"""Utilities for pricing, token cost calculations, and token counting used by analysis tools.

Exposes:
- MODEL_PRICING: pricing map per 1K tokens
- _get_pricing(model_name): internal helper to fetch pricing for a model
- calculate_cost_components(stats, model_name): compute cost breakdown
 - _get_encoding(model_name): stable tiktoken encoding lookup with fallbacks
 - _common_prefix_len(a, b): integer prefix overlap length for caching
 - calculate_llm_tokens_with_cache(llm_history, model_name): count tokens per step
 - calculate_optimizer_tokens_with_cache(history_file, model_name): count tokens per turn
"""
from __future__ import annotations
from typing import Dict, List, Optional
import os
import json
import sys
import subprocess
import hashlib
import pickle
import logging
from pathlib import Path
from collections import defaultdict
import glob
import re

# Pricing stored as cost per 1K tokens. Keep in sync with analyzer expectations.
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # $3 / 1M input, $0.75 / 1M cached input, $12 / 1M output
    'gpt-4.1': {
        'input': 3 / 1000,
        'cached_input': 0.75 / 1000,
        'output': 12 / 1000,
    },
    # Example mini pricing
    'gpt-4.1-mini': {
        'input': 0.8 / 1000,
        'cached_input': 0.2 / 1000,
        'output': 3.2 / 1000,
    },
}

# Increment when cached token stats schema changes (e.g., new metrics like dependency)
# Increment when token counting logic changes (advanced optimizer output logic updated)
CACHE_SCHEMA_VERSION = "v2.2"


def _get_pricing(model_name: str) -> Dict[str, float]:
    """Return pricing map for a given model name.

    If unknown, returns an empty dict which effectively makes all costs 0.
    """
    return MODEL_PRICING.get(model_name, {})


def calculate_cost_components(stats: Dict, model_name: str) -> Dict[str, float]:
    """Calculate cost components for a set of token stats and model pricing.

    stats keys expected:
    - total_input_tokens
    - total_input_cached_tokens
    - total_input_new_tokens
    - total_output_tokens

    Returns a dict with input/output costs and totals both with and without cache discount.
    """
    pricing = _get_pricing(model_name)
    input_rate = float(pricing.get('input', 0.0))
    cached_input_rate = float(pricing.get('cached_input', input_rate))  # fallback to input
    output_rate = float(pricing.get('output', 0.0))

    total_input = int(stats.get('total_input_tokens', 0) or 0)
    cached_input = int(stats.get('total_input_cached_tokens', 0) or 0)
    new_input = int(stats.get('total_input_new_tokens', total_input) or 0)
    output_tokens = int(stats.get('total_output_tokens', 0) or 0)

    # Costs: tokens are billed per 1K at the given rate
    input_cost_no_cache = (total_input / 1000.0) * input_rate
    input_cost_with_cache = (cached_input / 1000.0) * cached_input_rate + (new_input / 1000.0) * input_rate
    output_cost = (output_tokens / 1000.0) * output_rate

    total_cost_no_cache = input_cost_no_cache + output_cost
    total_cost_with_cache = input_cost_with_cache + output_cost

    return {
        'input_cost_no_cache': input_cost_no_cache,
        'input_cost_with_cache': input_cost_with_cache,
        'output_cost': output_cost,
        'total_cost_no_cache': total_cost_no_cache,
        'total_cost_with_cache': total_cost_with_cache,
    }


# ---------------- Tokenization utilities ---------------- #

def _get_encoding(model_name: str):
    """Return a tiktoken encoding object, with sensible fallbacks.

    Attempts to import tiktoken on demand and install it if missing.
    """
    try:
        import tiktoken  # type: ignore
    except Exception:
        # Try to install tiktoken if unavailable
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tiktoken'])
            import tiktoken  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("tiktoken is required for token counting. Please install it.") from e
    try:
        name = str(model_name).lower()
        if 'gpt-4' in name:
            return tiktoken.encoding_for_model('gpt-4')
        if 'gpt-3.5' in name:
            return tiktoken.encoding_for_model('gpt-3.5-turbo')
        return tiktoken.get_encoding('cl100k_base')
    except Exception:  # pragma: no cover
        return tiktoken.get_encoding('cl100k_base')


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    """Return length of common prefix between two token id sequences."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# ---------------- Calculation functions ---------------- #

def calculate_llm_tokens_with_cache(llm_history, model_name: str = 'gpt-4.1'):
    """Compute agent token usage with cache-aware input counting.

    llm_history is expected to be a list of sessions; each session is a list of
    messages with 'role' and 'content'. We treat each assistant message as a step.
    """
    encoding = _get_encoding(model_name)
    if not llm_history or not isinstance(llm_history, list):
        return {
            'total_input_tokens': 0,
            'total_input_cached_tokens': 0,
            'total_input_new_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens_no_cache': 0,
            'total_tokens_with_cache': 0,
            'step_breakdown': [],
            'num_steps': 0
        }

    step_breakdown = []
    peak_tokens = 0
    total_input_tokens = total_input_cached = total_input_new = total_output_tokens = 0
    prev_input_token_ids: List[int] = []
    last_session_assistant_content = []

    for session_idx, session in enumerate(llm_history):
        if not isinstance(session, list):
            continue
        conversation_history = []
        step_count = 0
        for message in session:
            if not isinstance(message, dict) or 'role' not in message:
                continue
            role = message.get('role', '')
            content = message.get('content', '')
            conversation_history.append(message)
            if role == 'assistant':
                # skip duplicates or empty assistant content
                if not content or (last_session_assistant_content and content in last_session_assistant_content):
                    continue
                step_count += 1
                # Build full prior input (all messages before assistant reply)
                input_messages = conversation_history[:-1]
                input_text = ''.join(f"{m.get('role','')}: {m.get('content','')}\n" for m in input_messages)
                # For dependency metric, exclude system prompt tokens from input count
                input_text_no_system = ''.join(
                    f"{m.get('role','')}: {m.get('content','')}\n" for m in input_messages if m.get('role') != 'system'
                )
                # Tokenize input and output
                if " " * 10 in input_text: # Quick fix for some errorneous input (errorneous output from optimizer)
                    input_text = input_text.replace(" " * 10, "")
                input_token_ids = encoding.encode(input_text) if input_text else []
                input_no_system_token_ids = encoding.encode(input_text_no_system) if input_text_no_system else []
                output_token_ids = encoding.encode(content) if content else []
                # Caching computation: overlap with previous step's input sequence (prefix)
                cached_len = _common_prefix_len(prev_input_token_ids, input_token_ids)
                new_len = len(input_token_ids) - cached_len
                prev_input_token_ids = input_token_ids  # update for next step

                total_input_tokens += len(input_token_ids)
                total_input_cached += cached_len
                total_input_new += new_len

                out_tokens = len(output_token_ids)
                total_output_tokens += out_tokens
                last_session_assistant_content.append(content)
                # Peak tokens should consider only input tokens at each step
                if len(input_token_ids) > peak_tokens:
                    peak_tokens = len(input_token_ids)
                # Dependency metric: ((n_input + 2 * n_output) * n_output) / 2 with n_input excluding system tokens
                n_input_dep = len(input_no_system_token_ids)
                n_output_dep = out_tokens
                dependency_val = ((n_input_dep + 2 * n_output_dep) * n_output_dep) / 2 if n_output_dep > 0 else 0.0
                step_breakdown.append({
                    'session': session_idx + 1,
                    'step': step_count,
                    'input_tokens_total': len(input_token_ids),
                    'input_tokens_cached': cached_len,
                    'input_tokens_new': new_len,
                    'output_tokens': out_tokens,
                    'dependency': dependency_val,
                    'total_tokens_step_no_cache': len(input_token_ids) + out_tokens,
                    'total_tokens_step_with_cache': new_len + out_tokens
                })
    total_tokens_no_cache = total_input_tokens + total_output_tokens
    # For with_cache, cached tokens are billed at discount, not zero; keep totals consistent
    total_tokens_with_cache = total_input_new + total_input_cached + total_output_tokens
    total_dependency = sum(sb.get('dependency', 0.0) for sb in step_breakdown)

    return {
        'total_input_tokens': total_input_tokens,
        'total_input_cached_tokens': total_input_cached,
        'total_input_new_tokens': total_input_new,
        'total_output_tokens': total_output_tokens,
        'total_tokens_no_cache': total_tokens_no_cache,
        'total_tokens_with_cache': total_tokens_with_cache,
        'step_breakdown': step_breakdown,
        'total_dependency': total_dependency,
        'num_steps': len(step_breakdown),
        'peak_tokens': peak_tokens,
    }


def calculate_optimizer_tokens_with_cache(
    history_file: str,
    model_name: str = 'gpt-4.1-mini',
    agent_llm_history_file: Optional[str] = None,
):
    """Compute optimizer token usage with two formats.

    Format A (legacy): each turn is a list where the second element is a string (user prompt).
      Example turn: [system_prompt, user_text, assistant_response]
      -> Uses simple previous-turn prefix caching.

    Format B (advanced): each turn is a list where the second element is a list of chat messages
      (each a dict with role/content). Example turn:
        [system_prompt, [ {role: user, content: ...}, {role: assistant, content: ...}, ...], optimizer_response]
      -> For caching we consider the full flattened input (system + all provided messages) and allow
         reuse from either (a) previous optimizer turn, or (b) any agent step input sequence built
         from llm_history (if available). We automatically attempt to load llm_history.json from the
         same directory if agent_llm_history_file not supplied.
    """
    if not os.path.exists(history_file):
        return {
            'total_input_tokens': 0,
            'total_input_cached_tokens': 0,
            'total_input_new_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens_no_cache': 0,
            'total_tokens_with_cache': 0,
            'num_turns': 0
        }
    try:
        with open(history_file, 'r') as f:
            history_data = json.load(f)
    except Exception:
        history_data = []
    encoding = _get_encoding(model_name)

    def _basic(history_list: List):
        total_input_tokens = total_input_cached = total_input_new = total_output_tokens = 0
        total_dependency = 0.0
        peak_tokens = 0
        num_turns = 0
        prev_input_ids: List[int] = []
        for turn in history_list:
            if isinstance(turn, list) and len(turn) >= 2 and isinstance(turn[1], str):
                num_turns += 1
                input_text_parts = []
                if turn[0]:
                    input_text_parts.append(f"system: {turn[0]}")
                if turn[1]:
                    if " " * 10 in turn[1]:
                        turn[1] = turn[1].replace(" " * 10, "")
                    input_text_parts.append(f"user: {turn[1]}")
                input_text = '\n'.join(input_text_parts)
                input_ids = encoding.encode(input_text) if input_text else []
                user_only_text = f"user: {turn[1]}" if turn[1] else ''
                user_only_ids = encoding.encode(user_only_text) if user_only_text else []
                cached_len = _common_prefix_len(prev_input_ids, input_ids)
                new_len = len(input_ids) - cached_len
                total_input_tokens += len(input_ids)
                total_input_cached += cached_len
                total_input_new += new_len
                prev_input_ids = input_ids
                out_len = 0
                if len(turn) >= 3 and isinstance(turn[2], str) and turn[2]:
                    out_ids = encoding.encode(turn[2].strip())
                    out_len = len(out_ids)
                    total_output_tokens += out_len
                if out_len > 0:
                    total_dependency += ((len(user_only_ids) + 2 * out_len) * out_len) / 2
                if len(input_ids) > peak_tokens:
                    peak_tokens = len(input_ids)
        total_tokens_no_cache = total_input_tokens + total_output_tokens
        total_tokens_with_cache = total_input_cached + total_input_new + total_output_tokens
        return {
            'total_input_tokens': total_input_tokens,
            'total_input_cached_tokens': total_input_cached,
            'total_input_new_tokens': total_input_new,
            'total_output_tokens': total_output_tokens,
            'total_tokens_no_cache': total_tokens_no_cache,
            'total_tokens_with_cache': total_tokens_with_cache,
            'num_turns': num_turns,
            'peak_tokens': peak_tokens,
            'total_dependency': total_dependency,
        }

    def _load_agent_sequences(llm_hist_path: str):
        if not os.path.exists(llm_hist_path):
            return []
        try:
            with open(llm_hist_path, 'r') as af:
                llm_hist = json.load(af)
        except Exception:
            return []
        seqs: List[List[int]] = []
        for session in llm_hist if isinstance(llm_hist, list) else []:
            if not isinstance(session, list):
                continue
            conv = []
            for msg in session:
                if not isinstance(msg, dict):
                    continue
                conv.append(msg)
                if msg.get('role') == 'assistant':
                    prior = conv[:-1]
                    text = ''.join(f"{m.get('role','')}: {m.get('content','') }\n" for m in prior)
                    if text:
                        if " " * 10 in text:
                            text = text.replace(" " * 10, "")
                        try:
                            seq = encoding.encode(text)
                        except Exception:
                            seq = []
                        if seq:
                            seqs.append(seq)
        return seqs

    def _advanced(history_list: List):
        # Build agent sequences
        if agent_llm_history_file is None:
            cand = os.path.join(os.path.dirname(history_file), 'llm_history.json')
        else:
            cand = agent_llm_history_file
        agent_seqs = _load_agent_sequences(cand)
        total_input_tokens = total_input_cached = total_input_new = total_output_tokens = 0
        total_dependency = 0.0
        peak_tokens = 0
        num_turns = 0
        prev_input_ids: List[int] = []
        cached_from_agent = 0
        cached_from_prev = 0
        for turn in history_list:
            if isinstance(turn, list) and len(turn) >= 2 and isinstance(turn[1], list):
                messages = turn[1]
                num_turns += 1
                parts = []
                # For Format B the first element is an empty string; system messages are inside messages list.
                # Only include turn[0] if non-empty AND there is no system message in messages (fallback safety).
                has_system_in_messages = any(isinstance(m, dict) and m.get('role') == 'system' for m in messages)
                if turn[0] and not has_system_in_messages:
                    parts.append(f"system: {turn[0]}")
                for m in messages:
                    if isinstance(m, dict):
                        parts.append(f"{m.get('role','')}: {m.get('content','')}")
                input_text = '\n'.join(parts)
                if " " * 10 in input_text:
                    input_text = input_text.replace(" " * 10, "")
                input_ids = encoding.encode(input_text) if input_text else []
                # For dependency exclude system tokens
                user_only_ids = []
                for m in messages:
                    if isinstance(m, dict) and m.get('role') != 'system':
                        c = m.get('content') or ''
                        if c:
                            user_only_ids.extend(encoding.encode(f"{m.get('role')}: {c}"))
                cached_prev = _common_prefix_len(prev_input_ids, input_ids)
                cached_agent_local = 0
                for seq in agent_seqs:
                    cl = _common_prefix_len(seq, input_ids)
                    if cl > cached_agent_local:
                        cached_agent_local = cl
                if cached_agent_local >= cached_prev:
                    cached_len = cached_agent_local
                    if cached_len > 0:
                        cached_from_agent += cached_len
                else:
                    cached_len = cached_prev
                    if cached_len > 0:
                        cached_from_prev += cached_len
                new_len = len(input_ids) - cached_len
                total_input_tokens += len(input_ids)
                total_input_cached += cached_len
                total_input_new += new_len
                prev_input_ids = input_ids
                # Output: ONLY turn[2] (first element after messages) counts as optimizer output.
                # Any additional trailing strings (e.g., replicated summaries) are ignored for cost.
                out_len = 0
                if len(turn) >= 3 and isinstance(turn[2], str) and turn[2]:
                    out_candidate = turn[2]
                    out_ids = encoding.encode(out_candidate.strip())
                    out_len = len(out_ids)
                    total_output_tokens += out_len
                if out_len > 0:
                    total_dependency += ((len(user_only_ids) + 2 * out_len) * out_len) / 2
                if len(input_ids) > peak_tokens:
                    peak_tokens = len(input_ids)
        total_tokens_no_cache = total_input_tokens + total_output_tokens
        total_tokens_with_cache = total_input_cached + total_input_new + total_output_tokens
        return {
            'total_input_tokens': total_input_tokens,
            'total_input_cached_tokens': total_input_cached,
            'total_input_new_tokens': total_input_new,
            'total_output_tokens': total_output_tokens,
            'total_tokens_no_cache': total_tokens_no_cache,
            'total_tokens_with_cache': total_tokens_with_cache,
            'num_turns': num_turns,
            'peak_tokens': peak_tokens,
            'total_dependency': total_dependency,
            'total_input_cached_tokens_from_agent': cached_from_agent,
            'total_input_cached_tokens_from_prev': cached_from_prev,
            'advanced_cache': True,
        }

    if isinstance(history_data, list) and history_data and isinstance(history_data[0], list) and len(history_data[0]) >= 2 and isinstance(history_data[0][1], list):
        return _advanced(history_data)
    return _basic(history_data if isinstance(history_data, list) else [])


# ---------------- Config and Base Paths ---------------- #

# Central config fallback to locate experiments and cache directories
try:  # pragma: no cover
    from experiments.analysis_tools.config import TOKEN_USAGE_CACHE_DIR, EXPERIMENTS_ROOT  # type: ignore
except Exception:  # pragma: no cover
    try:
        from .config import TOKEN_USAGE_CACHE_DIR, EXPERIMENTS_ROOT  # type: ignore
    except Exception:
        TOKEN_USAGE_CACHE_DIR = Path('analysis/cache/token_usage')
        TOKEN_USAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]

APPWORLD_BASE_PATH = EXPERIMENTS_ROOT / 'appworld' / 'outputs'
OFFICEBENCH_BASE_PATH = EXPERIMENTS_ROOT / 'officebench' / 'outputs'
SMOLAGENTS_BASE_PATH = EXPERIMENTS_ROOT / 'smolagents' / 'outputs'

# Support alternate outputs directory names (e.g. outputs_gpt5) while preserving existing API.
APPWORLD_BASE_PATHS = [p for p in [APPWORLD_BASE_PATH, EXPERIMENTS_ROOT / 'appworld' / 'outputs_gpt5'] if p.exists()]
OFFICEBENCH_BASE_PATHS = [p for p in [OFFICEBENCH_BASE_PATH, EXPERIMENTS_ROOT / 'officebench' / 'outputs_gpt5'] if p.exists()]
SMOLAGENTS_BASE_PATHS = [p for p in [SMOLAGENTS_BASE_PATH, EXPERIMENTS_ROOT / 'smolagents' / 'outputs_gpt5'] if p.exists()]


# ---------------- Cache helpers ---------------- #

def get_file_hash(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def get_cache_key(file_path: str, model_name: str, suffix: str = '') -> Optional[str]:
    h = get_file_hash(file_path)
    return f"{CACHE_SCHEMA_VERSION}_{h}_{model_name}{suffix}" if h else None


def load_token_cache(cache_key: Optional[str]):
    if not cache_key:
        return None
    cf = os.path.join(str(TOKEN_USAGE_CACHE_DIR), f"{cache_key}.pkl")
    if not os.path.exists(cf):
        return None
    try:
        with open(cf, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def save_token_cache(cache_key: Optional[str], token_data):  # pragma: no cover
    if not cache_key:
        return
    os.makedirs(str(TOKEN_USAGE_CACHE_DIR), exist_ok=True)
    cf = os.path.join(str(TOKEN_USAGE_CACHE_DIR), f"{cache_key}.pkl")
    try:
        with open(cf, 'wb') as f:
            pickle.dump(token_data, f)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save token cache: {e}")


# ---------------- Experiment helpers ---------------- #

def calculate_env_steps(task_dir: str, experiment_type='appworld'):
    if experiment_type == 'appworld':
        f = os.path.join(task_dir, 'appworld_trajectory.json')
        if os.path.exists(f):
            try:
                with open(f, 'r') as fh:
                    env = json.load(fh)
                return {'num_steps': len(env.get('trajectory', []))}
            except Exception:
                return {'num_steps': 0}
        return {'num_steps': 0}
    f = os.path.join(task_dir, 'env_history.json')
    if os.path.exists(f):
        try:
            with open(f, 'r') as fh:
                env = json.load(fh)
            return {'num_steps': len(env)}
        except Exception:
            return {'num_steps': 0}
    return {'num_steps': 0}


def load_experiment_evaluation(experiment_name: str, split='train'):
    clean = experiment_name[11:] if experiment_name.startswith('[AppWorld] ') else experiment_name
    eval_file = str(EXPERIMENTS_ROOT / 'appworld' / 'experiments' / 'outputs' / clean / 'evaluations' / f'{split}.json')
    if os.path.exists(eval_file):
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)
            return data.get('individual', {}), data.get('aggregate', {})
        except Exception:
            return {}, {}
    return {}, {}


def detect_experiment_type(experiment_name: str):
    # Prefixed names
    if experiment_name.startswith('[AppWorld] '):
        clean = experiment_name[11:]
        for base in APPWORLD_BASE_PATHS or [APPWORLD_BASE_PATH]:
            p = base / clean
            if p.exists():
                return 'appworld', p
    if experiment_name.startswith('[OfficeBench] '):
        clean = experiment_name[14:]
        for base in OFFICEBENCH_BASE_PATHS or [OFFICEBENCH_BASE_PATH]:
            p = base / clean
            if p.exists():
                return 'officebench', p
    if experiment_name.startswith('[SmolAgents] '):
        clean = experiment_name[13:]
        for base in SMOLAGENTS_BASE_PATHS or [SMOLAGENTS_BASE_PATH]:
            p = base / clean
            if p.exists():
                return 'smolagents', p
    # Raw names (search all bases)
    for base in APPWORLD_BASE_PATHS or [APPWORLD_BASE_PATH]:
        p = base / experiment_name
        if p.exists():
            return 'appworld', p
    for base in OFFICEBENCH_BASE_PATHS or [OFFICEBENCH_BASE_PATH]:
        p = base / experiment_name
        if p.exists():
            return 'officebench', p
    for base in SMOLAGENTS_BASE_PATHS or [SMOLAGENTS_BASE_PATH]:
        p = base / experiment_name
        if p.exists():
            return 'smolagents', p
    return None, None


def get_available_experiments():
    exps_set = set()
    for base in APPWORLD_BASE_PATHS or [APPWORLD_BASE_PATH]:
        if base.exists():
            for d in base.iterdir():
                if d.is_dir():
                    exps_set.add(f"[AppWorld] {d.name}")
    for base in OFFICEBENCH_BASE_PATHS or [OFFICEBENCH_BASE_PATH]:
        if base.exists():
            for d in base.iterdir():
                if d.is_dir():
                    exps_set.add(f"[OfficeBench] {d.name}")
    for base in SMOLAGENTS_BASE_PATHS or [SMOLAGENTS_BASE_PATH]:
        if base.exists():
            for d in base.iterdir():
                if d.is_dir():
                    exps_set.add(f"[SmolAgents] {d.name}")
    return sorted(exps_set)


def get_task_difficulty(task_id: str, evaluation_data=None, task_dir_path: Optional[str]=None, experiment_type='appworld'):
    if experiment_type == 'officebench':
        if re.match(r'^\d+-\d+$', task_id):
            return f"level_{task_id.split('-')[0]}"
        return 'unknown'
    clean_task_id = task_id[5:] if task_id.startswith('task_') else task_id
    meta_path = EXPERIMENTS_ROOT / 'appworld' / 'data' / 'tasks' / clean_task_id / 'ground_truth' / 'metadata.json'
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as mf:
                md = json.load(mf)
            md_diff = md.get('difficulty')
            if md_diff in [1,2,3,'1','2','3']:
                return str(md_diff)
        except Exception:
            pass
    if evaluation_data and clean_task_id in evaluation_data:
        diff = evaluation_data[clean_task_id].get('difficulty')
        if diff in [1,2,3,'1','2','3']:
            return str(diff)
    return 'unknown'


def get_task_fold(task_dir_path: Optional[str]) -> str:
    if not task_dir_path:
        return 'unknown'
    parts = [p.lower() for p in task_dir_path.split(os.sep)]
    for fold in ['train','dev','test_challenge','challenge','test_normal','test']:
        if fold in parts:
            if fold == 'challenge':
                return 'test_challenge'
            if fold == 'test':
                return 'test_normal'
            return fold
    return 'unknown'


# ---------------- Main analysis (moved from analyzer) ---------------- #

def analyze_experiment_tokens_v2(
    experiment_name: str, 
    agent_model='gpt-4.1', 
    optimizer_model='gpt-4.1', 
    folds: Optional[List[str]] = None,
    verbose=False
):
    experiment_type, experiment_path = detect_experiment_type(experiment_name)
    if not experiment_path:
        return None
    evaluation_data, aggregate_data = ({}, {})
    if experiment_type == 'appworld':
        evaluation_data, aggregate_data = load_experiment_evaluation(experiment_name, 'train')

    results = {
        'experiment_name': experiment_name,
        'experiment_type': experiment_type,
        'agent_model': agent_model,
        'optimizer_model': optimizer_model,
        'evaluation_data_available': bool(evaluation_data),
        'tasks': {},
        'by_difficulty': defaultdict(lambda: {
            'agent': {
                'input_tokens': 0,
                'input_cached_tokens': 0,
                'input_new_tokens': 0,
                'output_tokens': 0,
                'num_tasks': 0,
                'avg_steps': 0,
                'peak_tokens_total': 0,
                'dependency_total': 0.0,
                'input_cost_no_cache_total': 0.0,
                'input_cost_with_cache_total': 0.0,
                'output_cost_total': 0.0,
                'total_cost_no_cache_total': 0.0,
                'total_cost_with_cache_total': 0.0,
            },
            'optimizer': {
                'input_tokens': 0,
                'input_cached_tokens': 0,
                'input_new_tokens': 0,
                'output_tokens': 0,
                'num_tasks': 0,
                'peak_tokens_total': 0,
                'dependency_total': 0.0,
                'input_cost_no_cache_total': 0.0,
                'input_cost_with_cache_total': 0.0,
                'output_cost_total': 0.0,
                'total_cost_no_cache_total': 0.0,
                'total_cost_with_cache_total': 0.0,
            }
        }),
        'by_fold': defaultdict(lambda: {
            'agent': {
                'input_tokens': 0,
                'input_cached_tokens': 0,
                'input_new_tokens': 0,
                'output_tokens': 0,
                'num_tasks': 0,
                'avg_steps': 0,
                'peak_tokens_total': 0,
                'dependency_total': 0.0,
                'input_cost_no_cache_total': 0.0,
                'input_cost_with_cache_total': 0.0,
                'output_cost_total': 0.0,
                'total_cost_no_cache_total': 0.0,
                'total_cost_with_cache_total': 0.0,
            },
            'optimizer': {
                'input_tokens': 0,
                'input_cached_tokens': 0,
                'input_new_tokens': 0,
                'output_tokens': 0,
                'num_tasks': 0,
                'peak_tokens_total': 0,
                'dependency_total': 0.0,
                'input_cost_no_cache_total': 0.0,
                'input_cost_with_cache_total': 0.0,
                'output_cost_total': 0.0,
                'total_cost_no_cache_total': 0.0,
                'total_cost_with_cache_total': 0.0,
            }
        }),
        'by_fold_difficulty': defaultdict(lambda: defaultdict(lambda: {
            'agent': {
                'input_tokens': 0,
                'input_cached_tokens': 0,
                'input_new_tokens': 0,
                'output_tokens': 0,
                'num_tasks': 0,
                'avg_steps': 0,
                'peak_tokens_total': 0,
                'dependency_total': 0.0,
                'input_cost_no_cache_total': 0.0,
                'input_cost_with_cache_total': 0.0,
                'output_cost_total': 0.0,
                'total_cost_no_cache_total': 0.0,
                'total_cost_with_cache_total': 0.0,
            },
            'optimizer': {
                'input_tokens': 0,
                'input_cached_tokens': 0,
                'input_new_tokens': 0,
                'output_tokens': 0,
                'num_tasks': 0,
                'peak_tokens_total': 0,
                'dependency_total': 0.0,
                'input_cost_no_cache_total': 0.0,
                'input_cost_with_cache_total': 0.0,
                'output_cost_total': 0.0,
                'total_cost_no_cache_total': 0.0,
                'total_cost_with_cache_total': 0.0,
            }
        })),
        'step_analysis': {'by_step': defaultdict(list), 'max_steps': 0}
    }

    if experiment_type == 'appworld':
        task_pattern = os.path.join(str(experiment_path), '*', 'task_*')
    elif experiment_type == 'smolagents':
        # Folds (e.g. test, test_challenge, dev, train) / samples / <sample_id>
        task_pattern = os.path.join(str(experiment_path), '*', 'samples', '*')
    else:
        task_pattern = os.path.join(str(experiment_path), '*', '[1-3]-*', '*')
    task_dirs = glob.glob(task_pattern)

    for task_dir in task_dirs:
        if experiment_type in ('appworld', 'smolagents'):
            task_id = os.path.basename(task_dir)
        else:
            task_id = os.path.basename(os.path.dirname(task_dir))
        fold_val = get_task_fold(task_dir)
        if experiment_type in ('smolagents') and fold_val == 'test_normal':
            # Preserve original 'test' naming for these datasets
            fold_val = 'test'
        if folds and fold_val not in set(folds):
            continue
        difficulty = get_task_difficulty(task_id, evaluation_data, task_dir, experiment_type)
        if verbose:
            print("Processing task:", task_id, "Difficulty:", difficulty, "Fold:", fold_val)
        # Agent
        llm_history_file = os.path.join(task_dir, 'llm_history.json')
        agent_stats = {
            'total_input_tokens': 0,
            'total_input_cached_tokens': 0,
            'total_input_new_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens_no_cache': 0,
            'total_tokens_with_cache': 0,
            'num_steps': 0,
            'step_breakdown': []
        }
        if os.path.exists(llm_history_file):
            cache_key = get_cache_key(llm_history_file, agent_model, '_agent_v2')
            cached = load_token_cache(cache_key)
            if cached:
                agent_stats = cached
                # Backfill new dependency metric if absent
                if 'total_dependency' not in agent_stats:
                    try:
                        with open(llm_history_file, 'r') as f:
                            llm_history = json.load(f)
                        recomputed = calculate_llm_tokens_with_cache(llm_history, agent_model)
                        # Preserve existing num_steps if set externally
                        if 'num_steps' in agent_stats:
                            recomputed['num_steps'] = agent_stats['num_steps']
                        agent_stats = recomputed
                        save_token_cache(cache_key, agent_stats)
                    except Exception:  # pragma: no cover
                        agent_stats.setdefault('total_dependency', 0.0)
            else:
                try:
                    with open(llm_history_file, 'r') as f:
                        llm_history = json.load(f)
                    agent_stats = calculate_llm_tokens_with_cache(llm_history, agent_model)
                    env_stats = calculate_env_steps(task_dir, experiment_type)
                    agent_stats['num_steps'] = env_stats['num_steps']
                    save_token_cache(cache_key, agent_stats)
                except Exception as e:  # pragma: no cover
                    logging.getLogger(__name__).warning(f'Failed processing {llm_history_file}: {e}')
        # Ensure peak_tokens exists even for older cached entries
        if 'peak_tokens' not in agent_stats:
            peak = 0
            for sd in agent_stats.get('step_breakdown', []) or []:
                # Prefer input-only tokens for peak
                step_input = sd.get('input_tokens_total')
                if step_input is None:
                    # Fallback: derive from totals if needed
                    t_no_cache = sd.get('total_tokens_step_no_cache', 0) or 0
                    out_toks = sd.get('output_tokens', 0) or 0
                    step_input = max(0, t_no_cache - out_toks)
                if (step_input or 0) > peak:
                    peak = int(step_input or 0)
            # As a last resort, try recompute if no breakdown present
            if peak == 0 and os.path.exists(llm_history_file):
                try:
                    with open(llm_history_file, 'r') as f:
                        llm_history = json.load(f)
                    recomputed = calculate_llm_tokens_with_cache(llm_history, agent_model)
                    peak = recomputed.get('peak_tokens', 0) or 0
                except Exception:
                    peak = 0
            agent_stats['peak_tokens'] = peak

        # Optimizer
        history_optimizer_file = os.path.join(task_dir, 'history_optimizer_history.json')
        obs_optimizer_file = os.path.join(task_dir, 'obs_optimizer_history.json')
        history_stats = {
            'total_input_tokens': 0,
            'total_input_cached_tokens': 0,
            'total_input_new_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens_no_cache': 0,
            'total_tokens_with_cache': 0,
            'num_turns': 0
        }
        obs_stats = {
            'total_input_tokens': 0,
            'total_input_cached_tokens': 0,
            'total_input_new_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens_no_cache': 0,
            'total_tokens_with_cache': 0,
            'num_turns': 0
        }
        if os.path.exists(history_optimizer_file):
            ck = get_cache_key(history_optimizer_file, optimizer_model, '_opt_hist_v2')
            cached = load_token_cache(ck)
            if cached:
                history_stats = cached
                if 'total_dependency' not in history_stats:
                    try:
                        history_stats = calculate_optimizer_tokens_with_cache(history_optimizer_file, optimizer_model)
                        save_token_cache(ck, history_stats)
                    except Exception:
                        history_stats.setdefault('total_dependency', 0.0)
            else:
                history_stats = calculate_optimizer_tokens_with_cache(history_optimizer_file, optimizer_model)
                save_token_cache(ck, history_stats)
        # Ensure peak_tokens exists for cached optimizer stats
        if 'peak_tokens' not in history_stats and os.path.exists(history_optimizer_file):
            try:
                history_stats = calculate_optimizer_tokens_with_cache(history_optimizer_file, optimizer_model)
                # Optionally refresh cache
                ck = get_cache_key(history_optimizer_file, optimizer_model, '_opt_hist_v2')
                save_token_cache(ck, history_stats)
            except Exception:
                history_stats.setdefault('peak_tokens', 0)

        if os.path.exists(obs_optimizer_file):
            ck = get_cache_key(obs_optimizer_file, optimizer_model, '_opt_obs_v2')
            cached = load_token_cache(ck)
            if cached:
                obs_stats = cached
                if 'total_dependency' not in obs_stats:
                    try:
                        obs_stats = calculate_optimizer_tokens_with_cache(obs_optimizer_file, optimizer_model)
                        save_token_cache(ck, obs_stats)
                    except Exception:
                        obs_stats.setdefault('total_dependency', 0.0)
            else:
                obs_stats = calculate_optimizer_tokens_with_cache(obs_optimizer_file, optimizer_model)
                save_token_cache(ck, obs_stats)
        if 'peak_tokens' not in obs_stats and os.path.exists(obs_optimizer_file):
            try:
                obs_stats = calculate_optimizer_tokens_with_cache(obs_optimizer_file, optimizer_model)
                ck = get_cache_key(obs_optimizer_file, optimizer_model, '_opt_obs_v2')
                save_token_cache(ck, obs_stats)
            except Exception:
                obs_stats.setdefault('peak_tokens', 0)

        optimizer_stats = {
            'total_input_tokens': history_stats['total_input_tokens'] + obs_stats['total_input_tokens'],
            'total_input_cached_tokens': history_stats['total_input_cached_tokens'] + obs_stats['total_input_cached_tokens'],
            'total_input_new_tokens': history_stats['total_input_new_tokens'] + obs_stats['total_input_new_tokens'],
            'total_output_tokens': history_stats['total_output_tokens'] + obs_stats['total_output_tokens'],
            'total_tokens_no_cache': history_stats['total_tokens_no_cache'] + obs_stats['total_tokens_no_cache'],
            'total_tokens_with_cache': history_stats['total_tokens_with_cache'] + obs_stats['total_tokens_with_cache'],
            'num_turns': history_stats['num_turns'] + obs_stats['num_turns'],
            'peak_tokens': max(history_stats.get('peak_tokens', 0) or 0, obs_stats.get('peak_tokens', 0) or 0),
            'history_optimizer': history_stats,
            'obs_optimizer': obs_stats
        }

        agent_cost_components = calculate_cost_components(agent_stats, agent_model)
        optimizer_cost_components = calculate_cost_components(optimizer_stats, optimizer_model)

        task_entry = {
            'difficulty': difficulty,
            'fold': fold_val,
            'agent': agent_stats,
            'optimizer': optimizer_stats,
            'costs': {
                'agent': agent_cost_components,
                'optimizer': optimizer_cost_components,
                'total_no_cache': agent_cost_components['total_cost_no_cache'] + optimizer_cost_components['total_cost_no_cache'],
                'total_with_cache': agent_cost_components['total_cost_with_cache'] + optimizer_cost_components['total_cost_with_cache']
            }
        }
        results['tasks'][task_id] = task_entry

        diff_stats = results['by_difficulty'][difficulty]
        if agent_stats['total_tokens_no_cache'] > 0:
            diff_agent = diff_stats['agent']
            diff_agent['input_tokens'] += agent_stats['total_input_tokens']
            diff_agent['input_cached_tokens'] += agent_stats['total_input_cached_tokens']
            diff_agent['input_new_tokens'] += agent_stats['total_input_new_tokens']
            diff_agent['output_tokens'] += agent_stats['total_output_tokens']
            diff_agent['num_tasks'] += 1
            diff_agent['avg_steps'] += agent_stats['num_steps']
            diff_agent['peak_tokens_total'] += agent_stats.get('peak_tokens', 0) or 0
            diff_agent['dependency_total'] += agent_stats.get('total_dependency', 0.0) or 0.0
            diff_agent['input_cost_no_cache_total'] += agent_cost_components['input_cost_no_cache']
            diff_agent['input_cost_with_cache_total'] += agent_cost_components['input_cost_with_cache']
            diff_agent['output_cost_total'] += agent_cost_components['output_cost']
            diff_agent['total_cost_no_cache_total'] += agent_cost_components['total_cost_no_cache']
            diff_agent['total_cost_with_cache_total'] += agent_cost_components['total_cost_with_cache']
        if optimizer_stats['total_tokens_no_cache'] > 0:
            diff_opt = diff_stats['optimizer']
            diff_opt['input_tokens'] += optimizer_stats['total_input_tokens']
            diff_opt['input_cached_tokens'] += optimizer_stats['total_input_cached_tokens']
            diff_opt['input_new_tokens'] += optimizer_stats['total_input_new_tokens']
            diff_opt['output_tokens'] += optimizer_stats['total_output_tokens']
            diff_opt['num_tasks'] += 1
            diff_opt['peak_tokens_total'] += optimizer_stats.get('peak_tokens', 0) or 0
            diff_opt['dependency_total'] += (history_stats.get('total_dependency',0.0) or 0.0) + (obs_stats.get('total_dependency',0.0) or 0.0)
            diff_opt['input_cost_no_cache_total'] += optimizer_cost_components['input_cost_no_cache']
            diff_opt['input_cost_with_cache_total'] += optimizer_cost_components['input_cost_with_cache']
            diff_opt['output_cost_total'] += optimizer_cost_components['output_cost']
            diff_opt['total_cost_no_cache_total'] += optimizer_cost_components['total_cost_no_cache']
            diff_opt['total_cost_with_cache_total'] += optimizer_cost_components['total_cost_with_cache']

        fold = task_entry['fold']
        fold_stats = results['by_fold'][fold]
        if agent_stats['total_tokens_no_cache'] > 0:
            fA = fold_stats['agent']
            fA['input_tokens'] += agent_stats['total_input_tokens']
            fA['input_cached_tokens'] += agent_stats['total_input_cached_tokens']
            fA['input_new_tokens'] += agent_stats['total_input_new_tokens']
            fA['output_tokens'] += agent_stats['total_output_tokens']
            fA['num_tasks'] += 1
            fA['avg_steps'] += agent_stats['num_steps']
            fA['peak_tokens_total'] += agent_stats.get('peak_tokens', 0) or 0
            fA['dependency_total'] += agent_stats.get('total_dependency', 0.0) or 0.0
            fA['input_cost_no_cache_total'] += agent_cost_components['input_cost_no_cache']
            fA['input_cost_with_cache_total'] += agent_cost_components['input_cost_with_cache']
            fA['output_cost_total'] += agent_cost_components['output_cost']
            fA['total_cost_no_cache_total'] += agent_cost_components['total_cost_no_cache']
            fA['total_cost_with_cache_total'] += agent_cost_components['total_cost_with_cache']
        if optimizer_stats['total_tokens_no_cache'] > 0:
            fO = fold_stats['optimizer']
            fO['input_tokens'] += optimizer_stats['total_input_tokens']
            fO['input_cached_tokens'] += optimizer_stats['total_input_cached_tokens']
            fO['input_new_tokens'] += optimizer_stats['total_input_new_tokens']
            fO['output_tokens'] += optimizer_stats['total_output_tokens']
            fO['num_tasks'] += 1
            fO['peak_tokens_total'] += optimizer_stats.get('peak_tokens', 0) or 0
            fO['dependency_total'] += (history_stats.get('total_dependency',0.0) or 0.0) + (obs_stats.get('total_dependency',0.0) or 0.0)
            fO['input_cost_no_cache_total'] += optimizer_cost_components['input_cost_no_cache']
            fO['input_cost_with_cache_total'] += optimizer_cost_components['input_cost_with_cache']
            fO['output_cost_total'] += optimizer_cost_components['output_cost']
            fO['total_cost_no_cache_total'] += optimizer_cost_components['total_cost_no_cache']
            fO['total_cost_with_cache_total'] += optimizer_cost_components['total_cost_with_cache']

        fd_stats = results['by_fold_difficulty'][fold][difficulty]
        if agent_stats['total_tokens_no_cache'] > 0:
            fdA = fd_stats['agent']
            fdA['input_tokens'] += agent_stats['total_input_tokens']
            fdA['input_cached_tokens'] += agent_stats['total_input_cached_tokens']
            fdA['input_new_tokens'] += agent_stats['total_input_new_tokens']
            fdA['output_tokens'] += agent_stats['total_output_tokens']
            fdA['num_tasks'] += 1
            fdA['avg_steps'] += agent_stats['num_steps']
            fdA['peak_tokens_total'] += agent_stats.get('peak_tokens', 0) or 0
            fdA['dependency_total'] += agent_stats.get('total_dependency', 0.0) or 0.0
            fdA['input_cost_no_cache_total'] += agent_cost_components['input_cost_no_cache']
            fdA['input_cost_with_cache_total'] += agent_cost_components['input_cost_with_cache']
            fdA['output_cost_total'] += agent_cost_components['output_cost']
            fdA['total_cost_no_cache_total'] += agent_cost_components['total_cost_no_cache']
            fdA['total_cost_with_cache_total'] += agent_cost_components['total_cost_with_cache']
        if optimizer_stats['total_tokens_no_cache'] > 0:
            fdO = fd_stats['optimizer']
            fdO['input_tokens'] += optimizer_stats['total_input_tokens']
            fdO['input_cached_tokens'] += optimizer_stats['total_input_cached_tokens']
            fdO['input_new_tokens'] += optimizer_stats['total_input_new_tokens']
            fdO['output_tokens'] += optimizer_stats['total_output_tokens']
            fdO['num_tasks'] += 1
            fdO['peak_tokens_total'] += optimizer_stats.get('peak_tokens', 0) or 0
            fdO['dependency_total'] += (history_stats.get('total_dependency',0.0) or 0.0) + (obs_stats.get('total_dependency',0.0) or 0.0)
            fdO['input_cost_no_cache_total'] += optimizer_cost_components['input_cost_no_cache']
            fdO['input_cost_with_cache_total'] += optimizer_cost_components['input_cost_with_cache']
            fdO['output_cost_total'] += optimizer_cost_components['output_cost']
            fdO['total_cost_no_cache_total'] += optimizer_cost_components['total_cost_no_cache']
            fdO['total_cost_with_cache_total'] += optimizer_cost_components['total_cost_with_cache']

        for step_data in agent_stats.get('step_breakdown', []):
            step_num = step_data['step']
            results['step_analysis']['by_step'][step_num].append(step_data['input_tokens_total'] + step_data['output_tokens'])
            if step_num > results['step_analysis']['max_steps']:
                results['step_analysis']['max_steps'] = step_num

    for difficulty, diff_stats in results['by_difficulty'].items():
        agent = diff_stats['agent']
        if agent['num_tasks'] > 0:
            n = agent['num_tasks']
            agent['avg_steps'] /= n
            agent['avg_input_tokens'] = agent['input_tokens'] / n
            agent['avg_input_cached_tokens'] = agent['input_cached_tokens'] / n
            agent['avg_input_new_tokens'] = agent['input_new_tokens'] / n
            agent['avg_output_tokens'] = agent['output_tokens'] / n
            agent['avg_peak_tokens'] = agent['peak_tokens_total'] / n
            agent['avg_input_cost_no_cache'] = agent['input_cost_no_cache_total'] / n
            agent['avg_input_cost_with_cache'] = agent['input_cost_with_cache_total'] / n
            agent['avg_output_cost'] = agent['output_cost_total'] / n
            agent['avg_total_cost_no_cache'] = agent['total_cost_no_cache_total'] / n
            agent['avg_total_cost_with_cache'] = agent['total_cost_with_cache_total'] / n
            agent['avg_dependency'] = agent['dependency_total'] / n
        optimizer = diff_stats['optimizer']
        if optimizer['num_tasks'] > 0:
            # n2 = optimizer['num_tasks']
            n2 = agent['num_tasks']  # Use agent's num_tasks for consistency
            optimizer['avg_input_tokens'] = optimizer['input_tokens'] / n2
            optimizer['avg_input_cached_tokens'] = optimizer['input_cached_tokens'] / n2
            optimizer['avg_input_new_tokens'] = optimizer['input_new_tokens'] / n2
            optimizer['avg_output_tokens'] = optimizer['output_tokens'] / n2
            optimizer['avg_peak_tokens'] = optimizer['peak_tokens_total'] / n2
            optimizer['avg_input_cost_no_cache'] = optimizer['input_cost_no_cache_total'] / n2
            optimizer['avg_input_cost_with_cache'] = optimizer['input_cost_with_cache_total'] / n2
            optimizer['avg_output_cost'] = optimizer['output_cost_total'] / n2
            optimizer['avg_total_cost_no_cache'] = optimizer['total_cost_no_cache_total'] / n2
            optimizer['avg_total_cost_with_cache'] = optimizer['total_cost_with_cache_total'] / n2
            optimizer['avg_dependency'] = (optimizer.get('dependency_total',0.0) or 0.0) / n2

    for fold_name, fold_stats in results['by_fold'].items():
        agent = fold_stats['agent']
        if agent['num_tasks'] > 0:
            n = agent['num_tasks']
            agent['avg_steps'] /= n
            agent['avg_input_tokens'] = agent['input_tokens'] / n
            agent['avg_input_cached_tokens'] = agent['input_cached_tokens'] / n
            agent['avg_input_new_tokens'] = agent['input_new_tokens'] / n
            agent['avg_output_tokens'] = agent['output_tokens'] / n
            agent['avg_peak_tokens'] = agent['peak_tokens_total'] / n
            agent['avg_input_cost_no_cache'] = agent['input_cost_no_cache_total'] / n
            agent['avg_input_cost_with_cache'] = agent['input_cost_with_cache_total'] / n
            agent['avg_output_cost'] = agent['output_cost_total'] / n
            agent['avg_total_cost_no_cache'] = agent['total_cost_no_cache_total'] / n
            agent['avg_total_cost_with_cache'] = agent['total_cost_with_cache_total'] / n
            agent['avg_dependency'] = agent['dependency_total'] / n
        optimizer = fold_stats['optimizer']
        if optimizer['num_tasks'] > 0:
            # n2 = optimizer['num_tasks']
            n2 = agent['num_tasks']  # Use agent's num_tasks for consistency
            optimizer['avg_input_tokens'] = optimizer['input_tokens'] / n2
            optimizer['avg_input_cached_tokens'] = optimizer['input_cached_tokens'] / n2
            optimizer['avg_input_new_tokens'] = optimizer['input_new_tokens'] / n2
            optimizer['avg_output_tokens'] = optimizer['output_tokens'] / n2
            optimizer['avg_peak_tokens'] = optimizer['peak_tokens_total'] / n2
            optimizer['avg_input_cost_no_cache'] = optimizer['input_cost_no_cache_total'] / n2
            optimizer['avg_input_cost_with_cache'] = optimizer['input_cost_with_cache_total'] / n2
            optimizer['avg_output_cost'] = optimizer['output_cost_total'] / n2
            optimizer['avg_total_cost_no_cache'] = optimizer['total_cost_no_cache_total'] / n2
            optimizer['avg_total_cost_with_cache'] = optimizer['total_cost_with_cache_total'] / n2
            optimizer['avg_dependency'] = (optimizer.get('dependency_total',0.0) or 0.0) / n2

    for fold_name, diff_map in results['by_fold_difficulty'].items():
        for diff, fd_stats in diff_map.items():
            a = fd_stats['agent']
            if a['num_tasks'] > 0:
                n = a['num_tasks']
                a['avg_steps'] /= n
                a['avg_input_tokens'] = a['input_tokens'] / n
                a['avg_input_cached_tokens'] = a['input_cached_tokens'] / n
                a['avg_input_new_tokens'] = a['input_new_tokens'] / n
                a['avg_output_tokens'] = a['output_tokens'] / n
                a['avg_peak_tokens'] = a['peak_tokens_total'] / n
                a['avg_input_cost_no_cache'] = a['input_cost_no_cache_total'] / n
                a['avg_input_cost_with_cache'] = a['input_cost_with_cache_total'] / n
                a['avg_output_cost'] = a['output_cost_total'] / n
                a['avg_total_cost_no_cache'] = a['total_cost_no_cache_total'] / n
                a['avg_total_cost_with_cache'] = a['total_cost_with_cache_total'] / n
                a['avg_dependency'] = a['dependency_total'] / n
            o = fd_stats['optimizer']
            if o['num_tasks'] > 0:
                # n2 = o['num_tasks']
                n2 = a['num_tasks']  # Use agent's num_tasks for consistency
                o['avg_input_tokens'] = o['input_tokens'] / n2
                o['avg_input_cached_tokens'] = o['input_cached_tokens'] / n2
                o['avg_input_new_tokens'] = o['input_new_tokens'] / n2
                o['avg_output_tokens'] = o['output_tokens'] / n2
                o['avg_peak_tokens'] = o['peak_tokens_total'] / n2
                o['avg_input_cost_no_cache'] = o['input_cost_no_cache_total'] / n2
                o['avg_input_cost_with_cache'] = o['input_cost_with_cache_total'] / n2
                o['avg_output_cost'] = o['output_cost_total'] / n2
                o['avg_total_cost_no_cache'] = o['total_cost_no_cache_total'] / n2
                o['avg_total_cost_with_cache'] = o['total_cost_with_cache_total'] / n2
                o['avg_dependency'] = (o.get('dependency_total',0.0) or 0.0) / n2

    return results


# ---------------- Presentation helpers ---------------- #

def build_markdown_cost_table(title: str, by_difficulty: dict) -> str:
    """Return a Markdown document containing a per-difficulty cost table.

    by_difficulty maps difficulty -> { 'agent': {...}, 'optimizer': {...} }
    Only uses avg_* and num_tasks fields if present.
    """
    # Try to sort difficulties numerically when possible, else lexicographically
    def sort_key(k: str):
        try:
            return (0, int(str(k)))
        except Exception:
            return (1, str(k))

    lines = []
    lines.append(f"## {title}")
    lines.append("")
    header = (
        "| Difficulty | Agent Tasks | Agent Avg Steps | Agent Avg Peak Tokens | Agent Avg Dependency | Agent Avg Total Cost (no cache) | Agent Avg Total Cost (with cache) | "
        "Optimizer Tasks | Optimizer Avg Peak Tokens | Optimizer Avg Dependency | Optimizer Avg Total Cost (no cache) | Optimizer Avg Total Cost (with cache) |"
    )
    sep = (
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines.append(header)
    lines.append(sep)
    
    # Calculate overall averages across all difficulties
    total_agent_tasks = 0
    total_agent_steps = 0.0
    total_agent_peak_tokens = 0.0
    total_agent_cost_no_cache = 0.0
    total_agent_cost_with_cache = 0.0
    total_opt_tasks = 0
    total_opt_peak_tokens = 0.0
    total_opt_cost_no_cache = 0.0
    total_opt_cost_with_cache = 0.0
    
    for diff in sorted(by_difficulty.keys(), key=sort_key):
        stats = by_difficulty.get(diff, {}) or {}
        A = stats.get('agent', {}) or {}
        O = stats.get('optimizer', {}) or {}
        
        agent_tasks = A.get('num_tasks', 0)
        opt_tasks = O.get('num_tasks', 0)
        
        row = (
            f"| {diff} | "
            f"{agent_tasks} | "
            f"{A.get('avg_steps', 0.0):.2f} | "
            f"{A.get('avg_peak_tokens', 0.0):.1f} | "
            f"{A.get('avg_dependency', 0.0):.2f} | "
            f"{A.get('avg_total_cost_no_cache', 0.0):.6f} | "
            f"{A.get('avg_total_cost_with_cache', 0.0):.6f} | "
            f"{opt_tasks} | "
            f"{O.get('avg_peak_tokens', 0.0):.1f} | "
            f"{O.get('avg_dependency', 0.0):.2f} | "
            f"{O.get('avg_total_cost_no_cache', 0.0):.6f} | "
            f"{O.get('avg_total_cost_with_cache', 0.0):.6f} |"
        )
        lines.append(row)
        
        # Accumulate for overall averages
        total_agent_tasks += agent_tasks
        total_agent_steps += A.get('avg_steps', 0.0) * agent_tasks
        total_agent_peak_tokens += A.get('avg_peak_tokens', 0.0) * agent_tasks
        total_agent_cost_no_cache += A.get('avg_total_cost_no_cache', 0.0) * agent_tasks
        total_agent_cost_with_cache += A.get('avg_total_cost_with_cache', 0.0) * agent_tasks
        total_opt_tasks += opt_tasks
        total_opt_peak_tokens += O.get('avg_peak_tokens', 0.0) * opt_tasks
        total_opt_cost_no_cache += O.get('avg_total_cost_no_cache', 0.0) * opt_tasks
        total_opt_cost_with_cache += O.get('avg_total_cost_with_cache', 0.0) * opt_tasks

    # Add overall average row
    if total_agent_tasks > 0 or total_opt_tasks > 0:
        avg_agent_steps = total_agent_steps / total_agent_tasks if total_agent_tasks > 0 else 0.0
        avg_agent_peak_tokens = total_agent_peak_tokens / total_agent_tasks if total_agent_tasks > 0 else 0.0
        avg_agent_cost_no_cache = total_agent_cost_no_cache / total_agent_tasks if total_agent_tasks > 0 else 0.0
        avg_agent_cost_with_cache = total_agent_cost_with_cache / total_agent_tasks if total_agent_tasks > 0 else 0.0
        avg_opt_peak_tokens = total_opt_peak_tokens / total_opt_tasks if total_opt_tasks > 0 else 0.0
        avg_opt_cost_no_cache = total_opt_cost_no_cache / total_opt_tasks if total_opt_tasks > 0 else 0.0
        avg_opt_cost_with_cache = total_opt_cost_with_cache / total_opt_tasks if total_opt_tasks > 0 else 0.0
        
        overall_row = (
            f"| **Overall Average** | "
            f"**{total_agent_tasks}** | "
            f"**{avg_agent_steps:.2f}** | "
            f"**{avg_agent_peak_tokens:.1f}** | "
            f"**{(total_agent_tasks and (sum((by_difficulty[d].get('agent', {}).get('avg_dependency',0.0) or 0.0) * (by_difficulty[d].get('agent', {}).get('num_tasks',0) or 0) for d in by_difficulty) / total_agent_tasks) or 0.0):.2f}** | "
            f"**{avg_agent_cost_no_cache:.6f}** | "
            f"**{avg_agent_cost_with_cache:.6f}** | "
            f"**{total_opt_tasks}** | "
            f"**{avg_opt_peak_tokens:.1f}** | "
            f"**{(total_opt_tasks and (sum((by_difficulty[d].get('optimizer', {}).get('avg_dependency',0.0) or 0.0) * (by_difficulty[d].get('optimizer', {}).get('num_tasks',0) or 0) for d in by_difficulty) / total_opt_tasks) or 0.0):.2f}** | "
            f"**{avg_opt_cost_no_cache:.6f}** | "
            f"**{avg_opt_cost_with_cache:.6f}** |"
        )
        lines.append(overall_row)
    
    lines.append("")
    return "\n".join(lines)


def build_markdown_cost_doc(title: str, by_difficulty: dict, by_fold_difficulty: dict) -> str:
    """Build a Markdown document with an overall table and per-fold tables.

    by_fold_difficulty: fold -> difficulty -> { 'agent': {...}, 'optimizer': {...} }
    """
    sections = []
    sections.append(build_markdown_cost_table(title, by_difficulty))

    # Known fold order; display only if present and has data
    preferred_order = ['train', 'dev', 'test_normal', 'test_challenge', 'unknown']
    # Include any other folds deterministically as well
    other_folds = [f for f in by_fold_difficulty.keys() if f not in preferred_order]
    fold_order = preferred_order + sorted(other_folds)

    for fold in fold_order:
        diff_map = by_fold_difficulty.get(fold)
        if not diff_map:
            continue
        # Determine if there's any data
        has_data = False
        for stats in diff_map.values():
            a = stats.get('agent', {})
            o = stats.get('optimizer', {})
            if (a.get('num_tasks', 0) or 0) > 0 or (o.get('num_tasks', 0) or 0) > 0:
                has_data = True
                break
        if not has_data:
            continue
        fold_title = f"{title} — {fold}"
        sections.append(build_markdown_cost_table(fold_title, diff_map))
    return "\n\n".join(sections)


def build_cost_csv(by_difficulty: dict, by_fold_difficulty: dict) -> str:
    """Build a CSV string for overall + per-fold per-difficulty metrics.

    Columns:
    - fold: one of overall, train, dev, test_normal, test_challenge, unknown, ...
    - difficulty
    - agent_num_tasks, agent_avg_steps, agent_avg_peak_tokens,
      agent_avg_total_cost_no_cache, agent_avg_total_cost_with_cache
    - optimizer_num_tasks, optimizer_avg_peak_tokens,
      optimizer_avg_total_cost_no_cache, optimizer_avg_total_cost_with_cache
    """
    def sort_key(k: str):
        try:
            return (0, int(str(k)))
        except Exception:
            return (1, str(k))

    headers = [
        'fold', 'difficulty',
        'agent_num_tasks', 'agent_avg_steps', 'agent_avg_peak_tokens', 'agent_avg_dependency',
        'agent_avg_total_cost_no_cache', 'agent_avg_total_cost_with_cache',
        'optimizer_num_tasks', 'optimizer_avg_peak_tokens', 'optimizer_avg_dependency',
        'optimizer_avg_total_cost_no_cache', 'optimizer_avg_total_cost_with_cache',
    ]
    lines = [','.join(headers)]

    # Calculate overall averages across all difficulties
    def add_overall_avg(lines_list, by_diff_dict, fold_name):
        total_agent_tasks = 0
        total_agent_steps = 0.0
        total_agent_peak_tokens = 0.0
        total_agent_cost_no_cache = 0.0
        total_agent_cost_with_cache = 0.0
        total_opt_tasks = 0
        total_opt_peak_tokens = 0.0
        total_opt_cost_no_cache = 0.0
        total_opt_cost_with_cache = 0.0
        
        for diff in sorted(by_diff_dict.keys(), key=sort_key):
            stats = by_diff_dict.get(diff, {}) or {}
            A = stats.get('agent', {}) or {}
            O = stats.get('optimizer', {}) or {}
            
            agent_tasks = A.get('num_tasks', 0)
            opt_tasks = O.get('num_tasks', 0)
            
            total_agent_tasks += agent_tasks
            total_agent_steps += A.get('avg_steps', 0.0) * agent_tasks
            total_agent_peak_tokens += A.get('avg_peak_tokens', 0.0) * agent_tasks
            total_agent_cost_no_cache += A.get('avg_total_cost_no_cache', 0.0) * agent_tasks
            total_agent_cost_with_cache += A.get('avg_total_cost_with_cache', 0.0) * agent_tasks
            total_opt_tasks += opt_tasks
            total_opt_peak_tokens += O.get('avg_peak_tokens', 0.0) * opt_tasks
            total_opt_cost_no_cache += O.get('avg_total_cost_no_cache', 0.0) * opt_tasks
            total_opt_cost_with_cache += O.get('avg_total_cost_with_cache', 0.0) * opt_tasks

        if total_agent_tasks > 0 or total_opt_tasks > 0:
            avg_agent_steps = total_agent_steps / total_agent_tasks if total_agent_tasks > 0 else 0.0
            avg_agent_peak_tokens = total_agent_peak_tokens / total_agent_tasks if total_agent_tasks > 0 else 0.0
            avg_agent_cost_no_cache = total_agent_cost_no_cache / total_agent_tasks if total_agent_tasks > 0 else 0.0
            avg_agent_cost_with_cache = total_agent_cost_with_cache / total_agent_tasks if total_agent_tasks > 0 else 0.0
            avg_opt_peak_tokens = total_opt_peak_tokens / total_opt_tasks if total_opt_tasks > 0 else 0.0
            avg_opt_cost_no_cache = total_opt_cost_no_cache / total_opt_tasks if total_opt_tasks > 0 else 0.0
            avg_opt_cost_with_cache = total_opt_cost_with_cache / total_opt_tasks if total_opt_tasks > 0 else 0.0
            
            dep_weighted_total = 0.0
            for diff in by_diff_dict.keys():
                stats = by_diff_dict.get(diff, {}) or {}
                A = stats.get('agent', {}) or {}
                dep_weighted_total += (A.get('avg_dependency', 0.0) or 0.0) * (A.get('num_tasks', 0) or 0)
            avg_dep = dep_weighted_total / total_agent_tasks if total_agent_tasks > 0 else 0.0
            opt_dep_weighted_total = 0.0
            for diff in by_diff_dict.keys():
                stats = by_diff_dict.get(diff, {}) or {}
                O = stats.get('optimizer', {}) or {}
                opt_dep_weighted_total += (O.get('avg_dependency', 0.0) or 0.0) * (O.get('num_tasks', 0) or 0)
            avg_opt_dep = opt_dep_weighted_total / total_opt_tasks if total_opt_tasks > 0 else 0.0
            overall_row = [
                fold_name, 'Overall Average',
                str(total_agent_tasks), f"{avg_agent_steps:.2f}", f"{avg_agent_peak_tokens:.1f}", f"{avg_dep:.2f}",
                f"{avg_agent_cost_no_cache:.6f}", f"{avg_agent_cost_with_cache:.6f}",
                str(total_opt_tasks), f"{avg_opt_peak_tokens:.1f}", f"{avg_opt_dep:.2f}",
                f"{avg_opt_cost_no_cache:.6f}", f"{avg_opt_cost_with_cache:.6f}",
            ]
            lines_list.append(','.join(overall_row))

    # Overall rows
    for diff in sorted(by_difficulty.keys(), key=sort_key):
        stats = by_difficulty.get(diff, {}) or {}
        A = stats.get('agent', {}) or {}
        O = stats.get('optimizer', {}) or {}
        row = [
            'overall', str(diff),
            str(A.get('num_tasks', 0)), f"{A.get('avg_steps', 0.0):.2f}", f"{A.get('avg_peak_tokens', 0.0):.1f}", f"{A.get('avg_dependency', 0.0):.2f}",
            f"{A.get('avg_total_cost_no_cache', 0.0):.6f}", f"{A.get('avg_total_cost_with_cache', 0.0):.6f}",
            str(O.get('num_tasks', 0)), f"{O.get('avg_peak_tokens', 0.0):.1f}", f"{O.get('avg_dependency', 0.0):.2f}",
            f"{O.get('avg_total_cost_no_cache', 0.0):.6f}", f"{O.get('avg_total_cost_with_cache', 0.0):.6f}",
        ]
        lines.append(','.join(row))
    
    # Add overall average for overall section
    add_overall_avg(lines, by_difficulty, 'overall')

    # Per-fold rows
    preferred_order = ['train', 'dev', 'test_normal', 'test_challenge', 'unknown']
    other_folds = [f for f in by_fold_difficulty.keys() if f not in preferred_order]
    fold_order = preferred_order + sorted(other_folds)
    for fold in fold_order:
        diff_map = by_fold_difficulty.get(fold)
        if not diff_map:
            continue
        # Only include folds with any data
        has_data = False
        for stats in diff_map.values():
            a = stats.get('agent', {})
            o = stats.get('optimizer', {})
            if (a.get('num_tasks', 0) or 0) > 0 or (o.get('num_tasks', 0) or 0) > 0:
                has_data = True
                break
        if not has_data:
            continue
        for diff in sorted(diff_map.keys(), key=sort_key):
            stats = diff_map.get(diff, {}) or {}
            A = stats.get('agent', {}) or {}
            O = stats.get('optimizer', {}) or {}
            row = [
                str(fold), str(diff),
                str(A.get('num_tasks', 0)), f"{A.get('avg_steps', 0.0):.2f}", f"{A.get('avg_peak_tokens', 0.0):.1f}", f"{A.get('avg_dependency', 0.0):.2f}",
                f"{A.get('avg_total_cost_no_cache', 0.0):.6f}", f"{A.get('avg_total_cost_with_cache', 0.0):.6f}",
                str(O.get('num_tasks', 0)), f"{O.get('avg_peak_tokens', 0.0):.1f}", f"{O.get('avg_dependency', 0.0):.2f}",
                f"{O.get('avg_total_cost_no_cache', 0.0):.6f}", f"{O.get('avg_total_cost_with_cache', 0.0):.6f}",
            ]
            lines.append(','.join(row))
        
        # Add overall average for this fold
        add_overall_avg(lines, diff_map, str(fold))

    return "\n".join(lines)


__all__ = [
    'MODEL_PRICING',
    '_get_pricing',
    'calculate_cost_components',
    '_get_encoding',
    '_common_prefix_len',
    'calculate_llm_tokens_with_cache',
    'calculate_optimizer_tokens_with_cache',
    'EXPERIMENTS_ROOT',
    'APPWORLD_BASE_PATH',
    'APPWORLD_BASE_PATHS',
    'OFFICEBENCH_BASE_PATH',
    'OFFICEBENCH_BASE_PATHS',
    'SMOLAGENTS_BASE_PATH',
    'SMOLAGENTS_BASE_PATHS',
    'get_file_hash',
    'get_cache_key',
    'load_token_cache',
    'save_token_cache',
    'calculate_env_steps',
    'load_experiment_evaluation',
    'detect_experiment_type',
    'get_available_experiments',
    'get_task_difficulty',
    'get_task_fold',
    'analyze_experiment_tokens_v2',
    'build_markdown_cost_table',
    'build_markdown_cost_doc',
    'build_cost_csv',
]
