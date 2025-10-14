#!/usr/bin/env python3
"""
Minimal dashboard to visualize Smolagents trajectories and logs.

Features:
- Select a samples folder (defaults to experiments/smolagents/outputs/.../samples)
- Browse sample IDs, view question/answer/prediction
- Visualize smolagents_trajectory.json (actions/observations/rewards)
- Inspect llm_history.json as a chat and env_history.json as step logs

Run:
  streamlit run experiments/smolagents/dashboard.py

Dependencies:
  pip install streamlit
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import streamlit as st


# ---------- Utils ----------

def load_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_sample_dirs(samples_root: str) -> List[str]:
    if not os.path.isdir(samples_root):
        return []
    dirs = []
    for name in os.listdir(samples_root):
        full = os.path.join(samples_root, name)
        if os.path.isdir(full):
            dirs.append(name)
    return sorted(dirs)


def discover_sample_roots(outputs_root: str) -> List[str]:
    """Find all directories named 'samples' under outputs_root."""
    results: List[str] = []
    if not os.path.isdir(outputs_root):
        return results
    for root, dirs, files in os.walk(outputs_root):
        if os.path.basename(root) == "samples":
            results.append(os.path.abspath(root))
    results.sort()
    return results


def _normalize_answer(s: str) -> str:
    import re
    import string

    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: Any, gold: Any) -> bool:
    def norm_one(x: Any) -> str:
        if isinstance(x, (list, tuple)):
            x = x[0] if x else ""
        return _normalize_answer(str(x))

    p = norm_one(pred)
    if isinstance(gold, (list, tuple)):
        return any(p == norm_one(g) for g in gold)
    return p == norm_one(gold)


def f1_max(pred: Any, gold: Any) -> float:
    p = str(pred) if pred is not None else ""
    if isinstance(gold, (list, tuple)):
        return max((_f1_score(p, str(g)) for g in gold), default=0.0)
    return _f1_score(p, str(gold))


# ---------- Token counting (approx) ----------

def _get_tokenizer():
    try:
        import tiktoken  # type: ignore
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens_text(text: str) -> int:
    enc = _get_tokenizer()
    if enc is None:
        # Heuristic fallback: ~0.75 words per token
        words = len(text.split()) if text else 0
        return int(max(1, round(words / 0.75))) if words else 0
    try:
        return len(enc.encode(text or ""))
    except Exception:
        return len((text or "").split())


def count_tokens_messages(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        total += count_tokens_text(str(m.get("content", "")))
    return total


# ---------- Display helpers ----------

def _normalize_observation(obs: Any) -> str:
    # Ensure string
    if obs is None:
        return ""
    if not isinstance(obs, str):
        obs = str(obs)
    # Convert literal backslash-n sequences to real newlines (avoid double converting)
    # Heuristic: only do this if there are no real newlines yet or the ratio of literal to real is high
    literal = obs.count("\\n")
    real = obs.count("\n")
    if literal:
        # Replace literal escaped sequences
        obs = obs.replace("\\n", "\n")
    # Also unescape tabs if present
    if "\\t" in obs:
        obs = obs.replace("\\t", "\t")
    return obs

def _observation_widget(label: str, obs: Any, idx: int | None = None):
    text = _normalize_observation(obs)
    import hashlib
    base = f"{label}:{idx}" if idx is not None else label
    h = hashlib.blake2b((base + text).encode(), digest_size=8).hexdigest()
    st.text_area(label, text, height=min(400, 24 + 14 * (text.count('\n') + 1)), key=f"obs_{h}")


# ---------- UI ----------

st.set_page_config(page_title="Smolagents Trajectory Viewer", layout="wide")
st.title("Smolagents Trajectory Viewer")

outputs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
default_samples = os.path.abspath(os.path.join(outputs_root, "gpt-4.1_debug", "train", "samples"))

with st.sidebar:
    st.header("Browse")
    # Build dropdown options by discovering 'samples' directories
    candidates = discover_sample_roots(outputs_root)
    # Ensure default path appears first if present
    if os.path.isdir(default_samples) and default_samples not in candidates:
        candidates = [default_samples] + candidates
    if not candidates:
        st.warning("No 'samples' folders found under outputs. Please generate runs first.")
        st.stop()

    # Persist selected samples_root in session state
    if "samples_root" not in st.session_state:
        st.session_state.samples_root = candidates[0]

    samples_root = st.selectbox(
        "Samples folder",
        options=candidates,
        index=candidates.index(st.session_state.samples_root) if st.session_state.samples_root in candidates else 0,
        key="samples_root",
        help="Pick a 'samples' directory containing per-sample subfolders",
        # Show a shorter, relative path instead of a long absolute path
        format_func=lambda p: os.path.relpath(p, outputs_root),
    )

    all_dirs = list_sample_dirs(samples_root)
    if not all_dirs:
        st.warning("No sample directories found under the provided path.")
    filter_text = st.text_input("Filter IDs", value="", help="Substring filter for sample IDs")
    filtered = [d for d in all_dirs if filter_text.lower() in d.lower()]

    # Initialize session state for selected id
    if "selected_sample_id" not in st.session_state and filtered:
        st.session_state.selected_sample_id = filtered[0]

    # Ensure the persisted selection exists in the current filtered list
    if filtered:
        current_sel = st.session_state.get("selected_sample_id")
        if current_sel not in filtered:
            st.session_state.selected_sample_id = filtered[0]
        sel_index = filtered.index(st.session_state.selected_sample_id)
    else:
        sel_index = None

    # The selectbox controls the selected id
    selected = st.selectbox(
        "Sample ID",
        filtered,
        index=sel_index,
        key="selected_sample_id",
    )

    # Simple prev/next controls using session_state
    if filtered:
        idx = filtered.index(st.session_state.selected_sample_id) if st.session_state.selected_sample_id in filtered else 0
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("◀ Prev", use_container_width=True) and idx > 0:
                st.session_state.selected_sample_id = filtered[idx - 1]
        with col_next:
            if st.button("Next ▶", use_container_width=True) and idx < len(filtered) - 1:
                st.session_state.selected_sample_id = filtered[idx + 1]


if not selected:
    st.stop()

sample_dir = os.path.join(samples_root, selected)
st.subheader(f"Sample: {selected}")

# Load files
sample_json = load_json(os.path.join(sample_dir, "sample.json")) or {}
traj_json = load_json(os.path.join(sample_dir, "smolagents_trajectory.json")) or {}
env_hist = load_json(os.path.join(sample_dir, "env_history.json")) or []
llm_hist = load_json(os.path.join(sample_dir, "llm_history.json")) or []

# Prepare assistant messages list and code blocks (raw LLM actions)
def extract_first_code_block(text: str) -> Optional[str]:
    import re as _re
    # Match ```python ... ``` or ``` ... ```; non-greedy for content
    m = _re.search(r"```(?:python)?\n(.*?)```", text, flags=_re.DOTALL|_re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def extract_thought(text: str) -> Optional[str]:
    """Extracts the Thought: ... segment prior to the first code block, if present."""
    import re as _re
    m = _re.search(r"Thought:\s*(.*?)(?:```|$)", text, flags=_re.DOTALL|_re.IGNORECASE)
    if m:
        thought = m.group(1).strip()
        return thought if thought else None
    return None

assistant_msgs: List[Dict[str, Any]] = []
assistant_code_blocks: List[Optional[str]] = []
assistant_thoughts: List[Optional[str]] = []
assistant_input_tokens: List[int] = []
def _flatten_llm_history(raw: Any) -> List[Dict[str, Any]]:
    """Return a flat list of message dicts from possible nested llm_history structures.

    Accepts either:
      - A simple list of message dicts
      - A list containing one or more lists of message dicts
      - A mixed list of dicts and lists
    """
    if not isinstance(raw, list):
        return []
    flat: List[Dict[str, Any]] = []
    nested = any(isinstance(el, list) for el in raw)
    if nested:
        for el in raw:
            if isinstance(el, list):
                for msg in el:
                    if isinstance(msg, dict):
                        flat.append(msg)
            elif isinstance(el, dict):
                flat.append(el)
    else:
        for msg in raw:
            if isinstance(msg, dict):
                flat.append(msg)
    return flat

if llm_hist and isinstance(llm_hist, list):
    msgs = _flatten_llm_history(llm_hist)
    for idx, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "assistant":
            try:
                assistant_input_tokens.append(count_tokens_messages(msgs[:idx]))
            except Exception:
                assistant_input_tokens.append(0)
            assistant_msgs.append(m)
            content_str = str(m.get("content", ""))
            code = extract_first_code_block(content_str)
            assistant_code_blocks.append(code)
            assistant_thoughts.append(extract_thought(content_str))


# ---- Summary panel ----
q = sample_json.get("question")
ans = sample_json.get("answer")
pred = sample_json.get("prediction")

cols = st.columns([3, 2, 2, 2])
with cols[0]:
    st.markdown("### Question")
    st.write(q or "-")
with cols[1]:
    st.markdown("### Answer")
    st.write(ans)
with cols[2]:
    st.markdown("### Prediction")
    st.write(pred)
with cols[3]:
    st.markdown("### Metrics")
    if ans is not None and pred is not None:
        try:
            em = exact_match(pred, ans)
            f1 = f1_max(pred, ans)
            st.metric("Exact Match", "✅" if em else "❌")
            st.metric("F1", f"{f1:.3f}")
        except Exception:
            st.write("-")
    else:
        st.write("-")


st.divider()


# ---- Trajectory ----
st.markdown("## Trajectory")
if traj_json:
    meta_cols = st.columns(6)
    with meta_cols[0]:
        st.caption(f"Interactions: {traj_json.get('num_interactions', '-')}")
    with meta_cols[1]:
        st.caption(f"Completed: {traj_json.get('completed', '-')}")
    with meta_cols[2]:
        st.caption(f"Final reward: {traj_json.get('final_reward', '-')}")
    with meta_cols[3]:
        st.caption("Instruction shown below")
    with meta_cols[4]:
        if assistant_input_tokens:
            st.caption(f"Peak input tokens (approx): {max(assistant_input_tokens)}")
        else:
            st.caption("Peak input tokens: -")
    # Steps count column
    with meta_cols[5]:
        steps_preview = traj_json.get('trajectory', [])
        st.caption(f"Steps: {len(steps_preview)}")

    with st.expander("Task instruction", expanded=False):
        st.write(traj_json.get("task_instruction", "-"))

    steps: List[Dict[str, Any]] = traj_json.get("trajectory", [])
    for i, step in enumerate(steps, start=1):
        st.markdown(f"### Step {i}")
        a_cols = st.columns([3, 2])
        with a_cols[0]:
            # One Thought (if present)
            thought = assistant_thoughts[i-1] if i-1 < len(assistant_thoughts) else None
            if thought:
                st.caption("Thought")
                st.write(thought)

            # One Action: prefer LLM raw code, fallback to executed action
            raw = assistant_code_blocks[i-1] if i-1 < len(assistant_code_blocks) else None
            action_code = raw if raw else step.get("action", "")
            if action_code:
                st.caption("Action")
                st.code(action_code, language="python")
            elif assistant_msgs and i-1 < len(assistant_msgs):
                # Last resort: show message text if nothing else is available
                with st.expander("Raw LLM message", expanded=False):
                    st.text(assistant_msgs[i-1].get("content", ""))
        with a_cols[1]:
            st.caption("Reward / Done / Info")
            st.json({k: step.get(k) for k in ["reward", "done", "info"] if k in step})
            # Per-step input tokens
            if i-1 < len(assistant_input_tokens):
                st.metric("Input tokens (approx)", assistant_input_tokens[i-1])
        with st.expander("Observation / Logs", expanded=False):
            _observation_widget("Observation", step.get("observation", ""), i)
        st.divider()
else:
    st.info("No smolagents_trajectory.json found for this sample.")


# ---- LLM history (chat) ----
st.markdown("## LLM History")
if llm_hist and isinstance(llm_hist, list):
    msgs = _flatten_llm_history(llm_hist)
    if not msgs:
        st.info("llm_history.json present but no recognizable messages.")
    else:
        for msg in msgs:
            role = msg.get("role", "assistant") if isinstance(msg, dict) else "assistant"
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if role == "system":
                with st.expander("System prompt", expanded=False):
                    st.text(content)
            elif role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").write(content)
else:
    st.info("No llm_history.json found or empty.")


# ---- Env history (mirror of trajectory, if present) ----
st.markdown("## Env History")
if env_hist:
    st.caption(f"Env steps: {len(env_hist)}")
    for i, step in enumerate(env_hist, start=1):
        with st.expander(f"Step {i}"):
            st.caption("Action (code)")
            st.code(step.get("action", ""), language="python")
            st.caption("Observation / Logs")
            _observation_widget("Observation", step.get("observation", ""), i)
            st.caption("Meta")
            st.json({k: step.get(k) for k in ["reward", "done", "info"] if k in step})
else:
    st.info("No env_history.json found or empty.")
