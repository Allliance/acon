#!/usr/bin/env python3
"""
Minimal Smolagents + MuSiQue runner

- Loads minimal MuSiQue samples
- Builds SmolagentsEnv and SmolagentsAgent
- Evaluates with exact match on final answer
"""

import json
import os
import shutil
import time
import yaml
from dataclasses import asdict
from types import SimpleNamespace
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from eval_utils import exact_match, f1_max

from productive_agents.env.smolagents.env import SmolagentsEnv
from productive_agents.env.smolagents.config import SmolagentsEnvConfig
from productive_agents.agents.smolagents.agent import create_smolagents_agent
from productive_agents.agents.unified_agent import merge_configs
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

try:
    # When running as a module: experiments.smolagents.run
    from .dataset import QALoader, QAExample
except Exception:
    # When running as a script from this folder: python run.py
    from dataset import QALoader, QAExample


def _sanitize_for_path(name: str) -> str:
    # Keep alnum, dash, underscore, dot; replace others with '-'
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '-' for ch in name)


def _fmt_k(n: int) -> str:
    """Format a token count compactly: 8192 -> '8k', 4096 -> '4k', 1500 -> '1500'."""
    if n is None:
        return "x"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)

def run_sample(
    ex,
    model_name: str,
    max_iter: int,
    debug: bool,
    output_base: str,
    lora_name: Optional[str] = None,
    model_ctxopt: Optional[Any] = None,
    co_config: Optional[Dict[str, Any]] = None,
    experiment_name: str = "smolagents_musique",
    causal_generation: bool = False,
    seed: int = 42,
):
    os.makedirs(output_base, exist_ok=True)

    env_cfg = SmolagentsEnvConfig(
        experiment_name=experiment_name,
        max_interactions=max_iter,
        debug_mode=debug,
        local_workdir=os.path.join(output_base, "workdir"),
    )
    env = SmolagentsEnv(config=env_cfg)

    # Task: we pass the question as the instruction.
    # Seed is fixed across repetitions so the only run-to-run variation is
    # residual vLLM non-determinism (batching / float), which repeats average out.
    env.reset(seed=seed, task=ex.question)

    # Build exp config as attributes (MemoryManager expects attribute access)
    agent_cfg = SimpleNamespace(
        debug_mode=debug,
        max_iter=max_iter,
        co_config=co_config,
        causal_generation=causal_generation,
    )

    task_cfg = {
        "task_id": ex.id,
        "question": ex.question,
    }

    agent = create_smolagents_agent(
        model_name=model_name,
        key="",
        env=env,
        task_config=task_cfg,
    exp_config=agent_cfg,
        lora_name=lora_name,
    model_ctxopt=model_ctxopt,
        debug_mode=debug,
    )

    result = agent.run(env, max_iter=max_iter)
    # Try to pull final answer from info
    info = result.get("info", {}) or {}
    pred = info.get("final_answer_raw") or info.get("final_answer") or ""

    # Persist traces
    sample_dir = os.path.join(output_base, ex.id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(sample_dir, "sample.json"), "w", encoding="utf-8") as f:
        json.dump({
            "id": ex.id,
            "question": ex.question,
            "answer": ex.answer,
            "prediction": pred,
            "result": result,
        }, f, indent=2, ensure_ascii=False)

    # Save histories
    agent.dump_history(sample_dir)
    env.dump_history(sample_dir)

    # Causal generation: persist the (sample_id, posterior_step, prior_step, info) tuples.
    if causal_generation and hasattr(agent, "dump_causal_pairs"):
        agent.dump_causal_pairs(sample_dir)

    # Write human-readable trajectory log
    _dump_trajectory_text(
        sample_dir=sample_dir,
        question=ex.question,
        answer=ex.answer,
        prediction=pred,
        llm_history=agent.memory_manager.llm_history,
        env_trajectory=env.trajectory,
    )

    return pred, result


def _dump_trajectory_text(
    sample_dir: str,
    question: str,
    answer: Any,
    prediction: str,
    llm_history: list,
    env_trajectory: list,
) -> None:
    """Write a readable turn-by-turn trajectory to trajectory.txt."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"TASK:\n{question}")
    lines.append(f"\nGOLD ANSWER:\n{answer}")
    lines.append(f"\nPREDICTION:\n{prediction or '(none)'}")
    lines.append("=" * 80)

    # Flatten all sessions into one stream; mark session breaks
    step = 0
    for s_idx, session in enumerate(llm_history):
        if s_idx > 0:
            lines.append(f"\n{'─'*40} [Session {s_idx} — after compression] {'─'*40}\n")
        i = 0
        while i < len(session):
            msg = session[i]
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lines.append(f"[SYSTEM PROMPT — {len(content)} chars, omitted]\n")
                i += 1
            elif role == "user":
                if i <= 1 and s_idx == 0:
                    # First user turn is the task instruction — already shown above
                    lines.append(f"[TASK PROMPT — {len(content)} chars, omitted]\n")
                else:
                    lines.append(f"── Observation ──\n{content}\n")
                i += 1
            elif role == "assistant":
                step += 1
                lines.append(f"{'━'*60}")
                lines.append(f"Step {step}")
                lines.append(f"{'━'*60}")
                lines.append(content)
                lines.append("")
                i += 1

    lines.append("=" * 80)
    log_path = os.path.join(sample_dir, "trajectory.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


_TOKENIZER = None


def _count_tokens(text: str) -> int:
    """Token count using the same encoding the framework uses for thresholds.

    Matches ctxopt/base.py: gpt-4o encoding, falling back to cl100k_base, and
    finally a whitespace-word approximation if tiktoken is unavailable.
    """
    global _TOKENIZER
    if not text:
        return 0
    if _TOKENIZER is None:
        try:
            import tiktoken
            try:
                _TOKENIZER = tiktoken.encoding_for_model("gpt-4o")
            except Exception:
                _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKENIZER = False  # sentinel: tiktoken unavailable
    if _TOKENIZER is False:
        return len(text.split())
    return len(_TOKENIZER.encode(text))


def _safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _compute_round_metrics(samples_dir: str) -> Dict[str, float]:
    """Aggregate trajectory / compression metrics over all samples in a round.

    Per-sample signals (all token counts via tiktoken):
      * compression_turns       = len(llm_history) - 1
                                  (number of session resets; works for both
                                  LLM-based and selection-based baselines).
      * num_steps               = smolagents_trajectory.num_interactions
                                  (falls back to len(env_history)).
      * raw_trajectory_len      = sum of tokens over env_history of
                                  (action + observation); excludes compression.
      * compression_lengths     = [tokens(entry[2]) for entry in
                                  history_optimizer_history] — the compressor
                                  outputs (empty for selection-only baselines,
                                  which do not emit generated summaries).

    Round-level metrics:
      avg_compression_turns      mean compression_turns over samples
      avg_num_steps              mean num_steps over samples
      avg_raw_trajectory         mean raw_trajectory_len over samples
      avg_compression_length     mean over every compression event (token len)
      avg_max_compression_length mean over samples of that sample's max
                                 compression-output token length
    """
    if not os.path.isdir(samples_dir):
        return {}

    comp_turns, num_steps, raw_lens = [], [], []
    per_sample_max_comp, all_comp_lens = [], []

    for name in sorted(os.listdir(samples_dir)):
        sdir = os.path.join(samples_dir, name)
        if not os.path.isdir(sdir):
            continue

        lh = _safe_load_json(os.path.join(sdir, "llm_history.json"))
        if isinstance(lh, list):
            comp_turns.append(max(len(lh) - 1, 0))

        traj = _safe_load_json(os.path.join(sdir, "smolagents_trajectory.json"))
        eh = _safe_load_json(os.path.join(sdir, "env_history.json"))
        if isinstance(traj, dict) and isinstance(traj.get("num_interactions"), int):
            num_steps.append(traj["num_interactions"])
        elif isinstance(eh, list):
            num_steps.append(len(eh))

        if isinstance(eh, list):
            tot = 0
            for step in eh:
                if isinstance(step, dict):
                    tot += _count_tokens(str(step.get("action", "")))
                    tot += _count_tokens(str(step.get("observation", "")))
            raw_lens.append(tot)

        hoh = _safe_load_json(os.path.join(sdir, "history_optimizer_history.json"))
        sample_comp_lens = []
        if isinstance(hoh, list):
            for entry in hoh:
                if isinstance(entry, (list, tuple)) and len(entry) >= 3 and isinstance(entry[2], str):
                    sample_comp_lens.append(_count_tokens(entry[2]))
        all_comp_lens.extend(sample_comp_lens)
        per_sample_max_comp.append(max(sample_comp_lens) if sample_comp_lens else 0)

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    return {
        "avg_compression_turns": _mean(comp_turns),
        "avg_num_steps": _mean(num_steps),
        "avg_raw_trajectory": _mean(raw_lens),
        "avg_compression_length": _mean(all_comp_lens),
        "avg_max_compression_length": _mean(per_sample_max_comp),
    }


# Keys aggregated (mean + variance) across repetition rounds.
_AGG_KEYS = [
    "avg_em",
    "avg_f1",
    "avg_compression_turns",
    "avg_num_steps",
    "avg_raw_trajectory",
    "avg_compression_length",
    "avg_max_compression_length",
]


class _NullProgress:
    """Drop-in for rich.Progress when rounds run concurrently.

    Concurrent rich Live displays clash in one terminal, so parallel rounds
    suppress the bar and rely on each round's eval.log for progress.
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def add_task(self, *a, **k):
        return 0

    def advance(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    @property
    def console(self):
        return Console()


def main(
    split: str = "dev",
    output_dir: str = "outputs/smolagents_musique",
    model_name: str = "gpt-4o-mini",
    max_iter: int = 40,
    repeat: int = 3,
    parallel_rounds: bool = False,
    seed: int = 42,
    limit: Optional[int] = None,
    debug: bool = False,
    lora_name: Optional[str] = None,
    tag: Optional[str] = None,
    data_folder: Optional[str] = None,
    co_config_path: Optional[str] = None,
    id_list_file: Optional[str] = None,
    num_workers: int = 100,
    causal_generation: bool = False,
    history_threshold: int = 8192,
    compression_budget: int = 4096,
):
    # Resolve dataset path from split/data_folder if provided.
    # Desired: choose file by split (train/test), reading from data_folder.
    if data_folder:
        f = (split or "test").lower()
        # Map common aliases
        if f in {"dev", "validation", "val"}:
            f = "test"
        elif f not in {"train", "test"}:
            f = "test"
        root = data_folder or "data/4hop_hf"
        this_dir = os.path.dirname(__file__)
        root_abs = root if os.path.isabs(root) else os.path.abspath(os.path.join(this_dir, root))
        fname = "train.jsonl" if f == "train" else "test.jsonl"
        resolved_data_path = os.path.join(root_abs, fname)
        # Normalize split to resolved fold alias for downstream naming
        split = f

    # Load context optimization config if provided
    co_config = None
    model_ctxopt = None
    if co_config_path:
        if os.path.exists(co_config_path):
            try:
                with open(co_config_path, "r") as f:
                    co_config = yaml.safe_load(f)
            except Exception:
                co_config = None
        else:
            print(f"Warning: co_config_path {co_config_path} not found; continuing without ctxopt.")

    # Override threshold/budget from CLI so the tag matches the actual run.
    # These are always set (defaults: 8192 / 4096) so behavior is reproducible
    # even when the config doesn't declare them.
    if co_config is not None:
        co_config["history_summarization_threshold"] = history_threshold
        co_config["obs_summarization_threshold"] = history_threshold
        co_config["compression_budget"] = compression_budget

    # Initialize local ctxopt model if requested
    if co_config and co_config.get("model_type") == "local":
        try:
            from productive_agents.llm import vLLMLocal
            model_ctxopt = vLLMLocal(co_config["model"], lora_path=co_config.get("lora_name"))
        except Exception as e:
            print(f"Warning: failed to initialize local ctxopt model: {e}")
            model_ctxopt = None

    # Optional filtering: load ID list if provided
    filter_ids = None
    if id_list_file:
        if os.path.exists(id_list_file):
            with open(id_list_file, 'r', encoding='utf-8') as f:
                filter_ids = {line.strip() for line in f if line.strip() and not line.strip().startswith('#')}
            if not filter_ids:
                print(f"Warning: id_list_file {id_list_file} contained no usable IDs; proceeding without filtering.")
                filter_ids = None
        else:
            print(f"Warning: id_list_file {id_list_file} not found; proceeding without filtering.")
            filter_ids = None

    # Dataset: file-backed if provided (resolved or explicit), otherwise a tiny inline demo set
    if 'resolved_data_path' in locals() and resolved_data_path:
        loader = QALoader(resolved_data_path)
        # If filtering, materialize and filter for accurate counts; else use streaming iterator
        if filter_ids is not None:
            materialized = [ex for ex in loader.iter(limit=None) if ex.id in filter_ids]
            if limit is not None:
                materialized = materialized[:limit]
            iterator = materialized
            total_count = len(materialized)
        else:
            iterator = loader.iter(limit=limit)
            # Best-effort total for ETA
            try:
                total_count = loader_count = None
                if hasattr(loader, 'count'):
                    loader_count = loader.count(limit=limit)
                total_count = loader_count
            except Exception:
                total_count = None
    else:  # demo data path
        demo = [
            QAExample(id="demo1", question="Where did the leader of the largest European country after the collapse of the country that denied anything more than an advisory role in the Korean war die?", answer="Moscow"),
            QAExample(id="demo2", question="What is the capital of France?", answer="Paris"),
        ]
        if filter_ids is not None:
            demo = [ex for ex in demo if ex.id in filter_ids]
        demo_list = demo[: limit or len(demo)]
        iterator = demo_list
        total_count = len(demo_list)

    # Materialize the dataset once so every repetition round runs the exact
    # same set of examples (datasets here are small — ~100 samples).
    examples = list(iterator)
    total_count = len(examples)
    _resolved_data_path = resolved_data_path if 'resolved_data_path' in locals() else None

    # Save under outputs/{model_name}_{tag}/{split}_{round}.
    # Threshold / budget / worker count are appended ONLY when overridden from
    # the defaults (8k / 4k / 128) so default runs get a clean tag.
    model_part = _sanitize_for_path(model_name)
    base_tag = _sanitize_for_path(tag) if tag else "notag"
    param_bits = []
    if history_threshold != 8192:    param_bits.append(f"t{_fmt_k(history_threshold)}")
    if compression_budget != 4096:   param_bits.append(f"b{_fmt_k(compression_budget)}")
    if num_workers != 100:           param_bits.append(f"w{num_workers}")
    tag_part = base_tag + ("_" + "_".join(param_bits) if param_bits else "")
    split_part = _sanitize_for_path((split or "test").lower())
    outputs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    # Fixed run dir; repetitions live in {split}_1 / {split}_2 / ... beneath it.
    run_dir = os.path.join(outputs_root, f"{model_part}_{tag_part}")
    os.makedirs(run_dir, exist_ok=True)
    experiment_name = f"smolagents_musique_{split_part}" + (f"_{tag_part}" if tag_part else "")

    def _score_and_record(ex, pred_raw, result):
        pred = [p.strip() for p in (pred_raw or "").split(";")]
        em_list = [exact_match(_pred, _answer) for _pred, _answer in zip(pred, ex.answer)]
        f1_list = [f1_max(_pred, _answer) for _pred, _answer in zip(pred, ex.answer)]
        em_score = sum(em_list) / len(ex.answer)
        f1_score = sum(f1_list) / len(ex.answer)
        row = {
            "id": ex.id,
            "question": ex.question,
            "answer": ex.answer,
            "prediction": pred,
            "em": em_score,
            "f1": f1_score,
            "iterations": result.get("iterations", 0) if result else 0,
            "success": result.get("success", False) if result else False,
        }
        return em_score, f1_score, row

    def _round_complete(round_dir: str) -> Optional[Dict[str, Any]]:
        """Return a finished round's summary (with metrics) if it looks done.

        A round counts as complete when its summary.json exists with the
        expected sample total. Metrics are recomputed from samples/ if the
        stored summary predates them (resume / older runs)."""
        sp = os.path.join(round_dir, "summary.json")
        s = _safe_load_json(sp)
        if not isinstance(s, dict):
            return None
        if s.get("total", 0) != total_count or total_count == 0:
            return None
        if any(k not in s for k in ("avg_compression_turns", "avg_max_compression_length")):
            s.update(_compute_round_metrics(os.path.join(round_dir, "samples")))
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
        return s

    def _run_round(round_idx: int, use_progress: bool = True) -> Dict[str, Any]:
        output_dir = os.path.join(run_dir, f"{split_part}_{round_idx}")
        # Partial / crashed round (dir exists but no valid summary) → redo it.
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if co_config_path and os.path.exists(co_config_path):
            try:
                shutil.copy2(co_config_path, os.path.join(output_dir, os.path.basename(co_config_path)))
            except Exception as e:
                print(f"Warning: failed to copy co_config to output dir: {e}")

        log_fh = open(os.path.join(output_dir, "eval.log"), "a", buffering=1, encoding="utf-8")

        def _log(msg: str) -> None:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_fh.write(f"[{ts}] {msg}\n")

        n = 0
        correct = 0.0
        f1_sum = 0.0
        all_rows = []

        def _run_one(ex):
            try:
                pred_raw, result = run_sample(
                    ex=ex,
                    model_name=model_name,
                    max_iter=max_iter,
                    debug=debug,
                    output_base=os.path.join(output_dir, "samples"),
                    lora_name=lora_name,
                    model_ctxopt=model_ctxopt,
                    co_config=co_config,
                    experiment_name=experiment_name,
                    causal_generation=causal_generation,
                    seed=seed,
                )
                return ex, pred_raw, result, None
            except Exception as e:  # keep one bad sample from killing the whole run
                return ex, "", {}, e

        _log(f"Starting round {round_idx}/{repeat}: model={model_name} tag={tag} "
             f"split={split_part} total={total_count} num_workers={num_workers} "
             f"max_iter={max_iter} seed={seed} co_config={co_config_path}")
        start_ts = time.time()

        progress = (
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
            )
            if use_progress
            else _NullProgress()
        )
        with progress:
            task_id = progress.add_task(f"MuSiQue round {round_idx}/{repeat}", total=total_count)
            try:
                if num_workers and num_workers > 1:
                    # Parallel execution: each sample is independent (own env,
                    # agent, MemoryManager). vLLM clients are thread-safe.
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    with ThreadPoolExecutor(max_workers=num_workers) as pool:
                        futures = [pool.submit(_run_one, ex) for ex in examples]
                        for fut in as_completed(futures):
                            ex, pred_raw, result, err = fut.result()
                            if err is not None:
                                progress.console.print(f"[red]Sample {ex.id} failed: {err}")
                                _log(f"Sample {ex.id} FAILED: {err}")
                            em_score, f1_score, row = _score_and_record(ex, pred_raw, result)
                            correct += em_score
                            f1_sum += f1_score
                            n += 1
                            all_rows.append(row)
                            progress.advance(task_id, 1)
                            elapsed = time.time() - start_ts
                            rate = n / elapsed if elapsed > 0 else 0.0
                            remaining = (total_count - n) / rate if rate > 0 and total_count else 0
                            _log(
                                f"[{n}/{total_count}] id={ex.id} em={em_score:.2f} f1={f1_score:.2f} "
                                f"iters={row['iterations']} | "
                                f"running_em={correct/n:.3f} running_f1={f1_sum/n:.3f} | "
                                f"elapsed={timedelta(seconds=int(elapsed))} eta={timedelta(seconds=int(remaining))}"
                            )
                else:
                    for ex in examples:
                        ex, pred_raw, result, err = _run_one(ex)
                        if err is not None:
                            progress.console.print(f"[red]Sample {ex.id} failed: {err}")
                            _log(f"Sample {ex.id} FAILED: {err}")
                        em_score, f1_score, row = _score_and_record(ex, pred_raw, result)
                        correct += em_score
                        f1_sum += f1_score
                        n += 1
                        all_rows.append(row)
                        progress.advance(task_id, 1)
                        elapsed = time.time() - start_ts
                        rate = n / elapsed if elapsed > 0 else 0.0
                        remaining = (total_count - n) / rate if rate > 0 and total_count else 0
                        _log(
                            f"[{n}/{total_count}] id={ex.id} em={em_score:.2f} f1={f1_score:.2f} "
                            f"iters={row['iterations']} | "
                            f"running_em={correct/n:.3f} running_f1={f1_sum/n:.3f} | "
                            f"elapsed={timedelta(seconds=int(elapsed))} eta={timedelta(seconds=int(remaining))}"
                        )
            except KeyboardInterrupt:
                progress.console.print("\nInterrupted by user (Ctrl+C). Finishing up...")
                _log("Interrupted by user (Ctrl+C).")

        summary = {
            "total": n,
            "avg_em": (correct / n) if n else 0.0,
            "avg_f1": (f1_sum / n) if n else 0.0,
            "model": model_name,
            "split": split,
            "tag": tag,
            "round": round_idx,
            "repeat": repeat,
            "seed": seed,
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "limit": limit,
            "max_iter": max_iter,
            "co_config_path": co_config_path,
            "id_list_file": id_list_file,
            "history_threshold": history_threshold,
            "compression_budget": compression_budget,
            "num_workers": num_workers,
        }
        total_elapsed = time.time() - start_ts
        summary["wall_time_seconds"] = total_elapsed

        # Write predictions first so metrics can be recomputed from samples/.
        with open(os.path.join(output_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary.update(_compute_round_metrics(os.path.join(output_dir, "samples")))

        _log(
            f"Round {round_idx} complete: total={n} em={summary['avg_em']:.4f} "
            f"f1={summary['avg_f1']:.4f} comp_turns={summary['avg_compression_turns']:.2f} "
            f"steps={summary['avg_num_steps']:.2f} wall_time={timedelta(seconds=int(total_elapsed))}"
        )
        log_fh.close()

        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    # ── Repetition loop with resume ───────────────────────────────────────────
    # Completed rounds are reused; pending rounds run sequentially, or all at
    # once when --parallel_rounds is set (each pending round gets its own
    # sample ThreadPoolExecutor → up to len(pending) * num_workers in flight).
    summaries_by_round: Dict[int, Dict[str, Any]] = {}
    pending = []
    for r in range(1, repeat + 1):
        round_dir = os.path.join(run_dir, f"{split_part}_{r}")
        done = _round_complete(round_dir)
        if done is not None:
            print(f"[repeat] round {r}/{repeat}: reusing existing {os.path.basename(round_dir)}")
            summaries_by_round[r] = done
        else:
            pending.append(r)

    if parallel_rounds and len(pending) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"[repeat] running rounds {pending} in parallel "
              f"({num_workers} workers each); progress in each round's eval.log")
        with ThreadPoolExecutor(max_workers=len(pending)) as rpool:
            fut_to_r = {rpool.submit(_run_round, r, False): r for r in pending}
            for fut in as_completed(fut_to_r):
                r = fut_to_r[fut]
                summaries_by_round[r] = fut.result()
                print(f"[repeat] round {r}/{repeat}: done")
    else:
        for r in pending:
            print(f"[repeat] round {r}/{repeat}: running → {split_part}_{r}")
            summaries_by_round[r] = _run_round(r)

    round_summaries = [summaries_by_round[r] for r in sorted(summaries_by_round)]

    # ── Aggregate across rounds (mean + variance) ─────────────────────────────
    import statistics

    def _agg(values: list) -> Dict[str, Any]:
        return {
            "mean": statistics.fmean(values) if values else 0.0,
            "variance": statistics.variance(values) if len(values) >= 2 else 0.0,
            "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
            "values": values,
        }

    aggregate = {
        "model": model_name,
        "split": split,
        "tag": tag,
        "repeat": repeat,
        "rounds_completed": len(round_summaries),
        "seed": seed,
        "max_iter": max_iter,
        "total": total_count,
        "co_config_path": co_config_path,
        "history_threshold": history_threshold,
        "compression_budget": compression_budget,
        "num_workers": num_workers,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            k: _agg([float(s.get(k, 0.0)) for s in round_summaries])
            for k in _AGG_KEYS
        },
        "rounds": [
            {
                "round": s.get("round"),
                "dir": f"{split_part}_{s.get('round')}",
                **{k: s.get(k) for k in _AGG_KEYS},
                "wall_time_seconds": s.get("wall_time_seconds"),
            }
            for s in round_summaries
        ],
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(json.dumps(aggregate, indent=2))

    # Pretty aggregate table with rich
    console = Console()
    table = Table(title=f"Smolagents MuSiQue — {repeat}-round aggregate", show_lines=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Mean", style="magenta")
    table.add_column("Std", style="yellow")
    table.add_column("Per-round", style="green")
    for k in _AGG_KEYS:
        a = aggregate["metrics"][k]
        table.add_row(
            k,
            f"{a['mean']:.4f}",
            f"{a['std']:.4f}",
            ", ".join(f"{v:.3f}" for v in a["values"]),
        )
    console.print(table)
    meta = Table(show_lines=False)
    meta.add_column("Field", style="cyan", no_wrap=True)
    meta.add_column("Value", style="magenta")
    meta.add_row("Model", model_name)
    meta.add_row("Tag", str(tag) if tag else "-")
    meta.add_row("Split", str(split))
    meta.add_row("Repeat", f"{len(round_summaries)}/{repeat}")
    meta.add_row("Seed", str(seed))
    meta.add_row("Max Iter", str(max_iter))
    meta.add_row("Total", str(total_count))
    meta.add_row("Run Dir", run_dir)
    if _resolved_data_path:
        meta.add_row("Data", _resolved_data_path)
    console.print(meta)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run minimal Smolagents MuSiQue eval")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--output_dir", type=str, default="outputs/smolagents_multi_8")
    parser.add_argument("--model_name", type=str, default="gpt-4.1")
    parser.add_argument("--max_iter", type=int, default=40)
    parser.add_argument("--repeat", type=int, default=3, help="Number of repetition rounds; results land in {split}_1.. with an aggregate summary.json. Resumes by skipping rounds that already completed.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed used for every repetition round (env reset + sampling).")
    parser.add_argument("--parallel_rounds", action="store_true", help="Run all pending repetition rounds concurrently (each with its own --num_workers pool) instead of one after another. Per-round progress goes to eval.log.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lora_name", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None, help="Tag name to group outputs (saved under output_dir/tag)")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data/nq_multi_8",
        help="Folder containing dataset files (expects train_4hop.jsonl/test_4hop.jsonl)",
    )
    parser.add_argument("--co_config_path", type=str, default=None, help="Context optimization config file path")
    parser.add_argument("--id_list_file", type=str, default=None, help="Optional file containing example IDs (one per line) to restrict the run")
    parser.add_argument("--num_workers", type=int, default=100, help="Parallel sample workers (threads). Each runs one sample independently against the same vLLM endpoint. Baked into the output tag.")
    parser.add_argument("--history_threshold", type=int, default=8192, help="Token threshold that triggers history compression. Baked into the output tag (e.g. t8k). Overrides the config file.")
    parser.add_argument("--compression_budget", type=int, default=4096, help="Target token budget for the compressed history. Baked into the output tag (e.g. b4k). Overrides the config file.")
    parser.add_argument(
        "--causal_generation",
        action="store_true",
        help="Enable causal generation mode: agent emits per-step recall tags citing earlier steps; "
             "tags are stripped from stored history and persisted as causal_pairs.json per sample.",
    )

    args = parser.parse_args()

    main(
        split=args.split,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_iter=args.max_iter,
        repeat=args.repeat,
        parallel_rounds=args.parallel_rounds,
        seed=args.seed,
        limit=args.limit,
        debug=args.debug,
        lora_name=args.lora_name,
        tag=args.tag,
        data_folder=args.data_folder,
        co_config_path=args.co_config_path,
        id_list_file=args.id_list_file,
        num_workers=args.num_workers,
        causal_generation=args.causal_generation,
        history_threshold=args.history_threshold,
        compression_budget=args.compression_budget,
    )
