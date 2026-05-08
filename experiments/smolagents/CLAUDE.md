# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Experiment scripts and configs for **8-objective QA benchmarks** built on top of Hugging Face's [`smolagents`](https://github.com/huggingface/smolagents) framework. Evaluates LLMs on multi-hop QA tasks (MuSiQue, NQ-Multi-8) with support for context compression (history/observation summarization) and a local Wikipedia retriever.

## Setup

```bash
pip install smolagents

# Download retriever index
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir search/database/wikipedia
huggingface-cli download PeterJinGo/wiki-18-corpus --filename wiki-18.jsonl.gz --repo-type dataset --local-dir search/database/wikipedia
gzip -d search/database/wikipedia/wiki-18.jsonl.gz

# Start retriever server (must be running before evaluation)
python search/retriever_server.py --index_path search/database/wikipedia/bm25
```

## Common Commands

```bash
# Run evaluation (basic)
python run.py --model_name gpt-4o-mini --split test --limit 10

# Run with context optimization
python run_all.py --model_name gpt-4.1 --tag baseline \
    --co_config_path configs/context_opt/gpt-4.1_history.yaml

# Run context optimization pipeline (generates configs, runs eval, collects metrics)
python run_ctxopt_pipeline.py \
    --prompts-dir prompts/context_opt \
    --model-name gpt-4.1 \
    --split train \
    --ctxopt-type history \
    --tag my_experiment

# Convert outputs to AppWorld evaluation format
python evaluate_to_appworld_format.py --file outputs/<run>/predictions.jsonl

# Prepare training dataset from trajectories
cd ../training
python save_trajectories_dataset.py \
    --task smolagents \
    --folders gpt-4.1_history_compression \
    --file-types llm_history,history_optimizer_history \
    --outputs-root dataset \
    --split train \
    --min-f1 0.6 --require-success

# Launch interactive results dashboard
streamlit run dashboard.py
```

Outputs land in `outputs/<model>_<tag>/`.

## Architecture

**Evaluation entry points:**
- `run.py` / `run_all.py` — instantiate `SmolagentsEnv` and `SmolagentsAgent` (from `productive_agents` library), iterate over a dataset, dump per-sample results
- `run_ctxopt_pipeline.py` — orchestrates multi-run sweeps over context optimization configs; generates YAML configs from Jinja prompts, then calls the evaluator

**Data:**
- `dataset.py` — `QALoader` (lazy JSONL/JSON iteration) + `QAExample` dataclass; input format: `{"id", "question", "answer", "contexts"}`
- `data/nq_multi_8/` — train/test splits and fold indices

**Evaluation:**
- `eval_utils.py` — SQuAD-style exact match and F1; supports answer lists
- `evaluate_to_appworld_format.py` — converts `predictions.jsonl` to AppWorld report (`evaluations/dev.json`) with aggregate and per-sample metrics

**Context optimization:**
- `configs/context_opt/*.yaml` — configures compression type (`history` or `obs`), model, prompt templates, thresholds (`history_summarization_threshold`, `preserve_last_k_turns`), and `history_summary_rule`
- `prompts/context_opt/*.jinja` — Jinja2 templates for system prompt, history compression, and observation compression

**Retrieval:**
- `search/retriever_server.py` — FastAPI server wrapping BM25/FAISS indexes with dense embeddings (e5-base-v2); supports configurable top-k

**Visualization:**
- `dashboard.py` — Streamlit app for browsing samples, trajectories, and LLM/env histories

## Data Flow

```
dataset.py → run.py → productive_agents (SmolagentsEnv + SmolagentsAgent)
                            ↕
              search/retriever_server.py (Wikipedia BM25/FAISS)
                            ↓
              outputs/<model>_<tag>/{predictions.jsonl, per-sample dirs}
                            ↓
              evaluate_to_appworld_format.py → evaluations/dev.json
```

## Notes

Context optimization, prompt refinement, and distillation follow the same structure as the AppWorld experiments (`../appworld/`). Adjust paths accordingly when referencing AppWorld scripts.
