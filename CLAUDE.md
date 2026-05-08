# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ACON** is a research framework for optimizing context compression in long-horizon LLM agents. It minimizes redundant memory growth while preserving task-relevant information, evaluated across AppWorld, OfficeBench, and an 8-objective QA benchmark.

Paper: [arXiv:2510.00615](https://arxiv.org/abs/2510.00615)

## Setup

```bash
pip install -e .
# Add OpenAI API key to configs/private_config.yaml
```

Optional benchmark dependencies:
```bash
pip install smolagents         # for 8-objective QA
# OfficeBench requires additional setup (pandas, PyMuPDF, Docker, MySQL)
```

## Commands

**Linting:**
```bash
ruff check .
ruff format .
```

**Tests:**
```bash
pytest
```

**Running experiments:**
```bash
# AppWorld (single task)
cd experiments/appworld && python run.py --task_id <id> --model_name gpt-4.1 --tag baseline

# AppWorld (batch)
cd experiments/appworld && python run_all.py --split train --model_name gpt-4.1 --tag baseline --co_config_path configs/context_opt/gpt-4.1_history.yaml

# OfficeBench
cd experiments/officebench && python run.py --task_dir tasks/1-1 --model_name gpt-4o

# Smolagents / 8-objective QA
cd experiments/smolagents && python run.py
```

## Architecture

### Package structure (`src/productive_agents/`)

```
agents/
  base.py              # BaseAgent + BasePromptBuilder abstract classes
  unified_agent.py     # Concrete agent consolidating common patterns
  memory.py            # MemoryManager: conversation history + optimizer integration
  utils.py             # LLMManager factory
  appworld/            # AppWorld-specific agent + config
  officebench/         # OfficeBench-specific agent + config
  smolagents/          # Smolagents-specific agent
env/
  base.py              # BaseEnv + BaseLanguageBasedEnv abstract classes
  appworld/            # AppWorld environment wrapper
  officebench/         # OfficeBench environment wrapper
ctxopt/
  base.py              # BaseContextOptimizer (Jinja2 prompt loading + tiktoken)
  history_optimizer.py # Summarizes conversation history
  obs_optimizer.py     # Compresses environment observations
llm.py                 # Unified LLM interface: ChatGPT, Gemini, vLLM, Azure
```

### Data flow

1. `experiments/<benchmark>/run.py` loads config YAML, instantiates an agent and environment.
2. Agent's `run()` loop calls `env.step(action)` and receives observations.
3. `MemoryManager` stores history and invokes `HistoryOptimizer` / `ObservationOptimizer` when token thresholds are exceeded.
4. Context optimizers use Jinja2 templates from `prompts/context_opt/` to call a compressor LLM.
5. Compressed context is fed back to the main agent LLM.

### Key abstractions

**BaseAgent** (`agents/base.py`): abstract methods are `_initialize_llm()`, `_initialize_prompt_builder()`, `_initialize_action_processor()`, `_build_context_sections()`, `_process_response()`. The `forward()` / `run()` methods are concrete.

**BaseContextOptimizer** (`ctxopt/base.py`): subclasses implement `process()`. Token counting uses tiktoken; prompts are Jinja2 templates.

**LLM interface** (`llm.py`): all models expose a unified `generate()` method. `o1`/`o3`/`o4`-family models require special parameter handling (no `temperature`, use `max_completion_tokens`).

### Configuration

Experiments are configured via YAML files:
- `experiments/<benchmark>/configs/base_config.yaml` — task runner defaults
- `experiments/<benchmark>/configs/context_opt/<model>_<type>.yaml` — compression settings

Key context optimization parameters:
- `history_summarization_threshold` / `obs_summarization_threshold`: token counts that trigger compression
- `preserve_last_k_turns`: recent turns exempt from summarization
- `history_summary_rule`: `"accumulate"` (append summaries) or `"reset"` (replace history)
- `compressor_type`: `"full"` or `"stepwise"`

### Output layout

```
experiments/<benchmark>/outputs/<model>_<tag>/<split>/
  experiment_summary.json
  task_<id>_<rep>/
    appworld_trajectory.json   # or env_history.json
    llm_history.json
    history_optimizer_history.json
    step_alignment.json
    results.json
```

### API keys

`configs/private_config.yaml` — loaded by `LLMManager` in `agents/utils.py`. Never commit real keys.
