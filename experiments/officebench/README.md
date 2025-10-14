# OfficeBench Experiments

This folder contains all experiment scripts and configs for OfficeBench benchmarks.


## 1. Environment & Data Setup

Our implementation is based on [OfficeBench](https://github.com/zlwang-cs/OfficeBench).
Instead of using Docker, we install dependencies locally.

Install with OfficeBench extras:

```bash
pip install -e .[officebench]
```

## 2. Running Experiments

All experiments are located under `experiments/officebench`.
Outputs are stored in `experiments/officebench/outputs/<model>_<tag>`.

Example run:

```bash
cd experiments/officebench
python run_all.py \
    --model_name gpt-4.1 \
    --split train \
    --tag baseline \
    --co_config_path configs/context_opt/gpt-4.1_history.yaml
```

Evaluate:

```bash
python -m evaluation.main --model_name=gpt-4.1 --tag_name=baseline --split=train
```

## 3. Notes

All other experimental details — including **context optimization**,  
**prompt refinement**, and **distillation (compressor & agent)** —  
are identical to the [AppWorld pipeline](#1-appworld).  

You can reuse the same scripts under `experiments/training/` and `experiments/prompt_optimizer/`.  
Just update the file paths to point to the OfficeBench experiment directories.
