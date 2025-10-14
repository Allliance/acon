# 8-objective QA Experiments

This folder contains all experiment scripts and configs for 8-objective QA benchmarks based on [`smolagents`](https://github.com/huggingface/smolagents).


## 1. Environment & Data Setup

Our implementation is based on [`smolagents`](https://github.com/huggingface/smolagents).

Install dependencies:

```bash
pip install smolagents
```

Download and setup the retriever index following [Search-R1](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md#local-sparse-retriever):
```bash
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir search/database/wikipedia
huggingface-cli download PeterJinGo/wiki-18-corpus --filename wiki-18.jsonl.gz --repo-type dataset --local-dir search/database/wikipedia
gzip -d search/database/wikipedia/wiki-18.jsonl.gz
```

Run the retriever server:

```bash
python search/retriever_server.py --index_path search/database/wikipedia/bm25
```

## 2. Running Experiments

All experiments are located under `experiments/smolagents`.
Outputs are stored in `experiments/smolagents/outputs/<model>_<tag>`.

Example run (with context compression):

```bash
cd experiments/smolagents
python run_all.py \
    --model_name gpt-4.1 \
    --tag baseline \
    --co_config_path configs/context_opt/gpt-4.1_history.yaml
```

## 3. Dataset Preparation

To convert evaluation logs into AppWorld-style datasets and prepare for training, use the following script:

```bash
python evaluate_to_appworld_format.py
cd ../training
python save_trajectories_dataset.py \
    --task smolagents \
    --folders gpt-4.1_history_compression \
    --file-types llm_history,history_optimizer_history \
    --outputs-root dataset \
    --split train \
    --min-f1 0.6 --require-success
```
## 4. Notes

All other experimental details — including **context optimization**,  
**prompt refinement**, and **distillation (compressor & agent)** —  
follow the **same structure and scripts** as described in [AppWorld](experiments/appworld/README.md).

Simply adjust paths (e.g., `experiments/smolagents/` instead of `experiments/appworld/`)  
and use the appropriate benchmark names when running the scripts.