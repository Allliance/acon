# AppWorld Experiments

This folder contains all experiment scripts and configs for AppWorld benchmarks.


## 1. Environment & Data Setup

Install the upstream [AppWorld](https://github.com/StonyBrookNLP/appworld) package and download its data before running any experiments.

```bash
git lfs install
git clone https://github.com/StonyBrookNLP/appworld
cd appworld
pip install -e .
appworld install --repo
appworld download data
```

Move the produced `data/` directory into `experiments/appworld/` so the local runners can resolve paths as expected:

```bash
mv data /path/to/acon/experiments/appworld/
```

## 2. Running Baseline Experiments

All AppWorld experiments are located under `experiments/appworld`.
Outputs are stored in `experiments/appworld/outputs/<model>_<tag>`.

Run the following command to execute an agent **without context compression**:

```bash
cd experiments/appworld
python run_all.py \
    --split train \
    --model_name gpt-4.1 \
    --tag baseline \
    --co_config_path configs/context_opt/appworld/gpt-4.1_history_v2.yaml
appworld evaluate gpt-4.1_baseline train
```

> 💡 We recommend running with the `--debug` flag first to verify your setup.

Run the following to execute an agent **with context compression (default prompt)**:

```bash
cd experiments/appworld
python run_all.py \
    --split train \
    --model_name gpt-4.1 \
    --tag history_compression \
    --co_config_path configs/context_opt/appworld/gpt-4.1_history_v2.yaml
appworld evaluate gpt-4.1_history_compression train
```

After running both commands above, proceed to compression guideline optimization.

**Key Arguments**

* `--model_name`: e.g., `gpt-4.1` (currently OpenAI models only)
* `--split`: `train | dev | test_normal | test_challenge`
* `--co_config_path`: context optimization (history / observation) config file
* `--tag`: experiment grouping label (suffix for output directory names)

---

**Output Layout**

```text
experiments/appworld/outputs/
    <model_name>_<tag>/
        train/
            experiment_summary.json
            task_<id>_<rep>/
                appworld_trajectory.json
                env_history.json
                history_optimizer_history.json
                llm_history.json
                step_alignment.json
                results.json
        dev/
        test_normal/
        test_challenge/
```

**Evaluation Results**
You can find evaluation summaries under
`experiments/appworld/outputs/<model>_<tag>/experiment_summary.json`.

## 3. Compression Guideline Optimization

Optimize the compression guideline using prompt optimization.

**Prerequisites**

* A baseline run directory (no compression)
* An optimized run directory using a candidate compression config

Example:

```bash
cd experiments/prompt_optimizer
python unified_update_history_prompt.py \
    --baseline-run gpt-4.1_baseline \
    --optimized-run gpt-4.1_history_compression \
    --benchmark appworld \
    --base-prompt-template ../appworld/prompts/context_opt/prompt_history_v2.jinja \
    --output-dir outputs_appworld/history_regression
```

This produces improved prompt variants under
`outputs_appworld/history_regression/optimized_prompts/`.

Validate and select the best compression prompt:

```bash
cd ../appworld
cp data_copy/datasets/* data/datasets/  # ensure subset of training dataset is present
bash scripts/run_ctxopt_history.sh \
    ../prompt_optimizer/outputs_appworld/history_regression/optimized_prompts
```

The pipeline creates a dated folder under
`experiments/appworld/configs/context_opt/` containing generated configs.
The file starting with `best_` is the chosen `co_config_path` for subsequent experiments:

```bash
python run_all.py \
    --split train \
    --model_name gpt-4.1 \
    --tag history_optimized_best \
    --co_config_path configs/context_opt/{YYMMDD}_gpt-4.1_history_optimized_prompt/best_improved_history_prompt_samples.yaml
```

## 4. Distillation Stage 1: Compressor Model

Train a **local model** (e.g., `Qwen3-14B`) to perform context compression.

**Steps**

1. Ensure a full `train` split run exists that used compression (e.g.,
   `experiments/appworld/outputs/gpt-4.1_history_optimized_best/train/`).

2. Export trajectories into a training dataset:

   ```bash
   cd experiments/training
   python save_trajectories_dataset.py -f gpt-4.1_history_compression -t history_optimizer_history
   ```

   Output:
   `dataset/history_optimizer_history/gpt-4.1_history_compression_train.jsonl`

3. Finetune the compressor LoRA:

   ```bash
   bash scripts/run_finetune.sh \
       Qwen/Qwen3-14B \
       dataset/history_optimizer_history/gpt-4.1_history_compression_train.jsonl \
       history_compression_3epochs
   ```

4. Serve the compressor:

   ```bash
   bash scripts/serve_single.sh \
       Qwen/Qwen3-14B \
       finetuned_models/qwen-14B/history_compression_3epochs
   ```

5. Example config (`configs/context_opt/qwen3-14B/history_baseline.yaml`):

   ```yaml
   type: history
   model: "Qwen/Qwen3-14B"
   lora_name: "finetune=finetuned_models/qwen-14B/history_compression_3epochs"
   compressor_type: full
   prompts:
       prompt_system: system_prompt
       prompt_history_user: prompt_history_v2
   history_summarization_threshold: 4096
   preserve_last_k_turns: 1
   history_summary_rule: reset
   ```

6. Evaluate with frontier or local inference:

   ```bash
   cd ../appworld
   bash scripts/run_test_normal.sh \
       gpt-4.1 \
       Qwen3-14B_history_baseline \
       configs/context_opt/qwen3-14B/history_baseline.yaml
   ```

## 5. Distillation Stage 2: Agent Model

This stage builds on the compressor model, adding a second LoRA layer
to teach higher-level **reasoning and action selection**.

1. Export trajectories:

   ```bash
   cd experiments/training
   python save_trajectories_dataset.py -f gpt-4.1_250809_gpt-4.1_history_v2 -t llm_history
   ```

   Output:
   `dataset/llm_history/gpt-4.1_250809_gpt-4.1_history_v2_train.jsonl`

2. Finetune the agent LoRA:

   ```bash
   bash scripts/run_finetune_agent.sh \
       Qwen/Qwen3-14B \
       dataset/llm_history/gpt-4.1_250809_gpt-4.1_history_v2_train.jsonl \
       250916_agent_history_v2_3epochs
   ```

3. Serve compressor + agent stacked:

   ```bash
   bash scripts/serve_agent.sh \
       Qwen/Qwen3-14B \
       finetuned_models/qwen-14B/250916_history_v2_3epochs \
       finetuned_models/qwen-14B/250916_agent_history_v2_3epochs
   ```

4. Evaluate using the same config:

   ```bash
   cd ../appworld
   bash scripts/run_test_normal_lora.sh \
       Qwen/Qwen3-14B \
       250916_Qwen3-14B_history_baseline \
       configs/context_opt/qwen3-14B/history_baseline.yaml
   ```