from unsloth import FastLanguageModel, FastModel

import os
import sys
sys.path.append(".")
import json
import torch
import random
from datetime import datetime

import argparse
import transformers
from datasets import concatenate_datasets
from collections import defaultdict

from trl import (
    SFTTrainer,
    SFTConfig,
    DataCollatorForCompletionOnlyLM
)

from preprocess import (
    preprocess_sft_dataset,
)
from unsloth.chat_templates import train_on_responses_only
from data_utils import compute_io_token_stats

os.environ["DISABLE_MLFLOW_INTEGRATION"] = "1"

MODEL_IDENTIFIERS = {
    "meta-llama/Llama-3.2-1B-Instruct": "llama-1B-instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3B-instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-8B-instruct",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen-0.5B-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen-1.5B-instruct",
    "Qwen/Qwen2.5-3B-Instruct": "qwen-3B-instruct",
    "Qwen/Qwen2.5-7B-Instruct": "qwen-7B-instruct",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen-4B",
    "Qwen/Qwen3-8B": "qwen-8B",
    "Qwen/Qwen3-14B": "qwen-14B",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "qwen-coder-1.5B-instruct",
    "microsoft/phi-4": "phi-4",
}

def setup_savedir(args):
    # Step 3-1: Setup save dir
    if "finetuned_models" in args.model_name:
        # Extract model_identifier from the path
        path_parts = args.model_name.split('/')

        # Find the part that might be a model identifier
        for part in path_parts:
            # Check if this part is a value in MODEL_IDENTIFIERS
            for model_name, identifier in MODEL_IDENTIFIERS.items():
                if part == identifier:
                    model_identifier = part
                    break

            # If we found a match, break out of the outer loop
            if 'model_identifier' in locals():
                break

        # If no match was found in the path, use a default
        if 'model_identifier' not in locals():
            # Try to infer from the directory structure
            if len(path_parts) >= 3 and path_parts[-3] == "finetuned_models":
                model_identifier = path_parts[-2]
            else:
                model_identifier = "qwen-7B-instruct"  # Default fallback
    else:
        model_identifier = MODEL_IDENTIFIERS.get(args.model_name)
        if model_identifier is None:
            raise NotImplementedError

    print(f"Model: {args.model_name}")

    if args.exp_id:
        exp_id = f"{args.exp_id}"
    else:
        exp_id = f"baseline"
        if args.num_epochs > 1:
            exp_id += f"_{args.num_epochs}epochs"
        if args.full_finetuning:
            exp_id += "_full"
        if len(args.postfix) > 0:
            if args.postfix.startswith("_"):
                exp_id += args.postfix
            else:
                exp_id += "_" + args.postfix

    output_dir = f"./finetuned_models/{model_identifier}/{exp_id}"
    print("Output dir: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    metadata = vars(args)
    with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return output_dir

def main(args):
    # Set Seed
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.model_name not in MODEL_IDENTIFIERS.keys() and "finetuned_models" not in args.model_name:
        import pdb; pdb.set_trace()
        args.model_name = "meta-llama/Llama-3.2-1B-Instruct"

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name if args.prev_lora_path is None else args.prev_lora_path,
        max_seq_length=args.max_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=args.full_finetuning,
    )

    if not args.full_finetuning and args.prev_lora_path is None:
        if "qwen" in args.model_name.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",]
        elif "phi" in args.model_name.lower():
            target_modules = ["o_proj", "gate_up_proj", "qkv_proj", "down_proj"]
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = target_modules,
            lora_alpha = 32,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            max_seq_length = args.max_length,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
        peft_config = None
    else:
        peft_config = None

    print("peft config", peft_config)

    ########## Setup model done ###############

    # Setup the dataet
    # Load dataset

    preprocess_fn = preprocess_sft_dataset

    train_dataset = None
    for _train_filepath in args.train_filepath:
        _train_dataset = preprocess_fn(_train_filepath)
        if train_dataset:
            train_dataset = concatenate_datasets([train_dataset, _train_dataset])
        else:
            train_dataset = _train_dataset

    if args.valid_filepath is not None:
        eval_dataset = preprocess_fn(args.valid_filepath)
    else:
        eval_dataset = None

    if args.dataset_size > 0:
        train_dataset = train_dataset[:args.dataset_size]

    # Compute token stats BEFORE converting with chat template so we separate user/system vs assistant
    try:
        compute_io_token_stats(train_dataset, tokenizer, max_samples=min(500, len(train_dataset)))
    except Exception as e:
        print("[Warn] Failed to compute token stats:", e)

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(
            convo, 
            tokenize = False, 
            add_generation_prompt = False,
            enable_thinking = False 
        ).rstrip() for convo in convos]
        return { "text" : texts, }
    
    train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
    data_module = {
        "train_dataset": train_dataset
    }

    print("# Train Dataset: ", len(data_module["train_dataset"]))
    if "eval_dataset" in data_module.keys():
        print("# Valid Dataset: ", len(data_module["eval_dataset"]))
        eval_strategy = "epoch"
        save_strategy = "epoch"
        load_best_model_at_end = True
    else:
        eval_strategy = "no"
        save_strategy = "no"
        load_best_model_at_end = False

    output_dir = setup_savedir(args)
    ########## Setup dataset done ###############

    batch_size = args.batch_size
    # Step 3: Train
    train_args = SFTConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        bf16=True,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        deepspeed=args.deepspeed,
        fsdp=args.fsdp is not None,
        fsdp_config=args.fsdp,
        # assistant_only_loss=True,
        # Strategy
        logging_steps=10,
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        output_dir=output_dir,
        load_best_model_at_end=load_best_model_at_end,
        gradient_checkpointing=args.gradient_checkpointing,
        save_safetensors=False
    )

    # only for Qwen
    if "qwen" in args.model_name.lower():
        response_template = "<|im_start|>assistant"
        instruction_template = "<|im_start|>user"
    elif "phi" in args.model_name.lower():
        if "mini" in args.model_name.lower():
            response_template = "<|assistant|>"
            instruction_template = "<|user|>"
        else:
            response_template = "<|im_start|>assistant<|im_sep|>"
            instruction_template = "<|im_start|>user<|im_sep|>"
    else:
        raise NotImplementedError("Only Qwen is supported for now")

    # collator = DataCollatorForCompletionOnlyLM(
    #     response_template,
    #     instruction_template=instruction_template,
    #     tokenizer=tokenizer
    # ) # instead, use train_on_responses_only

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        peft_config=peft_config,
        # data_collator=collator,
        **data_module
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_template,
        response_part=response_template
    )
    # debug
    # print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))
    space = tokenizer(" ", add_special_tokens = False).input_ids[0]
    print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))
    
    trainer.train()
    ########## Train done ###############

    # Step 4: Save best model
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
        default="Qwen/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--peft_name", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--num_epochs', default=2, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--max_length", default=10000, type=int)
    parser.add_argument("--postfix", default="", type=str)
    parser.add_argument("--full_finetuning", action='store_true')
    parser.add_argument("--dataset_size", default=-1, type=int)
    parser.add_argument("--prev_lora_path", type=str, default=None, help="Previous iteration of lora for iteravtive fine-tuning")

    parser.add_argument(
        "--train_filepath",
        type=str,
        default=["./dataset/history_optimizer_history/gpt-4.1_250729_history_opt_last1_promptv2_train.jsonl"],
        nargs='+'
    )
    parser.add_argument("--valid_filepath", type=str, default=None)
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--rejection_sft", action='store_true')

    # Deepspeed
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--fsdp", type=str, default=None)

    args = parser.parse_args()

    main(args)