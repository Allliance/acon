import json
from datasets import Dataset

def preprocess_sft_dataset(filepath):
    dataset = []
    with open(filepath) as f:
        for line in f:
            dataset.append(json.loads(line))

    processed_dataset = []
    for data in dataset:
        messages = data['messages']
        processed_dataset.append({"messages": messages})
    processed_dataset = Dataset.from_list(processed_dataset)
    return processed_dataset

def preprocess_sft_rejection_dataset(filepath):
    dataset = []
    with open(filepath) as f:
        for line in f:
            dataset.append(json.loads(line))

    preprocess_dataset = []
    for data in dataset:
        positives = data["positives"]
        for pos in positives:
            # TODO: sorting with the length of generation
            messages = [
                {"role": "system", "content": data["system_prompt"]},
                {"role": "user", "content": data["user_prompt"]},
                {"role": "assistant", "content": pos}
            ]
            preprocess_dataset.append({"messages": messages})
            break # Use only one positive following the baseline paper

    print(f"Preprocessed {len(preprocess_dataset)} samples from rejection SFT dataset.")    
    processed_dataset = Dataset.from_list(preprocess_dataset)
    return processed_dataset

def preprocess_dpo_dataset(filepath):
    dataset = []
    with open(filepath) as f:
        for line in f:
            dataset.append(json.loads(line))

    preprocess_dataset = []
    for data in dataset:
        if len(data["positives"]) == 0 or len(data["negatives"]) == 0:
            continue

        # Make possible pairs using zip until the shorter list is exhausted
        min_length = min(len(data["positives"]), len(data["negatives"]))
        for i in range(min_length):
            prompt = [
                {"role": "system", "content": data["system_prompt"]},
                {"role": "user", "content": data["user_prompt"]}
            ]
            chosen = [
                {"role": "assistant", "content": data["positives"][i]},
            ]
            rejected = [
                {"role": "assistant", "content": data["negatives"][i]},
            ]
            preprocess_dataset.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    print(f"Preprocessed {len(preprocess_dataset)} samples from DPO dataset.")
    processed_dataset = Dataset.from_list(preprocess_dataset)
    return processed_dataset