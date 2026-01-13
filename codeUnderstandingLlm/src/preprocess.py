import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used is:", device)
    return device

def load_dataset_from_huggingface(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer

def tokenize_function(example, tokenizer):
    #start_prompt = 'Summarize the following codes.\n\n'
    #end_prompt = '\n\nSummary: '
   # prompt = [start_prompt + dialogue + end_prompt for dialogue in example["sentences"]]
    example['input_ids'] = tokenizer(example['sentences'], padding="max_length",max_length = 1024, truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["Explanation"], padding="max_length", max_length = 1024,  truncation=True, return_tensors="pt").input_ids
    return example

def preprocess_dataset(dataset, tokenizer):
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['sentences', 'Explanation'])
    return tokenized_datasets

if __name__ == "__main__":
    device = check_device()
    dataset_name = "salony/code_explanations_2"
    dataset = load_dataset_from_huggingface(dataset_name)
    tokenizer = load_tokenizer("t5-base")
    tokenized_datasets = preprocess_dataset(dataset, tokenizer)
    torch.save(tokenized_datasets, "tokenized_datasets.pt")
    print("Preprocessing complete. Tokenized datasets saved.")
