import torch
from huggingface_hub import login
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Define directory to save the raw dataset
raw_dataset_directory = './raw_dataset'
save_tokenModel_directory = './tokenModel'
tokenized_dataset_directory = './tokenized_dataset'

print("Now dataset is loading from huggingface...")
huggingface_dataset_name = "salony/code_explanations_2"
dataset = load_dataset(huggingface_dataset_name)
print("dataset has been loaded successfully...")
print("----"*50)
print()

# Save the raw dataset to disk
print(f"Saving the raw dataset to {raw_dataset_directory}...")
dataset.save_to_disk(raw_dataset_directory)
print("Raw dataset has been saved successfully...")
print("----" * 50)
print()

# Load tokenizer and model
print("Loading the tokenizer for the model....")
tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=True)
print("Tokenizer has been loaded successfully....")
print("---" * 50)
print()

print("Loading the model ....")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base", torch_dtype=torch.bfloat16)
print("Model has been loaded successfully....")
print("---" * 50)
print()

# Save tokenizer and model
tokenizer.save_pretrained(save_tokenModel_directory)
model.save_pretrained(save_tokenModel_directory)
print("model and tokenizer has been saved successfully")
print("---"*80)
print()

print("Tokenizer function is working now...")
def tokenize_function(example):

    input = tokenizer(example['sentences'], padding="max_length", max_length = 512, truncation=True, return_tensors="pt")
    output = tokenizer(example["Explanation"], padding="max_length", max_length = 512, truncation=True, return_tensors="pt")

    example['input_ids'] = input.input_ids
    example['attention_mask'] = input.attention_mask
    example['labels'] = output.input_ids

    return example
print("worked successfully....")
print("---"*50)
print()

print("mapping the inputs corresponding to labels....")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
print("mapped successfully...")
print("---"*50)
print()

print("The tokenized dataset content are:")
print(tokenized_datasets)
print("---"*50)
print()

print("after removing the columns....")
tokenized_datasets = tokenized_datasets.remove_columns(['sentences', 'Explanation'])
print("columns removed successsfullyy..")
print("---"*50)
print()

print("After removing, the tokenized dataset content is:")
print(tokenized_datasets)
print("---"*50)
print()

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)
print("---"*50)
print()

print(f"Saving the tokenized dataset to {tokenized_dataset_directory}...")
tokenized_datasets.save_to_disk(tokenized_dataset_directory)
print("Tokenized dataset has been saved successfully...")
print("---"*50)
print()
