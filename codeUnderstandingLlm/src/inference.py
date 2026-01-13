import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_scheduler
from datasets import load_dataset,load_from_disk
from transformers import get_cosine_schedule_with_warmup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device used is:", device)
# Directories
raw_dataset_directory = './raw_dataset'
token_model_directory = './tokenModel'
tokenizer_model_directory = './tokenized_dataset'

# Device checking
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used is:", device)
print("--" * 70)
print()

print("Libraries have been loaded.....")
print("--" * 70)
print()

print("Loading the dataset from local disk...")
dataset = load_from_disk(raw_dataset_directory)
print("Raw dataset has been loaded successfully...")
print("--" * 70)
print()

print("Loading the tokenizer and model from saved directory....")
tokenizer = AutoTokenizer.from_pretrained(token_model_directory, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(token_model_directory).to(device)
print("Tokenizer and model have been loaded successfully....")
print("--" * 70)

# Load the processed dataset from disk
print(f"Loading the tokenized dataset from {tokenizer_model_directory}...")
tokenized_datasets = load_from_disk(tokenizer_model_directory)
print("Tokenized dataset has been loaded successfully...")
print("--" * 70)
print()


lr_rate = 0.0001
num_epochs = 20
batch_size = 4
num_training_steps = 7019
params = model.parameters()
total_steps = num_epochs * (num_training_steps / batch_size)
print("the total steps are:", total_steps)

# Define the optimizer, scheduler, optimizer
#optimizer = AdamW(params=params,lr= lr_rate)
#scheduler = get_scheduler("linear",optimizer=optimizer,num_warmup_steps=0,num_training_steps=total_steps)
num_warmup_steps=1000
optimizer = AdamW(params=params, lr=lr_rate)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)




def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

# Load the checkpoint (replace with the correct checkpoint file path)
checkpoint_path = "train_3/epoch_119.pt"  # Update this to the latest checkpoint if needed
epoch, loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")

# Set the model to evaluation mode
model.eval()

print("Zero shot inference checking ....")
code = dataset['test'][0:10]['sentences']
csv_summary = dataset['test'][0:10]['Explanation']

original_model_summaries = []
instruct_model_summaries = []

for _, dialogue in enumerate(code):
    prompt = f"""
learn the explanations of the sentences and generate your own explanation based on the learnt explanation

{dialogue}

Description: """

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
 # Generate the output
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"],
                                 max_length=1000,          # Allows for long output
                                 min_length=300,           # Ensures the output is sufficiently long
                                 length_penalty=1.4,       # Slightly penalizes very long sequences to avoid verbosity
                                 num_beams=10,              # Balances quality and computational cost
                                 top_k=40,
                                 top_p=0.95,
                                 temperature=0.7,
                                 no_repeat_ngram_size=3,   # Avoids repeating phrases
                                 early_stopping=True)      # Stops when an end condition is met

    # Decode the generated output
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    instruct_model_summaries.append(output)

zipped_summaries = list(zip(csv_summary, instruct_model_summaries))

df = pd.DataFrame(zipped_summaries, columns=['csv_dataset_summaries', 'instruct_model_summaries'])
print()
print("The results are given below:")
for index, row in df.iterrows():
    print(f"csv_dataset_summaries: {row['csv_dataset_summaries']}")
    print("*"*70)
    print(f"instruct_model_summaries: {row['instruct_model_summaries']}")
    print("-" * 70)

print()
from rouge_score import rouge_scorer
print("ROUGE SCORE ANALYSIS")
# Calculate ROUGE score
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = []

for index, row in df.iterrows():
    scores.append(scorer.score(row['csv_dataset_summaries'], row['instruct_model_summaries']))

# Print ROUGE scores
print("ROUGE scores:")
for score in scores:
    print("ROUGE-1:", score['rouge1'].fmeasure)
    print("ROUGE-2:", score['rouge2'].fmeasure)
    print("ROUGE-L:", score['rougeL'].fmeasure)
    print("-" * 50)

output_dir = "fine_tuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
