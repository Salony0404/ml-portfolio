import math
import torch
import evaluate
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.optim import AdamW
from datasets import load_metric
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq,  get_scheduler
from transformers import get_cosine_schedule_with_warmup


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
original_model = AutoModelForSeq2SeqLM.from_pretrained(token_model_directory).to(device)
print("Tokenizer and model have been loaded successfully....")
print("--" * 70)

# Load the processed dataset from disk
print(f"Loading the tokenized dataset from {tokenizer_model_directory}...")
tokenized_datasets = load_from_disk(tokenizer_model_directory)
print("Tokenized dataset has been loaded successfully...")
print("--" * 70)
print()

print("Setting the data collator")
# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=original_model)

print()
print("---" * 50)
print()

print("Parameters and hyperparameters for the model ....")
lr_rate = 0.0001
num_epochs = 120
batch_size = 4
num_training_steps = 7019
num_warmup_steps = 1000
params = original_model.parameters()
total_steps = num_epochs * (num_training_steps / batch_size)
print("The total steps are:", total_steps)

warmup_steps = int(0.1 * total_steps)
print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")


print(tokenized_datasets)
print("---"*50)
print()

print("setting training arguments")
training_args = TrainingArguments(
    output_dir="check1",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # evaluation_strategy="epoch",
    gradient_accumulation_steps = 2,
    learning_rate=lr_rate,
    evaluation_strategy="steps",
    logging_steps=100,
    num_train_epochs=num_epochs,

    weight_decay=0.01
)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(7019))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(957))

print("passing TA to trainer")
trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator = data_collator
                                     )

tokenized_datasets.set_format("torch")
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(7019))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(957))



print("parameters and hyperparameters for the model ....")

params = original_model.parameters()
total_steps = num_epochs * (num_training_steps / batch_size)
print("the total steps are:", total_steps)

print()
print("DataLoader,optimizer and schedulers are...")
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)



optimizer = AdamW(params=params, lr=lr_rate)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

print("---"*50)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
original_model.to(device)

print("progress barr...")



progress_bar = tqdm(range(num_training_steps))

original_model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs =original_model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        epoch_loss += loss.item()

    epoch_loss /= len(train_dataloader)
    print(f"The average epoch loss for epoch {epoch + 1} is: {epoch_loss}")
    print()

    print("Saving checkpoints...")
    checkpoint_path = f"train_3/epoch_{epoch + 1}.pt"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': original_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': epoch_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved for epoch {epoch + 1} at {checkpoint_path}")

print()
print("Loading the metric for evaluation...")

metric = load_metric("accuracy")
original_model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = original_model(**batch)
        logits = outputs.logits

    labels = batch['labels'].cpu().numpy()
    predictions = torch.argmax(logits, dim=-1)
    predictions = predictions.flatten()
    labels = labels.flatten()

    metric.add_batch(predictions=predictions, references=labels)

accuracy = metric.compute()
print(f"Final Accuracy: {accuracy['accuracy']}")

output_dir = "fine_tuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
