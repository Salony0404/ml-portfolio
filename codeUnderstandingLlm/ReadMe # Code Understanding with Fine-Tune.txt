# Code Understanding with Fine-Tuned LLM

## Overview
This project focuses on fine-tuning a large language model (T5) to automatically generate natural language explanations for source code. The system learns from a custom dataset of code snippets and their corresponding explanations and exposes inference capabilities through a REST API.

The project covers the complete lifecycle of an LLM-based system: data preprocessing, model fine-tuning, evaluation, inference, and deployment.

---

## Problem Statement
Understanding unfamiliar code is time-consuming and error-prone, especially for large or poorly documented codebases.  
The goal of this project is to train a language model that can **generate meaningful, human-readable explanations of code automatically**, enabling faster comprehension and improved developer productivity.

---

## Dataset
- Custom dataset hosted on Hugging Face
- Each sample contains:
  - A code snippet
  - A corresponding natural language explanation
- The dataset is tokenized and stored locally for efficient training

---

## Methodology
1. Dataset loading and preprocessing using Hugging Face Datasets
2. Tokenization using a T5 tokenizer
3. Fine-tuning a T5-based sequence-to-sequence model
4. Model evaluation using ROUGE metrics
5. Zero-shot and fine-tuned inference comparison
6. REST API deployment using Flask

---

## Model & Training
- Base model: **T5**
- Frameworks: PyTorch, Hugging Face Transformers
- Training:
  - Custom training loop
  - Gradient accumulation
  - Cosine learning rate scheduler
  - Checkpoint saving per epoch

---

## Evaluation
- ROUGE-1, ROUGE-2, and ROUGE-L scores
- Qualitative comparison between ground-truth explanations and generated outputs

---

## Project Structure
