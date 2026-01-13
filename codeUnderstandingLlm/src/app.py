from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used is:", device)

# Load the tokenizer and model
model_name = 't5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16).to(device)

# Load fine-tuned model
fine_tuned_model_path = "/home/mlmachine/Documents/LLM_project/fine_tuned_model"
instruct_model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_path, torch_dtype=torch.bfloat16).to(device)


@app.route('/train', methods=['GET'])
def train():
    return jsonify(message="Model trained successfully!")

@app.route('/generate-description', methods=['POST'])
def generate_description():
    data = request.get_json()
    code = data.get('code', '')
    
    prompt = f"""
    Zero Shot Learning: Generate a description of the function based on the provided code.

    Code: '{code}'
    descriptions:'{data}'
    """
    
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    output_ids = instruct_model.generate(inputs["input_ids"], max_length=1000,
                                         num_return_sequences=1,
                                         num_beams=5,
                                         top_k=50, top_p=0.95,
                                         temperature=0.7)
    description = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return jsonify(description=description)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
