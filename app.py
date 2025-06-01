from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Inicjalizacja modelu i tokenizera
model_name = "facebook/opt-125m"  # Darmowy model językowy
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def replace_r_with_l(text):
    return text.replace('r', 'l').replace('R', 'L')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    # Przygotowanie wejścia dla modelu
    inputs = tokenizer(user_message, return_tensors="pt", max_length=100)
    
    # Generowanie odpowiedzi
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    
    # Dekodowanie odpowiedzi
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Zamiana R na L
    modified_response = replace_r_with_l(response)
    
    return jsonify({'response': modified_response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 