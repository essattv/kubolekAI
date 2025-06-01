from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import random
import gc

app = Flask(__name__)

# Zmienna globalna do modelu i tokenizera
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        model_name = "distilgpt2"  # Lżejszy model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,  # Możesz zmienić na float16 jeśli działa
            device_map='cpu'
        )
        model.eval()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

# Lista typowych zwrotów kibola
KIBOL_PHRASES = [
    "KULWA MAĆ!",
    "JEBANY PIELDOLNIK!",
    "CHUJOWA SYTUACJA!",
    "O KULWA!",
    "JEBANA SPLAWA!",
    "PIELDOLONE GÓWNO!",
    "KULWA JAKA PIĘKNA!",
    "JEBANY CUD!",
    "CHUJOWO TO WYGLĄDA!",
    "PIELDOLIĆ TO!",
]

def add_kibol_style(text):
    # Zamiana R na L
    text = text.replace('r', 'l').replace('R', 'L')
    
    # Dodanie losowego zwrotu kibola na początku lub końcu
    if random.random() < 0.7:
        kibol_phrase = random.choice(KIBOL_PHRASES)
        if random.random() < 0.5:
            text = f"{kibol_phrase} {text}"
        else:
            text = f"{text} {kibol_phrase}"
    
    # Dodanie wykrzykników
    if random.random() < 0.3:
        text = text.replace('.', '!!!')
    
    # Wielkie litery
    if random.random() < 0.4:
        text = text.upper()
    
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']

        # Wczytaj model tylko jeśli nie istnieje
        load_model()

        # Tokenizacja
        inputs = tokenizer(user_message, return_tensors="pt", max_length=20)

        # Generowanie odpowiedzi
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=30,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=10,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=20
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        modified_response = add_kibol_style(response)

        # Czyszczenie pamięci
        del outputs
        del inputs
        global model, tokenizer
        del model
        del tokenizer
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

        return jsonify({'response': modified_response})
    
    except Exception as e:
        return jsonify({'response': f"KULWA, COŚ SIĘ ZEPSUŁO! BŁĄD: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
