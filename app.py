from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import random
import gc

app = Flask(__name__)

# Inicjalizacja modelu i tokenizera
model_name = "gpt2"  # Używamy podstawowego GPT-2 zamiast DistilGPT2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

# Optymalizacja pamięci
model.eval()
model = model.cpu()  # Wymuszamy użycie CPU
torch.set_num_threads(1)  # Ograniczamy liczbę wątków

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
    if random.random() < 0.7:  # 70% szans na dodanie zwrotu
        kibol_phrase = random.choice(KIBOL_PHRASES)
        if random.random() < 0.5:
            text = f"{kibol_phrase} {text}"
        else:
            text = f"{text} {kibol_phrase}"
    
    # Dodanie wielokrotnych wykrzykników
    if random.random() < 0.3:  # 30% szans na dodanie wykrzykników
        text = text.replace('.', '!!!')
    
    # Dodanie wielkich liter
    if random.random() < 0.4:  # 40% szans na wielkie litery
        text = text.upper()
    
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']
        
        # Przygotowanie wejścia dla modelu
        inputs = tokenizer(user_message, return_tensors="pt", max_length=30)  # Zmniejszamy maksymalną długość
        
        # Generowanie odpowiedzi
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=50,  # Zmniejszamy maksymalną długość
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=20,  # Zmniejszamy top_k
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Dekodowanie odpowiedzi
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Dodanie stylu kibola
        modified_response = add_kibol_style(response)
        
        # Czyszczenie pamięci
        del outputs
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return jsonify({'response': modified_response})
    except Exception as e:
        return jsonify({'response': f"KULWA, COŚ SIĘ ZEPSUŁO! BŁĄD: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 
