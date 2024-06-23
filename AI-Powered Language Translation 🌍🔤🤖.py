from flask import Flask, request, jsonify, render_template
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text, src_lang='en', tgt_lang='fr'):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    translated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128, num_beams=4, early_stopping=True)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translated_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data['text']
    translated_text = translate_text(text)
    return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True)
