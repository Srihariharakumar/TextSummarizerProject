from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']

    input_token = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    output_token = model.generate(input_token['input_ids'])
    summary = tokenizer.decode(output_token[0], skip_special_tokens=True)

    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=1000,debug=True)