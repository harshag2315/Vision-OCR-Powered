from flask import Flask, render_template, request, jsonify
import easyocr
from transformers import pipeline
from pdf2image import convert_from_bytes
import torch
from io import BytesIO
from PIL import Image


app = Flask(__name__)
reader = easyocr.Reader(['en'])

# Load the Hugging Face model
model_name = "facebook/bart-large-mnli"
model = pipeline("zero-shot-classification", model=model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    uploaded_file = request.files['file']

    if uploaded_file.filename.lower().endswith('.pdf'):
        images = convert_from_bytes(uploaded_file.read())
    else:
        image = Image.open(uploaded_file.stream)
        images = [image]

    results = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        page_results = reader.readtext(buf.read())
        results.extend(page_results)

    if not results:
        return jsonify({'error': 'No text detected'}), 400

    extracted_text = '\n'.join([res[1] for res in results])
    cleaned_text = " ".join(extracted_text.split())

    candidate_labels = ["AI-Generated", "Human-Written"]
    prediction = model(cleaned_text, candidate_labels=candidate_labels)

    text_origin = prediction['labels'][0]

    return jsonify({
        'prediction': text_origin,
        'text': extracted_text
    })
if __name__ == "__main__":
    app.run(debug=True)
