from flask import Flask, render_template, request, jsonify
import easyocr
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
reader = easyocr.Reader(['en'])

# Example training data (expand later for better accuracy)
texts = [
    "The industrial revolution marked a major turning point in history.",
    "Humans have always found ways to adapt to their environments.",
    "Creativity is one of the most important aspects of human nature.",
    "As an AI language model, I can provide information on many topics.",
    "I am an AI model developed by OpenAI to assist with various tasks.",
    "According to my training data, the industrial revolution started in Britain.",
]
labels = [0, 0, 0, 1, 1, 1]  # 0 = Human, 1 = GPT

def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

# Preprocessing
texts = [preprocess(t) for t in texts]

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train simple classifier
model = LogisticRegression()
model.fit(X, labels)

def predict_text_origin(input_text):
    input_text = preprocess(input_text)
    input_vector = vectorizer.transform([input_text])
    prediction = model.predict(input_vector)
    return "GPT-Generated" if prediction[0] == 1 else "Human-Written"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    img_bytes = image.read()
    results = reader.readtext(img_bytes)

    if not results:
        return jsonify({'error': 'No text detected in image'}), 400

    extracted_text = '\n'.join([result[1] for result in results])

    # Predict if extracted text is GPT or Human
    prediction = predict_text_origin(extracted_text)

    # Now first show prediction, then the text
    return jsonify({
        'prediction': prediction,
        'text': extracted_text
    })

if __name__ == "__main__":
    app.run(debug=True)
