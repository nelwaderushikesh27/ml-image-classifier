"""
Flask API for image classifier.
"""

from flask import Flask, request, jsonify
from model.predict import ImageClassifier
import os

app = Flask(__name__)
classifier = ImageClassifier(
    model_path=os.getenv('MODEL_PATH', 'models/classifier.h5')
)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image'].read()
    result = classifier.predict(image)
    return jsonify(result)


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    images = request.files.getlist('images')
    results = classifier.predict_batch([img.read() for img in images])
    return jsonify({'results': results})


@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify({
        'model': 'CNN Image Classifier',
        'input_shape': '(224, 224, 3)',
        'num_classes': 10
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
