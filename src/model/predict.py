"""
Prediction script for image classifier.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import io


class ImageClassifier:
    def __init__(self, model_path='models/classifier.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = self._load_class_names()
    
    def _load_class_names(self):
        return ['class_0', 'class_1', 'class_2', 'class_3', 'class_4',
                'class_5', 'class_6', 'class_7', 'class_8', 'class_9']
    
    def preprocess_image(self, image_bytes):
        """Preprocess image for prediction."""
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, image_bytes):
        """Predict class for an image."""
        processed = self.preprocess_image(image_bytes)
        predictions = self.model.predict(processed)[0]
        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx])
        
        return {
            'class': self.class_names[class_idx],
            'confidence': confidence,
            'predictions': {
                name: float(conf)
                for name, conf in zip(self.class_names, predictions)
            }
        }
    
    def predict_batch(self, image_bytes_list):
        """Predict classes for multiple images."""
        results = []
        for image_bytes in image_bytes_list:
            results.append(self.predict(image_bytes))
        return results
