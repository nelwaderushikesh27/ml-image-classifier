"""
Training script for image classifier.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import argparse
import os


def build_model(input_shape=(224, 224, 3), num_classes=10):
    """Build CNN model."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def load_data(data_dir):
    """Load dataset from directory."""
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        image_size=(224, 224),
        batch_size=32,
        label_mode='categorical'
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        image_size=(224, 224),
        batch_size=32,
        label_mode='categorical'
    )
    return train_ds, val_ds


def train_model(data_dir='data', epochs=50, model_path='models/classifier.h5'):
    """Train the classifier."""
    train_ds, val_ds = load_data(data_dir)
    
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model-path', type=str, default='models/classifier.h5')
    args = parser.parse_args()
    
    model, history = train_model(args.data_dir, args.epochs, args.model_path)
    print(f"Model saved to {args.model_path}")
