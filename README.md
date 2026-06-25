# 🤖 ML Image Classifier

A deep learning image classifier built with TensorFlow and Keras.

## ✨ Features
- Real-time image classification
- Custom model training
- Transfer learning support
- GPU acceleration
- Batch prediction
- Model export (SavedModel, TFLite)

## 🛠️ Tech Stack
| Component | Technology |
|-----------|------------|
| **Framework** | TensorFlow 2.x |
| **Language** | Python 3.10+ |
| **Backend** | Flask/FastAPI |
| **Frontend** | Streamlit |
| **Deployment** | Docker |

## 🏗️ Project Structure

```
ml-image-classifier/
├── src/
│   ├── model/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── model.py
│   ├── data/
│   │   ├── loader.py
│   │   ├── augment.py
│   │   └── preprocess.py
│   ├── app/
│   │   ├── api.py
│   │   └── streamlit_app.py
│   └── utils/
│       ├── metrics.py
│       └── visualize.py
├── notebooks/
│   ├── exploratory.ipynb
│   └── training.ipynb
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/nelwaderushikesh27/ml-image-classifier.git
cd ml-image-classifier

# Install
pip install -r requirements.txt

# Train
python src/model/train.py --epochs 50

# Run API
python src/app/api.py

# Run Web App
streamlit run src/app/streamlit_app.py
```

## 📊 Model Architecture

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

## 📦 Dependencies

```txt
tensorflow>=2.13.0
numpy>=1.24.0
pillow>=10.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
streamlit>=1.28.0
flask>=3.0.0
opencv-python>=4.8.0
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Classify image |
| POST | `/predict/batch` | Batch classification |
| GET | `/model/info` | Model details |
| GET | `/health` | Health check |

## 🐳 Docker

```bash
docker build -t ml-classifier .
docker run -p 5000:5000 ml-classifier
```

---
*Built with 🤖 TensorFlow*
