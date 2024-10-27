# Face Recognition Model Using Deep Learning

## Description
This project implements a deep learning model for face recognition using TensorFlow/Keras. The model is trained on the Pins Face Recognition dataset and achieves high accuracy in classifying different individuals from facial images.

## Features
- Deep CNN architecture with multiple convolutional blocks
- Data augmentation for improved generalization
- Batch normalization and dropout for regularization
- Learning rate scheduling
- Early stopping to prevent overfitting
- Comprehensive model evaluation and visualization

## Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/face-recognition-model.git
cd face-recognition-model
```
## Install required packages:
bash
pip install -r requirements.txt

## Download the dataset:
The code will automatically download the dataset using kagglehub.
Usage
Run the main script:
```
python face_recognition_model.py
```
## The script will:
Validate and preprocess the dataset
Train the model
Generate performance visualizations
Save the trained model
To use the trained model for predictions:
python
from face_recognition_model import predict_image

# Predict a single image
image_path = "path_to_your_image.jpg"
class_name, confidence = predict_image(image_path)
print(f"Predicted class: {class_name}")
print(f"Confidence: {confidence:.4f}")

## Model Architecture
Input Layer: 100x100x3 (RGB images)
3 Convolutional Blocks with:
Conv2D layers
Batch Normalization
MaxPooling
Dropout
Dense Layers with:
L2 regularization
Batch Normalization
Dropout
Output Layer: Softmax activation
Performance
Target Accuracy: 85%+
Training includes:
Validation split: 80/20
Early stopping
Learning rate reduction on plateau
Data Augmentation
Rotation
Width/Height shifts
Horizontal flips
Zoom
Shear transformations
## Files Structure
text
├── face_recognition_model.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── pins_face_recognition/

## Model Output
The model saves:
Training history plots
Classification report
Trained model weights
Performance metrics
