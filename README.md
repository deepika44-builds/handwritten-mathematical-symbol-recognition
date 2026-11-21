
# Handwritten Mathematical Symbol Recognition

An AI-powered handwritten digit and mathematical symbol recognition system built using Convolutional Neural Networks (CNN).

## Project Overview

This project recognises handwritten mathematical digits and symbols from images using deep learning.

The model is trained on a dataset containing 27,000 images across 18 different classes.

## Features

- Handwritten digit and symbol classification
- CNN-based deep learning model
- Image preprocessing and normalisation
- Model evaluation using a confusion matrix
- Precision, recall and F1-score analysis
- Interactive Streamlit web application
- Prediction confidence visualisation

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Google Colab

## Dataset Information

- Total Images: 27,000
- Number of Classes: 18
- Training Images: 21,600
- Testing Images: 5,400
- Image Dimensions: 28 × 28 pixels

## Model Architecture

The project uses a Convolutional Neural Network (CNN) consisting of:

1. Convolutional layers
2. Max pooling layers
3. Flatten layer
4. Fully connected dense layer
5. Dropout layer
6. Softmax output layer

## Application

The Streamlit application allows users to test the trained model using sample images from the dataset.

## Project Structure

```text
handwritten-mathematical-symbol-recognition/
│
├── app.py
├── model_training.ipynb
├── requirements.txt
└── README.md
```

## Future Improvements

- Custom handwritten image upload
- Improved image preprocessing
- Model optimisation
- Public application deployment

## Author

Deepika

BSc Computer Science

