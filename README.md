Emotion Recognition with RAF-DB Dataset
A real-time emotion recognition system that uses deep learning to classify facial expressions into 7 emotion categories: Surprise, Fear, Disgust, Happy, Sad, Angry, and Neutral.

🎯 Features
Real-time Emotion Detection: Webcam-based emotion recognition

Deep Learning Models: Custom CNN architecture with optional transfer learning

Data Analysis: Comprehensive class distribution visualization

Data Augmentation: Improved model generalization with image augmentation

Class Balancing: Automatic class weight computation for imbalanced datasets

Model Evaluation: Detailed performance metrics and visualization

🚀 Quick Start
Prerequisites
Python 3.7+

Webcam for real-time demo

Kaggle account (for dataset download)

Installation
Clone the repository

bash
git clone <repository-url>
cd emotion-recognition
Install dependencies

bash
pip install -r requirements.txt
Set up Kaggle credentials (for automatic dataset download)

Create a Kaggle account at https://www.kaggle.com/

Go to your account settings and create API token

Place kaggle.json in ~/.kaggle/ directory

Usage
Run the complete pipeline:

bash
python emotion_classifier.py
The script will automatically:

Download the RAF-DB dataset

Build and train the CNN model

Launch real-time webcam emotion detection

📁 Project Structure
text
emotion-recognition/
├── emotion_classifier.py    # Main implementation file
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
└── models/                 # Saved models (generated after training)
🧠 Model Architecture
The system uses a custom CNN architecture with:

Input: 48×48 grayscale facial images

Architecture:

4 convolutional blocks with BatchNorm and Dropout

Global Average Pooling

Dense layers with regularization

Softmax output for 7 emotion classes

Training: Adam optimizer with learning rate scheduling

Regularization: Early stopping, data augmentation, class weights
