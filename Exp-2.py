# -*- coding: utf-8 -*-
"""ml_lab_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10yEXmh9z3ylEamv9qCFqJL9hTTHhVYs_
"""

#EXP2

import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import kagglehub

# Download latest version
path = kagglehub.dataset_download("ishantripathi93/plant-disease-dataset")

print("Path to dataset files:", path)

def preprocess_image(image_path, size=(128, 128)):
    """
    Load and preprocess the image (resize, normalize).
    """
    image = cv2.imread(image_path)
    if image is not None:
        # Resize the image to a fixed size
        image = cv2.resize(image, size)
        # Normalize the pixel values to the range [0, 1]
        image = image.astype('float32') / 255.0
        # Flatten the image (convert 2D image to 1D feature vector)
        image = image.flatten()
    return image

def load_dataset(image_folder):
    images = []
    labels = []

    for label_folder in os.listdir(image_folder):
        label_path = os.path.join(image_folder, label_folder)
        label = 1 if label_folder == 'diseased' else 0  # 1 for diseased, 0 for healthy

        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            image = preprocess_image(image_path)
            if image is not None:
                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)

image_folder = f'{path}/train'

# Load dataset
X, y = load_dataset(image_folder)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

def predict_image(image_path):
    image = preprocess_image(image_path)
    if image is not None:
        image = scaler.transform([image])  # Scale the image features
        prediction = svm_model.predict(image)
        return "Diseased" if prediction[0] == 1 else "Healthy"
    else:
        return "Invalid image"

new_image_path = f"{path}/train/healthy/30.jpg"  # Replace with the path to the new image
result = predict_image(new_image_path)
print(f"The given plant is: {result}")

#EXP3