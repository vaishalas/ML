# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/157Yd6kvEsNZrrt1f2pAe5LkHKnZ9mUr_
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
import kagglehub

path = kagglehub.dataset_download("ayuraj/asl-dataset")

print("Path to dataset files:", path)

train_dir = f'{path}/asl_dataset/asl_dataset'
test_dir = f'{path}/asl_dataset/asl_dataset'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(train_data.num_classes,activation='softmax')
])

model.compile(optimizer='adam',loss='crossentropy',metrics=['accuracy'])

history = model.fit(train_data, epochs=2, validation_data=test_data)

test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

y_true = test_data.classes
y_pred = np.argmax(model.predict(test_data), axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_data.class_indices.keys())))
print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")