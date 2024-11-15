# -*- coding: utf-8 -*-
"""HandWirtternDigit Recognition.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12n9G81lM0gm8BlI7csgn2ycaOgK3wpy8
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as mp

data = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = data.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='Crossentropy')

model.fit(x_train,y_train,epochs=10)

index = 11
image = x_test[index].reshape(1,28,28)
pre = model.predict(image)
pre_label = np.argmax(pre)
print(pre_label)
mp.imshow(x_test[index],cmap='gray')