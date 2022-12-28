# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 12:28:03 2022

@author: kmj84
"""

#이미지 불러오기
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

TRAINING_DIR = "C:/Users/kmj84/OneDrive/바탕 화면/aipSources/쓰레기/train"
VALIDATION_DIR = "C:/Users/kmj84/OneDrive/바탕 화면/aipSources/쓰레기/test"

batch_size=8

training_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

#이미지 전처리
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size = batch_size,
    target_size=(128,128),
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size = batch_size,
    target_size=(128,128),
    class_mode='categorical',
    shuffle=True)

img, label = next(train_generator)
plt.figure(figsize=(20,20))


'''
#이미지 확인하기
for i in range(8):
    plt.subplot(3, 3, i+1)
    plt.imshow(img[i])
    plt.title(label[i])
    plt.axis('off')
    
plt.show()
'''
num_classes = 5


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128, 128, 3)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Dropout((0.25)),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(5, activation='sigmoid')
])


'''
model = Sequential([
  layers.Conv2D(16, (3,3), padding='same', activation='relu',input_shape=(128, 128, 3)),
  layers.MaxPooling2D(2,2),
  layers.Conv2D(32, (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Conv2D(64, (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Conv2D(128, (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dense(num_classes)
])
'''

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

history=model.fit(train_generator, epochs=30,
                  validation_data=validation_generator, verbose=1)

res = model.evaluate(validation_generator, verbose=0)
print('"정확률은' , res[1]*100)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='best')
plt.grid()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='best')
plt.grid()
plt.show()

model.save("saved_model2.h5")
