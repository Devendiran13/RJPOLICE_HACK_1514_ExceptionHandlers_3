import os
import cv2
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import tensorflow
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization, Activation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score

vgg_network = VGG19(input_shape=(250, 250, 3), weights='imagenet', include_top=False)
for layer in vgg_network.layers:
    layer.trainable = False

X = Flatten()(vgg_network.output)
Prediction = Dense(3, activation='softmax')(X)
vgg_model = Model(inputs=vgg_network.input, outputs=Prediction)

vgg_model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

plot_model(model=vgg_model, show_shapes=True)

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history = vgg_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=[early_stop], batch_size=30, shuffle=True)

test_loss, test_accuracy = vgg_model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)
