import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


train_path = r'/content/Train'
test_path = r'/content/Test'
val_path = r'/content/Validation'

for folder in os.listdir(train_path):
    sub_path = os.path.join(train_path, folder)
    for i in range(2):
        if i < len(os.listdir(sub_path)):
            temp_path = os.path.join(sub_path, os.listdir(sub_path)[i])
            img = mpimg.imread(temp_path)
            imgplot = plt.imshow(img)
            plt.show()

def image_to_array(path, size):
    data = []
    for folder in os.listdir(path):
        sub_path = os.path.join(path, folder)
        for img in os.listdir(sub_path):
            image_path = os.path.join(sub_path, img)
            img_arr = cv2.imread(image_path)
            if img_arr is not None:
                img_arr = cv2.resize(img_arr, size)
                data.append(img_arr)
            else:
                print(f"Warning: Failed to read image {image_path}")
    return data
  
train = image_to_array(train_path, (250,250))
test = image_to_array(test_path,(250,250))
val = image_to_array(val_path, (250,250))

x_train = np.array(train)
x_test = np.array(test)
x_val = np.array(val)

x_train = x_train / 255.0
x_test = x_test / 255.0
x_val = x_val / 255.0

def data_inclass(data_path, size, class_mode):
    datageneration = ImageDataGenerator(rescale=1./255)
    classes = datageneration.flow_from_directory(data_path, target_size=size, batch_size=32, class_mode=class_mode)
    return classes

train_class = data_inclass(train_path, size, 'sparse')
test_class = data_inclass(test_path, size, 'sparse')
val_class = data_inclass(val_path, size, 'sparse')

y_train = train_class.classes
y_test = test_class.classes
y_val = val_class.classes
