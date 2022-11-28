# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:29:48 2018

@author: mohit123
"""
import numpy as np
from keras import applications
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras.utils import get_file

import numpy as np
import warnings

from keras import applications
from keras.models import Sequential
from keras.layers import  Convolution2D, Conv2D, MaxPool2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras.utils import get_file
from keras.datasets import mnist
#from sklearn.preprocessing import OneHotEncoder
from keras.layers import Activation
import random
import time

img_width, img_height = 224, 224        # Resolution of inputs
batch_size = 64                        # Batch size
epochs = 20                # Maximum number of epochs
# Load INCEPTIONV3
model=applications.VGG16(weights="imagenet", include_top=True, input_shape=(img_width, img_height, 3), classes=1000)

#initialise top model
model_final = model
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])

model_final.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical' )
label_map = (training_set.class_indices)

print(label_map)

itr = test_set = test_datagen.flow_from_directory(
        'val',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')
nb_train_samples = 1300000                # Number of train samples
nb_validation_samples = 50000            # Number of validation samples



model_final.fit_generator(
        training_set,
        samples_per_epoch=nb_train_samples, 
        epochs=1,
        validation_data=test_set,
        validation_steps=nb_validation_samples/64)


model_final.save_weights('keras_vgg16_main.h5')


X, y = itr.next()
arr = model_final.evaluate(X,y);
print(arr)