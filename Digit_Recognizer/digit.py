# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 18:00:57 2018

@author: 0x
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns



print("Loading Training and Testing Data =====>")
training_data = pd.read_csv('train.csv')
testing_data = pd.read_csv('test.csv')

X_train = training_data.iloc[:,1:].values
Y_train = training_data.iloc[:,:1]

# keras.utils.to_categorical(Y_train, num_classes)
Y_train = pd.get_dummies(Y_train, columns=['label'])
X_train= X_train.reshape(-1, 28,28,1)/255


import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.layers import Dropout


def neural_network_model():
    # Initialising the CNN
    classifier = Sequential()
    # Convolution
    # Creating feature map, with 3,3 filter
    classifier.add(Conv2D(32, kernel_size=(3, 3), input_shape = (28, 28, 1), activation = 'relu'))    
    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    # Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.25))
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Flattening
    classifier.add(Flatten())    
    # Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 10, activation = 'softmax'))
    
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  # cross entropy better for classification
    return classifier


classifier = neural_network_model()
classifier.fit(X_train, Y_train, batch_size = 32, epochs =25, verbose=1) # 0.8058

# Predict
X_test = testing_data.values.reshape(-1, 28, 28, 1)/255

Y_predict = classifier.predict_classes(X_test)
ids = np.arange(1,28001)

output = pd.DataFrame({ 'ImageId' : ids, 'Label': Y_predict })

output.to_csv('Digit_Recognizer.csv', index = False)







