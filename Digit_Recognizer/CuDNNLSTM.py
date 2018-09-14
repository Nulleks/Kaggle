# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 02:03:46 2018

@author: 0x
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

training_data = pd.read_csv('train.csv')
testing_data = pd.read_csv('test.csv')

X_train = training_data.iloc[:,1:].values
Y_train = training_data.iloc[:,:1]






Y_train = Y_train.iloc[:,0].values
X_train= X_train.reshape(-1, 28,28)/255
X_test = testing_data.values.reshape(-1, 28, 28)/255


#### Change loss to categorical_crossentropy if this uncommented  #####
#from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
#Y_train = to_categorical(Y_train, num_classes = 10)



from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=42)





g = plt.imshow(X_train[1][:,:,0])



print(X_train.shape)
print(X_train[0].shape)


model = Sequential()

# IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))
model.add(Dropout(0.1))


model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy', #categorical_crossentropy if label changed to categorical
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(X_train,
          Y_train,
          epochs=25)
,          
          validation_data=(X_val, Y_val))



# Predict


Y_predict = model.predict_classes(X_test)
ids = np.arange(1,28001)

output = pd.DataFrame({ 'ImageId' : ids, 'Label': Y_predict })

output.to_csv('LSTM_test.csv', index = False)
