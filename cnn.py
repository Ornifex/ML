#!/usr/bin/env python3

import numpy as np
import pandas as pd
import utils

from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling1D, Flatten
from keras.layers import Dense, Dropout, Activation, Embedding

train_x, test_x, train_y, test_y = utils.load_features("features.csv", scale=True)


data_dim = 14
timesteps = 14
num_classes = 10

model = Sequential()

model.add(Conv1D(128, 3, input_shape=(timesteps, data_dim), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(32, 3, activation='relu'))
model.add(MaxPooling1D(3))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))
from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001, decay=1e-6),
              # optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
              metrics=['accuracy'])

train_x = np.reshape(train_x, (train_x.shape[0], timesteps, data_dim))
test_x = np.reshape(test_x, (test_x.shape[0], timesteps, data_dim))

import tensorflow as tf

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

model.fit(train_x, train_y, batch_size=35, epochs=400,
          validation_data=(test_x, test_y), shuffle=True)
print(model.summary())
print(model.evaluate(test_x, test_y, batch_size=35))
