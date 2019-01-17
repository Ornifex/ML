#!/usr/bin/env python3

import numpy as np
import pandas as pd
import utils

from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense, Embedding
from keras.layers import Dense, Dropout, Activation

train_x, test_x, train_y, test_y = utils.load_features("features.csv", scale=True)


data_dim = 28
timesteps = 7
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
# model.add(GRU(96, input_shape=(timesteps, data_dim), dropout=0.4, recurrent_dropout=0.5))
model.add(GRU(96, input_shape=(timesteps, data_dim), dropout=0.4, recurrent_dropout=0.5, return_sequences=True))
model.add(GRU(48, dropout=0.3, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(10, activation='softmax'))
from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001, decay=1e-6),
              # optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
              metrics=['accuracy'])

# #zero pad spectral centroids
# initialPadIdx = 141
# for i in range(0,7):
#     train_x = np.insert(train_x,np.s_[initialPadIdx:initialPadIdx+19],0,axis=1)
#     test_x = np.insert(test_x,np.s_[initialPadIdx:initialPadIdx+19],0,axis=1)
#     initialCentroid = initialPadIdx+20

# initial = 288
# #zero pad spectral constant feature (being 7 elements so adding 13 zero's for 20 timesteps)
# for i in range(0,7):
#     train_x = np.insert(train_x,np.s_[initial:initial+13],0,axis=1)
#     test_x = np.insert(test_x,np.s_[initial:initial+13],0,axis=1)
#     initial = initial + 13

train_x = np.reshape(train_x, (train_x.shape[0], timesteps, data_dim))
test_x = np.reshape(test_x, (test_x.shape[0], timesteps, data_dim))

import tensorflow as tf

from keras.callbacks import ModelCheckpoint
filepath = "gru-weights.best1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

model.fit(train_x, train_y, batch_size=35, epochs=400,
          validation_data=(test_x, test_y), shuffle=True,
          callbacks=callbacks_list)
print(model.summary())
print(model.evaluate(test_x, test_y, batch_size=35))
