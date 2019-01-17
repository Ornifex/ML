#!/usr/bin/env python3

import numpy as np
import pandas as pd
import utils

from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, ZeroPadding2D
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers.normalization import BatchNormalization

from keras import regularizers

from keras import backend as K
# K.set_image_dim_ordering('th')

train_x, test_x, train_y, test_y = utils.load_features("features.csv", scale=True)


data_dim = 14
timesteps = 14
num_classes = 10


def alexnet_model(img_shape=(14, 14, 1), n_classes=10, l2_reg=0., weights=None):

    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(96, (3, 3), input_shape=img_shape,
                       padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(256, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(512, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(n_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    if weights is not None:
        alexnet.load_weights(weights)

    alexnet.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(lr=0.001, decay=1e-6),
                    metrics=['accuracy'])

    return alexnet

model = Sequential()

model.add(Conv1D(256, 3, input_shape=(timesteps, data_dim), activation='relu'))
model.add(Dropout(0.35))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001, decay=1e-6),
              # optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
              metrics=['accuracy'])

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', kernel_regularizer=regularizers.l2(0.01),
                 input_shape=(timesteps, data_dim, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adadelta(lr=0.001, decay=1e-6),
              metrics=['accuracy'])


train_x = np.reshape(train_x, (train_x.shape[0], timesteps, data_dim, 1))
test_x = np.reshape(test_x, (test_x.shape[0], timesteps, data_dim, 1))

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
filepath = "cnn_weights.best1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

alexnet_model().fit(train_x, train_y, batch_size=35, epochs=400,
          validation_data=(test_x, test_y), shuffle=True,
          callbacks=callbacks_list)
print(model.summary())
print(model.evaluate(test_x, test_y, batch_size=35))
