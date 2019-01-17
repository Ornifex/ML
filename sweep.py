from __future__ import print_function
import numpy as np
import os
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from itertools import chain
from random import randint
from hyperas import optim
from hyperas.distributions import choice, uniform
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import numpy as np
from keras.layers import Dense, Dropout, Activation, Bidirectional
import tensorflow as tf
import pandas as pd

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    features = pd.read_csv("features.csv", index_col=0, header=[0, 1, 2])

    # get current working directory
    cwd = os.getcwd()
    # get the subdirectories of the cwd
    dirs = next(os.walk(cwd))[1]

    train = pd.DataFrame()
    test = pd.DataFrame()
    train_y = np.array([])
    test_y = np.array([])
    
    for idx in range(0,10):
        randidx = randint(0,100)
        genre = features.iloc[idx*100:(1+idx)*100]
        if(randidx < 10):
            train = train.append(genre.iloc[randidx:randidx+90])
            rangeTest1 = genre.iloc[0:randidx]
            rangeTest2 = genre.iloc[randidx + 90:100]
            test = test.append(pd.concat([rangeTest1, rangeTest2]))
        else:
            rangeTrain1 = genre.iloc[randidx:100]
            rangeTrain2 = genre.iloc[0:randidx-10]
            train = train.append(pd.concat([rangeTrain1,rangeTrain2]))
            test = test.append(genre.iloc[randidx-10:randidx])
        train_y = np.append(train_y, np.repeat(idx, 90))
        test_y = np.append(test_y, np.repeat(idx, 10))

    train_x = train.values
    test_x = test.values
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(train_x)
    scaler.transform(test_x)

    data_dim = 21
    timesteps = 20
    num_classes = 10

    #zero pad spectral centroids
    initialPadIdx = 141
    for i in range(0,7):
        train_x = np.insert(train_x,np.s_[initialPadIdx:initialPadIdx+19],0,axis=1)
        test_x = np.insert(test_x,np.s_[initialPadIdx:initialPadIdx+19],0,axis=1)
        initialCentroid = initialPadIdx+20

    initial = 288
    #zero pad spectral constant feature (being 7 elements so adding 13 zero's for 20 timesteps)
    for i in range(0,7):
        train_x = np.insert(train_x,np.s_[initial:initial+13],0,axis=1)
        test_x = np.insert(test_x,np.s_[initial:initial+13],0,axis=1)
        initial = initial + 13

    train_x = np.reshape(train_x, (train_x.shape[0], timesteps, data_dim))
    test_x = np.reshape(test_x, (test_x.shape[0], timesteps, data_dim))

    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)
    return train_x, train_y, test_x, test_y

def create_model(train_x, train_y, test_x, test_y):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    data_dim = 21
    timesteps = 20
    num_classes = 10

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(Bidirectional(LSTM({{choice([128, 256, 512])}},
                input_shape=(timesteps, data_dim),
                dropout={{uniform(0, 1)}}, 
                recurrent_dropout={{uniform(0, 1)}}, 
                return_sequences=True)) ) # returns a sequence of vectors of dimension 32
    model.add(Bidirectional(LSTM({{choice([16, 32, 64])}},
                dropout={{uniform(0, 1)}}, 
                recurrent_dropout={{uniform(0, 1)}},
                return_sequences=False)))  # returns a sequence of vectors of dimension 32
    #model.add(LSTM(32))  # return a single vector of dimension 32
    #model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    from keras import optimizers

    model.compile(loss='categorical_crossentropy',
                optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                metrics=['accuracy'])

    result = model.fit(train_x, train_y,
          batch_size={{choice([10, 20, 40])}}, epochs=200,
          validation_data=(test_x, test_y),
          shuffle=True, verbose=0)
    print(model.summary())
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    #print(model.evaluate(test_x, test_y, batch_size=35))

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)