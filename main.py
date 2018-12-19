import pandas as pd
import os
import numpy as np

features = pd.read_csv("features.csv", index_col=0, header=[0, 1, 2])

# get current working directory
cwd = os.getcwd()
# get the subdirectories of the cwd
dirs = next(os.walk(cwd))[1]

train = pd.DataFrame()
test = pd.DataFrame()
train_y = np.array([])
test_y = np.array([])
from itertools import chain
from random import randint
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
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler
scaler = StandardScaler(copy=False)
scaler.fit_transform(train_x)
scaler.transform(test_x)

from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(train_x, train_y)
print(clf.score(test_x, test_y))

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(max_iter=2000)
clf.fit(train_x, train_y)
print(clf.score(test_x, test_y))

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import numpy as np
from keras.layers import Dense, Dropout, Activation
data_dim = 21
timesteps = 20
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(128,
               input_shape=(timesteps, data_dim),dropout=0.35, recurrent_dropout=0.55, ))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(16,
#               dropout=0.15, recurrent_dropout=0.4,return_sequences=False))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))
model.load_weights('weights-improvement-156-1.14.hdf5')
from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy'])

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

import tensorflow as tf

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

#create early stopping checkpoints based on minimum validation loss
from keras.callbacks import ModelCheckpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(train_x, train_y,
          batch_size=35, epochs=400,
          validation_data=(test_x, test_y),shuffle=True,callbacks=callbacks_list)
print(model.summary())
print(model.evaluate(test_x, test_y, batch_size=35))


#test model using best weights
#model.load_weights('weights-improvement-156-1.14.hdf5')
#scores = model.evaluate(test_x,test_y,verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
