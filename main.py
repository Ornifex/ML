import pandas as pd
import os
import numpy as np

features = pd.read_csv("features.csv", index_col=0, header=[0, 1, 2])

print(features)

# get current working directory
cwd = os.getcwd()
# get the subdirectories of the cwd
dirs = next(os.walk(cwd))[1]

train = pd.DataFrame()
test = pd.DataFrame()
train_y = np.array([])
test_y = np.array([])

for idx, dir in enumerate(dirs):
    genre = features.iloc[idx*100:(1+idx)*100]
    train = train.append(genre.iloc[0:90])
    test = test.append(genre.iloc[90:100])
    train_y = np.append(train_y, np.repeat(idx, 90))
    test_y = np.append(test_y, np.repeat(idx, 10))


train_x = train.values
test_x  = test.values

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
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 196
timesteps = 1
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(128, 
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

train_x = np.reshape(train_x, (train_x.shape[0], timesteps, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], timesteps, test_x.shape[1]))

from keras.utils import to_categorical
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

model.fit(train_x, train_y,
          batch_size=24, epochs=50,
          validation_data=(test_x, test_y))

print(model.evaluate(test_x, test_y, batch_size=64))
