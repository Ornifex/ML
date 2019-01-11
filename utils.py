import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler


def load_features(filename, test_ratio=0.1, scale=False, scaler=StandardScaler(copy=False)):
    features = pd.read_csv(filename, index_col=0, header=[0, 1, 2])
    train = pd.DataFrame()
    test = pd.DataFrame()
    train_y = np.array([])
    test_y = np.array([])

    for idx in range(0, 10):
        randidx = random.randint(0, 100)
        genre = features.iloc[idx*100:(1+idx)*100]
        if randidx < test_ratio * 100:
            train = train.append(genre.iloc[randidx:int(randidx + (1 - test_ratio) * 100)])
            rangeTest1 = genre.iloc[0:randidx]
            rangeTest2 = genre.iloc[int(randidx + (1 - test_ratio) * 100):100]
            test = test.append(pd.concat([rangeTest1, rangeTest2]))
        else:
            rangeTrain1 = genre.iloc[randidx:100]
            rangeTrain2 = genre.iloc[0:int(randidx - test_ratio * 100)]
            train = train.append(pd.concat([rangeTrain1, rangeTrain2]))
            test = test.append(genre.iloc[int(randidx - test_ratio * 100):randidx])
        train_y = np.append(train_y, np.repeat(idx, (1 - test_ratio) * 100))
        test_y = np.append(test_y, np.repeat(idx, test_ratio * 100))

    train_x = train.values
    test_x = test.values
    if scale:
        scaler.fit_transform(train_x)
        scaler.transform(test_x)
    print(test_x.shape)
    return train_x, test_x, train_y, test_y
