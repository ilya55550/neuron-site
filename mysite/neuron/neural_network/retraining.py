import os

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from random import randint
from datetime import datetime
from mysite.settings import BASE_DIR


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def retraining(dataset, time_step, path: str):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(np.array(dataset).reshape(-1, 1))

    train_size = int(len(dataset) * 0.65)
    train_data, test_data = dataset[:train_size], dataset[train_size:]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = load_model(str(BASE_DIR) + '/media/' + path)

    model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=1, batch_size=64)

    # Удаление прошлой модели
    os.remove(str(BASE_DIR) + '/media/' + path)

    # Сохранение новой модели
    model.save(str(BASE_DIR) + '/media/' + path)  # <FieldFile: save_model_nn/2022/04/03/СNN_model_kyjBosa.h5>
    return path
