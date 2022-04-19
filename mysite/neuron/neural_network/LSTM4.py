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
        a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def predict(dataset, date, form_data):
    date = np.array(date)
    # dataset = np.array(dataset)
    # print(dataset[:10])

    date = date.reshape(-1, 1)
    # dataset = dataset.reshape(-1, 1)
    # print(dataset[:10])

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(np.array(dataset).reshape(-1, 1))

    train_size = int(len(dataset) * 0.65)
    test_size = len(dataset) - train_size
    train_data, test_data = dataset[:train_size], dataset[train_size:]

    time_step = form_data['time_step']
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Initializing the Recurrent Neural Network
    model = Sequential()
    # Adding the first LSTM layer with a sigmoid activation function and some Dropout regularization
    # Units - dimensionality of the output space

    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))

    model.add(LSTM(units=50, return_sequences=True))

    model.add(LSTM(units=50))

    # Adding the output layer
    model.add(Dense(units=1))

    model.compile(optimizer=form_data['optimizer'], loss=form_data['loss'], metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=form_data['epochs'],
                        batch_size=form_data['batch_size'])

    # Сохранение обученной модели
    current_datetime = datetime.now()
    generate_name = lambda: str(randint(1, 100000000)) + '.h5'
    path = f'save_model_nn/' + str(current_datetime.year) + '/' + str(current_datetime.month) + '/' + str(
        current_datetime.day) + '/' + generate_name()
    model.save(str(BASE_DIR) + '/media/' + path)  # <FieldFile: save_model_nn/2022/04/03/СNN_model_kyjBosa.h5>

    # predict = model.predict(X_test)
    # predict = scaler.inverse_transform(predict)
    #
    # diff = predict - test_data
    #
    # print("MSE:", np.mean(diff ** 2))
    # print("MAE:", np.mean(abs(diff)))
    # print("RMSE:", np.sqrt(np.mean(diff ** 2)))

    return path
