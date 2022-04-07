import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


# from keras.optimizers import Adam, SGD


def predict(dataset, date):
    # print(f'date: {date[:10]}')
    # print(f'dataset: {dataset[:10]}')

    date = np.array(date)
    dataset = np.array(dataset)
    # print(np.shape(date))

    date = date.reshape(-1, 1)
    dataset = dataset.reshape(-1, 1)

    # print(np.shape(date))

    # print(np.shape(date))

    # print(f'date: {date[:10]}')
    # print()
    #
    # for i in date:
    #     print(i)
    #
    # print()

    # print(f'dataset: {dataset[:10]}')

    # plt.figure(figsize=(20, 7))
    # plt.plot(date, dataset, label='Tesla Stock Price', color='red')
    # plt.xticks(np.arange(100, len(dataset), 200))
    # plt.xlabel('Date')
    # plt.ylabel('Price ($)')
    # plt.legend()
    # plt.show()

    num_shape = 1900

    train = dataset[:num_shape]
    test = dataset[num_shape:]

    sc = MinMaxScaler(feature_range=(0, 1))
    train_scaled = sc.fit_transform(train)

    X_train = []

    # Price on next day
    y_train = []

    window = 60

    for i in range(window, num_shape):
        X_train_ = np.reshape(train_scaled[i - window:i, 0], (window, 1))
        X_train.append(X_train_)
        y_train.append(train_scaled[i, 0])
    X_train = np.stack(X_train)
    y_train = np.stack(y_train)

    if os.path.exists('LSTM.h5'):
        model = load_model('LSTM.h5')
    else:
        # Initializing the Recurrent Neural Network
        model = Sequential()
        # Adding the first LSTM layer with a sigmoid activation function and some Dropout regularization
        # Units - dimensionality of the output space

        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50))
        model.add(Dropout(0.2))

        # Adding the output layer
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=300, batch_size=32)

        # Сохранение обученной модели
        model.save('LSTM.h5')

    df_volume = np.vstack((train, test))

    inputs = df_volume[df_volume.shape[0] - test.shape[0] - window:]
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    num_2 = df_volume.shape[0] - num_shape + window

    X_test = []

    for i in range(window, num_2):
        X_test_ = np.reshape(inputs[i - window:i, 0], (window, 1))
        X_test.append(X_test_)

    X_test = np.stack(X_test)
    predict = model.predict(X_test)
    predict = sc.inverse_transform(predict)

    # # diff = predict - test
    # #
    # # print("MSE:", np.mean(diff ** 2))
    # # print("MAE:", np.mean(abs(diff)))
    # # print("RMSE:", np.sqrt(np.mean(diff ** 2)))
    #
    # # print(len(date[1800:]))
    # # print(len(dataset[1800:]))
    #
    # # print(f'date[-predict.shape[0]:]: {date[-predict.shape[0]:]}')
    #
    # # print()
    #
    # # print(f'predict: {predict}')
    #
    # date = np.reshape(date, (len(date),))
    # dataset = np.reshape(dataset, (len(dataset),))
    #
    #
    # # plt.figure(figsize=(20, 7))
    # plt.figure(figsize=(20, 7))
    # plt.plot(date[1800:], dataset[1800:], color='red', label='Real Tesla Stock Price')
    # plt.plot(date[-predict.shape[0]:], predict, color='blue', label='Predicted Tesla Stock Price')
    # # plt.xticks(np.arange(100, len(dataset), 200))
    # plt.title('Tesla Stock Price Prediction')
    # plt.xlabel('Date')
    # plt.ylabel('Price ($)')
    # plt.legend()
    # plt.show()

    # прогноз на 20 дней
    pred_ = predict[-1].copy()
    prediction_full = []
    window = 60
    df_copy = dataset[:]

    for j in range(20):
        df_ = np.vstack((df_copy, pred_))
        train_ = df_[:num_shape]
        test_ = df_[num_shape:]

        df_volume_ = np.vstack((train_, test_))

        inputs_ = df_volume_[df_volume_.shape[0] - test_.shape[0] - window:]
        inputs_ = inputs_.reshape(-1, 1)
        inputs_ = sc.transform(inputs_)

        X_test_2 = []

        for k in range(window, num_2):
            X_test_3 = np.reshape(inputs_[k - window:k, 0], (window, 1))
            X_test_2.append(X_test_3)

        X_test_ = np.stack(X_test_2)
        predict_ = model.predict(X_test_)
        pred_ = sc.inverse_transform(predict_)
        prediction_full.append(pred_[-1][0])
        df_copy = df_[j:]

    prediction_full_new = np.vstack((predict, np.array(prediction_full).reshape(-1, 1)))
    # print(f'prediction_full_new: {prediction_full_new}')

    # df_date = date
    # date2 = date

    # for h in range(20):
    #     df_date_add = pd.to_datetime(df_date['Date'].iloc[-1]) + pd.DateOffset(days=1)
    #     df_date_add = pd.DataFrame([df_date_add.strftime("%Y-%m-%d")], columns=['Date'])
    #     df_date = df_date.append(df_date_add)
    # df_date = df_date.reset_index(drop=True)

    date = np.reshape(date, (len(date),))
    dataset = np.reshape(dataset, (len(dataset),))
    df_volume = np.reshape(df_volume, (len(df_volume),))
    prediction_full_new = np.reshape(prediction_full_new, (len(prediction_full_new),))

    res_date = date[1700:]
    res_volume = df_volume[1700:]
    predict_date = date[-prediction_full_new.shape[0]:]
    predict_value = list(map(str, prediction_full_new))

    """Формирования результрующих данных"""
    print(type(res_date))
    print(type(res_volume))
    print(type(predict_date))
    print(type(predict_value))
    print('преобразуем в лист')

    res_date = list(res_date)
    res_volume = list(res_volume)
    predict_date = list(predict_date)
    predict_value = list(predict_value)

    print(type(res_date))
    print(type(res_volume))
    print(type(predict_date))
    print(type(predict_value))

    data_for_graphic_with_predict = []
    data_for_graphic_with_predict.append(res_date)
    data_for_graphic_with_predict.append(res_volume)
    data_for_graphic_with_predict.append(predict_date)
    data_for_graphic_with_predict.append(predict_value)

    print('-------')

    print(res_date[:10])
    print(res_volume[:10])
    print(predict_date[:10])
    print(predict_value[:10])
    print('-------')

    print(type(res_date))
    print(type(res_volume))
    print(type(predict_date))
    print(type(predict_value))

    print('-------')
    print(type(data_for_graphic_with_predict))

    return data_for_graphic_with_predict
    # return res_date, res_volume, predict_date, predict_value

    # print(res_date[:10])
    # print(res_volume[:10])
    #
    # print(predict_date[:10])
    # print(predict_value[:10])

    #
    # plt.figure(figsize=(20, 7))
    # plt.plot(date[1700:], df_volume[1700:], color='red', label='Real Tesla Stock Price')
    # plt.plot(date[-prediction_full_new.shape[0]:], prediction_full_new, color='blue',
    #          label='Predicted Tesla Stock Price')
    # # plt.xticks(np.arange(100, dataset[1700:], 200))
    # plt.title('Tesla Stock Price Prediction')
    # plt.xlabel('Date')
    # plt.ylabel('Price ($)')
    # plt.legend()
    # plt.show()
