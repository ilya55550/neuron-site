import os

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from mysite.settings import BASE_DIR


def predict(dataset, date, predict_daily, selected_trained_nn_path, selected_trained_nn_time_step):
    date = np.array(date)
    date = date.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(np.array(dataset).reshape(-1, 1))

    time_step = selected_trained_nn_time_step

    model = load_model(str(BASE_DIR) + '/media/' + selected_trained_nn_path)

    x_input = dataset[-time_step:].reshape(1, -1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    temp_input = list(map(float, temp_input))
    list_output = []
    i = 0

    while (i < predict_daily):
        x_input = np.array(temp_input[-time_step:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, time_step, 1))

        single_value_predict = model.predict(x_input, verbose=0)
        temp_input.extend(single_value_predict[0].tolist())
        list_output.extend(single_value_predict.tolist())

        i += 1

    list_output = scaler.inverse_transform(list_output).tolist()

    """Генерация новых дат для прогноза"""
    date_list = pd.bdate_range(datetime.today(), periods=predict_daily).tolist()
    list_date = list(map(lambda x: str(x.date()), date_list))

    """Формирования результрующих данных"""
    date = np.reshape(date, (len(date),))
    dataset = np.reshape(dataset, (len(dataset),))
    predict_date = np.reshape(list_date, (len(list_date),))
    list_output = np.reshape(list_output, (len(list_output),))

    res_date = date[-time_step:]
    res_volume = dataset[-time_step:]
    predict_value = list(map(str, list_output))

    res_date = list(res_date)
    res_volume = list(res_volume)
    predict_date = list(predict_date)
    predict_value = list(predict_value)

    data_for_graphic_with_predict = []
    data_for_graphic_with_predict.append(res_date)
    data_for_graphic_with_predict.append(res_volume)
    data_for_graphic_with_predict.append(predict_date)
    data_for_graphic_with_predict.append(predict_value)

    return data_for_graphic_with_predict


def predict_past_data(dataset, date, predict_daily, selected_trained_nn_path, selected_trained_nn_time_step):
    time_step = selected_trained_nn_time_step
    print(f'time_step: {time_step}')
    print(f'predict_daily: {predict_daily}')

    date = np.array(date[-(predict_daily + time_step):])
    date = date.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    res_dataset = dataset[-predict_daily:]
    dataset = scaler.fit_transform(np.array(dataset[-(predict_daily + time_step):]).reshape(-1, 1))

    print(f'len(dataset): {len(dataset)}')
    print(f'len(date): {len(date)}')

    model = load_model(str(BASE_DIR) + '/media/' + selected_trained_nn_path)

    x_input = dataset[-time_step:].reshape(1, -1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    temp_input = list(map(float, temp_input))
    list_output = []
    i = 0

    while (i < predict_daily):
        x_input = np.array(temp_input[-time_step:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, time_step, 1))

        single_value_predict = model.predict(x_input, verbose=0)
        temp_input.extend(single_value_predict[0].tolist())
        list_output.extend(single_value_predict.tolist())

        i += 1

    list_output = scaler.inverse_transform(list_output).tolist()
    predict_date = date[-predict_daily:]

    """Формирования результрующих данных"""
    date = np.reshape(date, (len(date),))

    predict_date = np.reshape(predict_date, (len(predict_date),))
    list_output = np.reshape(list_output, (len(list_output),))

    res_date = date[-predict_daily:]

    predict_value = list(map(str, list_output))

    res_date = list(res_date)

    predict_date = list(predict_date)
    predict_value = list(predict_value)

    print(f'res_date: {res_date}')
    print(f'res_dataset: {res_dataset}')
    print(f'predict_date: {predict_date}')
    print(f'predict_value: {predict_value}')

    print(f'len_res_date: {len(res_date)}')
    print(f'len_res_dataset: {len(res_dataset)}')
    print(f'len_predict_date: {len(predict_date)}')
    print(f'len_predict_value: {len(predict_value)}')

    data_for_graphic_with_predict = []
    data_for_graphic_with_predict.append(res_date)
    data_for_graphic_with_predict.append(res_dataset)
    data_for_graphic_with_predict.append(predict_date)
    data_for_graphic_with_predict.append(predict_value)

    return data_for_graphic_with_predict
