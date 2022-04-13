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


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def predict(dataset, date):
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

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # print('X_test.shape[1]: {X_test.shape[1]}')

    if os.path.exists('LSTM_with100.h5'):
        model = load_model('LSTM_with100.h5')
    else:
        # Initializing the Recurrent Neural Network
        model = Sequential()
        # Adding the first LSTM layer with a sigmoid activation function and some Dropout regularization
        # Units - dimensionality of the output space

        model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))

        model.add(LSTM(units=50, return_sequences=True))

        model.add(LSTM(units=50))

        # Adding the output layer
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        history = model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=100, batch_size=64)

        # Сохранение обученной модели
        model.save('LSTM_with100.h5')

    # time_step = 100

    x_input = test_data[-time_step:].reshape(1, -1)
    # print(f'x_input: {x_input}')

    temp_input = list(x_input)
    # print(f'temp_input: {temp_input}')

    temp_input = temp_input[0].tolist()
    # print(f'temp_input: {temp_input}')

    temp_input = list(map(float, temp_input))
    # print(f'temp_input: {temp_input}')

    list_output = []
    i = 0
    number_forecast_days = 30

    while (i < number_forecast_days):
        # print(f'iter: {i}')
        # print(f'temp_input: {temp_input}')
        # print(f'lentemp_input: {len(temp_input)}')

        x_input = np.array(temp_input[-time_step:])
        print(f'x_input: {x_input}')
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, time_step, 1))
        # print(f'x_input: {x_input}')
        # print(f'lenx_input: {len(x_input)}')

        single_value_predict = model.predict(x_input, verbose=0)
        temp_input.extend(single_value_predict[0].tolist())
        list_output.extend(single_value_predict.tolist())

        i += 1

    # print(f'list_output: {list_output}')
    # print(f'lenlist_output: {len(list_output)}')

    print(f'После: {list_output}')
    list_output = scaler.inverse_transform(list_output).tolist()
    print(f'До: {list_output}')

    # print(list_output)

    """Генерация новых дат для прогноза"""
    date_list = pd.bdate_range(datetime.today(), periods=number_forecast_days).tolist()
    list_date = list(map(lambda x: str(x.date()), date_list))

    """Формирования результрующих данных"""
    date = np.reshape(date, (len(date),))
    dataset = np.reshape(dataset, (len(dataset),))
    predict_date = np.reshape(list_date, (len(list_date),))
    list_output = np.reshape(list_output, (len(list_output),))

    # print(date)
    # print(dataset)
    # print(predict_date)
    # print(list_output)

    res_date = date[-time_step:]
    res_volume = dataset[-time_step:]
    predict_value = list(map(str, list_output))

    res_date = list(res_date)
    res_volume = list(res_volume)
    predict_date = list(predict_date)
    predict_value = list(predict_value)

    # print(f'train_date: {train_date[:2]}')
    # print(f'test_date: {test_date[:2]}')
    #
    # print(f'predict_date: {predict_date}')
    # print(f'predict_value: {predict_value}')

    data_for_graphic_with_predict = []
    data_for_graphic_with_predict.append(res_date)
    data_for_graphic_with_predict.append(res_volume)
    data_for_graphic_with_predict.append(predict_date)
    data_for_graphic_with_predict.append(predict_value)

    # plt.figure(figsize=(10, 20))
    # plt.subplot(2, 2, 1)
    # plt.plot(history.history['loss'])
    # plt.title("loss nn")
    # plt.subplot(2, 2, 2)
    # plt.plot(history.history['accuracy'])
    # plt.title("accuracy nn")
    # plt.show()

    return data_for_graphic_with_predict

# """Генерация новых дат для прогноза"""
#     date_list = pd.bdate_range(datetime.today(), periods=number_forecast_days).tolist()
#     res_predict_date = list(map(lambda x: str(x.date()), date_list))


# df_volume = np.vstack((train, test))
#
# inputs = df_volume[df_volume.shape[0] - test.shape[0] - window:]
# inputs = inputs.reshape(-1, 1)
# inputs = sc.transform(inputs)
#
# num_2 = df_volume.shape[0] - size_half_dataset + window
#
# X_test = []
#
# for i in range(window, num_2):
#     X_test_ = np.reshape(inputs[i - window:i, 0], (window, 1))
#     X_test.append(X_test_)
#
# X_test = np.stack(X_test)
# predict = model.predict(X_test)
# predict = sc.inverse_transform(predict)
#
# # # diff = predict - test
# # #
# # # print("MSE:", np.mean(diff ** 2))
# # # print("MAE:", np.mean(abs(diff)))
# # # print("RMSE:", np.sqrt(np.mean(diff ** 2)))
# #
# # # print(len(date[1800:]))
# # # print(len(dataset[1800:]))
# #
# # # print(f'date[-predict.shape[0]:]: {date[-predict.shape[0]:]}')
# #
# # # print()
# #
# # # print(f'predict: {predict}')
# #
# # date = np.reshape(date, (len(date),))
# # dataset = np.reshape(dataset, (len(dataset),))
# #
# #
# # # plt.figure(figsize=(20, 7))
# # plt.figure(figsize=(20, 7))
# # plt.plot(date[1800:], dataset[1800:], color='red', label='Real Tesla Stock Price')
# # plt.plot(date[-predict.shape[0]:], predict, color='blue', label='Predicted Tesla Stock Price')
# # # plt.xticks(np.arange(100, len(dataset), 200))
# # plt.title('Tesla Stock Price Prediction')
# # plt.xlabel('Date')
# # plt.ylabel('Price ($)')
# # plt.legend()
# # plt.show()
#
# # прогноз на n дней
# number_forecast_days = 60
#
#
# pred_ = predict[-1].copy()
# prediction_full = []
# window = 60
# df_copy = dataset[:]
#
# time_predict = 0
# for j in range(number_forecast_days):
#     df_ = np.vstack((df_copy, pred_))
#     train_ = df_[:size_half_dataset]
#     test_ = df_[size_half_dataset:]
#
#     df_volume_ = np.vstack((train_, test_))
#
#     inputs_ = df_volume_[df_volume_.shape[0] - test_.shape[0] - window:]
#     inputs_ = inputs_.reshape(-1, 1)
#     inputs_ = sc.transform(inputs_)
#
#     X_test_2 = []
#
#     for k in range(window, num_2):
#         X_test_3 = np.reshape(inputs_[k - window:k, 0], (window, 1))
#         X_test_2.append(X_test_3)
#
#     X_test_ = np.stack(X_test_2)
#     print(f'X_test_: {X_test_[0]}')
#     print(f'len(X_test_): {len(X_test_[0])}')
#
#     predict_ = model.predict(X_test_)
#     pred_ = sc.inverse_transform(predict_)
#     prediction_full.append(pred_[-1][0])
#     df_copy = df_[j:]
#
#     time_predict += 1
#     print(time_predict)
#
# prediction_full_new = np.vstack((predict, np.array(prediction_full).reshape(-1, 1)))
# # print(f'prediction_full_new: {prediction_full_new}')
#
# df_date = list(date)
# # date2 = date
#
# """Генерация новых дат для прогноза"""
# date_list = pd.bdate_range(datetime.today(), periods=number_forecast_days).tolist()
# res_predict_date = list(map(lambda x: str(x.date()), date_list))
#
#
#
#
# # for h in range(number_forecast_days):
# #     df_date_add = pd.to_datetime(date[-1]) + pd.DateOffset(days=1)
# #     df_date_add = pd.DataFrame([df_date_add.strftime("%Y-%m-%d")], columns=['Date'])
# #     res_predict_date.append(df_date_add)
# # # res_predict_date = res_predict_date.reset_index(drop=True)
#
# print(type(res_predict_date))
# print(f'res_predict_date {res_predict_date}')
#
# date = np.reshape(date, (len(date),))
# # dataset = np.reshape(dataset, (len(dataset),))
# df_volume = np.reshape(df_volume, (len(df_volume),))
# prediction_full_new = np.reshape(prediction_full_new, (len(prediction_full_new),))
#
# res_date = date[len(date) - 59:]
# res_volume = df_volume[len(dataset) - 59:]
# predict_date = date[-prediction_full_new.shape[0]:]
# predict_value = list(map(str, prediction_full_new))
#
# """Формирования результрующих данных"""
#
# res_date = list(res_date)
# res_volume = list(res_volume)
# predict_date = list(predict_date)
# predict_value = list(predict_value)
#
# print(f'train_date: {train_date[:2]}')
# print(f'test_date: {test_date[:2]}')
#
# print(f'predict_date: {predict_date}')
# print(f'predict_value: {predict_value}')
#
# data_for_graphic_with_predict = []
# data_for_graphic_with_predict.append(res_date)
# data_for_graphic_with_predict.append(res_volume)
# data_for_graphic_with_predict.append(predict_date)
# data_for_graphic_with_predict.append(predict_value)
#
# return data_for_graphic_with_predict
