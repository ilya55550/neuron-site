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



def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)



def predict(dataset, date, predict_daily):
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
        model.save('LSTM_with100.h5')  # <FieldFile: save_model_nn/2022/04/03/СNN_model_kyjBosa.h5>

    predict = model.predict(X_test)
    predict = scaler.inverse_transform(predict)

    diff = predict - test_data

    print("MSE:", np.mean(diff ** 2))
    print("MAE:", np.mean(abs(diff)))
    print("RMSE:", np.sqrt(np.mean(diff ** 2)))


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
    # number_forecast_days = 30

    while (i < predict_daily):
        # print(f'iter: {i}')
        # print(f'temp_input: {temp_input}')
        # print(f'lentemp_input: {len(temp_input)}')

        x_input = np.array(temp_input[-time_step:])
        # print(f'x_input: {x_input}')
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

    # print(f'После: {list_output}')
    list_output = scaler.inverse_transform(list_output).tolist()
    # print(f'До: {list_output}')

    # print(list_output)

    """Генерация новых дат для прогноза"""
    date_list = pd.bdate_range(datetime.today(), periods=predict_daily).tolist()
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

    # print(f'predict_date: {predict_date}')
    # print(f'len(predict_date): {len(predict_date)}')
    #
    # print(f'predict_value: {predict_value}')
    # print(f'len(predict_value): {len(predict_value)}')


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
