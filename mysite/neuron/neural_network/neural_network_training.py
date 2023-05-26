import numpy as np
from keras.models import Sequential, load_model
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
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


def training(dataset, date, form_data):
    date = np.array(date)

    date = date.reshape(-1, 1)

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

    if str(form_data['neural_network_architecture']) == 'LSTM':
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

    if str(form_data['neural_network_architecture']) == 'GRU':
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(GRU(units=50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(GRU(units=50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(GRU(units=50))
        model.add(Dense(units=1))

    model.compile(optimizer=form_data['optimizer'], loss=form_data['loss'], metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, ytest),
                        epochs=form_data['epochs'],
                        batch_size=form_data['batch_size'])

    # tf.keras.metrics.CategoricalAccuracy(), metrics.AUC(), metrics.Precision(), metrics.Recall()
    # model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer,
    #               metrics=[tf.keras.metrics.CategoricalAccuracy(), metrics.AUC(), metrics.Precision(),
    #                        metrics.Recall()])

    # Сохранение обученной модели
    current_datetime = datetime.now()
    generate_name = lambda: str(randint(1, 100000000)) + '.h5'
    path = f'save_model_nn/' + str(current_datetime.year) + '/' + str(current_datetime.month) + '/' + str(
        current_datetime.day) + '/' + generate_name()
    model.save(str(BASE_DIR) + '/media/' + path)  # <FieldFile: save_model_nn/2022/04/03/СNN_model_kyjBosa.h5>

    acurracy_dict = {history.epoch[i] + 1: history.history['accuracy'][i] for i in range(len(history.epoch))}
    loss_dict = {history.epoch[i] + 1: history.history['loss'][i] for i in range(len(history.epoch))}

    return path, acurracy_dict, loss_dict
