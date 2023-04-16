import datetime

from django.core.mail import send_mail
from mysite.celery import app
from neuron.neural_network.retraining import retraining
from neuron.utils import data_API
from .models import *
from decouple import config

@app.task
def retraining_models():
    print("Начало выполнения celery task")
    companies = ListСompanies.objects.all()
    for company in companies:
        try:
            model = TrainedNeuralNetwork.objects.get(company=company.name)
            """Обращаемся к апи"""
            data_for_graphic = data_API(company.ticker)
            """Формируем набор данных для нейронки"""
            value = list(reversed(data_for_graphic.values()))
            """Обучаем модель, возвращаем путь к файлу модели"""
            retraining(value, model.time_step, str(model.file_trained_nn))
            model.time_update = datetime.datetime.now()
            model.save()
            print(f"Для {str(company.name)} выполнилось!!!!!!!!!!")
        except Exception as e:
            continue