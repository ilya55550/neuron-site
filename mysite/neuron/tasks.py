import datetime

from django.core.mail import send_mail
from mysite.celery import app
from neuron.neural_network.retraining import retraining
from neuron.utils import data_API
from .models import *
from decouple import config

@app.task
def retraining_models():
    companies = ListСompanies.objects.all()
    for company in companies:
        try:
            model = TrainedNeuralNetwork.objects.get(company=company.name)
            data_for_graphic = data_API(company.ticker)
            value = list(reversed(data_for_graphic.values()))
            retraining(value, model.time_step, str(model.file_trained_nn))
            model.time_update = datetime.datetime.now()
            model.save()
        except Exception as e:
            continue