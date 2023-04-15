from django.db import models
from django.contrib.auth.models import User, AbstractUser
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.urls import reverse
from neuron.neural_network.Choices import *


class CustomUser(AbstractUser):
    researcher = models.BooleanField(default=False, blank=True)


class NeuralNetwork(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(max_length=100, unique=True, db_index=True)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='networks')

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse('show_network', kwargs={'network_slug': self.slug})

    class Meta:
        verbose_name = 'Нейронная сеть'
        verbose_name_plural = 'Нейронные сети'
        ordering = ['id']


class TrainedNeuralNetworkUser(models.Model):
    # creator = models.ForeignKey('CustomUser', on_delete=models.SET_NULL, null=True)
    creator = models.CharField(max_length=100, null=True)
    time_step = models.IntegerField(null=True)
    loss = models.CharField(max_length=100, choices=choices_loss)
    optimizer = models.CharField(max_length=100, choices=choices_optimizer)
    epochs = models.IntegerField()
    batch_size = models.IntegerField()
    file_trained_nn = models.FileField(upload_to='save_model_nn/%Y/%m/%d/')
    time_create = models.DateTimeField(auto_now_add=True)
    neural_network_architecture = models.ForeignKey('NeuralNetwork', on_delete=models.CASCADE)
    company = models.CharField(max_length=100, null=True)

    def __str__(self):
        return str(self.pk)

    def get_absolute_url(self):
        return reverse('predict', kwargs={'nn_id': self.pk})

    class Meta:
        verbose_name = 'Пользовательская обученная нейронная сеть'
        verbose_name_plural = 'Пользовательские обученные нейронные сети'
        ordering = ['-time_create']


class ListСompanies(models.Model):
    name = models.CharField(max_length=100)
    ticker = models.CharField(max_length=10, db_index=True, unique=True)

    def __str__(self):
        return self.name

    # def get_absolute_url(self):
    #     return reverse('predict', kwargs={'company_id': self.pk})

    class Meta:
        verbose_name = 'Компания'
        verbose_name_plural = 'Компании'
        ordering = ['ticker']


class TrainedNeuralNetwork(models.Model):
    file_trained_nn = models.FileField(upload_to='harvested_save_model_nn/')
    company = models.CharField(max_length=100, null=True)
    time_step = models.IntegerField(null=True)
    time_update = models.DateTimeField(null=True)

    def __str__(self):
        return str(self.pk)

    def get_absolute_url(self):
        return reverse('predict', kwargs={'nn_id': self.pk})

    class Meta:
        verbose_name = 'Обученная нейронная сеть'
        verbose_name_plural = 'Обученные нейронные сети'
