from django.db import models
from django.contrib.auth.models import User, AbstractUser
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.urls import reverse


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


class TrainedNeuralNetwork(models.Model):
    creator = models.CharField(max_length=100)
    time_step = models.IntegerField(null=True)
    loss = models.CharField(max_length=100)
    optimizer = models.CharField(max_length=100)
    epochs = models.IntegerField()
    batch_size = models.IntegerField()
    file_trained_nn = models.FileField(upload_to='save_model_nn/%Y/%m/%d/')
    time_create = models.DateTimeField(auto_now_add=True)
    neural_network_architecture = models.ForeignKey('NeuralNetwork', on_delete=models.CASCADE)

    def __str__(self):
        return self.creator

    def get_absolute_url(self):
        return reverse('predict', kwargs={'nn_id': self.pk})

    class Meta:
        verbose_name = 'Обученная нейронная сеть'
        verbose_name_plural = 'Обученные нейронные сети'
        ordering = ['-time_create']


class ListСompanies(models.Model):
    name = models.CharField(max_length=100)
    ticker = models.CharField(max_length=10, db_index=True, unique=True)

    def __str__(self):
        return self.ticker

    # def get_absolute_url(self):
    #     return reverse('predict', kwargs={'company_id': self.pk})

    class Meta:
        verbose_name = 'Компания'
        verbose_name_plural = 'Компании'
        ordering = ['ticker']
