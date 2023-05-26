import re
from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, UserChangeForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from captcha.fields import CaptchaField

from .models import *


class CustomUserCreationForm(UserCreationForm):
    username = forms.CharField(label='Логин', widget=forms.TextInput(attrs={'class': 'form-input'}))
    email = forms.EmailField(label='Email', widget=forms.EmailInput(attrs={'class': 'form-input'}))
    password1 = forms.CharField(label='Пароль', widget=forms.PasswordInput(attrs={'class': 'form-input'}))
    password2 = forms.CharField(label='Повтор пароля', widget=forms.PasswordInput(attrs={'class': 'form-input'}))
    researcher = forms.BooleanField(label="Исследователь", required=False,
                                    widget=forms.CheckboxInput(attrs={'class': 'checkbox-style'}))

    class Meta(UserCreationForm):
        model = CustomUser
        fields = ('username', 'email', 'researcher')


class CustomUserChangeForm(UserChangeForm):

    def __init__(self, *args, **kwargs):
        self.request = kwargs.pop('request', None)
        super(CustomUserChangeForm, self).__init__(*args, **kwargs)

    class Meta:
        model = CustomUser
        fields = ('username', 'password')


class LoginUserForm(AuthenticationForm):
    username = forms.CharField(label='Логин', widget=forms.TextInput(attrs={'class': 'form-input'}))
    password = forms.CharField(label='Пароль', widget=forms.PasswordInput(attrs={'class': 'form-input'}))

    class Meta:
        model = CustomUser
        fields = ('username', 'password')

    def clean_username(self):
        username = self.cleaned_data['username']
        if re.match(r'\d', username):
            raise ValidationError('Название не должно начинаться с цифры')
        return username


class ContactForm(forms.Form):
    name = forms.CharField(label='Имя', max_length=255)
    email = forms.EmailField(label='Email')
    content = forms.CharField(widget=forms.Textarea(attrs={'cols': 60, 'rows': 10}))
    captcha = CaptchaField()


class ChoiceParamResearcher(forms.Form):
    company = forms.ModelChoiceField(label='Тикер компании', widget=forms.Select(attrs={"class": "select-company",
                                                                                        "onchange": "if (this.selectedIndex) send_company();"}),
                                     queryset=ListСompanies.objects.all())

    trained_nn_id = forms.ModelChoiceField(label='Модель нейронной сети',
                                           widget=forms.Select(attrs={'class': 'form-input'}),
                                           queryset=TrainedNeuralNetworkUser.objects.all())
    predict_daily = forms.IntegerField(label='Количество дней прогноза',
                                       widget=forms.TextInput(attrs={'class': 'form-input'}))


class ChoiceParam(forms.Form):
    company = forms.ModelChoiceField(label='Тикер компании', widget=forms.Select(attrs={'class': 'form-input'}),
                                     queryset=ListСompanies.objects.all())
    predict_daily = forms.IntegerField(label='Количество дней прогноза',
                                       widget=forms.TextInput(attrs={'class': 'form-input'}))

    def clean_predict_daily(self):
        predict_daily = self.cleaned_data['predict_daily']
        if predict_daily > 365 or predict_daily < 2:
            raise ValidationError('Возможный диапазон прогноза 2-365 дней')
        return predict_daily


class TrainingForm(forms.ModelForm):
    company = forms.ModelChoiceField(label='Тикер компании', widget=forms.Select(attrs={'class': 'form-input'}),
                                     queryset=ListСompanies.objects.all())

    class Meta:
        model = TrainedNeuralNetworkUser
        fields = ('neural_network_architecture', 'time_step', 'loss', 'optimizer', 'epochs', 'batch_size', 'company')
        widgets = {
            "neural_network_architecture": forms.Select(attrs={'class': 'form-input'}),
            "time_step": forms.TextInput(attrs={'class': 'form-input'}),
            "loss": forms.Select(attrs={'class': 'form-input'}),
            "optimizer": forms.Select(attrs={'class': 'form-input'}),
            "epochs": forms.TextInput(attrs={'class': 'form-input'}),
            "batch_size": forms.TextInput(attrs={'class': 'form-input'}),
        }
