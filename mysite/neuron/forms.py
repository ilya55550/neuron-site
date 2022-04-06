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


# class RegisterUserForm(forms.ModelForm):
#     username = forms.CharField(label='Логин', widget=forms.TextInput(attrs={'class': 'form-input'}))
#     email = forms.EmailField(label='Email', widget=forms.EmailInput(attrs={'class': 'form-input'}))
#     password1 = forms.CharField(label='Пароль', widget=forms.PasswordInput(attrs={'class': 'form-input'}))
#     password2 = forms.CharField(label='Повтор пароля', widget=forms.PasswordInput(attrs={'class': 'form-input'}))
#     researcher = forms.BooleanField(label="Исследователь", required=False)
#
#     class Meta:
#         model = User
#         fields = '__all__'


class LoginUserForm(AuthenticationForm):
    username = forms.CharField(label='Логин', widget=forms.TextInput(attrs={'class': 'form-input'}))
    password = forms.CharField(label='Пароль', widget=forms.PasswordInput(attrs={'class': 'form-input'}))

    class Meta:
        model = CustomUser
        fields = ('username', 'password')

    def clean_title(self):
        username = self.cleaned_data['username']
        if re.match(r'\d', username):
            raise ValidationError('Название не должно начинаться с цифры')
        return username


class ContactForm(forms.Form):
    name = forms.CharField(label='Имя', max_length=255)
    email = forms.EmailField(label='Email')
    content = forms.CharField(widget=forms.Textarea(attrs={'cols': 60, 'rows': 10}))
    captcha = CaptchaField()


class ChoiceParam(forms.Form):
    company = forms.ModelChoiceField(queryset=ListСompanies.objects.all())
    time_frame = forms.ChoiceField(label='Layout',
                                   choices=(('Daily', 'Daily'),
                                            ('Weekly', 'Weekly'),
                                            ('Monthly', 'Monthly'),
                                            ))
    trained_nn_id = forms.ModelChoiceField(queryset=TrainedNeuralNetwork.objects.all())
