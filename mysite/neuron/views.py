from django.contrib.auth import logout, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.views import LoginView
from django.http import HttpResponse, HttpResponseNotFound, Http404
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy
from django.views.generic import ListView, DetailView, CreateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin

from .forms import *
from .models import *
from .utils import *


class HomePage(DataMixin, ListView):

    template_name = 'neuron/index.html'
    # context_object_name = 'posts'  # В эту переменную помещаются данные из указанной модели

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        c_def = self.get_user_context(title='Главная страница')
        return context | c_def

    def get_queryset(self):
        pass


class RegisterUser(DataMixin, CreateView):
    form_class = RegisterUserForm
    template_name = 'neuron/register.html'
    success_url = reverse_lazy('login')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        c_def = self.get_user_context(title='Регистрация')
        return context | c_def

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect('home')


class LoginUser(DataMixin, LoginView):
    form_class = LoginUserForm
    template_name = 'neuron/login.html'

    def get_user_context(self, **kwargs):
        context = super().get_user_context(**kwargs)
        c_def = self.get_user_context(title='Авторизация')
        return context | c_def

    def get_success_url(self):
        return reverse_lazy('home')


def logout_user(request):
    logout(request)
    return redirect('login')


def about(request):
    context = {
        'menu': menu,
    }
    return render(request, 'neuron/about.html', context=context)


class ContactFormView(DataMixin, FormView):
    form_class = ContactForm
    template_name = 'neuron/contact.html'
    success_url = reverse_lazy('home')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        c_def = self.get_user_context(title='Обратная связь')
        return context | c_def

    def form_valid(self, form):
        print(form.cleaned_data)
        return redirect('home')


class ResultLSTM(DataMixin, ListView):

    template_name = 'neuron/res_lstm.html'
    # context_object_name = 'posts'  # В эту переменную помещаются данные из указанной модели

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        c_def = self.get_user_context(title='LSTM')
        return context | c_def

    def get_queryset(self):
        pass
