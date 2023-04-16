import datetime
import json
import os

from mysite.settings import BASE_DIR
from .neural_network import predict, neural_network_training

from django.contrib.auth import logout, login
from django.contrib.auth.views import LoginView
from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.views.generic import ListView, DetailView, CreateView, FormView, View

from .forms import *
from .models import *
from .neural_network.retraining import retraining
from .utils import *


class HomePage(DataMixin, ListView):
    model = NeuralNetwork
    template_name = 'neuron/index.html'
    context_object_name = 'network'  # В эту переменную помещаются данные из указанной модели

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        c_def = self.get_user_context(title='Главная страница')
        context['count_trained_nn'] = TrainedNeuralNetworkUser.objects.count()
        context['count_user'] = CustomUser.objects.count()

        return context | c_def

    def get_queryset(self):
        return NeuralNetwork.objects.all()


class RegisterUser(DataMixin, CreateView):
    form_class = CustomUserCreationForm
    template_name = 'neuron/register.html'
    success_url = reverse_lazy('login')  # перенаправление, по умолчанию при реализации get_absolute_url в модели,

    # перенаправление происходит на созданную страницу

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

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        c_def = self.get_user_context(title='Авторизация')
        return context | c_def

    def get_success_url(self):
        return reverse_lazy('home')


def logout_user(request):
    logout(request)
    return redirect('login')


class About(DataMixin, ListView):
    model = NeuralNetwork
    template_name = 'neuron/about.html'

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        c_def = self.get_user_context(title='О сайте')
        return context | c_def


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


class Predict(DataMixin, View):
    model = ListСompanies
    context_object_name = 'companies'  # В эту переменную помещаются данные из указанной модели

    def get_context_data(self, *, request, object_list=None, **kwargs):
        context = {}
        c_def = self.get_user_context(title='Прогноз')
        """Достаём данные из сессии"""
        selected_company_ticker = request.session.get('selected_company_ticker')
        predict_daily = request.session.get('predict_daily')

        # selected_time_frame = request.session.get('selected_time_frame')
        selected_trained_nn_path = request.session.get('selected_trained_nn_path')
        selected_trained_nn_time_step = request.session.get('selected_trained_nn_time_step')

        # print(f'selected_trained_nn_path: {selected_trained_nn_path}')
        """Обращаемся к апи"""
        data_for_graphic = data_API(selected_company_ticker)
        """Формируем набор данных для нейронки"""
        date = list(reversed(data_for_graphic.keys()))
        value = list(reversed(data_for_graphic.values()))
        """Запись в сессию"""
        request.session['date'] = date
        request.session['value'] = value
        """Формируем контекс для отправки в js"""
        context['data_for_graphic_with_predict'] = json.dumps(
            predict.predict(value, date, predict_daily, selected_trained_nn_path,
                            selected_trained_nn_time_step))

        context['selected_company'] = json.dumps([request.session.get('selected_company_name'),
                                                  json.dumps(selected_company_ticker)])
        context['data_for_graphic'] = json.dumps(
            [

                {
                    'data': data,
                    'value': value,
                }
                for data, value in reversed(data_for_graphic.items())
            ]
        )
        return context | c_def

    def get(self, request):
        return render(request, 'neuron/predict.html', context=self.get_context_data(request=request))

    def get_queryset(self):
        return ListСompanies.objects.all()


class ShowNetwork(DataMixin, DetailView):
    model = NeuralNetwork
    template_name = 'neuron/shownetwork.html'
    slug_url_kwarg = 'network_slug'
    context_object_name = 'network'

    # allow_empty = False  # возврат ошибки 404 при несоответсвии идентификатора url и бд

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        c_def = self.get_user_context(title=context['network'])
        return context | c_def


class ChoiceForecastParam(DataMixin, View):

    def get_context_data(self, *, object_list=None, **kwargs):
        context = {}
        c_def = self.get_user_context(title='Выбор параметров прогнозирования')
        context['TrainedNeuralNetwork'] = TrainedNeuralNetworkUser.objects.all()
        return context | c_def

    def get(self, request):
        form = ChoiceParam()
        return render(request, 'neuron/choice_forecast_param.html', context=self.get_context_data() | {'form': form})

    def post(self, request):
        bound_form = ChoiceParam(request.POST)
        if bound_form.is_valid():
            company = bound_form.cleaned_data['company'].name
            request.session['selected_company_name'] = company
            request.session['selected_company_ticker'] = bound_form.cleaned_data['company'].ticker
            # request.session['selected_time_frame'] = bound_form.cleaned_data['time_frame']
            request.session['predict_daily'] = bound_form.cleaned_data['predict_daily']

            try:
                nn = TrainedNeuralNetwork.objects.get(company=company)
            except Exception as e:
                print('ку ' + str(e))
                return

            request.session['selected_trained_nn_path'] = str(nn.file_trained_nn)
            request.session['selected_trained_nn_time_step'] = nn.time_step

            return redirect('predict')

        return render(request, 'neuron/choice_forecast_param.html',
                      context=self.get_context_data() | {'form': bound_form})


class ChoiceForecastParamResearcher(DataMixin, View):

    def get_context_data(self, *, object_list=None, **kwargs):
        context = {}
        c_def = self.get_user_context(title='Выбор параметров прогнозирования')
        context['TrainedNeuralNetworkUser'] = TrainedNeuralNetworkUser.objects.all()
        return context | c_def

    def get(self, request):
        form = ChoiceParamResearcher()
        return render(request, 'neuron/choice_forecast_param_researcher.html',
                      context=self.get_context_data() | {'form': form})

    def post(self, request):
        bound_form = ChoiceParamResearcher(request.POST)
        if bound_form.is_valid():
            request.session['selected_company_name'] = bound_form.cleaned_data['company'].name
            request.session['selected_company_ticker'] = bound_form.cleaned_data['company'].ticker
            # request.session['selected_time_frame'] = bound_form.cleaned_data['time_frame']
            request.session['predict_daily'] = bound_form.cleaned_data['predict_daily']

            print(bound_form.cleaned_data['trained_nn_id'].__dict__)

            request.session['selected_trained_nn_path'] = str(bound_form.cleaned_data['trained_nn_id'].file_trained_nn)
            request.session['selected_trained_nn_time_step'] = bound_form.cleaned_data['trained_nn_id'].time_step

            return redirect('predict')

        return render(request, 'neuron/choice_forecast_param.html',
                      context=self.get_context_data() | {'form': bound_form})


class PredictPastData(DataMixin, View):
    def get_context_data(self, *, request, object_list=None, **kwargs):
        context = {}
        c_def = self.get_user_context(title='Прогноз прошлых значений')
        """Достаём данные из сессии"""
        date = request.session.get('date')
        value = request.session.get('value')
        predict_daily = request.session.get('predict_daily')
        selected_trained_nn_path = request.session.get('selected_trained_nn_path')
        selected_trained_nn_time_step = request.session.get('selected_trained_nn_time_step')
        selected_company_name = request.session.get('selected_company_name')
        selected_company_ticker = request.session.get('selected_company_ticker')

        """Формируем контекс для отправки в js"""

        context['data_for_graphic_with_predict'] = json.dumps(
            predict.predict_past_data(value, date, predict_daily, selected_trained_nn_path,
                                      selected_trained_nn_time_step))

        context['selected_company'] = json.dumps([selected_company_name, selected_company_ticker])

        return context | c_def

    def get(self, request):
        return render(request, 'neuron/predict_past_data.html', context=self.get_context_data(request=request))


class Training(DataMixin, View):
    def get_context_data(self, *, object_list=None, **kwargs):
        c_def = self.get_user_context(title='Обучение нейронной сети')
        return c_def

    def get(self, request):
        form = TrainingForm()
        return render(request, 'neuron/training.html', context=self.get_context_data() | {'form': form})

    def post(self, request):
        bound_form = TrainingForm(request.POST)

        if bound_form.is_valid():
            form_data = bound_form.cleaned_data

            print(str(form_data['neural_network_architecture']))

            selected_company_ticker = form_data['company'].ticker
            """Обращаемся к апи"""
            data_for_graphic = data_API(selected_company_ticker)
            """Формируем набор данных для нейронки"""
            date = list(reversed(data_for_graphic.keys()))
            value = list(reversed(data_for_graphic.values()))

            """Обучаем модель, возвращаем путь к файлу модели"""
            path, accuracy, loss = neural_network_training.training(value, date, form_data)

            """Возвращается модель с заполнеными полями с формы"""
            model = bound_form.save(commit=False)
            """Дополняем модель"""
            model.creator = request.user
            model.file_trained_nn = path
            model.save()

            #####################
            request.session['accuracy'] = accuracy
            request.session['loss'] = loss

            return redirect('training_metrics')

        return render(request, 'neuron/training.html',
                      context=self.get_context_data() | {'form': bound_form})


class TrainingMetrics(DataMixin, View):
    def get_context_data(self, *, request, object_list=None, **kwargs):
        context = {}
        c_def = self.get_user_context(title='Метрики')
        """Достаём данные из сессии"""
        accuracy = request.session.get('accuracy')
        loss = request.session.get('loss')

        """Формируем контекс для отправки в js"""
        context['accuracy'] = json.dumps(
            [

                {
                    'epoch': int(epoch),
                    'value': value,
                }
                for epoch, value in accuracy.items()
            ]
        )

        context['loss'] = json.dumps(
            [

                {
                    'epoch': int(epoch),
                    'value': value,
                }
                for epoch, value in loss.items()
            ]
        )

        return context | c_def

    def get(self, request):
        return render(request, 'neuron/training_metrics.html', context=self.get_context_data(request=request))
