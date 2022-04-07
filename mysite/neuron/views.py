import json
# import .neural_network.LSTM2 as lstm
from .neural_network import LSTM2
from pyexpat.errors import messages

from django.contrib.auth import logout, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.views import LoginView
from django.db import transaction
from django.http import HttpResponse, HttpResponseNotFound, Http404
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy
from django.views.generic import ListView, DetailView, CreateView, FormView, View
from django.contrib.auth.mixins import LoginRequiredMixin

from .forms import *
from .models import *
from .utils import *


class HomePage(DataMixin, ListView):
    model = NeuralNetwork
    template_name = 'neuron/index.html'
    context_object_name = 'network'  # В эту переменную помещаются данные из указанной модели

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        c_def = self.get_user_context(title='Главная страница')
        context['count_trained_nn'] = TrainedNeuralNetwork.objects.count()
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


# @login_required
# @transaction.atomic
# def update_profile(request):
#     if request.method == 'POST':
#         user_form = RegisterUserForm(request.POST, instance=request.user)
#         profile_form = ProfileUserForm(request.POST, instance=request.user.profile)
#         if user_form.is_valid() and profile_form.is_valid():
#             user_form.save()
#             profile_form.save()
#             messages.success(request, _('Ваш профиль был успешно обновлен!'))
#             return redirect('settings:profile')
#         else:
#             messages.error(request, _('Пожалуйста, исправьте ошибки.'))
#     else:
#         user_form = RegisterUserForm(instance=request.user)
#         profile_form = ProfileUserForm(instance=request.user.profile)
#     return render(request, 'neuron/register.html', {
#         'user_form': user_form,
#         'profile_form': profile_form
#     })


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
        c_def = self.get_user_context(title='Predict')
        """Достаём данные из сессии"""
        selected_company_ticker = request.session.get('selected_company_ticker')
        selected_time_frame = request.session.get('selected_time_frame')
        # selected_trained_nn_id = request.session.get['trained_nn_id']
        """Обращаемся к апи"""
        data_for_graphic = data_API(selected_time_frame, selected_company_ticker)
        """Формируем набор данных для нейронки"""
        date = list(reversed(data_for_graphic.keys()))
        value = list(reversed(data_for_graphic.values()))
        """Формируем контекс для отправки в js"""

        # res_date, res_volume, predict_date, predict_value = LSTM2.predict(value, date)
        # context['res_date'] = json.dumps(res_date)
        # context['res_volume'] = json.dumps(res_volume)
        # context['predict_date'] = json.dumps(predict_date)
        # context['predict_value'] = json.dumps(predict_value)

        context['data_for_graphic_with_predict'] = json.dumps(LSTM2.predict(value, date))
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
    # form_class = ChoiceParam
    # template_name = 'neuron/choice_forecast_param.html'

    def get_context_data(self, *, object_list=None, **kwargs):
        # context = super().get_context_data(**kwargs)
        context = {}
        c_def = self.get_user_context(title='Прогнозирование')
        context['TrainedNeuralNetwork'] = TrainedNeuralNetwork.objects.all()
        return context | c_def

    def get(self, request):
        form = ChoiceParam()
        return render(request, 'neuron/choice_forecast_param.html', context=self.get_context_data() | {'form': form})

    def post(self, request):
        bound_form = ChoiceParam(request.POST)
        if bound_form.is_valid():
            request.session['selected_company_name'] = bound_form.cleaned_data['company'].name
            request.session['selected_company_ticker'] = bound_form.cleaned_data['company'].ticker
            request.session['selected_time_frame'] = bound_form.cleaned_data['time_frame']
            request.session['selected_trained_nn_id'] = bound_form.cleaned_data['trained_nn_id'].id
            return redirect('predict')

        return render(request, 'neuron/choice_forecast_param.html',
                      context=self.get_context_data() | {'form': bound_form})
