import requests
from decouple import config
from django.db.models import Count
from django.core.cache import cache

from .models import *


def data_API(function, symbol):
    """https://www.alphavantage.co/"""
    function_for_url = 'TIME_SERIES_' + str(function).upper()
    url = f"https://www.alphavantage.co/query?function={function_for_url}&outputsize=full&symbol={symbol}" \
          f"&apikey={config('API_KEY')}"
    r = requests.get(url)
    data = r.json()
    function_for_dict = 'Time Series ' + f'({function})'
    data = {k: v['4. close'] for k, v in data[function_for_dict].items()}
    return data


menu = [
    {'title': "Прогнозирование", 'url_name': 'choice_forecast_param'},
    {'title': "О сайте", 'url_name': 'about'},
    {'title': "Обратная связь", 'url_name': 'contact'},
]


class DataMixin:

    def get_user_context(self, **kwargs):
        context = kwargs
        network = cache.get('network')  # кэширование
        if not network:
            # first_name_networks = NeuralNetwork.objects.get(pk=1)
            # other_name_networks = NeuralNetwork.objects.all()[1:]
            networks = NeuralNetwork.objects.all()
            cache.set('network', network, 600)

        user_menu = menu.copy()
        context['menu'] = user_menu
        # context['first_name_networks'] = first_name_networks
        # context['other_name_networks'] = other_name_networks
        context['networks'] = networks

        return context
