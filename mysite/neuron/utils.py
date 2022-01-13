from django.db.models import Count
from django.core.cache import cache

from .models import *

menu = [
    {'title': "О сайте", 'url_name': 'about'},
    {'title': "Обратная связь", 'url_name': 'contact'},
]


class DataMixin:

    def get_user_context(self, **kwargs):
        context = kwargs
        # cats = cache.get('cats')  # кэширование
        # if not cats:
        #     cats = Category.objects.annotate(Count('women'))
        #     cache.set('cats', cats, 60)

        user_menu = menu.copy()
        context['menu'] = user_menu
        return context