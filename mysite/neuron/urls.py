from django.urls import path, re_path
from .views import *

urlpatterns = [
    path('', HomePage.as_view(), name='home'),
    path('about/', about, name='about'),
    path('contact/', ContactFormView.as_view(), name='contact'),
    path('login/', LoginUser.as_view(), name='login'),
    path('logout/', logout_user, name='logout'),
    path('register/', RegisterUser.as_view(), name='register'),
    path('res_lstm/', ResultLSTM.as_view(), name='res_lstm'),
]