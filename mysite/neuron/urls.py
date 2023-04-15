from django.urls import path, re_path
from .views import *

urlpatterns = [
    path('', HomePage.as_view(), name='home'),
    path('about/', About.as_view(), name='about'),
    path('contact/', ContactFormView.as_view(), name='contact'),
    path('login/', LoginUser.as_view(), name='login'),
    path('logout/', logout_user, name='logout'),
    path('register/', RegisterUser.as_view(), name='register'),
    path('network/<slug:network_slug>/', ShowNetwork.as_view(), name='show_network'),
    path('choice_forecast_param_researcher/', ChoiceForecastParamResearcher.as_view(),
         name='choice_forecast_param_researcher'),

    path('choice_forecast_param/', ChoiceForecastParam.as_view(), name='choice_forecast_param'),
    path('choice_forecast_param/predict/', Predict.as_view(), name='predict'),
    path('choice_forecast_param/predict/predict_past', PredictPastData.as_view(), name='predict_past_data'),
    path('training/', Training.as_view(), name='training'),
]