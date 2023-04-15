from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin

# from .forms import CustomUserCreationForm, CustomUserChangeForm
from .models import *


class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'researcher', 'first_name', 'last_name', 'is_staff')
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal info', {'fields': ('first_name', 'researcher', 'last_name', 'email')}),
        ('Permissions', {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions'),
        }),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )


class NeuralNetworkAdmin(admin.ModelAdmin):
    prepopulated_fields = {'slug': ('name',)}


class ListСompaniesAdmin(admin.ModelAdmin):
    list_display = ('name', 'ticker')


class TrainedNeuralNetworkUserAdmin(admin.ModelAdmin):
    list_display = (
        'creator', 'time_step', 'loss', 'optimizer', 'epochs', 'batch_size', 'file_trained_nn', 'time_create',
        'neural_network_architecture')


class TrainedNeuralNetworkAdmin(admin.ModelAdmin):
    list_display = (
        'time_step','file_trained_nn'
    )


admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(NeuralNetwork, NeuralNetworkAdmin)
admin.site.register(ListСompanies, ListСompaniesAdmin)
admin.site.register(TrainedNeuralNetworkUser, TrainedNeuralNetworkUserAdmin)
admin.site.register(TrainedNeuralNetwork, TrainedNeuralNetworkAdmin)
