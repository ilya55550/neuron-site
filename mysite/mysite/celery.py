import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

app = Celery('mysite')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()


app.conf.beat_schedule = {
    'retraining_models': {
        'task': 'neuron.tasks.retraining_models',
        'schedule': crontab(minute=0, hour=0)
    },
}
