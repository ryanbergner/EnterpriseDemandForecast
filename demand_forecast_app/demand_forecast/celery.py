"""
Celery application configuration.
"""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "demand_forecast.settings")

app = Celery("demand_forecast")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
