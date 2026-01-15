"""Django app configuration for demand_forecast."""

from .celery import app as celery_app

__all__ = ("celery_app",)
