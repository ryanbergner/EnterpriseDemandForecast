"""
ASGI config for demand_forecast project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "demand_forecast.settings")

application = get_asgi_application()
