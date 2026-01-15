"""
WSGI config for demand_forecast project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "demand_forecast.settings")

application = get_wsgi_application()
