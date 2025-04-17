"""
ASGI config for attendance_system_facial_recognition project.
"""

import os
import django
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
django.setup()

# Thêm import này sau khi Django đã setup
from recognition import routing

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            routing.websocket_urlpatterns
        )
    ),
}) 