# Import đây là cần thiết để đảm bảo Celery app được khởi tạo
from .celery import app as celery_app

__all__ = ('celery_app',)

