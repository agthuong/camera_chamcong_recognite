import os
from celery import Celery
from django.conf import settings

# Thiết lập biến môi trường mặc định
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')

# Tạo instance Celery
app = Celery('attendance_system_facial_recognition')

# Sử dụng cấu hình từ settings.py
app.config_from_object('django.conf:settings', namespace='CELERY')

# Tự động tìm và đăng ký các tasks từ tất cả các app Django
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}') 