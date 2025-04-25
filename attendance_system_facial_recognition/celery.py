import os
from celery import Celery
from django.conf import settings
from .celery_logging import configure_logging  # Import hàm cấu hình logging

# Thiết lập biến môi trường mặc định
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')

# Cấu hình logging
configure_logging()

# Tạo instance Celery
app = Celery('attendance_system_facial_recognition')

# Sử dụng cấu hình từ settings.py
app.config_from_object('django.conf:settings', namespace='CELERY')

# Cấu hình tắt bớt log
app.conf.worker_hijack_root_logger = False  # Tắt việc chiếm quyền điều khiển root logger
app.conf.worker_log_color = False  # Tắt log màu để tránh mã ANSI trong file log
app.conf.task_store_errors_even_if_ignored = True  # Lưu lỗi ngay cả khi bị bỏ qua
app.conf.task_ignore_result = True  # Bỏ qua kết quả nếu không cần thiết
app.conf.worker_send_task_events = True  # Gửi sự kiện task cho monitor

# Tự động tìm và đăng ký các tasks từ tất cả các app Django
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}') 