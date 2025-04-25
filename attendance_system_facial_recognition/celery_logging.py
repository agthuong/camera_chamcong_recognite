import logging
import os
from logging.handlers import RotatingFileHandler

# Thiết lập thư mục log
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Cấu hình logging cho Celery
def configure_logging():
    # Thiết lập định dạng log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Tạo handler file log cho worker
    worker_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, 'celery_worker.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    worker_handler.setFormatter(formatter)
    worker_handler.setLevel(logging.WARNING)  # Chỉ log warning trở lên
    
    # Tạo handler file log cho beat
    beat_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, 'celery_beat.log'),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    beat_handler.setFormatter(formatter)
    beat_handler.setLevel(logging.WARNING)  # Chỉ log warning trở lên
    
    # Tạo handler file log cho task
    task_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, 'celery_tasks.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    task_handler.setFormatter(formatter)
    task_handler.setLevel(logging.INFO)  # Log cả info cho tasks
    
    # Cấu hình logger
    celery_logger = logging.getLogger('celery')
    celery_logger.setLevel(logging.WARNING)
    celery_logger.addHandler(worker_handler)
    
    celery_beat_logger = logging.getLogger('celery.beat')
    celery_beat_logger.setLevel(logging.WARNING)
    celery_beat_logger.addHandler(beat_handler)
    
    celery_task_logger = logging.getLogger('celery.task')
    celery_task_logger.setLevel(logging.INFO)
    celery_task_logger.addHandler(task_handler)
    
    # Tắt logger propagation để tránh log trùng lặp
    celery_logger.propagate = False
    celery_beat_logger.propagate = False
    celery_task_logger.propagate = False
    
    # Cấu hình console handler với mức WARNING
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(formatter)
    
    # Thêm console handler vào root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console)
    
    # Thiết lập mức log cho các thư viện khác
    logging.getLogger('kombu').setLevel(logging.WARNING)
    logging.getLogger('amqp').setLevel(logging.WARNING)
    logging.getLogger('billiard').setLevel(logging.WARNING) 