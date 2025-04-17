#!/usr/bin/env python
"""
Script để kiểm tra trạng thái của Celery (worker và beat)
"""
import os
import sys
import time
import datetime
import django
import subprocess
import json
import pytz
from celery import current_app

# Thiết lập Django để có thể truy vấn DB
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
django.setup()

# Import sau khi setup Django
from django.utils import timezone
from celery import Celery
from celery.app import app_or_default

# Màu sắc cho terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_line(char='-', length=80):
    """In một dòng ngăn cách"""
    print(char * length)

def get_local_time():
    """Lấy thời gian địa phương ở múi giờ Việt Nam"""
    utc_now = datetime.datetime.now(pytz.UTC)
    vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    local_now = utc_now.astimezone(vn_tz)
    return local_now

def check_celery_worker_status():
    """Kiểm tra trạng thái của Celery worker"""
    try:
        print(f"\n{Colors.HEADER}KIỂM TRA TRẠNG THÁI CELERY WORKER{Colors.ENDC}")
        print_line()
        
        # Sử dụng Celery inspector để lấy thông tin worker
        i = current_app.control.inspect()
        
        # Kiểm tra các worker đang hoạt động
        active_workers = i.ping() or {}
        if not active_workers:
            print(f"{Colors.RED}Không có Celery worker nào đang chạy!{Colors.ENDC}")
            return False
        
        print(f"{Colors.GREEN}Celery worker đang hoạt động:{Colors.ENDC}")
        for worker_name, response in active_workers.items():
            print(f"- {worker_name}: {response}")
        
        # Kiểm tra các task đang chạy
        active_tasks = i.active() or {}
        
        print(f"\n{Colors.HEADER}TASK ĐANG CHẠY{Colors.ENDC}")
        print_line()
        
        if not any(active_tasks.values()):
            print(f"{Colors.YELLOW}Không có task nào đang chạy.{Colors.ENDC}")
        else:
            for worker_name, tasks in active_tasks.items():
                if tasks:
                    print(f"\nWorker: {worker_name}")
                    print(f"{'Task ID':<36} {'Task':<30} {'Thời gian chạy':<20}")
                    print_line('-', 90)
                    
                    for task in tasks:
                        task_id = task.get('id', 'unknown')
                        task_name = task.get('name', 'unknown')
                        
                        # Tính thời gian chạy
                        start_time = task.get('time_start', 0)
                        if start_time:
                            # Chuyển đổi từ unix timestamp
                            start_dt = datetime.datetime.fromtimestamp(start_time)
                            now = datetime.datetime.now()
                            runtime = now - start_dt
                            runtime_str = str(runtime).split('.')[0]  # Bỏ phần microseconds
                        else:
                            runtime_str = "unknown"
                        
                        print(f"{task_id:<36} {task_name:<30} {runtime_str:<20}")
                        
                        # In các tham số của task
                        args = task.get('args', '[]')
                        kwargs = task.get('kwargs', '{}')
                        print(f"   Args: {args}")
                        print(f"   Kwargs: {kwargs}")
                        print()
                else:
                    print(f"Worker {worker_name}: Không có task nào đang chạy")
        
        # Kiểm tra task đã được lên lịch (scheduled)
        scheduled_tasks = i.scheduled() or {}
        
        print(f"\n{Colors.HEADER}TASK ĐÃ LÊN LỊCH{Colors.ENDC}")
        print_line()
        
        if not any(scheduled_tasks.values()):
            print(f"{Colors.YELLOW}Không có task nào đã lên lịch.{Colors.ENDC}")
        else:
            for worker_name, tasks in scheduled_tasks.items():
                if tasks:
                    print(f"\nWorker: {worker_name}")
                    print(f"{'Task ID':<36} {'Task':<30} {'Lịch chạy tiếp':<20}")
                    print_line('-', 90)
                    
                    for task in tasks:
                        task_id = task.get('request', {}).get('id', 'unknown')
                        task_name = task.get('request', {}).get('name', 'unknown')
                        
                        # Thời gian chạy tiếp
                        eta = task.get('eta', None)
                        if eta:
                            eta_str = eta
                        else:
                            eta_str = "unknown"
                        
                        print(f"{task_id:<36} {task_name:<30} {eta_str:<20}")
                else:
                    print(f"Worker {worker_name}: Không có task nào đã lên lịch")
        
        # Kiểm tra task định kỳ (registered)
        registered_tasks = i.registered() or {}
        
        print(f"\n{Colors.HEADER}TASK ĐÃ ĐĂNG KÝ{Colors.ENDC}")
        print_line()
        
        if not any(registered_tasks.values()):
            print(f"{Colors.YELLOW}Không có task nào đã đăng ký.{Colors.ENDC}")
        else:
            print(f"Danh sách task đã đăng ký theo worker:")
            for worker_name, tasks in registered_tasks.items():
                print(f"\nWorker: {worker_name}")
                for task_name in tasks:
                    print(f"- {task_name}")
        
        return True
        
    except ConnectionRefusedError:
        print(f"{Colors.RED}Không thể kết nối đến Celery broker. Celery có thể không đang chạy.{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Lỗi khi kiểm tra Celery worker: {str(e)}{Colors.ENDC}")
        return False

def check_celery_beat_status():
    """Kiểm tra trạng thái của Celery beat"""
    try:
        print(f"\n{Colors.HEADER}KIỂM TRA TRẠNG THÁI CELERY BEAT{Colors.ENDC}")
        print_line()
        
        # Lấy lịch trình từ Celery Beat
        scheduler = current_app.conf.CELERYBEAT_SCHEDULE or current_app.conf.beat_schedule
        
        if not scheduler:
            print(f"{Colors.YELLOW}Không tìm thấy lịch trình Celery Beat nào.{Colors.ENDC}")
            return
        
        print(f"Lịch trình Celery Beat:")
        print(f"{'Task':<40} {'Lịch trình':<40}")
        print_line()
        
        for task_name, task_config in scheduler.items():
            schedule_str = str(task_config.get('schedule', 'unknown'))
            print(f"{task_name:<40} {schedule_str:<40}")
        
        # Kiểm tra xem Celery Beat có đang chạy không trên Windows bằng cách sử dụng wmic thay vì ps
        is_windows = sys.platform.startswith('win')
        try:
            if is_windows:
                # Sử dụng tasklist trên Windows
                cmd = "tasklist | findstr celery"
                output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
                if "celery" in output.lower():
                    print(f"\n{Colors.GREEN}Celery Beat có thể đang chạy.{Colors.ENDC}")
                    print(f"Các tiến trình Celery đang chạy:")
                    print(output)
                else:
                    print(f"\n{Colors.RED}Không tìm thấy tiến trình Celery Beat đang chạy.{Colors.ENDC}")
            else:
                # Sử dụng ps trên Linux/Mac
                ps_output = subprocess.check_output(["ps", "aux"], universal_newlines=True)
                if "celery -A attendance_system_facial_recognition beat" in ps_output:
                    print(f"\n{Colors.GREEN}Celery Beat đang chạy.{Colors.ENDC}")
                else:
                    print(f"\n{Colors.RED}Không tìm thấy tiến trình Celery Beat đang chạy.{Colors.ENDC}")
        except subprocess.SubprocessError as e:
            print(f"\n{Colors.YELLOW}Không thể kiểm tra tiến trình Celery Beat: {str(e)}{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}Lỗi khi kiểm tra Celery Beat: {str(e)}{Colors.ENDC}")

def main():
    """Hàm chính để chạy script"""
    # Lấy thời gian hiện tại theo múi giờ Việt Nam
    local_now = get_local_time()
    system_now = datetime.datetime.now()
    django_now = timezone.now()
    django_local = timezone.localtime(django_now)
    
    print(f"{Colors.BOLD}{Colors.HEADER}KIỂM TRA TRẠNG THÁI CELERY{Colors.ENDC}")
    print(f"Thời gian hệ thống:     {system_now.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Thời gian Việt Nam:     {local_now.strftime('%d/%m/%Y %H:%M:%S')} (GMT+7)")
    print(f"Thời gian Django (UTC): {django_now.strftime('%d/%m/%Y %H:%M:%S')} UTC")
    print(f"Thời gian Django (VN):  {django_local.strftime('%d/%m/%Y %H:%M:%S')} ({django_local.tzinfo})")
    print_line('=')
    
    # Kiểm tra worker status
    worker_running = check_celery_worker_status()
    
    # Kiểm tra beat status nếu worker đang chạy
    if worker_running:
        check_celery_beat_status()
    
    print_line('=')
    print(f"{Colors.BOLD}Kết thúc kiểm tra!{Colors.ENDC}")
    print(f"Nếu thấy thời gian không đúng, hãy đảm bảo TIME_ZONE trong settings.py là 'Asia/Ho_Chi_Minh'")
    print(f"và CELERY_TIMEZONE cũng được đặt thành 'Asia/Ho_Chi_Minh'")

if __name__ == "__main__":
    main() 