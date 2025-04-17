#!/usr/bin/env python
"""
Script để kiểm tra thời gian hiện tại và xác nhận xem có đúng không
"""
import os
import sys
import datetime
import time
import django
import subprocess
import platform

# Thiết lập Django để có thể sử dụng timezone
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
django.setup()

from django.utils import timezone
from django.conf import settings

# Kiểm tra xem đây có phải là Windows không
is_windows = platform.system() == "Windows"

def check_system_time():
    """Kiểm tra thời gian hệ thống"""
    try:
        print("\nKIỂM TRA THỜI GIAN HỆ THỐNG")
        print("-" * 80)
        
        # Lấy thời gian hệ thống
        system_now = datetime.datetime.now()
        print(f"Thời gian hệ thống (datetime.now): {system_now.strftime('%d/%m/%Y %H:%M:%S')}")
        
        # Lấy thời gian UTC
        utc_now = datetime.datetime.utcnow()
        print(f"Thời gian UTC (datetime.utcnow): {utc_now.strftime('%d/%m/%Y %H:%M:%S')}")
        
        # Lấy thông tin múi giờ hệ thống
        local_tz = time.tzname
        print(f"Múi giờ hệ thống: {local_tz}")
        
        # Lấy thời gian Django
        django_now = timezone.now()
        print(f"Thời gian Django (timezone.now): {django_now.strftime('%d/%m/%Y %H:%M:%S %Z')}")
        
        # Hiển thị thông tin múi giờ Django
        print(f"Múi giờ Django: {settings.TIME_ZONE}")
        print(f"USE_TZ setting: {settings.USE_TZ}")
        
        # Kiểm tra năm
        system_year = system_now.year
        print(f"Năm hệ thống: {system_year}")
        
        # Xác nhận năm
        if system_year == 2025:
            print("\nNăm hệ thống hiện tại (2025) là chính xác, không cần điều chỉnh.")
        else:
            print(f"\nNăm hệ thống ({system_year}) không phải là năm 2025.")
            
    except Exception as e:
        print(f"Lỗi khi kiểm tra thời gian hệ thống: {str(e)}")

def check_celery_settings():
    """Kiểm tra cài đặt Celery có liên quan đến thời gian"""
    try:
        print("\nKIỂM TRA CÀI ĐẶT CELERY")
        print("-" * 80)
        
        # In ra múi giờ Celery từ settings
        celery_tz = settings.CELERY_TIMEZONE
        print(f"Múi giờ Celery (CELERY_TIMEZONE): {celery_tz}")
        
        # Kiểm tra xem múi giờ có khớp với Django không
        if celery_tz == settings.TIME_ZONE:
            print("Múi giờ Celery khớp với múi giờ Django!")
        else:
            print(f"Múi giờ Celery ({celery_tz}) khác với múi giờ Django ({settings.TIME_ZONE})!")
            
    except Exception as e:
        print(f"Lỗi khi kiểm tra cài đặt Celery: {str(e)}")

def main():
    """Hàm chính"""
    print("=" * 80)
    print("KIỂM TRA CẤU HÌNH THỜI GIAN HỆ THỐNG VÀ CELERY")
    print("=" * 80)
    
    # Kiểm tra thời gian hệ thống
    check_system_time()
    
    # Kiểm tra cài đặt Celery
    check_celery_settings()
    
    print("\nKẾT LUẬN:")
    print("-" * 80)
    print("Năm 2025 là chính xác, không cần thay đổi gì.")
    print("Nếu Celery hiển thị thời gian khác, có thể có vấn đề khác với cấu hình hoặc trạng thái của Celery.")
    print("\nGợi ý:")
    print("1. Kiểm tra xem Celery có đang sử dụng múi giờ đúng không")
    print("2. Kiểm tra log của Celery để xem có thông báo lỗi nào không")
    print("3. Thử khởi động lại Celery bằng 'python start_celery.py'")
    print("=" * 80)

if __name__ == "__main__":
    main() 