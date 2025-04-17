#!/usr/bin/env python
"""
Script để kiểm tra thời gian từ nhiều nguồn khác nhau
"""
import os
import sys
import time
import datetime
import django
import subprocess
import platform

# Thiết lập Django để có thể sử dụng timezone
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
django.setup()

# Import sau khi setup Django
from django.utils import timezone
from django.conf import settings

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

def check_system_time():
    """Kiểm tra thời gian hệ thống"""
    try:
        print(f"\n{Colors.HEADER}KIỂM TRA THỜI GIAN HỆ THỐNG{Colors.ENDC}")
        print_line()
        
        # Lấy thời gian hệ thống
        system_now = datetime.datetime.now()
        print(f"Thời gian hệ thống (datetime.now): {system_now.strftime('%d/%m/%Y %H:%M:%S')}")
        
        # Lấy thời gian UTC
        utc_now = datetime.datetime.utcnow()
        print(f"Thời gian UTC (datetime.utcnow): {utc_now.strftime('%d/%m/%Y %H:%M:%S')}")
        
        # Lấy thông tin múi giờ hệ thống
        local_tz = time.tzname
        print(f"Múi giờ hệ thống: {local_tz}")
        
        # Lấy timestamp hiện tại
        timestamp = time.time()
        print(f"Timestamp hiện tại: {timestamp}")
        print(f"Chuyển timestamp thành datetime: {datetime.datetime.fromtimestamp(timestamp).strftime('%d/%m/%Y %H:%M:%S')}")
        
    except Exception as e:
        print(f"{Colors.RED}Lỗi khi kiểm tra thời gian hệ thống: {str(e)}{Colors.ENDC}")

def check_django_time():
    """Kiểm tra thời gian từ Django timezone"""
    try:
        print(f"\n{Colors.HEADER}KIỂM TRA THỜI GIAN DJANGO{Colors.ENDC}")
        print_line()
        
        # Lấy thời gian hiện tại từ Django timezone
        django_now = timezone.now()
        print(f"Thời gian Django (timezone.now): {django_now.strftime('%d/%m/%Y %H:%M:%S %Z')}")
        
        # Hiển thị thông tin múi giờ Django
        print(f"Múi giờ Django: {settings.TIME_ZONE}")
        print(f"USE_TZ setting: {settings.USE_TZ}")
        
        # Lấy thời gian local từ Django
        django_localtime = timezone.localtime(django_now)
        print(f"Thời gian Django local (timezone.localtime): {django_localtime.strftime('%d/%m/%Y %H:%M:%S %Z')}")
        
    except Exception as e:
        print(f"{Colors.RED}Lỗi khi kiểm tra thời gian Django: {str(e)}{Colors.ENDC}")

def check_ntp_time():
    """Kiểm tra thời gian từ máy chủ NTP"""
    try:
        print(f"\n{Colors.HEADER}KIỂM TRA THỜI GIAN TỪ MÁY CHỦ NTP{Colors.ENDC}")
        print_line()
        
        # Kiểm tra OS và chọn lệnh phù hợp
        os_type = platform.system()
        
        if os_type == "Windows":
            try:
                # Sử dụng lệnh w32tm trên Windows
                output = subprocess.check_output("w32tm /stripchart /computer:time.windows.com /samples:1 /dataonly", shell=True)
                print(f"Kết quả từ máy chủ NTP Windows: {output.decode('utf-8').strip()}")
            except subprocess.SubprocessError as e:
                print(f"{Colors.YELLOW}Không thể kiểm tra thời gian NTP từ Windows: {str(e)}{Colors.ENDC}")
                
            # Thử phương pháp khác với Python
            try:
                from ntplib import NTPClient
                client = NTPClient()
                response = client.request('pool.ntp.org')
                ntp_time = datetime.datetime.fromtimestamp(response.tx_time)
                print(f"Thời gian NTP từ pool.ntp.org: {ntp_time.strftime('%d/%m/%Y %H:%M:%S')}")
                print(f"Độ lệch so với hệ thống: {response.offset:.6f} giây")
            except ImportError:
                print(f"{Colors.YELLOW}Không thể sử dụng ntplib (thư viện không được cài đặt){Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.YELLOW}Lỗi khi lấy thời gian NTP sử dụng ntplib: {str(e)}{Colors.ENDC}")
        else:
            try:
                # Sử dụng lệnh ntpdate trên Linux/Mac
                output = subprocess.check_output("ntpdate -q pool.ntp.org", shell=True)
                print(f"Kết quả từ máy chủ NTP: {output.decode('utf-8').strip()}")
            except subprocess.SubprocessError:
                print(f"{Colors.YELLOW}Không thể kiểm tra thời gian NTP từ terminal{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}Lỗi khi kiểm tra thời gian NTP: {str(e)}{Colors.ENDC}")

def main():
    """Hàm chính để chạy script"""
    print(f"{Colors.BOLD}{Colors.HEADER}KIỂM TRA THỜI GIAN HỆ THỐNG{Colors.ENDC}")
    print(f"Thực hiện kiểm tra: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print_line('=')
    
    # Kiểm tra thời gian hệ thống
    check_system_time()
    
    # Kiểm tra thời gian Django
    check_django_time()
    
    # Kiểm tra thời gian NTP
    check_ntp_time()
    
    print_line('=')
    print(f"{Colors.BOLD}Kết thúc kiểm tra!{Colors.ENDC}")

if __name__ == "__main__":
    main() 