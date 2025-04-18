"""
Script khởi động cả Celery worker và Celery beat trong một tiến trình
Sử dụng cho mục đích kiểm thử nhanh
"""
import os
import subprocess
import sys
import threading
import time
import django

# Thiết lập Django để có thể truy vấn DB
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
django.setup()

# Fix múi giờ cho Celery logs - thêm dòng này để sửa lỗi múi giờ
# Import sau khi setup Django
from django.utils import timezone
from recognition.models import ContinuousAttendanceSchedule

def print_schedule_info():
    """In thông tin về tất cả các lịch trình trong database"""
    try:
        # Lấy thời gian UTC và thời gian địa phương
        now_utc = timezone.now()
        now_local = timezone.localtime(now_utc)
        today = now_local.date()
        current_time_local = now_local.time()
        day_of_week = str(now_local.isoweekday())  # 1-7 (1 là thứ Hai)
        
        print("\n" + "="*80)
        print(f"THÔNG TIN LỊCH TRÌNH CHẤM CÔNG")
        print(f"Thời gian UTC: {now_utc.strftime('%d/%m/%Y %H:%M:%S')} UTC")
        print(f"Thời gian VN:  {now_local.strftime('%d/%m/%Y %H:%M:%S')} (GMT+7)")
        print(f"Ngày trong tuần: {day_of_week} (1=Thứ Hai, 7=Chủ Nhật)")
        print("="*80)
        
        # Lấy tất cả lịch trình
        all_schedules = ContinuousAttendanceSchedule.objects.all().order_by('start_time')
        active_schedules = 0
        
        print(f"Tổng số lịch trình: {all_schedules.count()}")
        print("\n{:<5} {:<20} {:<10} {:<15} {:<15} {:<20} {:<20}".format(
            "ID", "Tên", "Loại", "Bắt đầu", "Kết thúc", "Trạng thái", "Thời gian kích hoạt"
        ))
        print("-"*100)
        
        for schedule in all_schedules:
            # Kiểm tra xem lịch trình có hoạt động vào ngày này không
            days_active = schedule.active_days.split(',')
            is_active_today = day_of_week in days_active
            
            # Kiểm tra xem có đang trong khung giờ chạy không
            # Quan trọng: sử dụng thời gian địa phương để so sánh
            in_time_range = schedule.start_time <= current_time_local <= schedule.end_time
            
            # Tính thời gian còn lại trước khi kích hoạt
            time_until_active = ""
            
            if is_active_today and schedule.status == 'active':
                if current_time_local < schedule.start_time:
                    # Tính thời gian còn lại đến khi bắt đầu
                    hours_until = schedule.start_time.hour - current_time_local.hour
                    minutes_until = schedule.start_time.minute - current_time_local.minute
                    if minutes_until < 0:
                        hours_until -= 1
                        minutes_until += 60
                    time_until_active = f"Còn {hours_until}h:{minutes_until:02d}p"
                elif in_time_range:
                    # Đang trong thời gian hoạt động
                    time_until_active = "ĐANG TRONG GIỜ"
                else:
                    # Đã kết thúc hôm nay
                    time_until_active = "Đã kết thúc hôm nay"
            elif not is_active_today and schedule.status == 'active':
                # Không hoạt động hôm nay
                next_day = 0
                for day in sorted([int(d) for d in days_active]):
                    if day > int(day_of_week):
                        next_day = day
                        break
                if next_day == 0:  # Nếu không tìm thấy ngày tiếp theo trong tuần này
                    next_day = min([int(d) for d in days_active])
                    days_until = next_day + 7 - int(day_of_week)
                else:
                    days_until = next_day - int(day_of_week)
                time_until_active = f"Còn {days_until} ngày"
            
            # Kiểm tra xem lịch trình có sẽ được chạy hôm nay không
            will_run_today = is_active_today and in_time_range and schedule.status == 'active'
            
            if will_run_today:
                active_schedules += 1
            
            status_str = f"{schedule.status}"
            if schedule.is_running:
                status_str = f"*** ĐANG CHẠY ***"
            elif is_active_today and schedule.status == 'active':
                if in_time_range:
                    status_str = f"active (trong giờ)"
                else:
                    if current_time_local < schedule.start_time:
                        status_str = f"active (chưa đến giờ)"
                    else:
                        status_str = f"active (đã hết giờ)"
            elif not is_active_today and schedule.status == 'active':
                status_str = f"active (khác ngày)"
            
            print("{:<5} {:<20} {:<10} {:<15} {:<15} {:<20} {:<20}".format(
                schedule.id,
                schedule.name[:20],
                schedule.schedule_type,
                schedule.start_time.strftime("%H:%M"),
                schedule.end_time.strftime("%H:%M"),
                status_str,
                time_until_active
            ))
        
        print("-"*100)
        print(f"Số lịch trình sẽ chạy hôm nay (trong giờ): {active_schedules}")
        
        if active_schedules > 0:
            print("\nCác lịch trình sẽ hoạt động:")
            for schedule in all_schedules:
                days_active = schedule.active_days.split(',')
                is_active_today = day_of_week in days_active
                in_time_range = schedule.start_time <= current_time_local <= schedule.end_time
                will_run_today = is_active_today and in_time_range and schedule.status == 'active'
                
                if will_run_today:
                    print(f"- {schedule.id}: {schedule.name} ({schedule.camera.name}, {schedule.schedule_type})")
        
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"[ERROR] Không thể in thông tin lịch trình: {e}")
        import traceback
        traceback.print_exc()

def run_command(command):
    """Chạy một lệnh trong tiến trình con và theo dõi đầu ra"""
    print(f"[INFO] Running command: {' '.join(command)}") # Log lệnh sẽ chạy
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        # shell=True # Không nên dùng shell=True với list command
    )
    
    for line in iter(process.stdout.readline, ''):
        print(line.strip())
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code:
        print(f"[ERROR] Lệnh thất bại với mã trả về {return_code}")
        # Không nên exit ở đây vì beat vẫn có thể chạy
        # sys.exit(return_code)
    return return_code # Trả về return code

def run_worker():
    """Khởi động Celery worker với eventlet pool"""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
    # Sử dụng eventlet pool cho Windows
    command = [
        sys.executable, # Đường dẫn python hiện tại
        "-m", "celery", # Chạy module celery
        "-A", "attendance_system_facial_recognition", 
        "worker", 
        "--loglevel=info", 
        "-P", "eventlet" # Chỉ định pool là eventlet
    ]
    run_command(command)

def run_beat():
    """Khởi động Celery beat"""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
    # Lệnh cho beat không cần thay đổi pool
    command = [
        sys.executable, 
        "-m", "celery",
        "-A", "attendance_system_facial_recognition", 
        "beat", 
        "--loglevel=info",
        # Thêm pidfile để tránh chạy nhiều instance beat cùng lúc (tùy chọn)
        # "--pidfile=celerybeat.pid" 
    ]
    run_command(command)

if __name__ == "__main__":
    print("[INFO] Bắt đầu khởi động Celery Worker và Beat...")
    
    # In thông tin về các lịch trình
    print_schedule_info()
    
    # Khởi động worker trong thread riêng
    worker_thread = threading.Thread(target=run_worker, name="CeleryWorkerThread")
    worker_thread.daemon = True
    worker_thread.start()
    
    # Đợi một chút để worker khởi động
    print("[INFO] Đợi worker khởi động (5 giây)...")
    time.sleep(5)
    
    # Khởi động beat trong thread chính (hoặc thread riêng nếu muốn)
    print("[INFO] Khởi động Celery Beat...")
    # Chạy beat trong thread riêng để có thể bắt KeyboardInterrupt dễ hơn
    beat_thread = threading.Thread(target=run_beat, name="CeleryBeatThread")
    beat_thread.daemon = True
    beat_thread.start()

    try:
        # Giữ thread chính chạy để bắt KeyboardInterrupt
        while worker_thread.is_alive() and beat_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Nhận tín hiệu KeyboardInterrupt. Đang dừng các tiến trình Celery...")
        # Worker và Beat là daemon threads nên sẽ tự động dừng khi thread chính kết thúc
        sys.exit(0)
    except Exception as e:
         print(f"[ERROR] Lỗi không xác định trong thread chính: {e}")
         sys.exit(1)
    finally:
         print("[INFO] Kết thúc script start_celery.") 