#!/usr/bin/env python
"""
Script để kiểm tra và hiển thị thông tin về lịch trình chấm công và camera
"""
import os
import sys
import time
import datetime
import django

# Thiết lập Django để có thể truy vấn DB
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
django.setup()

# Import sau khi setup Django
from django.utils import timezone
from django.contrib.auth.models import User
from recognition.models import ContinuousAttendanceSchedule, CameraConfig
from recognition.tasks import active_processors, camera_locks

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

def print_schedules():
    """In thông tin về tất cả các lịch trình trong database"""
    try:
        # Lấy thời gian UTC và thời gian địa phương
        now_utc = timezone.now()
        now_local = timezone.localtime(now_utc)
        today = now_local.date()
        current_time_local = now_local.time()
        day_of_week = str(now_local.isoweekday())  # 1-7 (1 là thứ Hai)
        
        print(f"\n{Colors.HEADER}THÔNG TIN LỊCH TRÌNH CHẤM CÔNG{Colors.ENDC}")
        print(f"Thời gian UTC: {now_utc.strftime('%d/%m/%Y %H:%M:%S')} UTC")
        print(f"Thời gian VN:  {now_local.strftime('%d/%m/%Y %H:%M:%S')} (GMT+7)")
        print(f"Ngày trong tuần: {day_of_week} (1=Thứ Hai, 7=Chủ Nhật)")
        print_line()
        
        # Lấy tất cả lịch trình
        all_schedules = ContinuousAttendanceSchedule.objects.all().order_by('camera', 'start_time')
        active_schedules = []
        
        print(f"Tổng số lịch trình: {all_schedules.count()}")
        
        if all_schedules.count() == 0:
            print(f"{Colors.YELLOW}Không có lịch trình nào trong database.{Colors.ENDC}")
            return
        
        print(f"\n{'ID':<5} {'Tên':<20} {'Loại':<10} {'Camera':<15} {'Giờ':<15} {'Trạng thái':<20} {'Thời gian kích hoạt':<20}")
        print_line("-", 100)
        
        for schedule in all_schedules:
            # Kiểm tra xem lịch trình có hoạt động vào ngày này không
            days_active = schedule.active_days.split(',')
            is_active_today = day_of_week in days_active
            
            # Kiểm tra xem có đang trong khung giờ chạy không - sử dụng thời gian địa phương
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
                    time_until_active = f"{Colors.GREEN}ĐANG TRONG GIỜ{Colors.ENDC}"
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
                active_schedules.append(schedule)
            
            status_str = f"{schedule.status}"
            if schedule.is_running:
                status_str = f"{Colors.GREEN}*** ĐANG CHẠY ***{Colors.ENDC}"
            elif is_active_today and schedule.status == 'active':
                if in_time_range:
                    status_str = f"{Colors.YELLOW}active (trong giờ){Colors.ENDC}"
                else:
                    if current_time_local < schedule.start_time:
                        status_str = f"active (chưa đến giờ)"
                    else:
                        status_str = f"active (đã hết giờ)"
            elif not is_active_today and schedule.status == 'active':
                status_str = f"active (khác ngày)"
            
            # In thông tin lịch trình
            time_str = f"{schedule.start_time.strftime('%H:%M')}-{schedule.end_time.strftime('%H:%M')}"
            camera_name = schedule.camera.name if schedule.camera else "N/A"
            
            print(f"{schedule.id:<5} {schedule.name[:20]:<20} {schedule.schedule_type:<10} {camera_name[:15]:<15} {time_str:<15} {status_str:<20} {time_until_active:<20}")
        
        print_line("-", 100)
        print(f"Số lịch trình hoạt động hôm nay (trong giờ): {len(active_schedules)}")
        
        # Hiển thị thông tin chi tiết về các lịch trình đang hoạt động
        if active_schedules:
            print(f"\n{Colors.BOLD}Lịch trình đang hoạt động:{Colors.ENDC}")
            for schedule in active_schedules:
                print(f"- {schedule.id}: {schedule.name} ({schedule.camera.name}, {schedule.schedule_type})")
    
    except Exception as e:
        print(f"{Colors.RED}Lỗi khi hiển thị lịch trình: {str(e)}{Colors.ENDC}")
        import traceback
        traceback.print_exc()

def print_cameras():
    """In thông tin về tất cả các camera trong database"""
    try:
        print(f"\n{Colors.HEADER}THÔNG TIN CAMERA{Colors.ENDC}")
        print_line()
        
        # Lấy tất cả camera
        all_cameras = CameraConfig.objects.all().order_by('name')
        
        print(f"Tổng số camera: {all_cameras.count()}")
        
        if all_cameras.count() == 0:
            print(f"{Colors.YELLOW}Không có camera nào trong database.{Colors.ENDC}")
            return
        
        print(f"\n{'ID':<5} {'Tên':<25} {'Nguồn':<40} {'ROI':<20} {'Đang sử dụng':<15}")
        print_line()
        
        for camera in all_cameras:
            # Kiểm tra xem camera có đang được sử dụng không
            in_use = camera.source in camera_locks
            in_use_str = f"{Colors.RED}Có{Colors.ENDC}" if in_use else "Không"
            
            # Lấy thông tin ROI
            roi = camera.get_roi_tuple()
            roi_str = f"{roi}" if roi else "Không có"
            
            # In thông tin camera
            print(f"{camera.id:<5} {camera.name[:25]:<25} {camera.source[:40]:<40} {roi_str:<20} {in_use_str:<15}")
            
            # Nếu camera đang được sử dụng, hiển thị chi tiết
            if in_use:
                schedule_id = camera_locks[camera.source]
                try:
                    schedule = ContinuousAttendanceSchedule.objects.get(id=schedule_id)
                    print(f"   → Đang sử dụng bởi: {schedule.name} (ID: {schedule_id})")
                except:
                    print(f"   → Đang sử dụng bởi schedule ID: {schedule_id} (không tìm thấy trong DB)")
        
    except Exception as e:
        print(f"{Colors.RED}Lỗi khi hiển thị camera: {str(e)}{Colors.ENDC}")

def print_processor_info():
    """In thông tin về các processor đang hoạt động"""
    try:
        print(f"\n{Colors.HEADER}THÔNG TIN PROCESSOR ĐANG CHẠY{Colors.ENDC}")
        print_line()
        
        if not active_processors:
            print(f"{Colors.YELLOW}Không có processor nào đang chạy.{Colors.ENDC}")
            return
        
        print(f"Số processor đang chạy: {len(active_processors)}")
        
        print(f"\n{'Schedule ID':<15} {'Camera Source':<40} {'Đang chạy':<10} {'Thời gian chạy':<20}")
        print_line()
        
        for schedule_id, processor in active_processors.items():
            # Lấy thông tin về processor
            is_running = processor.is_running
            is_running_str = f"{Colors.GREEN}Có{Colors.ENDC}" if is_running else f"{Colors.RED}Không{Colors.ENDC}"
            
            # In thông tin processor
            print(f"{schedule_id:<15} {processor.camera_source[:40]:<40} {is_running_str:<10} {'N/A':<20}")
            
            # Thêm thông tin về lịch trình nếu có
            try:
                schedule = ContinuousAttendanceSchedule.objects.get(id=schedule_id)
                print(f"   → Lịch trình: {schedule.name}")
            except:
                print(f"   → Lịch trình: Không tìm thấy trong DB")
        
    except Exception as e:
        print(f"{Colors.RED}Lỗi khi hiển thị thông tin processor: {str(e)}{Colors.ENDC}")

def print_camera_locks():
    """In thông tin về các camera lock"""
    try:
        print(f"\n{Colors.HEADER}THÔNG TIN CAMERA LOCK{Colors.ENDC}")
        print_line()
        
        if not camera_locks:
            print(f"{Colors.YELLOW}Không có camera nào đang bị khóa.{Colors.ENDC}")
            return
        
        print(f"Số camera đang bị khóa: {len(camera_locks)}")
        
        print(f"\n{'Camera Source':<40} {'Schedule ID':<15} {'Schedule Name':<25}")
        print_line()
        
        for camera_source, schedule_id in camera_locks.items():
            # Tìm tên lịch trình nếu có
            try:
                schedule = ContinuousAttendanceSchedule.objects.get(id=schedule_id)
                schedule_name = schedule.name
            except:
                schedule_name = "Không tìm thấy trong DB"
            
            # In thông tin lock
            print(f"{camera_source[:40]:<40} {schedule_id:<15} {schedule_name[:25]:<25}")
        
    except Exception as e:
        print(f"{Colors.RED}Lỗi khi hiển thị thông tin camera lock: {str(e)}{Colors.ENDC}")

def main():
    """Hàm chính để chạy script"""
    print(f"{Colors.BOLD}{Colors.HEADER}KIỂM TRA LỊCH TRÌNH CHẤM CÔNG{Colors.ENDC}")
    print(f"Thời gian: {timezone.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print_line('=')
    
    print_schedules()
    print_cameras()
    print_processor_info()
    print_camera_locks()
    
    print_line('=')
    print(f"{Colors.BOLD}Kết thúc kiểm tra!{Colors.ENDC}")

if __name__ == "__main__":
    main() 