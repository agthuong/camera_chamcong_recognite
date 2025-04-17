#!/usr/bin/env python
import os
import sys
import django
from datetime import datetime, timedelta
import pytz

# Thiết lập Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
django.setup()

from django.utils import timezone as django_timezone
from django.conf import settings
from recognition.models import Attendance
from management.models import Employee

def display_time_info():
    """Hiển thị thông tin thời gian hệ thống"""
    print("\n===== THÔNG TIN THỜI GIAN HIỆN TẠI =====")
    now_system = datetime.now()
    now_utc = datetime.utcnow()
    now_django = django_timezone.now()
    now_local = django_timezone.localtime(now_django)
    
    print(f"System time: {now_system}")
    print(f"UTC time: {now_utc}")
    print(f"Django time (UTC): {now_django}")
    print(f"Django time (local): {now_local}")
    print(f"Django timezone setting: {settings.TIME_ZONE}")
    print("======================================\n")

def check_attendance_records():
    """Kiểm tra các bản ghi attendance trong cơ sở dữ liệu"""
    print("===== KIỂM TRA THỜI GIAN CHECK-IN TRONG CƠ SỞ DỮ LIỆU =====")
    
    # Lấy 10 bản ghi chấm công gần nhất
    recent_records = Attendance.objects.all().order_by('-date', '-check_in')[:10]
    
    if not recent_records:
        print("Không tìm thấy bản ghi chấm công nào.")
        return
    
    print(f"Tìm thấy {len(recent_records)} bản ghi chấm công gần nhất:")
    
    issues_found = False
    for record in recent_records:
        employee = Employee.objects.filter(user=record.user).first()
        employee_name = employee.name if employee else record.user.username
        
        # Kiểm tra xem thời gian check-in có ở múi giờ UTC không
        if record.check_in:
            is_utc = record.check_in.tzinfo == pytz.UTC
            local_time = django_timezone.localtime(record.check_in) if is_utc else record.check_in
            time_diff = local_time - record.check_in if is_utc else timedelta(0)
            
            print(f"ID: {record.id}, Nhân viên: {employee_name}, Ngày: {record.date}")
            print(f"  Check-in (DB): {record.check_in} (Timezone: {record.check_in.tzinfo})")
            print(f"  Check-in (Local): {local_time} (Timezone: {local_time.tzinfo})")
            print(f"  Chênh lệch thời gian: {time_diff}")
            
            if is_utc and abs(time_diff) > timedelta(minutes=1):
                issues_found = True
                print(f"  [CẢNH BÁO] Thời gian check-in có thể không chính xác (múi giờ UTC)")
            
            if record.check_out:
                is_utc_out = record.check_out.tzinfo == pytz.UTC
                local_time_out = django_timezone.localtime(record.check_out) if is_utc_out else record.check_out
                print(f"  Check-out (DB): {record.check_out} (Timezone: {record.check_out.tzinfo})")
                print(f"  Check-out (Local): {local_time_out} (Timezone: {local_time_out.tzinfo})")
            
            print("-" * 50)
    
    if issues_found:
        return True
    else:
        print("Không tìm thấy vấn đề nào với múi giờ trong các bản ghi chấm công.")
        return False

def fix_attendance_records(dry_run=True):
    """Sửa múi giờ cho các bản ghi attendance từ UTC sang múi giờ Việt Nam"""
    print("\n===== CẬP NHẬT MÚI GIỜ CHO BẢN GHI CHẤM CÔNG =====")
    
    if dry_run:
        print("CHẾ ĐỘ THỬ NGHIỆM: Không thực hiện thay đổi trong cơ sở dữ liệu")
    
    # Lấy tất cả các bản ghi có check-in ở múi giờ UTC
    utc_records = Attendance.objects.filter(check_in__isnull=False)
    updated_count = 0
    
    for record in utc_records:
        if record.check_in and record.check_in.tzinfo == pytz.UTC:
            # Kiểm tra xem thời gian check-in có ở múi giờ UTC không
            local_time = django_timezone.localtime(record.check_in)
            time_diff = local_time - record.check_in
            
            employee = Employee.objects.filter(user=record.user).first()
            employee_name = employee.name if employee else record.user.username
            
            # Nếu có sự khác biệt đáng kể về thời gian, cập nhật bản ghi
            if abs(time_diff) > timedelta(minutes=1):
                print(f"Cập nhật bản ghi ID: {record.id}, Nhân viên: {employee_name}, Ngày: {record.date}")
                print(f"  Check-in cũ (UTC): {record.check_in}")
                print(f"  Check-in mới (VN): {local_time}")
                
                if not dry_run:
                    # Cập nhật thời gian check-in sang múi giờ VN
                    vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
                    new_checkin = record.check_in.astimezone(vn_timezone)
                    record.check_in = new_checkin
                    
                    # Cập nhật thời gian check-out nếu có
                    if record.check_out and record.check_out.tzinfo == pytz.UTC:
                        new_checkout = record.check_out.astimezone(vn_timezone)
                        record.check_out = new_checkout
                    
                    record.save()
                
                updated_count += 1
    
    print(f"\nTìm thấy {updated_count} bản ghi cần cập nhật")
    if not dry_run:
        print("Đã cập nhật thành công các bản ghi.")
    print("======================================")

def main():
    """Hàm chính để chạy script"""
    print("KIỂM TRA VÀ CẬP NHẬT MÚI GIỜ CHẤM CÔNG")
    print("======================================")
    
    # Hiển thị thông tin thời gian hiện tại
    display_time_info()
    
    # Kiểm tra các bản ghi
    issues_found = check_attendance_records()
    
    if issues_found:
        choice = input("\nPhát hiện vấn đề với múi giờ trong bản ghi chấm công. "
                       "Bạn có muốn cập nhật múi giờ cho các bản ghi này không? (y/n): ")
        
        if choice.lower() == 'y':
            # Chạy thử nghiệm để xem trước các thay đổi
            fix_attendance_records(dry_run=True)
            
            confirm = input("\nCÁC THAY ĐỔI TRÊN SẼ ĐƯỢC ÁP DỤNG. Xác nhận cập nhật? (y/n): ")
            if confirm.lower() == 'y':
                # Chạy cập nhật thực tế
                fix_attendance_records(dry_run=False)
                print("\nĐã cập nhật thành công múi giờ cho các bản ghi chấm công.")
            else:
                print("\nĐã hủy cập nhật.")
        else:
            print("\nKhông thực hiện cập nhật.")
    else:
        print("\nKhông cần thực hiện cập nhật nào.")

if __name__ == "__main__":
    main() 