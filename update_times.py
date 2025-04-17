#!/usr/bin/env python
"""
Script để cập nhật các bản ghi check-in trong cơ sở dữ liệu,
chuyển đổi thời gian từ UTC sang múi giờ Việt Nam (GMT+7)
"""
import os
import sys
import datetime
import logging
import django
import pytz

# Thiết lập Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
django.setup()

# Import sau khi Django đã được thiết lập
from django.utils import timezone
from django.conf import settings
from recognition.models import AttendanceRecord

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('update_times.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Múi giờ Việt Nam
VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')
UTC_TZ = pytz.UTC

def convert_utc_to_vn(dt):
    """
    Chuyển đổi thời gian UTC sang múi giờ Việt Nam
    """
    if dt is None:
        return None
        
    # Đảm bảo dt có thông tin múi giờ
    if dt.tzinfo is None:
        dt = UTC_TZ.localize(dt)
    
    # Chuyển sang múi giờ Việt Nam
    return dt.astimezone(VN_TZ)

def check_time_records():
    """
    Kiểm tra xem có bản ghi nào có thời gian check-in khác biệt đáng kể
    giữa UTC và múi giờ Việt Nam không
    """
    logger.info("Bắt đầu kiểm tra dữ liệu thời gian trong cơ sở dữ liệu...")
    records = AttendanceRecord.objects.filter(check_in__isnull=False)
    
    count = 0
    issues = 0
    
    for record in records:
        count += 1
        utc_time = record.check_in
        vn_time = convert_utc_to_vn(utc_time)
        
        # Tính toán sự khác biệt về giờ
        hour_diff = vn_time.hour - utc_time.hour
        if hour_diff < 0:  # Xử lý trường hợp vượt qua nửa đêm
            hour_diff += 24
            
        # Hiển thị các bản ghi có sự khác biệt đáng kể
        # Múi giờ VN là UTC+7, nên chênh lệch chuẩn là 7 giờ
        if hour_diff != 7:
            issues += 1
            logger.info(f"ID: {record.id}, User: {record.user.username}, Date: {record.date}")
            logger.info(f"  UTC time: {utc_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"  VN time: {vn_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"  Hour diff: {hour_diff} (Expected: 7)")
            
    logger.info(f"Đã kiểm tra {count} bản ghi, tìm thấy {issues} bản ghi có vấn đề")
    return issues

def update_time_records(dry_run=True):
    """
    Cập nhật các bản ghi check-in trong cơ sở dữ liệu
    
    Args:
        dry_run: Nếu True, chỉ hiển thị các thay đổi mà không thực hiện
    """
    if dry_run:
        logger.info("Chạy ở chế độ DRY RUN - không thực hiện thay đổi thực tế")
    else:
        logger.info("Chạy ở chế độ cập nhật thực tế - các thay đổi sẽ được lưu vào cơ sở dữ liệu")
    
    # Lấy tất cả bản ghi có check-in
    records = AttendanceRecord.objects.filter(check_in__isnull=False)
    logger.info(f"Tìm thấy {records.count()} bản ghi cần cập nhật")
    
    updated = 0
    for record in records:
        utc_time = record.check_in
        vn_time = convert_utc_to_vn(utc_time)
        
        # Tính toán sự khác biệt về giờ
        hour_diff = vn_time.hour - utc_time.hour
        if hour_diff < 0:
            hour_diff += 24
        
        # Cập nhật nếu chênh lệch giờ không phải là 7
        if hour_diff != 7:
            logger.info(f"Cập nhật bản ghi ID: {record.id}, User: {record.user.username}")
            logger.info(f"  Trước: {utc_time.strftime('%Y-%m-%d %H:%M:%S')} (UTC)")
            logger.info(f"  Sau:   {vn_time.strftime('%Y-%m-%d %H:%M:%S')} (VN)")
            
            if not dry_run:
                # Đặt lại thời gian check-in thành giờ Việt Nam
                # nhưng với thông tin múi giờ của UTC
                # Vì Django lưu trữ thời gian dưới dạng UTC
                vn_as_utc = UTC_TZ.localize(
                    datetime.datetime(
                        vn_time.year, vn_time.month, vn_time.day,
                        vn_time.hour, vn_time.minute, vn_time.second
                    )
                )
                record.check_in = vn_as_utc
                record.save(update_fields=['check_in'])
                updated += 1
    
    if not dry_run:
        logger.info(f"Đã cập nhật {updated} bản ghi trong cơ sở dữ liệu")
    else:
        logger.info(f"Sẽ cập nhật {updated} bản ghi nếu chạy trong chế độ thực tế")

def main():
    """
    Hàm chính để chạy script
    """
    logger.info("=== Bắt đầu kiểm tra và cập nhật thời gian trong cơ sở dữ liệu ===")
    
    # Kiểm tra cài đặt múi giờ Django
    logger.info(f"Múi giờ của Django: {settings.TIME_ZONE}")
    logger.info(f"USE_TZ: {settings.USE_TZ}")
    
    # Kiểm tra xem có vấn đề với dữ liệu không
    issues = check_time_records()
    
    if issues > 0:
        # Chạy ở chế độ dry run trước
        update_time_records(dry_run=True)
        
        # Hỏi người dùng có muốn cập nhật thực tế không
        response = input("\nBạn có muốn cập nhật thực tế các bản ghi này không? (y/n): ")
        if response.lower() == 'y':
            update_time_records(dry_run=False)
            logger.info("Đã hoàn thành cập nhật cơ sở dữ liệu")
        else:
            logger.info("Đã hủy cập nhật cơ sở dữ liệu")
    else:
        logger.info("Không tìm thấy bản ghi nào cần cập nhật")
    
    logger.info("=== Kết thúc kiểm tra và cập nhật thời gian ===")

if __name__ == "__main__":
    main() 