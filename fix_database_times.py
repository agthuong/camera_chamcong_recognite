#!/usr/bin/env python
"""
Script để kiểm tra và sửa thời gian trong database
"""
import os
import sys
import traceback
import datetime
import pytz
import logging
import django

# Thiết lập Django để truy vấn DB
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
django.setup()

from django.utils import timezone
from django.contrib.auth.models import User
from recognition.models import AttendanceRecord, ContinuousAttendanceLog
from django.db.models import Q

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Múi giờ Việt Nam
vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')

def show_time_info():
    """Hiển thị thông tin về thời gian hiện tại"""
    try:
        now_utc = timezone.now()
        now_local = timezone.localtime(now_utc)
        
        logger.info("-" * 80)
        logger.info("THÔNG TIN THỜI GIAN HỆ THỐNG:")
        logger.info(f"Thời gian UTC: {now_utc.strftime('%d/%m/%Y %H:%M:%S')} UTC")
        logger.info(f"Thời gian địa phương: {now_local.strftime('%d/%m/%Y %H:%M:%S')} ({now_local.tzinfo})")
        logger.info(f"Múi giờ Django: {timezone.get_current_timezone_name()}")
        logger.info("-" * 80)
    except Exception as e:
        logger.error(f"Lỗi khi hiển thị thông tin thời gian: {str(e)}")

def check_attendance_records():
    """Kiểm tra các bản ghi chấm công có thời gian trong database"""
    try:
        # Lấy 10 bản ghi chấm công gần nhất
        latest_records = AttendanceRecord.objects.all().order_by('-date', '-updated_at')[:10]
        
        logger.info("-" * 80)
        logger.info("10 BẢN GHI CHẤM CÔNG MỚI NHẤT:")
        logger.info("{:<4} {:<15} {:<12} {:<20} {:<20} {:<20}".format(
            "ID", "User", "Ngày", "Check-in", "Check-out", "Cập nhật"
        ))
        logger.info("-" * 80)
        
        for record in latest_records:
            check_in_str = record.check_in.strftime('%d/%m/%Y %H:%M:%S') if record.check_in else "Chưa có"
            check_out_str = record.check_out.strftime('%d/%m/%Y %H:%M:%S') if record.check_out else "Chưa có"
            updated_str = record.updated_at.strftime('%d/%m/%Y %H:%M:%S')
            
            logger.info("{:<4} {:<15} {:<12} {:<20} {:<20} {:<20}".format(
                record.id, record.user.username, record.date.strftime('%d/%m/%Y'),
                check_in_str, check_out_str, updated_str
            ))
        
        # Kiểm tra bản ghi có check-in là 05:00:01
        fixed_time_records = AttendanceRecord.objects.filter(
            Q(check_in__hour=5) & Q(check_in__minute=0) & Q(check_in__second=1)
        ).order_by('-date')[:10]
        
        if fixed_time_records.exists():
            logger.info("\nBẢN GHI CÓ THỜI GIAN CHECK-IN CỐ ĐỊNH (05:00:01):")
            logger.info("{:<4} {:<15} {:<12} {:<20}".format(
                "ID", "User", "Ngày", "Check-in"
            ))
            logger.info("-" * 60)
            
            for record in fixed_time_records:
                check_in_str = record.check_in.strftime('%d/%m/%Y %H:%M:%S') if record.check_in else "Chưa có"
                logger.info("{:<4} {:<15} {:<12} {:<20}".format(
                    record.id, record.user.username, record.date.strftime('%d/%m/%Y'), check_in_str
                ))
                
            logger.info(f"\nTìm thấy {fixed_time_records.count()} bản ghi có check-in cố định là 05:00:01")
            logger.info("Sử dụng tùy chọn 'fix' để sửa các bản ghi này.")
        else:
            logger.info("\nKhông tìm thấy bản ghi nào có check-in cố định là 05:00:01")
            
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra bản ghi chấm công: {str(e)}")
        logger.error(traceback.format_exc())

def fix_fixed_time_records():
    """Sửa các bản ghi có check-in cố định 05:00:01"""
    try:
        # Tìm các bản ghi có check-in là 05:00:01
        fixed_time_records = AttendanceRecord.objects.filter(
            Q(check_in__hour=5) & Q(check_in__minute=0) & Q(check_in__second=1)
        ).order_by('-date')
        
        count = fixed_time_records.count()
        if count == 0:
            logger.info("Không tìm thấy bản ghi nào có check-in cố định là 05:00:01")
            return
            
        logger.info(f"Tìm thấy {count} bản ghi có check-in cố định là 05:00:01")
        logger.info("Bắt đầu sửa chữa...")
        
        fixed_count = 0
        for record in fixed_time_records:
            logger.info(f"Xử lý bản ghi ID: {record.id}, User: {record.user.username}, Date: {record.date}")
            logger.info(f"  Check-in hiện tại: {record.check_in}")
            
            # Tạo check-in mới giống với thời gian updated_at nhưng đảm bảo cùng ngày với record.date
            if record.updated_at:
                new_time = record.updated_at.time()
                new_check_in = datetime.datetime.combine(record.date, new_time)
                new_check_in = timezone.make_aware(new_check_in, timezone=vn_tz)
                
                # Lưu giá trị cũ để hiển thị
                old_check_in = record.check_in
                
                # Cập nhật bản ghi
                record.check_in = new_check_in
                record.save(update_fields=['check_in'])
                
                logger.info(f"  Check-in mới: {record.check_in}")
                logger.info(f"  Đã cập nhật từ {old_check_in} thành {new_check_in}")
                fixed_count += 1
            else:
                logger.warning(f"  Không có thời gian updated_at, không thể sửa")
                
        logger.info(f"Đã sửa {fixed_count}/{count} bản ghi có check-in cố định")
        
    except Exception as e:
        logger.error(f"Lỗi khi sửa bản ghi check-in cố định: {str(e)}")
        logger.error(traceback.format_exc())

def fix_attendance_records():
    """Sửa các bản ghi chấm công có thời gian không đúng múi giờ"""
    try:
        # Lấy tất cả bản ghi chấm công trong 7 ngày gần nhất
        recent_date = timezone.now().date() - datetime.timedelta(days=7)
        records = AttendanceRecord.objects.filter(date__gte=recent_date)
        
        logger.info(f"Tìm thấy {records.count()} bản ghi chấm công trong 7 ngày gần đây")
        
        # Tìm các bản ghi có vấn đề về múi giờ
        records_to_fix = []
        for record in records:
            if record.check_in and record.check_in.tzinfo is None:
                records_to_fix.append(record)
                continue
                
            # Kiểm tra xem thời gian có khác biệt đáng kể so với múi giờ hiện tại
            if record.check_in:
                check_in_vn = record.check_in.astimezone(vn_tz) if record.check_in.tzinfo else record.check_in
                hour_diff = abs(check_in_vn.hour - 12)  # So sánh với 12 giờ trưa
                if hour_diff > 8:  # Chênh lệch lớn
                    records_to_fix.append(record)
        
        logger.info(f"Tìm thấy {len(records_to_fix)} bản ghi cần được sửa múi giờ")
        
        for record in records_to_fix:
            logger.info(f"ID: {record.id}, User: {record.user.username}")
            logger.info(f"  Check-in trước: {record.check_in}")
            
            # Fix bản ghi
            if record.check_in and record.check_in.tzinfo is None:
                # Nếu không có tzinfo, giả định là UTC
                record.check_in = pytz.UTC.localize(record.check_in)
            
            # Chuyển đổi về múi giờ địa phương
            if record.check_in:
                # Điều chỉnh check_in để phù hợp với múi giờ hiện tại
                # Nếu check_in là buổi sáng nhưng hiện tại là buổi chiều
                check_in_vn = record.check_in.astimezone(vn_tz)
                
                # Tạo một thời gian mới với giờ, phút, giây giữ nguyên nhưng ngày và tháng là ngày hiện tại
                fixed_check_in = timezone.make_aware(
                    datetime.datetime.combine(
                        record.date,
                        check_in_vn.time()
                    )
                )
                
                record.check_in = fixed_check_in
                record.save(update_fields=['check_in'])
                
                logger.info(f"  Check-in sau: {record.check_in}")
        
        logger.info("-" * 80)
        logger.info(f"Đã cập nhật {len(records_to_fix)} bản ghi.")
        
    except Exception as e:
        logger.error(f"Lỗi khi sửa múi giờ chấm công: {str(e)}")
        logger.error(traceback.format_exc())

def fix_logs():
    """Kiểm tra log các bản ghi chấm công"""
    try:
        # Lấy 10 log gần nhất
        latest_logs = ContinuousAttendanceLog.objects.all().order_by('-timestamp')[:10]
        
        logger.info("-" * 80)
        logger.info("10 LOG CHẤM CÔNG GẦN NHẤT:")
        logger.info("{:<4} {:<10} {:<20} {:<30}".format(
            "ID", "Loại", "Thời gian", "Thông báo"
        ))
        logger.info("-" * 80)
        
        for log in latest_logs:
            message = log.message[:30] + "..." if len(log.message) > 30 else log.message
            timestamp_str = log.timestamp.strftime('%d/%m/%Y %H:%M:%S')
            
            logger.info("{:<4} {:<10} {:<20} {:<30}".format(
                log.id, log.event_type, timestamp_str, message
            ))
            
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra log chấm công: {str(e)}")

def fix_time_display():
    """Đảm bảo hiển thị thời gian đúng múi giờ Việt Nam"""
    try:
        # Áp dụng định dạng múi giờ Việt Nam cho tất cả các logger
        class VNTimeFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                # Lấy thời gian UTC từ timestamp
                utc_dt = datetime.datetime.fromtimestamp(record.created, pytz.UTC)
                # Chuyển đổi sang múi giờ Việt Nam
                vn_dt = utc_dt.astimezone(vn_tz)
                
                if datefmt:
                    return vn_dt.strftime(datefmt)
                return vn_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Định dạng datetime mặc định
        date_format = "%Y-%m-%d %H:%M:%S"
        # Định dạng log mặc định
        log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
        
        # Tạo formatter với múi giờ VN
        vn_formatter = VNTimeFormatter(log_format, datefmt=date_format)
        
        # Áp dụng formatter cho tất cả handler của root logger
        for handler in logging.root.handlers:
            handler.setFormatter(vn_formatter)
        
        logger.info("Đã áp dụng định dạng thời gian Việt Nam cho logging")
        
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập định dạng thời gian: {str(e)}")

def main():
    """Hàm chính"""
    # Thiết lập hiển thị thời gian đúng
    fix_time_display()
    
    # Hiển thị thông tin thời gian
    show_time_info()
    
    # Nếu có tham số command line
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'check':
            # Chỉ kiểm tra
            check_attendance_records()
            fix_logs()
            
        elif command == 'fix':
            # Sửa các bản ghi có vấn đề
            check_attendance_records()
            
            # Hỏi xác nhận trước khi sửa
            confirm = input("\nBạn có chắc chắn muốn sửa các bản ghi có vấn đề không? (y/n): ")
            if confirm.lower() == 'y':
                # Sửa các bản ghi có check-in 12:00:01
                fix_fixed_time_records()
                
                # Sửa các bản ghi có vấn đề về múi giờ
                fix_attendance_records()
            else:
                logger.info("Đã hủy việc sửa bản ghi")
                
        elif command == 'fixtime':
            # Sửa chỉ các bản ghi có check-in 05:00:01
            check_attendance_records()
            
            # Hỏi xác nhận trước khi sửa
            confirm = input("\nBạn có chắc chắn muốn sửa các bản ghi có check-in 05:00:01 không? (y/n): ")
            if confirm.lower() == 'y':
                fix_fixed_time_records()
            else:
                logger.info("Đã hủy việc sửa bản ghi")
        else:
            logger.info("Lệnh không hợp lệ. Sử dụng: check, fix, hoặc fixtime")
    else:
        # Mặc định kiểm tra
        check_attendance_records()
        logger.info("\nSử dụng 'python fix_database_times.py fix' để sửa các bản ghi có vấn đề")
        logger.info("Sử dụng 'python fix_database_times.py fixtime' để chỉ sửa các bản ghi có check-in 05:00:01")

if __name__ == "__main__":
    main() 