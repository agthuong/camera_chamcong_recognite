#!/usr/bin/env python
"""
Module để đảm bảo Celery hiển thị thời gian đúng múi giờ Việt Nam khi log.
"""
import datetime
import time
import logging
import pytz
from django.conf import settings

# Lấy timezone Việt Nam
vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')

# Class formatter tùy chỉnh cho logging
class VNTimeFormatter(logging.Formatter):
    """
    Định dạng log hiển thị thời gian ở múi giờ Việt Nam.
    """
    def formatTime(self, record, datefmt=None):
        """
        Ghi đè phương thức formatTime để chuyển đổi thời gian sang múi giờ VN.
        """
        # Lấy thời gian UTC từ timestamp
        utc_dt = datetime.datetime.fromtimestamp(record.created, pytz.UTC)
        # Chuyển đổi sang múi giờ Việt Nam
        vn_dt = utc_dt.astimezone(vn_tz)
        
        if datefmt:
            return vn_dt.strftime(datefmt)
        return vn_dt.strftime("%Y-%m-%d %H:%M:%S")

def apply_timezone_fix():
    """
    Áp dụng fix múi giờ cho tất cả logger hiện có.
    """
    # Định dạng datetime mặc định
    date_format = "%Y-%m-%d %H:%M:%S"
    # Định dạng log mặc định
    log_format = "[%(asctime)s,%(msecs)03d] [%(name)s] [%(levelname)s] %(message)s"
    
    # Tạo formatter với múi giờ VN
    vn_formatter = VNTimeFormatter(log_format, datefmt=date_format)
    
    # Áp dụng formatter cho tất cả handler của root logger
    for handler in logging.root.handlers:
        handler.setFormatter(vn_formatter)
    
    # Kiểm tra các logger đã đăng ký
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            handler.setFormatter(vn_formatter)
    
    print("Đã áp dụng hiệu chỉnh múi giờ Việt Nam cho log system")

# Áp dụng fix khi import module này
apply_timezone_fix()

# Hiển thị thông tin múi giờ hiện tại
print(f"Múi giờ Celery: {settings.CELERY_TIMEZONE}")
print(f"Múi giờ Django: {settings.TIME_ZONE}")
print(f"Múi giờ hệ thống đã điều chỉnh: Asia/Ho_Chi_Minh (GMT+7)")
print("Logging sẽ hiển thị thời gian theo múi giờ Việt Nam")

def convertUTCtoVN(dt):
    """
    Chuyển đổi một đối tượng datetime từ UTC sang múi giờ Việt Nam.
    Hàm này hỗ trợ cả datetime có timezone và không có timezone.

    Args:
        dt: đối tượng datetime cần chuyển đổi

    Returns:
        đối tượng datetime đã được chuyển sang múi giờ Việt Nam
    """
    if dt is None:
        return None
        
    # Nếu không có thông tin múi giờ, giả định là UTC
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return dt.astimezone(vn_tz)

# Tạo một hàm giúp chuyển đổi thời gian UTC sang múi giờ VN
def to_vn_time(dt):
    """
    Chuyển đổi một đối tượng datetime sang múi giờ Việt Nam (wrapper cho convertUTCtoVN).
    """
    return convertUTCtoVN(dt)

# Monkey patch để đảm bảo Django luôn hiển thị giờ VN
import django.utils.timezone
original_now = django.utils.timezone.now
original_localtime = django.utils.timezone.localtime

def now_in_vn():
    """
    Trả về thời gian hiện tại ở múi giờ Việt Nam.
    """
    return to_vn_time(original_now())

def localtime_in_vn(dt=None):
    """
    Đảm bảo localtime luôn trả về múi giờ Việt Nam.
    """
    if dt is None:
        dt = original_now()
    return to_vn_time(dt)

# Áp dụng monkey patch cho Django
django.utils.timezone.now = now_in_vn
django.utils.timezone.localtime = localtime_in_vn
print("Đã áp dụng patch cho django.utils.timezone.now và localtime")

# Thêm các patch khác nếu cần
# Chẳng hạn patch cho celery.utils.time cũng có thể giúp ích
try:
    import celery.utils.time
    original_utcnow = celery.utils.time.utcnow
    
    def celery_utcnow_in_vn():
        """Trả về thời gian hiện tại cho Celery trong múi giờ Việt Nam."""
        return to_vn_time(original_utcnow())
    
    # Monkey patch cho Celery
    celery.utils.time.utcnow = celery_utcnow_in_vn
    print("Đã áp dụng patch cho celery.utils.time.utcnow")
except (ImportError, AttributeError):
    print("Không thể áp dụng patch cho celery.utils.time")

# Test xem tất cả hàm thời gian đã được patch chưa
current_time = now_in_vn()
print(f"Thời gian hiện tại sau khi patch: {current_time} ({current_time.tzinfo})") 