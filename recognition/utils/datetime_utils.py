"""
Mô-đun trợ giúp xử lý thời gian và timezone trong ứng dụng

Cung cấp các hàm trợ giúp để đảm bảo sử dụng timezone nhất quán trong toàn bộ ứng dụng.
Luôn sử dụng các hàm này thay vì gọi trực tiếp datetime.now() hoặc các hàm tương tự.
"""

from django.utils import timezone
from django.conf import settings
import datetime

def get_current_time():
    """
    Lấy thời gian hiện tại với múi giờ được xác định trong settings.TIME_ZONE
    Đây là hàm chính để sử dụng thay cho datetime.now()
    
    Returns:
        datetime: Thời gian hiện tại (timezone aware)
    """
    return timezone.now()

def get_current_date():
    """
    Lấy ngày hiện tại (chỉ phần date) theo múi giờ địa phương
    
    Returns:
        date: Ngày hiện tại
    """
    return timezone.localdate()

def localize_datetime(dt):
    """
    Chuyển đổi một datetime object sang múi giờ địa phương
    
    Args:
        dt: Đối tượng datetime (naive hoặc aware)
        
    Returns:
        datetime: Đối tượng datetime đã được chuyển đổi sang múi giờ địa phương
    """
    if dt is None:
        return None
        
    # Nếu là naive datetime, giả định nó là UTC
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, timezone.utc)
        
    # Chuyển đổi sang múi giờ địa phương
    return timezone.localtime(dt)

def format_time(dt, format_str="%H:%M:%S"):
    """
    Định dạng thời gian theo một format cụ thể
    
    Args:
        dt: Đối tượng datetime cần định dạng
        format_str: Chuỗi định dạng (mặc định: HH:MM:SS)
        
    Returns:
        str: Chuỗi thời gian đã định dạng
    """
    if dt is None:
        return "-"
        
    dt = localize_datetime(dt)
    return dt.strftime(format_str)

def format_datetime(dt, format_str="%d/%m/%Y %H:%M:%S"):
    """
    Định dạng ngày tháng theo một format cụ thể
    
    Args:
        dt: Đối tượng datetime cần định dạng
        format_str: Chuỗi định dạng (mặc định: DD/MM/YYYY HH:MM:SS)
        
    Returns:
        str: Chuỗi ngày tháng đã định dạng
    """
    if dt is None:
        return "-"
        
    dt = localize_datetime(dt)
    return dt.strftime(format_str)

def make_aware_datetime(year, month, day, hour=0, minute=0, second=0):
    """
    Tạo một datetime object có timezone từ các giá trị ngày giờ
    
    Args:
        year, month, day, hour, minute, second: Các thành phần thời gian
        
    Returns:
        datetime: Đối tượng datetime có timezone
    """
    naive_dt = datetime.datetime(year, month, day, hour, minute, second)
    return timezone.make_aware(naive_dt)

def combine_date_time(date_obj, time_obj):
    """
    Kết hợp đối tượng date và time thành một datetime với timezone
    
    Args:
        date_obj: Đối tượng date
        time_obj: Đối tượng time
        
    Returns:
        datetime: Đối tượng datetime có timezone
    """
    if date_obj is None or time_obj is None:
        return None
        
    naive_dt = datetime.datetime.combine(date_obj, time_obj)
    return timezone.make_aware(naive_dt) 