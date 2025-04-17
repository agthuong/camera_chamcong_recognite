from django.db import models
from django.contrib.auth.models import User # Nếu bạn muốn liên kết với User sau này
from django.utils import timezone

# Model để lưu cấu hình Camera và ROI tương ứng
class CameraConfig(models.Model):
    name = models.CharField(
        max_length=100, 
        unique=True, # Tên camera nên là duy nhất

    )
    source = models.CharField(
        max_length=255, 
        unique=True, # Nguồn cũng nên là duy nhất

    )
    roi_x = models.IntegerField(null=True, blank=True, help_text="Tọa độ X của góc trên bên trái ROI")
    roi_y = models.IntegerField(null=True, blank=True, help_text="Tọa độ Y của góc trên bên trái ROI")
    roi_w = models.IntegerField(null=True, blank=True, help_text="Chiều rộng (Width) của ROI")
    roi_h = models.IntegerField(null=True, blank=True, help_text="Chiều cao (Height) của ROI")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_roi_tuple(self):
        """Trả về tuple (x, y, w, h) nếu ROI hợp lệ, ngược lại trả về None."""
        if self.roi_x is not None and self.roi_y is not None and \
           self.roi_w is not None and self.roi_h is not None and \
           self.roi_w > 0 and self.roi_h > 0:
            return (self.roi_x, self.roi_y, self.roi_w, self.roi_h)
        return None

    def __str__(self):
        roi_str = f"ROI={self.get_roi_tuple()}" if self.get_roi_tuple() else "ROI Chưa cấu hình"
        return f"{self.name} ({self.source}) - {roi_str}"

    class Meta:
        verbose_name = "Cấu hình Camera"
        verbose_name_plural = "Các Cấu hình Camera"
        ordering = ['name'] # Sắp xếp theo tên khi hiển thị

# Model lưu trữ thông tin vai trò người dùng
class UserRole(models.Model):
    ROLE_CHOICES = [
        ('supervisor', 'Supervisor'),
        ('worker', 'Worker'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='role_info')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='worker')
    supervisor = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='workers',
        help_text="Chọn giám sát trưởng (chỉ áp dụng cho worker)"
    )
    custom_supervisor = models.CharField(
        max_length=100, 
        null=True, 
        blank=True,
        help_text="Tên giám sát trưởng nếu không có trong hệ thống"
    )
    supervisor_email = models.EmailField(
        max_length=255, 
        null=True, 
        blank=True,
        help_text="Email của giám sát trưởng"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        if self.role == 'supervisor':
            return f"{self.user.username} - Supervisor"
        else:
            supervisor_name = self.supervisor.username if self.supervisor else self.custom_supervisor or "Chưa xác định"
            return f"{self.user.username} - Worker (Giám sát: {supervisor_name})"
    
    class Meta:
        verbose_name = "Vai trò người dùng"
        verbose_name_plural = "Vai trò người dùng"

class AttendanceRecord(models.Model):
    COMPANY_CHOICES = [
        ('DBplus', 'DBplus'),
        ('DBhomes', 'DBhomes'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    employee_id = models.CharField(max_length=50, null=True, blank=True, verbose_name="ID nhân viên")
    project = models.CharField(max_length=255, null=True, blank=True, verbose_name="Dự án")
    company = models.CharField(max_length=50, choices=COMPANY_CHOICES, default='DBplus', verbose_name="Công ty")
    date = models.DateField(default=timezone.now)
    check_in = models.DateTimeField(null=True, blank=True)
    check_out = models.DateTimeField(null=True, blank=True)
    check_in_image_url = models.ImageField(upload_to='attendance_faces/check_in/', null=True, blank=True, verbose_name="Ảnh Check-in")
    check_out_image_url = models.ImageField(upload_to='attendance_faces/check_out/', null=True, blank=True, verbose_name="Ảnh Check-out")
    recognized_by_camera = models.CharField(
        max_length=100, 
        null=True, 
        blank=True, 
        verbose_name="Nhận diện bởi Camera",
        help_text="Tên của camera đã thực hiện nhận diện (nếu có)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Bản ghi chấm công"
        verbose_name_plural = "Các bản ghi chấm công"
        ordering = ['-date', '-check_in']
        unique_together = ['user', 'date']  # Mỗi người chỉ có 1 bản ghi mỗi ngày

    def __str__(self):
        check_in_time = self.check_in.strftime("%H:%M:%S") if self.check_in else "Chưa có"
        check_out_time = self.check_out.strftime("%H:%M:%S") if self.check_out else "Chưa có"
        return f"{self.user.username} - {self.project} - {self.company} - {self.date} - Check in: {check_in_time}, Check out: {check_out_time}"

class ScheduledCameraRecognition(models.Model):
    """
    Model cho việc lên lịch nhận diện tự động cho camera
    """
    TIME_INTERVAL_CHOICES = [
        (15, '15 phút'),
        (30, '30 phút'),
        (60, '1 giờ'),
        (120, '2 giờ'),
        (180, '3 giờ'),
        (240, '4 giờ'),
        (360, '6 giờ'),
        (720, '12 giờ'),
    ]
    
    STATUS_CHOICES = [
        ('active', 'Đang hoạt động'),
        ('paused', 'Tạm dừng'),
        ('completed', 'Đã hoàn thành'),
    ]
    
    name = models.CharField(max_length=100, verbose_name="Tên lịch trình")
    camera = models.ForeignKey(
        CameraConfig, 
        on_delete=models.CASCADE, 
        related_name="schedules",
        verbose_name="Camera"
    )
    start_time = models.TimeField(verbose_name="Thời gian bắt đầu")
    end_time = models.TimeField(verbose_name="Thời gian kết thúc")
    interval_minutes = models.IntegerField(
        choices=TIME_INTERVAL_CHOICES,
        default=60,
        verbose_name="Khoảng thời gian kiểm tra (phút)"
    )
    active_days = models.CharField(
        max_length=20, 
        default="1,2,3,4,5", 
        verbose_name="Ngày hoạt động",
        help_text="Các ngày trong tuần (1-7, 1 là thứ Hai)"
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='active',
        verbose_name="Trạng thái"
    )
    last_run = models.DateTimeField(null=True, blank=True, verbose_name="Lần chạy cuối")
    next_run = models.DateTimeField(null=True, blank=True, verbose_name="Lần chạy tiếp theo")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} ({self.camera.name}) - {self.get_status_display()}"
    
    class Meta:
        verbose_name = "Lịch trình nhận diện tự động"
        verbose_name_plural = "Các lịch trình nhận diện tự động"
        ordering = ['camera__name', 'start_time']

class ScheduledRecognitionLog(models.Model):
    """
    Ghi lại nhật ký hoạt động của lịch trình nhận diện tự động
    """
    schedule = models.ForeignKey(
        ScheduledCameraRecognition, 
        on_delete=models.CASCADE,
        related_name='logs',
        verbose_name="Lịch trình"
    )
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name="Thời điểm")
    success = models.BooleanField(default=False, verbose_name="Thành công")
    message = models.TextField(verbose_name="Thông báo")
    recognized_users = models.TextField(
        blank=True, 
        null=True, 
        verbose_name="Người dùng đã nhận diện",
        help_text="Danh sách username đã nhận diện, ngăn cách bởi dấu phẩy"
    )
    attendance_records = models.ManyToManyField(
        AttendanceRecord,
        blank=True,
        related_name="recognition_logs",
        verbose_name="Bản ghi chấm công"
    )
    
    def __str__(self):
        return f"{self.schedule.name} - {self.timestamp.strftime('%d/%m/%Y %H:%M:%S')} - {'Thành công' if self.success else 'Thất bại'}"
    
    class Meta:
        verbose_name = "Nhật ký nhận diện tự động"
        verbose_name_plural = "Nhật ký nhận diện tự động"
        ordering = ['-timestamp']

class ContinuousAttendanceSchedule(models.Model):
    """
    Model cho việc lên lịch chấm công tự động theo khung giờ cố định
    """
    STATUS_CHOICES = [
        ('active', 'Đang hoạt động'),
        ('paused', 'Tạm dừng'),
    ]
    
    SCHEDULE_TYPE_CHOICES = [
        ('check_in', 'Giờ vào (Check-in)'),
        ('check_out', 'Giờ ra (Check-out)'),
    ]
    
    name = models.CharField(max_length=100, verbose_name="Tên lịch trình")
    camera = models.ForeignKey(
        CameraConfig, 
        on_delete=models.CASCADE, 
        related_name="continuous_schedules",
        verbose_name="Camera"
    )
    schedule_type = models.CharField(
        max_length=10,
        choices=SCHEDULE_TYPE_CHOICES,
        default='check_in',
        verbose_name="Loại chấm công"
    )
    start_time = models.TimeField(verbose_name="Thời gian bắt đầu")
    end_time = models.TimeField(verbose_name="Thời gian kết thúc")
    active_days = models.CharField(
        max_length=20, 
        default="1,2,3,4,5", 
        verbose_name="Ngày hoạt động",
        help_text="Các ngày trong tuần (1-7, 1 là thứ Hai)"
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='active',
        verbose_name="Trạng thái"
    )
    is_running = models.BooleanField(
        default=False,
        verbose_name="Đang chạy",
        help_text="Trạng thái hoạt động hiện tại (đang quét hay không)"
    )
    worker_id = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        verbose_name="ID Worker",
        help_text="ID của Celery worker đang xử lý lịch trình này"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} - {self.get_schedule_type_display()} ({self.camera.name})"
    
    class Meta:
        verbose_name = "Lịch trình chấm công liên tục"
        verbose_name_plural = "Các lịch trình chấm công liên tục"
        ordering = ['camera__name', 'schedule_type', 'start_time']

class ContinuousAttendanceLog(models.Model):
    """
    Ghi lại nhật ký hoạt động của lịch trình chấm công liên tục
    """
    schedule = models.ForeignKey(
        ContinuousAttendanceSchedule, 
        on_delete=models.CASCADE,
        related_name='logs',
        verbose_name="Lịch trình"
    )
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name="Thời điểm")
    event_type = models.CharField(
        max_length=20,
        verbose_name="Loại sự kiện",
        choices=[
            ('start', 'Bắt đầu'),
            ('stop', 'Kết thúc'),
            ('recognition', 'Nhận diện'),
            ('error', 'Lỗi'),
        ]
    )
    message = models.TextField(verbose_name="Thông báo")
    recognized_user = models.ForeignKey(
        User,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="continuous_recognitions",
        verbose_name="Người dùng đã nhận diện"
    )
    attendance_record = models.ForeignKey(
        AttendanceRecord,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="continuous_recognition_logs",
        verbose_name="Bản ghi chấm công"
    )
    
    def __str__(self):
        return f"{self.schedule.name} - {self.timestamp.strftime('%d/%m/%Y %H:%M:%S')} - {self.get_event_type_display()}"
    
    class Meta:
        verbose_name = "Nhật ký chấm công liên tục"
        verbose_name_plural = "Nhật ký chấm công liên tục"
        ordering = ['-timestamp']

