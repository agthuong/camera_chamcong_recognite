from django.db import models
from django.contrib.auth.models import User # Nếu bạn muốn liên kết với User sau này
from django.utils import timezone
from django.core.exceptions import ValidationError
from django.db.models import Q
import datetime

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
    start_time = models.DateTimeField(verbose_name="Thời gian bắt đầu")
    end_time = models.DateTimeField(verbose_name="Thời gian kết thúc")
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
    start_time = models.DateTimeField(verbose_name="Thời gian bắt đầu")
    end_time = models.DateTimeField(verbose_name="Thời gian kết thúc")
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

    def _time_ranges_overlap(self, start1, end1, start2, end2):
        """
        Kiểm tra xem hai khoảng thời gian có chồng chéo không.
        Hỗ trợ xử lý lịch qua đêm (end_time < start_time).
        """
        if start1 is None or end1 is None or start2 is None or end2 is None:
            return False

        # Chuyển đổi TimeField thành datetime để so sánh dễ dàng hơn
        # Sử dụng một ngày cố định (ví dụ: 1970-01-01)
        ref_date = datetime.date(1970, 1, 1)
        dt1_start = datetime.datetime.combine(ref_date, start1)
        dt1_end = datetime.datetime.combine(ref_date, end1)
        dt2_start = datetime.datetime.combine(ref_date, start2)
        dt2_end = datetime.datetime.combine(ref_date, end2)

        # Xử lý lịch qua đêm cho khoảng 1
        range1_overnight = dt1_end < dt1_start
        if range1_overnight:
            # Chia thành 2 khoảng: [start, midnight) và [midnight, end) của ngày hôm sau
            dt1_end_next_day = dt1_end + datetime.timedelta(days=1)
            # Kiểm tra xem dt2 có overlap với [dt1_start, midnight_next_day) HOẶC [midnight, dt1_end)
            midnight_next_day = datetime.datetime.combine(ref_date + datetime.timedelta(days=1), datetime.time.min)
            midnight_this_day = datetime.datetime.combine(ref_date, datetime.time.min)

            overlap1 = (dt1_start < dt2_end and dt2_start < midnight_next_day)
            overlap2 = (midnight_this_day < dt2_end and dt2_start < dt1_end)
            # Xử lý trường hợp range 2 cũng qua đêm
            if dt2_end < dt2_start:
                 dt2_end_next_day = dt2_end + datetime.timedelta(days=1)
                 # Check overlap với [dt1_start, midnight_next_day) vs [dt2_start, midnight_next_day)
                 overlap1 = (dt1_start < dt2_end_next_day and dt2_start < midnight_next_day)
                 # Check overlap với [midnight, dt1_end) vs [midnight, dt2_end)
                 overlap2 = (midnight_this_day < dt1_end and midnight_this_day < dt2_end)
            return overlap1 or overlap2
        
        # Xử lý lịch qua đêm cho khoảng 2
        range2_overnight = dt2_end < dt2_start
        if range2_overnight:
             # Tương tự như trên, đảo vai trò range 1 và 2
             dt2_end_next_day = dt2_end + datetime.timedelta(days=1)
             midnight_next_day = datetime.datetime.combine(ref_date + datetime.timedelta(days=1), datetime.time.min)
             midnight_this_day = datetime.datetime.combine(ref_date, datetime.time.min)
             # Check overlap [dt2_start, midnight_next_day) vs dt1
             overlap1 = (dt2_start < dt1_end and dt1_start < midnight_next_day)
             # Check overlap [midnight, dt2_end) vs dt1
             overlap2 = (midnight_this_day < dt1_end and dt1_start < dt2_end)
             return overlap1 or overlap2

        # Trường hợp cả hai không qua đêm
        # Overlap khi: start1 < end2 và start2 < end1
        return dt1_start < dt2_end and dt2_start < dt1_end

    def clean(self):
        super().clean() # Gọi clean của lớp cha

        # Chỉ kiểm tra nếu lịch trình đang active
        if self.status != 'active':
            return

        # Tìm TẤT CẢ các lịch trình active khác cùng camera và khác ID
        conflicting_schedules = ContinuousAttendanceSchedule.objects.filter(
            camera=self.camera,
            status='active'
        ).exclude(pk=self.pk) # Loại trừ chính lịch trình đang kiểm tra

        if not conflicting_schedules.exists():
            return # Không có lịch trình nào xung đột tiềm năng

        # Lấy danh sách ngày hoạt động của lịch trình hiện tại
        current_active_days = set(self.active_days.split(','))

        # Kiểm tra từng lịch trình xung đột tiềm năng
        for conflict_schedule in conflicting_schedules:
            conflict_active_days = set(conflict_schedule.active_days.split(','))

            # Tìm các ngày trùng lặp
            common_days = current_active_days.intersection(conflict_active_days)

            if common_days:
                # Nếu có ngày trùng, kiểm tra giờ chồng chéo
                if self._time_ranges_overlap(self.start_time, self.end_time,
                                             conflict_schedule.start_time, conflict_schedule.end_time):
                    # Nếu giờ cũng chồng chéo, báo lỗi với thông báo đơn giản
                    error_message = "Đã có lịch trình trùng giờ trên camera này. Vui lòng điều chỉnh thời gian/ngày."
                    raise ValidationError(error_message)

    def save(self, *args, **kwargs):
        # Gọi full_clean để chạy validation (bao gồm cả clean) trước khi lưu
        # Không cần check force_insert hay force_update vì full_clean xử lý các trường hợp đó
        self.full_clean()
        super().save(*args, **kwargs) # Gọi save gốc

    def get_active_days_display(self):
        """
        Trả về danh sách các tuple (số ngày, tên ngày Tiếng Việt) cho các ngày hoạt động.
        """
        if not self.active_days:
            return []
        
        days_map = {
            '1': 'Thứ 2',
            '2': 'Thứ 3',
            '3': 'Thứ 4',
            '4': 'Thứ 5',
            '5': 'Thứ 6',
            '6': 'Thứ 7',
            '7': 'CN'
        }
        
        active_day_numbers = self.active_days.split(',')
        display_list = []
        # Sắp xếp theo đúng thứ tự ngày trong tuần
        for day_num in sorted(active_day_numbers, key=lambda x: int(x) if x.isdigit() else 0):
            day_name = days_map.get(day_num)
            if day_name:
                display_list.append((day_num, day_name))
        return display_list

    def __str__(self):
        # Cập nhật __str__ để rõ ràng hơn một chút
        return f"{self.name} ({self.camera.name}) - {self.get_schedule_type_display()} [{self.get_status_display()}]"

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

