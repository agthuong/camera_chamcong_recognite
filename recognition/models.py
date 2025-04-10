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

