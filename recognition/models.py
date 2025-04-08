from django.db import models
from django.contrib.auth.models import User # Nếu bạn muốn liên kết với User sau này
from django.utils import timezone

# Model để lưu cấu hình Camera và ROI tương ứng
class CameraConfig(models.Model):
    name = models.CharField(
        max_length=100, 
        unique=True, # Tên camera nên là duy nhất
        help_text="Tên gợi nhớ cho camera (ví dụ: Camera Cổng Trước)"
    )
    source = models.CharField(
        max_length=255, 
        unique=True, # Nguồn cũng nên là duy nhất
        help_text="Nguồn video (ID Webcam, đường dẫn file, URL RTSP)"
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
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    check_in = models.DateTimeField(null=True, blank=True)
    check_out = models.DateTimeField(null=True, blank=True)
    check_in_face = models.ImageField(upload_to='attendance_faces/check_in/', null=True, blank=True)
    check_out_face = models.ImageField(upload_to='attendance_faces/check_out/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Bản ghi chấm công"
        verbose_name_plural = "Các bản ghi chấm công"
        ordering = ['-date', '-check_in']
        unique_together = ['user', 'date']  # Mỗi người chỉ có 1 bản ghi mỗi ngày

    def __str__(self):
        return f"{self.user.username} - {self.date} - Check in: {self.check_in}, Check out: {self.check_out}"

# Create your models here. (Các model khác nếu có)
