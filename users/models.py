from django.db import models
from django.contrib.auth.models import User
from django.conf import settings # Import settings
import datetime
# Create your models here.
	

class Present(models.Model):
	user=models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE) # Sử dụng settings.AUTH_USER_MODEL
	date = models.DateField(default=datetime.date.today)
	present=models.BooleanField(default=False)
	
class Time(models.Model):
	user=models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE) # Sử dụng settings.AUTH_USER_MODEL
	date = models.DateField(default=datetime.date.today)
	time=models.DateTimeField(null=True,blank=True)
	out=models.BooleanField(default=False)

# Mô hình Profile mới
class Profile(models.Model):
    ROLE_CHOICES = (
        ('supervisor', 'Supervisor'),
        ('worker', 'Worker'),
    )
    # Liên kết một-một với mô hình User có sẵn
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='profile')
    # Trường lưu vai trò
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, blank=False, null=False)
    # Trường lưu công ty
    company = models.CharField(max_length=100, blank=True, null=True)
    # Thêm trường nhà thầu
    contractor = models.CharField(max_length=100, blank=True, null=True, verbose_name='Nhà thầu')
    # Thêm trường lĩnh vực
    field = models.CharField(max_length=100, blank=True, null=True, verbose_name='Lĩnh vực')
    # Liên kết tới Profile của supervisor (chỉ dành cho worker)
    supervisor = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='workers', limit_choices_to={'role': 'supervisor'})

    def __str__(self):
        # Hiển thị thông tin cơ bản khi gọi đối tượng Profile
        return f"{self.user.username} - {self.get_role_display()}"

# (Tùy chọn) Bạn có thể thêm tín hiệu (signals) để tự động tạo Profile khi User được tạo
# from django.db.models.signals import post_save
# from django.dispatch import receiver
#
# @receiver(post_save, sender=settings.AUTH_USER_MODEL)
# def create_user_profile(sender, instance, created, **kwargs):
#     if created:
#         # Tự động tạo Profile nếu chưa có và gán vai trò mặc định nếu cần
#         Profile.objects.get_or_create(user=instance) # Cân nhắc gán role mặc định ở đây nếu muốn
#
# @receiver(post_save, sender=settings.AUTH_USER_MODEL)
# def save_user_profile(sender, instance, **kwargs):
#     # Đảm bảo Profile được lưu khi User được lưu
#     try:
#         instance.profile.save()
#     except Profile.DoesNotExist:
#         # Xử lý trường hợp Profile chưa tồn tại (mặc dù signal trên nên xử lý việc tạo)
#         Profile.objects.create(user=instance)

