# Generated by Django 4.2.20 on 2025-04-16 09:42

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('recognition', '0012_scheduledcamerarecognition_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='ContinuousAttendanceSchedule',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, verbose_name='Tên lịch trình')),
                ('schedule_type', models.CharField(choices=[('check_in', 'Giờ vào (Check-in)'), ('check_out', 'Giờ ra (Check-out)')], default='check_in', max_length=10, verbose_name='Loại chấm công')),
                ('start_time', models.TimeField(verbose_name='Thời gian bắt đầu')),
                ('end_time', models.TimeField(verbose_name='Thời gian kết thúc')),
                ('active_days', models.CharField(default='1,2,3,4,5', help_text='Các ngày trong tuần (1-7, 1 là thứ Hai)', max_length=20, verbose_name='Ngày hoạt động')),
                ('status', models.CharField(choices=[('active', 'Đang hoạt động'), ('paused', 'Tạm dừng')], default='active', max_length=20, verbose_name='Trạng thái')),
                ('is_running', models.BooleanField(default=False, help_text='Trạng thái hoạt động hiện tại (đang quét hay không)', verbose_name='Đang chạy')),
                ('worker_id', models.CharField(blank=True, help_text='ID của Celery worker đang xử lý lịch trình này', max_length=100, null=True, verbose_name='ID Worker')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('camera', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='continuous_schedules', to='recognition.cameraconfig', verbose_name='Camera')),
            ],
            options={
                'verbose_name': 'Lịch trình chấm công liên tục',
                'verbose_name_plural': 'Các lịch trình chấm công liên tục',
                'ordering': ['camera__name', 'schedule_type', 'start_time'],
            },
        ),
        migrations.CreateModel(
            name='ContinuousAttendanceLog',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now_add=True, verbose_name='Thời điểm')),
                ('event_type', models.CharField(choices=[('start', 'Bắt đầu'), ('stop', 'Kết thúc'), ('recognition', 'Nhận diện'), ('error', 'Lỗi')], max_length=20, verbose_name='Loại sự kiện')),
                ('message', models.TextField(verbose_name='Thông báo')),
                ('attendance_record', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='continuous_recognition_logs', to='recognition.attendancerecord', verbose_name='Bản ghi chấm công')),
                ('recognized_user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='continuous_recognitions', to=settings.AUTH_USER_MODEL, verbose_name='Người dùng đã nhận diện')),
                ('schedule', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='logs', to='recognition.continuousattendanceschedule', verbose_name='Lịch trình')),
            ],
            options={
                'verbose_name': 'Nhật ký chấm công liên tục',
                'verbose_name_plural': 'Nhật ký chấm công liên tục',
                'ordering': ['-timestamp'],
            },
        ),
    ]
