# Generated by Django 3.1.8 on 2025-04-07 19:51

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('recognition', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='AttendanceRecord',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField(default=django.utils.timezone.now)),
                ('check_in', models.DateTimeField(blank=True, null=True)),
                ('check_out', models.DateTimeField(blank=True, null=True)),
                ('check_in_face', models.ImageField(blank=True, null=True, upload_to='attendance_faces/check_in/')),
                ('check_out_face', models.ImageField(blank=True, null=True, upload_to='attendance_faces/check_out/')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Bản ghi chấm công',
                'verbose_name_plural': 'Các bản ghi chấm công',
                'ordering': ['-date', '-check_in'],
                'unique_together': {('user', 'date')},
            },
        ),
    ]
