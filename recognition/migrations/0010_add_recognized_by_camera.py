from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('recognition', '0009_auto_20250414_1008'),
    ]

    operations = [
        migrations.AddField(
            model_name='attendancerecord',
            name='recognized_by_camera',
            field=models.CharField(
                blank=True, 
                help_text='Tên của camera đã thực hiện nhận diện (nếu có)',
                max_length=100, 
                null=True, 
                verbose_name='Nhận diện bởi Camera'
            ),
        ),
    ] 