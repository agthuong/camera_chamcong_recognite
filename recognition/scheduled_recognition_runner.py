"""
Module quản lý và chạy lịch trình nhận diện tự động
Chạy file này như một tiến trình riêng biệt để thực hiện lịch trình nhận diện
"""

import os
import sys
import time
import logging
import traceback
import django
from datetime import datetime, timedelta
import threading

# Thiết lập Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
django.setup()

# Import sau khi Django đã được thiết lập
from django.utils import timezone
from django.conf import settings
from recognition.models import ScheduledCameraRecognition
from recognition.views import perform_scheduled_recognition

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.BASE_DIR, 'scheduled_recognition.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_and_run_schedules():
    """
    Kiểm tra và chạy các lịch trình cần thực hiện
    """
    try:
        # Lấy thời gian hiện tại
        now = timezone.now()
        logger.info(f"Checking schedules at {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Kiểm tra xem có lịch trình nào cần chạy không
        schedules_to_run = ScheduledCameraRecognition.objects.filter(
            status='active',
            next_run__lte=now
        )
        
        logger.info(f"Found {schedules_to_run.count()} schedules to run")
        
        # Chạy từng lịch trình trong một thread riêng
        threads = []
        for schedule in schedules_to_run:
            logger.info(f"Starting thread for schedule: {schedule.name}")
            thread = threading.Thread(
                target=perform_scheduled_recognition,
                args=(schedule.id,)
            )
            thread.start()
            threads.append(thread)
        
        # Đợi các thread hoàn thành
        for thread in threads:
            thread.join(timeout=600)  # Timeout 10 phút
        
        logger.info("Completed running schedules")
        
    except Exception as e:
        logger.error(f"Error checking schedules: {str(e)}")
        logger.error(traceback.format_exc())


def update_next_run_times():
    """
    Cập nhật thời gian chạy tiếp theo cho tất cả các lịch trình
    """
    try:
        schedules = ScheduledCameraRecognition.objects.filter(status='active', next_run__isnull=True)
        logger.info(f"Updating next run time for {schedules.count()} schedules")
        
        for schedule in schedules:
            try:
                now = timezone.now()
                day_of_week = str(now.isoweekday())
                active_days = schedule.active_days.split(',')
                
                # Kiểm tra xem hôm nay có phải là ngày hoạt động không
                if day_of_week in active_days:
                    # Kiểm tra xem thời gian hiện tại có nằm trong khoảng thời gian hoạt động không
                    current_time = now.time()
                    if schedule.start_time <= current_time <= schedule.end_time:
                        # Nếu thời gian hiện tại nằm trong khoảng, thì lịch trình sẽ chạy ngay
                        schedule.next_run = now
                        schedule.save()
                        logger.info(f"Schedule {schedule.name} will run immediately")
                        continue
                
                # Nếu không, tìm ngày tiếp theo trong danh sách ngày hoạt động
                next_day = now.date()
                days_checked = 0
                found_next_run = False
                
                # Kiểm tra tối đa 7 ngày
                while days_checked < 7 and not found_next_run:
                    next_day = next_day + timedelta(days=1)
                    next_day_of_week = str(next_day.weekday() + 1)  # Chuyển đổi từ 0-6 sang 1-7
                    
                    if next_day_of_week in active_days:
                        # Tạo datetime cho ngày tiếp theo với thời gian bắt đầu
                        next_run = timezone.make_aware(
                            datetime.combine(next_day, schedule.start_time)
                        )
                        schedule.next_run = next_run
                        schedule.save()
                        found_next_run = True
                        logger.info(f"Schedule {schedule.name} next run set to {next_run}")
                    
                    days_checked += 1
                
                if not found_next_run:
                    logger.warning(f"Could not find next run time for schedule {schedule.name}")
            
            except Exception as e:
                logger.error(f"Error updating next run time for schedule {schedule.id}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in update_next_run_times: {str(e)}")
        logger.error(traceback.format_exc())


def main():
    """
    Hàm chính để chạy scheduler
    """
    logger.info("Starting scheduled recognition runner...")
    
    # Cập nhật lần đầu
    update_next_run_times()
    
    while True:
        try:
            # Kiểm tra và thực hiện các lịch trình
            check_and_run_schedules()
            
            # Sleep một khoảng thời gian
            time.sleep(60)  # Kiểm tra mỗi phút
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Exiting...")
            break
        
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            logger.error(traceback.format_exc())
            time.sleep(60)  # Ngủ 1 phút trước khi thử lại


if __name__ == "__main__":
    main() 