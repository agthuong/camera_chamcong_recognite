#!/usr/bin/env python
"""
Script để làm sạch tài nguyên camera và đặt lại trạng thái của lịch trình.
Sử dụng khi có nhiều lịch trình bị treo hoặc khi có vấn đề với camera.
"""
import os
import sys
import django
import argparse
import time

# Thiết lập Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
django.setup()

from django.utils import timezone
from recognition.models import ContinuousAttendanceSchedule, ContinuousAttendanceLog
from recognition.tasks import stop_continuous_recognition, camera_locks, active_processors
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup_resources.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def cleanup_schedules(schedule_ids=None, force=False):
    """
    Làm sạch tài nguyên cho các lịch trình được chỉ định.
    
    Args:
        schedule_ids: Danh sách ID của các lịch trình cần làm sạch, nếu None sẽ làm sạch tất cả.
        force: Nếu True, sẽ đặt lại trạng thái là_running=False ngay cả khi lịch trình vẫn hoạt động.
    """
    try:
        # Lấy danh sách lịch trình cần làm sạch
        if schedule_ids:
            schedules = ContinuousAttendanceSchedule.objects.filter(id__in=schedule_ids)
            logger.info(f"Đã tìm thấy {schedules.count()} lịch trình từ danh sách ID cụ thể.")
        else:
            # Mặc định chỉ làm sạch những lịch trình đang chạy (is_running=True)
            schedules = ContinuousAttendanceSchedule.objects.filter(is_running=True)
            logger.info(f"Đã tìm thấy {schedules.count()} lịch trình đang chạy.")
        
        # Không có lịch trình nào cần làm sạch
        if not schedules.exists():
            logger.info("Không có lịch trình nào cần làm sạch.")
            return
        
        # In danh sách các lịch trình sẽ được làm sạch
        logger.info("Danh sách lịch trình sẽ được làm sạch:")
        for schedule in schedules:
            logger.info(f"  - ID: {schedule.id}, Tên: {schedule.name}, Camera: {schedule.camera.name}, is_running: {schedule.is_running}")
        
        # Dừng lịch trình và đặt lại trạng thái
        for schedule in schedules:
            schedule_id = schedule.id
            camera_source = schedule.camera.source
            
            logger.info(f"Đang làm sạch lịch trình {schedule_id} - {schedule.name}...")
            
            # Gọi stop_continuous_recognition nếu lịch trình đang chạy
            if schedule.is_running or force:
                # Thử dừng lịch trình qua Celery task
                if not force:
                    try:
                        logger.info(f"Gửi task stop_continuous_recognition cho lịch trình {schedule_id}...")
                        stop_task = stop_continuous_recognition.apply_async(args=[schedule_id])
                        logger.info(f"Đã gửi task stop, task ID: {stop_task.id}")
                        
                        # Chờ task hoàn thành (tối đa 5 giây)
                        stop_task.get(timeout=5)
                        logger.info(f"Task stop đã hoàn thành cho lịch trình {schedule_id}")
                    except Exception as e:
                        logger.error(f"Lỗi khi dừng lịch trình {schedule_id} qua Celery: {str(e)}")
                
                # Làm sạch các biến toàn cục
                if schedule_id in active_processors:
                    try:
                        logger.info(f"Dừng processor cho lịch trình {schedule_id}...")
                        processor = active_processors[schedule_id]
                        processor.stop()
                        del active_processors[schedule_id]
                        logger.info(f"Đã xóa processor khỏi active_processors cho lịch trình {schedule_id}")
                    except Exception as e:
                        logger.error(f"Lỗi khi dừng processor cho lịch trình {schedule_id}: {str(e)}")
                else:
                    logger.info(f"Không tìm thấy processor cho lịch trình {schedule_id} trong active_processors")
                
                # Giải phóng khóa camera nếu có
                if camera_source in camera_locks and camera_locks[camera_source] == schedule_id:
                    try:
                        del camera_locks[camera_source]
                        logger.info(f"Đã giải phóng khóa camera {camera_source} cho lịch trình {schedule_id}")
                    except Exception as e:
                        logger.error(f"Lỗi khi giải phóng khóa camera {camera_source}: {str(e)}")
                
                # Đặt lại trạng thái trong DB
                try:
                    schedule.refresh_from_db()
                    if schedule.is_running:
                        schedule.is_running = False
                        schedule.worker_id = None
                        schedule.save(update_fields=['is_running', 'worker_id'])
                        logger.info(f"Đã đặt lại is_running=False cho lịch trình {schedule_id} trong DB")
                        
                        # Ghi log sự kiện
                        ContinuousAttendanceLog.objects.create(
                            schedule=schedule,
                            event_type='stop',
                            message=f"Đã dừng lịch trình thông qua script cleanup_resources.py"
                        )
                    else:
                        logger.info(f"Lịch trình {schedule_id} đã được đặt is_running=False")
                except Exception as e:
                    logger.error(f"Lỗi khi cập nhật DB cho lịch trình {schedule_id}: {str(e)}")
        
        # In thông tin trạng thái sau khi làm sạch
        logger.info("Trạng thái sau khi làm sạch:")
        logger.info(f"  - Số lượng processor còn trong active_processors: {len(active_processors)}")
        logger.info(f"  - Số lượng camera còn trong camera_locks: {len(camera_locks)}")
        if active_processors:
            logger.info(f"  - Các lịch trình còn trong active_processors: {list(active_processors.keys())}")
        if camera_locks:
            logger.info(f"  - Các camera còn trong camera_locks: {camera_locks}")
        
        # Kiểm tra lại các lịch trình trong DB
        running_schedules = ContinuousAttendanceSchedule.objects.filter(is_running=True)
        if running_schedules.exists():
            logger.warning(f"Vẫn còn {running_schedules.count()} lịch trình có is_running=True trong DB:")
            for schedule in running_schedules:
                logger.warning(f"  - ID: {schedule.id}, Tên: {schedule.name}, Camera: {schedule.camera.name}")
        else:
            logger.info("Tất cả lịch trình đã được đặt is_running=False trong DB")
        
        logger.info("Quá trình làm sạch hoàn tất!")
        
    except Exception as e:
        logger.error(f"Lỗi không xác định trong quá trình làm sạch: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def list_running_schedules():
    """
    Liệt kê tất cả các lịch trình đang chạy và trạng thái tài nguyên.
    """
    try:
        # Lấy danh sách lịch trình đang chạy từ DB
        running_schedules = ContinuousAttendanceSchedule.objects.filter(is_running=True)
        logger.info(f"Tìm thấy {running_schedules.count()} lịch trình có is_running=True trong DB:")
        
        for schedule in running_schedules:
            schedule_id = schedule.id
            camera_source = schedule.camera.source
            
            # Kiểm tra xem lịch trình có đang chạy thực sự không
            is_in_active_processors = schedule_id in active_processors
            is_camera_locked = camera_source in camera_locks and camera_locks[camera_source] == schedule_id
            
            logger.info(f"  - ID: {schedule_id}, Tên: {schedule.name}, Camera: {schedule.camera.name}")
            logger.info(f"    + Worker ID: {schedule.worker_id}")
            logger.info(f"    + Có trong active_processors: {is_in_active_processors}")
            logger.info(f"    + Camera có bị khóa: {is_camera_locked}")
            
            # Kiểm tra sự không nhất quán
            if is_in_active_processors != is_camera_locked:
                logger.warning(f"    ! CẢNH BÁO: Không nhất quán giữa processor và khóa camera")
        
        # Hiển thị tài nguyên đang sử dụng
        logger.info("\nTài nguyên đang sử dụng:")
        logger.info(f"  - active_processors: {len(active_processors)} processors")
        if active_processors:
            for schedule_id, processor in active_processors.items():
                logger.info(f"    + Schedule ID: {schedule_id}, Camera: {processor.camera_source}")
        
        logger.info(f"  - camera_locks: {len(camera_locks)} cameras")
        if camera_locks:
            for camera_source, schedule_id in camera_locks.items():
                logger.info(f"    + Camera: {camera_source}, Schedule ID: {schedule_id}")
        
        # Kiểm tra sự không nhất quán
        inconsistent = False
        for schedule_id in active_processors:
            # Kiểm tra xem schedule_id có tồn tại trong DB không
            try:
                schedule = ContinuousAttendanceSchedule.objects.get(id=schedule_id)
                if not schedule.is_running:
                    logger.warning(f"  ! CẢNH BÁO: Schedule ID {schedule_id} có trong active_processors nhưng is_running=False trong DB")
                    inconsistent = True
            except ContinuousAttendanceSchedule.DoesNotExist:
                logger.warning(f"  ! CẢNH BÁO: Schedule ID {schedule_id} có trong active_processors nhưng không tồn tại trong DB")
                inconsistent = True
        
        for camera_source, schedule_id in camera_locks.items():
            # Kiểm tra xem camera có được sử dụng bởi lịch trình có is_running=True không
            try:
                schedule = ContinuousAttendanceSchedule.objects.get(id=schedule_id)
                if not schedule.is_running:
                    logger.warning(f"  ! CẢNH BÁO: Camera {camera_source} bị khóa bởi schedule ID {schedule_id} nhưng is_running=False trong DB")
                    inconsistent = True
            except ContinuousAttendanceSchedule.DoesNotExist:
                logger.warning(f"  ! CẢNH BÁO: Camera {camera_source} bị khóa bởi schedule ID {schedule_id} nhưng không tồn tại trong DB")
                inconsistent = True
        
        if inconsistent:
            logger.warning("\n! Phát hiện sự không nhất quán. Hãy cân nhắc chạy script với tùy chọn cleanup.")
        
        return running_schedules
        
    except Exception as e:
        logger.error(f"Lỗi không xác định khi liệt kê lịch trình: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Công cụ làm sạch tài nguyên camera và lịch trình')
    parser.add_argument('--list', action='store_true', help='Liệt kê các lịch trình đang chạy')
    parser.add_argument('--cleanup', action='store_true', help='Làm sạch tài nguyên của các lịch trình đang chạy')
    parser.add_argument('--force', action='store_true', help='Bắt buộc làm sạch ngay cả khi lịch trình đang hoạt động bình thường')
    parser.add_argument('--schedule-ids', type=int, nargs='+', help='Danh sách ID của các lịch trình cần làm sạch')
    
    args = parser.parse_args()
    
    # Nếu không có tùy chọn nào được cung cấp, hiển thị trợ giúp
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)
    
    # Liệt kê các lịch trình đang chạy
    if args.list:
        logger.info("=== BẮT ĐẦU LIỆT KÊ LỊCH TRÌNH ===")
        running_schedules = list_running_schedules()
        logger.info("=== KẾT THÚC LIỆT KÊ LỊCH TRÌNH ===")
    
    # Làm sạch tài nguyên
    if args.cleanup:
        logger.info("=== BẮT ĐẦU LÀM SẠCH TÀI NGUYÊN ===")
        cleanup_schedules(args.schedule_ids, args.force)
        logger.info("=== KẾT THÚC LÀM SẠCH TÀI NGUYÊN ===") 