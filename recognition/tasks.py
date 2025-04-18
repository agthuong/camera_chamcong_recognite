"""
Celery tasks cho hệ thống nhận diện liên tục
"""
import os
import time
import logging

import traceback
import numpy as np
import cv2
from datetime import datetime
import threading
import face_recognition

from celery import shared_task
from django.utils import timezone as django_timezone
from django.conf import settings
from django.contrib.auth.models import User


from .models import (
    AttendanceRecord, ContinuousAttendanceSchedule, 
    ContinuousAttendanceLog
)

# Import các hàm cần thiết từ module views
#from .recognition_utils import predict, update_attendance_in_db_in, update_attendance_in_db_out
from imutils.face_utils import FaceAligner
from imutils import face_utils

# Cấu hình logging an toàn hơn
logger = logging.getLogger(__name__)

# Nếu chưa có handlers, thêm handlers mới
if not logger.handlers:
    # Sử dụng StreamHandler không chỉ định stream
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)
    
    # Thêm file handler với encoding utf-8
    try:
        file_handler = logging.FileHandler('celery_recognition.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Không thể tạo file handler: {str(e)}")

logger.setLevel(logging.INFO)

# Lưu trữ các processor đang hoạt động
active_processors = {}

# Lưu trữ các camera đang được sử dụng - dùng để kiểm soát truy cập vào camera
camera_locks = {}

# --- Lớp đọc video bằng thread --- 
class ThreadedVideoCapture:
    """
    Xử lý nguồn video trong một thread riêng để tránh việc buffer bị đầy,
    đặc biệt hữu ích cho luồng RTSP.
    """
    def __init__(self, source):
        """
        Khởi tạo ThreadedVideoCapture
        
        Args:
            source: Có thể là đường dẫn RTSP, ID của webcam, hoặc đường dẫn file video
        """
        try:
            # Thử chuyển source thành số nguyên (ID của webcam)
            self.source = int(source)
            self.is_rtsp = False
        except ValueError:
            # Nếu không phải số nguyên, coi là đường dẫn
            self.source = source
            # Kiểm tra nếu là luồng RTSP
            self.is_rtsp = source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))
            
        self.capture = None
        self.thread = None
        self.lock = threading.Lock()
        self.running = False
        self.current_frame = None
        self.frame_available = False
        self.last_error_time = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        logger.info(f"Khởi tạo ThreadedVideoCapture với nguồn: {source} {'(RTSP/Streaming)' if self.is_rtsp else ''}")
        
    def start(self):
        """Khởi động thread đọc frame"""
        self.capture = cv2.VideoCapture(self.source)
        
        # Đặt các tham số đặc biệt cho RTSP
        if self.is_rtsp:
            # Cấu hình RTSP để giảm độ trễ và tăng reliability
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giữ buffer nhỏ để luôn lấy frame mới nhất
            # Đặt Codec cho FFMPEG (backend phổ biến của OpenCV cho RTSP)
            # Không sử dụng MJPEG có thể giảm lag với một số camera
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            # Đặt thời gian timeout nếu có thể
            try:
                self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            except AttributeError:
                logger.warning("Phiên bản OpenCV không hỗ trợ CAP_PROP_OPEN_TIMEOUT_MSEC")

            # Cho phép kết nối lại nhanh hơn
            self.max_consecutive_errors = 3  # Giảm số lỗi trước khi kết nối lại
            logger.info("Đã áp dụng cấu hình đặc biệt cho luồng RTSP/Streaming")
            
        if not self.capture.isOpened():
            logger.error(f"Không thể mở nguồn video: {self.source}")
            return False
            
        # Tùy chọn kiểm tra kích thước frame
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        logger.info(f"Kích thước video: {width}x{height}, FPS: {fps:.1f}")
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, name=f"VideoCaptureThread-{self.source}", daemon=True)
        self.thread.start()
        logger.info(f"Đã khởi động thread đọc frame từ nguồn: {self.source}")
        return self
        
    def _read_loop(self):
        """Loop chạy trong thread để liên tục đọc frame mới"""
        retry_delay = 0.01  # Giá trị cơ bản
        while self.running:
            try:
                ret, frame = self.capture.read()
                if not ret or frame is None or frame.size == 0:
                    self.consecutive_errors += 1
                    current_time = time.time()
                    
                    # Đặt lại last_error_time nếu đây là lỗi đầu tiên
                    if self.last_error_time is None:
                        self.last_error_time = current_time
                    
                    # In cảnh báo mỗi 3 giây để tránh làm tràn console
                    if self.last_error_time is None or (current_time - self.last_error_time) > 3:
                        logger.warning(f"Lỗi khi đọc frame. Số lỗi liên tiếp: {self.consecutive_errors}")
                        self.last_error_time = current_time
                    
                    # Điều chỉnh retry_delay tùy thuộc vào loại luồng
                    retry_delay = 0.01 if not self.is_rtsp else 0.05 * self.consecutive_errors
                    retry_delay = min(retry_delay, 2.0)  # Giới hạn tối đa 2 giây
                    
                    # Thử kết nối lại camera nếu có quá nhiều lỗi liên tiếp
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        logger.info(f"Thử kết nối lại với nguồn video: {self.source}")
                        self.capture.release()
                        time.sleep(retry_delay)  # Đợi một chút trước khi thử lại
                        self.capture = cv2.VideoCapture(self.source)
                        
                        # Áp dụng lại các cấu hình đặc biệt cho RTSP
                        if self.is_rtsp:
                            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                            try:
                                self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                            except AttributeError:
                                pass  # Đã báo lỗi ở phần trên, không cần báo lại
                            
                        self.consecutive_errors = 0
                        
                    time.sleep(retry_delay)  # Tránh sử dụng 100% CPU khi gặp lỗi
                    continue
                
                # Reset counter nếu đọc thành công
                self.consecutive_errors = 0
                self.last_error_time = None
                
                # Kiểm tra frame hợp lệ (không quá nhỏ)
                if frame.shape[0] < 10 or frame.shape[1] < 10:
                    logger.warning(f"Frame không hợp lệ, kích thước quá nhỏ: {frame.shape}")
                    time.sleep(0.01)
                    continue
                
                # Cập nhật frame mới nhất với lock để tránh race condition
                with self.lock:
                    self.current_frame = frame
                    self.frame_available = True
                
                # Nhường CPU cho các thread khác
                # Điều chỉnh thời gian sleep tùy thuộc vào loại luồng
                sleep_time = 0.001 if not self.is_rtsp else 0.005
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Lỗi trong thread đọc frame: {e}")
                logger.error(traceback.format_exc())
                time.sleep(0.1)  # Tránh vòng lặp vô hạn khi gặp lỗi
    
    def read(self):
        """Lấy frame mới nhất từ thread"""
        with self.lock:
            if not self.frame_available:
                return None
            return self.current_frame.copy()  # Trả về bản sao để tránh conflict

    def isOpened(self):
        """Kiểm tra xem thread có đang chạy và capture có mở không"""
        return self.running and (self.capture is not None and self.capture.isOpened())

    def release(self):
        """Dừng thread và giải phóng tài nguyên"""
        logger.info("Yêu cầu dừng thread đọc video...")
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)  # Đợi thread kết thúc tối đa 1 giây
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        logger.info("Đã dừng ThreadedVideoCapture và giải phóng tài nguyên")

# --- Kết thúc lớp đọc video bằng thread ---

class VideoProcessor:
    """
    Lớp xử lý video từ camera, thực hiện nhận diện liên tục
    (Sử dụng ThreadedVideoCapture)
    """
    def __init__(self, camera_source, schedule_id=None, schedule_type='check_in'):
        self.camera_source = camera_source
        self.schedule_id = schedule_id
        self.schedule_type = schedule_type
        self.is_running = False
        self.capture = None # Sẽ được thay bằng ThreadedVideoCapture
        self.roi = None
        self.frame_count = 0
        self.recognized_faces = {}
        self.fa = None
        

        self.load_recognition_model()
    
    def load_recognition_model(self):
        """
        Tải model nhận diện khuôn mặt
        """
        try:
            import pickle
            import dlib
            from sklearn.svm import SVC
            
            # Load model SVC
            self.svc = pickle.load(open(settings.RECOGNITION_SVC_PATH, 'rb'))
            
            # Load classes (người dùng đã train)
            self.classes = np.load(settings.RECOGNITION_CLASSES_PATH, allow_pickle=True)
            
            # Tạo face detector và shape predictor
            self.detector = dlib.get_frontal_face_detector()
            
            predictor_path = settings.RECOGNITION_PREDICTOR_PATH
            if not os.path.exists(predictor_path):
                 logger.error(f"Không tìm thấy file shape predictor tại: {predictor_path}")
                 raise FileNotFoundError(f"Shape predictor not found at {predictor_path}")
                 
            self.predictor = dlib.shape_predictor(predictor_path)
            
            # Khởi tạo FaceAligner
            self.fa = FaceAligner(self.predictor, desiredFaceWidth=settings.RECOGNITION_FACE_WIDTH)
            
            logger.info("Model nhận diện (detector, predictor, aligner, svc, classes) đã tải")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải model nhận diện: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def set_roi(self, x, y, w, h):
        """
        Thiết lập vùng ROI cho camera
        """
        self.roi = (x, y, w, h)
    
    def start_continuous_recognition(self):
        """
        Bắt đầu quá trình nhận diện liên tục
        """
        try:
            # Khởi tạo và bắt đầu ThreadedVideoCapture
            self.capture = ThreadedVideoCapture(self.camera_source).start()
            
            # Đợi một chút để thread đọc frame đầu tiên
            time.sleep(0.5) 
            
            # Kiểm tra lại sau khi start
            if not self.capture.isOpened(): 
                logger.error(f"ThreadedVideoCapture không thể bắt đầu cho nguồn: {self.camera_source}")
                self.log_event('error', f"Không thể bắt đầu đọc video từ: {self.camera_source}")
                return False
            
            self.is_running = True
            self.log_event('start', f"Bắt đầu quét liên tục từ nguồn {self.camera_source}")
            
            # Thực hiện quá trình nhận diện
            self.process_video_stream()
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi bắt đầu nhận diện: {str(e)}")
            logger.error(traceback.format_exc())
            self.log_event('error', f"Lỗi khi bắt đầu nhận diện: {str(e)}")
            self.stop() # Đảm bảo gọi stop để giải phóng tài nguyên
            return False
    
    def process_video_stream(self):
        """
        Xử lý luồng video từ camera
        """
        # Import ngay từ đầu để tránh import trong vòng lặp
        from recognition.models import ContinuousAttendanceSchedule
        
        # Có thể bật/tắt ROI để kiểm tra
        use_roi = True  # Đặt thành False để bỏ qua ROI
        
        if not self.capture.isOpened():
            error_msg = f"Không thể mở video source: {self.camera_source}"
            logger.error(error_msg)
            self.log_event('error', error_msg)
            return
        
        self.frame_count = 0
        self.consecutive_errors = 0  # Khởi tạo bộ đếm lỗi
        last_schedule_check_time = time.time()  # Thêm biến theo dõi thời điểm kiểm tra lịch trình cuối cùng
        
        logger.info(f"Bắt đầu xử lý luồng video từ camera {self.camera_source}")
        if use_roi and self.roi:
            logger.info(f"Sử dụng ROI: {self.roi}")
        else:
            logger.info("Không sử dụng ROI - sẽ xử lý toàn bộ frame")
        
        # Vòng lặp xử lý video
        while self.is_running:
            try:
                # Kiểm tra lịch trình mỗi 5 giây để đảm bảo dừng khi cần
                current_time = time.time()
                if self.schedule_id and current_time - last_schedule_check_time > 5:
                    try:
                        schedule = ContinuousAttendanceSchedule.objects.get(id=self.schedule_id)
                        if not schedule.is_running:
                            logger.info(f"Phát hiện lịch trình {self.schedule_id} đã được đánh dấu dừng trong DB, dừng processor...")
                            self.is_running = False
                            break
                    except Exception as schedule_check_err:
                        logger.error(f"Lỗi khi kiểm tra trạng thái lịch trình: {schedule_check_err}")
                    last_schedule_check_time = current_time

                # Đọc frame từ ThreadedVideoCapture
                frame = self.capture.read()
                
                if frame is None:
                    # Nếu không đọc được frame, log lỗi và tiếp tục
                    self.consecutive_errors += 1
                    if self.consecutive_errors % 10 == 0:  # Log mỗi 10 lần lỗi
                        logger.warning(f"Không đọc được frame từ {self.camera_source}, lỗi thứ {self.consecutive_errors}")
                    if self.consecutive_errors > 100:  # Đóng và mở lại sau 100 lỗi liên tiếp
                        logger.error(f"Quá nhiều lỗi đọc frame ({self.consecutive_errors}). Thử kết nối lại.")
                        self.capture.release()
                        self.capture = ThreadedVideoCapture(self.camera_source)
                        self.capture.start()
                        time.sleep(1)  # Chờ 1 giây trước khi thử lại
                    time.sleep(0.1)  # Ngủ ngắn trước khi thử lại
                    continue
                
                # Đọc thành công, reset bộ đếm lỗi
                self.consecutive_errors = 0
                
                # Tăng bộ đếm frame
                self.frame_count += 1
                
                # Xử lý mỗi n frame (để giảm tải CPU)
                if self.frame_count % settings.RECOGNIZE_FRAME_SKIP != 0:
                    time.sleep(0.01) 
                    continue
                
                # --- Chuẩn bị frame cho việc xử lý ---
                frame_to_process = None
                
                # Resize frame nếu quá lớn (tối ưu performance)
                orig_h, orig_w = frame.shape[:2]
                if orig_w > settings.RECOGNITION_FRAME_WIDTH:
                    scale_factor = settings.RECOGNITION_FRAME_WIDTH / orig_w
                    resized_frame = cv2.resize(frame, (settings.RECOGNITION_FRAME_WIDTH, int(orig_h * scale_factor)), interpolation=cv2.INTER_AREA)
                    resized_h, resized_w = resized_frame.shape[:2]
                    logger.debug(f"Frame được resize từ {orig_w}x{orig_h} sang {resized_w}x{resized_h}")
                else:
                    resized_frame = frame.copy()
                    scale_factor = 1.0
                
                # Áp dụng ROI nếu có và được bật
                if use_roi and self.roi:
                    rx, ry, rw, rh = self.roi
                    
                    # Tính lại ROI cho frame gốc
                    if scale_factor != 1.0:
                        # Nếu frame đã được resize, cần tính lại tọa độ cho frame gốc
                        orig_rx = int(rx / scale_factor)
                        orig_ry = int(ry / scale_factor)
                        orig_rw = int(rw / scale_factor)
                        orig_rh = int(rh / scale_factor)
                    else:
                        orig_rx, orig_ry, orig_rw, orig_rh = rx, ry, rw, rh
                    
                    # Đảm bảo ROI nằm trong frame
                    orig_rx = max(0, orig_rx)
                    orig_ry = max(0, orig_ry)
                    orig_rw = min(orig_rw, orig_w - orig_rx)
                    orig_rh = min(orig_rh, orig_h - orig_ry)
                    
                    if orig_rw > 0 and orig_rh > 0:
                        try:
                            # Cắt frame theo ROI
                            frame_to_process = frame[orig_ry:orig_ry + orig_rh, orig_rx:orig_rx + orig_rw].copy()
                            if self.frame_count % 100 == 0:  # Log mỗi 100 frame 
                                logger.info(f"Áp dụng ROI: ({orig_rx}, {orig_ry}, {orig_rw}, {orig_rh})")
                        except Exception as e:
                            logger.error(f"Lỗi khi cắt ROI: {e}")
                            frame_to_process = None
                    else:
                        logger.warning(f"ROI không hợp lệ: ({orig_rx}, {orig_ry}, {orig_rw}, {orig_rh})")
                        frame_to_process = None
                else:
                    # Không áp dụng ROI, sử dụng toàn bộ frame
                    frame_to_process = frame.copy()
                    if self.frame_count % 100 == 0:  # Log mỗi 100 frame
                        logger.info("Xử lý toàn bộ frame (không có ROI)")
                
                # Kiểm tra frame được xử lý có hợp lệ không
                if frame_to_process is None or frame_to_process.size == 0:
                    logger.warning("Frame được xử lý không hợp lệ hoặc rỗng, bỏ qua frame này")
                    continue
                
                # Nhận diện khuôn mặt
                self.recognize_faces(frame_to_process)
                
                # Đợi một khoảng thời gian nhỏ trước khi xử lý frame tiếp theo
                time.sleep(0.01)  # 10ms
            
            except Exception as e:
                logger.error(f"Lỗi trong quá trình xử lý video: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(0.5)  # Đợi 0.5 giây trước khi tiếp tục
        
        logger.info("Kết thúc xử lý luồng video")
        
    def recognize_faces(self, frame):
        """
        Xử lý frame để nhận diện khuôn mặt
        """
        # Kiểm tra frame
        if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            logger.warning("Frame không hợp lệ, bỏ qua")
            return []
        
        # Xử lý frame
        try:
            # Chuyển sang thang màu xám
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Tăng cường độ tương phản
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Phát hiện khuôn mặt
            faces = self.detector(gray, 0)
            
            if len(faces) > 0:
                logger.info(f"Đã phát hiện {len(faces)} khuôn mặt trong frame")
            
            # Danh sách người đã nhận diện
            recognized_users = []
            
            # Xử lý từng khuôn mặt phát hiện được
            for i, face in enumerate(faces):
                try:
                    # Lấy tọa độ hình chữ nhật
                    #(fx, fy, fw, fh) = face_utils.rect_to_bb(face)
                    
                    # Căn chỉnh khuôn mặt cho nhận diện - quan trọng để tăng độ chính xác
                    face_aligned = self.fa.align(frame, gray, face)
                    
                    # Kiểm tra face_aligned có hợp lệ không
                    if face_aligned is None or face_aligned.size == 0:
                        logger.warning(f"Căn chỉnh khuôn mặt {i} thất bại, bỏ qua")
                        continue
                            
                    # Nhận diện khuôn mặt - sử dụng phương thức đúng từ hàm predict của video_roi_processor
                    try:
                        # Mã hóa khuôn mặt - sử dụng model='hog' giống như trong video_roi_processor
                        face_locations = face_recognition.face_locations(face_aligned, model='hog')
                        if not face_locations:
                            logger.debug(f"Không tìm thấy khuôn mặt trong ảnh đã căn chỉnh, bỏ qua")
                            continue
                            
                        face_encodings = face_recognition.face_encodings(face_aligned, known_face_locations=face_locations)
                        if not face_encodings:
                            logger.debug(f"Không thể mã hóa khuôn mặt, bỏ qua")
                            continue
                            
                        # Dự đoán từ face encoding
                        face_encoding_array = np.zeros((1, 128))
                        face_encoding_array[0] = face_encodings[0]
                        
                        # Lấy xác suất từ SVC
                        probabilities = self.svc.predict_proba(face_encoding_array)
                        best_class_index = np.argmax(probabilities[0])
                        best_class_probability = probabilities[0][best_class_index]
                        
                        # Kiểm tra xác suất so với ngưỡng
                        if best_class_probability >= settings.RECOGNITION_PREDICTION_THRESHOLD:
                            result = [best_class_index]
                            confidence = best_class_probability
                        else:
                            result = [-1]
                            confidence = best_class_probability
                    except Exception as predict_err:
                        logger.error(f"Lỗi khi dự đoán khuôn mặt: {predict_err}")
                        logger.error(traceback.format_exc())
                        result = [-1]
                        confidence = 0.0
                    
                    # Log kết quả nhận diện
                    conf_value = float(confidence)
                    if result[0] != -1:
                        # Nhận diện thành công
                        username = self.classes[result[0]]
                        recognized_users.append(username)
                        logger.info(f"[PREDICT] Khuôn mặt {i}: Nhận diện '{username}' - Độ tin cậy: {conf_value:.4f}")
                    else:
                        # Không nhận diện được
                        if conf_value > 0:
                            logger.info(f"[PREDICT] Khuôn mặt {i}: KHÔNG NHẬN DIỆN - Độ tin cậy thấp: {conf_value:.4f}")
                        else:
                            logger.info(f"[PREDICT] Khuôn mặt {i}: KHÔNG NHẬN DIỆN")
                    
                    # Xử lý kết quả nhận diện nếu thành công
                    if result[0] != -1:
                        username = self.classes[result[0]]
                        
                        # Cập nhật bộ đếm nhận diện cho người dùng
                        if username in self.recognized_faces:
                            self.recognized_faces[username] += 1
                        else:
                            self.recognized_faces[username] = 1
                        
                        # Xác định ngưỡng nhận diện theo loại chấm công
                        threshold = (
                            settings.RECOGNITION_CHECK_IN_THRESHOLD 
                            if self.schedule_type == 'check_in' 
                            else settings.RECOGNITION_CHECK_OUT_THRESHOLD
                        )
                        
                        # Log bộ đếm
                        logger.info(f"[COUNTER] '{username}': {self.recognized_faces[username]}/{threshold} frame")
                        
                        # Nếu đạt ngưỡng, thực hiện chấm công
                        if self.recognized_faces[username] >= threshold:
                            logger.info(f"===== ĐẠT NGƯỠNG CHO {username} =====")
                            # Truyền dữ liệu ảnh đã căn chỉnh vào process_attendance
                            self.process_attendance(username, face_image_data=face_aligned)
                            self.recognized_faces[username] = 0  # Reset counter sau khi xử lý
                
                except Exception as face_err:
                    logger.error(f"Lỗi xử lý khuôn mặt {i}: {face_err}")
                    logger.error(traceback.format_exc())
            
            # Reset bộ đếm nếu không còn phát hiện khuôn mặt
            if not recognized_users:
                if len(faces) == 0:
                    # Reset tất cả bộ đếm nếu không có khuôn mặt nào
                    if self.recognized_faces:
                        logger.info("Không phát hiện khuôn mặt, reset tất cả bộ đếm")
                        self.recognized_faces = {}
                else:
                    # Reset bộ đếm cho những người không xuất hiện trong frame
                    users_to_reset = set(self.recognized_faces.keys()) - set(recognized_users)
                    if users_to_reset:
                        logger.info(f"Reset bộ đếm cho {len(users_to_reset)} người không còn xuất hiện")
                        for user in users_to_reset:
                            if user in self.recognized_faces:
                                del self.recognized_faces[user]

            return recognized_users

        except Exception as e:
            logger.error(f"Lỗi trong nhận diện khuôn mặt: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def process_attendance(self, username, face_image_data=None):
        """
        Xử lý chấm công cho người dùng đã được nhận diện

        Args:
            username (str): Tên người dùng.
            face_image_data (np.ndarray, optional): Dữ liệu ảnh khuôn mặt đã căn chỉnh (numpy array). Mặc định là None.
        """
        try:
            if not self.is_running:
                logger.warning(f"Processor đã dừng, bỏ qua chấm công cho {username}")
                return None
            
            # Đánh dấu thời điểm bắt đầu xử lý
            start_time = time.time()
            
            # Tìm user trong hệ thống
            try:
                user = User.objects.get(username=username)
                logger.info(f"Tìm thấy user {username} (ID: {user.id}) trong cơ sở dữ liệu")
            except User.DoesNotExist:
                logger.warning(f"User {username} không tồn tại. Cố gắng tạo mới.")
                try:
                    # Tự động tạo user nếu không tồn tại (giống như video_roi_processor)
                    user = User.objects.create_user(
                        username=username,
                        password=f"default_{username}", 
                        first_name=username
                    )
                    logger.info(f"Đã tạo mới người dùng '{username}' trong cơ sở dữ liệu.")
                except Exception as create_err:
                    logger.error(f"Không thể tạo người dùng mới: {create_err}")
                    self.log_event('error', f"Lỗi: Không thể tạo người dùng mới {username}: {create_err}")
                    return
            
            # Tìm camera name nếu có schedule_id
            camera_name = None
            if self.schedule_id:
                try:
                    schedule = ContinuousAttendanceSchedule.objects.get(id=self.schedule_id)
                    camera_name = schedule.camera.name
                    logger.info(f"Chấm công bởi camera: {camera_name}")
                except Exception as cam_err:
                    logger.warning(f"Không thể xác định tên camera: {cam_err}")
            
            # Xử lý chấm công
            now = django_timezone.now()
            today = now.date()
            attendance_record = None
            message = ""
            
            try:
                # Chuẩn hóa tên file
                sanitized_username = self.sanitize_filename(username)
                
                if self.schedule_type == 'check_in':
                    # Xử lý chấm công vào
                    logger.info(f"Thực hiện chấm công VÀO cho {username}")
                    
                    # Tìm hoặc tạo bản ghi chấm công
                    record, created = AttendanceRecord.objects.get_or_create(
                        user=user,
                        date=today,
                        defaults={
                            'check_in': now,
                            'recognized_by_camera': camera_name
                        }
                    )
                    
                    if created:
                        # Nếu bản ghi mới, lưu ảnh check-in
                        logger.info(f"Đã tạo bản ghi chấm công mới cho {username}")
                        
                        # Lưu ảnh khuôn mặt (nếu có)
                        try:
                            # Sử dụng face_image_data được truyền vào
                            if face_image_data is not None and face_image_data.size > 0:
                                # Tạo tên file dựa trên username và thời gian
                                face_filename = f'{sanitized_username}_{today}_{now.strftime("%H%M%S")}.jpg'
                                face_path = os.path.join(settings.MEDIA_ROOT, settings.RECOGNITION_ATTENDANCE_FACES_DIR, 
                                                       settings.RECOGNITION_CHECK_IN_SUBDIR, face_filename)
                                relative_face_path = os.path.join(settings.RECOGNITION_ATTENDANCE_FACES_DIR, 
                                                               settings.RECOGNITION_CHECK_IN_SUBDIR, face_filename)
                                
                                # Đảm bảo thư mục tồn tại
                                os.makedirs(os.path.dirname(face_path), exist_ok=True)
                                
                                # Lưu ảnh và cập nhật đường dẫn vào record
                                saved_img = cv2.imwrite(face_path, face_image_data)
                                if saved_img:
                                    record.check_in_image_url = relative_face_path
                                    logger.info(f"Đã lưu ảnh check-in cho {username}")
                                else:
                                    logger.error(f"Không thể lưu ảnh check-in")
                            else:
                                 logger.warning(f"Không có dữ liệu ảnh để lưu cho check-in của {username}")
                        except Exception as img_err:
                            logger.error(f"Lỗi khi lưu ảnh check-in: {img_err}")
                        
                        try:
                            # Chuyển đổi thời gian check-in từ UTC sang múi giờ địa phương
                            local_check_in_time = django_timezone.localtime(record.check_in) if record.check_in else None
                            check_in_time = local_check_in_time.strftime('%H:%M:%S') if local_check_in_time else "N/A"
                            
                            # Hiển thị thêm thông tin thời gian hiện tại và múi giờ để debug
                            now_local = django_timezone.localtime(now)
                            logger.info(f"DEBUG TIME: DB time={record.check_in}, local time={local_check_in_time}, current time={now_local}")
                        except Exception as e:
                            logger.warning(f"Lỗi khi chuyển đổi múi giờ check-in: {e}")
                            check_in_time = record.check_in.strftime('%H:%M:%S') if record.check_in else "N/A"
                        
                        message = f"Đã nhận diện và chấm công VÀO cho {username}"
                    else:
                        # Nếu đã tồn tại, cập nhật check_in nếu chưa có
                        if not record.check_in:
                            record.check_in = now
                            if not record.recognized_by_camera and camera_name:
                                record.recognized_by_camera = camera_name
                            message = f"Đã cập nhật check-in cho {username}"
                        else:
                            # Đã có check-in, không cập nhật
                            try:
                                # Chuyển đổi thời gian check-in từ UTC sang múi giờ địa phương
                                local_check_in_time = django_timezone.localtime(record.check_in) if record.check_in else None
                                check_in_time = local_check_in_time.strftime('%H:%M:%S') if local_check_in_time else "N/A"
                                
                                # Hiển thị thêm thông tin thời gian hiện tại và múi giờ để debug
                                now_local = django_timezone.localtime(now)
                                logger.info(f"DEBUG TIME: DB time={record.check_in} ({record.check_in.tzinfo}), local time={local_check_in_time} ({local_check_in_time.tzinfo}), current time={now_local} ({now_local.tzinfo})")
                            except Exception as e:
                                logger.warning(f"Lỗi khi chuyển đổi múi giờ check-in: {e}")
                                check_in_time = record.check_in.strftime('%H:%M:%S') if record.check_in else "N/A"
                            
                            message = f"{username} đã check-in từ lúc {check_in_time}"
                            logger.info(message)
                            self.log_event('recognition', message, user, record)
                            return record
                else:
                    # Xử lý chấm công ra
                    logger.info(f"Thực hiện chấm công RA cho {username}")
                    
                    try:
                        # Tìm bản ghi chấm công hiện có
                        record = AttendanceRecord.objects.get(user=user, date=today)
                        
                        # Đã có bản ghi, cập nhật check-out
                        record.check_out = now
                        if not record.recognized_by_camera and camera_name:
                            record.recognized_by_camera = camera_name
                        
                        # Lưu ảnh khuôn mặt check-out (nếu có)
                        try:
                            # Sử dụng face_image_data được truyền vào
                            if face_image_data is not None and face_image_data.size > 0:
                                # Tạo tên file chỉ dựa trên username và ngày (không thêm timestamp)
                                face_filename = f'{sanitized_username}_{today}_out.jpg'
                                face_path = os.path.join(settings.MEDIA_ROOT, settings.RECOGNITION_ATTENDANCE_FACES_DIR, 
                                               settings.RECOGNITION_CHECK_OUT_SUBDIR, face_filename)
                                relative_face_path = os.path.join(settings.RECOGNITION_ATTENDANCE_FACES_DIR, 
                                                               settings.RECOGNITION_CHECK_OUT_SUBDIR, face_filename)
                                
                                # Không cần tạo thư mục nữa vì đã tạo trong __init__
                                # os.makedirs(os.path.dirname(face_path), exist_ok=True)
                                
                                # Giảm kích thước ảnh và chất lượng để tăng tốc độ lưu
                                # Giảm kích thước xuống một nửa
                                face_height, face_width = face_image_data.shape[:2]
                                face_resized = cv2.resize(face_image_data, (face_width//2, face_height//2), 
                                               interpolation=cv2.INTER_AREA)
                                
                                # Lưu ảnh với chất lượng thấp hơn
                                saved_img = cv2.imwrite(face_path, face_resized, 
                                              [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                                
                                if saved_img:
                                    record.check_out_image_url = relative_face_path
                                    logger.info(f"Đã lưu/cập nhật ảnh check-out cho {username}")
                                else:
                                    logger.error(f"Không thể lưu ảnh check-out")
                            else:
                                logger.warning(f"Không có dữ liệu ảnh để lưu cho check-out của {username}")
                        except Exception as img_err:
                            logger.error(f"Lỗi khi lưu ảnh check-out: {img_err}")
                        
                        #check_in_time_str = record.check_in.strftime('%H:%M:%S') if record.check_in else "chưa có"
                        #check_out_time_str = now.strftime('%H:%M:%S')
                        time_check_out = datetime.now()
                        message = f"Đã nhận diện và chấm công RA cho {username} check-out: {time_check_out})"
                    except AttendanceRecord.DoesNotExist:
                        # Chưa có bản ghi nào, tạo mới với check-out
                        record = AttendanceRecord.objects.create(
                            user=user,
                            date=today,
                            check_out=now,
                            recognized_by_camera=camera_name
                        )
                        message = f"Đã nhận diện và tạo mới bản ghi check-out cho {username}"
                        logger.info(f"Đã tạo bản ghi mới với check-out cho {username}")
                
                # Lưu bản ghi
                record.save()
                attendance_record = record
                logger.info(f"Đã lưu bản ghi chấm công cho {username}")
                
                # Cố gắng đẩy dữ liệu lên Firebase nếu có
                try:
                    from .firebase_util import push_attendance_to_firebase
                    # Ghi lại thời gian bắt đầu đẩy Firebase
                    firebase_start_time = time.time()
                    success = push_attendance_to_firebase(record, camera_name)
                    firebase_time = time.time() - firebase_start_time
                    
                    if success:
                        logger.info(f"Đã đẩy dữ liệu chấm công lên Firebase (thời gian: {firebase_time:.2f}s)")
                    else:
                        logger.warning(f"Không thể đẩy dữ liệu lên Firebase")
                except ImportError:
                    logger.debug("Module firebase_util không khả dụng, bỏ qua đồng bộ")
                except Exception as firebase_err:
                    logger.error(f"Lỗi khi đẩy dữ liệu lên Firebase: {firebase_err}")
            
                # Ghi log tổng thời gian xử lý
                total_time = time.time() - start_time
                logger.info(f"Tổng thời gian xử lý chấm công: {total_time:.2f} giây")
            
            except Exception as att_err:
                message = f"Lỗi khi cập nhật chấm công: {str(att_err)}"
                logger.error(message)
                logger.error(traceback.format_exc())
                self.log_event('error', message, user)
                return
            
            # Lưu nhật ký hoạt động
            try:
                self.log_event('recognition', message, user, attendance_record)
                logger.info(f"Đã lưu nhật ký cho {username}")
            except Exception as log_err:
                logger.error(f"Lỗi khi lưu nhật ký: {str(log_err)}")
            
            logger.info(f"===== KẾT THÚC XỬ LÝ CHẤM CÔNG CHO {username} =====")
            return attendance_record
            
        except Exception as e:
            err_msg = f"Lỗi không xác định khi xử lý chấm công cho {username}: {str(e)}"
            logger.error(err_msg)
            logger.error(traceback.format_exc())
            self.log_event('error', err_msg)
            return None
    
    def sanitize_filename(self, filename):
        """
        Chuẩn hóa tên file: bỏ dấu, thay ký tự đặc biệt bằng '_'
        """
        if not filename:
            return ""
        
        try:
            from unidecode import unidecode
            import re
            
            # Bỏ dấu
            sanitized = unidecode(filename)
            # Thay ký tự đặc biệt bằng dấu gạch dưới
            sanitized = re.sub(r'[^\w\-]+', '_', sanitized)
            # Chuyển về chữ thường và loại bỏ gạch dưới thừa
            sanitized = sanitized.lower().strip('_')
            
            if not sanitized:
                return "_"  # Giá trị mặc định nếu tên rỗng
                
            return sanitized
        except ImportError:
            logger.warning("Thư viện unidecode không được cài đặt, dùng sanitize đơn giản")
            import re
            # Sanitize đơn giản
            sanitized = re.sub(r'[^\w\-]+', '_', filename)
            return sanitized.lower().strip('_') or "_"
        except Exception as e:
            logger.error(f"Lỗi khi sanitize filename: {e}")
            return f"user_{hash(filename) % 10000}"  # Fallback
    
    def log_event(self, event_type, message, user=None, attendance_record=None):
        """
        Ghi nhật ký sự kiện liên quan đến lịch trình
        
        Args:
            event_type: Loại sự kiện ('start', 'stop', 'recognition', 'error')
            message: Nội dung thông báo
            user: User đã được nhận diện (nếu có)
            attendance_record: Bản ghi chấm công liên quan (nếu có)
        """
        try:
            if not self.schedule_id:
                logger.warning(f"Không thể log sự kiện '{event_type}' do không có schedule_id")
                return
                
            # Tìm lịch trình
            try:
                from .models import ContinuousAttendanceSchedule, ContinuousAttendanceLog
                schedule = ContinuousAttendanceSchedule.objects.get(id=self.schedule_id)
                logger.info(f"Log sự kiện '{event_type}' cho lịch trình {schedule.name}")
            except ContinuousAttendanceSchedule.DoesNotExist:
                logger.error(f"Không tìm thấy lịch trình với ID={self.schedule_id}")
                return
            except Exception as sch_err:
                logger.error(f"Lỗi khi tìm lịch trình với ID={self.schedule_id}: {sch_err}")
                return
            
            # Tạo log cho sự kiện
            try:
                logger.info(f"Đang lưu log: {event_type} - {message}")
                log_entry = ContinuousAttendanceLog(
                    schedule=schedule,
                    event_type=event_type,
                    message=message,
                    recognized_user=user,
                    attendance_record=attendance_record
                )
                log_entry.save()
                logger.info(f"Đã lưu log ID={log_entry.id}")
                return log_entry
            except Exception as save_err:
                logger.error(f"Lỗi khi lưu log: {save_err}")
                logger.error(traceback.format_exc())
                
        except Exception as e:
            logger.error(f"Lỗi không xác định trong log_event: {e}")
            logger.error(traceback.format_exc())
    
    def stop(self):
        """
        Dừng quá trình nhận diện và giải phóng tài nguyên
        """
        if not self.is_running:
            return
        
        logger.info("Dừng VideoProcessor...")
        self.is_running = False
        time.sleep(0.5)
        # Gọi release cho threaded video capture
        if self.capture:
            self.capture.release()
            self.capture = None
        
        self.log_event('stop', "Đã dừng quét liên tục")
        logger.info("VideoProcessor đã dừng hoàn toàn")


@shared_task(bind=True)
def start_continuous_recognition(self, schedule_id):
    """
    Task Celery để bắt đầu nhận diện liên tục cho một lịch trình
    """
    if schedule_id in active_processors:
        logger.warning(f"Processor đã tồn tại cho lịch trình {schedule_id}, dừng processor cũ trước khi tạo mới")
        old_processor = active_processors[schedule_id]
        old_processor.stop()
        time.sleep(1)  # Chờ processor dừng
        del active_processors[schedule_id]
        
    logger.info(f"TASK start_continuous_recognition BẮT ĐẦU cho schedule_id: {schedule_id}, Worker Task ID: {self.request.id}")
    
    try:
        # Lấy thông tin lịch trình
        logger.info(f"Đang lấy thông tin lịch trình {schedule_id} từ DB...")
        schedule = ContinuousAttendanceSchedule.objects.get(id=schedule_id)
        logger.info(f"Đã lấy lịch trình: {schedule.name}")
        
        # Kiểm tra trạng thái *chỉ khi không phải là test run*
        if schedule.status != 'active': 
            logger.warning(f"Lịch trình {schedule_id} không hoạt động (status={schedule.status}), bỏ qua")
            return False
        
        # Lấy camera source
        camera_source = schedule.camera.source
        
        # Kiểm tra xem camera đã có người dùng chưa (khóa này chỉ trong phạm vi tiến trình)
        if camera_source in camera_locks:
            logger.warning(f"Camera {camera_source} đang được sử dụng bởi một lịch trình khác")
            

        # Kiểm tra xem lịch trình này đã đang chạy chưa (trong DB)
        if schedule.is_running:
            # Nếu là test run và đang chạy -> có thể là lỗi từ lần test trước chưa dừng hẳn
            if is_test_run:
                logger.warning(f"Lịch trình {schedule_id} đã is_running=True khi bắt đầu test. Sẽ thử dừng lịch trình cũ trước.")
                try:
                    # Cố gắng dừng lịch trình cũ
                    if schedule_id in active_processors:
                        logger.info(f"Tìm thấy processor cũ cho {schedule_id}, dừng lại trước")
                        old_processor = active_processors[schedule_id]
                        old_processor.stop()
                        del active_processors[schedule_id]
                except Exception as e:
                    logger.error(f"Lỗi khi dừng processor cũ: {e}")
                
                # Reset is_running trong DB
                schedule.is_running = False
                schedule.worker_id = None
                schedule.save(update_fields=['is_running', 'worker_id'])
            else: 
                # Nếu là chạy thường và đang chạy
                logger.warning(f"Lịch trình {schedule_id} đã is_running=True, kiểm tra worker_id")
                
                # Kiểm tra worker_id để xác định xem task đang chạy có còn active không
                if schedule.worker_id:
                    # Đối với bản production cần kiểm tra worker_id với Celery inspector
                    # Ở đây đơn giản hóa bằng cách kiểm tra xem processor có trong biến active_processors không
                    if schedule_id in active_processors:
                        logger.info(f"Lịch trình {schedule_id} đang chạy với processor trong active_processors, bỏ qua")
                        return False
                    else:
                        logger.warning(f"Lịch trình {schedule_id} đánh dấu là running nhưng không có processor, reset trạng thái")
                        schedule.is_running = False
                        schedule.worker_id = None
                        schedule.save(update_fields=['is_running', 'worker_id'])
                else:
                    logger.warning(f"Lịch trình {schedule_id} đánh dấu là running nhưng không có worker_id, reset trạng thái")
                    schedule.is_running = False
                    schedule.save(update_fields=['is_running'])

        # Tạo processor mới
        logger.info(f"Tạo VideoProcessor cho camera {camera_source}, type: {schedule.schedule_type}")
        processor = VideoProcessor(
            camera_source=camera_source,
            schedule_id=schedule_id,
            schedule_type=schedule.schedule_type
        )
        
        # Thiết lập ROI nếu có
        roi = schedule.camera.get_roi_tuple()
        if roi:
            logger.info(f"Thiết lập ROI: {roi}")
            processor.set_roi(*roi)
        else:
            logger.info(f"Không có ROI được cấu hình cho camera.")
        
        # Đánh dấu camera đang được sử dụng
        camera_locks[camera_source] = schedule_id
        
        # Cập nhật trạng thái lịch trình *trước* khi bắt đầu processor
        logger.info(f"Cập nhật DB: is_running=True, worker_id={self.request.id} cho schedule {schedule_id}")
        schedule.is_running = True
        schedule.worker_id = self.request.id
        schedule.save(update_fields=['is_running', 'worker_id']) # Chỉ cập nhật trường cần thiết
        logger.info(f"Đã cập nhật DB.")
        
        # Lưu processor vào dictionary toàn cục
        active_processors[schedule_id] = processor
        logger.info(f"Đã lưu processor vào active_processors (key={schedule_id})")
        
        # Bắt đầu nhận diện
        logger.info(f"Gọi processor.start_continuous_recognition()...")
        result = processor.start_continuous_recognition()
        logger.info(f"processor.start_continuous_recognition() trả về: {result}")

        # Nếu processor kết thúc (có thể do lỗi hoặc hoàn thành), xóa khỏi active_processors
        if schedule_id in active_processors and active_processors[schedule_id] == processor:
            logger.info(f"Xóa processor khỏi active_processors sau khi kết thúc.")
            del active_processors[schedule_id]
            
            # Giải phóng khóa camera
            if camera_source in camera_locks and camera_locks[camera_source] == schedule_id:
                logger.info(f"Giải phóng khóa camera {camera_source}")
                del camera_locks[camera_source]
            
            # Cập nhật lại DB nếu processor dừng mà chưa có task stop nào gọi
            try:
                # Cần lấy lại schedule object vì có thể nó đã thay đổi
                schedule_final = ContinuousAttendanceSchedule.objects.get(id=schedule_id)
                if schedule_final.is_running and schedule_final.worker_id == self.request.id:
                     logger.warning(f"Processor dừng đột ngột, cập nhật is_running=False cho schedule {schedule_id}")
                     schedule_final.is_running = False
                     schedule_final.worker_id = None
                     schedule_final.save(update_fields=['is_running', 'worker_id'])
            except Exception as db_err:
                 logger.error(f"Lỗi khi cập nhật DB sau khi processor dừng: {db_err}")

        logger.info(f"TASK start_continuous_recognition KẾT THÚC cho schedule_id: {schedule_id}")
        return result
    
    except ContinuousAttendanceSchedule.DoesNotExist:
        logger.error(f"Lịch trình {schedule_id} không tồn tại")
        return False
    except Exception as e:
        logger.error(f"LỖI không xác định trong start_continuous_recognition: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Cố gắng cập nhật trạng thái lịch trình về False nếu có lỗi
        try:
            # Lấy lại schedule object
            schedule_err = ContinuousAttendanceSchedule.objects.get(id=schedule_id)
            
            # Giải phóng khóa camera nếu có
            camera_source = schedule_err.camera.source
            if camera_source in camera_locks and camera_locks[camera_source] == schedule_id:
                logger.info(f"Giải phóng khóa camera {camera_source} do lỗi")
                del camera_locks[camera_source]
            
            if schedule_err.is_running and schedule_err.worker_id == self.request.id: # Chỉ cập nhật nếu task này đang quản lý
                 schedule_err.is_running = False
                 schedule_err.worker_id = None
                 schedule_err.save(update_fields=['is_running', 'worker_id'])
                 logger.info(f"Đã cập nhật is_running=False do lỗi.")
            
                 # Ghi log lỗi vào DB
                 # Cần tạo instance tạm để ghi log nếu processor chưa kịp tạo hoặc đã bị xóa
                 if schedule_id not in active_processors: 
                     processor_temp = VideoProcessor(schedule_err.camera.source, schedule_id) 
                     processor_temp.log_event('error', f"Lỗi nghiêm trọng khi bắt đầu: {str(e)}")
                 else:
                      active_processors[schedule_id].log_event('error', f"Lỗi nghiêm trọng khi bắt đầu: {str(e)}")

        except Exception as db_err:
            logger.error(f"Lỗi khi cập nhật DB hoặc ghi log lỗi: {db_err}")
        
        logger.info(f"TASK start_continuous_recognition KẾT THÚC (trong finally) cho schedule_id: {schedule_id}") 


@shared_task
def stop_continuous_recognition(schedule_id):
    """
    Task Celery để dừng nhận diện liên tục cho một lịch trình
    """
    logger.info(f"Dừng nhận diện liên tục cho lịch trình {schedule_id}")
    max_attempts = 3
    success = False
    
    try:
        # Lấy thông tin lịch trình để biết camera_source
        schedule = ContinuousAttendanceSchedule.objects.get(id=schedule_id)
        camera_source = schedule.camera.source
        
        # Lấy processor từ dictionary toàn cục
        processor = active_processors.get(schedule_id)
        
        # Nếu có processor đang chạy, dừng lại
        if processor:
            logger.info(f"Tìm thấy processor cho lịch trình {schedule_id}, đang tiến hành dừng...")
            processor.stop()
            
            # Kiểm tra lặp lại để đảm bảo processor đã dừng hoàn toàn
            for attempt in range(max_attempts):
                time.sleep(1)  # Đợi 1 giây
                if not processor.is_running:
                    logger.info(f"Processor đã dừng thành công sau {attempt+1} lần kiểm tra")
                    success = True
                    break
                else:
                    logger.warning(f"Processor vẫn đang chạy sau lần thử {attempt+1}, thử lại...")
                    # Gọi stop() lần nữa
                    processor.stop()
            
            # Sau khi kiểm tra, xóa processor khỏi dictionary
            if schedule_id in active_processors:
                logger.info(f"Xóa processor khỏi active_processors")
                del active_processors[schedule_id]
            
            # Giải phóng khóa camera
            if camera_source in camera_locks and camera_locks[camera_source] == schedule_id:
                logger.info(f"Giải phóng khóa camera {camera_source} khi dừng lịch trình {schedule_id}")
                del camera_locks[camera_source]
        else:
            logger.warning(f"Không tìm thấy processor cho lịch trình {schedule_id} trong active_processors")
            success = True  # Coi như thành công vì không cần dừng
        
        # Cập nhật trạng thái lịch trình
        schedule.is_running = False
        schedule.worker_id = None
        schedule.save(update_fields=['is_running', 'worker_id'])
        logger.info(f"Đã cập nhật is_running=False và worker_id=None cho lịch trình {schedule_id}")
        
        # Ghi log
        ContinuousAttendanceLog.objects.create(
            schedule=schedule,
            event_type='stop',
            message=f"Đã dừng nhận diện liên tục theo lịch trình"
        )
        logger.info(f"Đã ghi log dừng cho lịch trình {schedule_id}")
        
        # Kiểm tra lại một lần nữa sau khi đã cập nhật DB để đảm bảo
        if not success and schedule_id in active_processors:
            logger.warning(f"Processor vẫn còn trong active_processors sau tất cả các nỗ lực, buộc xóa")
            del active_processors[schedule_id]
        
        return True
    
    except ContinuousAttendanceSchedule.DoesNotExist:
        logger.error(f"Lịch trình {schedule_id} không tồn tại")
        return False
    except Exception as e:
        logger.error(f"Lỗi khi dừng nhận diện liên tục: {str(e)}")
        logger.error(traceback.format_exc())
        return False


@shared_task
def schedule_continuous_recognition():
    """
    Task Celery chạy định kỳ để kiểm tra và khởi động/dừng các lịch trình.
    """
    logger.info("===== [Scheduler] BẮT ĐẦU KIỂM TRA LỊCH TRÌNH =====")
    
    try:
        # Lấy thời gian hiện tại với múi giờ chính xác
        now_utc = django_timezone.now()  # Lấy thời gian hiện tại theo UTC
        now_local = django_timezone.localtime(now_utc)  # Chuyển sang múi giờ địa phương (settings.TIME_ZONE)
        
        # Sử dụng thời gian địa phương để so sánh với thời gian lịch trình
        current_time = now_local.time()  # Đây là thời gian sẽ dùng để so sánh
        current_day = str(now_local.isoweekday())  # 1-7 (1 là thứ Hai)
        
        # Log rõ ràng cả hai múi giờ để dễ debug
        logger.info(f"[Scheduler] Thời gian UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"[Scheduler] Thời gian địa phương: {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"[Scheduler] Thời gian so sánh: {current_time}, Ngày: {current_day}")
        
        # Lấy lịch trình active
        schedules = ContinuousAttendanceSchedule.objects.filter(status='active')
        logger.info(f"[Scheduler] Tìm thấy {schedules.count()} lịch trình active")
        
        for schedule in schedules:
            # QUAN TRỌNG: In ra log chi tiết cho phân tích
            # Kiểm tra điều kiện thời gian từng phần
            is_after_start = current_time >= schedule.start_time
            is_before_end = current_time <= schedule.end_time
            
            # Xử lý trường hợp lịch qua đêm (end_time < start_time)
            if schedule.end_time < schedule.start_time:
                # Nếu end < start, thì có 2 khoảng thời gian hợp lệ:
                # 1. Từ start_time đến cuối ngày (23:59:59)
                # 2. Từ đầu ngày (00:00:00) đến end_time
                is_after_start = current_time >= schedule.start_time or current_time <= schedule.end_time
                is_before_end = current_time >= schedule.start_time or current_time <= schedule.end_time
                
                # Log giải thích thêm cho trường hợp đặc biệt này
                logger.info(f"[Scheduler] - Lịch qua đêm được phát hiện (end < start)")
            
            in_time_range = is_after_start and is_before_end
            # Log tương tự đến file
            logger.info(f"[Scheduler] Lịch ID {schedule.id}: {schedule.name}")
            logger.info(f"[Scheduler] - Thời gian: {schedule.start_time} đến {schedule.end_time}")
            logger.info(f"[Scheduler] - Current time (địa phương): {current_time}")
            logger.info(f"[Scheduler] - Current day: {current_day} in active days: {schedule.active_days}")
            logger.info(f"[Scheduler] - Điều kiện ngày: {current_day in schedule.active_days.split(',')}")
            logger.info(f"[Scheduler] - After start: {is_after_start}, Before end: {is_before_end}")
            logger.info(f"[Scheduler] - In time range: {in_time_range}, Running: {schedule.is_running}")
            
            # Kiểm tra ngày trong tuần
            if current_day in schedule.active_days.split(','):
                # SỬA ĐỔI LOGIC: Sử dụng in_time_range đã tính ở trên
                if in_time_range and not schedule.is_running:
                    # Bắt đầu nhận diện
                    logger.info(f"[Scheduler] >>> BẮT ĐẦU lịch trình {schedule.id} - {schedule.name}")
                    start_continuous_recognition.delay(schedule.id)
                
                # PHẦN QUAN TRỌNG NHẤT - Kiểm tra điều kiện dừng
                # Nếu không trong time range và đang chạy -> dừng
                elif not in_time_range and schedule.is_running:
                    # Dừng nhận diện
                    logger.info(f"[Scheduler] <<< DỪNG lịch trình {schedule.id} - {schedule.name} (Ngoài giờ)")
                    stop_continuous_recognition.delay(schedule.id)
                else:
                    status = "đang chạy đúng" if schedule.is_running and in_time_range else \
                             "đang dừng đúng" if not schedule.is_running and not in_time_range else \
                             "cần START" if not schedule.is_running and in_time_range else \
                             "cần STOP" if schedule.is_running and not in_time_range else "không xác định"

                    logger.info(f"[Scheduler] --- Lịch trình {schedule.id}: {status}")
            else:
                if schedule.is_running:
                    logger.info(f"[Scheduler] <<< DỪNG lịch trình {schedule.id} - {schedule.name} (Không hoạt động ngày này)")
                    stop_continuous_recognition.delay(schedule.id)
        
        logger.info("===== [Scheduler] KẾT THÚC KIỂM TRA LỊCH TRÌNH =====")
        return True
    except Exception as e:
        logger.error(f"[Scheduler] !!! LỖI: {str(e)}")
        logger.error(traceback.format_exc())
        return False