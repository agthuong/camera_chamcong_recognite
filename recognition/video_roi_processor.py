import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.face_utils import FaceAligner
import numpy as np
import face_recognition
import pickle
import os
import time
from sklearn.preprocessing import LabelEncoder
from django.utils import timezone
from django.conf import settings
import threading
import re
from unidecode import unidecode
from attendance_system_facial_recognition.logger import setup_logger
from .utils.datetime_utils import get_current_time, get_current_date, format_datetime

# Thiết lập logger cho module này
logger = setup_logger(__name__, 'video_processor.log')

# --- Global state for tracking collection progress ---
# Format: { 'sanitized_username': {'current': 0, 'total': 150, 'active': False} }
# --- Chuyển về dùng username gốc làm key ---
# Format: { 'username': {'current': 0, 'total': 150, 'active': False, 'sanitized': 'sanitized_username'} }
collect_progress_tracker = {}

# --- Constants and Paths --- (Adjust if necessary)
# PREDICTOR_PATH = 'face_recognition_data/shape_predictor_68_face_landmarks.dat' # Thay bằng settings
# SVC_PATH = "face_recognition_data/svc.sav" # Thay bằng settings
# CLASSES_PATH = 'face_recognition_data/classes.npy' # Thay bằng settings
# TRAINING_DATA_DIR = 'face_recognition_data/training_dataset/' # Thay bằng settings
# FACE_WIDTH = 96 # Width for aligned faces # Thay bằng settings
# FRAME_WIDTH = 800 # Width to resize video frames # Thay bằng settings
# DEFAULT_MAX_SAMPLES = 50 # Default samples to collect # Thay bằng settings
# DEFAULT_RECOGNITION_THRESHOLD = 3 # Default consecutive detections for recognition # Thay bằng settings
# --- Thêm ngưỡng check-out --- 
# CHECK_OUT_RECOGNITION_THRESHOLD = 4 # Số lần nhận diện liên tiếp cho check-out # Thay bằng settings
# --- Kết thúc --- 
# FRAME_SKIP = 3 # Process every Nth frame # Thay bằng settings

# --- Lấy giá trị từ settings ---

class StreamOutput:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.running = False # Thêm trạng thái đang chạy

    def set_frame(self, frame):
        # Chỉ encode nếu đang có yêu cầu stream (running=True)
        if self.running:
            with self.lock:
                # Resize frame trước khi encode để giảm băng thông nếu cần
                # frame_display = imutils.resize(frame, width=640) # Ví dụ resize
                frame_display = frame # Giữ nguyên kích thước gốc nếu muốn video to
                ret, jpeg = cv2.imencode('.jpg', frame_display, [int(cv2.IMWRITE_JPEG_QUALITY), 85]) # Giảm chất lượng JPEG để tiết kiệm băng thông
                if ret:
                    self.frame = jpeg.tobytes()

    def get_frame_bytes(self):
        with self.lock:
            return self.frame

    def start_stream(self):
        self.running = True

    def stop_stream(self):
        self.running = False
        with self.lock:
            self.frame = None # Xóa frame khi dừng

# Global instance (Cân nhắc giải pháp tốt hơn cho production)
stream_output = StreamOutput()

def predict(face_aligned, svc, threshold=settings.RECOGNITION_PREDICTION_THRESHOLD):
    """
    Predicts face from aligned image using the loaded SVC model.
    Returns: Tuple (list containing predicted class index or -1, highest probability)
    """
    face_encodings_list = np.zeros((1, 128))
    try:
        # Use model='hog' which is faster. Use 'cnn' for higher accuracy if GPU available.
        x_face_locations = face_recognition.face_locations(face_aligned, model='hog')
        faces_encodings = face_recognition.face_encodings(face_aligned, known_face_locations=x_face_locations)

        if not faces_encodings: # Check if list is empty
            return ([-1], [0.0])

        # Assuming only one face encoding per aligned image
        face_encodings_list[0] = faces_encodings[0]

    except Exception as e:
        return ([-1], [0.0])

    try:
        prob = svc.predict_proba(face_encodings_list)
        best_class_index = np.argmax(prob[0])
        best_class_probability = prob[0][best_class_index]

        # --- Sử dụng ngưỡng từ settings --- 
        if best_class_probability >= threshold:
            return ([best_class_index], [best_class_probability])
        else:

            return ([-1], [best_class_probability])
            # --- Kết thúc thay đổi ---

    except Exception as e:
        # print(f"Debug: Error during prediction: {e}")
        return ([-1], [0.0])

def select_roi_from_source(video_source, frame_skip_on_next=5):
    """
    Opens the video source, allows frame navigation and ROI selection.
    Returns the ROI coordinates (x, y, w, h) or None if cancelled.
    
    Args:
        video_source: Path to video file, stream URL, or webcam ID.
        frame_skip_on_next: Number of frames to skip when 'n' is pressed.
    """
    try:
        source_int = int(video_source)
        cap = cv2.VideoCapture(source_int)
    except ValueError:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Lỗi: Không thể mở nguồn video: {video_source}")
        return None

    roi = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Option: Break or loop back if it's a stream? For now, break.
            if roi is None: # If we never selected an ROI
                print("Không có frame nào để chọn ROI.")
            break

        frame_count += 1

    # Cleanup
    cap.release()

    # roi will be None if cancelled, or the tuple (x,y,w,h) if selected
    return roi

# --- Hàm chuẩn hóa tên file/thư mục ---
def sanitize_filename(filename):
    """
    Chuẩn hóa tên file/thư mục: bỏ dấu, thay ký tự đặc biệt bằng '_', chuyển về chữ thường.
    """
    if not filename:
        return ""
    # Bỏ dấu
    sanitized = unidecode(filename)
    # Thay thế các ký tự không phải chữ/số/gạch dưới/gạch ngang bằng gạch dưới
    sanitized = re.sub(r'[^\w\-]+', '_', sanitized)
    # Chuyển về chữ thường và loại bỏ gạch dưới thừa ở đầu/cuối
    sanitized = sanitized.lower().strip('_')
    # Xử lý trường hợp tên rỗng sau khi chuẩn hóa
    if not sanitized:
        return "_" # Hoặc trả về một giá trị mặc định khác
    return sanitized
# --- Kết thúc hàm chuẩn hóa ---

# --- Xóa định nghĩa FRAME_WIDTH ở đây vì nó sẽ được lấy từ settings --- 
# FRAME_WIDTH = 800 

# Thêm class VideoSourceHandler để xử lý luồng video tốt hơn
class VideoSourceHandler:
    """
    Xử lý nguồn video trong một thread riêng để tránh việc buffer bị đầy,
    đặc biệt hữu ích cho luồng RTSP.
    """
    def __init__(self, source):
        """
        Khởi tạo VideoSourceHandler
        
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
        # Thêm thuộc tính stop_event - sẽ được gán từ bên ngoài
        self.stop_event = None

    def start(self):
        """Khởi động thread đọc frame"""
        print(f"[INFO] Khởi tạo nguồn video: {self.source}")
        self.capture = cv2.VideoCapture(self.source)
        
        # Đặt các tham số đặc biệt cho RTSP
        if self.is_rtsp:
            # Cấu hình RTSP để giảm độ trễ và tăng reliability
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giữ buffer nhỏ để luôn lấy frame mới nhất
            # Đặt Codec cho FFMPEG (backend phổ biến của OpenCV cho RTSP)
            # Không sử dụng MJPEG có thể giảm lag với một số camera
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            # Đặt thời gian timeout - có thể điều chỉnh tùy camera

            # Cho phép kết nối lại nhanh hơn
            self.max_consecutive_errors = 3  # Giảm số lỗi trước khi kết nối lại
            print(f"[INFO] Đã áp dụng cấu hình đặc biệt cho luồng RTSP/Streaming")
            
        if not self.capture.isOpened():
            error_message = f"Không thể mở nguồn video: {self.source}"
            print(f"[ERROR] {error_message}")
            return False
            
        # Tùy chọn kiểm tra kích thước frame
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        
        # Kiểm tra nếu kích thước không hợp lệ, có thể là dấu hiệu của kết nối thất bại
        if width <= 0 or height <= 0:
            error_message = f"Kích thước video không hợp lệ: {width}x{height}, có thể camera không khả dụng"
            print(f"[ERROR] {error_message}")
            self.capture.release()
            return False
            
        print(f"[INFO] Kích thước video: {width}x{height}, FPS: {fps:.1f}")
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.daemon = True  # Thread sẽ tự động kết thúc khi chương trình kết thúc
        self.thread.start()
        print(f"[INFO] Đã khởi động thread đọc frame từ nguồn: {self.source}")
        return True
        
    def stop(self):
        """Dừng thread đọc frame và giải phóng tài nguyên"""
        print(f"[VSH STOP] Bắt đầu dừng VideoSourceHandler cho nguồn: {self.source}")
        
        # Đặt cờ running = False để dừng vòng lặp _read_loop
        self.running = False
        
        # Giải phóng tài nguyên thread
        try:
            if self.thread and self.thread.is_alive():
                print(f"[VSH STOP] Đợi thread đọc frame kết thúc...")
                join_timeout = 3.0  # Giảm xuống vì đã cải thiện _read_loop với nhiều kiểm tra stop_event
                start_join_time = time.time()
                self.thread.join(timeout=join_timeout)
                end_join_time = time.time()
                join_duration = end_join_time - start_join_time
                print(f"[VSH STOP] Thời gian join thread đọc frame: {join_duration:.2f} giây.")
                
                # Kiểm tra nếu thread vẫn còn sống sau timeout
                if self.thread.is_alive():
                    print(f"[VSH STOP WARNING] Thread đọc frame vẫn đang chạy sau {join_timeout} giây timeout.")
                    # Xóa tham chiếu thread ngay cả khi vẫn chạy - THÊM MỚI
                    print("[VSH STOP] Đặt self.thread = None và tiếp tục")
                    self.thread = None
                else:
                    print("[VSH STOP] Thread đọc frame đã kết thúc.")
                    self.thread = None
            elif self.thread:
                 print("[VSH STOP] Thread đọc frame đã kết thúc trước khi gọi stop.")
                 self.thread = None
            else:
                 print("[VSH STOP] Không có thread đọc frame để dừng.")
                 
        except Exception as e:
            print(f"[VSH STOP ERROR] Lỗi khi dừng thread đọc frame: {e}")
            # Xóa tham chiếu thread ngay cả khi có lỗi - THÊM MỚI
            self.thread = None
            
        # --- Thực hiện release() trong thread riêng để tránh treo ---
        print("[VSH STOP] Bắt đầu giải phóng OpenCV capture với thread riêng...")
        capture_released = False
        
        # Tạo hàm release an toàn chạy trong thread riêng
        def safe_release():
            nonlocal capture_released
            try:
                if self.capture and self.capture.isOpened():
                    print(f"[VSH SAFE RELEASE] Gọi self.capture.release() cho nguồn: {self.source}")
                    release_start_time = time.time()
            self.capture.release()
                    release_end_time = time.time()
                    release_duration = release_end_time - release_start_time
                    print(f"[VSH SAFE RELEASE] Thời gian thực thi self.capture.release(): {release_duration:.2f} giây.")
                    capture_released = True
            except Exception as e:
                print(f"[VSH SAFE RELEASE ERROR] Lỗi khi giải phóng capture: {e}")
        
        # Chỉ tạo release thread nếu có capture cần giải phóng
        if self.capture:
            try:
                release_thread = threading.Thread(target=safe_release)
                release_thread.daemon = True
                release_thread.start()
                
                # Chỉ đợi release thread một khoảng thời gian nhất định
                release_timeout = 3.0  # 3 giây là đủ cho hầu hết trường hợp
                print(f"[VSH STOP] Đợi release thread tối đa {release_timeout} giây...")
                release_thread.join(timeout=release_timeout)
                
                if release_thread.is_alive():
                    print(f"[VSH STOP WARNING] Release thread vẫn đang chạy sau {release_timeout} giây.")
                    print("[VSH STOP] Tiếp tục mà không đợi release hoàn thành")
                else:
                    print("[VSH STOP] Release thread đã hoàn thành")
                    
                # Bất kể release thành công hay không, vẫn đặt capture = None
                self.capture = None
                print("[VSH STOP] Đã đặt capture = None")
                
            except Exception as e:
                print(f"[VSH STOP ERROR] Lỗi khi tạo/đợi release thread: {e}")
                # Đảm bảo đặt capture = None trong mọi trường hợp
                self.capture = None
                
        elif self.capture is not None:
            # Nếu capture không null nhưng không isOpened(), vẫn đặt = None
            self.capture = None
            print("[VSH STOP] Đặt capture = None (capture không mở)")
        else:
            print("[VSH STOP] Không có capture để giải phóng")
            
        # Làm sạch tài nguyên khác
        print("[VSH STOP] Làm sạch các tài nguyên khác (lock, frame)...")
        with self.lock:
            self.current_frame = None
            self.frame_available = False
        
        print(f"[VSH STOP] Đã hoàn thành dừng VideoSourceHandler cho nguồn: {self.source}")
        
    def _read_loop(self):
        """Loop chạy trong thread để liên tục đọc frame mới"""
        retry_delay = 0.01  # Giá trị cơ bản
        frame_read_count = 0
        while self.running:
            # Kiểm tra stop_event nếu có - THÊM MỚI
            if self.stop_event and self.stop_event.is_set():
                print(f"[VSH LOOP {self.source}] Phát hiện stop_event, thoát _read_loop")
                self.running = False  # Đảm bảo dừng ngay lập tức
                break
                
            try:
                # print(f"[VSH LOOP {self.source}] Bắt đầu đọc frame...") # Có thể gây quá nhiều log
                read_start_time = time.time()
                
                # Đọc frame với cơ chế timeout đơn giản - THÊM MỚI
                ret, frame = None, None
                read_success = False
                max_read_wait = 0.3  # 300ms timeout cho đọc frame
                
                while time.time() - read_start_time < max_read_wait and not read_success:
                    # Kiểm tra stop_event thường xuyên trong quá trình đợi đọc frame - THÊM MỚI
                    if self.stop_event and self.stop_event.is_set():
                        print(f"[VSH LOOP {self.source}] Phát hiện stop_event khi cố gắng đọc frame")
                        return  # Thoát ngay lập tức
                    
                    try:
                        if self.capture is None or not self.capture.isOpened():
                            break
                        ret, frame = self.capture.read()  # Thử đọc frame
                        if ret and frame is not None:
                            read_success = True
                            break
                    except Exception as e:
                        print(f"[VSH LOOP ERROR {self.source}] Lỗi khi đọc frame: {e}")
                    
                    # Đợi một chút trước khi thử lại
                    time.sleep(0.01)
                    
                read_duration = time.time() - read_start_time
                
                # Kiểm tra lại stop_event sau khi đọc frame - THÊM MỚI
                if self.stop_event and self.stop_event.is_set():
                    print(f"[VSH LOOP {self.source}] Phát hiện stop_event sau khi đọc frame")
                    return
                
                # In log nếu thời gian đọc frame > 0.5 giây
                if read_duration > 0.5:
                    print(f"[VSH LOOP WARNING {self.source}] Thời gian đọc frame: {read_duration:.2f} giây (success={read_success})")
                    
                if not read_success or frame is None or (frame is not None and frame.size == 0):
                    # ... (xử lý lỗi đọc frame như cũ)
                    self.consecutive_errors += 1
                    current_time = time.time()
                    
                    # Đặt lại last_error_time nếu đây là lỗi đầu tiên
                    if self.last_error_time is None:
                        self.last_error_time = current_time
                    
                    # In cảnh báo mỗi 3 giây để tránh làm tràn console
                    if self.last_error_time is None or (current_time - self.last_error_time) > 3:
                        print(f"[VSH LOOP WARNING {self.source}] Lỗi khi đọc frame. Số lỗi liên tiếp: {self.consecutive_errors}")
                        self.last_error_time = current_time
                    
                    # Kiểm tra stop_event thường xuyên khi xử lý lỗi - THÊM MỚI
                    if self.stop_event and self.stop_event.is_set():
                        print(f"[VSH LOOP {self.source}] Phát hiện stop_event khi xử lý lỗi đọc frame")
                        return
                    
                    # Điều chỉnh retry_delay tùy thuộc vào loại luồng
                    retry_delay = 0.01 if not self.is_rtsp else 0.05 * self.consecutive_errors
                    retry_delay = min(retry_delay, 2.0)  # Giới hạn tối đa 2 giây
                    
                    # Thử kết nối lại camera nếu có quá nhiều lỗi liên tiếp
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        # Kiểm tra stop_event trước khi kết nối lại - THÊM MỚI
                        if self.stop_event and self.stop_event.is_set():
                            print(f"[VSH LOOP {self.source}] Phát hiện stop_event trước khi thử kết nối lại")
                            return
                            
                        print(f"[VSH LOOP INFO {self.source}] Thử kết nối lại với nguồn video...")
                        # --- Bổ sung: Gọi release trước khi tạo lại --- 
                        if self.capture and self.capture.isOpened():
                            try:
                                print(f"[VSH LOOP INFO {self.source}] Gọi release() trước khi kết nối lại...")
                        self.capture.release()
                            except Exception as release_err:
                                print(f"[VSH LOOP ERROR {self.source}] Lỗi khi release trước khi kết nối lại: {release_err}")
                        # --- Kết thúc bổ sung ---
                        time.sleep(retry_delay)  # Đợi một chút trước khi thử lại
                        
                        # Kiểm tra stop_event trước khi tạo capture mới - THÊM MỚI
                        if self.stop_event and self.stop_event.is_set():
                            print(f"[VSH LOOP {self.source}] Phát hiện stop_event trước khi tạo capture mới")
                            return
                            
                        self.capture = cv2.VideoCapture(self.source)
                        
                        # Áp dụng lại các cấu hình đặc biệt cho RTSP
                        if self.is_rtsp:
                            # ... (cấu hình RTSP như cũ)
                            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                            try:
                                self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                            except AttributeError:
                                pass  # Đã báo lỗi ở phần trên, không cần báo lại
                            
                        self.consecutive_errors = 0
                        
                    # Kiểm tra stop_event trước khi sleep - THÊM MỚI
                    if self.stop_event and self.stop_event.is_set():
                        print(f"[VSH LOOP {self.source}] Phát hiện stop_event trước khi sleep khi xử lý lỗi")
                        return
                        
                    time.sleep(retry_delay)  # Tránh sử dụng 100% CPU khi gặp lỗi
                    continue
                
                # Reset counter nếu đọc thành công
                frame_read_count += 1
                self.consecutive_errors = 0
                self.last_error_time = None
                
                # Kiểm tra frame hợp lệ (không quá nhỏ)
                if frame.shape[0] < 10 or frame.shape[1] < 10:
                    print(f"[VSH LOOP WARNING {self.source}] Frame không hợp lệ, kích thước quá nhỏ: {frame.shape}")
                    
                    # Kiểm tra stop_event trước khi tiếp tục - THÊM MỚI
                    if self.stop_event and self.stop_event.is_set():
                        print(f"[VSH LOOP {self.source}] Phát hiện stop_event sau khi phát hiện frame không hợp lệ")
                        return
                        
                    time.sleep(0.01)
                    continue
                
                # Cập nhật frame mới nhất với lock để tránh race condition
                with self.lock:
                    self.current_frame = frame
                    self.frame_available = True
                
                # Kiểm tra stop_event trước khi sleep - THÊM MỚI
                if self.stop_event and self.stop_event.is_set():
                    print(f"[VSH LOOP {self.source}] Phát hiện stop_event sau khi cập nhật frame")
                    return
                
                # Nhường CPU cho các thread khác
                sleep_time = 0.001 if not self.is_rtsp else 0.005
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"[VSH LOOP ERROR {self.source}] Lỗi trong thread đọc frame: {e}")
                import traceback
                traceback.print_exc()
                
                # Kiểm tra stop_event khi gặp exception - THÊM MỚI
                if self.stop_event and self.stop_event.is_set():
                    print(f"[VSH LOOP {self.source}] Phát hiện stop_event sau khi xử lý exception")
                    return
                    
                time.sleep(0.1)  # Tránh vòng lặp vô hạn khi gặp lỗi
                
        print(f"[VSH LOOP] Thread đọc frame cho nguồn {self.source} kết thúc. Đã đọc {frame_read_count} frames.")
                
    def get_frame(self):
        """Lấy frame mới nhất từ thread"""
        with self.lock:
            if not self.frame_available:
                return None
            return self.current_frame.copy()  # Trả về bản sao để tránh conflict

def process_video_with_roi(video_source, mode, roi, stop_event, output_handler, username=None,
                           max_samples=settings.RECOGNITION_DEFAULT_MAX_SAMPLES,
                           recognition_threshold=settings.RECOGNITION_CHECK_IN_THRESHOLD,
                           employee_id=None, project=None, company=None,
                           camera_name=None, contractor=None, field=None):
    
    global collect_progress_tracker # Để có thể cập nhật biến global
    # --- DEBUG PRINTS --- 
    print("-"*30)
    print(f"[PROCESS START] Mode: {mode}, Source: {video_source}, ROI: {roi}, Username: {username}")
    print(f"[PROCESS START] Employee ID: {employee_id}, Project: {project}, Company: {company}")
    print(f"[PROCESS START] Contractor: {contractor}, Field: {field}")
    print(f"[PROCESS START] stop_event set: {stop_event.is_set()}")
    # --- END DEBUG PRINTS --- 
    
    # Xác định tên camera từ nguồn video nếu có thể và chưa có tên camera
    if camera_name is None:
        try:
            from .models import CameraConfig
            # Thử tìm camera trong cơ sở dữ liệu
            camera_obj = CameraConfig.objects.filter(source=str(video_source)).first()
            if camera_obj:
                camera_name = camera_obj.name
                print(f"[PROCESS INFO] Đã xác định camera: {camera_name}")
        except Exception as cam_err:
            print(f"[PROCESS INFO] Không thể xác định camera từ nguồn {video_source}: {cam_err}")
    else:
        print(f"[PROCESS INFO] Sử dụng tên camera đã cung cấp: {camera_name}")

    output_handler.start_stream() 

    if mode == 'collect' and not username:
        print("[PROCESS ERROR] Cần cung cấp username cho chế độ 'collect'")
        output_handler.stop_stream()
        return 0
    if roi is None and mode == 'collect':
         print("[PROCESS ERROR] Cần có ROI cho chế độ 'collect'")
         output_handler.stop_stream()
         return 0
    elif roi is not None and len(roi) != 4:
         print("[PROCESS ERROR] ROI không hợp lệ")
         output_handler.stop_stream()
         return 0 if mode == 'collect' else {}

    rx, ry, rw, rh = (0,0,0,0) 
    if roi:
        rx, ry, rw, rh = [int(v) for v in roi]

    # Khởi tạo và bắt đầu xử lý video
    video_handler = VideoSourceHandler(video_source)
    # Truyền stop_event cho VideoSourceHandler - THÊM MỚI
    video_handler.stop_event = stop_event
    
    # Sử dụng khối try-finally để đảm bảo giải phóng tài nguyên
    try:
    if not video_handler.start():
        error_msg = f"Không thể mở nguồn video: {video_source}"
        print(f"[PROCESS ERROR] {error_msg}")
        output_handler.stop_stream()
        return {"error": error_msg} if mode != 'collect' else 0
    
    detector, predictor, fa = None, None, None
    svc, encoder = None, None
            
    if mode in ['collect', 'recognize']:
        print("[PROCESS INFO] Đang tải bộ phát hiện khuôn mặt...")
        try:
            detector = dlib.get_frontal_face_detector()
            if not os.path.exists(settings.RECOGNITION_PREDICTOR_PATH):
                print(f"[PROCESS ERROR] Không tìm thấy file shape predictor tại: {settings.RECOGNITION_PREDICTOR_PATH}")
                output_handler.stop_stream()
                return 0 if mode == 'collect' else {}
            predictor = dlib.shape_predictor(settings.RECOGNITION_PREDICTOR_PATH)
            fa = FaceAligner(predictor, desiredFaceWidth=settings.RECOGNITION_FACE_WIDTH)
        except Exception as e:
            print(f"[PROCESS ERROR] Lỗi khi khởi tạo dlib/FaceAligner: {e}")
            output_handler.stop_stream()
            return 0 if mode == 'collect' else {}

    sample_count = 0
    recognized_persons = {}
    recognition_counts = {}
    output_dir = None
    last_save_time = {}

    if mode == 'collect':
        # Sử dụng sanitized_username
        # --- Hoàn nguyên sử dụng sanitized_username --- 
        sanitized_username = None
        if username:
            sanitized_username = sanitize_filename(username)
            print(f"[PROCESS INFO] Original username: '{username}', Sanitized: '{sanitized_username}'")
            if not sanitized_username:
                 print("[PROCESS ERROR] Username sau khi chuẩn hóa bị rỗng.")
                 output_handler.stop_stream()
                 return 0
        else: # Vẫn cần kiểm tra username gốc
            print("[PROCESS ERROR] Username bị thiếu cho chế độ collect.")
            output_handler.stop_stream()
            return 0
        output_dir = os.path.join(settings.RECOGNITION_TRAINING_DIR, sanitized_username)
        # --- Kết thúc hoàn nguyên --- 

        # --- Sử dụng username gốc trực tiếp --- 
        # if not username: # Kiểm tra lại username gốc
        #     print("[PROCESS ERROR] Username bị thiếu cho chế độ collect.")
        #     output_handler.stop_stream()
        #     return 0
        # output_dir = os.path.join(TRAINING_DATA_DIR, username)
        # --- Kết thúc sử dụng username gốc ---
        try:
            # Khởi tạo/Reset tracker cho sanitized_username này
            # collect_progress_tracker[sanitized_username] = {'current': 0, 'total': max_samples, 'active': True}
            # print(f"[Tracker INIT] Progress for {sanitized_username}: {collect_progress_tracker[sanitized_username]}")
            # --- Sử dụng username gốc làm key --- 
            collect_progress_tracker[username] = {
                'current': 0, 
                'total': max_samples, 
                'active': True,
                'completed': False,
                    'sanitized': sanitized_username,
                    'stopping': False  # THÊM MỚI - Đánh dấu trạng thái đang dừng
            }
            print(f"[Tracker INIT] Progress for {username}: {collect_progress_tracker[username]}")
            # --- Kết thúc thay đổi --- 

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"[PROCESS INFO] Đã tạo thư mục: {output_dir}")
                # --- Tạo file _info.txt --- 
                info_path = os.path.join(output_dir, '_info.txt')
                try:
                    with open(info_path, 'w', encoding='utf-8') as f_info:
                        f_info.write(username)
                    print(f"[PROCESS INFO] Đã tạo file thông tin: {info_path}")
                except Exception as e_info:
                    print(f"[PROCESS WARNING] Không thể tạo file thông tin '{info_path}': {e_info}")
                # --- Kết thúc tạo file _info.txt ---
            else:
                print(f"[PROCESS INFO] Thư mục đã tồn tại: {output_dir}")
                # --- Kiểm tra và tạo file _info.txt nếu thiếu --- 
                info_path = os.path.join(output_dir, '_info.txt')
                if not os.path.exists(info_path):
                    try:
                        with open(info_path, 'w', encoding='utf-8') as f_info:
                           f_info.write(username)
                        print(f"[PROCESS INFO] Đã tạo file thông tin (thiếu): {info_path}")
                    except Exception as e_info:
                        print(f"[PROCESS WARNING] Không thể tạo file thông tin (thiếu) '{info_path}': {e_info}")
                # --- Kết thúc kiểm tra ---
        except OSError as e:
            print(f"[PROCESS ERROR] Không thể tạo thư mục '{output_dir}': {e}")
            output_handler.stop_stream()
            return 0
        print(f"[PROCESS INFO] Chế độ COLLECT: Sẽ lưu tối đa {max_samples} mẫu vào '{output_dir}'")

    elif mode == 'recognize':
        if not os.path.exists(settings.RECOGNITION_SVC_PATH) or not os.path.exists(settings.RECOGNITION_CLASSES_PATH):
            print("Lỗi: Không tìm thấy file model (svc.sav) hoặc classes (classes.npy). Vui lòng huấn luyện trước.")
            output_handler.stop_stream()
            return {}
        try:
            with open(settings.RECOGNITION_SVC_PATH, 'rb') as f:
                svc = pickle.load(f)
            encoder = LabelEncoder()
            encoder.classes_ = np.load(settings.RECOGNITION_CLASSES_PATH)
            for name in encoder.classes_:
                recognized_persons[name] = False
                recognition_counts[name] = 0
                last_save_time[name] = None
            print(f"[PROCESS INFO] Chế độ RECOGNIZE: Đã tải model cho {len(encoder.classes_)} người.")
        except Exception as e:
            print(f"[PROCESS ERROR] Lỗi khi tải model/classes: {e}")
            output_handler.stop_stream()
            return {}

    frame_index = 0
        last_processed_frame = 0  # THÊM MỚI - theo dõi frame cuối cùng được xử lý
        loop_start_time = time.time()
        last_log_time = loop_start_time
        print(f"[PROCESS {camera_name}] Bắt đầu vòng lặp xử lý (Mode: {mode})...")
        
        while not stop_event.is_set():
            loop_iter_start_time = time.time()
            # Kiểm tra stop_event NGAY ĐẦU vòng lặp
            if stop_event.is_set():
                print(f"[PROCESS {camera_name}] Phát hiện stop_event ngay đầu vòng lặp, thoát...")
                # Cập nhật trạng thái 'stopping' nếu đang ở chế độ thu thập
                if mode == 'collect' and username in collect_progress_tracker:
                    collect_progress_tracker[username]['stopping'] = True
                    print(f"[PROCESS {camera_name}] Cập nhật trạng thái stopping=True cho {username}")
                break
                
            frame = video_handler.get_frame()
            if frame is None:
                # Kiểm tra stop_event sau khi get_frame() trả về None - THÊM MỚI
                if stop_event.is_set():
                    print(f"[PROCESS {camera_name}] Phát hiện stop_event sau khi get_frame() trả về None, thoát...")
                    if mode == 'collect' and username in collect_progress_tracker:
                        collect_progress_tracker[username]['stopping'] = True
                    break
                # print(f"[PROCESS {camera_name}] No frame available, waiting...") # Có thể gây nhiều log
                time.sleep(0.01)
                continue

            frame_to_display = frame.copy()
            roi_frame_for_processing = None

            # Kiểm tra stop_event sau khi copy frame và trước khi xử lý ROI - THÊM MỚI
            if stop_event.is_set():
                print(f"[PROCESS {camera_name}] Phát hiện stop_event sau khi copy frame, thoát...")
                if mode == 'collect' and username in collect_progress_tracker:
                    collect_progress_tracker[username]['stopping'] = True
                break

            # Xử lý ROI
            if roi:
                orig_h, orig_w = frame.shape[:2]
                scale = orig_w / settings.RECOGNITION_FRAME_WIDTH
                orig_rx = int(rx * scale)
                orig_ry = int(ry * scale)
                orig_rw = int(rw * scale)
                orig_rh = int(rh * scale)
                orig_rx = max(0, orig_rx)
                orig_ry = max(0, orig_ry)
                orig_rw = min(orig_rw, orig_w - orig_rx)
                orig_rh = min(orig_rh, orig_h - orig_ry)

                if orig_rw > 0 and orig_rh > 0:
                    cv2.rectangle(frame_to_display, (orig_rx, orig_ry), (orig_rx + orig_rw, orig_ry + orig_rh), (255, 0, 0), 2)
                    try:
                        if mode in ['collect', 'recognize']:
                            roi_frame_for_processing = frame[orig_ry:orig_ry + orig_rh, orig_rx:orig_rx + orig_rw].copy()
                            if roi_frame_for_processing.size == 0:
                                roi_frame_for_processing = None
                    except Exception as e:
                         print(f"[PROCESS ERROR] Lỗi khi crop ROI: {e}")
                         roi_frame_for_processing = None
                else:
                    roi_frame_for_processing = None

            elif mode == 'recognize': 
                roi_frame_for_processing = frame.copy() # Xử lý toàn bộ frame
                orig_rx, orig_ry = 0, 0 # Đặt tọa độ gốc để vẽ bounding box đúng

            # Kiểm tra stop_event sau khi xử lý ROI và trước khi xử lý frame - THÊM MỚI
            if stop_event.is_set():
                print(f"[PROCESS {camera_name}] Phát hiện stop_event sau khi xử lý ROI, thoát...")
                if mode == 'collect' and username in collect_progress_tracker:
                    collect_progress_tracker[username]['stopping'] = True
                break

            if roi_frame_for_processing is not None and mode in ['collect', 'recognize']:
                frame_index += 1
                
                # Kiểm tra stop_event TRƯỚC khi quyết định bỏ qua frame - THÊM MỚI
                if stop_event.is_set():
                    print(f"[PROCESS {camera_name}] Phát hiện stop_event trước khi kiểm tra frame_skip, thoát...")
                    if mode == 'collect' and username in collect_progress_tracker:
                        collect_progress_tracker[username]['stopping'] = True
                    break
                
                # THAY ĐỔI: Cải tiến logic bỏ qua frame
                process_this_frame = True
                if mode == 'recognize':
                    process_this_frame = (frame_index % settings.RECOGNIZE_FRAME_SKIP == 0)
                elif mode == 'collect':
                    process_this_frame = (frame_index % settings.COLLECT_FRAME_SKIP == 0) 
                
                if not process_this_frame:
                    output_handler.set_frame(frame_to_display)
                    
                    # Kiểm tra stop_event sau khi quyết định bỏ qua frame - THÊM MỚI
                    if stop_event.is_set():
                        print(f"[PROCESS {camera_name}] Phát hiện stop_event sau khi bỏ qua frame, thoát...")
                        if mode == 'collect' and username in collect_progress_tracker:
                            collect_progress_tracker[username]['stopping'] = True
                        break
                    
                     continue
                    
                # Ghi nhớ frame cuối cùng được xử lý - THÊM MỚI
                last_processed_frame = frame_index

                try:
                    roi_gray = cv2.cvtColor(roi_frame_for_processing, cv2.COLOR_BGR2GRAY)
                    faces = detector(roi_gray, 0)
                except Exception as detect_err:
                    print(f"[PROCESS ERROR] Lỗi trong quá trình phát hiện khuôn mặt: {detect_err}")
                    faces = []

                # Kiểm tra stop_event sau khi phát hiện khuôn mặt - THÊM MỚI
                if stop_event.is_set():
                    print(f"[PROCESS {camera_name}] Phát hiện stop_event sau khi phát hiện {len(faces)} khuôn mặt, thoát...")
                    if mode == 'collect' and username in collect_progress_tracker:
                        collect_progress_tracker[username]['stopping'] = True
                    break

                detected_in_frame = set()
                if faces:
                    for face in faces:
                        # Kiểm tra stop_event trong vòng lặp mặt - THÊM MỚI
                        if stop_event.is_set():
                            print(f"[PROCESS {camera_name}] Phát hiện stop_event trong vòng lặp mặt, thoát...")
                            if mode == 'collect' and username in collect_progress_tracker:
                                collect_progress_tracker[username]['stopping'] = True
                            break
                            
                        (fx, fy, fw, fh) = face_utils.rect_to_bb(face)
                        if roi:
                            draw_x = orig_rx + int(fx * (orig_rw / roi_frame_for_processing.shape[1]))
                            draw_y = orig_ry + int(fy * (orig_rh / roi_frame_for_processing.shape[0]))
                            draw_w = int(fw * (orig_rw / roi_frame_for_processing.shape[1]))
                            draw_h = int(fh * (orig_rh / roi_frame_for_processing.shape[0]))
                        else:
                            draw_x, draw_y, draw_w, draw_h = fx, fy, fw, fh

                        try:
                            face_aligned = fa.align(roi_frame_for_processing, roi_gray, face)
                        except Exception as align_err:
                            print(f"[PROCESS ERROR] Lỗi căn chỉnh mặt: {align_err}")
                            continue

                        if mode == 'collect':
                            # Kiểm tra stop_event trước khi lưu mẫu - THÊM MỚI
                            if stop_event.is_set():
                                print(f"[PROCESS {camera_name}] Phát hiện stop_event trước khi lưu mẫu, thoát...")
                                if username in collect_progress_tracker:
                                    collect_progress_tracker[username]['stopping'] = True
                                break
                                
                            sample_count += 1
                            img_path = os.path.join(output_dir, f"{sanitized_username}_{sample_count}.jpg")
                            cv2.imwrite(img_path, face_aligned)

                            if username in collect_progress_tracker:
                                collect_progress_tracker[username]['current'] = sample_count

                            cv2.rectangle(frame_to_display, (draw_x, draw_y), (draw_x + draw_w, draw_y + draw_h), (0, 255, 0), 1)
                            cv2.putText(frame_to_display, f"Sample: {sample_count}", (draw_x, draw_y - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                            if sample_count >= max_samples:
                                print(f"[PROCESS INFO] Đã thu thập đủ {max_samples} mẫu.")
                                
                                # Đánh dấu là hoàn thành thay vì chỉ đặt stop_event - THÊM MỚI
                                if username in collect_progress_tracker:
                                    collect_progress_tracker[username]['completed'] = True
                                    print(f"[PROCESS INFO] Đánh dấu completed=True cho {username}")
                                
                                stop_event.set()
                                break

                        elif mode == 'recognize':
                            # Kiểm tra stop_event trước khi nhận diện - THÊM MỚI
                            if stop_event.is_set():
                                print(f"[PROCESS {camera_name}] Phát hiện stop_event trước khi nhận diện, thoát...")
                                break
                                
                            (pred_idx, prob) = predict(face_aligned, svc)
                            prob_value = float(prob[0])
                            person_name = "Unknown"
                            color = (0, 0, 255)

                            if pred_idx != [-1]:
                                person_name = encoder.inverse_transform(pred_idx)[0]
                                detected_in_frame.add(person_name)
                                color = (0, 255, 0)

                                if prob_value >= 0.5:
                                    recognition_counts[person_name] = recognition_counts.get(person_name, 0) + 1
                                else:
                                    recognition_counts[person_name] = 0

                                if recognition_counts[person_name] >= recognition_threshold:
                                    if not recognized_persons.get(person_name, False):
                                        print(f"[RECOGNIZED] {person_name}")
                                        recognized_persons[person_name] = True
                                    
                                    # Kiểm tra stop_event trước khi lưu ảnh chấm công - THÊM MỚI
                                    if stop_event.is_set():
                                        print(f"[PROCESS {camera_name}] Phát hiện stop_event trước khi lưu ảnh chấm công, thoát...")
                                        break
                                    
                                    # Xử lý lưu ảnh chấm công như cũ
                                    try:
                                        # Xác định tên file và đường dẫn
                                        now = get_current_time()
                                        today = get_current_date()
                                        sanitized_person_name_for_file = sanitize_filename(person_name)
                                        
                                        # Xác định xem đây là check-in hay check-out
                                        check_in_file_pattern = f'{sanitized_person_name_for_file}_{today}_*.jpg'
                                        check_in_dir = os.path.join(settings.MEDIA_ROOT, settings.RECOGNITION_ATTENDANCE_FACES_DIR, settings.RECOGNITION_CHECK_IN_SUBDIR)
                                        
                                        import glob
                                        existing_check_ins = glob.glob(os.path.join(check_in_dir, check_in_file_pattern))
                                        
                                        # Nếu chưa có ảnh check-in hôm nay -> đây là check-in
                                        is_check_in = len(existing_check_ins) == 0
                                        
                                        if is_check_in:
                                            # Xử lý check-in
                                            face_filename = f'{sanitized_person_name_for_file}_{today}_{now.strftime("%H%M%S")}.jpg'
                                            face_path = os.path.join(settings.MEDIA_ROOT, settings.RECOGNITION_ATTENDANCE_FACES_DIR, 
                                                                    settings.RECOGNITION_CHECK_IN_SUBDIR, face_filename)
                                            relative_face_path = os.path.join(settings.RECOGNITION_ATTENDANCE_FACES_DIR, 
                                                                        settings.RECOGNITION_CHECK_IN_SUBDIR, face_filename)
                                            
                                            # Tạo thư mục nếu chưa tồn tại
                                            os.makedirs(os.path.dirname(face_path), exist_ok=True)
                                            
                                            # Lưu ảnh
                                            saved_img = cv2.imwrite(face_path, face_aligned)
                                            if saved_img:
                                                print(f"[PROCESS INFO] Phát hiện check-in cho '{person_name}' - đã lưu ảnh nhưng không lưu dữ liệu chấm công")
                                                last_save_time[person_name] = now
                                            else:
                                                print(f"[PROCESS ERROR] Không thể lưu ảnh check-in tại: {face_path}")
                                        else:
                                            # Xử lý check-out
                                            face_filename = f'{sanitized_person_name_for_file}_{today}_{now.strftime("%H%M%S")}_out.jpg'
                                            face_path = os.path.join(settings.MEDIA_ROOT, settings.RECOGNITION_ATTENDANCE_FACES_DIR, 
                                                                    settings.RECOGNITION_CHECK_OUT_SUBDIR, face_filename)
                                            relative_face_path = os.path.join(settings.RECOGNITION_ATTENDANCE_FACES_DIR, 
                                                                        settings.RECOGNITION_CHECK_OUT_SUBDIR, face_filename)
                                            
                                            # Tạo thư mục nếu chưa tồn tại
                                            os.makedirs(os.path.dirname(face_path), exist_ok=True)
                                            
                                            # Lưu ảnh
                                            saved_img = cv2.imwrite(face_path, face_aligned)
                                            if saved_img:
                                                # Lấy thời gian check-in từ tên file
                                                check_in_time_str = "không rõ"
                                                if existing_check_ins:
                                                    # Lấy file đầu tiên
                                                    check_in_file = os.path.basename(existing_check_ins[0])
                                                    # Trích xuất thời gian từ tên file (định dạng: name_YYYY-MM-DD_HHMMSS.jpg)
                                                    try:
                                                        time_part = check_in_file.split('_')[-1].split('.')[0]
                                                        check_in_time_str = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                                                    except:
                                                        pass
                                                
                                                check_out_time_str = now.strftime('%H:%M:%S')
                                                print(f"[PROCESS INFO] Phát hiện check-out cho '{person_name}'")
                                                last_save_time[person_name] = now
                                            else:
                                                print(f"[PROCESS ERROR] Không thể lưu ảnh check-out tại: {face_path}")
                                    except Exception as e:
                                        print(f"[PROCESS ERROR] Lỗi khi xử lý ảnh chấm công: {e}")
                                        import traceback
                                        traceback.print_exc()

                                    color = (0, 255, 255)
                            else:
                                if prob_value > 0:
                                    person_name = f"Unknown ({prob_value:.2f})"
                                else:
                                    person_name = "Unknown (0.00)"

                            cv2.rectangle(frame_to_display, (draw_x, draw_y), (draw_x + draw_w, draw_y + draw_h), color, 1)
                            cv2.putText(frame_to_display, person_name, (draw_x, draw_y - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    # Kiểm tra nếu vòng lặp mặt bị thoát do stop_event
                    if stop_event.is_set():
                        print(f"[PROCESS {camera_name}] Thoát vòng lặp mặt do stop_event")
                        break

                    if mode == 'recognize':
                         for name in recognition_counts:
                             if name not in detected_in_frame:
                         recognition_counts[name] = 0

            # Kiểm tra stop_event trước khi cập nhật output_handler
            if stop_event.is_set():
                print(f"[PROCESS {camera_name}] Phát hiện stop_event trước khi cập nhật output_handler, thoát...")
                if mode == 'collect' and username in collect_progress_tracker:
                    collect_progress_tracker[username]['stopping'] = True
                break

            output_handler.set_frame(frame_to_display)

            # Kiểm tra stop_event SAU KHI xử lý frame
            if stop_event.is_set():
                print(f"[PROCESS {camera_name}] Phát hiện stop_event sau khi xử lý frame, thoát...")
                if mode == 'collect' and username in collect_progress_tracker:
                    collect_progress_tracker[username]['stopping'] = True
                break

            # Log định kỳ để theo dõi tiến trình
            current_time = time.time()
            if current_time - last_log_time > 10.0: # Log mỗi 10 giây
                 print(f"[PROCESS LOOP {camera_name}] Đang chạy... Frame index: {frame_index}, Processed: {last_processed_frame}, Mode: {mode}")
                 # Thêm trạng thái stop_event vào log định kỳ
                 print(f"[PROCESS LOOP {camera_name}] stop_event.is_set() = {stop_event.is_set()}")
                 last_log_time = current_time
            
            # Log nếu một vòng lặp mất quá nhiều thời gian
            loop_iter_duration = current_time - loop_iter_start_time
            if loop_iter_duration > 2.0: # Cảnh báo nếu vòng lặp > 2 giây
                print(f"[PROCESS LOOP WARNING {camera_name}] Vòng lặp xử lý mất {loop_iter_duration:.2f} giây (frame {frame_index})")

            time.sleep(0.001) # Nhường CPU
            
        # Vòng lặp kết thúc - kiểm tra lý do - THÊM MỚI
        if stop_event.is_set():
            print(f"[PROCESS END {camera_name}] Vòng lặp kết thúc do stop_event được đặt")
        else:
            print(f"[PROCESS END {camera_name}] Vòng lặp kết thúc do lý do khác")

    # Đánh dấu tiến trình là không hoạt động khi kết thúc (hoặc bị dừng)
    if mode == 'collect' and username and username in collect_progress_tracker:
            # Đánh dấu 'stopping' = False vì đã dừng xong
            collect_progress_tracker[username]['stopping'] = False
            
        if sample_count >= max_samples:
            # Chỉ đặt active=False nếu đã thu thập đủ mẫu
            collect_progress_tracker[username]['active'] = False
            collect_progress_tracker[username]['completed'] = True
            print(f"[Tracker END] Thu thập hoàn thành cho {username}: {collect_progress_tracker[username]}")
        else:
            # Nếu bị dừng giữa chừng, giữ active=True thêm một thời gian để frontend hiển thị
            collect_progress_tracker[username]['completed'] = False
            print(f"[Tracker END] Thu thập bị dừng giữa chừng cho {username}: {collect_progress_tracker[username]}")
            # Đặt một timer để đặt active=False sau một khoảng thời gian
            def delay_deactivate():
                time.sleep(10) # Đợi 10 giây để frontend hiển thị
                if username in collect_progress_tracker:
                    collect_progress_tracker[username]['active'] = False
                    print(f"[Tracker DELAYED] Đã đặt active=False sau khi hiển thị: {collect_progress_tracker[username]}")
            
            # Tạo một thread riêng để xử lý việc delay deactivate
            deactivate_thread = threading.Thread(target=delay_deactivate, daemon=True)
            deactivate_thread.start()

    print("[PROCESS END] Kết thúc hàm process_video_with_roi")
    if mode == 'collect':
        print(f"[PROCESS INFO] Đã lưu tổng cộng {sample_count} mẫu cho '{username}' vào thư mục '{sanitized_username}'.")
        return sample_count
    elif mode == 'recognize':
        print("--- Kết quả Nhận diện Cuối cùng (khi dừng) ---")
        final_recognized = {name: status for name, status in recognized_persons.items() if status}
        if not final_recognized:
            print("Không có ai được nhận diện trong suốt quá trình.")
        else:
            for name in final_recognized:
                print(f"- {name}")
        print("------------------------------------")
        return final_recognized
    else:
        return {}

    except Exception as e: # Thêm khối except cho try chính
        print(f"[PROCESS ERROR] Lỗi nghiêm trọng trong quá trình xử lý video: {e}")
        import traceback
        traceback.print_exc()
        # Đảm bảo dừng output handler nếu có lỗi nghiêm trọng
        try:
            output_handler.stop_stream()
        except Exception as stop_err:
            print(f"[PROCESS ERROR] Lỗi khi dừng output handler sau khi có lỗi: {stop_err}")
        # Cập nhật tracker nếu đang thu thập
        if mode == 'collect' and username and username in collect_progress_tracker:
            collect_progress_tracker[username]['active'] = False
            collect_progress_tracker[username]['completed'] = False # Đánh dấu là chưa hoàn thành
            print(f"[Tracker ERROR] Đặt active=False do lỗi cho {username}: {collect_progress_tracker[username]}")
        # Trả về kết quả lỗi phù hợp
        return 0 if mode == 'collect' else {}
        
    finally:
        # Đảm bảo luôn giải phóng tài nguyên ngay cả khi có lỗi hoặc kết thúc bình thường
        print(f"[PROCESS FINALLY {camera_name}] Bắt đầu giải phóng tài nguyên...")
        try:
            # Dừng VideoSourceHandler trước
            if video_handler:
                print("[PROCESS FINALLY] Dừng VideoSourceHandler...")
                video_handler.stop()
        except Exception as e:
            print(f"[PROCESS ERROR] Lỗi khi dừng VideoSourceHandler trong finally: {e}")
            
        try:
            # Dừng output handler (có thể đã được gọi nếu có lỗi)
            print("[PROCESS FINALLY] Dừng OutputHandler...")
            output_handler.stop_stream()
        except Exception as e:
            print(f"[PROCESS ERROR] Lỗi khi dừng OutputHandler trong finally: {e}")
            
        # Giải phóng bộ nhớ các mô hình nếu cần
        detector, predictor, fa = None, None, None
        svc, encoder = None, None
        
        print("[PROCESS FINALLY] Đã giải phóng tất cả tài nguyên")
        
        # *** Lưu ý: Không nên trả về giá trị từ khối finally *** 
        # Việc return ở đây sẽ ghi đè giá trị return từ khối try hoặc except
        # Giá trị return nên được xử lý trong khối try hoặc except


