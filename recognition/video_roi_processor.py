import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import numpy as np
import face_recognition
import pickle
import os
import time
from sklearn.preprocessing import LabelEncoder
from django.utils import timezone
from recognition.models import AttendanceRecord
from django.contrib.auth.models import User
from django.conf import settings
import threading
import re
from unidecode import unidecode
from .firebase_util import push_attendance_to_firebase  # Thêm import này

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
            # print("Debug: No face encodings found.")
            return ([-1], [0.0])

        # Assuming only one face encoding per aligned image
        face_encodings_list[0] = faces_encodings[0]

    except Exception as e:
        # print(f"Debug: Error during face encoding: {e}")
        return ([-1], [0.0])

    try:
        prob = svc.predict_proba(face_encodings_list)
        best_class_index = np.argmax(prob[0])
        best_class_probability = prob[0][best_class_index]

        # print(f"Debug: Probabilities: {prob[0]}, Best Index: {best_class_index}, Prob: {best_class_probability:.2f}")

        # --- Sử dụng ngưỡng từ settings --- 
        if best_class_probability >= threshold:
            return ([best_class_index], [best_class_probability])
        else:
            # print(f"Debug: Probability {best_class_probability:.2f} below threshold {threshold}")
            # --- Thay đổi: Trả về ngưỡng đã dùng để debug --- 
            # return ([-1], [best_class_probability]) # Return -1 but still provide the probability
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
        print(f"[INFO] Sử dụng nguồn video ID: {source_int}")
    except ValueError:
        cap = cv2.VideoCapture(video_source)
        print(f"[INFO] Sử dụng nguồn video: {video_source}")

    if not cap.isOpened():
        print(f"Lỗi: Không thể mở nguồn video: {video_source}")
        return None

    window_title = "Chon ROI - Nhan 'n'=Next Frame, ENTER/SPACE=Chon ROI, ESC/q=Huy"
    cv2.namedWindow(window_title)
    roi = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cảnh báo: Đã hết video hoặc không thể đọc frame tiếp theo.")
            # Option: Break or loop back if it's a stream? For now, break.
            if roi is None: # If we never selected an ROI
                print("Không có frame nào để chọn ROI.")
            break

        frame_count += 1
        # --- Sử dụng FRAME_WIDTH từ settings --- 
        frame_display = imutils.resize(frame, width=settings.RECOGNITION_FRAME_WIDTH).copy() # Thay FRAME_WIDTH bằng settings
        cv2.putText(frame_display, f"Frame: {frame_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame_display, "Nhan 'n': Next, ENTER/SPACE: Chon, ESC/q: Huy", (10, frame_display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(window_title, frame_display)

        key = cv2.waitKey(0) & 0xFF # Wait indefinitely for key press

        if key == ord('n'): # Next frame(s)
            print(f"[INFO] Bỏ qua {frame_skip_on_next} frames...")
            for _ in range(frame_skip_on_next -1): # Read and discard frames
                 ret, _ = cap.read()
                 if not ret:
                      break # Stop if video ends
                 frame_count += 1
            continue # Go to the next iteration to read and display

        elif key == ord(' ') or key == 13: # Space or Enter - Select ROI on *this* frame
            print("Vui lòng vẽ hình chữ nhật ROI và nhấn ENTER/SPACE lần nữa...")
            # Use selectROI on the *current* frame that was displayed
            roi_selected = cv2.selectROI(window_title, frame_display, showCrosshair=True, fromCenter=False)
            
            # Check if selection was cancelled during selectROI itself
            if roi_selected == (0, 0, 0, 0):
                 print("Đã hủy trong lúc vẽ ROI. Nhan 'n' để thử frame khác, ESC/q để thoát.")
                 # Show the frame again to allow choosing 'n' or 'esc'
                 cv2.putText(frame_display, "DA HUY ROI. Nhan 'n'/ENTER/ESC", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                 cv2.imshow(window_title, frame_display)
                 continue # Go back to waiting for 'n', Enter, or ESC
            else:
                roi = roi_selected # Valid ROI selected
                print(f"ROI được chọn trên Frame {frame_count}: {roi}")
                break # Exit the loop

        elif key == ord('q') or key == 27: # q or ESC - Cancel
            print("Hủy chọn ROI.")
            roi = None
            break
        else:
             print(f"Phím không hợp lệ: {chr(key)}. Chỉ sử dụng 'n', SPACE, ENTER, ESC, 'q'.")

    # Cleanup
    cap.release()
    cv2.destroyWindow(window_title)
    
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
        print(f"[INFO] Khởi tạo VideoSourceHandler với nguồn: {source} {'(RTSP/Streaming)' if self.is_rtsp else ''}")
        
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
            # Đặt thời gian timeout - có thể điều chỉnh tùy camera

            # Cho phép kết nối lại nhanh hơn
            self.max_consecutive_errors = 3  # Giảm số lỗi trước khi kết nối lại
            print("[INFO] Đã áp dụng cấu hình đặc biệt cho luồng RTSP/Streaming")
            
        if not self.capture.isOpened():
            print(f"[ERROR] Không thể mở nguồn video: {self.source}")
            return False
            
        # Tùy chọn kiểm tra kích thước frame
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Kích thước video: {width}x{height}, FPS: {fps:.1f}")
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.daemon = True  # Thread sẽ tự động kết thúc khi chương trình kết thúc
        self.thread.start()
        print(f"[INFO] Đã khởi động thread đọc frame từ nguồn: {self.source}")
        return True
        
    def stop(self):
        """Dừng thread đọc frame và giải phóng tài nguyên"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)  # Đợi thread kết thúc tối đa 1 giây
        if self.capture:
            self.capture.release()
        print("[INFO] Đã dừng VideoSourceHandler")
        
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
                        print(f"[WARNING] Lỗi khi đọc frame. Số lỗi liên tiếp: {self.consecutive_errors}")
                        self.last_error_time = current_time
                    
                    # Điều chỉnh retry_delay tùy thuộc vào loại luồng
                    retry_delay = 0.01 if not self.is_rtsp else 0.05 * self.consecutive_errors
                    retry_delay = min(retry_delay, 2.0)  # Giới hạn tối đa 2 giây
                    
                    # Thử kết nối lại camera nếu có quá nhiều lỗi liên tiếp
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        print(f"[INFO] Thử kết nối lại với nguồn video: {self.source}")
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
                    print(f"[WARNING] Frame không hợp lệ, kích thước quá nhỏ: {frame.shape}")
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
                print(f"[ERROR] Lỗi trong thread đọc frame: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Tránh vòng lặp vô hạn khi gặp lỗi
                
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
                           camera_name=None):
    
    global collect_progress_tracker # Để có thể cập nhật biến global
    # --- DEBUG PRINTS --- 
    print("-"*30)
    print(f"[PROCESS START] Mode: {mode}, Source: {video_source}, ROI: {roi}, Username: {username}")
    print(f"[PROCESS START] Employee ID: {employee_id}, Project: {project}, Company: {company}")
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
    if roi is None and mode != 'stream':
         print("[PROCESS ERROR] Cần có ROI cho chế độ 'collect' hoặc 'recognize'")
         output_handler.stop_stream()
         return 0 if mode == 'collect' else {}
    elif roi is not None and len(roi) != 4:
         print("[PROCESS ERROR] ROI không hợp lệ")
         output_handler.stop_stream()
         return 0 if mode == 'collect' else {}

    rx, ry, rw, rh = (0,0,0,0) 
    if roi:
        rx, ry, rw, rh = [int(v) for v in roi]

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
                # 'sanitized': username # Chỗ này bị sai ở lần sửa trước, nên là sanitized_username
                'sanitized': sanitized_username 
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

    print(f"[PROCESS INFO] Khởi tạo luồng video từ nguồn: {video_source}")
    video_handler = VideoSourceHandler(video_source)
    if not video_handler.start():
        print(f"[PROCESS ERROR] Không thể khởi động VideoSourceHandler cho nguồn: {video_source}")
        output_handler.stop_stream()
        return 0 if mode == 'collect' else recognized_persons

    frame_index = 0
    print(f"[PROCESS INFO] Bắt đầu vòng lặp xử lý (Mode: {mode})...")
    try:
        while not stop_event.is_set():
            frame = video_handler.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_to_display = frame.copy()
            roi_frame_for_processing = None

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

            if mode in ['collect', 'recognize'] and roi_frame_for_processing is not None:
                frame_index += 1
                if frame_index % settings.RECOGNITION_FRAME_SKIP != 0:
                    output_handler.set_frame(frame_to_display)
                    continue

                try:
                    roi_frame_resized = roi_frame_for_processing
                    roi_gray = cv2.cvtColor(roi_frame_resized, cv2.COLOR_BGR2GRAY)
                    faces = detector(roi_gray, 0)
                except Exception as detect_err:
                    print(f"[PROCESS ERROR] Lỗi trong quá trình phát hiện khuôn mặt: {detect_err}")
                    faces = []

                detected_in_frame = set()
                if faces:
                    for face in faces:
                        (fx, fy, fw, fh) = face_utils.rect_to_bb(face)
                        draw_x = orig_rx + int(fx * (orig_rw / roi_frame_resized.shape[1]))
                        draw_y = orig_ry + int(fy * (orig_rh / roi_frame_resized.shape[0]))
                        draw_w = int(fw * (orig_rw / roi_frame_resized.shape[1]))
                        draw_h = int(fh * (orig_rh / roi_frame_resized.shape[0]))

                        try:
                            face_aligned = fa.align(roi_frame_resized, roi_gray, face)
                        except Exception as align_err:
                            print(f"[PROCESS ERROR] Lỗi căn chỉnh mặt: {align_err}")
                            continue

                        if mode == 'collect':
                            sample_count += 1
                            # Sử dụng sanitized_username
                            # img_path = os.path.join(output_dir, f"{sanitized_username}_{sample_count}.jpg")
                            # --- Sử dụng username gốc trực tiếp --- 
                            # img_path = os.path.join(output_dir, f"{username}_{sample_count}.jpg")
                            # --- Kết thúc sử dụng username gốc ---
                            # --- Hoàn nguyên sử dụng sanitized_username --- 
                            img_path = os.path.join(output_dir, f"{sanitized_username}_{sample_count}.jpg")
                            # --- Kết thúc hoàn nguyên ---
                            cv2.imwrite(img_path, face_aligned)

                            # Cập nhật tiến trình cho sanitized_username
                            # if sanitized_username in collect_progress_tracker:
                            #    collect_progress_tracker[sanitized_username]['current'] = sample_count
                            # --- Cập nhật tiến trình cho username gốc --- 
                            if username in collect_progress_tracker:
                                collect_progress_tracker[username]['current'] = sample_count
                            # --- Kết thúc thay đổi ---

                            cv2.rectangle(frame_to_display, (draw_x, draw_y), (draw_x + draw_w, draw_y + draw_h), (0, 255, 0), 1)
                            cv2.putText(frame_to_display, f"Sample: {sample_count}", (draw_x, draw_y - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                            if sample_count >= max_samples:
                                print(f"[PROCESS INFO] Đã thu thập đủ {max_samples} mẫu.")
                                stop_event.set()
                                break

                        elif mode == 'recognize':
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
                                    try:
                                        # --- Chuẩn hóa tên chỉ cho filename --- 
                                        sanitized_person_name_for_file = sanitize_filename(person_name)
                                        # --- Kết thúc chuẩn hóa ---
                                        try:
                                            user = User.objects.get(username=person_name)
                                        except User.DoesNotExist:
                                            print(f"[PROCESS INFO] Tạo mới người dùng '{person_name}' trong cơ sở dữ liệu.")
                                            user = User.objects.create_user(
                                                username=person_name,
                                                password=f"default_{person_name}",
                                                first_name=person_name
                                            )

                                        now = timezone.now()
                                        today = now.date()

                                        record, created = AttendanceRecord.objects.get_or_create(
                                            user=user,
                                            date=today,
                                            defaults={
                                                'check_in': now,
                                                'project': project,
                                                'company': company,
                                                'employee_id': employee_id
                                            }
                                        )

                                        # Cập nhật thông tin dự án và công ty nếu có
                                        if project and not record.project:
                                            record.project = project
                                        if company and not record.company:
                                            record.company = company
                                        if employee_id and not record.employee_id:
                                            record.employee_id = employee_id

                                        if created:
                                            # --- Check-in: Vẫn dùng ngưỡng mặc định (recognition_threshold đã lấy từ settings) ---
                                            face_filename = f'{sanitized_person_name_for_file}_{today}_{now.strftime("%H%M%S")}.jpg'
                                            # --- Sử dụng settings cho đường dẫn ảnh check-in ---
                                            # face_path = os.path.join(settings.MEDIA_ROOT, 'attendance_faces', 'check_in', face_filename)
                                            # relative_face_path = os.path.join('attendance_faces', 'check_in', face_filename)
                                            face_path = os.path.join(settings.MEDIA_ROOT, settings.RECOGNITION_ATTENDANCE_FACES_DIR, settings.RECOGNITION_CHECK_IN_SUBDIR, face_filename)
                                            relative_face_path = os.path.join(settings.RECOGNITION_ATTENDANCE_FACES_DIR, settings.RECOGNITION_CHECK_IN_SUBDIR, face_filename)
                                            # --- Kết thúc sử dụng settings ---
                                            os.makedirs(os.path.dirname(face_path), exist_ok=True)
                                            # --- Thêm kiểm tra lưu ảnh --- 
                                            saved_img_in = cv2.imwrite(face_path, face_aligned)
                                            if not saved_img_in:
                                                print(f"[PROCESS ERROR] Không thể lưu ảnh check-in tại: {face_path}")
                                            else:
                                                record.check_in_image_url = relative_face_path # Lưu đường dẫn tương đối đã chuẩn hóa
                                                record.save() # Lưu record chỉ khi ảnh lưu thành công (hoặc di chuyển save ra ngoài if này nếu ảnh không bắt buộc)
                                                last_save_time[person_name] = now # Cập nhật thời gian chỉ khi save thành công
                                                print(f"[PROCESS INFO] Đã lưu check-in cho '{person_name}'")
                                                
                                                # Đẩy dữ liệu lên Firebase sau khi check-in
                                                try:
                                                    success = push_attendance_to_firebase(record, camera_name)
                                                    if success:
                                                        print(f"[PROCESS INFO] Đã đẩy dữ liệu check-in lên Firebase cho '{person_name}'")
                                                    else:
                                                        print(f"[PROCESS ERROR] Không thể đẩy dữ liệu lên Firebase cho '{person_name}'")
                                                except Exception as firebase_err:
                                                    print(f"[PROCESS ERROR] Lỗi khi đẩy dữ liệu check-in lên Firebase: {firebase_err}")
                                        else:
                                            # Người dùng đã check-in trước đó, cập nhật thành check-out
                                            # Lưu thời gian nhận diện gần nhất làm check-out time
                                            record.check_out = now
                                            
                                            # Lưu ảnh check-out
                                            face_filename = f'{sanitized_person_name_for_file}_{today}_{now.strftime("%H%M%S")}_out.jpg'
                                            face_path = os.path.join(settings.MEDIA_ROOT, settings.RECOGNITION_ATTENDANCE_FACES_DIR, 
                                                                    settings.RECOGNITION_CHECK_OUT_SUBDIR, face_filename)
                                            relative_face_path = os.path.join(settings.RECOGNITION_ATTENDANCE_FACES_DIR, 
                                                                        settings.RECOGNITION_CHECK_OUT_SUBDIR, face_filename)
                                            
                                            os.makedirs(os.path.dirname(face_path), exist_ok=True)
                                            saved_img_out = cv2.imwrite(face_path, face_aligned)
                                            
                                            if not saved_img_out:
                                                print(f"[PROCESS ERROR] Không thể lưu ảnh check-out tại: {face_path}")
                                            else:
                                                record.check_out_image_url = relative_face_path
                                                record.save()  # Lưu record với thông tin check-out mới
                                                last_save_time[person_name] = now
                                                
                                                check_in_time_str = record.check_in.strftime('%H:%M:%S') if record.check_in else "không rõ"
                                                check_out_time_str = record.check_out.strftime('%H:%M:%S')
                                                print(f"[PROCESS INFO] Đã cập nhật check-out cho '{person_name}' (check-in: {check_in_time_str}, check-out: {check_out_time_str})")
                                                
                                                # Đẩy dữ liệu lên Firebase sau khi check-out
                                                try:
                                                    success = push_attendance_to_firebase(record, camera_name)
                                                    if success:
                                                        print(f"[PROCESS INFO] Đã đẩy dữ liệu check-out lên Firebase cho '{person_name}'")
                                                    else:
                                                        print(f"[PROCESS ERROR] Không thể đẩy dữ liệu check-out lên Firebase cho '{person_name}'")
                                                except Exception as firebase_err:
                                                    print(f"[PROCESS ERROR] Lỗi khi đẩy dữ liệu check-out lên Firebase: {firebase_err}")
                                    except Exception as e:
                                        print(f"[PROCESS ERROR] Lỗi khi lưu thông tin chấm công: {e}")
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

                    if mode == 'recognize':
                         for name in recognition_counts:
                             if name not in detected_in_frame:
                                 recognition_counts[name] = 0
                elif mode == 'recognize':
                     for name in recognition_counts:
                         recognition_counts[name] = 0

            output_handler.set_frame(frame_to_display)

            if mode == 'collect' and stop_event.is_set():
                break

            if frame_index % 100 == 0: # Print every 100 frames processed in ROI
                 print(f"[PROCESS LOOP] Frame index: {frame_index}, Mode: {mode}, stop_event: {stop_event.is_set()}")

            time.sleep(0.001)

    except Exception as e:
        print(f"[PROCESS ERROR] Lỗi trong vòng lặp xử lý: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[PROCESS INFO] Kết thúc vòng lặp, dừng video handler...")
        video_handler.stop()
        output_handler.stop_stream()

    # Đánh dấu tiến trình là không hoạt động khi kết thúc (hoặc bị dừng)
    # Sử dụng sanitized_username
    # if mode == 'collect' and sanitized_username and sanitized_username in collect_progress_tracker:
    #    collect_progress_tracker[sanitized_username]['active'] = False
    #    print(f"[Tracker END] Progress for {sanitized_username}: {collect_progress_tracker[sanitized_username]}")
    # --- Sử dụng username gốc --- 
    if mode == 'collect' and username and username in collect_progress_tracker:
        collect_progress_tracker[username]['active'] = False
        print(f"[Tracker END] Progress for {username}: {collect_progress_tracker[username]}")
    # --- Kết thúc thay đổi ---

    print("[PROCESS END] Kết thúc hàm process_video_with_roi")
    if mode == 'collect':
        # Sử dụng sanitized_username để thông báo
        # print(f"[PROCESS INFO] Đã lưu tổng cộng {sample_count} mẫu cho {sanitized_username}.")
        # --- Sử dụng username gốc --- 
        # print(f"[PROCESS INFO] Đã lưu tổng cộng {sample_count} mẫu cho {username}.")
        # --- Kết thúc sử dụng username gốc ---
        # --- Hoàn nguyên sử dụng username gốc và sanitized --- 
        print(f"[PROCESS INFO] Đã lưu tổng cộng {sample_count} mẫu cho '{username}' vào thư mục '{sanitized_username}'.")
        # --- Kết thúc hoàn nguyên ---
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

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    # ... (Phần __main__ có thể giữ nguyên hoặc cập nhật để test chế độ stream nếu muốn)
     pass # Thay pass bằng nội dung __main__ hiện tại của bạn 