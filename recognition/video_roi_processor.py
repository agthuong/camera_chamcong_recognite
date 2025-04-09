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

# --- Constants and Paths --- (Adjust if necessary)
PREDICTOR_PATH = 'face_recognition_data/shape_predictor_68_face_landmarks.dat'
SVC_PATH = "face_recognition_data/svc.sav"
CLASSES_PATH = 'face_recognition_data/classes.npy'
TRAINING_DATA_DIR = 'face_recognition_data/training_dataset/'
FACE_WIDTH = 96 # Width for aligned faces
FRAME_WIDTH = 800 # Width to resize video frames
DEFAULT_MAX_SAMPLES = 150 # Default samples to collect
DEFAULT_RECOGNITION_THRESHOLD = 3 # Default consecutive detections for recognition
FRAME_SKIP = 3 # Process every Nth frame

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
processing_thread = None
stop_processing_event = threading.Event()

def predict(face_aligned, svc, threshold=0.7):
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
        # prob is like [[0.1, 0.8, 0.1]]
        best_class_index = np.argmax(prob[0])
        best_class_probability = prob[0][best_class_index]

        # print(f"Debug: Probabilities: {prob[0]}, Best Index: {best_class_index}, Prob: {best_class_probability:.2f}")

        if best_class_probability >= threshold:
            return ([best_class_index], [best_class_probability])
        else:
            # print(f"Debug: Probability {best_class_probability:.2f} below threshold {threshold}")
            return ([-1], [best_class_probability]) # Return -1 but still provide the probability

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
        frame_display = imutils.resize(frame, width=FRAME_WIDTH).copy()
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
            try:
                self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 giây timeout
            except AttributeError:
                print("[INFO] cv2.CAP_PROP_OPEN_TIMEOUT_MSEC không khả dụng trong phiên bản OpenCV này")
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
                           max_samples=DEFAULT_MAX_SAMPLES,
                           recognition_threshold=DEFAULT_RECOGNITION_THRESHOLD):
    """
    Xử lý video (thu thập hoặc nhận diện) trong vùng ROI đã chọn.
    Đưa frame đã xử lý (đã vẽ hoặc gốc) vào output_handler để stream.

    Args:
        video_source: Đường dẫn video hoặc ID webcam
        mode: 'collect', 'recognize', hoặc 'stream' (mới)
        roi: Tuple (x, y, w, h) xác định vùng ROI
        stop_event: threading.Event để dừng xử lý từ bên ngoài
        output_handler: Instance của StreamOutput để đặt frame JPEG
        username: Bắt buộc nếu mode là 'collect'
        max_samples: Số mẫu tối đa cần thu thập
        recognition_threshold: Ngưỡng nhận diện
    """
    output_handler.start_stream() # Báo cho output handler biết là đang stream

    if mode == 'collect' and not username:
        print("Lỗi: Cần cung cấp username cho chế độ 'collect'")
        output_handler.stop_stream()
        return 0
    if roi is None and mode != 'stream': # Cho phép stream không cần ROI
         print("Lỗi: Cần có ROI cho chế độ 'collect' hoặc 'recognize'")
         output_handler.stop_stream()
         return 0 if mode == 'collect' else {}
    elif roi is not None and len(roi) != 4:
         print("Lỗi: ROI không hợp lệ")
         output_handler.stop_stream()
         return 0 if mode == 'collect' else {}

    # Giải nén ROI (nếu có)
    rx, ry, rw, rh = (0,0,0,0) # Giá trị mặc định nếu không có ROI (chế độ stream)
    if roi:
        rx, ry, rw, rh = [int(v) for v in roi]

    # Khởi tạo các công cụ phát hiện khuôn mặt (chỉ khi cần)
    detector, predictor, fa = None, None, None
    svc, encoder = None, None
    if mode in ['collect', 'recognize']:
        print("[INFO] Đang tải bộ phát hiện khuôn mặt...")
        try:
            detector = dlib.get_frontal_face_detector()
            if not os.path.exists(PREDICTOR_PATH):
                print(f"Lỗi: Không tìm thấy file shape predictor tại: {PREDICTOR_PATH}")
                output_handler.stop_stream()
                return 0 if mode == 'collect' else {}
            predictor = dlib.shape_predictor(PREDICTOR_PATH)
            fa = FaceAligner(predictor, desiredFaceWidth=FACE_WIDTH)
        except Exception as e:
            print(f"Lỗi khi khởi tạo dlib/FaceAligner: {e}")
            output_handler.stop_stream()
            return 0 if mode == 'collect' else {}

    # Thiết lập dựa trên mode
    sample_count = 0
    recognized_persons = {}  # Cho chế độ recognize
    recognition_counts = {}  # Theo dõi số lần nhận diện liên tiếp
    output_dir = None
    last_save_time = {}  # Dictionary để lưu thời gian lưu cuối cùng cho mỗi người

    if mode == 'collect':
        output_dir = os.path.join(TRAINING_DATA_DIR, username)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Đã tạo thư mục: {output_dir}")
        print(f"[INFO] Chế độ COLLECT: Sẽ lưu tối đa {max_samples} mẫu vào '{output_dir}'")

    elif mode == 'recognize':
        # Tải model SVC và encoder
        if not os.path.exists(SVC_PATH) or not os.path.exists(CLASSES_PATH):
            print("Lỗi: Không tìm thấy file model (svc.sav) hoặc classes (classes.npy). Vui lòng huấn luyện trước.")
            output_handler.stop_stream()
            return {}
        try:
            with open(SVC_PATH, 'rb') as f:
                svc = pickle.load(f)
            encoder = LabelEncoder()
            encoder.classes_ = np.load(CLASSES_PATH)
            # Khởi tạo dictionary theo dõi
            for name in encoder.classes_:
                recognized_persons[name] = False
                recognition_counts[name] = 0
                last_save_time[name] = None  # Khởi tạo thời gian lưu cuối cùng
            print(f"[INFO] Chế độ RECOGNIZE: Đã tải model cho {len(encoder.classes_)} người.")
        except Exception as e:
            print(f"Lỗi khi tải model/classes: {e}")
            output_handler.stop_stream()
            return {}

    # Khởi tạo VideoSourceHandler
    print(f"[INFO] Khởi tạo luồng video từ nguồn: {video_source}")
    video_handler = VideoSourceHandler(video_source)
    if not video_handler.start():
        print(f"[ERROR] Không thể khởi động VideoSourceHandler cho nguồn: {video_source}")
        output_handler.stop_stream()
        return 0 if mode == 'collect' else recognized_persons

    frame_index = 0
    print(f"[INFO] Bắt đầu xử lý video (Mode: {mode})...")
    try:
        while not stop_event.is_set(): # Kiểm tra sự kiện dừng
            frame = video_handler.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_to_display = frame.copy() # Frame để vẽ lên và gửi đi
            roi_frame_for_processing = None # Frame con ROI để xử lý

            # Nếu có ROI, crop frame để xử lý và vẽ hình chữ nhật lên frame hiển thị
            if roi:
                orig_h, orig_w = frame.shape[:2]
                scale = orig_w / FRAME_WIDTH # Giả sử ROI được chọn trên frame có width=FRAME_WIDTH
                orig_rx = int(rx * scale)
                orig_ry = int(ry * scale)
                orig_rw = int(rw * scale)
                orig_rh = int(rh * scale)
                orig_rx = max(0, orig_rx)
                orig_ry = max(0, orig_ry)
                orig_rw = min(orig_rw, orig_w - orig_rx)
                orig_rh = min(orig_rh, orig_h - orig_ry)

                if orig_rw > 0 and orig_rh > 0:
                    # Vẽ ROI lên frame_to_display
                    cv2.rectangle(frame_to_display, (orig_rx, orig_ry), (orig_rx + orig_rw, orig_ry + orig_rh), (255, 0, 0), 2)
                    try:
                        # Chỉ lấy ROI để xử lý nếu mode là collect hoặc recognize
                        if mode in ['collect', 'recognize']:
                            roi_frame_for_processing = frame[orig_ry:orig_ry + orig_rh, orig_rx:orig_rx + orig_rw].copy()
                            if roi_frame_for_processing.size == 0:
                                roi_frame_for_processing = None # Đặt lại nếu ROI rỗng
                    except Exception as e:
                         print(f"[ERROR] Lỗi khi crop ROI: {e}")
                         roi_frame_for_processing = None
                else:
                    roi_frame_for_processing = None # ROI không hợp lệ

            # Chỉ xử lý frame ROI nếu cần và ROI hợp lệ
            if mode in ['collect', 'recognize'] and roi_frame_for_processing is not None:
                frame_index += 1
                if frame_index % FRAME_SKIP != 0:
                    output_handler.set_frame(frame_to_display) # Vẫn gửi frame gốc có vẽ ROI
                    continue

                try:
                    # Resize roi_frame về kích thước chuẩn để xử lý (nếu cần)
                    # roi_frame_resized = imutils.resize(roi_frame_for_processing, width=FACE_WIDTH*2) # Ví dụ
                    roi_frame_resized = roi_frame_for_processing # Sử dụng ROI gốc
                    roi_gray = cv2.cvtColor(roi_frame_resized, cv2.COLOR_BGR2GRAY)
                    faces = detector(roi_gray, 0)
                except Exception as detect_err:
                    print(f"[ERROR] Lỗi trong quá trình phát hiện khuôn mặt: {detect_err}")
                    faces = [] # Coi như không có khuôn mặt nếu lỗi

                detected_in_frame = set() # Reset cho mỗi frame xử lý
                if faces:
                    for face in faces:
                        (fx, fy, fw, fh) = face_utils.rect_to_bb(face)
                        # Tọa độ để vẽ trên frame gốc (frame_to_display)
                        draw_x = orig_rx + int(fx * (orig_rw / roi_frame_resized.shape[1]))
                        draw_y = orig_ry + int(fy * (orig_rh / roi_frame_resized.shape[0]))
                        draw_w = int(fw * (orig_rw / roi_frame_resized.shape[1]))
                        draw_h = int(fh * (orig_rh / roi_frame_resized.shape[0]))

                        try:
                            face_aligned = fa.align(roi_frame_resized, roi_gray, face)
                        except Exception as align_err:
                            print(f"[ERROR] Lỗi căn chỉnh mặt: {align_err}")
                            continue

                        if mode == 'collect':
                            sample_count += 1
                            img_path = os.path.join(output_dir, f"{username}_{sample_count}.jpg")
                            cv2.imwrite(img_path, face_aligned)

                            # Vẽ lên frame_to_display
                            cv2.rectangle(frame_to_display, (draw_x, draw_y), (draw_x + draw_w, draw_y + draw_h), (0, 255, 0), 1)
                            cv2.putText(frame_to_display, f"Sample: {sample_count}", (draw_x, draw_y - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                            if sample_count >= max_samples:
                                print(f"[INFO] Đã thu thập đủ {max_samples} mẫu.")
                                stop_event.set() # Dừng xử lý
                                break

                        elif mode == 'recognize':
                            (pred_idx, prob) = predict(face_aligned, svc)
                            prob_value = float(prob[0])
                            person_name = "Unknown"
                            color = (0, 0, 255) # Đỏ cho Unknown

                            if pred_idx != [-1]:
                                person_name = encoder.inverse_transform(pred_idx)[0]
                                detected_in_frame.add(person_name)
                                color = (0, 255, 0) # Xanh lá nếu nhận diện được

                                if prob_value >= 0.5: # Chỉ tăng count nếu prob đủ cao
                                    recognition_counts[person_name] = recognition_counts.get(person_name, 0) + 1
                                else:
                                    recognition_counts[person_name] = 0 # Reset nếu không chắc chắn

                                if recognition_counts[person_name] >= recognition_threshold:
                                    if not recognized_persons.get(person_name, False):
                                        print(f"[RECOGNIZED] {person_name}")
                                        recognized_persons[person_name] = True
                                    # Xử lý chấm công (giữ nguyên logic)
                                    # ... (logic chấm công hiện tại của bạn) ...
                                    try:
                                        # Tìm hoặc tạo người dùng
                                        try:
                                            user = User.objects.get(username=person_name)
                                        except User.DoesNotExist:
                                            print(f"[INFO] Tạo mới người dùng '{person_name}' trong cơ sở dữ liệu.")
                                            user = User.objects.create_user(
                                                username=person_name,
                                                password=f"default_{person_name}",
                                                first_name=person_name
                                            )

                                        now = timezone.now()
                                        today = now.date()

                                        # Tìm hoặc tạo bản ghi chấm công
                                        record, created = AttendanceRecord.objects.get_or_create(
                                            user=user,
                                            date=today,
                                            defaults={'check_in': now}
                                        )

                                        if created:
                                            # Nếu là check in đầu tiên
                                            face_path = os.path.join(settings.MEDIA_ROOT, 'attendance_faces', 'check_in', f'{person_name}_{today}_{now.strftime("%H%M%S")}.jpg')
                                            os.makedirs(os.path.dirname(face_path), exist_ok=True)
                                            cv2.imwrite(face_path, face_aligned)
                                            record.check_in_face = os.path.join('attendance_faces', 'check_in', f'{person_name}_{today}_{now.strftime("%H%M%S")}.jpg')
                                            record.save()
                                            last_save_time[person_name] = now
                                            print(f"[INFO] Đã lưu check-in cho '{person_name}'")
                                        else:
                                            # Luôn cập nhật check out
                                            last_saved = last_save_time.get(person_name) # Lấy thời gian lưu cuối cùng
                                            # Chỉ cập nhật check out nếu đã qua 10 giây kể từ lần lưu cuối cùng
                                            if last_saved is None or (now - last_saved).total_seconds() >= 10:
                                                record.check_out = now
                                                face_path = os.path.join(settings.MEDIA_ROOT, 'attendance_faces', 'check_out', f'{person_name}_{today}_{now.strftime("%H%M%S")}.jpg')
                                                os.makedirs(os.path.dirname(face_path), exist_ok=True)
                                                cv2.imwrite(face_path, face_aligned)
                                                record.check_out_face = os.path.join('attendance_faces', 'check_out', f'{person_name}_{today}_{now.strftime("%H%M%S")}.jpg')
                                                record.save()
                                                last_save_time[person_name] = now # Cập nhật thời gian lưu cuối
                                                print(f"[INFO] Đã cập nhật check-out cho '{person_name}'")

                                    except Exception as e:
                                        print(f"[ERROR] Lỗi khi lưu thông tin chấm công: {e}")
                                        import traceback
                                        traceback.print_exc()

                                    color = (0, 255, 255) # Vàng nếu đã xác nhận
                            else: # pred_idx == [-1]
                                if prob_value > 0: # Hiển thị prob nếu có
                                    person_name = f"Unknown ({prob_value:.2f})"
                                else:
                                    person_name = "Unknown (0.00)"

                            # Vẽ bounding box và tên lên frame_to_display
                            cv2.rectangle(frame_to_display, (draw_x, draw_y), (draw_x + draw_w, draw_y + draw_h), color, 1)
                            cv2.putText(frame_to_display, person_name, (draw_x, draw_y - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    # Reset recognition_counts cho những người không được phát hiện trong frame này
                    if mode == 'recognize':
                         for name in recognition_counts:
                             if name not in detected_in_frame:
                                 recognition_counts[name] = 0
                # Nếu không có khuôn mặt nào được phát hiện trong frame ROI
                elif mode == 'recognize':
                     # Reset tất cả recognition_counts
                     for name in recognition_counts:
                         recognition_counts[name] = 0

            # --- Kết thúc phần xử lý ROI ---

            # Đặt frame cuối cùng (đã vẽ nếu cần) vào output_handler
            output_handler.set_frame(frame_to_display)

            # Ngắt nếu chế độ collect đã xong
            if mode == 'collect' and stop_event.is_set():
                break

            # Thêm một khoảng nghỉ nhỏ để giảm tải CPU
            time.sleep(0.001)

    except Exception as e:
        print(f"[ERROR] Lỗi không mong đợi trong quá trình xử lý: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Dừng xử lý video thread.")
        video_handler.stop()
        output_handler.stop_stream() # Báo dừng stream và xóa frame cuối

    # Trả về kết quả tùy theo mode
    if mode == 'collect':
        print(f"[INFO] Đã lưu tổng cộng {sample_count} mẫu cho {username}.")
        return sample_count
    elif mode == 'recognize':
        # ... (giữ nguyên phần trả về kết quả recognize) ...
        print("--- Kết quả Nhận diện Cuối cùng (khi dừng) ---")
        final_recognized = {name: status for name, status in recognized_persons.items() if status}
        if not final_recognized:
            print("Không có ai được nhận diện trong suốt quá trình.")
        else:
            for name in final_recognized:
                print(f"- {name}")
        print("------------------------------------")
        return final_recognized
    else: # mode == 'stream'
         return {} # Không có kết quả cụ thể để trả về


# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    # ... (Phần __main__ có thể giữ nguyên hoặc cập nhật để test chế độ stream nếu muốn)
     pass # Thay pass bằng nội dung __main__ hiện tại của bạn 