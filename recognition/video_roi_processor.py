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

def process_video_with_roi(video_source, mode, roi, username=None,
                           max_samples=DEFAULT_MAX_SAMPLES,
                           recognition_threshold=DEFAULT_RECOGNITION_THRESHOLD,
                           show_window=True):
    """
    Xử lý video (thu thập hoặc nhận diện) trong vùng ROI đã chọn.
    Chỉ gửi vùng ROI cho mô hình nhận diện, bỏ qua phần còn lại của frame.

    Args:
        video_source: Đường dẫn video hoặc ID webcam
        mode: 'collect' hoặc 'recognize'
        roi: Tuple (x, y, w, h) xác định vùng ROI
        username: Bắt buộc nếu mode là 'collect'
        max_samples: Số mẫu tối đa cần thu thập
        recognition_threshold: Ngưỡng nhận diện
        show_window: Hiển thị cửa sổ OpenCV hay không
    """
    if mode == 'collect' and not username:
        print("Lỗi: Cần cung cấp username cho chế độ 'collect'")
        return 0
    if not roi or len(roi) != 4:
        print("Lỗi: ROI không hợp lệ")
        return 0 if mode == 'collect' else {}

    # Giải nén ROI
    rx, ry, rw, rh = [int(v) for v in roi]

    # Khởi tạo các công cụ phát hiện khuôn mặt
    print("[INFO] Đang tải bộ phát hiện khuôn mặt...")
    try:
        detector = dlib.get_frontal_face_detector()
        if not os.path.exists(PREDICTOR_PATH):
            print(f"Lỗi: Không tìm thấy file shape predictor tại: {PREDICTOR_PATH}")
            return 0 if mode == 'collect' else {}
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        fa = FaceAligner(predictor, desiredFaceWidth=FACE_WIDTH)
    except Exception as e:
        print(f"Lỗi khi khởi tạo dlib/FaceAligner: {e}")
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
            return {}

    # Khởi tạo VideoSourceHandler thay vì sử dụng cv2.VideoCapture trực tiếp
    print(f"[INFO] Khởi tạo luồng video từ nguồn: {video_source}")
    video_handler = VideoSourceHandler(video_source)
    if not video_handler.start():
        print(f"[ERROR] Không thể khởi động VideoSourceHandler cho nguồn: {video_source}")
        return 0 if mode == 'collect' else recognized_persons
    
    # Xử lý video frames
    frame_index = 0
    window_name = "Xu Ly Trong ROI"
    if show_window:
        cv2.namedWindow(window_name)

    try:
        # Biến đếm cho việc gỡ lỗi
        empty_frame_count = 0
        processed_frame_count = 0
        detection_success_count = 0
        face_detected_count = 0
        
        # Thời gian theo dõi FPS
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        print("[INFO] Bắt đầu xử lý video...")
        
        while True:
            # Lấy frame từ VideoSourceHandler thay vì đọc trực tiếp
            frame = video_handler.get_frame()
            if frame is None:
                empty_frame_count += 1
                if empty_frame_count % 100 == 0:  # In thông báo mỗi 100 frame rỗng
                    print(f"[WARNING] Không lấy được frame từ VideoSourceHandler ({empty_frame_count} lần)")
                time.sleep(0.01)  # Đợi một chút trước khi thử lại
                continue
            
            # Reset biến đếm frame rỗng nếu đã nhận được frame
            if empty_frame_count > 0:
                print(f"[INFO] Đã tiếp tục nhận được frame sau {empty_frame_count} lần thất bại")
                empty_frame_count = 0
            
            # Tính FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:  # Tính FPS mỗi 10 frame
                current_time = time.time()
                elapsed_time = current_time - fps_start_time
                if elapsed_time > 0:
                    fps = fps_frame_count / elapsed_time
                    fps_frame_count = 0
                    fps_start_time = current_time
                    # Chỉ in FPS mỗi 50 frame để tránh làm tràn log
                    if processed_frame_count % 50 == 0:
                        print(f"[INFO] FPS: {fps:.1f}, Detected faces: {face_detected_count}/{processed_frame_count} frames")

            frame_index += 1
            if frame_index % FRAME_SKIP != 0:  # Bỏ qua một số frame
                continue
                
            processed_frame_count += 1

            # Lấy kích thước gốc của frame
            orig_h, orig_w = frame.shape[:2]

            # Tính toán tỉ lệ scale giữa frame gốc và frame lúc chọn ROI (FRAME_WIDTH)
            scale = orig_w / FRAME_WIDTH

            # Chuyển đổi tọa độ ROI về hệ tọa độ gốc
            orig_rx = int(rx * scale)
            orig_ry = int(ry * scale) 
            orig_rw = int(rw * scale)
            orig_rh = int(rh * scale)

            # Đảm bảo tọa độ nằm trong khung hình gốc
            orig_rx = max(0, orig_rx)
            orig_ry = max(0, orig_ry)
            orig_rw = min(orig_rw, orig_w - orig_rx)
            orig_rh = min(orig_rh, orig_h - orig_ry)

            if orig_rw <= 0 or orig_rh <= 0:
                print(f"[WARNING] Kích thước ROI không hợp lệ: ({orig_rx}, {orig_ry}, {orig_rw}, {orig_rh}), frame size: {orig_w}x{orig_h}, bỏ qua frame.")
                continue

            # Crop trên frame gốc
            try:
                roi_frame_orig_crop = frame[orig_ry:orig_ry + orig_rh, orig_rx:orig_rx + orig_rw]
                if roi_frame_orig_crop.size == 0:
                    print(f"[WARNING] ROI frame bị rỗng, bỏ qua frame.")
                    continue
            except Exception as e:
                print(f"[ERROR] Lỗi khi xử lý ROI: {e}")
                continue

            # Resize ROI đã crop về kích thước phù hợp để xử lý
            try:
                # Thử tính tỷ lệ khung hình để không bị biến dạng
                aspect_ratio = roi_frame_orig_crop.shape[1] / roi_frame_orig_crop.shape[0]
                target_width = 1280
                target_height = int(target_width / aspect_ratio)
                
                # Đảm bảo kích thước không quá lớn hoặc quá nhỏ
                if target_height > 960:
                    target_height = 960
                    target_width = int(target_height * aspect_ratio)
                    
                roi_frame = cv2.resize(roi_frame_orig_crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                
                # Hiển thị thông tin trên ROI frame nếu debug mode
                if show_window and processed_frame_count % 30 == 0:  # Mỗi 30 frame
                    # Chỉ hiển thị thông tin FPS mỗi 30 frame
                    cv2.putText(roi_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
            except Exception as e:
                print(f"[ERROR] Lỗi khi resize/chuyển đổi màu: {e}")
                continue

            # Phát hiện khuôn mặt trong ROI
            try:
                faces = detector(roi_gray, 0)
                detection_success_count += 1
                
                if faces:
                    face_detected_count += 1
                    
                if not faces:
                    if mode == 'recognize':
                        for name in recognition_counts:
                            recognition_counts[name] = 0
                else:
                    detected_in_frame = set()

                    for face in faces:
                        (fx, fy, fw, fh) = face_utils.rect_to_bb(face)

                        try:
                            face_aligned = fa.align(roi_frame, roi_gray, face)
                        except Exception as e:
                            print(f"[ERROR] Lỗi khi căn chỉnh khuôn mặt: {e}")
                            continue

                        if mode == 'collect':
                            sample_count += 1
                            img_path = os.path.join(output_dir, f"{username}_{sample_count}.jpg")
                            cv2.imwrite(img_path, face_aligned)

                            cv2.rectangle(roi_frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 1)
                            cv2.putText(roi_frame, f"Sample: {sample_count}", (fx, fy - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                            if sample_count >= max_samples:
                                print(f"[INFO] Đã thu thập đủ {max_samples} mẫu.")
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
                                    
                                    # Xử lý chấm công - không thay đổi
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
                                            if last_save_time[person_name] is None or (now - last_save_time[person_name]).total_seconds() >= 10:
                                                record.check_out = now
                                                face_path = os.path.join(settings.MEDIA_ROOT, 'attendance_faces', 'check_out', f'{person_name}_{today}_{now.strftime("%H%M%S")}.jpg')
                                                os.makedirs(os.path.dirname(face_path), exist_ok=True)
                                                cv2.imwrite(face_path, face_aligned)
                                                record.check_out_face = os.path.join('attendance_faces', 'check_out', f'{person_name}_{today}_{now.strftime("%H%M%S")}.jpg')
                                                record.save()
                                                last_save_time[person_name] = now
                                                print(f"[INFO] Đã cập nhật check-out cho '{person_name}' - khoảng cách từ lần lưu trước: {(now - last_save_time.get(person_name, now)).total_seconds():.1f}s")
                                        
                                    except Exception as e:
                                        print(f"[ERROR] Lỗi khi lưu thông tin chấm công: {e}")
                                        import traceback
                                        traceback.print_exc()
                                    
                                    color = (0, 255, 255)
                            else:
                                if prob_value > 0:
                                    person_name = f"Unknown ({prob_value:.2f})"
                                else:
                                    person_name = "Unknown (0.00)"

                            cv2.rectangle(roi_frame, (fx, fy), (fx + fw, fy + fh), color, 1)
                            cv2.putText(roi_frame, f"{person_name}", (fx, fy - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    if mode == 'recognize' and faces:
                        for name in recognition_counts:
                            if name not in detected_in_frame:
                                recognition_counts[name] = 0
            except Exception as e:
                print(f"[ERROR] Lỗi khi phát hiện khuôn mặt: {e}")
                continue

            # Hiển thị thông tin debug
            if show_window and processed_frame_count % 30 == 0:
                # Thêm thông tin FPS và trạng thái vào frame
                cv2.putText(roi_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(roi_frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            if show_window:
                cv2.imshow(window_name, roi_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Người dùng nhấn 'q', dừng xử lý.")
                    break

            if mode == 'collect' and sample_count >= max_samples:
                break

    except KeyboardInterrupt:
        print("[INFO] Người dùng ngắt chương trình.")
    except Exception as e:
        print(f"[ERROR] Lỗi không mong đợi trong quá trình xử lý: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Đảm bảo luôn giải phóng tài nguyên
        video_handler.stop()
        if show_window:
            cv2.destroyAllWindows()
        print("[INFO] Hoàn tất xử lý video.")

    if mode == 'collect':
        print(f"[INFO] Đã lưu tổng cộng {sample_count} mẫu cho {username}.")
        return sample_count
    elif mode == 'recognize':
        print("\n--- Kết quả Nhận diện Cuối cùng ---")
        final_recognized = {name: status for name, status in recognized_persons.items() if status}
        if not final_recognized:
            print("Không có ai được nhận diện.")
        else:
            for name in final_recognized:
                print(f"- {name}")
        print("------------------------------------")
        return final_recognized

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    print("Chạy module video_roi_processor.py trực tiếp để test...")

    # Test Case 1: Collect from webcam
    test_source = 0 # Webcam ID
    test_mode = 'collect'
    test_username = 'test_user_roi'

    # Test Case 2: Recognize from video file
    # test_source = 'path/to/your/test_video.mp4' # <<<=== IMPORTANT: Replace with your video path
    # test_mode = 'recognize'
    # test_username = None # Not needed for recognize

    print(f"\nĐang test với Nguồn: {test_source}, Chế độ: {test_mode}")

    # Check if source exists only if it's not an integer (webcam ID)
    if not isinstance(test_source, int) and not os.path.exists(test_source):
         print(f"Lỗi: File video test '{test_source}' không tồn tại. Vui lòng cập nhật đường dẫn trong __main__.")
    else:
        # 1. Select ROI
        # Pass the video source directly
        selected_roi = select_roi_from_source(test_source) 

        # 2. Process if ROI was selected
        if selected_roi:
            print(f"\nBắt đầu xử lý với ROI: {selected_roi}")
            result = process_video_with_roi(
                video_source=test_source,
                mode=test_mode,
                roi=selected_roi,
                username=test_username,
                show_window=True # Show the processing window during test
            )
            print(f"\nKết quả xử lý ({test_mode}):")
            print(result)
        else:
            print("Không có ROI nào được chọn, kết thúc test.") 