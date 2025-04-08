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
import argparse

def predict(face_aligned, svc, threshold=0.7):
    """
    Dự đoán khuôn mặt từ hình ảnh đã căn chỉnh
    """
    face_encodings = np.zeros((1, 128))
    try:
        x_face_locations = face_recognition.face_locations(face_aligned)
        faces_encodings = face_recognition.face_encodings(face_aligned, known_face_locations=x_face_locations)
        if len(faces_encodings) == 0:
            return ([-1], [0])
    except:
        return ([-1], [0])

    prob = svc.predict_proba(faces_encodings)
    result = np.where(prob[0] == np.amax(prob[0]))
    if prob[0][result[0]] <= threshold:
        return ([-1], prob[0][result[0]])
    return (result[0], prob[0][result[0]])

def collect_face_from_video(username, video_path):
    """
    Thu thập dữ liệu khuôn mặt từ video
    """
    if not os.path.exists(video_path):
        print(f"Không tìm thấy video tại đường dẫn: {video_path}")
        return
    
    # Tạo thư mục để lưu hình ảnh khuôn mặt
    if not os.path.exists(f'face_recognition_data/training_dataset/{username}/'):
        os.makedirs(f'face_recognition_data/training_dataset/{username}/')
    directory = f'face_recognition_data/training_dataset/{username}/'
    
    # Khởi tạo các công cụ phát hiện khuôn mặt
    print("[INFO] Đang tải bộ phát hiện khuôn mặt...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video")
        return
    
    print(f"[INFO] Đang xử lý video: {video_path}")
    
    frame_count = 0
    sample_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Xử lý mỗi 5 frame để tăng tốc độ
        if frame_count % 5 != 0:
            continue
        
        frame = imutils.resize(frame, width=800)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)
        
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_aligned = fa.align(frame, gray_frame, face)
            sample_count += 1
            
            if face is None:
                print("Không phát hiện khuôn mặt")
                continue
            
            # Lưu hình ảnh khuôn mặt
            cv2.imwrite(directory + '/' + str(sample_count) + '.jpg', face_aligned)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Sample: {sample_count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Hiển thị frame
            cv2.imshow("Thu thập dữ liệu", frame)
            cv2.waitKey(1)
            
            # Dừng khi đủ mẫu
            if sample_count >= 300:
                break
                
        if sample_count >= 300:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Đã thu thập {sample_count} mẫu khuôn mặt cho {username}")

def recognize_from_video(video_path):
    """
    Nhận diện khuôn mặt từ video
    """
    if not os.path.exists(video_path):
        print(f"Không tìm thấy video tại đường dẫn: {video_path}")
        return
    
    # Tải model đã huấn luyện
    print("[INFO] Đang tải mô hình...")
    
    # Kiểm tra xem các file cần thiết đã tồn tại chưa
    if not os.path.exists("face_recognition_data/svc.sav"):
        print("Chưa có model được huấn luyện. Vui lòng huấn luyện trước.")
        return
    
    if not os.path.exists("face_recognition_data/classes.npy"):
        print("Chưa có dữ liệu classes. Vui lòng huấn luyện trước.")
        return
    
    # Tải model
    with open("face_recognition_data/svc.sav", 'rb') as f:
        svc = pickle.load(f)
    
    # Tải detector và aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    
    # Tải encoder
    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')
    
    # Khởi tạo các biến để theo dõi nhận diện
    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    
    for i in range(no_of_faces):
        name = encoder.inverse_transform([i])[0]
        count[name] = 0
        present[name] = False
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video")
        return
    
    print(f"[INFO] Đang xử lý video: {video_path}")
    
    # Tạo video output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = video_path.replace('.mp4', '_recognized.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Xử lý mỗi 3 frame để tăng tốc độ
        if frame_count % 3 != 0:
            out.write(frame)
            continue
        
        frame_copy = frame.copy()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)
        
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_aligned = fa.align(frame, gray_frame, face)
            
            # Dự đoán khuôn mặt
            (pred, prob) = predict(face_aligned, svc)
            
            if pred != [-1]:
                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                pred = person_name
                
                if count.get(pred, 0) == 0:
                    count[pred] = count.get(pred, 0) + 1
                    
                # Nếu nhận diện được 3 lần liên tiếp
                if count.get(pred, 0) >= 3:
                    present[pred] = True
                    count[pred] = 0  # reset
                else:
                    count[pred] = count.get(pred, 0) + 1
                
                # Vẽ bounding box và tên
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                prob_value = float(prob) if isinstance(prob, (int, float, np.number)) else float(prob[0]) if len(prob) > 0 else 0.0
                cv2.putText(frame, f"{person_name} ({prob_value:.2f})", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Nếu không nhận diện được
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Hiển thị frame
        cv2.imshow("Nhận diện", frame)
        out.write(frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    # Hiển thị những người có mặt
    print("\nDanh sách người được nhận diện:")
    for person, status in present.items():
        if status:
            print(f"- {person}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Đã lưu video đã nhận diện tại: {out_path}")

def main():
    parser = argparse.ArgumentParser(description='Test nhận dạng khuôn mặt từ video')
    parser.add_argument('--mode', type=str, required=True, choices=['collect', 'recognize'],
                        help='Mode: collect (thu thập dữ liệu) hoặc recognize (nhận diện)')
    parser.add_argument('--video', type=str, required=True,
                        help='Đường dẫn tới file video')
    parser.add_argument('--username', type=str,
                        help='Username của người dùng (chỉ cần cho mode collect)')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        if not args.username:
            print("Vui lòng cung cấp username với tham số --username")
            return
        collect_face_from_video(args.username, args.video)
    elif args.mode == 'recognize':
        recognize_from_video(args.video)

if __name__ == "__main__":
    main() 