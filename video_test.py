import cv2

cap = cv2.VideoCapture("test_face_resize.mp4")
print("Video opened:", cap.isOpened())

if cap.isOpened():
    ret, frame = cap.read()
    print("Read frame:", ret, "Frame shape:", frame.shape if ret else None)

cap.release()