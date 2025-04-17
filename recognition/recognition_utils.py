"""
Module chứa các hàm tiện ích cho recognition app
Tách riêng để tránh circular imports giữa views.py và tasks.py
"""

import os
import numpy as np
import cv2
import face_recognition
import datetime
from django.contrib.auth.models import User
from django.utils import timezone
from django.conf import settings
from users.models import Present, Time


def predict(face_aligned, svc, threshold=0.7):
    """
    Dự đoán người dùng từ ảnh khuôn mặt đã căn chỉnh
    
    Args:
        face_aligned: Ảnh khuôn mặt đã căn chỉnh
        svc: Model SVM Classifier đã train
        threshold: Ngưỡng xác suất để chấp nhận nhận diện
        
    Returns:
        (result, probability): Kết quả dự đoán (ID người dùng) và xác suất
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


def update_attendance_in_db_in(present):
    """
    Cập nhật thông tin chấm công vào của người dùng
    
    Args:
        present: Dictionary hoặc list chứa thông tin người dùng
        
    Returns:
        Bản ghi chấm công đã tạo/cập nhật
    """
    today = datetime.date.today()
    time_now = datetime.datetime.now()
    attendance_records = []
    
    # Chuyển đổi list thành dict nếu cần
    if isinstance(present, list):
        present_dict = {username: True for username in present}
    else:
        present_dict = present
    
    print(f"Cập nhật chấm công vào cho: {present_dict}")
    
    for person in present_dict:
        try:
            user = User.objects.get(username=person)
            print(f"Tìm thấy user: {user.username}")
            
            try:
                qs = Present.objects.get(user=user, date=today)
                print(f"Tìm thấy bản ghi Present hiện tại cho {user.username}")
            except Present.DoesNotExist:
                qs = None
                print(f"Chưa có bản ghi Present cho {user.username}, sẽ tạo mới")
            
            if qs is None:
                if present_dict[person]:
                    a = Present(user=user, date=today, present=True)
                    a.save()
                    print(f"Đã tạo bản ghi Present mới (present=True) cho {user.username}")
                else:
                    a = Present(user=user, date=today, present=False)
                    a.save()
                    print(f"Đã tạo bản ghi Present mới (present=False) cho {user.username}")
            else:
                if present_dict[person]:
                    qs.present = True
                    qs.save(update_fields=['present'])
                    print(f"Đã cập nhật bản ghi Present hiện tại thành present=True cho {user.username}")
            
            if present_dict[person]:
                a = Time(user=user, date=today, time=time_now, out=False)
                a.save()
                print(f"Đã tạo bản ghi Time mới (out=False) cho {user.username}")
                
                # Bổ sung trả về bản ghi chấm công nếu cần
                from recognition.models import AttendanceRecord
                try:
                    record, created = AttendanceRecord.objects.get_or_create(
                        user=user,
                        date=today,
                        defaults={
                            'check_in': time_now
                        }
                    )
                    
                    if created:
                        print(f"Đã tạo bản ghi AttendanceRecord mới với check_in={time_now} cho {user.username}")
                    elif not record.check_in:
                        record.check_in = time_now
                        record.save(update_fields=['check_in'])
                        print(f"Đã cập nhật bản ghi AttendanceRecord hiện tại với check_in={time_now} cho {user.username}")
                    else:
                        print(f"Đã tìm thấy bản ghi AttendanceRecord hiện tại với check_in={record.check_in} cho {user.username}")
                    
                    attendance_records.append(record)
                except Exception as e:
                    print(f"Lỗi khi tạo/cập nhật bản ghi AttendanceRecord: {str(e)}")
        except User.DoesNotExist:
            print(f"Không tìm thấy user với username={person}")
        except Exception as e:
            print(f"Lỗi khi xử lý chấm công cho {person}: {str(e)}")
    
    print(f"Kết quả: {len(attendance_records)} bản ghi chấm công được cập nhật")
    return attendance_records[0] if attendance_records else None


def update_attendance_in_db_out(present):
    """
    Cập nhật thông tin chấm công ra của người dùng
    
    Args:
        present: Dictionary hoặc list chứa thông tin người dùng
        
    Returns:
        Bản ghi chấm công đã cập nhật
    """
    today = datetime.date.today()
    time_now = datetime.datetime.now()
    attendance_records = []
    
    # Chuyển đổi list thành dict nếu cần
    if isinstance(present, list):
        present_dict = {username: True for username in present}
    else:
        present_dict = present
    
    print(f"Cập nhật chấm công ra cho: {present_dict}")
    
    for person in present_dict:
        try:
            user = User.objects.get(username=person)
            print(f"Tìm thấy user: {user.username}")
            
            if present_dict[person]:
                a = Time(user=user, date=today, time=time_now, out=True)
                a.save()
                print(f"Đã tạo bản ghi Time mới (out=True) cho {user.username}")
                
                # Cập nhật bản ghi check-out và trả về
                from recognition.models import AttendanceRecord
                try:
                    record = AttendanceRecord.objects.get(user=user, date=today)
                    print(f"Tìm thấy bản ghi AttendanceRecord hiện tại cho {user.username}")
                    
                    record.check_out = time_now
                    record.save(update_fields=['check_out', 'updated_at'])
                    print(f"Đã cập nhật bản ghi với check_out={time_now} cho {user.username}")
                    
                    attendance_records.append(record)
                except AttendanceRecord.DoesNotExist:
                    # Tạo bản ghi mới với check_out nếu chưa có check_in
                    print(f"Không tìm thấy bản ghi AttendanceRecord cho {user.username}, tạo mới")
                    record = AttendanceRecord.objects.create(
                        user=user,
                        date=today,
                        check_out=time_now
                    )
                    print(f"Đã tạo bản ghi mới với check_out={time_now} cho {user.username}")
                    attendance_records.append(record)
                except Exception as e:
                    print(f"Lỗi khi tạo/cập nhật bản ghi AttendanceRecord: {str(e)}")
        except User.DoesNotExist:
            print(f"Không tìm thấy user với username={person}")
        except Exception as e:
            print(f"Lỗi khi xử lý chấm công ra cho {person}: {str(e)}")
    
    print(f"Kết quả: {len(attendance_records)} bản ghi chấm công được cập nhật")
    return attendance_records[0] if attendance_records else None 