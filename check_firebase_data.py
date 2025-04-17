#!/usr/bin/env python
"""
Script kiểm tra dữ liệu trên Firebase và hiển thị chi tiết
"""
import os
import sys
import django
import json
from datetime import datetime

# Thiết lập Django để có thể sử dụng các model
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system_facial_recognition.settings')
django.setup()

# Sau khi setup Django, import module Firebase
from recognition.firebase_util import initialize_firebase, push_attendance_to_firebase
from recognition.models import AttendanceRecord, User
from django.utils import timezone

# Hàm hiển thị thông tin dữ liệu
def print_line(char="=", length=80):
    print(char * length)

def check_firebase_connection():
    """Kiểm tra kết nối đến Firebase"""
    print_line()
    print("KIỂM TRA KẾT NỐI FIREBASE")
    print_line()
    
    db = initialize_firebase()
    if db is None:
        print("LỖI: Không thể kết nối đến Firebase. Vui lòng kiểm tra file xác thực và kết nối mạng.")
        return None
    
    print("✓ Kết nối Firebase thành công!")
    return db

def list_recent_records():
    """Hiển thị các bản ghi chấm công gần đây trong cơ sở dữ liệu"""
    print_line()
    print("CÁC BẢN GHI CHẤM CÔNG GẦN ĐÂY")
    print_line()
    
    records = AttendanceRecord.objects.all().order_by('-date', '-check_in')[:5]
    
    if not records:
        print("Không tìm thấy bản ghi chấm công nào trong cơ sở dữ liệu.")
        return None
    
    print(f"Tìm thấy {len(records)} bản ghi gần đây:")
    for i, record in enumerate(records, 1):
        print(f"\n{i}. Bản ghi: {record.user.username} - {record.date}")
        print(f"   Check-in: {record.check_in}")
        print(f"   Check-out: {record.check_out}")
        print(f"   Camera: {record.recognized_by_camera}")
    
    return records

def check_firebase_data(db, record_id=None):
    """Kiểm tra dữ liệu trên Firebase"""
    print_line()
    print("KIỂM TRA DỮ LIỆU TRÊN FIREBASE")
    print_line()
    
    if record_id:
        # Kiểm tra bản ghi cụ thể
        try:
            record = AttendanceRecord.objects.get(id=record_id)
            check_user_data(db, record.user)
        except AttendanceRecord.DoesNotExist:
            print(f"Không tìm thấy bản ghi với ID={record_id}")
    else:
        # Kiểm tra tất cả người dùng
        users = User.objects.all()
        print(f"Đang kiểm tra dữ liệu cho {users.count()} người dùng...")
        
        for user in users:
            check_user_data(db, user)

def check_user_data(db, user):
    """Kiểm tra dữ liệu của một người dùng cụ thể"""
    print_line("-")
    print(f"Kiểm tra dữ liệu cho người dùng: {user.username} (ID: {user.id})")
    
    # Đường dẫn của người dùng
    user_ref = db.collection('users').document(str(user.id))
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        print(f"⨯ Không tìm thấy dữ liệu cho người dùng {user.username} tại đường dẫn: users/{user.id}")
        return
    
    # Hiển thị thông tin người dùng
    user_data = user_doc.to_dict()
    print(f"✓ Tìm thấy dữ liệu người dùng tại users/{user.id}:")
    print(json.dumps(user_data, indent=2, ensure_ascii=False))
    
    # Kiểm tra dữ liệu chấm công
    logs_ref = user_ref.collection('attendance_logs')
    logs = logs_ref.stream()
    
    logs_count = 0
    print("\nDữ liệu chấm công:")
    
    for log in logs:
        logs_count += 1
        log_data = log.to_dict()
        print(f"\n- Ngày: {log.id}")
        print(json.dumps(log_data, indent=2, ensure_ascii=False))
    
    if logs_count == 0:
        print("⨯ Không tìm thấy dữ liệu chấm công nào!")
    else:
        print(f"\n✓ Tìm thấy {logs_count} bản ghi chấm công.")

def check_specific_user(db, username):
    """Kiểm tra dữ liệu của người dùng với username cụ thể"""
    try:
        user = User.objects.get(username=username)
        print(f"\nĐang kiểm tra dữ liệu cho người dùng: {user.username}")
        check_user_data(db, user)
    except User.DoesNotExist:
        print(f"Không tìm thấy người dùng với username: {username}")

def push_data_again(record_id):
    """Thử đẩy lại dữ liệu lên Firebase"""
    print_line()
    print(f"THỬ ĐẨY LẠI DỮ LIỆU CHO BẢN GHI ID={record_id}")
    print_line()
    
    try:
        record = AttendanceRecord.objects.get(id=record_id)
        print(f"Đang đẩy lại dữ liệu cho {record.user.username}...")
        
        # Lấy tên camera nếu có
        camera_name = record.recognized_by_camera if record.recognized_by_camera else None
        
        # Thử đẩy lại dữ liệu
        success = push_attendance_to_firebase(record, camera_name)
        
        if success:
            print(f"✓ Đẩy dữ liệu thành công cho bản ghi ID={record_id}")
        else:
            print(f"⨯ Không thể đẩy dữ liệu cho bản ghi ID={record_id}")
        
    except AttendanceRecord.DoesNotExist:
        print(f"Không tìm thấy bản ghi với ID={record_id}")

def main():
    """Hàm chính"""
    print("KIỂM TRA DỮ LIỆU FIREBASE")
    print(f"Thời gian hiện tại: {timezone.now()}")
    print_line()
    
    # Kiểm tra kết nối
    db = check_firebase_connection()
    if not db:
        return
    
    # Hiển thị các bản ghi gần đây
    records = list_recent_records()
    
    # Menu tương tác
    while True:
        print_line()
        print("MENU TÙY CHỌN:")
        print("1. Kiểm tra tất cả dữ liệu người dùng trên Firebase")
        print("2. Kiểm tra dữ liệu của một người dùng cụ thể")
        print("3. Đẩy lại dữ liệu cho một bản ghi")
        print("0. Thoát")
        
        choice = input("\nNhập lựa chọn của bạn: ")
        
        if choice == "1":
            check_firebase_data(db)
        
        elif choice == "2":
            username = input("Nhập username cần kiểm tra: ")
            check_specific_user(db, username)
        
        elif choice == "3":
            if records:
                record_id = input("Nhập ID bản ghi cần đẩy lại (hoặc Enter để chọn bản ghi đầu tiên): ")
                if not record_id and records:
                    record_id = records[0].id
                
                if record_id:
                    push_data_again(record_id)
            else:
                print("Không có bản ghi nào để đẩy lại.")
        
        elif choice == "0":
            print("Kết thúc chương trình.")
            break
        
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main() 