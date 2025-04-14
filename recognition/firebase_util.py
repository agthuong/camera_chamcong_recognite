import firebase_admin
from firebase_admin import credentials, firestore
import os
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# Biến global để lưu trữ kết nối Firebase
firebase_app = None
db = None

def initialize_firebase():
    """
    Khởi tạo kết nối với Firebase nếu chưa được khởi tạo.
    Cần phải có file service account key JSON từ Firebase.
    """
    global firebase_app, db
    
    if firebase_app is not None:
        logger.info("Đã khởi tạo Firebase trước đó, sử dụng kết nối hiện có")
        return db  # Đã được khởi tạo, trả về instance db hiện tại
    
    try:
        # Đường dẫn đến file service account key
        service_account_path = os.path.join(settings.BASE_DIR, 'firebase-credentials.json')
        
        if not os.path.exists(service_account_path):
            logger.error(f"File service account key không tồn tại tại: {service_account_path}")
            print(f"[FIREBASE ERROR] File service account key không tồn tại tại: {service_account_path}")
            return None
        
        # Khởi tạo ứng dụng Firebase
        print(f"[FIREBASE] Đang khởi tạo Firebase với file xác thực: {service_account_path}")
        cred = credentials.Certificate(service_account_path)
        firebase_app = firebase_admin.initialize_app(cred)
        
        # Lấy instance Firestore
        db = firestore.client()
        logger.info("Đã kết nối thành công với Firebase")
        print("[FIREBASE] Đã kết nối thành công với Firebase")
        
        return db
    except Exception as e:
        logger.error(f"Lỗi kết nối Firebase: {e}")
        print(f"[FIREBASE ERROR] Lỗi kết nối Firebase: {e}")
        return None

def push_attendance_to_firebase(record, camera_name=None):
    """
    Đẩy thông tin chấm công lên Firebase theo cấu trúc:
    companies/[company]/projects/[project]/ID/[id]/
    
    Args:
        record: Đối tượng AttendanceRecord chứa dữ liệu chấm công
        camera_name: Tên camera đã nhận diện người dùng (tùy chọn)
    """
    # Khởi tạo Firebase nếu chưa được khởi tạo
    db = initialize_firebase()
    if db is None:
        logger.error("Không thể đẩy dữ liệu lên Firebase: Kết nối Firebase không khả dụng")
        print("[FIREBASE ERROR] Không thể đẩy dữ liệu lên Firebase: Kết nối Firebase không khả dụng")
        return False
    
    try:
        # Lưu thông tin camera vào record nếu được cung cấp và model có trường này
        if camera_name:
            try:
                has_camera_field = hasattr(record, 'recognized_by_camera')
                if has_camera_field and not record.recognized_by_camera:
                    record.recognized_by_camera = camera_name
                    record.save(update_fields=['recognized_by_camera'])
                    print(f"[FIREBASE] Đã cập nhật thông tin camera '{camera_name}' cho record của {record.user.username}")
            except AttributeError:
                print(f"[FIREBASE] Model không hỗ trợ trường recognized_by_camera, bỏ qua cập nhật")
            except Exception as e:
                print(f"[FIREBASE] Lỗi khi cập nhật camera: {e}")
        
        # Lấy dữ liệu từ record
        company_name = record.company if record.company else "Unknown"
        project_name = record.project if record.project else "Unknown"
        # Sử dụng employee_id nếu có, nếu không thì dùng username
        employee_id = record.employee_id if record.employee_id else record.user.username
        
        # Chuẩn bị dữ liệu để đẩy lên Firebase
        data = {
            'username': record.user.username,
            'employee_id': employee_id,
            'check_in_time': record.check_in.strftime('%H:%M:%S') if record.check_in else None,
            'check_out_time': record.check_out.strftime('%H:%M:%S') if record.check_out else None,
            'date': record.date.strftime('%Y-%m-%d'),
            'project': project_name,
            'company': company_name,
        }
        
        # Thêm thông tin camera nếu có
        if camera_name:
            data['camera'] = camera_name
        else:
            # Chỉ thêm từ model nếu trường tồn tại
            try:
                if hasattr(record, 'recognized_by_camera') and record.recognized_by_camera:
                    data['camera'] = record.recognized_by_camera
            except Exception:
                pass
        
        # Thêm URLs ảnh nếu có
        if record.check_in_image_url:
            # Tạo URL đầy đủ cho ảnh check-in
            data['check_in_image_url'] = f"{settings.MEDIA_URL}{record.check_in_image_url}"
        
        if record.check_out_image_url:
            # Tạo URL đầy đủ cho ảnh check-out
            data['check_out_image_url'] = f"{settings.MEDIA_URL}{record.check_out_image_url}"
        
        # Đường dẫn tới document trên Firestore
        doc_ref = db.collection('companies').document(company_name) \
                   .collection('projects').document(project_name) \
                   .collection('ID').document(employee_id)
        
        # Log dữ liệu sẽ đẩy lên
        print(f"[FIREBASE] Đẩy dữ liệu lên Firebase cho {record.user.username} với đường dẫn: companies/{company_name}/projects/{project_name}/ID/{employee_id}")
        print(f"[FIREBASE] Dữ liệu: {data}")
        
        # Đẩy dữ liệu lên Firestore
        doc_ref.set(data)
        logger.info(f"Đẩy thành công dữ liệu chấm công lên Firebase cho {record.user.username}")
        print(f"[FIREBASE SUCCESS] Đẩy thành công dữ liệu chấm công lên Firebase cho {record.user.username}")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi đẩy dữ liệu lên Firebase: {e}")
        print(f"[FIREBASE ERROR] Lỗi khi đẩy dữ liệu lên Firebase: {e}")
        return False 