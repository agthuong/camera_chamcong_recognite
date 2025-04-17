import firebase_admin
from firebase_admin import credentials, firestore
import os
from django.conf import settings
import logging
from django.contrib.auth.models import User
import io
import sys
from django.utils import timezone as django_timezone

# Cấu hình logger với encoding utf-8 một cách an toàn hơn
logger = logging.getLogger(__name__)

# Thêm handler với encoding utf-8 nếu chưa có
if not logger.handlers:
    # Sử dụng StringIO thay vì sys.stdout trực tiếp
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)
    
    # Thêm file handler với encoding utf-8
    try:
        file_handler = logging.FileHandler('firebase.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Không thể tạo file handler: {str(e)}")

logger.setLevel(logging.INFO)

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
    Đẩy thông tin chấm công lên Firebase theo cấu trúc mới:
    users/[user_id]/attendance_logs/[date]
    Đồng thời cập nhật collection list_worker/{supervisor_email} với danh sách ID worker.
    
    Với worker, sẽ sử dụng email của supervisor thay vì email riêng.
    Worker không có email, chỉ có supervisor_email.
    
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
        
        # Lấy thông tin cơ bản từ record
        user = record.user
        date_str = record.date.strftime('%Y-%m-%d')
        company_name = record.company if hasattr(record, 'company') and record.company else "Unknown"
        
        # Mặc định vai trò là "unknown"
        user_role = "unknown"
        supervisor_username = None
        supervisor_email = None
        user_email = user.email if user.email else None
        worker_id_for_list = str(user.id) # Lấy ID worker để thêm vào list
        
        # Khởi tạo biến để lưu thông tin nhà thầu và lĩnh vực
        contractor_info = None
        field_info = None
        
        # Kiểm tra thông tin vai trò từ Profile model
        try:
            # Kiểm tra xem user có Profile không
            if hasattr(user, 'profile'):
                profile = user.profile
                
                # Lấy vai trò từ profile
                user_role = profile.role  # 'supervisor' hoặc 'worker'
                
                # Lấy thông tin công ty nếu có
                if profile.company:
                    company_name = profile.company
                
                # Lấy thông tin nhà thầu và lĩnh vực nếu có
                if profile.contractor:
                    contractor_info = profile.contractor
                if profile.field:
                    field_info = profile.field
                
                # Nếu là worker, lấy thông tin supervisor
                if user_role == 'worker' and profile.supervisor:
                    supervisor = profile.supervisor.user
                    supervisor_username = supervisor.username
                    supervisor_email = supervisor.email
            # Nếu không tìm thấy profile, thử tìm trong role_info (nếu có)
            elif hasattr(user, 'role_info'):
                role_info = user.role_info
                user_role = role_info.role
                
                if user_role == 'worker':
                    if role_info.supervisor:
                        supervisor = role_info.supervisor
                        supervisor_username = supervisor.username
                        supervisor_email = supervisor.email
                    # Nếu có supervisor_email được lưu trữ trực tiếp
                    elif hasattr(role_info, 'supervisor_email') and role_info.supervisor_email:
                        supervisor_email = role_info.supervisor_email
                    
                    # Nếu có custom_supervisor
                    if hasattr(role_info, 'custom_supervisor') and role_info.custom_supervisor:
                        supervisor_username = role_info.custom_supervisor
        except Exception as e:
            print(f"[FIREBASE] Lỗi khi lấy thông tin vai trò: {e}")
        
        # Chuẩn bị dữ liệu người dùng để đẩy lên Firebase - không còn user_id
        user_data = {
            'username': user.username,
            'role': user_role,  # Đảm bảo luôn có trường role
            'company': company_name,
        }
        
        # Thêm thông tin nhà thầu và lĩnh vực nếu có
        if user_role == 'worker':
            if contractor_info:
                user_data['contractor'] = contractor_info
            if field_info:
                user_data['field'] = field_info
        
        # Chỉ thêm email cho supervisor, không thêm cho worker
        if user_role == 'supervisor' and user_email:
            user_data['email'] = user_email
            
        # Thêm thông tin supervisor nếu có
        if supervisor_username:
            user_data['supervisor'] = supervisor_username
        if supervisor_email:
            user_data['supervisor_email'] = supervisor_email
        
        # Chuyển đổi thời gian check-in và check-out sang múi giờ Việt Nam
        local_check_in = django_timezone.localtime(record.check_in) if record.check_in else None
        local_check_out = django_timezone.localtime(record.check_out) if record.check_out else None
        
        # Chuẩn bị dữ liệu chấm công với thời gian ở múi giờ Việt Nam
        attendance_data = {
            'date': date_str,
            'check_in_time': local_check_in.strftime('%H:%M:%S') if local_check_in else None,
            'check_out_time': local_check_out.strftime('%H:%M:%S') if local_check_out else None,
            'timezone': 'Asia/Ho_Chi_Minh',  # Thêm thông tin múi giờ để rõ ràng hơn
        }
        
        # Thêm thông tin camera nếu có
        if camera_name:
            attendance_data['camera'] = camera_name
        elif hasattr(record, 'recognized_by_camera') and record.recognized_by_camera:
            attendance_data['camera'] = record.recognized_by_camera
        
        # Thêm URLs ảnh nếu có
        if hasattr(record, 'check_in_image_url') and record.check_in_image_url:
            attendance_data['check_in_image_url'] = f"{settings.MEDIA_URL}{record.check_in_image_url}"
        
        if hasattr(record, 'check_out_image_url') and record.check_out_image_url:
            attendance_data['check_out_image_url'] = f"{settings.MEDIA_URL}{record.check_out_image_url}"
        
        # Đường dẫn tới document người dùng trên Firestore
        user_doc_ref = db.collection('users').document(str(user.id))
        
        # Lưu hoặc cập nhật thông tin người dùng
        user_doc_ref.set(user_data, merge=True)
        
        # Lưu dữ liệu chấm công vào subcollection attendance_logs
        attendance_log_ref = user_doc_ref.collection('attendance_logs').document(date_str)
        attendance_log_ref.set(attendance_data, merge=True)
        
        # === Cập nhật danh sách worker cho supervisor ===
        if user_role == 'worker' and supervisor_email:
            try:
                # Dữ liệu worker cần đẩy lên danh sách
                worker_data = {
                    'id': worker_id_for_list,
                    'username': user.username
                }
                
                # Thêm thông tin nhà thầu và lĩnh vực
                if contractor_info:
                    worker_data['contractor'] = contractor_info
                if field_info:
                    worker_data['field'] = field_info
                
                # Cập nhật danh sách worker với cả thông tin chi tiết
                list_worker_doc_ref = db.collection('list_worker').document(supervisor_email)
                
                # Đầu tiên, thêm worker ID vào mảng workers (để duy trì tương thích với code cũ)
                list_worker_doc_ref.set({
                    'workers': firestore.ArrayUnion([worker_id_for_list])
                }, merge=True)
                
                # Sau đó, lưu thông tin chi tiết của worker vào subcollection worker_details
                worker_details_ref = list_worker_doc_ref.collection('worker_details').document(worker_id_for_list)
                worker_details_ref.set(worker_data, merge=True)
                
                print(f"[FIREBASE] Đã cập nhật worker ID {worker_id_for_list} với thông tin chi tiết vào danh sách của supervisor {supervisor_email}")
            except Exception as e:
                print(f"[FIREBASE ERROR] Lỗi khi cập nhật list_worker cho {supervisor_email}: {e}")
        # === Kết thúc cập nhật danh sách worker ===
        
        # Log dữ liệu sẽ đẩy lên
        print(f"[FIREBASE] Đẩy dữ liệu lên Firebase cho {user.username} với đường dẫn: users/{user.id}/attendance_logs/{date_str}")
        print(f"[FIREBASE] Dữ liệu user: {user_data}")
        print(f"[FIREBASE] Dữ liệu chấm công: {attendance_data}")
        
        logger.info(f"Đẩy thành công dữ liệu chấm công lên Firebase cho {user.username}")
        print(f"[FIREBASE SUCCESS] Đẩy thành công dữ liệu chấm công lên Firebase cho {user.username}")
        
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi đẩy dữ liệu lên Firebase: {e}")
        print(f"[FIREBASE ERROR] Lỗi khi đẩy dữ liệu lên Firebase: {e}")
        return False 