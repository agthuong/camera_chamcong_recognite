# Hệ thống chấm công bằng nhận diện khuôn mặt

Dự án này là một hệ thống chấm công sử dụng nhận diện khuôn mặt, được xây dựng trên nền tảng Django và Celery.

## Cài đặt với Docker

### Yêu cầu
- Docker và Docker Compose

### Triển khai
1. Clone repository:
```
git clone <repository_url>
cd chamcong
```

2. Tạo các thư mục cần thiết cho Nginx (nếu chưa có):
```
mkdir -p nginx/conf.d
```

3. Tạo và khởi chạy các container với Docker Compose:
```
docker-compose up -d
```

4. Truy cập ứng dụng:
- Ứng dụng web: http://localhost

### Kiến trúc Docker
Dự án được chia thành các dịch vụ sau:

1. **web**: Ứng dụng Django chính với Gunicorn
   - Chức năng: Xử lý các request HTTP, giao diện người dùng
   - Sử dụng Gunicorn làm WSGI server

2. **celery-worker**: Worker xử lý các tác vụ bất đồng bộ
   - Chức năng: Xử lý các tác vụ nặng như nhận diện khuôn mặt, phân tích video

3. **celery-beat**: Lập lịch cho các tác vụ định kỳ
   - Chức năng: Quản lý và lập lịch cho các tác vụ định kỳ

4. **nginx**: Web server và reverse proxy
   - Port: 80
   - Chức năng: Phục vụ file tĩnh, proxy request đến Gunicorn

5. **redis**: Broker cho Celery
   - Chức năng: Quản lý hàng đợi công việc giữa ứng dụng và worker

### Cấu hình
Các biến môi trường có thể được cấu hình trong file `docker-compose.yml`:

- `CELERY_BROKER_URL`: URL của Redis broker (mặc định: redis://redis:6379/0)
- `CELERY_RESULT_BACKEND`: URL của Redis result backend (mặc định: redis://redis:6379/0)
- `DJANGO_SETTINGS_MODULE`: Module cài đặt Django (mặc định: attendance_system_facial_recognition.settings)

### Sử dụng Nginx bên ngoài (đã có sẵn)
Nếu bạn đã có Nginx server chạy trên máy host, bạn có thể:

1. Bỏ dịch vụ nginx trong file docker-compose.yml
2. Mở port 8000 cho dịch vụ web:
   ```yaml
   web:
     ports:
       - "8000:8000"
   ```
3. Cấu hình Nginx server trên máy host để proxy request đến container web:
   ```nginx
   server {
       listen 80;
       server_name your_domain.com;

       location /static/ {
           alias /path/to/your/staticfiles/;
       }

       location /media/ {
           alias /path/to/your/media/;
       }

       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Khởi động lại dịch vụ
```
docker-compose restart <service_name>
```

### Xem log
```
docker-compose logs -f <service_name>
```

### Dừng dịch vụ
```
docker-compose down
```

## Các lệnh hữu ích

### Tạo admin user
```
docker-compose exec web python manage.py createsuperuser
```

### Thực hiện migrations
```
docker-compose exec web python manage.py makemigrations
docker-compose exec web python manage.py migrate
```

### Kiểm tra trạng thái Celery
```
docker-compose exec web celery -A attendance_system_facial_recognition inspect active
```

## Lưu ý khi triển khai
- Đảm bảo các thư mục `media`, `face_recognition_data` có quyền ghi cho container
- Nếu sử dụng GPU, cần cấu hình thêm Docker để hỗ trợ CUDA
- Hệ thống đã được cấu hình với Gunicorn để xử lý nhiều request đồng thời 