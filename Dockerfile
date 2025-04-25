FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Cài đặt cmake và các dependencies cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libpq-dev \
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt
COPY requirements.txt /app/requirements.txt

# Cài đặt các dependencies Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn eventlet

# Sao chép mã nguồn
COPY . /app/

# Tạo tập tin cấu hình cho gunicorn
RUN echo 'import eventlet\neventlet.monkey_patch()' > /app/gunicorn_conf.py

# Tạo thư mục static, media và logs
RUN mkdir -p /app/staticfiles /app/media /app/logs && \
    chmod -R 755 /app/logs

# Sửa lỗi trong settings.py
RUN sed -i 's|BASE_DIR / '\''db.sqlite3'\''|os.path.join(BASE_DIR, '\''db.sqlite3'\'')|g' /app/attendance_system_facial_recognition/settings.py || true

# Mở cổng
EXPOSE 8000
# Tạo một khối VOLUME cho database và logs
VOLUME ["/app/db", "/app/logs"]
# Khởi động server với worker eventlet
CMD ["bash", "-c", "ulimit -n 65536 && python manage.py migrate && python manage.py collectstatic --noinput && gunicorn attendance_system_facial_recognition.wsgi:application --bind 0.0.0.0:8000 --worker-class eventlet --workers 4 --timeout 300 --preload --config /app/gunicorn_conf.py"]