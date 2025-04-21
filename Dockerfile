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
    pip install gunicorn

# Sao chép mã nguồn
COPY . /app/

# Tạo thư mục static và media
RUN mkdir -p /app/staticfiles /app/media

# Sửa lỗi trong settings.py
RUN sed -i 's|BASE_DIR / '\''db.sqlite3'\''|os.path.join(BASE_DIR, '\''db.sqlite3'\'')|g' /app/attendance_system_facial_recognition/settings.py || true

# Mở cổng
EXPOSE 8000

# Khởi động server - chỉ chạy Django, không chạy Celery nữa
CMD ["bash", "-c", "python manage.py migrate && python manage.py collectstatic --noinput && gunicorn attendance_system_facial_recognition.wsgi:application --bind 0.0.0.0:8000"]