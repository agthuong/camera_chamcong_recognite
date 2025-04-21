import os
import logging
from logging.handlers import RotatingFileHandler

# Đường dẫn thư mục log
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

# Đảm bảo thư mục logs tồn tại
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name, log_file, level=logging.INFO, max_bytes=10*1024*1024, backup_count=5):
    """
    Thiết lập logger với RotatingFileHandler
    
    Args:
        name: Tên của logger
        log_file: Tên tệp log (không bao gồm đường dẫn)
        level: Mức độ log (mặc định: INFO)
        max_bytes: Kích thước tối đa của mỗi tệp log (mặc định: 10MB)
        backup_count: Số lượng tệp backup tối đa (mặc định: 5)
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Nếu logger đã có handlers, có thể đã được thiết lập, trả về ngay
    if logger.handlers:
        return logger
    
    # Tạo đường dẫn đầy đủ đến tệp log
    log_path = os.path.join(LOG_DIR, log_file)
    
    # Thiết lập định dạng log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Cấu hình stream handler (ghi ra console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Cấu hình file handler với rotation
    try:
        file_handler = RotatingFileHandler(
            log_path, 
            maxBytes=max_bytes,       # 10MB mỗi tệp (mặc định)
            backupCount=backup_count, # Giữ tối đa 5 tệp backup (mặc định)
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Không thể tạo file handler: {str(e)}")
    
    # Thiết lập mức độ log
    logger.setLevel(level)
    
    return logger 