import flask
from flask import Flask, jsonify, request
import sqlite3
import datetime
import os
import logging

# Thiết lập logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

app = Flask(__name__)

# Hàm kết nối đến database
def get_db_connection():
    try:
        conn = sqlite3.connect('attendance.db')
        conn.row_factory = sqlite3.Row  # Để kết quả trả về dạng dictionary
        return conn
    except Exception as e:
        logging.error(f"Lỗi kết nối database: {e}")
        return None

# API trả về tất cả dữ liệu điểm danh
@app.route('/api/attendance', methods=['GET'])
def get_all_attendance():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Không thể kết nối đến database"}), 500
            
        cursor = conn.cursor()
        
        # Lấy tham số từ query string (nếu có)
        date = request.args.get('date')
        
        if date:
            # Kiểm tra định dạng ngày
            try:
                datetime.datetime.strptime(date, '%Y-%m-%d')
                cursor.execute("SELECT ID, name, date, check_in, check_out FROM attendance WHERE date = ?", (date,))
            except ValueError:
                return jsonify({"error": "Định dạng ngày không hợp lệ. Vui lòng sử dụng định dạng YYYY-MM-DD"}), 400
        else:
            cursor.execute("SELECT ID, name, date, check_in, check_out FROM attendance")
            
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            result.append({
                "ID": row["ID"],
                "Name": row["name"],
                "Date": row["date"],
                "Check_In": row["check_in"],
                "Check_Out": row["check_out"]
            })
            
        conn.close()
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Lỗi khi truy vấn dữ liệu điểm danh: {e}")
        return jsonify({"error": f"Đã xảy ra lỗi: {str(e)}"}), 500

# API để lấy dữ liệu điểm danh theo ID
@app.route('/api/attendance/<int:user_id>', methods=['GET'])
def get_attendance_by_id(user_id):
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Không thể kết nối đến database"}), 500
            
        cursor = conn.cursor()
        
        # Lấy tham số từ query string (nếu có)
        date = request.args.get('date')
        
        if date:
            # Kiểm tra định dạng ngày
            try:
                datetime.datetime.strptime(date, '%Y-%m-%d')
                cursor.execute("SELECT ID, name, date, check_in, check_out FROM attendance WHERE ID = ? AND date = ?", 
                             (user_id, date))
            except ValueError:
                return jsonify({"error": "Định dạng ngày không hợp lệ. Vui lòng sử dụng định dạng YYYY-MM-DD"}), 400
        else:
            cursor.execute("SELECT ID, name, date, check_in, check_out FROM attendance WHERE ID = ?", (user_id,))
            
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            result.append({
                "ID": row["ID"],
                "Name": row["name"],
                "Date": row["date"],
                "Check_In": row["check_in"],
                "Check_Out": row["check_out"]
            })
            
        conn.close()
        
        if not result:
            return jsonify({"message": f"Không tìm thấy dữ liệu điểm danh cho ID {user_id}"}), 404
            
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Lỗi khi truy vấn dữ liệu điểm danh theo ID: {e}")
        return jsonify({"error": f"Đã xảy ra lỗi: {str(e)}"}), 500

# API để lấy dữ liệu điểm danh theo ngày
@app.route('/api/attendance/date/<date>', methods=['GET'])
def get_attendance_by_date(date):
    try:
        # Kiểm tra định dạng ngày
        try:
            datetime.datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            return jsonify({"error": "Định dạng ngày không hợp lệ. Vui lòng sử dụng định dạng YYYY-MM-DD"}), 400
            
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Không thể kết nối đến database"}), 500
            
        cursor = conn.cursor()
        cursor.execute("SELECT ID, name, date, check_in, check_out FROM attendance WHERE date = ?", (date,))
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            result.append({
                "ID": row["ID"],
                "Name": row["name"],
                "Date": row["date"],
                "Check_In": row["check_in"],
                "Check_Out": row["check_out"]
            })
            
        conn.close()
        
        if not result:
            return jsonify({"message": f"Không tìm thấy dữ liệu điểm danh cho ngày {date}"}), 404
            
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Lỗi khi truy vấn dữ liệu điểm danh theo ngày: {e}")
        return jsonify({"error": f"Đã xảy ra lỗi: {str(e)}"}), 500

# API để lấy tất cả người dùng đã điểm danh
@app.route('/api/users', methods=['GET'])
def get_all_users():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Không thể kết nối đến database"}), 500
            
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ID, name FROM attendance ORDER BY ID")
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            result.append({
                "ID": row["ID"],
                "Name": row["name"]
            })
            
        conn.close()
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Lỗi khi truy vấn danh sách người dùng: {e}")
        return jsonify({"error": f"Đã xảy ra lỗi: {str(e)}"}), 500

# Endpoint để lấy báo cáo tổng hợp theo khoảng thời gian
@app.route('/api/report', methods=['GET'])
def get_attendance_report():
    try:
        # Lấy tham số từ query string
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Kiểm tra tham số
        if not start_date or not end_date:
            return jsonify({"error": "Vui lòng cung cấp start_date và end_date"}), 400
            
        # Kiểm tra định dạng ngày
        try:
            datetime.datetime.strptime(start_date, '%Y-%m-%d')
            datetime.datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({"error": "Định dạng ngày không hợp lệ. Vui lòng sử dụng định dạng YYYY-MM-DD"}), 400
            
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Không thể kết nối đến database"}), 500
            
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ID, name, date, check_in, check_out FROM attendance WHERE date BETWEEN ? AND ? ORDER BY date, ID",
            (start_date, end_date)
        )
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            result.append({
                "ID": row["ID"],
                "Name": row["name"],
                "Date": row["date"],
                "Check_In": row["check_in"],
                "Check_Out": row["check_out"]
            })
            
        conn.close()
        
        if not result:
            return jsonify({"message": f"Không tìm thấy dữ liệu điểm danh từ {start_date} đến {end_date}"}), 404
            
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Lỗi khi tạo báo cáo điểm danh: {e}")
        return jsonify({"error": f"Đã xảy ra lỗi: {str(e)}"}), 500

# API để hiển thị hình ảnh check-in và check-out theo tên người dùng
@app.route('/api/images/<username>/<status>', methods=['GET'])
def get_image(username, status):
    try:
        # Xử lý tên thư mục đặc biệt
        sanitized_user = ''.join(c if c.isalnum() else '_' for c in username)
        image_path = os.path.join('static', 'attendance_images', sanitized_user, f'{status}.jpg')
        
        if os.path.exists(image_path):
            return flask.send_file(image_path, mimetype='image/jpeg')
        else:
            return jsonify({"error": f"Không tìm thấy hình ảnh {status} cho {username}"}), 404
    except Exception as e:
        logging.error(f"Lỗi khi lấy hình ảnh: {e}")
        return jsonify({"error": f"Đã xảy ra lỗi: {str(e)}"}), 500

if __name__ == '__main__':
    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'attendance_api.log')),
            logging.StreamHandler()
        ]
    )
    
    # Cung cấp thông tin khi khởi động
    logging.info("Khởi động API Attendance...")
    
    # Lấy port từ biến môi trường hoặc sử dụng 5000 mặc định
    port = int(os.environ.get('PORT', 5000))
    
    # Chạy Flask app
    app.run(host='0.0.0.0', port=port, debug=False) 