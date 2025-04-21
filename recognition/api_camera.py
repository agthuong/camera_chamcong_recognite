import requests
import json
import urllib3
from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from recognition.models import CameraConfig

# Tắt cảnh báo SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# URL của API bên ngoài để lấy danh sách camera
API_CAMERA_URL = "https://camera.dbplus.com.vn:8085/api/cameras/list"
API_KEY = "ca2faf96cd28f289f8580c82b8bb2052bac8109028c4dec0a51a7ddce6cd6f3d"

@login_required
@require_GET
def get_existing_cameras(request):
    """
    API để lấy danh sách camera đã thêm vào hệ thống
    """
    try:
        cameras = CameraConfig.objects.all().order_by('name')
        camera_list = [{'id': camera.id, 'name': camera.name} for camera in cameras]
        
        return JsonResponse({
            "status": "success",
            "cameras": camera_list
        })
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": f"Lỗi khi lấy danh sách camera: {str(e)}"
        }, status=500)

@login_required
@require_GET
def fetch_external_cameras(request):
    """
    API để lấy danh sách camera từ hệ thống bên ngoài
    """
    try:
        # Gọi API bên ngoài để lấy danh sách camera
        response = requests.get(f"{API_CAMERA_URL}?apiKey={API_KEY}", verify=False)
        
        # Kiểm tra nếu request thành công
        if response.status_code == 200:
            camera_data = response.json()
            
            # Chuẩn bị danh sách camera
            camera_list = []
            
            # Lặp qua các nhóm camera
            for group_name, cameras in camera_data.items():
                for camera in cameras:
                    camera_list.append({
                        "group": group_name,
                        "name": camera["name"],
                        "rtsp_link": camera["rtsp_link"]
                    })
            
            return JsonResponse({
                "status": "success",
                "data": camera_list
            })
        else:
            return JsonResponse({
                "status": "error",
                "message": f"Không thể kết nối đến API: {response.status_code}"
            }, status=500)
    
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": f"Lỗi khi lấy danh sách camera: {str(e)}"
        }, status=500)

@login_required
@require_POST
def import_all_cameras(request):
    """
    API để tự động thêm tất cả camera từ API bên ngoài vào hệ thống
    """
    try:
        # Gọi API bên ngoài để lấy danh sách camera
        response = requests.get(f"{API_CAMERA_URL}?apiKey={API_KEY}", verify=False)
        
        # Kiểm tra nếu request thành công
        if response.status_code == 200:
            camera_data = response.json()
            
            # Theo dõi kết quả
            success_count = 0
            skipped_count = 0
            error_count = 0
            error_messages = []
            
            # Lặp qua các nhóm camera và thêm vào hệ thống
            for group_name, cameras in camera_data.items():
                for camera in cameras:
                    name = camera["name"]
                    rtsp_link = camera["rtsp_link"]
                    
                    # Kiểm tra xem camera đã tồn tại hay chưa
                    if CameraConfig.objects.filter(name=name).exists():
                        skipped_count += 1
                        continue
                    
                    if CameraConfig.objects.filter(source=rtsp_link).exists():
                        skipped_count += 1
                        continue
                    
                    # Thêm camera mới
                    try:
                        CameraConfig.objects.create(
                            name=name,
                            source=rtsp_link
                        )
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        error_messages.append(f"Lỗi khi thêm camera {name}: {str(e)}")
            
            return JsonResponse({
                "status": "success",
                "message": f"Đã thêm {success_count} camera, bỏ qua {skipped_count} camera đã tồn tại, {error_count} lỗi",
                "details": {
                    "success_count": success_count,
                    "skipped_count": skipped_count,
                    "error_count": error_count,
                    "errors": error_messages
                }
            })
        else:
            return JsonResponse({
                "status": "error",
                "message": f"Không thể kết nối đến API: {response.status_code}"
            }, status=500)
    
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": f"Lỗi khi nhập camera: {str(e)}"
        }, status=500)

@login_required
@require_POST
def delete_camera(request):
    """
    API để xóa camera khỏi hệ thống
    """
    try:
        # Lấy camera_id từ request
        camera_id = request.POST.get('camera_id')
        
        if not camera_id:
            return JsonResponse({
                "status": "error",
                "message": "Thiếu camera_id"
            }, status=400)
        
        # Tìm camera
        try:
            camera = CameraConfig.objects.get(id=camera_id)
            camera_name = camera.name
            
            # Xóa camera
            camera.delete()
            
            return JsonResponse({
                "status": "success",
                "message": f"Đã xóa camera: {camera_name}"
            })
        except CameraConfig.DoesNotExist:
            return JsonResponse({
                "status": "error",
                "message": f"Không tìm thấy camera với ID: {camera_id}"
            }, status=404)
    
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": f"Lỗi khi xóa camera: {str(e)}"
        }, status=500) 