"""attendance_system_facial_recognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from recognition import views as recog_views
from users import views as users_views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', recog_views.process_video_roi_view, name='home'),
    path('attendance-log/', recog_views.attendance_log_view, name='attendance-log'),
    path('add_camera/', recog_views.add_camera_view, name='add-camera'),
    path('video_feed/', recog_views.video_feed, name='video-feed'),
    path('login/', auth_views.LoginView.as_view(template_name='users/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='recognition/home.html'), name='logout'),
    path('register/', users_views.register, name='register'),
    path('not_authorised/', recog_views.not_authorised, name='not-authorised'),
    
    # --- URLs for ROI selection --- 
    path('save_roi/<int:camera_id>/', recog_views.save_roi_view, name='save-roi'),
    path('get_static_frame/<int:camera_id>/', recog_views.get_static_frame_view, name='get-static-frame'),
    # --- URL for getting collect progress --- 
    path('get_collect_progress/', recog_views.get_collect_progress_view, name='get-collect-progress'),
    # --- URL for AJAX training --- 
    path('ajax_train/', recog_views.ajax_train_view, name='ajax-train'),

    # --- URLs for Dataset Viewer --- 
    path('get_dataset_usernames/', recog_views.get_dataset_usernames_view, name='get-dataset-usernames'),
    path('get_random_images/<str:username>/', recog_views.get_random_dataset_images_view, name='get-random-images'),

    # --- URL Endpoint kiểm tra trạng thái xử lý ---
    path('get_processing_status/', recog_views.get_processing_status_view, name='get-processing-status'),

    # API endpoints
    path('api/attendance/', recog_views.AttendanceRecordList.as_view(), name='api-attendance'),
    path('api/attendance/user/<str:username>/', recog_views.UserAttendanceRecordList.as_view(), name='api-user-attendance'),
    path('api/attendance/today/', recog_views.today_attendance, name='api-today-attendance'),
    path('api/attendance/me/', recog_views.my_attendance, name='api-my-attendance'),
    # API đồng bộ dữ liệu lên Firebase với cấu trúc mới (users/[userID]) chứa thông tin chấm công trực tiếp
    path('api/sync-firebase/', recog_views.sync_to_firebase, name='api-sync-firebase'),
    # API kiểm tra kết nối Firebase
    path('api/test-firebase/', recog_views.test_firebase_api, name='api-test-firebase'),
    # API lấy danh sách supervisors
    path('api/supervisors/', recog_views.get_supervisors_api, name='api-get-supervisors'),

    # Trang hiển thị mối quan hệ Supervisor-Worker
    path('supervisor-worker/', recog_views.supervisor_worker_view, name='supervisor-worker'),

    # APIs cho Supervisor và Worker
    path('api/update-supervisor/', recog_views.update_supervisor, name='update-supervisor'),
    path('api/add-supervisor/', recog_views.add_supervisor, name='add-supervisor'),
    path('api/add-worker/', recog_views.add_worker, name='add-worker'),
    path('api/assign-worker/', recog_views.assign_worker, name='assign-worker'),
    path('api/get-worker-info/', recog_views.get_worker_info, name='get-worker-info'),
    path('api/update-worker/', recog_views.update_worker, name='update-worker'),
    # Thêm API xóa supervisor và worker
    path('api/delete-worker/', recog_views.delete_worker, name='delete-worker'),
    path('api/delete-supervisor/', recog_views.delete_supervisor, name='delete-supervisor'),
    # API đồng bộ danh sách worker theo supervisor vào Firebase
    path('api/sync-supervisor-worker-firebase/', recog_views.sync_supervisor_worker_firebase, name='sync-supervisor-worker-firebase'),
    # API lưu danh sách nhà thầu vào Firebase
    path('api/save-contractor/', recog_views.save_contractor_to_firebase, name='save-contractor-to-firebase'),

    # Quản lý cơ sở dữ liệu
    path('database_manager/', recog_views.database_manager_view, name='database-manager'),

    # URLs cho trang lên lịch nhận diện tự động
    path('scheduled-recognition/', recog_views.scheduled_recognition_view, name='scheduled-recognition'),
    path('edit-schedule/<int:schedule_id>/', recog_views.edit_schedule_view, name='edit-schedule'),
    path('toggle-schedule-status/', recog_views.toggle_schedule_status, name='toggle-schedule-status'),
    path('delete-schedule/', recog_views.delete_schedule, name='delete-schedule'),
    path('test-schedule/<int:schedule_id>/', recog_views.test_scheduled_recognition, name='test-schedule'),
    
    # Thêm URL mới cho chấm công liên tục
    path('continuous-schedule/', recog_views.continuous_schedule_view, name='continuous-schedule'),
    path('edit-continuous-schedule/<int:schedule_id>/', recog_views.edit_continuous_schedule_view, name='edit-continuous-schedule'),
    path('toggle-continuous-schedule-status/', recog_views.toggle_continuous_schedule_status, name='toggle-continuous-schedule-status'),
    path('delete-continuous-schedule/', recog_views.delete_continuous_schedule, name='delete-continuous-schedule'),
    path('test-continuous-schedule/<int:schedule_id>/', recog_views.test_continuous_schedule, name='test-continuous-schedule'),
    path('monitor-schedules/', recog_views.monitor_continuous_schedules, name='monitor-schedules'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
