from django.contrib import admin
from .models import CameraConfig, AttendanceRecord, UserRole, ScheduledCameraRecognition, ScheduledRecognitionLog, ContinuousAttendanceSchedule, ContinuousAttendanceLog # Import model mới

# Tùy chỉnh hiển thị trong Admin (tùy chọn)
@admin.register(CameraConfig)
class CameraConfigAdmin(admin.ModelAdmin):
    list_display = ('name', 'source', 'get_roi_tuple', 'updated_at') # Các cột hiển thị trong danh sách
    list_filter = ('name',) # Thêm bộ lọc theo tên
    search_fields = ('name', 'source') # Cho phép tìm kiếm theo tên và nguồn
    # Các trường chỉ đọc khi chỉnh sửa (không cho sửa trực tiếp ROI ở đây)
    # readonly_fields = ('roi_x', 'roi_y', 'roi_w', 'roi_h', 'created_at', 'updated_at') 
    # -> Tạm thời cho phép sửa ROI trong admin để dễ test, sau này có thể khóa lại
    fieldsets = (
        ('Thông tin Camera', {
            'fields': ('name', 'source')
        }),
        ('Cấu hình ROI (Tọa độ)', {
            'fields': ('roi_x', 'roi_y', 'roi_w', 'roi_h'),
            'classes': ('collapse',) # Thu gọn mặc định
        }),
         ('Thông tin Thêm', {
            'fields': ('created_at', 'updated_at'),
             'classes': ('collapse',)
        }),
    )
    readonly_fields = ('created_at', 'updated_at') # Không cho sửa ngày tạo/cập nhật

    # Có thể thêm action để chọn ROI hàng loạt nếu cần

@admin.register(AttendanceRecord)
class AttendanceRecordAdmin(admin.ModelAdmin):
    list_display = ('user', 'project', 'company', 'date', 'get_check_in_time', 'get_check_out_time')
    list_filter = ('date', 'company', 'project')
    search_fields = ('user__username', 'user__first_name', 'user__last_name', 'project')
    
    fieldsets = (
        ('Thông tin Cơ bản', {
            'fields': ('user', 'project', 'company', 'date')
        }),
        ('Giờ Làm', {
            'fields': ('check_in', 'check_out')
        }),
        ('Ảnh Xác nhận', {
            'fields': ('check_in_image_url', 'check_out_image_url')
        }),
        ('Thông tin Khác', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    readonly_fields = ('created_at', 'updated_at')
    
    def get_check_in_time(self, obj):
        return obj.check_in.strftime("%H:%M:%S") if obj.check_in else "Chưa có"
    get_check_in_time.short_description = "Giờ check-in"
    
    def get_check_out_time(self, obj):
        return obj.check_out.strftime("%H:%M:%S") if obj.check_out else "Chưa có"
    get_check_out_time.short_description = "Giờ check-out"

@admin.register(UserRole)
class UserRoleAdmin(admin.ModelAdmin):
    list_display = ('user', 'role', 'get_supervisor_name', 'updated_at')
    list_filter = ('role',)
    search_fields = ('user__username', 'supervisor__username', 'custom_supervisor')
    
    fieldsets = (
        ('Thông tin người dùng', {
            'fields': ('user', 'role')
        }),
        ('Thông tin giám sát', {
            'fields': ('supervisor', 'custom_supervisor'),
            'description': 'Chỉ áp dụng cho worker. Có thể chọn supervisor từ hệ thống hoặc nhập tên nếu không có.'
        }),
        ('Thông tin thêm', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    readonly_fields = ('created_at', 'updated_at')
    
    def get_supervisor_name(self, obj):
        if obj.role == 'supervisor':
            return "-"
        return obj.supervisor.username if obj.supervisor else obj.custom_supervisor or "Chưa xác định"
    get_supervisor_name.short_description = "Giám sát trưởng"

# Đăng ký model lên lịch nhận diện
@admin.register(ScheduledCameraRecognition)
class ScheduledCameraRecognitionAdmin(admin.ModelAdmin):
    list_display = ('name', 'camera', 'start_time', 'end_time', 'interval_minutes', 'status', 'last_run', 'next_run')
    list_filter = ('status', 'camera')
    search_fields = ('name', 'camera__name')
    readonly_fields = ('last_run', 'next_run', 'created_at', 'updated_at')
    fieldsets = (
        ('Thông tin cơ bản', {
            'fields': ('name', 'camera', 'status')
        }),
        ('Thời gian', {
            'fields': ('start_time', 'end_time', 'interval_minutes', 'active_days')
        }),
        ('Thông tin hệ thống', {
            'fields': ('last_run', 'next_run', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(ScheduledRecognitionLog)
class ScheduledRecognitionLogAdmin(admin.ModelAdmin):
    list_display = ('schedule', 'timestamp', 'success', 'message_short')
    list_filter = ('success', 'schedule')
    search_fields = ('message', 'recognized_users', 'schedule__name')
    readonly_fields = ('schedule', 'timestamp', 'success', 'message', 'recognized_users', 'attendance_records')
    
    def message_short(self, obj):
        if len(obj.message) > 50:
            return f"{obj.message[:50]}..."
        return obj.message
    
    message_short.short_description = "Thông báo"

@admin.register(ContinuousAttendanceSchedule)
class ContinuousAttendanceScheduleAdmin(admin.ModelAdmin):
    list_display = ('name', 'camera', 'schedule_type', 'start_time', 'end_time', 'status', 'is_running')
    list_filter = ('status', 'schedule_type', 'camera', 'is_running')
    search_fields = ('name', 'camera__name')
    readonly_fields = ('is_running', 'worker_id', 'created_at', 'updated_at')
    fieldsets = (
        ('Thông tin cơ bản', {
            'fields': ('name', 'camera', 'schedule_type', 'status')
        }),
        ('Thời gian', {
            'fields': ('start_time', 'end_time', 'active_days')
        }),
        ('Thông tin hệ thống', {
            'fields': ('is_running', 'worker_id', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(ContinuousAttendanceLog)
class ContinuousAttendanceLogAdmin(admin.ModelAdmin):
    list_display = ('schedule', 'timestamp', 'event_type', 'message_short', 'recognized_user')
    list_filter = ('event_type', 'schedule', 'recognized_user')
    search_fields = ('message', 'schedule__name', 'recognized_user__username')
    readonly_fields = ('schedule', 'timestamp', 'event_type', 'message', 'recognized_user', 'attendance_record')
    
    def message_short(self, obj):
        if len(obj.message) > 50:
            return f"{obj.message[:50]}..."
        return obj.message
    
    message_short.short_description = "Thông báo"

# Register your models here.
# admin.site.register(CameraConfig) # Cách đăng ký đơn giản nếu không cần tùy chỉnh
