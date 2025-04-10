from django.contrib import admin
from .models import CameraConfig, AttendanceRecord # Import model mới

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

# Register your models here.
# admin.site.register(CameraConfig) # Cách đăng ký đơn giản nếu không cần tùy chỉnh
