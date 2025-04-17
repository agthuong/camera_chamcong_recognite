from rest_framework import serializers
from django.contrib.auth.models import User
from .models import AttendanceRecord, UserRole
from django.conf import settings

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['username']

class AttendanceRecordSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    check_in_time = serializers.SerializerMethodField()
    check_out_time = serializers.SerializerMethodField()
    check_in_image = serializers.SerializerMethodField()
    check_out_image = serializers.SerializerMethodField()
    role = serializers.SerializerMethodField()
    supervisor_name = serializers.SerializerMethodField()
    supervisor_email_from_role = serializers.SerializerMethodField(method_name='get_supervisor_email_from_role')
    
    class Meta:
        model = AttendanceRecord
        fields = ['id', 'username', 'employee_id', 'project', 'company', 'date', 'check_in_time', 'check_out_time', 'check_in_image', 'check_out_image', 'role', 'supervisor_name', 'supervisor_email_from_role', 'recognized_by_camera']
    
    def get_check_in_time(self, obj):
        return obj.check_in.strftime('%H:%M:%S') if obj.check_in else None
    
    def get_check_out_time(self, obj):
        return obj.check_out.strftime('%H:%M:%S') if obj.check_out else None
        
    def get_check_in_image(self, obj):
        if obj.check_in_image_url and hasattr(obj.check_in_image_url, 'url'):
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.check_in_image_url.url)
            return obj.check_in_image_url.url
        return None
        
    def get_check_out_image(self, obj):
        if obj.check_out_image_url and hasattr(obj.check_out_image_url, 'url'):
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.check_out_image_url.url)
            return obj.check_out_image_url.url
        return None
        
    def get_role(self, obj):
        try:
            return obj.user.role_info.get_role_display()
        except UserRole.DoesNotExist:
            return None
        except AttributeError:
            return None

    def get_supervisor_name(self, obj):
        try:
            role_info = obj.user.role_info
            if role_info.role == 'worker':
                if role_info.supervisor:
                    return role_info.supervisor.username
                elif role_info.custom_supervisor:
                    return role_info.custom_supervisor
                else:
                    return "Chưa xác định"
            return None
        except UserRole.DoesNotExist:
            return None
        except AttributeError:
            return None
            
    def get_supervisor_email_from_role(self, obj):
        try:
            return obj.user.role_info.supervisor_email
        except UserRole.DoesNotExist:
            return None
        except AttributeError:
            return None 