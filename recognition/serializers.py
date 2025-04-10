from rest_framework import serializers
from django.contrib.auth.models import User
from .models import AttendanceRecord
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
    
    class Meta:
        model = AttendanceRecord
        fields = ['id', 'username', 'employee_id', 'project', 'company', 'date', 'check_in_time', 'check_out_time', 'check_in_image', 'check_out_image']
    
    def get_check_in_time(self, obj):
        return obj.check_in.strftime('%H:%M:%S') if obj.check_in else None
    
    def get_check_out_time(self, obj):
        return obj.check_out.strftime('%H:%M:%S') if obj.check_out else None
        
    def get_check_in_image(self, obj):
        if obj.check_in_image_url and hasattr(obj.check_in_image_url, 'url'):
            return self.context['request'].build_absolute_uri(obj.check_in_image_url.url)
        return None
        
    def get_check_out_image(self, obj):
        if obj.check_out_image_url and hasattr(obj.check_out_image_url, 'url'):
            return self.context['request'].build_absolute_uri(obj.check_out_image_url.url)
        return None 