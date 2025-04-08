from rest_framework import serializers
from django.contrib.auth.models import User
from .models import AttendanceRecord

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'first_name', 'last_name']

class AttendanceRecordSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    check_in_time = serializers.SerializerMethodField()
    check_out_time = serializers.SerializerMethodField()
    
    class Meta:
        model = AttendanceRecord
        fields = ['id', 'user', 'date', 'check_in_time', 'check_out_time', 'check_in_face', 'check_out_face']
    
    def get_check_in_time(self, obj):
        return obj.check_in.strftime('%H:%M:%S') if obj.check_in else None
    
    def get_check_out_time(self, obj):
        return obj.check_out.strftime('%H:%M:%S') if obj.check_out else None 