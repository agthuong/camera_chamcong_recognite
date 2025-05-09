from rest_framework import generics, filters, status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from datetime import datetime
from django.utils.dateparse import parse_date
from .models import AttendanceRecord
from .serializers import AttendanceRecordSerializer
from django.http import JsonResponse
from django.core.exceptions import ObjectDoesNotExist
from django.views.decorators.csrf import csrf_exempt
import json
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.models import User
from django.utils import timezone
from recognition.utils.datetime_utils import get_current_time, get_current_date

class AttendanceRecordList(generics.ListAPIView):
    """
    API endpoint để lấy tất cả dữ liệu chấm công 
    """
    queryset = AttendanceRecord.objects.all().order_by('-date', '-check_in')
    serializer_class = AttendanceRecordSerializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['user__username', 'user__first_name', 'user__last_name']
    
    def get_queryset(self):
        queryset = AttendanceRecord.objects.all().order_by('-date', '-check_in')
        
        # Lọc theo ngày
        start_date = self.request.query_params.get('start_date', None)
        end_date = self.request.query_params.get('end_date', None)
        
        if start_date:
            try:
                start_date = parse_date(start_date)
                queryset = queryset.filter(date__gte=start_date)
            except Exception:
                pass
                
        if end_date:
            try:
                end_date = parse_date(end_date)
                queryset = queryset.filter(date__lte=end_date)
            except Exception:
                pass
        
        # Lọc theo username
        username = self.request.query_params.get('username', None)
        if username:
            queryset = queryset.filter(user__username=username)
            
        return queryset

class UserAttendanceRecordList(generics.ListAPIView):
    """
    API endpoint để lấy dữ liệu chấm công của một người dùng cụ thể
    """
    serializer_class = AttendanceRecordSerializer
    
    def get_queryset(self):
        username = self.kwargs['username']
        queryset = AttendanceRecord.objects.filter(user__username=username).order_by('-date')
        
        # Lọc theo ngày
        start_date = self.request.query_params.get('start_date', None)
        end_date = self.request.query_params.get('end_date', None)
        
        if start_date:
            try:
                start_date = parse_date(start_date)
                queryset = queryset.filter(date__gte=start_date)
            except Exception:
                pass
                
        if end_date:
            try:
                end_date = parse_date(end_date)
                queryset = queryset.filter(date__lte=end_date)
            except Exception:
                pass
            
        return queryset

@csrf_exempt
def today_attendance(request):
    """
    API endpoint để lấy dữ liệu chấm công của ngày hôm nay.
    """
    if request.method == 'GET':
        try:
            # Lấy ngày hiện tại
            today = get_current_date()
            
            # Lấy tất cả bản ghi chấm công của ngày hôm nay
            records = AttendanceRecord.objects.filter(date=today)
            
            # Serialize dữ liệu
            serializer = AttendanceRecordSerializer(records, many=True)
            
            # Trả về dữ liệu dưới dạng JSON
            return JsonResponse(serializer.data, safe=False)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Phương thức không được hỗ trợ'}, status=405)

@csrf_exempt
def my_attendance(request):
    """
    API endpoint để lấy dữ liệu chấm công của người dùng cụ thể.
    Yêu cầu tham số: username, start_date, end_date
    """
    ###
    if request.method == 'GET':
        try:
            # Lấy tham số từ URL
            username = request.GET.get('username')
            start_date_str = request.GET.get('start_date')
            end_date_str = request.GET.get('end_date')
            
            # Kiểm tra tham số bắt buộc
            if not username:
                return JsonResponse({'error': 'Thiếu tham số username'}, status=400)
                
            # Mặc định start_date và end_date là ngày hôm nay nếu không được cung cấp
            if not start_date_str:
                start_date = get_current_date()
            else:
                # Sử dụng parse_date từ django.utils.dateparse
                start_date = parse_date(start_date_str)
                if start_date is None:
                    return JsonResponse({'error': 'Định dạng start_date không hợp lệ. Sử dụng định dạng YYYY-MM-DD'}, status=400)
            
            if not end_date_str:
                end_date = get_current_date()
            else:
                # Sử dụng parse_date từ django.utils.dateparse
                end_date = parse_date(end_date_str)
                if end_date is None:
                    return JsonResponse({'error': 'Định dạng end_date không hợp lệ. Sử dụng định dạng YYYY-MM-DD'}, status=400)
            
            # Kiểm tra xem người dùng tồn tại
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                return JsonResponse({'error': f'Không tìm thấy người dùng với username: {username}'}, status=404)
            
            # Lấy bản ghi chấm công trong khoảng thời gian
            records = AttendanceRecord.objects.filter(
                user=user,
                date__gte=start_date,
                date__lte=end_date
            ).order_by('-date', '-time_in')  # Sắp xếp theo ngày và giờ vào giảm dần
            
            # Serialize dữ liệu
            serializer = AttendanceRecordSerializer(records, many=True)
            
            # Trả về dữ liệu dưới dạng JSON
            return JsonResponse(serializer.data, safe=False)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Phương thức không được hỗ trợ'}, status=405)
