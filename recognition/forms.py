from django.forms import ModelForm
from django.contrib.auth.models import User
from django import forms
#from django.contrib.admin.widgets import AdminDateWidget
from .models import CameraConfig, ScheduledCameraRecognition, ContinuousAttendanceSchedule
# Import Profile từ app 'users'
from users.models import Profile 
# Giữ lại Q nếu cần cho các form khác, hoặc bỏ nếu chỉ dùng cho logic cũ
# from django.db.models import Q 

class usernameForm(forms.Form):
	username=forms.CharField(max_length=30)



class DateForm(forms.Form):
	date=forms.DateField(widget = forms.SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day")))


class UsernameAndDateForm(forms.Form):
	username=forms.CharField(max_length=30)
	date_from=forms.DateField(widget = forms.SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day")))
	date_to=forms.DateField(widget = forms.SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day")))


class DateForm_2(forms.Form):
	date_from=forms.DateField(widget = forms.SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day")))
	date_to=forms.DateField(widget = forms.SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day")))


# Form mới cho chức năng ROI (ĐÃ CẬP NHẬT theo Profile model)
class VideoRoiForm(forms.Form):
    camera = forms.ModelChoiceField(
        queryset=CameraConfig.objects.all(), 
        label='Chọn Camera',
        empty_label="--- Chọn Camera ---", # Thêm empty_label
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    mode = forms.ChoiceField(
        label='Chế độ xử lý',
        choices=[
            # Bỏ 'stream' nếu không xử lý trong view?
            # ('stream', 'Chỉ Stream Video'), 
            ('recognize', 'Nhận diện'),
            ('collect', 'Thu thập dữ liệu')
        ],
        initial='recognize', # Có thể đổi initial nếu muốn
        widget=forms.RadioSelect
    )
    username = forms.CharField(
        label='Username', # Đơn giản label
        max_length=150, # Tăng max_length khớp với User model
        required=False, # Sẽ validate trong clean()
        widget=forms.TextInput(attrs={'placeholder': 'Nhập username (bắt buộc khi Thu thập)'})
    )
    # --- Thêm trường Email (từ User model) --- 
    email = forms.EmailField(
        label='Email',
        required=False, # Chỉ bắt buộc cho Supervisor khi collect
        widget=forms.EmailInput(attrs={'placeholder': 'Nhập email (bắt buộc cho Supervisor)'})
    )
    # --- Cập nhật Role dựa trên Profile model ---
    role = forms.ChoiceField(
        label='Vai trò',
        choices=Profile.ROLE_CHOICES, # Lấy choices từ Profile model
        required=False, # Sẽ validate trong clean()
        initial='worker', # Giữ lại initial nếu muốn
        widget=forms.RadioSelect # Giữ RadioSelect hoặc đổi thành Select nếu muốn
    )
    # --- Cập nhật Supervisor dựa trên Profile model ---
    supervisor = forms.ModelChoiceField(
        queryset=User.objects.filter(profile__role='supervisor'),
        required=False, # Sẽ validate trong clean()
        label="Thuộc Supervisor",
        empty_label="--- Chọn Supervisor ---",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    # --- Chuyển Company thành ChoiceField ---
    company = forms.ChoiceField(
        label='Công ty',
        choices=[
            ('DBplus', 'DBplus'),
            ('DBhomes', 'DBhomes')
        ],
        required=True, # Đặt là True nếu luôn bắt buộc khi collect, validate trong clean()
        initial='DBplus', # Giữ lại initial nếu muốn
        widget=forms.Select(attrs={'class': 'form-control'}) # Dùng Select widget
    )
    
    # --- Thêm trường nhà thầu (Contractor) ---
    contractor = forms.CharField(
        label='Nhà thầu',
        max_length=100,
        required=False, # Sẽ validate trong clean() chỉ cho worker
        widget=forms.HiddenInput()
    )
    
    # --- Thêm trường lĩnh vực (Field) ---
    field = forms.CharField(
        label='Lĩnh vực',
        max_length=100,
        required=False, # Sẽ validate trong clean() chỉ cho worker
        widget=forms.HiddenInput()
    )
    
    # --- Bỏ trường employee_id --- 
    # employee_id = forms.CharField(max_length=50, required=False, label="Mã NV")
    # --- Bỏ trường project --- 
    # project = forms.CharField(max_length=100, required=False, label="Dự án")

    def clean(self):
        cleaned_data = super().clean()
        mode = cleaned_data.get("mode")
        username = cleaned_data.get("username")
        email = cleaned_data.get("email") # Lấy email
        role = cleaned_data.get("role")
        supervisor = cleaned_data.get("supervisor")
        # Lấy giá trị của các trường mới
        contractor = cleaned_data.get("contractor")
        field = cleaned_data.get("field")
        # --- Bỏ employee_id và project --- 
        # employee_id = cleaned_data.get("employee_id")
        company = cleaned_data.get("company")

        if mode == 'collect':
            # Username là bắt buộc khi collect
            if not username:
                self.add_error('username', "Vui lòng nhập Username khi chọn chế độ 'Thu thập dữ liệu'.")
            
            # Role là bắt buộc khi collect
            if not role:
                self.add_error('role', "Vui lòng chọn vai trò cho người dùng.")
            # Nếu là worker, supervisor, contractor và field là bắt buộc
            elif role == 'worker':
                if not supervisor:
                    self.add_error('supervisor', "Vui lòng chọn Supervisor cho Worker.")
                # Kiểm tra nhà thầu và lĩnh vực cho worker
                if not contractor:
                    self.add_error('contractor', "Vui lòng nhập Nhà thầu cho Worker.")
                if not field:
                    self.add_error('field', "Vui lòng nhập Lĩnh vực công việc cho Worker.")
            # Nếu là supervisor, email là bắt buộc
            elif role == 'supervisor':
                if not email:
                    self.add_error('email', "Vui lòng nhập Email cho Supervisor.")
                 
            # Có thể thêm validation cho các trường khác khi collect nếu cần
            if not company:
               self.add_error('company', "Vui lòng nhập Công ty.")
             # Bỏ validation cho employee_id
             # if not employee_id:
             #    self.add_error('employee_id', "Vui lòng nhập Mã NV.")
             # Bỏ validation cho project
             # if not project:
             #    self.add_error('project', "Vui lòng nhập Dự án.")
            
            # Kiểm tra nếu email đã tồn tại (nếu email phải là duy nhất)
            if email and role == 'supervisor':
                # Kiểm tra xem email đã được sử dụng bởi người dùng khác chưa
                existing_user = User.objects.filter(email=email).exclude(username=username).first()
                
                # Kiểm tra xem existing_user có tồn tại không
                if existing_user:
                    # Email đã được sử dụng bởi người dùng khác
                    self.add_error('email', "Email này đã được sử dụng bởi người dùng khác.")

        return cleaned_data

# Form để thêm Camera mới từ giao diện người dùng
class AddCameraForm(forms.ModelForm):
    class Meta:
        model = CameraConfig
        fields = ['name', 'source'] # Chỉ yêu cầu tên và nguồn khi thêm mới
        labels = {
            'name': 'Tên Camera',
            'source': 'Nguồn Video (ID Webcam, đường dẫn, URL RTSP)',
        }
        help_texts = {
            'name': 'Đặt tên gợi nhớ duy nhất cho camera.',
            'source': 'Nhập ID webcam (vd: 0), đường dẫn file, hoặc URL RTSP duy nhất.',
        }
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Ví dụ: Camera Cổng Trước'}),
            'source': forms.TextInput(attrs={'placeholder': 'Ví dụ: 0 hoặc rtsp://...'}),
        }

# Form để quản lý lịch trình nhận diện tự động camera
class ScheduledCameraRecognitionForm(forms.ModelForm):
    active_days = forms.MultipleChoiceField(
        choices=[
            ('1', 'Thứ 2'),
            ('2', 'Thứ 3'),
            ('3', 'Thứ 4'),
            ('4', 'Thứ 5'),
            ('5', 'Thứ 6'),
            ('6', 'Thứ 7'),
            ('7', 'Chủ nhật'),
        ],
        widget=forms.CheckboxSelectMultiple(),
        required=True,
        initial=['1', '2', '3', '4', '5'],
        label="Ngày hoạt động"
    )
    
    class Meta:
        model = ScheduledCameraRecognition
        fields = ['name', 'camera', 'start_time', 'end_time', 'interval_minutes', 'active_days', 'status']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Nhập tên lịch trình'}),
            'camera': forms.Select(attrs={'class': 'form-control select2'}),
            'start_time': forms.TimeInput(attrs={'class': 'form-control', 'type': 'time'}),
            'end_time': forms.TimeInput(attrs={'class': 'form-control', 'type': 'time'}),
            'interval_minutes': forms.Select(attrs={'class': 'form-control'}),
            'status': forms.Select(attrs={'class': 'form-control'}),
        }
    
    def clean(self):
        cleaned_data = super().clean()
        start_time = cleaned_data.get('start_time')
        end_time = cleaned_data.get('end_time')
        active_days = cleaned_data.get('active_days')
        
        if start_time and end_time and start_time >= end_time:
            self.add_error('end_time', "Thời gian kết thúc phải sau thời gian bắt đầu")
        
        if active_days:
            # Chuyển đổi list thành chuỗi ngăn cách bởi dấu phẩy
            cleaned_data['active_days'] = ','.join(active_days)
        
        return cleaned_data
    
    def __init__(self, *args, **kwargs):
        instance = kwargs.get('instance', None)
        super().__init__(*args, **kwargs)
        
        # Nếu đang chỉnh sửa form hiện có
        if instance:
            # Chuyển chuỗi ngày thành list để hiển thị đúng trong form
            active_days_str = instance.active_days
            active_days_list = active_days_str.split(',') if active_days_str else []
            self.initial['active_days'] = active_days_list

# Form để quản lý lịch trình chấm công liên tục
class ContinuousAttendanceScheduleForm(forms.ModelForm):
    active_days = forms.MultipleChoiceField(
        choices=[
            ('1', 'Thứ 2'),
            ('2', 'Thứ 3'),
            ('3', 'Thứ 4'),
            ('4', 'Thứ 5'),
            ('5', 'Thứ 6'),
            ('6', 'Thứ 7'),
            ('7', 'Chủ nhật'),
        ],
        widget=forms.CheckboxSelectMultiple(),
        required=True,
        initial=['1', '2', '3', '4', '5'],
        label="Ngày hoạt động"
    )
    
    class Meta:
        model = ContinuousAttendanceSchedule
        fields = ['name', 'camera', 'schedule_type', 'start_time', 'end_time', 'active_days', 'status']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Nhập tên lịch trình'}),
            'camera': forms.Select(attrs={'class': 'form-control select2'}),
            'schedule_type': forms.Select(attrs={'class': 'form-control'}),
            'start_time': forms.TextInput(attrs={'class': 'form-control timepicker-24hr', 'placeholder': 'HH:MM'}),
            'end_time': forms.TextInput(attrs={'class': 'form-control timepicker-24hr', 'placeholder': 'HH:MM'}),
            'status': forms.Select(attrs={'class': 'form-control'}),
        }
        help_texts = {
            'schedule_type': 'Chọn loại chấm công (Check-in hoặc Check-out)',
            'start_time': 'Thời điểm bắt đầu quét (định dạng 24 giờ, ví dụ: 08:30)',
            'end_time': 'Thời điểm kết thúc quét (định dạng 24 giờ, ví dụ: 17:00)',
        }
    
    def clean(self):
        cleaned_data = super().clean()
        start_time = cleaned_data.get('start_time')
        end_time = cleaned_data.get('end_time')
        active_days = cleaned_data.get('active_days')
        
        if start_time and end_time and start_time >= end_time:
            self.add_error('end_time', "Thời gian kết thúc phải sau thời gian bắt đầu")
        
        if active_days:
            # Chuyển đổi list thành chuỗi ngăn cách bởi dấu phẩy
            cleaned_data['active_days'] = ','.join(active_days)
        
        return cleaned_data
    
    def __init__(self, *args, **kwargs):
        instance = kwargs.get('instance', None)
        super().__init__(*args, **kwargs)
        
        # Nếu đang chỉnh sửa form hiện có
        if instance:
            # Chuyển chuỗi ngày thành list để hiển thị đúng trong form
            active_days_str = instance.active_days
            active_days_list = active_days_str.split(',') if active_days_str else []
            self.initial['active_days'] = active_days_list

       

