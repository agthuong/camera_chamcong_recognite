from django.forms import ModelForm
from django.contrib.auth.models import User
from django import forms
#from django.contrib.admin.widgets import AdminDateWidget
from .models import CameraConfig # Import model mới

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


# Form mới cho chức năng ROI (ĐÃ CẬP NHẬT)
class VideoRoiForm(forms.Form):
    camera = forms.ModelChoiceField(
        queryset=CameraConfig.objects.all(), 
        label='Chọn Camera',
        empty_label="-- Chọn một Camera đã cấu hình --",
        widget=forms.Select(attrs={'class': 'form-control'}) # Thêm class để styling nếu cần
    )
    mode = forms.ChoiceField(
        label='Chế độ xử lý',
        choices=[('collect', 'Thu thập dữ liệu'), ('recognize', 'Nhận diện')],
        widget=forms.RadioSelect
    )
    username = forms.CharField(
        label='Username (nếu Thu thập dữ liệu)', 
        max_length=30, 
        required=False, 
        help_text='Nhập tên người dùng cần thu thập dữ liệu.',
        widget=forms.TextInput(attrs={'placeholder': 'Nhập username'})
    )

    def clean(self):
        cleaned_data = super().clean()
        mode = cleaned_data.get("mode")
        username = cleaned_data.get("username")

        if mode == 'collect' and not username:
            raise forms.ValidationError(
                "Vui lòng nhập Username khi chọn chế độ 'Thu thập dữ liệu'."
            )
        
        # Kiểm tra xem camera đã chọn có ROI chưa (tùy chọn, có thể kiểm tra trong view)
        # camera = cleaned_data.get('camera')
        # if camera and not camera.get_roi_tuple():
        #     raise forms.ValidationError(
        #         f"Camera '{camera.name}' chưa được cấu hình ROI. Vui lòng vào trang Admin để cấu hình."
        #     )

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

       
