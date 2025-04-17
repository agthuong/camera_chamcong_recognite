from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required, permission_required
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse, Http404
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q, Count
from django.utils import timezone
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.conf import settings
from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition
import threading
import time
import json
import cv2
import os
import platform
import uuid
import base64
import numpy as np
import random
from datetime import date, datetime, timedelta
from django.utils.timezone import make_aware, now
from pathlib import Path
import re
import traceback
from django.shortcuts import render, redirect
from .forms import usernameForm, DateForm, UsernameAndDateForm, DateForm_2, VideoRoiForm, AddCameraForm
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import FaceAligner
import time
from attendance_system_facial_recognition.settings import BASE_DIR
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from django_pandas.io import read_frame
from users.models import Present, Time
import seaborn as sns
import pandas as pd
from django.db.models import Count
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math
from .models import CameraConfig, AttendanceRecord, UserRole, ScheduledCameraRecognition, ScheduledRecognitionLog, ContinuousAttendanceSchedule, ContinuousAttendanceLog
from django.utils import timezone
from django.db.models import Q
from rest_framework import generics, filters, status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAdminUser
from .serializers import AttendanceRecordSerializer
from django.utils.dateparse import parse_date
from django.http import StreamingHttpResponse, JsonResponse
import threading
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
import json
import random
import base64
from .video_roi_processor import sanitize_filename
from django.conf import settings
from .firebase_util import push_attendance_to_firebase
# REST API
from rest_framework import status, generics, filters
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser

# Import Profile từ ứng dụng users
from users.models import Profile

# Import các task Celery
from .tasks import test_continuous_recognition, start_continuous_recognition, stop_continuous_recognition

# Import các hàm utility từ recognition_utils
from .recognition_utils import predict, update_attendance_in_db_in, update_attendance_in_db_out

# Import từ firebase_util
from .firebase_util import push_attendance_to_firebase, initialize_firebase
from firebase_admin import firestore

# Import các form từ forms.py
from .forms import usernameForm, DateForm, UsernameAndDateForm, DateForm_2, VideoRoiForm, AddCameraForm

# Import các model từ models.py
from .models import CameraConfig, AttendanceRecord, UserRole, ScheduledCameraRecognition, ScheduledRecognitionLog, ContinuousAttendanceSchedule, ContinuousAttendanceLog

# Import các hàm xử lý video từ video_roi_processor.py
from .video_roi_processor import sanitize_filename, stream_output, process_video_with_roi, select_roi_from_source

# Import các hàm xử lý dữ liệu từ sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from django_pandas.io import read_frame
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math
from django.utils.dateparse import parse_date
from django.http import StreamingHttpResponse, JsonResponse
import threading
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
import json
import random
import base64
from django.conf import settings
from django.utils.translation import gettext_lazy as _
from .serializers import AttendanceRecordSerializer, UserSerializer
from django.utils.dateparse import parse_date
from django.http import StreamingHttpResponse, JsonResponse
import threading
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
import json
import random
import base64
from django.conf import settings
from .firebase_util import push_attendance_to_firebase
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.utils import timezone
import datetime
from django.db.models import Q
import io
import matplotlib
matplotlib.use('Agg')
from .forms import * # Import các form mới từ forms.py
from django.views.decorators.http import require_POST
from django.http import JsonResponse, FileResponse
from rest_framework import generics, permissions, filters
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.utils.translation import gettext_lazy as _
from .serializers import AttendanceRecordSerializer, UserSerializer
from .video_roi_processor import process_video_with_roi, stream_output, select_roi_from_source
from .firebase_util import push_attendance_to_firebase
from users.models import Profile

# Thêm import matplotlib
import matplotlib as mpl
mpl.use('Agg') # Đảm bảo mpl.use('Agg') được gọi trước plt

# --- Các biến global và phần còn lại của file ---
processing_thread = None
stop_processing_event = threading.Event()
processing_status = {
    'is_processing': False,
    'mode': None,
    'camera_source': None,
    'username': None, # Chỉ lưu username khi ở mode collect
}
# --- Thêm Lock và biến theo dõi camera hiện tại --- 
processing_lock = threading.Lock()
current_processing_camera_id = None 
# --- Kết thúc thêm --- 

# Định nghĩa RTSP stream URL, hãy thay đổi URL này theo cấu hình của bạn.
RTSP_STREAM_URL = 0

# utility functions:
def username_present(username):
    if User.objects.filter(username=username).exists():
        return True
    return False

def create_dataset(username):
    id = username
    if not os.path.exists('face_recognition_data/training_dataset/{}/'.format(id)):
        os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
    directory = 'face_recognition_data/training_dataset/{}/'.format(id)

    # Detect face
    # Loading the HOG face detector and the shape predictor for alignment
    print("[INFO] Loading the facial detector")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')  # CHANGE: đảm bảo đường dẫn tương đối
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    # Capture images từ RTSP stream thay vì webcam
    print("[INFO] Initializing RTSP video stream")
    vs = VideoStream(src=RTSP_STREAM_URL).start()
    time.sleep(2.0)

    sampleNum = 0
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)
        
        for face in faces:
            print("inside for loop")
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_aligned = fa.align(frame, gray_frame, face)
            sampleNum += 1

            if face is None:
                print("face is none")
                continue

            cv2.imwrite(directory + '/' + str(sampleNum) + '.jpg', face_aligned)
            face_aligned = imutils.resize(face_aligned, width=400)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.waitKey(50)

        cv2.imshow("Add Images", frame)
        cv2.waitKey(1)
        if sampleNum > 300:
            break

    vs.stop()
    cv2.destroyAllWindows()

def predict(face_aligned, svc, threshold=0.7):
    face_encodings = np.zeros((1, 128))
    try:
        x_face_locations = face_recognition.face_locations(face_aligned)
        faces_encodings = face_recognition.face_encodings(face_aligned, known_face_locations=x_face_locations)
        if len(faces_encodings) == 0:
            return ([-1], [0])
    except:
        return ([-1], [0])

    prob = svc.predict_proba(faces_encodings)
    result = np.where(prob[0] == np.amax(prob[0]))
    if prob[0][result[0]] <= threshold:
        return ([-1], prob[0][result[0]])
    return (result[0], prob[0][result[0]])

def vizualize_Data(embedded, targets):
    X_embedded = TSNE(n_components=2).fit_transform(embedded)
    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)
    plt.legend(bbox_to_anchor=(1, 1))
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()    
    plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
    plt.close()

def update_attendance_in_db_in(present):
    today = datetime.date.today()
    time_now = datetime.datetime.now()
    for person in present:
        user = User.objects.get(username=person)
        try:
            qs = Present.objects.get(user=user, date=today)
        except:
            qs = None
        if qs is None:
            if present[person]:
                a = Present(user=user, date=today, present=True)
                a.save()
            else:
                a = Present(user=user, date=today, present=False)
                a.save()
        else:
            if present[person]:
                qs.present = True
                qs.save(update_fields=['present'])
        if present[person]:
            a = Time(user=user, date=today, time=time_now, out=False)
            a.save()

def update_attendance_in_db_out(present):
    today = datetime.date.today()
    time_now = datetime.datetime.now()
    for person in present:
        user = User.objects.get(username=person)
        if present[person]:
            a = Time(user=user, date=today, time=time_now, out=True)
            a.save()

def check_validity_times(times_all):
    if len(times_all) > 0:
        sign = times_all.first().out
    else:
        sign = True
    times_in = times_all.filter(out=False)
    times_out = times_all.filter(out=True)
    if len(times_in) != len(times_out):
        sign = True
    break_hourss = 0
    if sign:
        check = False
        break_hourss = 0
        return (check, break_hourss)
    prev = True
    prev_time = times_all.first().time
    for obj in times_all:
        curr = obj.out
        if curr == prev:
            check = False
            break_hourss = 0
            return (check, break_hourss)
        if curr == False:
            curr_time = obj.time
            to = curr_time
            ti = prev_time
            break_time = ((to - ti).total_seconds()) / 3600
            break_hourss += break_time
        else:
            prev_time = obj.time
        prev = curr
    return (True, break_hourss)

def convert_hours_to_hours_mins(hours):
    h = int(hours)
    hours -= h
    m = hours * 60
    m = math.ceil(m)
    return str(str(h) + " hrs " + str(m) + "  mins")

def hours_vs_date_given_employee(present_qs, time_qs, admin=True):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    qs = present_qs

    for obj in qs:
        date = obj.date
        times_in = time_qs.filter(date=date).filter(out=False).order_by('time')
        times_out = time_qs.filter(date=date).filter(out=True).order_by('time')
        times_all = time_qs.filter(date=date).order_by('time')
        obj.time_in = None
        obj.time_out = None
        obj.hours = 0
        obj.break_hours = 0
        if len(times_in) > 0:            
            obj.time_in = times_in.first().time
        if len(times_out) > 0:
            obj.time_out = times_out.last().time
        if obj.time_in is not None and obj.time_out is not None:
            ti = obj.time_in
            to = obj.time_out
            hours = ((to - ti).total_seconds()) / 3600
            obj.hours = hours
        else:
            obj.hours = 0
        (check, break_hourss) = check_validity_times(times_all)
        if check:
            obj.break_hours = break_hourss
        else:
            obj.break_hours = 0
        df_hours.append(obj.hours)
        df_break_hours.append(obj.break_hours)
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)
    
    df = read_frame(qs)
    df["hours"] = df_hours
    df["break_hours"] = df_break_hours
    print(df)
    sns.barplot(data=df, x='date', y='hours')
    plt.xticks(rotation='vertical')
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    if admin:
        plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
        plt.close()
    else:
        plt.savefig('./recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
        plt.close()
    return qs

def hours_vs_employee_given_date(present_qs, time_qs):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    df_username = []
    qs = present_qs

    for obj in qs:
        user = obj.user
        times_in = time_qs.filter(user=user).filter(out=False)
        times_out = time_qs.filter(user=user).filter(out=True)
        times_all = time_qs.filter(user=user)
        obj.time_in = None
        obj.time_out = None
        obj.hours = 0
        if len(times_in) > 0:
            obj.time_in = times_in.first().time
        if len(times_out) > 0:
            obj.time_out = times_out.last().time
        if obj.time_in is not None and obj.time_out is not None:
            ti = obj.time_in
            to = obj.time_out
            hours = ((to - ti).total_seconds()) / 3600
            obj.hours = hours
        else:
            obj.hours = 0
        (check, break_hourss) = check_validity_times(times_all)
        if check:
            obj.break_hours = break_hourss
        else:
            obj.break_hours = 0
        df_hours.append(obj.hours)
        df_username.append(user.username)
        df_break_hours.append(obj.break_hours)
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)
    
    df = read_frame(qs)
    df['hours'] = df_hours
    df['username'] = df_username
    df["break_hours"] = df_break_hours

    sns.barplot(data=df, x='username', y='hours')
    plt.xticks(rotation='vertical')
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
    plt.close()
    return qs

def total_number_employees():
    qs = User.objects.all()
    return (len(qs) - 1)  # -1 để trừ admin

def employees_present_today():
    today = datetime.date.today()
    qs = Present.objects.filter(date=today).filter(present=True)
    return len(qs)

def this_week_emp_count_vs_date():
    today = datetime.date.today()
    some_day_last_week = today - datetime.timedelta(days=7)
    monday_of_last_week = some_day_last_week - datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    qs = Present.objects.filter(date__gte=monday_of_this_week).filter(date__lte=today)
    str_dates = []
    emp_count = []
    str_dates_all = []
    emp_cnt_all = []
    cnt = 0

    for obj in qs:
        date = obj.date
        str_dates.append(str(date))
        qs_date = Present.objects.filter(date=date).filter(present=True)
        emp_count.append(len(qs_date))

    while cnt < 5:
        date = str(monday_of_this_week + datetime.timedelta(days=cnt))
        cnt += 1
        str_dates_all.append(date)
        if str_dates.count(date) > 0:
            idx = str_dates.index(date)
            emp_cnt_all.append(emp_count[idx])
        else:
            emp_cnt_all.append(0)
    
    df = pd.DataFrame()
    df["date"] = str_dates_all
    df["Number of employees"] = emp_cnt_all
    
    sns.lineplot(data=df, x='date', y='Number of employees')
    plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_week/1.png')
    plt.close()

def last_week_emp_count_vs_date():
    today = datetime.date.today()
    some_day_last_week = today - datetime.timedelta(days=7)
    monday_of_last_week = some_day_last_week - datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    qs = Present.objects.filter(date__gte=monday_of_last_week).filter(date__lt=monday_of_this_week)
    str_dates = []
    emp_count = []
    str_dates_all = []
    emp_cnt_all = []
    cnt = 0

    for obj in qs:
        date = obj.date
        str_dates.append(str(date))
        qs_date = Present.objects.filter(date=date).filter(present=True)
        emp_count.append(len(qs_date))

    while cnt < 5:
        date = str(monday_of_last_week + datetime.timedelta(days=cnt))
        cnt += 1
        str_dates_all.append(date)
        if str_dates.count(date) > 0:
            idx = str_dates.index(date)
            emp_cnt_all.append(emp_count[idx])
        else:
            emp_cnt_all.append(0)
    
    df = pd.DataFrame()
    df["date"] = str_dates_all
    df["emp_count"] = emp_cnt_all

    sns.lineplot(data=df, x='date', y='emp_count')
    plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_week/1.png')
    plt.close()

# Create your views here.
def home(request):
    return render(request, 'recognition/home.html')

@login_required
def dashboard(request):
    if request.user.username == 'admin':
        print("admin")
        return render(request, 'recognition/admin_dashboard.html')
    else:
        print("not admin")
        return render(request, 'recognition/employee_dashboard.html')

@login_required
def add_photos(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    if request.method == 'POST':
        form = usernameForm(request.POST)
        data = request.POST.copy()
        username = data.get('username')
        if username_present(username):
            create_dataset(username)
            messages.success(request, f'Dataset Created')
            return redirect('add-photos')
        else:
            messages.warning(request, f'No such username found. Please register employee first.')
            return redirect('dashboard')
    else:
        form = usernameForm()
        return render(request, 'recognition/add_photos.html', {'form': form})

def mark_your_attendance(request):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path = "face_recognition_data/svc.sav"
    
    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        name = encoder.inverse_transform([i])[0]
        count[name] = 0
        present[name] = False

    # Sử dụng RTSP stream thay vì webcam
    #vs = VideoStream(src=RTSP_STREAM_URL).start()
    vs = VideoStream(src=0).start()
    sampleNum = 0

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)

        for face in faces:
            print("INFO : inside for loop")
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_aligned = fa.align(frame, gray_frame, face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            (pred, prob) = predict(face_aligned, svc)
            if pred != [-1]:
                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                pred = person_name
                if count[pred] == 0:
                    start[pred] = time.time()
                    count[pred] = count.get(pred, 0) + 1

                if count[pred] == 4 and (time.time() - start[pred]) > 1.2:
                    count[pred] = 0
                else:
                    present[pred] = True
                    log_time[pred] = datetime.datetime.now()
                    count[pred] = count.get(pred, 0) + 1
                    print(pred, present[pred], count[pred])
                cv2.putText(frame, str(person_name) + str(prob), (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                person_name = "unknown"
                cv2.putText(frame, str(person_name), (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Mark Attendance - In - Press q to exit", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()
    update_attendance_in_db_in(present)
    return redirect('home')

def mark_your_attendance_out(request):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path = "face_recognition_data/svc.sav"
    
    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        name = encoder.inverse_transform([i])[0]
        count[name] = 0
        present[name] = False

    # Sử dụng RTSP stream thay vì webcam
    vs = VideoStream(src=RTSP_STREAM_URL).start()
    sampleNum = 0

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)

        for face in faces:
            print("INFO : inside for loop")
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_aligned = fa.align(frame, gray_frame, face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            (pred, prob) = predict(face_aligned, svc)
            if pred != [-1]:
                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                pred = person_name
                if count[pred] == 0:
                    start[pred] = time.time()
                    count[pred] = count.get(pred, 0) + 1

                if count[pred] == 4 and (time.time() - start[pred]) > 1.5:
                    count[pred] = 0
                else:
                    present[pred] = True
                    log_time[pred] = datetime.datetime.now()
                    count[pred] = count.get(pred, 0) + 1
                    print(pred, present[pred], count[pred])
                cv2.putText(frame, str(person_name) + str(prob), (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                person_name = "unknown"
                cv2.putText(frame, str(person_name), (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Mark Attendance- Out - Press q to exit", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()
    update_attendance_in_db_out(present)
    return redirect('home')

@login_required
def train(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')

    training_dir = 'face_recognition_data/training_dataset'
    count = 0
    for person_name in os.listdir(training_dir):
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue
        for imagefile in image_files_in_folder(curr_directory):
            count += 1

    X = []
    y = []
    i = 0

    for person_name in os.listdir(training_dir):
        print(str(person_name))
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue
        for imagefile in image_files_in_folder(curr_directory):
            print(str(imagefile))
            image = cv2.imread(imagefile)
            try:
                X.append((face_recognition.face_encodings(image)[0]).tolist())
                y.append(person_name)
                i += 1
            except:
                print("removed")
                os.remove(imagefile)

    targets = np.array(y)
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    X1 = np.array(X)
    print("shape: " + str(X1.shape))
    np.save('face_recognition_data/classes.npy', encoder.classes_)
    svc = SVC(kernel='linear', probability=True)
    svc.fit(X1, y)
    svc_save_path = "face_recognition_data/svc.sav"
    with open(svc_save_path, 'wb') as f:
        pickle.dump(svc, f)

    vizualize_Data(X1, targets)
    messages.success(request, f'Training Complete.')
    return render(request, "recognition/train.html")

@login_required
def not_authorised(request):
    return render(request, 'recognition/not_authorised.html')

@login_required
def view_attendance_home(request):
    total_num_of_emp = total_number_employees()
    emp_present_today = employees_present_today()
    this_week_emp_count_vs_date()
    last_week_emp_count_vs_date()
    return render(request, "recognition/view_attendance_home.html", {'total_num_of_emp': total_num_of_emp, 'emp_present_today': emp_present_today})

@login_required
def view_attendance_date(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    qs = None
    time_qs = None
    present_qs = None

    if request.method == 'POST':
        form = DateForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data.get('date')
            print("date:" + str(date))
            time_qs = Time.objects.filter(date=date)
            present_qs = Present.objects.filter(date=date)
            if len(time_qs) > 0 or len(present_qs) > 0:
                qs = hours_vs_employee_given_date(present_qs, time_qs)
                return render(request, 'recognition/view_attendance_date.html', {'form': form, 'qs': qs})
            else:
                messages.warning(request, f'No records for selected date.')
                return redirect('view-attendance-date')
    else:
        form = DateForm()
        return render(request, 'recognition/view_attendance_date.html', {'form': form, 'qs': qs})

@login_required
def view_attendance_employee(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    time_qs = None
    present_qs = None
    qs = None

    if request.method == 'POST':
        form = UsernameAndDateForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            if username_present(username):
                u = User.objects.get(username=username)
                time_qs = Time.objects.filter(user=u)
                present_qs = Present.objects.filter(user=u)
                date_from = form.cleaned_data.get('date_from')
                date_to = form.cleaned_data.get('date_to')
                if date_to < date_from:
                    messages.warning(request, f'Invalid date selection.')
                    return redirect('view-attendance-employee')
                else:
                    time_qs = time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
                    present_qs = present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
                    if len(time_qs) > 0 or len(present_qs) > 0:
                        qs = hours_vs_date_given_employee(present_qs, time_qs, admin=True)
                        return render(request, 'recognition/view_attendance_employee.html', {'form': form, 'qs': qs})
                    else:
                        messages.warning(request, f'No records for selected duration.')
                        return redirect('view-attendance-employee')
            else:
                print("invalid username")
                messages.warning(request, f'No such username found.')
                return redirect('view-attendance-employee')
    else:
        form = UsernameAndDateForm()
        return render(request, 'recognition/view_attendance_employee.html', {'form': form, 'qs': qs})

@login_required
def view_my_attendance_employee_login(request):
    if request.user.username == 'admin':
        return redirect('not-authorised')
    qs = None
    time_qs = None
    present_qs = None
    if request.method == 'POST':
        form = DateForm_2(request.POST)
        if form.is_valid():
            u = request.user
            time_qs = Time.objects.filter(user=u)
            present_qs = Present.objects.filter(user=u)
            date_from = form.cleaned_data.get('date_from')
            date_to = form.cleaned_data.get('date_to')
            if date_to < date_from:
                messages.warning(request, f'Invalid date selection.')
                return redirect('view-my-attendance-employee-login')
            else:
                time_qs = time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
                present_qs = present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
                if len(time_qs) > 0 or len(present_qs) > 0:
                    qs = hours_vs_date_given_employee(present_qs, time_qs, admin=False)
                    return render(request, 'recognition/view_my_attendance_employee_login.html', {'form': form, 'qs': qs})
                else:
                    messages.warning(request, f'No records for selected duration.')
                    return redirect('view-my-attendance-employee-login')
    else:
        form = DateForm_2()
        return render(request, 'recognition/view_my_attendance_employee_login.html', {'form': form, 'qs': qs})

@login_required
def process_video_roi_view(request):
    global processing_thread, stop_processing_event, processing_status
    
    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'stop':
            print("[View] Nhận yêu cầu DỪNG xử lý.")
            if processing_thread and processing_thread.is_alive():
                stop_processing_event.set()
                processing_thread.join(timeout=2.0) # Đợi tối đa 2 giây
                if processing_thread.is_alive():
                    print("[View Warning] Luồng xử lý không dừng kịp thời.")
                else:
                    print("[View] Luồng xử lý đã dừng.")
            else:
                print("[View] Không có luồng xử lý nào đang chạy để dừng.")
            
            processing_status['is_processing'] = False
            processing_status['mode'] = None
            processing_status['camera_source'] = None
            processing_status['username'] = None
            processing_thread = None
            return JsonResponse({'status': 'success', 'message': 'Đã dừng xử lý.'})

        elif action == 'start':
            print("[View] Nhận yêu cầu BẮT ĐẦU xử lý.")
            form = VideoRoiForm(request.POST)
            if form.is_valid():
                if processing_thread and processing_thread.is_alive():
                    print("[View Warning] Đã có luồng xử lý chạy, bỏ qua yêu cầu start.")
                    return JsonResponse({'status': 'warning', 'message': 'Đã có quá trình xử lý đang chạy.'}, status=400)
                
                camera_config = form.cleaned_data['camera']
                mode = form.cleaned_data['mode']
                username = form.cleaned_data.get('username')
                role = form.cleaned_data.get('role')
                supervisor_user = form.cleaned_data.get('supervisor')
                company = form.cleaned_data.get('company')
                contractor = form.cleaned_data.get('contractor')
                field = form.cleaned_data.get('field')
                email = form.cleaned_data.get('email')

                # *** BẮT ĐẦU: Lấy và xác thực ROI ***
                roi = None
                try:
                    roi_x = camera_config.roi_x
                    roi_y = camera_config.roi_y
                    roi_w = camera_config.roi_w
                    roi_h = camera_config.roi_h
                    
                    print(f"[View] ROI gốc từ DB: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

                    if roi_w is not None and roi_h is not None and roi_w > 0 and roi_h > 0:
                        if roi_x is not None and roi_y is not None:
                            roi = (roi_x, roi_y, roi_w, roi_h)
                            print(f"[View] ROI hợp lệ được sử dụng: {roi}")
                        else:
                            print("[View Warning] ROI không hợp lệ (x hoặc y là None), sẽ xử lý toàn bộ frame.")
                            roi = None
                    else:
                        print("[View Warning] ROI không hợp lệ (w hoặc h <= 0 hoặc None), sẽ xử lý toàn bộ frame.")
                        roi = None
                    
                    if mode == 'collect' and roi is None:
                         print("[View Error] Chế độ 'Thu thập dữ liệu' yêu cầu ROI hợp lệ.")
                         return JsonResponse({'status': 'error', 'message': 'Chế độ Thu thập dữ liệu yêu cầu phải cấu hình ROI hợp lệ trước.'}, status=400)

                except AttributeError as attr_err:
                    print(f"[View Error] Lỗi thuộc tính khi lấy ROI cho camera {camera_config.id}: {attr_err}")
                    roi = None
                    if mode == 'collect':
                        return JsonResponse({'status': 'error', 'message': f'Lỗi cấu hình ROI cho camera đã chọn: {attr_err}'}, status=500)
                except Exception as e:
                    print(f"[View Error] Lỗi không xác định khi lấy ROI: {e}")
                    roi = None
                    if mode == 'collect':
                        return JsonResponse({'status': 'error', 'message': f'Lỗi khi lấy ROI: {e}'}, status=500)
                # *** KẾT THÚC: Lấy và xác thực ROI ***

                # Xử lý tạo/cập nhật User và Profile nếu là mode 'collect'
                if mode == 'collect':
                    from users.models import Profile # Import Profile tại đây
                    if not username:
                        return JsonResponse({'status': 'error', 'message': 'Username là bắt buộc cho chế độ Thu thập.'}, status=400)
                    
                    user, created = User.objects.get_or_create(
                        username=username,
                        defaults={'first_name': username}
                    )
                    if created: user.set_unusable_password(); user.save()
                    
                    if role == 'supervisor' and email:
                        if user.email != email:
                            if User.objects.filter(email=email).exclude(pk=user.pk).exists():
                                return JsonResponse({'status': 'error', 'message': f'Email "{email}" đã được sử dụng bởi người dùng khác.'}, status=400)
                            user.email = email
                            user.save(update_fields=['email'])
                        elif not user.email:
                            if User.objects.filter(email=email).exclude(pk=user.pk).exists():
                                return JsonResponse({'status': 'error', 'message': f'Email "{email}" đã được sử dụng bởi người dùng khác.'}, status=400)
                            user.email = email
                            user.save(update_fields=['email'])
                             
                    profile, profile_created = Profile.objects.update_or_create(
                        user=user,
                        defaults={
                            'role': role,
                            'company': company,
                            'contractor': contractor if role == 'worker' else None,
                            'field': field if role == 'worker' else None,
                            'supervisor': supervisor_user.profile if role == 'worker' and supervisor_user else None
                        }
                    )
                    if profile_created:
                        print(f"[View] Đã tạo Profile cho {username}")
                    else:
                        print(f"[View] Đã cập nhật Profile cho {username}")

                # Khởi tạo và bắt đầu luồng xử lý video
                stop_processing_event = threading.Event()
                video_source = camera_config.source
                camera_name = camera_config.name
                
                processing_thread = threading.Thread(
                    target=process_video_with_roi,
                    args=(video_source, mode, roi, stop_processing_event, stream_output),
                    kwargs={
                        'username': username,
                        'max_samples': settings.RECOGNITION_DEFAULT_MAX_SAMPLES,
                        'recognition_threshold': settings.RECOGNITION_CHECK_IN_THRESHOLD, 
                        'company': company,
                        'contractor': contractor if mode == 'collect' and role == 'worker' else None,
                        'field': field if mode == 'collect' and role == 'worker' else None,
                        'camera_name': camera_name
                    },
                    daemon=True
                )
                processing_thread.start()
                
                processing_status['is_processing'] = True
                processing_status['mode'] = mode
                processing_status['camera_source'] = video_source
                processing_status['username'] = username if mode == 'collect' else None

                print(f"[View] Đã bắt đầu luồng xử lý: Mode={mode}, Source={video_source}, ROI={roi}, User={username}")
                return JsonResponse({'status': 'success', 'message': 'Bắt đầu xử lý.'})
            else:
                print("[View Error] Form không hợp lệ:", form.errors.as_json())
                first_error_key = next(iter(form.errors), None)
                error_message = f"Dữ liệu không hợp lệ: {form.errors[first_error_key][0]}" if first_error_key else "Dữ liệu không hợp lệ."
                return JsonResponse({'status': 'error', 'message': error_message, 'errors': form.errors}, status=400)
        # Chỗ này cần else hoặc bỏ hẳn để không lỗi cú pháp
        # else: 
        #     return JsonResponse({'status': 'error', 'message': 'Hành động không hợp lệ.'}, status=400)

    # GET Request
    else: 
        form = VideoRoiForm()
        is_processing = processing_status.get('is_processing', False)
        context = {
            'form': form,
            'is_processing': is_processing,
        }
        return render(request, 'recognition/process_video_roi.html', context)

@login_required
def select_roi_view(request, camera_id):
    if request.user.username != 'admin':
        return redirect('not-authorised')

    try:
        camera_config = CameraConfig.objects.get(pk=camera_id)
    except CameraConfig.DoesNotExist:
        messages.error(request, "Không tìm thấy cấu hình camera được yêu cầu.")
        return redirect('home')

    video_source = camera_config.source
    print(f"[View] Bắt đầu chọn ROI cho Camera: {camera_config.name} (ID: {camera_id}), Nguồn: {video_source}")
    
    selected_roi_tuple = video_roi_processor.select_roi_from_source(video_source)

    if selected_roi_tuple:
        try:
             rx, ry, rw, rh = selected_roi_tuple
             camera_config.roi_x = rx
             camera_config.roi_y = ry
             camera_config.roi_w = rw
             camera_config.roi_h = rh
             camera_config.save()
             messages.success(request, f"Đã cập nhật ROI thành công cho camera '{camera_config.name}': {selected_roi_tuple}")
        except Exception as e:
             messages.error(request, f"Lỗi khi lưu ROI vào database: {e}")
    else:
        messages.warning(request, f"Đã hủy chọn ROI hoặc không thể mở video cho camera '{camera_config.name}'. ROI hiện tại không thay đổi.")

    return redirect('home')

@login_required
def add_camera_view(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    
    if request.method == 'POST':
        form = AddCameraForm(request.POST)
        if form.is_valid():
            try:
                form.save()
                messages.success(request, f"Đã thêm camera '{form.cleaned_data['name']}' thành công.")
                return redirect('home')
            except Exception as e:
                error_message = f"Lỗi khi thêm camera: {e}. Tên hoặc Nguồn có thể đã tồn tại."
                print(f"[Add Camera View] {error_message}")
                messages.error(request, error_message)
                return render(request, 'recognition/add_camera.html', {'form': form})
        else:
            messages.error(request, "Dữ liệu nhập không hợp lệ. Vui lòng kiểm tra lại.")
    else:
        form = AddCameraForm()
    
    return render(request, 'recognition/add_camera.html', {'form': form})

def generate_frames():
    print("[Stream] Bắt đầu generate_frames")
    from .video_roi_processor import stream_output
    while True:
        frame_bytes = stream_output.get_frame_bytes()
        if frame_bytes:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n'
            )
        else:
            if not stream_output.running and processing_thread is None:
                 print("[Stream] Stream stopped. Ending generate_frames.")
                 break
            time.sleep(0.05)

@login_required
def video_feed(request):
    print("[Stream] Yêu cầu video_feed")
    with threading.Lock():
         is_proc_alive = processing_thread is not None and processing_thread.is_alive()

    if not is_proc_alive:
        print("[Stream] Không có tiến trình nào đang chạy. Không bắt đầu stream.")
        from django.templatetags.static import static
        placeholder_path = static('recognition/img/placeholder.png')
        pass

    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

class AttendanceRecordList(generics.ListAPIView):
    queryset = AttendanceRecord.objects.all().order_by('-date', '-check_in')
    serializer_class = AttendanceRecordSerializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['user__username', 'user__first_name', 'user__last_name']
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        return context
    
    def get_queryset(self):
        queryset = AttendanceRecord.objects.all().order_by('-date', '-check_in')
        
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
        
        username = self.request.query_params.get('username', None)
        if username:
            queryset = queryset.filter(user__username=username)
            
        return queryset

class UserAttendanceRecordList(generics.ListAPIView):
    serializer_class = AttendanceRecordSerializer
    
    def get_queryset(self):
        username = self.kwargs['username']
        queryset = AttendanceRecord.objects.filter(user__username=username).order_by('-date')
        
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

@api_view(['GET'])
def today_attendance(request):
    today = datetime.now().date()
    records = AttendanceRecord.objects.filter(date=today)
    serializer = AttendanceRecordSerializer(records, many=True, context={'request': request})
    return Response(serializer.data)

@api_view(['GET'])
def my_attendance(request):
    username = request.query_params.get('username', None)
    start_date = request.query_params.get('start_date', None)
    end_date = request.query_params.get('end_date', None)
    
    if not username:
        return Response({"error": "Vui lòng cung cấp tham số username"}, status=400)
    
    queryset = AttendanceRecord.objects.filter(user__username=username).order_by('-date')
    
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
            
    serializer = AttendanceRecordSerializer(queryset, many=True, context={'request': request})
    return Response(serializer.data)

@require_POST
@login_required
def save_roi_view(request, camera_id):
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập.'}, status=403)

    try:
        camera_config = CameraConfig.objects.get(pk=camera_id)
    except CameraConfig.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Không tìm thấy camera.'}, status=404)

    try:
        # *** THAY ĐỔI: Đọc dữ liệu từ JSON body ***
        # data = request.POST -> Thay bằng đọc từ request.body
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
             print("[Save ROI View] Lỗi: Dữ liệu gửi lên không phải JSON hợp lệ.")
             return JsonResponse({'status': 'error', 'message': 'Dữ liệu yêu cầu không hợp lệ (không phải JSON).'}, status=400)

        # Lấy dữ liệu từ JSON object
        crop_x = data.get('roi_x')
        crop_y = data.get('roi_y')
        crop_width = data.get('roi_width')
        crop_height = data.get('roi_height')
        natural_width = data.get('natural_width')
        natural_height = data.get('natural_height')
        # *** KẾT THÚC THAY ĐỔI ĐỌC JSON ***

        # Kiểm tra dữ liệu nhận được
        if None in [crop_x, crop_y, crop_width, crop_height, natural_width, natural_height]:
             missing = [k for k, v in data.items() if v is None and k in ['roi_x', 'roi_y', 'roi_width', 'roi_height', 'natural_width', 'natural_height']]
             print(f"[Save ROI View] Lỗi: Thiếu dữ liệu bắt buộc: {missing}")
             return JsonResponse({'status': 'error', 'message': f'Thiếu dữ liệu bắt buộc: {", ".join(missing)}'}, status=400)

        print(f"[Save ROI View] Nhận dữ liệu JSON: CamID={camera_id}, Crop=({crop_x},{crop_y},{crop_width},{crop_height}), NaturalSize={natural_width}x{natural_height}")

        # *** THAY ĐỔI: Tính toán lại tọa độ ROI ***
        try:
            # Chuyển đổi sang số float trước khi tính toán
            crop_x = float(crop_x)
            crop_y = float(crop_y)
            crop_width = float(crop_width)
            crop_height = float(crop_height)
            natural_width = float(natural_width)
            natural_height = float(natural_height)

            # Kiểm tra natural_width để tránh chia cho 0
            if natural_width <= 0:
                raise ValueError("Kích thước ảnh gốc (natural_width) không hợp lệ.")

            # Tính tỉ lệ scale dựa trên chiều rộng chuẩn trong settings
            scale_factor = settings.RECOGNITION_FRAME_WIDTH / natural_width
            # Tính toán chiều cao chuẩn sau khi scale theo chiều rộng
            standard_height = natural_height * scale_factor

            # Tính toán tọa độ ROI đã scale và làm tròn thành số nguyên
            roi_x = int(round(crop_x * scale_factor))
            roi_y = int(round(crop_y * scale_factor))
            roi_w = int(round(crop_width * scale_factor))
            roi_h = int(round(crop_height * scale_factor))

            # Đảm bảo tọa độ và kích thước nằm trong giới hạn của frame đã scale
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            # Chiều rộng không vượt quá chiều rộng chuẩn trừ đi vị trí x
            roi_w = max(1, min(roi_w, settings.RECOGNITION_FRAME_WIDTH - roi_x)) # Đảm bảo w tối thiểu là 1
            # Chiều cao không vượt quá chiều cao chuẩn trừ đi vị trí y
            roi_h = max(1, min(roi_h, int(round(standard_height)) - roi_y)) # Đảm bảo h tối thiểu là 1

        except (ValueError, TypeError) as calc_err:
             print(f"[Save ROI View] Lỗi tính toán tọa độ: {calc_err}")
             return JsonResponse({'status': 'error', 'message': f'Lỗi dữ liệu tọa độ hoặc kích thước: {calc_err}'}, status=400)
        # *** KẾT THÚC THAY ĐỔI TÍNH TOÁN ***

        # Lưu vào database
        camera_config.roi_x = roi_x
        camera_config.roi_y = roi_y
        camera_config.roi_w = roi_w
        camera_config.roi_h = roi_h
        camera_config.save()

        print(f"[Save ROI View] Đã lưu ROI đã tính toán: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

        return JsonResponse({
            'status': 'success',
            'message': 'Đã lưu ROI thành công.',
            'saved_roi': [roi_x, roi_y, roi_w, roi_h]
        })

    except Exception as e:
        print(f"[Save ROI View] Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': 'Lỗi server nội bộ khi lưu ROI.'}, status=500)

@login_required
def get_static_frame_view(request, camera_id):
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Forbidden'}, status=403)

    try:
        camera_config = CameraConfig.objects.get(pk=camera_id)
        video_source = camera_config.source
        print(f"[Get Frame] Lấy frame tĩnh từ camera ID: {camera_id}, nguồn: {video_source}")

        cap = None
        try:
            try:
                 source_int = int(video_source)
                 cap = cv2.VideoCapture(source_int)
            except ValueError:
                 cap = cv2.VideoCapture(video_source)

            if not cap or not cap.isOpened():
                print(f"[Get Frame] Lỗi: Không thể mở nguồn video: {video_source}")
                return JsonResponse({'status': 'error', 'message': 'Không thể mở nguồn video'}, status=500)

            ret, frame = False, None
            for _ in range(5):
                 ret, frame = cap.read()
                 if ret and frame is not None and frame.size > 0:
                      break
                 time.sleep(0.1)

            if not ret or frame is None:
                print(f"[Get Frame] Lỗi: Không thể đọc frame từ nguồn: {video_source}")
                return JsonResponse({'status': 'error', 'message': 'Không thể đọc frame từ camera'}, status=500)

            frame_resized = imutils.resize(frame, width=settings.RECOGNITION_FRAME_WIDTH)
            print(f"[Get Frame] Đã resize frame về width={settings.RECOGNITION_FRAME_WIDTH}")

            ret, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                 print("[Get Frame] Lỗi: Không thể encode frame thành JPEG")
                 return JsonResponse({'status': 'error', 'message': 'Lỗi xử lý ảnh'}, status=500)

            # *** THAY ĐỔI: Encode sang Base64 và trả về JSON ***
            # Chuyển buffer thành chuỗi bytes base64
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            # Tạo Data URL
            frame_base64_data_url = f"data:image/jpeg;base64,{jpg_as_text}"
            
            print(f"[Get Frame] Đã encode frame thành Base64 Data URL (độ dài: {len(frame_base64_data_url)}).")

            # Trả về JsonResponse chứa Data URL
            return JsonResponse({
                'status': 'success',
                'message': 'Lấy frame thành công.',
                'frame_base64': frame_base64_data_url # Trả về base64 thay vì URL
            })
            # *** KẾT THÚC THAY ĐỔI ***

        finally:
             if cap and cap.isOpened():
                 cap.release()
                 print(f"[Get Frame] Đã giải phóng camera.")

    except CameraConfig.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Không tìm thấy camera'}, status=404)
    except Exception as e:
        print(f"[Get Frame] Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': 'Lỗi server nội bộ'}, status=500)

@login_required
def get_collect_progress_view(request):
    from .video_roi_processor import collect_progress_tracker
    if not request.user.is_superuser:
         return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập.'}, status=403)
    
    # Đóng gói dữ liệu theo định dạng chuẩn
    return JsonResponse({
        'status': 'success',
        'progress': collect_progress_tracker
    })

from django.core.exceptions import PermissionDenied

@login_required
@require_POST
def ajax_train_view(request):
    if not request.user.is_superuser:
         return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập.'}, status=403)

    print("[AJAX Train] Bắt đầu quá trình huấn luyện...")
    try:
        training_dir = settings.RECOGNITION_TRAINING_DIR
        person_folders = [f for f in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, f))]

        if not person_folders:
            return JsonResponse({'status': 'error', 'message': 'Thư mục training_dataset trống hoặc không có thư mục con hợp lệ.'})

        total_image_files = 0
        valid_folders_and_names = {}

        for folder_name in person_folders:
            curr_directory = os.path.join(training_dir, folder_name)
            info_path = os.path.join(curr_directory, '_info.txt')
            original_name = folder_name
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8') as f_info:
                        read_name = f_info.read().strip()
                        if read_name:
                           original_name = read_name
                        else:
                           print(f"[AJAX Train] Cảnh báo: File _info.txt trong {folder_name} rỗng, sử dụng tên thư mục.")
                except Exception as e_read:
                    print(f"[AJAX Train] Cảnh báo: Không thể đọc _info.txt trong {folder_name}: {e_read}")
            else:
                 print(f"[AJAX Train] Cảnh báo: Không tìm thấy _info.txt trong {folder_name}, sử dụng tên thư mục.")

            image_files = image_files_in_folder(curr_directory)
            if image_files:
                total_image_files += len(image_files)
                valid_folders_and_names[folder_name] = original_name
            else:
                 print(f"[AJAX Train] Cảnh báo: Thư mục '{folder_name}' không chứa ảnh, bỏ qua.")

        if not valid_folders_and_names:
             return JsonResponse({'status': 'error', 'message': 'Không tìm thấy thư mục con hợp lệ nào chứa ảnh trong training_dataset.'})
        if total_image_files == 0:
            return JsonResponse({'status': 'error', 'message': 'Không tìm thấy ảnh nào trong các thư mục con của training_dataset.'})

        print(f"[AJAX Train] Tìm thấy tổng cộng {total_image_files} ảnh trong {len(valid_folders_and_names)} thư mục hợp lệ.")
        X = []
        y = []
        i = 0
        removed_count = 0

        for folder_name, original_name in valid_folders_and_names.items():
            print(f"[AJAX Train] Đang xử lý thư mục: {folder_name} (Tên gốc: {original_name})")
            curr_directory = os.path.join(training_dir, folder_name)
            image_files = image_files_in_folder(curr_directory)

            for imagefile in image_files:
                if not os.path.exists(imagefile):
                    print(f"[AJAX Train] Lỗi: File không tồn tại '{imagefile}', bỏ qua.")
                    removed_count += 1
                    continue
                image = cv2.imread(imagefile)
                if image is None:
                    print(f"[AJAX Train] Lỗi: Không thể đọc ảnh '{imagefile}', bỏ qua.")
                    removed_count += 1
                    continue

                try:
                    face_encodings = face_recognition.face_encodings(image, model='hog')
                    if face_encodings:
                        X.append(face_encodings[0].tolist())
                        y.append(original_name)
                        i += 1
                    else:
                         print(f"[AJAX Train] Cảnh báo: Không tìm thấy khuôn mặt trong '{imagefile}', bỏ qua.")
                         removed_count += 1

                except Exception as e:
                    print(f"[AJAX Train] Lỗi khi xử lý encoding ảnh '{imagefile}': {e}, bỏ qua.")
                    removed_count += 1

        print(f"[AJAX Train] Đã xử lý {i} ảnh thành công, {removed_count} ảnh bị lỗi/bỏ qua.")
        if not X or not y:
             return JsonResponse({'status': 'error', 'message': 'Không có dữ liệu hợp lệ để huấn luyện sau khi xử lý ảnh.'})

        targets = np.array(y)
        encoder = LabelEncoder()
        encoder.fit(y)
        y_encoded = encoder.transform(y)
        X_train = np.array(X)

        print(f"[AJAX Train] Shape dữ liệu huấn luyện X: {X_train.shape}, y: {y_encoded.shape}")
        if X_train.shape[0] == 0:
            return JsonResponse({'status': 'error', 'message': 'Không có vector đặc trưng nào được tạo ra để huấn luyện.'})

        print("[AJAX Train] Bắt đầu huấn luyện SVC model...")
        svc = SVC(kernel='linear', probability=True, C=1.0)
        svc.fit(X_train, y_encoded)

        print("[AJAX Train] Lưu model SVC và classes...")
        svc_save_path = settings.RECOGNITION_SVC_PATH
        classes_save_path = settings.RECOGNITION_CLASSES_PATH
        try:
            with open(svc_save_path, 'wb') as f:
                 pickle.dump(svc, f)
            np.save(classes_save_path, encoder.classes_)
        except IOError as e:
             print(f"[AJAX Train] Lỗi I/O khi lưu model/classes: {e}")
             return JsonResponse({'status': 'error', 'message': f'Lỗi khi lưu file model: {e}'})

        try:
             print("[AJAX Train] Tạo ảnh visualize...")
             if X_train.shape[1] > 2:
                 X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_train)
                 plt.figure(figsize=(10, 8))
                 for i_target, t in enumerate(encoder.classes_):
                      idx = targets == t
                      plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)
                 plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                 plt.title('t-SNE Visualization of Face Encodings')
                 plt.xlabel('t-SNE Component 1')
                 plt.ylabel('t-SNE Component 2')
                 plt.tight_layout(rect=[0, 0, 0.85, 1])
                 viz_path = settings.RECOGNITION_VISUALIZATION_PATH
                 os.makedirs(os.path.dirname(viz_path), exist_ok=True)
                 plt.savefig(viz_path)
                 plt.close()
                 print(f"[AJAX Train] Đã lưu ảnh visualize vào: {viz_path}")
             else:
                 print("[AJAX Train] Bỏ qua visualize vì số chiều không đủ cho t-SNE.")
        except Exception as e_viz:
             print(f"[AJAX Train] Lỗi khi tạo ảnh visualize: {e_viz}")
             import traceback
             traceback.print_exc()

        print("[AJAX Train] Huấn luyện hoàn tất.")
        return JsonResponse({'status': 'success', 'message': f'Huấn luyện thành công cho {len(encoder.classes_)} người!'})

    except Exception as e:
        print(f"[AJAX Train] Lỗi không xác định trong quá trình huấn luyện: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': f'Lỗi không xác định: {e}'}, status=500)

TRAINING_DATASET_PATH = settings.RECOGNITION_TRAINING_DIR

@login_required
def get_dataset_usernames_view(request):
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập.'}, status=403)

    original_usernames = []
    try:
        if os.path.exists(TRAINING_DATASET_PATH) and os.path.isdir(TRAINING_DATASET_PATH):
            for folder_name in os.listdir(TRAINING_DATASET_PATH):
                curr_directory = os.path.join(TRAINING_DATASET_PATH, folder_name)
                if os.path.isdir(curr_directory):
                    info_path = os.path.join(curr_directory, '_info.txt')
                    if os.path.exists(info_path):
                        try:
                            with open(info_path, 'r', encoding='utf-8') as f_info:
                                original_name = f_info.read().strip()
                                if original_name:
                                    original_usernames.append(original_name)
                        except Exception as e_read:
                             print(f"[Dataset Viewer] Lỗi đọc _info.txt trong {folder_name}: {e_read}")
                    else:
                         print(f"[Dataset Viewer] Cảnh báo: Không tìm thấy _info.txt trong {folder_name}")

            original_usernames.sort()
    except Exception as e:
        print(f"[Dataset Viewer] Lỗi khi liệt kê usernames: {e}")
        return JsonResponse({'status': 'error', 'message': 'Lỗi khi lấy danh sách username.'}, status=500)

    return JsonResponse({'status': 'success', 'usernames': original_usernames})

@login_required
def get_random_dataset_images_view(request, username):
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập.'}, status=403)

    sanitized_username = sanitize_filename(username)
    if not sanitized_username:
        return JsonResponse({'status': 'error', 'message': f'Username "{username}" không hợp lệ sau khi chuẩn hóa.'}, status=400)

    user_folder_path = os.path.join(TRAINING_DATASET_PATH, sanitized_username)
    images_base64 = []
    max_images_to_show = 5

    try:
        if not os.path.exists(user_folder_path) or not os.path.isdir(user_folder_path):
            return JsonResponse({'status': 'error', 'message': f'Thư mục cho username "{username}" không tồn tại.'}, status=404)

        image_files = [f for f in os.listdir(user_folder_path) 
                       if os.path.isfile(os.path.join(user_folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
             return JsonResponse({'status': 'success', 'images': []})

        selected_files = random.sample(image_files, k=min(len(image_files), max_images_to_show))

        for file_name in selected_files:
            file_path = os.path.join(user_folder_path, file_name)
            try:
                with open(file_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    images_base64.append(encoded_string)
            except Exception as e:
                print(f"[Dataset Viewer] Error reading/encoding image {file_path}: {e}")

    except Exception as e:
        print(f"[Dataset Viewer] Error getting images for {username}: {e}")
        return JsonResponse({'status': 'error', 'message': 'Lỗi server khi lấy ảnh.'}, status=500)

    return JsonResponse({'status': 'success', 'images': images_base64})

@api_view(['GET'])
@permission_classes([IsAuthenticated]) # Chỉ cần đăng nhập để lấy danh sách
def get_supervisors_api(request):
    """
    API trả về danh sách các user là supervisor (username và id).
    """
    try:
        supervisors = User.objects.filter(
            Q(role_info__role='supervisor') | Q(is_superuser=True)
        ).distinct().values('id', 'username').order_by('username')
        return JsonResponse({'status': 'success', 'supervisors': list(supervisors)})
    except Exception as e:
        print(f"[API Supervisors Error] Lỗi lấy danh sách supervisor: {e}")
        return JsonResponse({'status': 'error', 'message': 'Lỗi server khi lấy danh sách supervisor.'}, status=500)

@api_view(['POST'])
@permission_classes([IsAdminUser])  # Chỉ admin mới có thể gọi API này
def sync_to_firebase(request):
    """
    Đẩy dữ liệu chấm công lên Firebase. Có thể lọc theo date hoặc đẩy tất cả.
    
    Params:
        date (string, optional): Ngày cần đồng bộ theo định dạng YYYY-MM-DD.
        camera_name (string, optional): Tên camera để gán cho các bản ghi.
        camera_id (int, optional): ID của camera để gán cho các bản ghi (sẽ được chuyển đổi thành camera_name).
    """
    date_str = request.data.get('date')
    camera_name = request.data.get('camera_name')  # Thêm thông tin camera từ request
    camera_id = request.data.get('camera_id')  # Thêm thông tin camera_id từ request
    
    # Nếu có camera_id nhưng không có camera_name, tìm tên camera từ ID
    if camera_id and not camera_name:
        try:
            camera_id = int(camera_id)
            from .models import CameraConfig
            camera = CameraConfig.objects.filter(id=camera_id).first()
            if camera:
                camera_name = camera.name
                print(f"[SYNC] Đã tìm thấy tên camera '{camera_name}' từ ID {camera_id}")
            else:
                print(f"[SYNC] Không tìm thấy camera với ID {camera_id}")
        except (ValueError, Exception) as e:
            print(f"[SYNC] Lỗi khi tìm camera từ ID: {e}")
    
    if date_str:
        try:
            sync_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            records = AttendanceRecord.objects.filter(date=sync_date)
            target_desc = f"cho ngày {date_str}"
        except ValueError:
            return Response({"error": "Định dạng ngày không hợp lệ. Sử dụng định dạng YYYY-MM-DD."}, status=400)
    else:
        records = AttendanceRecord.objects.all()
        target_desc = "tất cả"
    
    if camera_name:
        target_desc += f" với camera '{camera_name}'"
    
    if not records.exists():
        return Response({"message": f"Không có dữ liệu chấm công {target_desc} để đồng bộ."}, status=200)
    
    success_count = 0
    error_count = 0
    
    for record in records:
        try:
            if push_attendance_to_firebase(record, camera_name):
                success_count += 1
            else:
                error_count += 1
        except Exception:
            error_count += 1
    
    return Response({
        "message": f"Đồng bộ dữ liệu {target_desc} lên Firebase hoàn tất.",
        "total": records.count(),
        "success": success_count,
        "error": error_count
    }, status=200)

@api_view(['POST'])
@permission_classes([IsAdminUser])  # Chỉ admin mới có thể gọi API này
def test_firebase_api(request):
    """
    API để kiểm tra kết nối tới Firebase.
    
    Returns:
        Kết quả kiểm tra kết nối Firebase: success, message, và thông tin chi tiết.
    """
    from .firebase_util import test_firebase_connection
    
    result = test_firebase_connection()
    return Response(result, status=200 if result['success'] else 500)

# ---------------------- Thêm view để quản lý cơ sở dữ liệu ----------------------
import sqlite3
import pandas as pd
from django.shortcuts import render
from django.db import connection

@login_required
@permission_required('is_staff', raise_exception=True)
def database_manager_view(request):
    """
    Trang để xem và quản lý cơ sở dữ liệu SQLite3.
    """
    # Lấy danh sách các bảng trong cơ sở dữ liệu
    tables = []
    with connection.cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall() if not table[0].startswith('sqlite_')]
    
    # Khởi tạo các biến
    selected_table = request.GET.get('table', '')
    search_query = request.GET.get('search', '')
    columns = []
    data = []
    total_records = 0
    filtered_records = 0
    
    # Lấy dữ liệu của bảng đã chọn
    if selected_table in tables:
        with connection.cursor() as cursor:
            # Lấy thông tin cột
            cursor.execute(f"PRAGMA table_info({selected_table});")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Đếm tổng số bản ghi
            cursor.execute(f"SELECT COUNT(*) FROM {selected_table};")
            total_records = cursor.fetchone()[0]
            
            # Tìm kiếm nếu có
            if search_query:
                # Xây dựng điều kiện tìm kiếm cho tất cả các cột
                search_conditions = " OR ".join([f"{col} LIKE '%{search_query}%'" for col in columns])
                
                # Đếm số bản ghi sau khi lọc
                cursor.execute(f"SELECT COUNT(*) FROM {selected_table} WHERE {search_conditions};")
                filtered_records = cursor.fetchone()[0]
                
                # Lấy dữ liệu đã lọc
                cursor.execute(f"SELECT * FROM {selected_table} WHERE {search_conditions} LIMIT 100;")
            else:
                filtered_records = total_records
                cursor.execute(f"SELECT * FROM {selected_table} LIMIT 100;")
            
            # Chuyển đổi dữ liệu thành danh sách các từ điển
            data = []
            for row in cursor.fetchall():
                data.append(dict(zip(columns, row)))
    
    context = {
        'tables': tables,
        'selected_table': selected_table,
        'columns': columns,
        'data': data,
        'total_records': total_records,
        'filtered_records': filtered_records,
        'search_query': search_query,
        'database_path': connection.settings_dict['NAME'],
    }
    
    return render(request, 'recognition/database_manager.html', context)

@login_required
def get_processing_status_view(request):
    """
    API Endpoint để kiểm tra trạng thái của tiến trình xử lý video.
    """
    global processing_thread, processing_lock
    if not request.user.is_superuser:
         return JsonResponse({'is_processing': False, 'error': 'Permission denied'}, status=403)

    with processing_lock:
        is_processing = processing_thread is not None and processing_thread.is_alive()
        
    return JsonResponse({'is_processing': is_processing})

@login_required
def supervisor_worker_view(request):
    """Hiển thị mối quan hệ giữa Supervisor và Worker theo dạng trực quan"""
    from users.models import Profile
    from django.contrib.auth.models import User
    
    # Lấy tất cả supervisor
    supervisors = Profile.objects.filter(role='supervisor')
    
    # Chuẩn bị dữ liệu để hiển thị
    supervisor_data = []
    for supervisor in supervisors:
        # Lấy danh sách workers thuộc supervisor này
        workers = Profile.objects.filter(role='worker', supervisor=supervisor)
        
        # Thêm thông tin supervisor và workers vào danh sách
        supervisor_data.append({
            'supervisor': supervisor,
            'workers': workers,
            'worker_count': workers.count(),
        })
    
    # Thêm thông tin workers không có supervisor (nếu có)
    workers_without_supervisor = Profile.objects.filter(role='worker', supervisor__isnull=True)
    has_unassigned = workers_without_supervisor.exists()
    
    # Tổng số supervisor và worker
    total_supervisors = supervisors.count()
    total_workers = Profile.objects.filter(role='worker').count()
    
    context = {
        'supervisor_data': supervisor_data,
        'workers_without_supervisor': workers_without_supervisor,
        'has_unassigned': has_unassigned,
        'total_supervisors': total_supervisors,
        'total_workers': total_workers,
    }
    
    return render(request, 'recognition/supervisor_worker.html', context)

@login_required
@require_POST
def update_supervisor(request):
    """API để cập nhật thông tin Supervisor"""
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập'}, status=403)
    
    supervisor_id = request.POST.get('supervisor_id')
    email = request.POST.get('email')
    company = request.POST.get('company')
    
    try:
        supervisor = Profile.objects.get(id=supervisor_id, role='supervisor')
        
        # Cập nhật email cho user
        if email:
            # Kiểm tra xem email đã tồn tại chưa
            existing_user = User.objects.filter(email=email).exclude(id=supervisor.user.id).first()
            if existing_user:
                return JsonResponse({'status': 'error', 'message': f'Email {email} đã được sử dụng bởi người dùng khác'})
            
            # Cập nhật email cho user
            supervisor.user.email = email
            supervisor.user.save()
        
        # Cập nhật company cho profile
        if company:
            supervisor.company = company
            supervisor.save()
        
        return JsonResponse({'status': 'success', 'message': 'Đã cập nhật thông tin supervisor thành công'})
    except Profile.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Không tìm thấy supervisor'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Lỗi khi cập nhật supervisor: {str(e)}'}, status=500)

@login_required
@require_POST
def add_supervisor(request):
    """API để thêm Supervisor mới"""
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập'}, status=403)
    
    username = request.POST.get('username')
    email = request.POST.get('email')
    company = request.POST.get('company')
    
    if not username or not email:
        return JsonResponse({'status': 'error', 'message': 'Vui lòng cung cấp đầy đủ thông tin'})
    
    try:
        # Kiểm tra tên người dùng đã tồn tại chưa
        if User.objects.filter(username=username).exists():
            return JsonResponse({'status': 'error', 'message': 'Username đã tồn tại'})
        
        # Kiểm tra email đã tồn tại chưa
        if User.objects.filter(email=email).exists():
            return JsonResponse({'status': 'error', 'message': 'Email đã tồn tại'})
        
        # Tạo người dùng mới
        user = User.objects.create_user(
            username=username,
            email=email,
            password=f"default_{username}"  # Mật khẩu mặc định
        )
        
        # Tạo profile
        Profile.objects.create(
            user=user,
            role='supervisor',
            company=company
        )
        
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@login_required
@require_POST
def add_worker(request):
    """API để thêm Worker mới"""
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập'}, status=403)
    
    username = request.POST.get('username')
    company = request.POST.get('company')
    supervisor_id = request.POST.get('supervisor_id')
    # Lấy thêm contractor và field
    contractor = request.POST.get('contractor')
    field = request.POST.get('field')
    
    if not username:
        return JsonResponse({'status': 'error', 'message': 'Vui lòng cung cấp username'})
    
    try:
        # Kiểm tra tên người dùng đã tồn tại chưa
        if User.objects.filter(username=username).exists():
            return JsonResponse({'status': 'error', 'message': 'Username đã tồn tại'})
        
        # Tạo người dùng mới
        user = User.objects.create_user(
            username=username,
            password=f"default_{username}"  # Mật khẩu mặc định
        )
        
        # Lấy supervisor nếu có
        supervisor = None
        if supervisor_id:
            try:
                supervisor = Profile.objects.get(id=supervisor_id, role='supervisor')
            except Profile.DoesNotExist:
                pass
        
        # Tạo profile với thông tin contractor và field
        Profile.objects.create(
            user=user,
            role='worker',
            company=company,
            supervisor=supervisor,
            contractor=contractor, # Thêm contractor
            field=field # Thêm field
        )
        
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@login_required
@require_POST
def assign_worker(request):
    """API để phân công Worker cho Supervisor"""
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập'}, status=403)
    
    worker_id = request.POST.get('worker_id')
    supervisor_id = request.POST.get('supervisor_id')
    
    if not worker_id or not supervisor_id:
        return JsonResponse({'status': 'error', 'message': 'Vui lòng cung cấp đầy đủ thông tin'})
    
    try:
        worker = Profile.objects.get(id=worker_id, role='worker')
        supervisor = Profile.objects.get(id=supervisor_id, role='supervisor')
        
        # Cập nhật supervisor cho worker
        worker.supervisor = supervisor
        worker.save()
        
        return JsonResponse({'status': 'success'})
    except Profile.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Không tìm thấy Worker hoặc Supervisor'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@login_required
def get_worker_info(request):
    """API để lấy thông tin Worker để chỉnh sửa"""
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập'}, status=403)
    
    worker_id = request.GET.get('worker_id')
    
    if not worker_id:
        return JsonResponse({'status': 'error', 'message': 'Vui lòng cung cấp worker_id'})
    
    try:
        worker = Profile.objects.get(id=worker_id, role='worker')
        
        data = {
            'username': worker.user.username,
            'company': worker.company or '',
            'supervisor_id': worker.supervisor.id if worker.supervisor else '',
            'contractor': worker.contractor or '', # Thêm contractor
            'field': worker.field or '' # Thêm field
        }
        
        return JsonResponse({'status': 'success', 'data': data})
    except Profile.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Không tìm thấy Worker'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@login_required
@require_POST
def update_worker(request):
    """API để cập nhật thông tin Worker"""
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập'}, status=403)
    
    worker_id = request.POST.get('worker_id')
    company = request.POST.get('company')
    supervisor_id = request.POST.get('supervisor_id')
    # Lấy thêm contractor và field
    contractor = request.POST.get('contractor')
    field = request.POST.get('field')
    
    if not worker_id:
        return JsonResponse({'status': 'error', 'message': 'Vui lòng cung cấp worker_id'})
    
    try:
        worker = Profile.objects.get(id=worker_id, role='worker')
        
        # Cập nhật công ty, contractor, field
        worker.company = company
        worker.contractor = contractor # Thêm contractor
        worker.field = field # Thêm field
        
        # Cập nhật supervisor
        if supervisor_id:
            supervisor = Profile.objects.get(id=supervisor_id, role='supervisor')
            worker.supervisor = supervisor
        else:
            worker.supervisor = None
        
        worker.save()
        
        return JsonResponse({'status': 'success'})
    except Profile.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Không tìm thấy Worker hoặc Supervisor'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@require_POST
@login_required
@permission_required('is_staff', raise_exception=True)
def sync_supervisor_worker_firebase(request):
    """
    Đồng bộ danh sách worker theo supervisor vào collection 'list_worker' trên Firebase.
    Mỗi document trong collection sẽ là email của supervisor và chứa:
    - Danh sách ID của các worker
    - ID của supervisor
    - Thông tin thêm của worker: nhà thầu, lĩnh vực
    """
    from .firebase_util import initialize_firebase, firestore
    
    db = initialize_firebase()
    if db is None:
        return JsonResponse({'status': 'error', 'message': 'Không thể kết nối đến Firebase.'}, status=500)

    try:
        # Lấy tất cả các Supervisor
        supervisors = Profile.objects.filter(role='supervisor').select_related('user')
        
        # Sử dụng batch để ghi nhiều document hiệu quả
        batch = db.batch()
        sync_count = 0
        
        for supervisor_profile in supervisors:
            supervisor_user = supervisor_profile.user
            supervisor_email = supervisor_user.email
            supervisor_id = str(supervisor_user.id)  # ID của supervisor

            if not supervisor_email:
                print(f"[SYNC WARNING] Supervisor {supervisor_user.username} không có email, bỏ qua.")
                continue # Bỏ qua supervisor nếu không có email

            # Lấy danh sách ID của các worker thuộc supervisor này
            worker_profiles = supervisor_profile.workers.all().select_related('user')
            
            # Chuyển đổi sang list string
            worker_ids_str = [str(worker.user_id) for worker in worker_profiles]
            
            # Tạo hoặc ghi đè document trong list_worker
            doc_ref = db.collection('list_worker').document(supervisor_email)
            batch.set(doc_ref, {
                'workers': worker_ids_str,
                'supervisor_id': supervisor_id,  # Thêm ID của supervisor
                'username': supervisor_user.username,  # Thêm username cho dễ đọc
                'company': supervisor_profile.company or "Unknown"  # Thêm thông tin công ty
            })
            
            # Thêm thông tin chi tiết của từng worker vào subcollection
            for worker_profile in worker_profiles:
                worker_user = worker_profile.user
                worker_id = str(worker_user.id)
                
                # Tạo dữ liệu worker chi tiết
                worker_data = {
                    'id': worker_id,
                    'username': worker_user.username,
                }
                
                # Thêm thông tin nhà thầu và lĩnh vực nếu có
                if worker_profile.contractor:
                    worker_data['contractor'] = worker_profile.contractor
                if worker_profile.field:
                    worker_data['field'] = worker_profile.field
                
                # Tạo reference tới document chi tiết worker
                worker_detail_ref = doc_ref.collection('worker_details').document(worker_id)
                batch.set(worker_detail_ref, worker_data)
            
            sync_count += 1
            
        # Thực hiện commit batch
        batch.commit()
        
        return JsonResponse({
            'status': 'success', 
            'message': f'Đã đồng bộ thành công {sync_count} supervisor và thông tin worker của họ lên Firebase.'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': f'Lỗi khi đồng bộ: {str(e)}'}, status=500)

# Thêm hàm xóa worker
@login_required
@require_POST
def delete_worker(request):
    """API để xóa Worker"""
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập'}, status=403)
    
    worker_id = request.POST.get('worker_id')
    
    if not worker_id:
        return JsonResponse({'status': 'error', 'message': 'Vui lòng cung cấp worker_id'})
    
    try:
        worker = Profile.objects.get(id=worker_id, role='worker')
        user = worker.user
        
        # Xóa profile trước
        worker.delete()
        
        # Xóa user
        user.delete()
        
        return JsonResponse({'status': 'success', 'message': 'Đã xóa worker thành công'})
    except Profile.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Không tìm thấy Worker'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Lỗi khi xóa worker: {str(e)}'}, status=500)

# Thêm hàm xóa supervisor
@login_required
@require_POST
def delete_supervisor(request):
    """API để xóa Supervisor"""
    if not request.user.is_superuser:
        return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập'}, status=403)
    
    supervisor_id = request.POST.get('supervisor_id')
    
    if not supervisor_id:
        return JsonResponse({'status': 'error', 'message': 'Vui lòng cung cấp supervisor_id'})
    
    try:
        supervisor = Profile.objects.get(id=supervisor_id, role='supervisor')
        
        # Kiểm tra xem supervisor có worker nào không
        if Profile.objects.filter(supervisor=supervisor).exists():
            return JsonResponse({
                'status': 'error', 
                'message': 'Supervisor này đang có workers. Vui lòng chuyển workers sang supervisor khác trước khi xóa.'
            })
        
        user = supervisor.user
        
        # Xóa profile trước
        supervisor.delete()
        
        # Xóa user
        user.delete()
        
        return JsonResponse({'status': 'success', 'message': 'Đã xóa supervisor thành công'})
    except Profile.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Không tìm thấy Supervisor'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Lỗi khi xóa supervisor: {str(e)}'}, status=500)

@api_view(['POST'])
@login_required
def save_contractor_to_firebase(request):
    """API để lưu danh sách nhà thầu và lĩnh vực vào Firebase"""
    if not request.user.is_superuser:
        return Response({'status': 'error', 'message': 'Không có quyền truy cập'}, status=403)
    
    try:
        # Khởi tạo Firebase
        db = initialize_firebase()
        if not db:
            return Response({'status': 'error', 'message': 'Không thể kết nối đến Firebase'}, status=500)
        
        # Lấy dữ liệu từ request.data
        data = request.data
        
        # Kiểm tra các định dạng dữ liệu khác nhau
        contractors = []
        
        # Trường hợp 1: data là list các đối tượng có thuộc tính name và field
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            print(f"[CONTRACTOR] Dữ liệu dạng danh sách đối tượng: {data}")
            contractors = data
        # Trường hợp 2: data có thuộc tính contractors
        elif isinstance(data, dict) and 'contractors' in data:
            print(f"[CONTRACTOR] Dữ liệu có thuộc tính contractors: {data['contractors']}")
            if isinstance(data['contractors'], list):
                contractors = data['contractors']
            else:
                # Trường hợp contractors không phải là list
                return Response({'status': 'error', 'message': 'Dữ liệu contractors không đúng định dạng'}, status=400)
        # Trường hợp 3: data không phải là định dạng hợp lệ
        else:
            print(f"[CONTRACTOR] Dữ liệu không đúng định dạng: {data}")
            return Response({'status': 'error', 'message': 'Dữ liệu không đúng định dạng'}, status=400)
        
        if not contractors:
            return Response({'status': 'error', 'message': 'Không có dữ liệu nhà thầu'}, status=400)
        
        # Lưu danh sách nhà thầu vào collection 'contractors'
        contractors_ref = db.collection('contractors')
        batch = db.batch()
        
        # Xử lý từng nhà thầu
        for contractor in contractors:
            # Tùy thuộc vào định dạng của mỗi contractor
            if isinstance(contractor, dict) and 'name' in contractor:
                contractor_data = {
                    'name': contractor['name']
                }
                # Thêm field nếu có
                if 'field' in contractor:
                    contractor_data['field'] = contractor['field']
            else:
                # Nếu contractor chỉ là chuỗi
                contractor_data = {'name': str(contractor)}
            
            # Tạo ID dựa trên tên để tránh trùng lặp
            doc_id = contractor_data['name'].lower().replace(' ', '_')
            doc_ref = contractors_ref.document(doc_id)
            batch.set(doc_ref, contractor_data, merge=True)
        
        # Thực hiện commit batch
        batch.commit()
        
        return Response({
            'status': 'success', 
            'message': f'Đã lưu {len(contractors)} nhà thầu vào Firebase'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'status': 'error', 'message': f'Lỗi: {str(e)}'}, status=500)

@login_required
def attendance_log_view(request):
    """
    View để hiển thị và lọc log chấm công trên một trang riêng.
    """
    start_date_str = request.GET.get('start_date')
    end_date_str = request.GET.get('end_date')
    search_query = request.GET.get('search', '').strip()

    # Mặc định là ngày hôm nay nếu không có ngày bắt đầu/kết thúc
    today = timezone.localdate()
    start_date = parse_date(start_date_str) if start_date_str else today
    end_date = parse_date(end_date_str) if end_date_str else today

    attendance_records = AttendanceRecord.objects.select_related('user', 'user__profile', 'user__profile__supervisor__user') \
                                              .order_by('-date', 'user__username', '-check_in')

    # Lọc theo ngày
    if start_date and end_date:
        if start_date > end_date:
             messages.warning(request, "Ngày bắt đầu không thể lớn hơn ngày kết thúc.")
             # Hoặc set end_date = start_date
             end_date = start_date
        # Chuyển đổi date thành datetime để so sánh bao gồm cả ngày kết thúc
        start_datetime = timezone.make_aware(datetime.datetime.combine(start_date, datetime.time.min))
        end_datetime = timezone.make_aware(datetime.datetime.combine(end_date, datetime.time.max))
        attendance_records = attendance_records.filter(date__gte=start_date, date__lte=end_date)
        print(f"[Log View] Filtering date between: {start_date} and {end_date}")

    # Lọc theo tên username (tìm kiếm gần đúng)
    if search_query:
        attendance_records = attendance_records.filter(
            Q(user__username__icontains=search_query) |
            Q(user__first_name__icontains=search_query) |
            Q(user__last_name__icontains=search_query)
        )
        print(f"[Log View] Filtering by search query: {search_query}")

    # Gắn thêm tên supervisor vào từng record (Tối ưu hóa)
    for record in attendance_records:
        supervisor_name = "-"
        try:
            if record.user.profile and record.user.profile.supervisor:
                supervisor_name = record.user.profile.supervisor.user.username
        except ObjectDoesNotExist:
            pass # Bỏ qua nếu profile hoặc supervisor không tồn tại
        record.supervisor_name = supervisor_name # Gán thuộc tính tạm thời

    context = {
        'attendance_records': attendance_records,
        'start_date': start_date_str if start_date_str else today.strftime('%Y-%m-%d'),
        'end_date': end_date_str if end_date_str else today.strftime('%Y-%m-%d'),
        'search_query': search_query,
    }

    # Nếu là request AJAX (từ filter), chỉ trả về partial
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        print("[Log View] AJAX request detected, returning partial.")
        # Đảm bảo trả về đúng partial template
        return render(request, 'recognition/partials/attendance_log_table.html', context)
    else:
        print("[Log View] Standard request detected, returning full page.")
        # Trả về template đầy đủ cho trang log
        return render(request, 'recognition/attendance_log_page.html', context)

# --- Views cho lên lịch nhận diện tự động ---

@login_required
def scheduled_recognition_view(request):
    """
    View để hiển thị và quản lý lịch trình nhận diện tự động
    """
    if not request.user.is_staff:
        return redirect('not-authorised')

    if request.method == 'POST':
        # Xử lý thêm/cập nhật lịch trình
        schedule_id = request.POST.get('schedule_id')
        if schedule_id:
            # Cập nhật lịch trình hiện có
            schedule = get_object_or_404(ScheduledCameraRecognition, id=schedule_id)
            form = ScheduledCameraRecognitionForm(request.POST, instance=schedule)
        else:
            # Tạo lịch trình mới
            form = ScheduledCameraRecognitionForm(request.POST)
        
        if form.is_valid():
            schedule = form.save()
            messages.success(request, f"Đã {'cập nhật' if schedule_id else 'tạo'} lịch trình thành công!")
            return redirect('scheduled-recognition')
    else:
        # Hiển thị form trống
        form = ScheduledCameraRecognitionForm()
    
    # Lấy danh sách lịch trình
    schedules = ScheduledCameraRecognition.objects.all()
    
    # Lấy nhật ký gần đây
    logs = ScheduledRecognitionLog.objects.all()[:20]
    
    context = {
        'form': form,
        'schedules': schedules,
        'logs': logs,
    }
    
    return render(request, 'recognition/scheduled_recognition.html', context)


@login_required
def edit_schedule_view(request, schedule_id):
    """
    View để chỉnh sửa lịch trình nhận diện
    """
    if not request.user.is_staff:
        return redirect('not-authorised')
    
    schedule = get_object_or_404(ScheduledCameraRecognition, id=schedule_id)
    
    if request.method == 'POST':
        form = ScheduledCameraRecognitionForm(request.POST, instance=schedule)
        if form.is_valid():
            form.save()
            messages.success(request, "Đã cập nhật lịch trình thành công!")
            return redirect('scheduled-recognition')
    else:
        form = ScheduledCameraRecognitionForm(instance=schedule)
    
    # Lấy danh sách lịch trình
    schedules = ScheduledCameraRecognition.objects.all()
    
    # Lấy nhật ký gần đây
    logs = ScheduledRecognitionLog.objects.all()[:20]
    
    context = {
        'form': form,
        'schedules': schedules,
        'logs': logs,
    }
    
    return render(request, 'recognition/scheduled_recognition.html', context)


@login_required
@require_POST
def toggle_schedule_status(request):
    """
    API để bật/tắt trạng thái lịch trình
    """
    if not request.user.is_staff:
        return JsonResponse({'status': 'error', 'message': 'Bạn không có quyền thực hiện thao tác này'})
    
    schedule_id = request.POST.get('schedule_id')
    new_status = request.POST.get('status')
    
    if not schedule_id or new_status not in ['active', 'paused']:
        return JsonResponse({'status': 'error', 'message': 'Thông tin không hợp lệ'})
    
    try:
        schedule = ScheduledCameraRecognition.objects.get(id=schedule_id)
        schedule.status = new_status
        schedule.save()
        return JsonResponse({'status': 'success'})
    except ScheduledCameraRecognition.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Không tìm thấy lịch trình'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
@require_POST
def delete_schedule(request):
    """
    API để xóa lịch trình
    """
    if not request.user.is_staff:
        return JsonResponse({'status': 'error', 'message': 'Bạn không có quyền thực hiện thao tác này'})
    
    schedule_id = request.POST.get('schedule_id')
    
    if not schedule_id:
        return JsonResponse({'status': 'error', 'message': 'Thông tin không hợp lệ'})
    
    try:
        schedule = ScheduledCameraRecognition.objects.get(id=schedule_id)
        schedule.delete()
        return JsonResponse({'status': 'success'})
    except ScheduledCameraRecognition.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Không tìm thấy lịch trình'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})


# Hàm helper để thực hiện nhận diện tự động (sẽ được gọi bởi scheduler)
def perform_scheduled_recognition(schedule_id):
    """
    Thực hiện nhận diện tự động cho một lịch trình cụ thể
    Hàm này sẽ được gọi bởi scheduler hoặc từ view test
    """
    try:
        schedule = ScheduledCameraRecognition.objects.get(id=schedule_id)
        
        # Cập nhật thời gian chạy cuối
        schedule.last_run = timezone.now()
        schedule.save()
        
        # Khởi tạo log
        log = ScheduledRecognitionLog.objects.create(
            schedule=schedule,
            message="Đang bắt đầu nhận diện tự động...",
            success=False
        )
        
        # Lấy camera config
        camera_config = schedule.camera
        
        if not camera_config:
            log.message = "Không tìm thấy cấu hình camera."
            log.save()
            return log
        
        # Kiểm tra roi
        roi = camera_config.get_roi_tuple()
        
        # Khởi tạo bộ xử lý video với roi
        from .video_roi_processor import VideoProcessorROI
        
        processor = VideoProcessorROI()
        processor.camera_source = camera_config.source
        
        # Nếu có ROI, thiết lập nó
        if roi:
            processor.set_roi(roi[0], roi[1], roi[2], roi[3])
        
        # Bắt đầu xử lý
        processor.start(mode='recognize', save_to_db=True)
        
        # Đợi một khoảng thời gian để xử lý video và nhận diện
        time.sleep(10)  # Đợi 10 giây để có thể nhận diện
        
        # Lấy kết quả nhận diện
        recognized_users = processor.get_recognized_users()
        
        # Dừng xử lý
        processor.stop()
        
        # Cập nhật log
        if recognized_users:
            recognized_users_str = ', '.join(recognized_users)
            log.recognized_users = recognized_users_str
            log.message = f"Đã nhận diện thành công {len(recognized_users)} người dùng: {recognized_users_str}"
            log.success = True
        else:
            log.message = "Hoàn thành nhận diện nhưng không tìm thấy người dùng nào."
            log.success = True
        
        # Lưu log
        log.save()
        
        # Cập nhật lịch trình cho lần chạy tiếp theo
        update_next_run_time(schedule)
        
        return log
    
    except Exception as e:
        # Xử lý lỗi và ghi log
        try:
            log = ScheduledRecognitionLog.objects.create(
                schedule_id=schedule_id,
                message=f"Lỗi khi thực hiện nhận diện tự động: {str(e)}",
                success=False
            )
            return log
        except Exception as inner_e:
            # Nếu không thể tạo log, in lỗi và trả về None
            print(f"Error creating log: {str(inner_e)}")
            return None
        
        # In lỗi ra console
        print(f"Error in scheduled recognition: {str(e)}")
        traceback.print_exc()


def update_next_run_time(schedule):
    """
    Cập nhật thời gian chạy tiếp theo cho lịch trình
    """
    if schedule.status != 'active':
        schedule.next_run = None
        schedule.save()
        return
    
    now = timezone.now()
    interval_minutes = schedule.interval_minutes
    
    # Tính toán thời gian tiếp theo
    next_run = now + timedelta(minutes=interval_minutes)
    
    # Kiểm tra nếu next_run nằm trong khoảng thời gian hợp lệ của ngày
    next_run_time = next_run.time()
    if next_run_time < schedule.start_time or next_run_time > schedule.end_time:
        # Nếu vượt quá end_time, đặt lại vào start_time của ngày tiếp theo
        tomorrow = (now + timedelta(days=1)).date()
        next_run = timezone.make_aware(datetime.combine(tomorrow, schedule.start_time))
    
    # Kiểm tra nếu ngày tiếp theo là ngày hoạt động
    active_days = schedule.active_days.split(',')
    while str(next_run.isoweekday()) not in active_days:
        # Nếu không phải ngày hoạt động, tăng thêm 1 ngày
        next_run = next_run + timedelta(days=1)
        # Đặt lại về start_time
        next_run = timezone.make_aware(datetime.combine(next_run.date(), schedule.start_time))
    
    schedule.next_run = next_run
    schedule.save()


@login_required
def test_scheduled_recognition(request, schedule_id):
    """
    View để kiểm tra lịch trình nhận diện
    """
    if not request.user.is_staff:
        return JsonResponse({'status': 'error', 'message': 'Bạn không có quyền thực hiện thao tác này'})
    
    try:
        # Thực hiện nhận diện tự động
        log = perform_scheduled_recognition(schedule_id)
        
        if log and log.success:
            return JsonResponse({'status': 'success', 'message': log.message})
        else:
            return JsonResponse({'status': 'error', 'message': log.message if log else 'Không thể thực hiện nhận diện'})
    
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

@login_required
def continuous_schedule_view(request):
    """
    Hiển thị và quản lý lịch trình chấm công liên tục
    """
    # Xử lý form thêm mới
    if request.method == 'POST':
        form = ContinuousAttendanceScheduleForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Đã tạo lịch trình chấm công liên tục thành công!")
            return redirect('continuous-schedule')
    else:
        form = ContinuousAttendanceScheduleForm()
    
    # Lấy danh sách lịch trình
    schedules = ContinuousAttendanceSchedule.objects.all().order_by('schedule_type', 'start_time')
    
    # Lấy log gần đây
    logs = ContinuousAttendanceLog.objects.all().order_by('-timestamp')[:20]
    
    context = {
        'form': form,
        'schedules': schedules,
        'logs': logs,
    }
    
    return render(request, 'recognition/continuous_schedule.html', context)

@login_required
def edit_continuous_schedule_view(request, schedule_id):
    """
    Chỉnh sửa lịch trình chấm công liên tục
    """
    schedule = get_object_or_404(ContinuousAttendanceSchedule, id=schedule_id)
    
    if request.method == 'POST':
        form = ContinuousAttendanceScheduleForm(request.POST, instance=schedule)
        if form.is_valid():
            form.save()
            messages.success(request, "Đã cập nhật lịch trình chấm công liên tục thành công!")
            return redirect('continuous-schedule')
    else:
        form = ContinuousAttendanceScheduleForm(instance=schedule)
    
    # Lấy danh sách lịch trình
    schedules = ContinuousAttendanceSchedule.objects.all().order_by('schedule_type', 'start_time')
    
    # Lấy log gần đây
    logs = ContinuousAttendanceLog.objects.all().order_by('-timestamp')[:20]
    
    context = {
        'form': form,
        'schedules': schedules,
        'logs': logs,
        'is_edit': True,
    }
    
    return render(request, 'recognition/continuous_schedule.html', context)

@login_required
@require_POST
def toggle_continuous_schedule_status(request):
    """
    Bật/tắt trạng thái lịch trình chấm công liên tục
    """
    schedule_id = request.POST.get('schedule_id')
    status = request.POST.get('status')
    
    if not schedule_id or not status or status not in ['active', 'paused']:
        return JsonResponse({'status': 'error', 'message': 'Thông tin không hợp lệ'})
    
    try:
        schedule = ContinuousAttendanceSchedule.objects.get(id=schedule_id)
        schedule.status = status
        schedule.save()
        
        # Nếu tắt lịch trình, dừng quá trình nhận diện nếu đang chạy
        if status == 'paused' and schedule.is_running:
            stop_continuous_recognition.delay(schedule_id)
        
        return JsonResponse({
            'status': 'success', 
            'message': f'Đã {"kích hoạt" if status == "active" else "tạm dừng"} lịch trình thành công'
        })
    
    except ContinuousAttendanceSchedule.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Lịch trình không tồn tại'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

@login_required
@require_POST
def delete_continuous_schedule(request):
    """
    Xóa lịch trình chấm công liên tục
    """
    schedule_id = request.POST.get('schedule_id')
    
    if not schedule_id:
        return JsonResponse({'status': 'error', 'message': 'Thông tin không hợp lệ'})
    
    try:
        schedule = ContinuousAttendanceSchedule.objects.get(id=schedule_id)
        
        # Dừng quá trình nhận diện nếu đang chạy
        if schedule.is_running:
            stop_continuous_recognition.delay(schedule_id)
        
        # Xóa lịch trình
        schedule.delete()
        
        return JsonResponse({'status': 'success', 'message': 'Đã xóa lịch trình thành công'})
    
    except ContinuousAttendanceSchedule.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Lịch trình không tồn tại'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

@login_required
def test_continuous_schedule(request, schedule_id):
    """
    Chạy thử lịch trình chấm công liên tục
    """
    try:
        schedule = ContinuousAttendanceSchedule.objects.get(id=schedule_id)
        
        # Bắt đầu task test
        test_continuous_recognition.delay(schedule_id)
        
        return JsonResponse({
            'status': 'success',
            'message': f'Đã bắt đầu chạy thử lịch trình {schedule.name}. Quá trình sẽ chạy trong khoảng 30 giây.'
        })
    
    except ContinuousAttendanceSchedule.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Lịch trình không tồn tại'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

from .tasks import test_continuous_recognition, start_continuous_recognition, stop_continuous_recognition

@login_required
def monitor_continuous_schedules(request):
    """
    Hiển thị trạng thái của các lịch trình chấm công liên tục và log hoạt động
    """
    # Lấy tất cả lịch trình chấm công liên tục
    schedules = ContinuousAttendanceSchedule.objects.all().order_by('-status', 'schedule_type', 'camera__name')
    
    # Lấy log mới nhất
    all_logs = ContinuousAttendanceLog.objects.all().order_by('-timestamp')[:50]
    
    # Lấy thời gian hiện tại
    now = timezone.now()
    current_time = now.time()
    current_day = str(now.isoweekday())  # 1-7 (1 là thứ Hai)
    
    # Kiểm tra xem các lịch trình có đang trong khung giờ hoạt động không
    for schedule in schedules:
        schedule.should_be_running = False
        # Kiểm tra ngày trong tuần
        if current_day in schedule.active_days.split(','):
            # Kiểm tra thời gian
            if schedule.start_time <= current_time <= schedule.end_time:
                schedule.should_be_running = True
        
        # Lấy 5 log gần nhất của từng lịch trình
        schedule.recent_logs = ContinuousAttendanceLog.objects.filter(
            schedule=schedule
        ).order_by('-timestamp')[:5]
    
    context = {
        'schedules': schedules,
        'all_logs': all_logs,
        'current_time': now,
        'refresh_interval': 10,  # Tự động refresh trang sau 10 giây
    }
    
    return render(request, 'recognition/monitor_schedules.html', context)
