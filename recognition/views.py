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
from django.contrib.auth.decorators import login_required, permission_required
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
from .models import CameraConfig, AttendanceRecord
from django.utils import timezone
from django.db.models import Q
from rest_framework import generics, filters, status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAdminUser, IsAuthenticated
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

mpl.use('Agg') # Đảm bảo mpl.use('Agg') được gọi trước plt

# Khởi tạo biến global ở cấp độ module
processing_thread = None
stop_processing_event = threading.Event()
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
    # --- Thêm current_processing_camera_id vào global --- 
    global processing_thread, stop_processing_event, current_processing_camera_id
    from .video_roi_processor import stream_output, process_video_with_roi

    if not request.user.is_superuser:
        return redirect('not-authorised')

    if request.method == 'POST':
        action = request.POST.get('action')
        
        # Xử lý đặc biệt cho hành động dừng
        if action == 'stop':
            with processing_lock:
                if processing_thread and processing_thread.is_alive():
                    print("[View] Nhận lệnh dừng quá trình xử lý video.")
                    stop_processing_event.set()
                    # Không cần .join() ở đây vì có thể gây treo giao diện người dùng
                    return JsonResponse({'status': 'success', 'message': 'Đã gửi lệnh dừng quá trình xử lý.'})
                else:
                    return JsonResponse({'status': 'info', 'message': 'Không có quá trình xử lý nào đang chạy.'})
                    
        # Các hành động khác như bắt đầu xử lý - cần xác thực form đầy đủ
        form = VideoRoiForm(request.POST)
        if form.is_valid():
            form_data = form.cleaned_data
            camera_id = form_data.get('camera').id
            mode = form_data.get('mode')
            username = form_data.get('username')
            role = form_data.get('role')
            supervisor = form_data.get('supervisor')
            company = form_data.get('company') # Lấy company từ form
            email = form_data.get('email') # Lấy email từ form nếu là supervisor

            # --- Đảm bảo có giá trị company hợp lệ --- 
            if not company:
                company = 'DBplus' # Sử dụng giá trị mặc định nếu không được cung cấp
                print(f"[View Info] Company không được cung cấp, sử dụng mặc định: {company}")
            
            requested_camera = CameraConfig.objects.filter(id=camera_id).first()
            if not requested_camera:
                return JsonResponse({'status': 'error', 'message': 'Không tìm thấy camera được chọn.'}) 
            
            # --- Xử lý logic User & Profile chỉ khi mode='collect' --- 
            if mode == 'collect':
                if not username:
                    return JsonResponse({'status': 'error', 'message': 'Vui lòng nhập Username khi chọn chế độ Thu thập dữ liệu.'})
                
                # Validate role-specific inputs
                if role == 'supervisor' and not email:
                    return JsonResponse({'status': 'error', 'message': 'Vui lòng nhập Email cho Supervisor.'})
                elif role == 'worker' and not supervisor:
                    return JsonResponse({'status': 'error', 'message': 'Vui lòng chọn Supervisor cho Worker.'})
                
                try:
                    # Kiểm tra username và tạo User nếu chưa có
                    user_obj, user_created = User.objects.get_or_create(
                        username=username,
                        defaults={
                            'first_name': username, 
                            'is_active': True,
                            'email': email if role == 'supervisor' else ''
                        }
                    )
                    
                    if user_created:
                        user_obj.set_unusable_password() # Không cần đặt password thật
                        user_obj.save()
                        print(f"[View] Đã tạo người dùng mới: {username}")
                    # Nếu user đã tồn tại và là supervisor, cập nhật email nếu cần
                    elif role == 'supervisor' and email:
                        # Kiểm tra xem email đã được sử dụng bởi người dùng khác chưa
                        existing_user = User.objects.filter(email=email).exclude(username=username).first()
                        if existing_user:
                            return JsonResponse({'status': 'error', 'message': f'Email {email} đã được sử dụng bởi người dùng khác.'})
                        
                        # Cập nhật email nếu khác với email hiện tại
                        if user_obj.email != email:
                            user_obj.email = email
                            user_obj.save()
                            print(f"[View] Đã cập nhật email cho supervisor {username}: {email}")
                    
                    # Tạo hoặc cập nhật Profile
                    profile_data = {
                        'role': role,
                        'company': company,
                        'supervisor': Profile.objects.get(user=supervisor) if role == 'worker' and supervisor else None
                    }
                    
                    # Sử dụng get_or_create hoặc update_or_create
                    profile, created = Profile.objects.get_or_create(
                        user=user_obj,
                        defaults=profile_data
                    )
                    
                    if not created:
                        # Nếu profile đã tồn tại, cập nhật các trường
                        for key, value in profile_data.items():
                            setattr(profile, key, value)
                        profile.save()
                        print(f"[View] Đã cập nhật Profile cho: {username} - Role: {role}")
                    else:
                        print(f"[View] Đã tạo Profile mới cho: {username} - Role: {role}")
                    
                except Exception as e:
                    print(f"[View Error] Lỗi khi tạo/cập nhật người dùng và profile: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return JsonResponse({'status': 'error', 'message': f'Lỗi khi lưu thông tin người dùng: {str(e)}'})
            # --- Kết thúc logic User & Profile --- 

            # --- Logic bắt đầu thread --- 
            requested_roi = requested_camera.get_roi_tuple()
            if mode in ['collect', 'recognize'] and not requested_roi:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Camera "{requested_camera.name}" chưa có ROI. Vui lòng cấu hình ROI.',
                    'action_required': 'configure_roi',
                    'camera_id': requested_camera.id
                })
            
            with processing_lock:
                is_thread_alive = processing_thread is not None and processing_thread.is_alive()
                if is_thread_alive:
                    if current_processing_camera_id == requested_camera.id:
                        print(f"[View] Camera {requested_camera.id} đã đang chạy, nhưng tiếp tục mà không thông báo.")
                        return JsonResponse({'status': 'success', 'message': f'Đang tiếp tục xử lý camera "{requested_camera.name}" chế độ {mode}.'})
                    else:
                        print(f"[View] Dừng camera {current_processing_camera_id} để chuyển sang {requested_camera.id}.")
                        stop_processing_event.set()
                        try: 
                            processing_thread.join(timeout=5.0)
                        except RuntimeError: 
                            pass
                        if processing_thread.is_alive(): 
                            print("[View] Cảnh báo: Thread cũ không dừng kịp.")
                        else: 
                            print("[View] Thread cũ đã dừng.")
                        processing_thread = None
                        current_processing_camera_id = None
                        stream_output.stop_stream()
                        time.sleep(0.5)

                print(f"[View] Bắt đầu thread (Mode: {mode}, Camera: {requested_camera.name}, ROI: {requested_roi}, Username: {username})")
                stop_processing_event.clear()
                stream_output.start_stream()
                
                # Lấy các tham số khác cần truyền cho thread
                thread_kwargs = {
                    'company': company, # Đã lấy và xử lý ở trên
                    'project': None, # Tạm thời để None vì đã bỏ trường này
                    'camera_name': requested_camera.name # Thêm tên camera vào kwargs
                }

                processing_thread = threading.Thread(
                    target=process_video_with_roi,
                    args=(
                        requested_camera.source,
                        mode, 
                        requested_roi,
                        stop_processing_event,
                        stream_output,
                        username, 
                    ),
                    kwargs=thread_kwargs,
                    daemon=True
                )
                processing_thread.start()
                current_processing_camera_id = requested_camera.id
                request.session['last_camera_id'] = requested_camera.id
                request.session['last_mode'] = mode

                return JsonResponse({'status': 'success', 'message': f'Đã bắt đầu xử lý camera "{requested_camera.name}" chế độ {mode}.'})
        else:
            # Form không hợp lệ, trả về lỗi
            print(f"[View] Form không hợp lệ: {form.errors}")
            return JsonResponse({'status': 'error', 'message': 'Dữ liệu không hợp lệ.', 'errors': form.errors.as_json()})

    else:
        last_camera_id = request.session.get('last_camera_id')
        last_mode = request.session.get('last_mode', 'recognize') # Đổi default thành 'recognize'
        
        initial_data = {'mode': last_mode}
        if last_camera_id:
            try:
                initial_data['camera'] = CameraConfig.objects.get(pk=last_camera_id)
            except CameraConfig.DoesNotExist:
                 request.session.pop('last_camera_id', None)

        form = VideoRoiForm(initial=initial_data)
        with processing_lock:
             is_processing = processing_thread is not None and processing_thread.is_alive()

        records_qs = AttendanceRecord.objects.select_related('user').all().order_by('-date', '-check_in')
        
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        search_query = request.GET.get('search')
        
        if start_date:
            records_qs = records_qs.filter(date__gte=start_date)
        if end_date:
            records_qs = records_qs.filter(date__lte=end_date)
        if search_query:
            records_qs = records_qs.filter(
                Q(user__username__icontains=search_query) |
                Q(user__first_name__icontains=search_query) |
                Q(user__last_name__icontains=search_query)
            )
        
        records = records_qs[:30]

        context = {
            'form': form,
            'is_processing': is_processing,
            'records': records,
            'start_date': start_date,
            'end_date': end_date,
            'search_query': search_query
        }
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
             return render(request, 'recognition/partials/attendance_log_table.html', context)
        return render(request, 'recognition/process_video_roi.html', context)

@login_required
def select_roi_view(request, camera_id):
    if request.user.username != 'admin':
        return redirect('not-authorised')

    try:
        camera_config = CameraConfig.objects.get(pk=camera_id)
    except CameraConfig.DoesNotExist:
        messages.error(request, "Không tìm thấy cấu hình camera được yêu cầu.")
        return redirect('process-video-roi')

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

    return redirect('process-video-roi')

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
                return redirect('process-video-roi') 
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
        data = json.loads(request.body)
        crop_data = data.get('crop_data')
        natural_width = data.get('natural_width')
        natural_height = data.get('natural_height')

        if not all([crop_data, natural_width, natural_height]):
            raise ValueError("Dữ liệu crop hoặc kích thước ảnh gốc bị thiếu.")
        
        crop_x = crop_data.get('x')
        crop_y = crop_data.get('y')
        crop_width = crop_data.get('width')
        crop_height = crop_data.get('height')

        if None in [crop_x, crop_y, crop_width, crop_height]:
             raise ValueError("Dữ liệu tọa độ crop không đầy đủ.")

        print(f"[Save ROI View] Nhận dữ liệu: CamID={camera_id}, Crop={crop_data}, NaturalSize={natural_width}x{natural_height}")

        scale_factor = settings.RECOGNITION_FRAME_WIDTH / natural_width
        roi_x = int(crop_x * scale_factor)
        roi_y = int(crop_y * scale_factor)
        roi_w = int(crop_width * scale_factor)
        roi_h = int(crop_height * scale_factor)

        standard_height = int(natural_height * scale_factor)
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_w = min(roi_w, settings.RECOGNITION_FRAME_WIDTH - roi_x)
        roi_h = min(roi_h, standard_height - roi_y)

        camera_config.roi_x = roi_x
        camera_config.roi_y = roi_y
        camera_config.roi_w = roi_w
        camera_config.roi_h = roi_h
        camera_config.save()

        print(f"[Save ROI View] Đã lưu ROI đã tính toán: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

        return JsonResponse({'status': 'success', 'message': 'Đã lưu ROI thành công.', 'saved_roi': [roi_x, roi_y, roi_w, roi_h]})

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Dữ liệu JSON không hợp lệ.'}, status=400)
    except ValueError as ve:
         print(f"[Save ROI View] Lỗi giá trị: {ve}")
         return JsonResponse({'status': 'error', 'message': str(ve)}, status=400)
    except Exception as e:
        print(f"[Save ROI View] Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': 'Lỗi server nội bộ khi lưu ROI.'}, status=500)

from django.http import HttpResponse

@login_required
def get_static_frame_view(request, camera_id):
    if not request.user.is_superuser:
        return HttpResponse("Forbidden", status=403)

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
                return HttpResponse("Không thể mở nguồn video", status=500)

            ret, frame = False, None
            for _ in range(5):
                 ret, frame = cap.read()
                 if ret and frame is not None and frame.size > 0:
                      break
                 time.sleep(0.1)

            if not ret or frame is None:
                print(f"[Get Frame] Lỗi: Không thể đọc frame từ nguồn: {video_source}")
                return HttpResponse("Không thể đọc frame từ camera", status=500)

            frame_resized = imutils.resize(frame, width=settings.RECOGNITION_FRAME_WIDTH)
            print(f"[Get Frame] Đã resize frame về width={settings.RECOGNITION_FRAME_WIDTH}")

            ret, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                 print("[Get Frame] Lỗi: Không thể encode frame thành JPEG")
                 return HttpResponse("Lỗi xử lý ảnh", status=500)

            response = HttpResponse(buffer.tobytes(), content_type='image/jpeg')
            response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response['Pragma'] = 'no-cache'
            response['Expires'] = '0'
            print(f"[Get Frame] Đã gửi frame tĩnh thành công.")
            return response

        finally:
             if cap and cap.isOpened():
                 cap.release()
                 print(f"[Get Frame] Đã giải phóng camera.")

    except CameraConfig.DoesNotExist:
        return HttpResponse("Không tìm thấy camera", status=404)
    except Exception as e:
        print(f"[Get Frame] Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()
        return HttpResponse("Lỗi server nội bộ", status=500)

@login_required
def get_collect_progress_view(request):
    from .video_roi_processor import collect_progress_tracker
    if not request.user.is_superuser:
         return JsonResponse({'status': 'error', 'message': 'Không có quyền truy cập.'}, status=403)
    return JsonResponse(collect_progress_tracker)

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
