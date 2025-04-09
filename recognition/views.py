from django.shortcuts import render, redirect
from .forms import usernameForm, DateForm, UsernameAndDateForm, DateForm_2, VideoRoiForm, AddCameraForm
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
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
#import mpld3
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math
from . import video_roi_processor
from .models import CameraConfig, AttendanceRecord
from django.utils import timezone
from django.db.models import Q
from rest_framework import generics, filters, status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import AttendanceRecordSerializer
from django.utils.dateparse import parse_date
from django.http import StreamingHttpResponse
import threading
mpl.use('Agg')

# Khởi tạo biến global ở cấp độ module
processing_thread = None
stop_processing_event = threading.Event()

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
    global processing_thread, stop_processing_event # Vẫn cần khai báo global để sửa đổi
    if not request.user.is_superuser:
        return redirect('not-authorised')

    # --- Xử lý yêu cầu POST (Bắt đầu/Dừng xử lý) ---
    if request.method == 'POST':
        action = request.POST.get('action')
        
        # --- Dừng xử lý --- 
        if action == 'stop':
            if processing_thread and processing_thread.is_alive():
                print("[View] Gửi tín hiệu dừng...")
                stop_processing_event.set() # Gửi tín hiệu dừng
                processing_thread.join(timeout=5.0) # Đợi thread kết thúc
                if processing_thread.is_alive():
                     print("[View] Cảnh báo: Thread xử lý không dừng kịp thời.")
                else:
                     print("[View] Thread xử lý đã dừng.")
                processing_thread = None
                video_roi_processor.stream_output.stop_stream() # Đảm bảo dừng stream
                messages.info(request, 'Đã dừng xử lý video.')
            else:
                messages.warning(request, 'Không có tiến trình nào đang chạy để dừng.')
            return redirect('process-video-roi')
        
        # --- Bắt đầu xử lý --- 
        elif action == 'start':
             if processing_thread and processing_thread.is_alive():
                 messages.warning(request, 'Tiến trình xử lý đã chạy. Vui lòng dừng trước khi bắt đầu lại.')
                 return redirect('process-video-roi')
                 
             form = VideoRoiForm(request.POST)
             if form.is_valid():
                 camera = form.cleaned_data['camera']
                 mode = form.cleaned_data['mode'] # Chế độ xử lý (collect, recognize, stream)
                 username = form.cleaned_data.get('username')
                 
                 if mode == 'collect' and not username:
                     messages.error(request, 'Vui lòng nhập username khi chọn chế độ thu thập dữ liệu')
                     return redirect('process-video-roi')
                 
                 roi = camera.get_roi_tuple()
                 # Cho phép stream không cần ROI, nhưng collect/recognize thì cần
                 if mode in ['collect', 'recognize'] and not roi:
                     messages.error(request, f'Camera {camera.name} chưa được cấu hình ROI. Cần ROI cho chế độ {mode}.')
                     return redirect('process-video-roi')
                 
                 # Reset cờ dừng và khởi tạo thread
                 stop_processing_event.clear()
                 video_roi_processor.stream_output.start_stream() # Bật stream output
                 
                 print(f"[View] Bắt đầu thread xử lý (Mode: {mode}, Camera: {camera.name}, ROI: {roi})")
                 processing_thread = threading.Thread(
                     target=video_roi_processor.process_video_with_roi,
                     args=(
                         camera.source,
                         mode,
                         roi,
                         stop_processing_event, # Truyền sự kiện dừng
                         video_roi_processor.stream_output, # Truyền output handler
                         username, # Có thể là None nếu mode không phải collect
                     ),
                     daemon=True # Thread sẽ tự kết thúc khi chương trình chính thoát
                 )
                 processing_thread.start()
                 messages.success(request, f'Đã bắt đầu xử lý video ở chế độ: {mode}.')
                 # Không cần đợi thread ở đây, redirect ngay
                 return redirect('process-video-roi')
             else:
                 # Form không hợp lệ khi nhấn Start
                 messages.error(request, "Dữ liệu form không hợp lệ để bắt đầu.")
                 # Render lại trang với form lỗi
                 context = {
                    'form': form, # Hiển thị lỗi form
                    'is_processing': False, # Giả sử chưa chạy
                    'records': AttendanceRecord.objects.all().order_by('-date', '-check_in')[:10] # Hiển thị vài bản ghi
                 }
                 return render(request, 'recognition/process_video_roi.html', context)
        else:
             messages.error(request, "Hành động không hợp lệ.")
             return redirect('process-video-roi')
             
    # --- Xử lý yêu cầu GET (Hiển thị trang) --- 
    else:
        form = VideoRoiForm() # Form trống cho GET request
        is_processing = processing_thread is not None and processing_thread.is_alive()

    # Lấy dữ liệu chấm công gần đây (có thể giới hạn lại nếu nhiều)
    records = AttendanceRecord.objects.all().order_by('-date', '-check_in')[:20] # Lấy 20 bản ghi mới nhất
    
    # Xử lý tìm kiếm (giữ nguyên logic tìm kiếm của bạn)
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    search_query = request.GET.get('search')
    
    if start_date:
        records = records.filter(date__gte=start_date)
    if end_date:
        records = records.filter(date__lte=end_date)
    if search_query:
        records = records.filter(
            Q(user__username__icontains=search_query) |
            Q(user__first_name__icontains=search_query) |
            Q(user__last_name__icontains=search_query)
        )
    
    context = {
        'form': form,
        'is_processing': is_processing,
        'records': records,
        'start_date': start_date,
        'end_date': end_date,
        'search_query': search_query
    }
    return render(request, 'recognition/process_video_roi.html', context)

# View cập nhật ROI cho CameraConfig cụ thể
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
        # Lưu ROI vào database
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

    return redirect('process-video-roi') # Quay lại trang form chính

# View mới để thêm Camera Config từ UI
@login_required
def add_camera_view(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    
    if request.method == 'POST':
        form = AddCameraForm(request.POST)
        if form.is_valid():
            try:
                form.save() # Lưu CameraConfig mới
                messages.success(request, f"Đã thêm camera '{form.cleaned_data['name']}' thành công.")
                # Chuyển hướng đến trang ROI để họ có thể chọn camera mới
                return redirect('process-video-roi') 
            except Exception as e:
                # Xử lý lỗi nếu tên hoặc nguồn bị trùng (do unique=True trong model)
                error_message = f"Lỗi khi thêm camera: {e}. Tên hoặc Nguồn có thể đã tồn tại."
                print(f"[Add Camera View] {error_message}")
                messages.error(request, error_message)
                # Hiển thị lại form với lỗi
                return render(request, 'recognition/add_camera.html', {'form': form})
        else:
            # Form không hợp lệ
            messages.error(request, "Dữ liệu nhập không hợp lệ. Vui lòng kiểm tra lại.")
    else:
        # Cho GET request, hiển thị form trống
        form = AddCameraForm()
    
    return render(request, 'recognition/add_camera.html', {'form': form})

@login_required
def attendance_records(request):
    """
    Hiển thị bảng chấm công
    """
    if request.user.username != 'admin':
        return redirect('not-authorised')
        
    # Lấy ngày bắt đầu và kết thúc từ request
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    # Lấy tất cả bản ghi chấm công
    records = AttendanceRecord.objects.all()
    
    # Lọc theo ngày nếu có
    if start_date:
        records = records.filter(date__gte=start_date)
    if end_date:
        records = records.filter(date__lte=end_date)
        
    # Lọc theo tên người dùng nếu có
    search_query = request.GET.get('search')
    if search_query:
        records = records.filter(
            Q(user__username__icontains=search_query) |
            Q(user__first_name__icontains=search_query) |
            Q(user__last_name__icontains=search_query)
        )
    
    # Sắp xếp theo ngày và thời gian check in
    records = records.order_by('-date', '-check_in')
    
    context = {
        'records': records,
        'start_date': start_date,
        'end_date': end_date,
        'search_query': search_query,
    }
    
    return render(request, 'recognition/attendance_records.html', context)

# --- Video Stream View ---
def generate_frames():
    """Generator function to yield JPEG encoded frames."""
    print("[Stream] Bắt đầu generate_frames")
    while True:
        frame_bytes = video_roi_processor.stream_output.get_frame_bytes()
        if frame_bytes:
            # Yield the frame in the required format for multipart response
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n'
            )
        else:
            # Nếu không có frame, đợi một chút
            # print("[Stream] Không có frame, đợi...")
            time.sleep(0.05) # Giảm tải CPU khi không có frame mới
            # Hoặc có thể yield một ảnh placeholder

@login_required # Đảm bảo chỉ người dùng đăng nhập mới xem được stream
def video_feed(request):
    """View function that returns the MJPEG stream."""
    print("[Stream] Yêu cầu video_feed")
    # Check if processing is running? (Optional)
    # if not video_roi_processor.stream_output.running:
    #    return HttpResponse("Video processing is not running.")

    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

# API Views
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

@api_view(['GET'])
def today_attendance(request):
    """
    API endpoint để lấy dữ liệu chấm công của ngày hôm nay
    """
    today = datetime.now().date()
    records = AttendanceRecord.objects.filter(date=today)
    serializer = AttendanceRecordSerializer(records, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def my_attendance(request):
    """
    API endpoint để lấy dữ liệu chấm công của một người dùng theo tham số username
    """
    # Lấy tham số từ request
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
            
    serializer = AttendanceRecordSerializer(queryset, many=True)
    return Response(serializer.data)
