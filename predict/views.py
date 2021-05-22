from django.shortcuts import render, redirect
import pickle
# data wrangling & pre-processing
import pandas as pd
import numpy as np
# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from sklearn.model_selection import train_test_split
from django.contrib.auth import login, authenticate, logout
#model validation
from sklearn.metrics import log_loss,roc_auc_score,precision_score,f1_score,recall_score,roc_curve,auc
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,fbeta_score,matthews_corrcoef
from sklearn import metrics

# machine learning algorithms
# machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import csv
from xhtml2pdf import pisa
import io
import pickle
from django.shortcuts import render, redirect, reverse, HttpResponseRedirect, HttpResponse
from django.contrib.auth.decorators import login_required, user_passes_test
from datetime import datetime, timedelta, date
from django.conf import settings
from .models import  Doctor, Patient, Feature
import joblib
from .forms import PatientForm, DoctorForm, FeatureForm, CreateUserForm
from scipy import stats

dt = pd.read_csv('new_heart.csv')
dt["sex"] = dt.sex.apply(lambda  x:'male' if x==1 else 'female')

# Split the Dataset
X = dt.drop('target', axis=1)
y = dt.target
X.head()
X['st_slope'].value_counts()

# Convert the data
X = pd.get_dummies(X)
X.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=5)



def addPatient(request):
    if request.method == "POST":
        form = FeatureForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            age = form.cleaned_data['age']
            sex = form.cleaned_data['sex']
            cp = form.cleaned_data['chest_pain_type']
            rp = form.cleaned_data['resting_blood_pressure']
            cho = form.cleaned_data['cholesterol']
            fs = form.cleaned_data['fasting_blood_sugar']
            re = form.cleaned_data['rest_ecg']
            mah = form.cleaned_data['max_heart_rate_achieved']
            ex = form.cleaned_data['exercise_induced_angina']
            std = form.cleaned_data['st_depression']
            sts = form.cleaned_data['st_slope']

            print('YOUR NAME IS: {}'.format(name))
            if (cp == 'Typical Angina'):
                chest_pain_type_asymptomatic = 0
                chest_pain_type_atypical = 0
                chest_pain_type_anginal = 1
                chest_pain_type_typical = 0
            elif (cp == 'Atypical Angina'):
                chest_pain_type_anginal = 0
                chest_pain_type_asymptomatic = 0
                chest_pain_type_atypical = 1
                chest_pain_type_typical = 0
            elif (cp == 'Non-anginal pain'):
                chest_pain_type_anginal = 0
                chest_pain_type_asymptomatic = 0
                chest_pain_type_atypical = 0
                chest_pain_type_typical = 1
            else:
                chest_pain_type_anginal = 0
                chest_pain_type_asymptomatic = 1
                chest_pain_type_atypical = 0
                chest_pain_type_typical = 0

            if (re == 'Normal'):
                rest_ecg_abnormality = 0
                rest_ecg_normal = 1
                rest_ecg_ventricular = 0
            elif(re == 'abnormality'):
                rest_ecg_abnormality = 1
                rest_ecg_normal = 0
                rest_ecg_ventricular = 0
            else:
                rest_ecg_abnormality = 0
                rest_ecg_normal = 0
                rest_ecg_ventricular = 1

            if (sts == 'Upsloping'):
                st_slope_0 = 0
                st_slope_downsloping = 0
                st_slope_flat = 0
                st_slope_upsloping = 1
            elif (sts == 'Flat'):
                st_slope_0 = 0
                st_slope_downsloping = 0
                st_slope_flat = 1
                st_slope_upsloping = 0
            elif (sts == 'Downslpoing'):
                st_slope_0 = 0
                st_slope_downsloping = 1
                st_slope_flat = 0
                st_slope_upsloping = 0
            else:
                st_slope_0 = 1
                st_slope_downsloping = 0
                st_slope_flat = 0
                st_slope_upsloping = 0

            if (sex == 'Male'):
                sex_male = 1
                sex_female = 0
            else:
                sex_male = 0
                sex_female = 1

            feature = [[age,
                        sex_male,
                        sex_female,
                        chest_pain_type_asymptomatic,
                        chest_pain_type_atypical,
                        chest_pain_type_anginal,
                        chest_pain_type_typical,
                        rp, cho, fs,
                        rest_ecg_abnormality,
                        rest_ecg_normal,
                        rest_ecg_ventricular,
                        mah, ex, std,
                        st_slope_0,
                        st_slope_downsloping,
                        st_slope_flat,
                        st_slope_upsloping]]

            featur = pd.DataFrame(feature)
            f = pd.get_dummies(featur)
       
            rf = RandomForestClassifier(criterion='entropy', n_estimators=100)
            rf.fit(X_train, y_train)
            y_rf = rf.predict(X_test)
            y_pred_rf = rf.predict(f)
            acc_rf = "{:.2f}%".format(accuracy_score(y_test, y_rf)*100)
            #print(y_rf.shape)
            #print("y_pred_rf.shape {}".format(y_pred_rf.shape))
            #print("X_train: {}".format(X_train.shape))
            #print("X_test: {}".format(X_test.shape))
            #print("y_train: {}".format(y_train.shape))
            #print("y_test: {}".format(y_test.shape))
            #print(accuracy_score(y_test, y_pred_rf))
            if y_pred_rf == 0:
                heart_rf = "No heart disease"
                #print("The has no heart disease")
            else:
                heart_rf = "Heart disease"
                #print("The has heart disease")
       
            decc = DecisionTreeClassifier()
            decc.fit(X_train, y_train)
            y_pred_decc = decc.predict(X_test)
            y_pred_dt = rf.predict(f)
            acc_dt = "{:.2f}%".format(accuracy_score(y_test, y_pred_decc)*100)
            #print("dt_pred {}".format(y_pred_dt))
            if y_pred_dt == 0:
                heart_dt = "No heart disease"
                #print("The has no heart disease")
            else:
                heart_dt = "Heart disease"
                #print("The has heart disease")
            
            mlp = MLPClassifier()
            mlp.fit(X_train, y_train)
            y_pred_mlp = mlp.predict(X_test)
            y_pred_m = rf.predict(f)
            acc_ml = "{:.2f}%".format(accuracy_score(y_test, y_pred_mlp)*100)
            #print("dt_pred {}".format(y_pred_m))
            if y_pred_m == 0:
                heart_ml = "No heart disease"
                #print("The has no heart disease")
            else:
                heart_ml = "Heart disease"
                #print("The has heart disease")
            
            if  (y_pred_rf == 0 ) and (y_pred_dt == 0) and (y_pred_m== 0):
                target = "No Heart disease"
            elif (y_pred_dt == 0) and (y_pred_rf == 0):
                target = "No Heart disease"
            elif (y_pred_rf == 0) and (y_pred_m == 0):
                target = "No Heart disease"
            elif (y_pred_dt == 0) and (y_pred_m == 0):
                target = "No Heart disease"
            else:
                target = "Heart disease"
            d = Feature.objects.create(name=name,
                                        age=age,
                                        sex=sex,
                                        chest_pain_type=cp,
                                        resting_blood_pressure=rp,
                                        cholesterol=cho,
                                        fasting_blood_sugar=fs,
                                        rest_ecg=re,
                                        max_heart_rate_achieved=mah,
                                        exercise_induced_angina=ex,
                                        st_depression=std,
                                        st_slope=sts,
                                       rf = acc_rf,
                                       dt = acc_dt,
                                       ml = acc_ml,
                                       rf_ped=heart_rf,
                                       dt_pred=heart_dt,
                                       ml_pred=heart_ml,
                                       target=target)
            d.save()
            return redirect('scores')

    form = FeatureForm()
    context = {'form': form}
    return render(request, "predict/result.html", context)


@login_required(login_url='login')
def add_doctor(request):
    form = DoctorForm()
    if request.method == 'POST':
        form = DoctorForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('homepage')
    context = {'form': form}
    return render(request, 'predict/add_doctor.html', context)


def registerPage(request):
	if request.user.is_authenticated:
		return redirect('homepage')
	else:
		form = CreateUserForm()
		if request.method == 'POST':
			form = CreateUserForm(request.POST)
			if form.is_valid():
				form.save()
				user = form.cleaned_data.get('username')
				messages.success(request, 'Account was created for ' + user)
				return redirect('login')
		context = {'form': form}
		return render(request, 'predict/register.html', context)


def loginPage(request):
	if request.user.is_authenticated:
		return redirect('homepage')
	else:
		if request.method == 'POST':
			username = request.POST.get('username')
			password = request.POST.get('password')

			user = authenticate(request, username=username, password=password)

			if user is not None:
				login(request, user)
				return redirect('homepage')
			else:
				messages.info(request, 'Username OR password is incorrect')

		context = {}
		return render(request, 'predict/login.html', context)


def logoutUser(request):
	logout(request)
	return redirect('login')


@login_required(login_url='login')
def dashboard(request):
    withheartdisease = Feature.objects.all().filter(target="Heart disease").count()
    total = Feature.objects.all().count()
    withoutheartdisease = Feature.objects.all().filter(
        target="No Heart disease").count()

    data = Feature.objects.order_by('-data_added')[:5]
    context = {'data': data}
    context = {
        'withheartdisease': withheartdisease,
        'withoutheartdisease': withoutheartdisease,
        'total': total,
        'data': data
        }
    return render(request, 'predict/dashboard.html', context)



def homeDisplay(request):
    context = {}
    return render(request, 'predict/home.html', context)


@login_required(login_url='login')
def homePage(request):
    context = {}
    return render(request, 'predict/home_page.html', context)


@login_required(login_url='login')
def update_doctor(request, pk):
    doctor = Doctor.objects.get(id=pk)
    form = DoctorForm(request.FILES, instance=doctor)
    if request.method == 'POST':
        form = DoctorForm(request.POST, request.FILES, instance=doctor)
        if form.is_valid():
            form.save()
            return redirect('homepage')
    context = {'form': form}
    return render(request, 'predict/update_doctor.html', context)


@login_required(login_url='login')
def view_patient(request):
    patients = Patient.objects.all()
    context = {'patients': patients}
    return render(request, 'predict/view_patient.html', context)


@login_required(login_url='login')
def view_doctor(request):
    doctors = Doctor.objects.all()
    context = {'doctors': doctors}
    return render(request, 'predict/view_doctor.html', context)


@login_required(login_url='login')
def getfile(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="file.csv"'
    patients = Feature.objects.all()
    writer = csv.writer(response)
    for patient in patients:
        writer.writerow([patient.age, patient.sex,
                         patient.chest_pain_type,
                         patient.resting_blood_pressure,
                         patient.cholesterol,
                         patient.fasting_blood_sugar,
                         patient.rest_ecg,
                         patient.max_heart_rate_achieved,
                         patient.exercise_induced_angina,
                         patient.st_depression,
                         patient.st_slope,
                         patient.target])
    return response


@login_required(login_url='login')
def add_patient(request):
    if request.method == 'POST':
        form = PatientForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('homepage')
    form = PatientForm()
    context = {'form': form}
    return render(request, 'predict/add_patient.html', context)


@login_required(login_url='login')
def displayData(request):
    data = Feature.objects.order_by('-data_added')
    context = {'data': data}
    return render(request, 'predict/displaydata.html', context)


@login_required(login_url='login')
def getScores(request):
    predictions = Feature.objects.order_by('-data_added')[0]
    context = {"p": predictions}
    return render(request, "predict/pred.html", context)
