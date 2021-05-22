from . import models
from django.contrib.auth.models import User
from django.forms import ModelForm
#from .models import Feature, DisChargePatient,DoctorDetails, PatientDetails,AppointmentDetails
from django import forms
from .models import *
from django.contrib.auth.forms import UserCreationForm


class CreateUserForm(UserCreationForm):
	class Meta:
		model = User
		fields = ['username', 'email', 'password1', 'password2']

class FeatureForm(ModelForm):

    class Meta:
        model = Feature
        fields = ['name',
                  'age', 'sex', 'chest_pain_type',
                  'resting_blood_pressure', 'cholesterol',
                  'fasting_blood_sugar', 'rest_ecg',
                  'max_heart_rate_achieved', 'exercise_induced_angina',
                  'st_depression', 'st_slope'
                  ]


#for admin signup
class AdminSigupForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'username', 'password']
        widgets = {
            'password': forms.PasswordInput()
        }


#for student related form
class DoctorUserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'username', 'password']
        widgets = {
            'password': forms.PasswordInput()
        }


class DoctorForm(forms.ModelForm):
    class Meta:
        model = Doctor
        fields = ['name', 'address', 'mobile', 'department',  'profile_pic']


#for teacher related form
class PatientUserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'username', 'password']
        widgets = {
            'password': forms.PasswordInput()
        }


class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ['name', 'address', 'patientContact',
                  'patientSymptoms', 'profile_pic']

