from django.contrib import admin
from .models import Patient, Feature, Doctor


class FeatureClass(admin.ModelAdmin):
    list_display = ('name', 'age', 'sex', 'target', 'data_added')
    

class DoctorsClass(admin.ModelAdmin):
    list_display = ('name', 'address', 'mobile', 'department')
    

class PatientsClass(admin.ModelAdmin):
    list_display = ('name', 'address', 'patientContact',
                    'patientSymptoms', 'admittedOn')


admin.site.register(Patient, PatientsClass)
admin.site.register(Feature, FeatureClass)
admin.site.register(Doctor, DoctorsClass)
