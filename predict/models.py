from django.db import models


class Feature(models.Model):
    gender = (
        ('Male', 'Male'), ('Female', 'Female')
    )
    cp = (
        ('Typical Angina', 'Typical Angina'),
        ('Atypical Angina', 'Atypical Angina'),
        ('Non-anginal pain', 'Non-anginal pain'),
        ('Asymptomatic', 'Asymptomatic')
    )
    ecg = (
        ('Normal', 'Normal'), ('ST-T wave abnormality', 'ST-T wave abnormality'),
        ('Left ventricular hypertrophy', 'Left ventricular hypertrophy')
    )
    st = (
        ('Upsloping', 'Upsloping'), ('Flat', 'Flat'), ('Downslpoing', 'Downslpoing')
    )
    name = models.CharField(max_length=100, blank=True, null=True)
    age = models.IntegerField()
    sex = models.CharField(max_length=250, choices=gender,
                           default="Select Gender")
    chest_pain_type = models.CharField(
        max_length=250, choices=cp, default='Typical Angina')
    resting_blood_pressure = models.IntegerField()
    cholesterol = models.IntegerField()
    fasting_blood_sugar = models.IntegerField()
    rest_ecg = models.CharField(max_length=250, choices=ecg, default='Normal')
    max_heart_rate_achieved = models.IntegerField()
    exercise_induced_angina = models.IntegerField()
    st_depression = models.FloatField(max_length=250)
    st_slope = models.CharField(
        max_length=250, choices=st, default='Upsloping')
    rf = models.CharField(max_length=50)
    dt = models.CharField(max_length=50)
    ml = models.CharField(max_length=50)
    rf_ped = models.CharField(max_length=250)
    dt_pred = models.CharField(max_length=250)
    ml_pred = models.CharField(max_length=250)
    target = models.CharField(max_length=250, default='No heart disease')
    data_added = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


departments = [('Physician', 'Physician'),
               ('Dentist', 'Dentist'),
               ('Cardiologist', 'Cardiologist'),
               ('Dermatologists', 'Dermatologists'),
               ('Emergency Medicine Specialists',
                'Emergency Medicine Specialists'),
               ('Anesthesiologists', 'Anesthesiologists'),
               ('Colon and Rectal Surgeons', 'Colon and Rectal Surgeons')
               ]


class Doctor(models.Model):
    #user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=250)
    profile_pic = models.ImageField(
        upload_to='profile_pic', null=True, blank=True)
    address = models.CharField(max_length=100)
    mobile = models.CharField(max_length=25, null=True)
    department = models.CharField(
        max_length=100, choices=departments, default='Physician')

    def __str__(self):
        return self.name


class Patient(models.Model):
    #user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=250)
    profile_pic = models.ImageField(
        upload_to='profile_pic', null=True, blank=True)
    address = models.CharField(max_length=100)
    patientContact = models.CharField(max_length=25, null=False)
    patientSymptoms = models.CharField(max_length=100, null=False)
    admittedOn = models.DateField(auto_now=True)

    def __str__(self):
        return self.name


