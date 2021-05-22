from django.contrib import admin
from django.urls import path
from . import views
from django.contrib.auth.views import LoginView, LogoutView

from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    #path('', views.data, name = 'result'),
    path('scores', views.getScores, name='scores'),
    path('register/', views.registerPage, name="register"),
   	path('login/', views.loginPage, name="login"),
   	path('logout/', views.logoutUser, name="logout"),
    
    path('', views.homeDisplay, name='home'),
    path('display/', views.displayData, name='display'),
    path('add/', views.addPatient, name='add'),
    path('dash/', views.dashboard, name='dash'),
    path('add_patient/', views.add_patient, name='add_patient'),
    path('add_doctor/', views.add_doctor, name='add_doctor'),
    path('view_patient/', views.view_patient, name='view_patient'),
    path('view_doctor/', views.view_doctor, name='view_doctor'),
    path('csv/', views.getfile, name="csv"),
    path('homepage/', views.homePage, name='homepage'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
