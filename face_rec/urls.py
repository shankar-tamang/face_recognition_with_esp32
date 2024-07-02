from django.contrib import admin
from django.urls import path, include
import face_rec

from face_rec import views, views2

urlpatterns = [

    path('recognize', views.recognize_face_view ),
    path('data', views2.receive_data)
]
