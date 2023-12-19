from django.contrib import admin
from django.urls import path, include
from mainApp import views

urlpatterns = [
    #http://127.0.0.1:8000/
    path("", views.index, name = 'index'),
    path("scalp/", views.scalp, name = 'scalp'),
    path("shampoo/", views.shampoo, name = 'shampoo'),
    path("my/", views.my, name = 'my'),
    path("login/", views.login, name = 'login'),
    path("join/", views.join, name = 'join'),
    path("upload/", views.upload),
    path('shampooButton/', views.shampooButton, name='shampooButton'),
    path('loss/', views.loss),
    path('dandruff/', views.dandruff),
]



