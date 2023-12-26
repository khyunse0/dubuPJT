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
    path("logout/", views.logout_view, name='logout'),
    path("join/", views.join, name = 'join'),
    path("register/", views.register, name='register'),
    path("find_my_account/", views.find_my_account),
    path("upload/", views.upload),
    path('shampooButton/', views.shampooButton, name='shampooButton'),
    path('loss/', views.loss),
    path('dandruff/', views.dandruff),
    path('check-user-id/', views.check_user_id, name='check_user_id'),
    path('oauth/kakao/callback/', views.KakaoSignInCallBackView.as_view(), name='kakao_callback')
]



