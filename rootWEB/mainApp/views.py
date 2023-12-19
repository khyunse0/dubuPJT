from datetime import timezone

from django.http import JsonResponse
from django.shortcuts import render, redirect
from .models import *
from django.core.paginator import Paginator
from django.core.files.storage import  FileSystemStorage
from datetime import timezone

from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect
from .models import *
from django.core.paginator import Paginator
from django.core.files.storage import  FileSystemStorage

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
from PIL import Image

import os
from dotenv import load_dotenv

# Create your views here.

def index(request) :
    print('debug >>> mainApp /index')
    return render(request, 'mainpage/index.html')


def scalp(request) :
    print('debug >>> mainApp /scalp')
    return render(request, 'mainpage/scalp.html')


def shampoo(request) :
    print('debug >>> mainApp /shampoo')
    return render(request, 'mainpage/shampoo.html')


def my(request) :
    print('debug >>> mainApp /my')
    return render(request, 'mainpage/my.html')


def login(request) :
    print('debug >>> mainApp /login')
    print('debug >>> request method ' , request.method)
    if request.method == "POST":
        username = request.POST.get("id")
        password = request.POST.get("pwd")
        print("debug params >>> ", username, password)
        try:
            user = User_tbl.objects.get(user_id = username , user_pwd = password)
            print('debug >>> user ', user)
            if user is not None:
                request.session['user_id']=user.user_id
                print('debug >>> 로그인 성공!')
                return redirect('index')
            else:
                print('debug >>> 로그인 실패!')
                return render(request, 'mainpage/login.html', {'message': '로그인 실패!'})
        except User_tbl.DoesNotExist:
            print('debug >>> 로그인 예외!')
            return render(request, 'mainpage/login.html', {'message': '아이디와 비밀번호를 다시 확인해주세요'})
    else:
        print('debug >>> 로그인 실패1')
        return render(request, 'mainpage/login.html')


def logout_view(request):
    if 'user_id' in request.session:
        del request.session['user_id']
        print('debug >>> user deleted')
    return redirect('index')


def join(request):
    print('debug >>> mainpage /join')
    return redirect('register')


def register(request):
    if request.method == 'POST':
        id = request.POST['id']
        pwd = request.POST['pwd']
        email = request.POST['email']
        User_tbl.objects.create(user_id=id, user_pwd=pwd, user_email=email)
        return redirect('login') # login 페이지로 이동하게 만들기.
    return render(request, 'mainpage/register.html')  # 'register.html'은 회원가입 폼 템플릿


# media 폴더에 사진 업로드
def upload(request) :
    print('debug >>>> upload ')
    file = request.FILES['image']
    print('debug >>>> img ' , file , file.name)
    fs = FileSystemStorage()
    fileName = fs.save(file ,file)
    print('debug >>>> filename ', fileName)
    img_file = Image.open(file)
    img_file = img_file.resize((60, 80))
    img = image.img_to_array(img_file)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    # 모델 위치는 본인 컴퓨터 환경에 맞춰서 재설정 부탁드립니다 !
    pre_train_model = keras.models.load_model('C:/Users/user/PycharmProjects/Team2PJT/rootWEB/mainApp/static/hair_predict_model2')
    guess = pre_train_model.predict(img)
    labels = ['양호', '경증 비듬', '중등도 비듬', '중증 비듬', '경증 탈모', '중등도 탈모', '중증 탈모']
    links  = ['/shampoo', '/dandruff', '/dandruff', '/dandruff', '/loss', '/loss', '/loss']
    predicted_label = labels[np.argmax(guess)]
    links_label     = links[np.argmax(guess)]
    return render(request, 'mainpage/scalp_result.html', {'predicted_label': predicted_label, 'fileName' : fileName, 'links_label' : links_label})



def shampooButton(request):
    return render(request, 'mainpage/loss.html')

def loss(request) :
    print('debug >>> mainApp /loss')
    return render(request, 'mainpage/loss.html')

def dandruff(request) :
    print('debug >>> mainApp /dandruff')
    return render(request, 'mainpage/dandruff.html')


def toNaverMap(request) :
    print('debug >> mainApp /toNaverMap')
    load_dotenv()
    client_id = os.getenv('X-NCP-APIGW-API-KEY-ID')
    if not client_id:
        print ("debug >>> Client ID not found in environment")
        return render(request, 'error.html', {'error': 'Client ID not found'})
    context = {'client_id': client_id}
    return render(request,  'mainpage/toNaverMap.html', context)


def find_my_account(request) :
    print('debug >> mainApp /find_pwd')
    return render(request, 'mainpage/find_my_account.html')
