from django.db import models

class Product(models.Model):
    imgfile = models.ImageField(null=True, upload_to="", blank=True)  # 이미지 컬럼 추가