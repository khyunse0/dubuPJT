from django.db import models

class Product(models.Model):
    imgfile = models.ImageField(null=True, upload_to="", blank=True)  # 이미지 컬럼 추가


class User_tbl(models.Model) :
    user_id    = models.CharField(primary_key=True , max_length=50)
    user_pwd   = models.CharField(max_length=50)
    user_email = models.EmailField()
    user_name  = models.CharField(max_length=50)
    user_point = models.IntegerField(default=1000)
    user_img   = models.CharField(max_length=50)

    def __str__(self) :
        return self.user_id+'\t'+self.user_pwd+'\t'+self.user_name

