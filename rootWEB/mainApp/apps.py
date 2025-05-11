from django.apps import AppConfig
import torch
import timm
import os

class MainappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "mainApp"

    def ready(self) :
        model_path = os.path.abspath("mainApp/static/model/alopecia_model.pt")
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.eval()
        print("debug >>>>> EfficientNet-B4 모델 로드")
