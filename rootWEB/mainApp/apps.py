from django.apps import AppConfig
import torch
from efficientnet_pytorch import EfficientNet
import os

class MainappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "mainApp"

    def ready(self):
        model_path = os.path.abspath("mainApp/static/model/alopecia_model.pt")
        print(f"MODEL_PATH >>> {model_path}")

        # 전체 모델 로드
        self.model = torch.load(model_path, map_location="cpu")
        self.model.eval()

        print("debug >>>>> EfficientNet-B4 모델 로드 완료")
