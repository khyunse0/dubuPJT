from django.apps import AppConfig
import torch
import timm
import os

class MainappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "mainApp"

    def ready(self):
        # 모델 파일 경로
        model_path = os.path.abspath("mainApp/static/model/alopecia_model.pt")
        print(f"MODEL_PATH >>> {model_path}")

        # EfficientNet-B4 모델 정의 (timm 사용)
        self.model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=4)

        # 모델 가중치 로드
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print("debug >>>>> EfficientNet-B4 모델 로드 완료")
