import yaml
import torch
from ultralytics import YOLO

# 1. data.yaml 경로
yaml_path = '/home/planti/yoloTest/Lettuce Disease.v1i.yolov11/data.yaml'

# 2. data.yaml 생성
data = {
    'train': '/home/planti/yoloTest/Lettuce Disease.v1i.yolov11/train/images',
    'val': '/home/planti/yoloTest/Lettuce Disease.v1i.yolov11/valid/images',
    'test': '/home/planti/yoloTest/Lettuce Disease.v1i.yolov11/test/images',
    'names': ['Bacterial', 'Downy_mildew_on_lettuce', 'Powdery_mildew_on_lettuce', 'Septoria_Blight_on_lettuce', 'Viral', 'Wilt_and_leaf_blight_on_lettuce', 'healthy'],
    'nc': 7
}

with open(yaml_path, 'w') as f:
    yaml.dump(data, f)

# 확인
with open(yaml_path, 'r') as f:
    print(yaml.safe_load(f))


torch.cuda.empty_cache()

# 3. 모델 로드
model = YOLO('yolo11n.pt')

# 4. 클래스 정보 출력
print("Before Training:")
print(type(model.names), len(model.names))
print(model.names)

# 5. 학습 시작
model.train(
    data=yaml_path,
    epochs=30,
    patience=10,
    imgsz=416,
    project='/home/planti/yoloTest/Lettuce Disease.v1i.yolov11',  # ⬅️ 결과 저장 위치
    name='train_result'                                     # 폴더명: train_result
)

# 6. 학습 후 클래스 정보 출력
print("After Training:")
print(type(model.names), len(model.names))
print(model.names)