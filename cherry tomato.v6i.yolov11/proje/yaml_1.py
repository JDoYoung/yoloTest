import yaml
from ultralytics import YOLO

# 1. data.yaml 경로
yaml_path = '/home/aa/cherry tomato.v6i.yolov11/data.yaml'

# 2. data.yaml 생성
data = {
    'train': '/home/aa/yoloTest/cherry tomato.v6i.yolov11/train/images',
    'val': '/home/aa/yoloTest/cherry tomato.v6i.yolov11/valid/images',
    'test': '/home/aa/yoloTest/cherry tomato.v6i.yolov11/test/images',
    'names': ['bug', 'level 1', 'level 2', 'level 3', 'level 4', 'level 5', 'level 6'],
    'nc': 7
}

with open(yaml_path, 'w') as f:
    yaml.dump(data, f)

# 확인
with open(yaml_path, 'r') as f:
    print(yaml.safe_load(f))

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
    patience=5,
    imgsz=416,
    project='/home/aa/yoloTest/cherry tomato.v6i.yolov11',  # ⬅️ 결과 저장 위치
    name='train_result'                                     # 폴더명: train_result
)

# 6. 학습 후 클래스 정보 출력
print("After Training:")
print(type(model.names), len(model.names))
print(model.names)