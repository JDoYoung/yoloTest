import os
from ultralytics import YOLO

# 모델 경로 (학습한 모델 경로 확인)
model_path = '/home/aa/runs/detect/train/weights/best.pt'  # 또는 last.pt
model = YOLO(model_path)

# 예측할 이미지 폴더
input_folder = '/home/aa/yoloTest/cherry tomato.v6i.yolov11/sample_data'
output_folder = '/home/aa/yoloTest/cherry tomato.v6i.yolov11/predicted'

# 출력 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# 이미지 예측 루프
for image_name in image_files:
    image_path = os.path.join(input_folder, image_name)
    
    # 예측
    results = model(image_path)
    
    # 예측 결과 저장
    save_path = os.path.join(output_folder, f'pred_{image_name}')
    results[0].save(filename=save_path)

    print(f"\n{image_name} 예측 결과:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  → Class: {model.names[cls_id]}, Confidence: {conf:.2f}")

print("\n 모든 이미지 예측 완료! 결과는 다음 위치에 저장됨:")
print(f"→ {output_folder}")