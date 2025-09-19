# 확인된 객체의 총 개수와 각 레벨별 개수 코드 반영
import os
from ultralytics import YOLO
from collections import Counter

# 모델 경로
model_path = '/home/aa/runs/detect/train/weights/best.pt'
model = YOLO(model_path)

# 폴더 경로
input_folder = '/home/aa/yoloTest/cherry tomato.v6i.yolov11/sample_data'
output_folder = '/home/aa/yoloTest/cherry tomato.v6i.yolov11/predicted'
os.makedirs(output_folder, exist_ok=True)

# 카운터 초기화
class_counts = Counter()
total_detected = 0

# 이미지 처리
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for image_name in image_files:
    image_path = os.path.join(input_folder, image_name)
    results = model(image_path)

    # 예측된 이미지 저장
    save_path = os.path.join(output_folder, f'pred_{image_name}')
    results[0].save(filename=save_path)

    print(f"\n{image_name} 예측 결과:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        # 카운트 누적
        class_counts[class_name] += 1
        total_detected += 1

        print(f"  → Class: {class_name}, Confidence: {conf:.2f}")

# 결과 출력
print("\n최종 감지 결과 요약:")
print(f"총 감지된 객체 수: {total_detected}개\n")
for class_name in model.names.values():
    print(f"{class_name}: {class_counts[class_name]}개")