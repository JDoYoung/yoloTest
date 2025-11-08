import os
from ultralytics import YOLO
from collections import Counter

# --- 1. 경로 설정 ---
model_path = '/home/planti/yoloTest/Lettuce Disease.v1i.yolov11/train_result/weights/best.pt'
input_folder = '/home/planti/yoloTest/Lettuce Disease.v1i.yolov11/sample_data'
output_folder = '/home/planti/yoloTest/Lettuce Disease.v1i.yolov11/predicted'
os.makedirs(output_folder, exist_ok=True)

# --- 2. 모델 로드 ---
model = YOLO(model_path)
CLASS_NAMES = model.names
print(f"모델 로드 성공. 클래스: {CLASS_NAMES}")

# --- 3. 클래스별 객체 카운터 초기화 ---
total_object_counts = Counter()       # 클래스 이름별
total_object_counts_by_id = Counter() # 클래스 ID별
total_images_processed = 0

# --- 4. 이미지 파일 리스트 ---
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"\n--- 총 {len(image_files)}개의 이미지 분석 시작 ---\n")

# --- 5. 이미지 처리 루프 ---
for image_name in image_files:
    image_path = os.path.join(input_folder, image_name)
    total_images_processed += 1

    # 검출 수행 및 결과 이미지 저장
    results = model(image_path, conf=0.25, verbose=False, save=True, project=output_folder, name='results', exist_ok=True)

    # 검출된 객체 카운트
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES[class_id]
            total_object_counts[class_name] += 1
            total_object_counts_by_id[class_id] += 1

    print(f"✓ {image_name} 처리 완료")

# --- 6. 최종 결과 출력 ---
print("\n" + "="*60)
print("✨ 전체 샘플 데이터 분석 결과")
print("="*60)
print(f"총 처리된 이미지 수: {total_images_processed}개\n")

print("📊 검출된 클래스별 객체 개수 (클래스 이름):")
print("-"*60)
for class_id, class_name in CLASS_NAMES.items():
    count = total_object_counts[class_name]
    print(f"  {class_name:30s}: {count:4d}개")
print("-"*60)
print(f"  {'총합':30s}: {sum(total_object_counts.values()):4d}개\n")

print("📊 검출된 클래스별 객체 개수 (클래스 ID):")
print("-"*60)
for class_id, class_name in CLASS_NAMES.items():
    count = total_object_counts_by_id[class_id]
    print(f"  클래스 {class_id} ({class_name:30s}): {count:4d}개")
print("-"*60)
print(f"  {'총합':40s}: {sum(total_object_counts_by_id.values()):4d}개")
print("="*60)
print(f"\n📁 결과 이미지 저장 위치: {output_folder}/results/")