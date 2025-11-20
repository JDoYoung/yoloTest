import yaml
import torch
import os
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

# GPU 메모리 정리
torch.cuda.empty_cache()

# 3. 모델 로드
model = YOLO('yolo11n.pt')

# 4. 클래스 정보 출력
print("Before Training:")
print(type(model.names), len(model.names))
print(model.names)

# ==========================================
# 🎯 선택: Tune 사용 여부
# ==========================================
USE_TUNE = True  # True: Tune 사용 (느림, 최고 성능) / False: 바로 Train (빠름)

if USE_TUNE:
    # ==========================================
    # 5-A. 하이퍼파라미터 튜닝 (Tune)
    # ==========================================
    print("\n" + "="*60)
    print("🚀 하이퍼파라미터 튜닝(HPT) 시작...")
    print("   (iterations=20, 각 30 epochs, 총 약 600 epochs 소요)")
    print("   예상 시간: 6-12시간")
    print("="*60)
    
    results = model.tune(
        data=yaml_path,
        epochs=30,          # 튜닝의 각 시도(iteration)마다 최대 30 epoch
        iterations=20,      # 총 20가지의 다른 하이퍼파라미터 조합 시도
        optimizer='AdamW',  # YOLOv11에서 권장되는 옵티마이저
        patience=10,        # 10번 동안 성능 향상 없으면 해당 iteration 조기 종료
        imgsz=640,          # 이미지 크기 (질병 디테일을 위해 640 권장)
        batch=8,            # batch 크기는 GPU 메모리에 맞춰 고정
        project='/home/planti/yoloTest/Lettuce Disease.v1i.yolov11',
        name='tune_result'  # 결과 폴더명
    )
    
    # 6-A. 튜닝 결과 확인
    print("\n" + "="*60)
    print("✨ 튜닝 완료! 최적의 하이퍼파라미터:")
    print(results.best_hyp)
    print(f"\n📁 최적의 결과(best_hyp.yaml) 저장 위치: {results.save_dir}")
    print("="*60)
    
    # 7-A. 최적의 파라미터로 본 학습 시작
    print("\n" + "="*60)
    print("🚀 튜닝된 최적의 값으로 본 학습을 시작합니다...")
    print("="*60)
    
    # 튜닝으로 찾은 최고의 하이퍼파라미터 불러오기
    best_hyp_path = os.path.join(results.save_dir, 'best_hyp.yaml')
    
    # 새 모델 로드 (가중치 초기화)
    final_model = YOLO('yolo11n.pt')
    
    # 본 학습 실행 (early stopping 적용)
    final_model.train(
        data=yaml_path,
        hyp=best_hyp_path,      # ✨ 튜닝으로 찾은 최적의 값 적용!
        epochs=150,             # 본 학습은 더 길게 (150 epochs)
        patience=25,            # ✨ early stopping: 25번 동안 개선 없으면 자동 중단
        imgsz=640,
        batch=8,
        project='/home/planti/yoloTest/Lettuce Disease.v1i.yolov11',
        name='final_train_result'  # 최종 학습 결과 저장 폴더
    )
    
    print("\n" + "="*60)
    print("🎉 최종 학습 완료!")
    print(f"📁 최종 모델 저장 위치: /home/planti/yoloTest/Lettuce Disease.v1i.yolov11/final_train_result/weights/best.pt")
    print("="*60)
    
    # 학습 후 클래스 정보 출력
    print("\nAfter Training:")
    print(type(final_model.names), len(final_model.names))
    print(final_model.names)

else:
    # ==========================================
    # 5-B. 직접 학습 (Train) - early stopping 적용
    # ==========================================
    print("\n" + "="*60)
    print("🚀 학습 시작 (early stopping 적용)...")
    print("   예상 시간: 4-6시간")
    print("="*60)
    
    model.train(
        data=yaml_path,
        epochs=150,             # 최대 150 epochs (early stopping으로 조기 종료 가능)
        patience=25,            # ✨ early stopping: 25번 동안 개선 없으면 자동 중단
        imgsz=640,              # 이미지 크기 (질병 디테일을 위해 640 권장)
        weight_decay=0.0005,    # L2 정규화
        batch=8,
        mosaic=1.0,             # 모자이크 증강 (질병 데이터셋에 중요)
        lr0=0.008,              # 초기 학습률 (Roboflow 데이터셋 최적화)
        lrf=0.001,              # 최종 학습률
        optimizer='AdamW',      # YOLOv11 권장 옵티마이저
        project='/home/planti/yoloTest/Lettuce Disease.v1i.yolov11',
        name='train_result'     # 폴더명: train_result
    )
    
    print("\n" + "="*60)
    print("🎉 학습 완료!")
    print(f"📁 모델 저장 위치: /home/planti/yoloTest/Lettuce Disease.v1i.yolov11/train_result/weights/best.pt")
    print("="*60)
    
    # 6-B. 학습 후 클래스 정보 출력
    print("\nAfter Training:")
    print(type(model.names), len(model.names))
    print(model.names)

# ==========================================
# 8. 최종 모델 평가
# ==========================================
print("\n" + "="*60)
print("🔍 최종 모델 평가 중...")
print("="*60)

# 학습된 모델 경로 설정
if USE_TUNE:
    best_model_path = '/home/planti/yoloTest/Lettuce Disease.v1i.yolov11/final_train_result/weights/best.pt'
    eval_model = YOLO(best_model_path)
else:
    best_model_path = '/home/planti/yoloTest/Lettuce Disease.v1i.yolov11/train_result/weights/best.pt'
    eval_model = model  # 이미 로드된 모델 사용

# 검증 데이터로 평가
metrics = eval_model.val()

# 결과 출력
print(f"\n📊 최종 성능 지표:")
print(f"{'='*60}")
print(f"   mAP50:     {metrics.box.map50:.4f}  (IoU=0.5)")
print(f"   mAP50-95:  {metrics.box.map:.4f}  (IoU=0.5~0.95)")
print(f"   Precision: {metrics.box.mp:.4f}  (정밀도)")
print(f"   Recall:    {metrics.box.mr:.4f}  (재현율)")
print(f"{'='*60}")

# 클래스별 성능 출력
print(f"\n📋 클래스별 AP50:")
print(f"{'='*60}")
for i, class_name in enumerate(data['names']):
    if i < len(metrics.box.ap50):
        ap_value = metrics.box.ap50[i]
        bar_length = int(ap_value * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        print(f"   {class_name:30s}: {ap_value:.4f} {bar}")
print(f"{'='*60}")

# 성능 분석 및 권장사항
print(f"\n💡 성능 분석:")
mAP50 = metrics.box.map50
if mAP50 >= 0.85:
    print(f"   ✅ 우수한 성능! (mAP50: {mAP50:.3f})")
    print(f"   → 즉시 배포 가능")
elif mAP50 >= 0.75:
    print(f"   ✅ 좋은 성능 (mAP50: {mAP50:.3f})")
    print(f"   → 배포 가능, 필요시 미세 조정")
else:
    print(f"   ⚠️ 개선 필요 (mAP50: {mAP50:.3f})")
    print(f"   → 데이터 추가 수집 또는 더 큰 모델 사용 권장")

print(f"\n{'='*60}")
print(f"✨ 모든 작업 완료!")
print(f"{'='*60}")