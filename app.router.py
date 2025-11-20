from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import tempfile
from collections import defaultdict
from inference_sdk import InferenceHTTPClient
import logging
import base64

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Flask 앱 초기화 ---
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# --- 환경변수 또는 설정 ---
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', 'IyocuknmhCfOMjUGWF9H')
ROBOFLOW_MODEL_ID = "dddd-gq5sp/1"

# ❗ 각 모델의 실제 경로
TOMATO_MODEL_PATH = '/home/hyunjunoh/yoloTest/cherry tomato.v6i.yolov11/train_result/weights/best.pt'
LETTUCE_GROWTH_MODEL_PATH = '/home/hyunjunoh/yoloTest/lettuce_growth_50epochs/weights/best.pt'

# ❗❗❗ 최적화된 신뢰도 임계값 (AI 성능 기반)
CONFIDENCE_THRESHOLDS = {
    'tomato': 0.70,
    'lettuce_disease': 0,  # ✅ 질병 AI (mAP 69%) - 보수적
    'lettuce_growth': 0    # ✅ 성장 AI (mAP 87%) - 높게 설정
}

# # ❗ GPT 검증 임계값 (향후 사용 예정)
# GPT_VERIFICATION_THRESHOLD = 0.70

# --- Roboflow 클라이언트 초기화 ---
RF_CLIENT = None
try:
    RF_CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )
    logger.info("✅ Roboflow 클라이언트 초기화 성공")
except Exception as e:
    logger.error(f"❌ Roboflow 클라이언트 초기화 실패: {e}")

# --- 모델 로드 ---
MODELS = {}

# 토마토 모델
try:
    MODELS['tomato'] = YOLO(TOMATO_MODEL_PATH)
    logger.info(f"✅ 토마토 모델 로드 성공")
except Exception as e:
    logger.error(f"❌ 토마토 모델 로드 실패: {e}")
    MODELS['tomato'] = None

# 상추 질병 모델 (Roboflow API)
MODELS['lettuce_disease'] = RF_CLIENT is not None
if MODELS['lettuce_disease']:
    logger.info(f"✅ 상추 질병 모델 (Roboflow API) 준비 완료 - 임계값: {CONFIDENCE_THRESHOLDS['lettuce_disease']}")
else:
    logger.error(f"❌ 상추 질병 모델 (Roboflow API) 사용 불가")

# 상추 성장 단계 모델
try:
    MODELS['lettuce_growth'] = YOLO(LETTUCE_GROWTH_MODEL_PATH)
    logger.info(f"✅ 상추 성장 단계 모델 로드 성공 - 임계값: {CONFIDENCE_THRESHOLDS['lettuce_growth']}")
    if MODELS['lettuce_growth']:
        logger.info(f"   클래스: {MODELS['lettuce_growth'].names}")
except Exception as e:
    logger.warning(f"⚠️ 상추 성장 단계 모델 로드 실패: {e}")
    MODELS['lettuce_growth'] = None


# --- 유틸리티 함수 ---
def map_growth_stage_to_english(indonesian_stage):
    """인도네시아어 성장 단계를 영어로 맵핑"""
    growth_stage_mapping = {
        # Confusion Matrix 클래스
        'belum matang': 'GERMINATION',           # 미성숙
        'rusak': 'DAMAGED',                   # 손상됨
        'siap panen': 'MATURE',     # 수확 준비
        'background': 'NO_DETECTION'          # 배경 (검출 없음)
    }
    
    # 소문자로 변환하여 매칭
    stage_lower = indonesian_stage.lower().strip()
    
    return growth_stage_mapping.get(stage_lower, indonesian_stage.upper())


def map_to_korean(class_name):
    """
    상추 질병 이름만 한글로 맵핑
    - 나머지는 원본 그대로 반환
    """
    korean_mapping = {
        # === 상추 질병 한글화 (Roboflow API 결과) ===
        'healthy': '건강함',
        'Bacterial': '세균병',
        'Downy Mildew': '노균병',
        'Mosaic Virus': '모자이크 바이러스',
        'Powdery Mildew': '흰가루병',
        'Septoria_Blight': '셉토리아 마름병',
    }
    
    # 질병 이름이 매핑에 있으면 한글 반환, 없으면 원본 반환
    return korean_mapping.get(class_name, class_name)


def translate_result_to_korean(result):
    """
    분석 결과의 질병 이름만 한글로 변환
    - bestResult_ko: 질병인 경우만 한글, 나머지는 원본
    - detections의 className_ko: 질병만 한글화
    - classSummary_ko: 질병만 한글화
    """
    # bestResult 한글화 (질병인 경우만)
    if 'bestResult' in result:
        korean_name = map_to_korean(result['bestResult'])
        # 한글로 변환된 경우에만 _ko 필드 추가
        if korean_name != result['bestResult']:
            result['bestResult_ko'] = korean_name
    
    # detections 배열의 각 항목 한글화 (질병만)
    if 'detections' in result:
        for detection in result['detections']:
            if 'className' in detection:
                korean_name = map_to_korean(detection['className'])
                # 한글로 변환된 경우에만 _ko 필드 추가
                if korean_name != detection['className']:
                    detection['className_ko'] = korean_name
    
    # classSummary 한글화 (질병만)
    if 'classSummary' in result:
        korean_summary = {}
        has_korean = False
        for k, v in result['classSummary'].items():
            korean_name = map_to_korean(k)
            if korean_name != k:
                korean_summary[korean_name] = v
                has_korean = True
        
        # 한글로 변환된 항목이 있는 경우에만 _ko 필드 추가
        if has_korean:
            result['classSummary_ko'] = korean_summary
    
    return result


def allowed_file(filename):
    """업로드된 파일의 확장자 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_detections(results, threshold, model_name="unknown"):
    """YOLO 검출 결과를 처리하여 임계값 이상인 것들만 반환"""
    if not results or not hasattr(results[0], 'boxes') or results[0].boxes is None:
        return {
            'totalDetected': 0,
            'bestResult': 'no_detection',
            'avgConfidence': 0.0,
            'classSummary': {},
            'detections': [],
            'model': model_name
        }
    
    names = results[0].names
    valid_detections = []
    class_counts = defaultdict(int)
    confidences = []

    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            confidence = float(box.conf[0])
            
            if confidence >= threshold:
                class_id = int(box.cls[0])
                class_name = names[class_id]
                
                # 성장 단계 모델인 경우 영어로 변환
                display_name = map_growth_stage_to_english(class_name) if model_name == 'lettuce_growth' else class_name
                
                valid_detections.append({
                    'className': display_name,
                    'confidence': round(confidence, 4)
                })
                
                class_counts[display_name] += 1
                confidences.append(confidence)
    
    if valid_detections:
        avg_confidence = sum(confidences) / len(confidences)
        best_detection = max(valid_detections, key=lambda x: x['confidence'])
        
        result = {
            'totalDetected': len(valid_detections),
            'bestResult': best_detection['className'],
            'avgConfidence': round(avg_confidence, 4),
            'classSummary': dict(class_counts),
            'detections': valid_detections,
            'model': model_name
        }
        
        logger.info(f"   📊 검출: {result['bestResult']} (신뢰도: {result['avgConfidence']:.2f}, 개수: {result['totalDetected']})")
        return result
    else:
        result = {
            'totalDetected': 0,
            'bestResult': 'no_detection',
            'avgConfidence': 0.0,
            'classSummary': {},
            'detections': [],
            'model': model_name
        }
        
        logger.info(f"   ℹ️ 검출 없음 (임계값 {threshold*100:.0f}% 이상 없음)")
        return result


def process_roboflow_detections(rf_response, threshold, model_name="roboflow_lettuce_disease"):
    """Roboflow API 응답(JSON)을 기존 포맷으로 변환"""
    try:
        valid_detections = []
        class_counts = defaultdict(int)
        confidences = []
        
        predictions = rf_response.get('predictions', [])

        for pred in predictions:
            confidence = float(pred.get('confidence', 0))
            
            if confidence >= threshold:
                class_name = pred.get('class', 'unknown')
                
                valid_detections.append({
                    'className': class_name,
                    'confidence': round(confidence, 4)
                })
                
                class_counts[class_name] += 1
                confidences.append(confidence)

        if valid_detections:
            avg_confidence = sum(confidences) / len(confidences)
            best_detection = max(valid_detections, key=lambda x: x['confidence'])
            
            result = {
                'totalDetected': len(valid_detections),
                'bestResult': best_detection['className'],
                'avgConfidence': round(avg_confidence, 4),
                'classSummary': dict(class_counts),
                'detections': valid_detections,
                'model': model_name
            }
            
            logger.info(f"   📊 [Roboflow] 검출: {result['bestResult']} (신뢰도: {result['avgConfidence']:.2f}, 개수: {result['totalDetected']})")
            return result
        else:
            result = {
                'totalDetected': 0,
                'bestResult': 'no_detection',
                'avgConfidence': 0.0,
                'classSummary': {},
                'detections': [],
                'model': model_name
            }
            
            logger.info(f"   ℹ️ [Roboflow] 검출 없음 (임계값 {threshold*100:.0f}% 이상 없음)")
            return result

    except Exception as e:
        logger.error(f"❌ Roboflow 응답 파싱 오류: {e}")
        return {
            'totalDetected': 0,
            'bestResult': 'analysis_failed',
            'avgConfidence': 0.0,
            'classSummary': {},
            'detections': [],
            'model': 'roboflow_parser_error'
        }


def analyze_lettuce_growth_first(temp_path):
    """
    상추 2단계 분석 로직 (성장 단계 우선)
    
    Step 1: AI1 (성장 단계 검사) - 로컬 YOLO (임계값 0.60, mAP 87%)
    Step 2: 성장 단계 미검출 시 → AI2 (질병 검사) - Roboflow API (임계값 0.60)
    Step 3: 둘 다 실패 시 → "i don't know" 반환
    """
    
    # ===== Step 1: 성장 단계 검사 (AI1) - 로컬 YOLO 우선! =====
    logger.info(f"   🌱 [Step 1] 성장 단계 검사 (AI1 - 로컬 YOLO) 시작 (임계값: {CONFIDENCE_THRESHOLDS['lettuce_growth']})")
    
    growth_analysis = None
    
    if MODELS['lettuce_growth'] is None:
        logger.warning(f"   ⚠️ 성장 단계 모델(AI1) 로드 안 됨")
        growth_analysis = {
            'totalDetected': 0,
            'bestResult': 'no_detection',
            'avgConfidence': 0.0,
            'classSummary': {},
            'detections': [],
            'model': 'growth_model_unavailable'
        }
    else:
        try:
            growth_results = MODELS['lettuce_growth'](
                temp_path, 
                verbose=False
            )
            
            growth_analysis = process_detections(
                growth_results, 
                CONFIDENCE_THRESHOLDS['lettuce_growth'],
                model_name='lettuce_growth'
            )
            
        except Exception as e:
            logger.error(f"❌ 성장 단계 검사 오류: {e}")
            growth_analysis = {
                'totalDetected': 0,
                'bestResult': 'analysis_failed',
                'avgConfidence': 0.0,
                'classSummary': {},
                'detections': [],
                'model': 'growth_analysis_error'
            }
    
    # --- 성장 단계 검사 결과 분석 ---
    
    # Case 1: '정상 성장 단계' (GERMINATION, MATURE) 검출 시에만 바로 반환
    if growth_analysis['bestResult'] in ['GERMINATION', 'MATURE']:
        logger.info(f"   ✅ [AI1] 정상 성장 단계 검출: {growth_analysis['bestResult']}")
        logger.info(f"   📊 신뢰도: {growth_analysis['avgConfidence']:.2%}")
        logger.info(f"   ➡️ 명확한 성장 단계가 감지되어 질병 검사를 건너뜁니다.")
        
        return {
            'stage': 'growth',
            'result': growth_analysis
        }
    
    # Case 2: 'DAMAGED' 또는 '미검출' 시 → 질병 검사 시도
    else:
        # 'DAMAGED'는 질병 검사가 필요하므로 기록
        if growth_analysis['bestResult'] == 'DAMAGED':
            logger.info(f"   ℹ️ [AI1] 성장 단계 검사 결과: {growth_analysis['bestResult']} (질병 확인 필요)")
        else:
             logger.warning(f"   ⚠️ [AI1] 성장 단계 검사 결과: {growth_analysis['bestResult']}")
             
        logger.info(f"   ➡️ 질병 검사(AI2)로 전환합니다...")
        
        # ===== Step 2: 질병 검사 (AI2) - Roboflow API =====
        if not MODELS['lettuce_disease']:
            logger.warning(f"   ⚠️ 질병 모델(AI2) 사용 불가")
            return {
                'stage': 'unknown',
                'result': {
                    'totalDetected': 0,
                    'bestResult': "i don't know",
                    'avgConfidence': 0.0,
                    'classSummary': {},
                    'detections': [],
                    'model': 'both_ai_unavailable',
                    'message': '성장 단계 검사 실패 및 질병 모델이 로드되지 않았습니다.'
                }
            }
        
        try:
            logger.info(f"   🔬 [Step 2] 질병 검사 (AI2 - Roboflow API) 시작 (임계값: {CONFIDENCE_THRESHOLDS['lettuce_disease']})")
            logger.info(f"   🔄 이미지를 Base64로 인코딩 중...")
            
            with open(temp_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            logger.info(f"   📤 Roboflow API 호출 중... (이미지 크기: {len(image_data)} bytes)")
            
            # Base64 인코딩된 이미지로 API 호출
            disease_results_json = RF_CLIENT.infer(
                image_data,
                model_id=ROBOFLOW_MODEL_ID
            )
            
            logger.info(f"   ✅ Roboflow API 호출 성공")
            
            disease_analysis = process_roboflow_detections(
                disease_results_json,
                CONFIDENCE_THRESHOLDS['lettuce_disease']
            )
            
            # [상황 A] 특정 질병이 명확히 검출됨
            if disease_analysis['bestResult'] not in ['no_detection', 'analysis_failed', 'healthy']:
                logger.info(f"   ✅ [AI2] 질병 검출: {disease_analysis['bestResult']}")
                return {
                    'stage': 'disease',
                    'result': disease_analysis,
                    'previous_check': 'DAMAGED' if growth_analysis['bestResult'] == 'DAMAGED' else '성장 단계 미검출'
                }
            
            # [상황 B] Healthy 검출 (병 없음)
            elif disease_analysis['bestResult'] == 'healthy':
                logger.info(f"   ✅ [AI2] healthy 검출")
                return {
                    'stage': 'healthy',
                    'result': {
                        'totalDetected': disease_analysis['totalDetected'],
                        'bestResult': 'healthy',
                        'avgConfidence': disease_analysis['avgConfidence'],
                        'classSummary': disease_analysis['classSummary'],
                        'detections': disease_analysis['detections'],
                        'model': disease_analysis['model'],
                        'message': '질병 없이 건강한 상태입니다.'
                    }
                }
            
            # [상황 C] 질병 미검출 (AI2가 아무것도 못 찾음)
            else:
                logger.warning(f"   ⚠️ [AI2] 질병 미검출")
                
                # 🔥 [핵심 수정] AI1은 'DAMAGED'인데 AI2는 '모름'인 경우 -> 생리 장해로 판단
                if growth_analysis['bestResult'] == 'DAMAGED':
                    logger.info(f"   💡 [판단] 병균은 없으나 잎 손상(DAMAGED)이 확인됨 -> 생리 장해")
                    
                    return {
                        'stage': 'caution',  # 주의 단계 (프론트엔드 처리용)
                        'result': {
                            'totalDetected': 1,
                            'bestResult': 'physiological_disorder', # 생리장해 코드
                            'bestResult_ko': '생리 장해 (물/온도 확인)', # 강제 한글 변환
                            'avgConfidence': growth_analysis['avgConfidence'], # AI1의 신뢰도 계승
                            'classSummary': {'physiological_disorder': 1},
                            'detections': [],
                            'model': 'combined_logic',
                            'message': '병해충은 발견되지 않았으나 잎 마름 등 생육 부진이 의심됩니다. 물과 온도를 점검해주세요.'
                        }
                    }
                
                # [상황 D] 진짜 아무것도 모름 (AI1 실패 + AI2 실패)
                else:
                    logger.warning(f"   ➡️ AI1(성장)과 AI2(질병) 모두 검출 실패")
                    return {
                        'stage': 'unknown',
                        'result': {
                            'totalDetected': 0,
                            'bestResult': "i don't know",
                            'avgConfidence': 0.0,
                            'classSummary': {},
                            'detections': [],
                            'model': 'both_ai_failed',
                            'message': '분석 대상을 찾을 수 없습니다.'
                        }
                    }
                
        except Exception as e:
            logger.error(f"❌ Roboflow API 호출 오류: {e}")
            return {
                'stage': 'unknown',
                'result': {
                    'totalDetected': 0,
                    'bestResult': "i don't know",
                    'avgConfidence': 0.0,
                    'classSummary': {},
                    'detections': [],
                    'model': 'disease_api_error',
                    'message': '분석 중 오류가 발생했습니다.'
                }
            }


# --- API 엔드포인트 ---
@app.route('/analyze', methods=['POST'])
def analyze_image_router():
    """이미지 분석 API"""
    crop_type = request.form.get('crop_type')
    if not crop_type:
        return jsonify({'error': "'crop_type' (tomato 또는 lettuce) 필드가 필요합니다."}), 400
    
    if crop_type not in ['tomato', 'lettuce']:
        return jsonify({'error': "'crop_type'은 'tomato' 또는 'lettuce'여야 합니다."}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': "'file' 필드가 필요합니다."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if file and allowed_file(file.filename):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name
            
            logger.info(f"📸 이미지 분석 요청: {file.filename} (crop_type: {crop_type})")
            
            if crop_type == 'lettuce':
                analysis = analyze_lettuce_growth_first(temp_path)
                
                result = analysis['result']
                result['analysis_stage'] = analysis['stage']
                
                if 'previous_check' in analysis:
                    result['previous_check'] = analysis['previous_check']
                
                # ✅ 한글 번역 적용
                result = translate_result_to_korean(result)
                
                return jsonify(result)
                
            else:
                logger.info(f"   🍅 토마토 분석 시작 (임계값: {CONFIDENCE_THRESHOLDS['tomato']})")
                
                if MODELS['tomato'] is None:
                    return jsonify({'error': '토마토 모델이 로드되지 않았습니다.'}), 500
                
                results = MODELS['tomato'](temp_path, verbose=False)
                result = process_detections(results, CONFIDENCE_THRESHOLDS['tomato'], model_name='tomato')
                
                # ✅ 한글 번역 적용
                result = translate_result_to_korean(result)
                
                return jsonify(result)

        except Exception as e:
            logger.error(f"❌ 오류: {e}")
            return jsonify({'error': '이미지 분석 중 오류 발생', 'details': str(e)}), 500
        
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.info(f"🗑️ 임시 파일 삭제 완료: {temp_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 임시 파일 삭제 실패: {e}")
    
    else:
        return jsonify({'error': '허용되지 않는 파일 형식입니다. (png, jpg, jpeg, gif, bmp만 허용)'}), 400


@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'tomato': MODELS['tomato'] is not None,
            'lettuce_disease': MODELS['lettuce_disease'],
            'lettuce_growth': MODELS['lettuce_growth'] is not None
        },
        'confidence_thresholds': CONFIDENCE_THRESHOLDS
    })


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Flask 서버 시작")
    logger.info("=" * 60)
    logger.info(f"   토마토 모델: {'✅ 로드됨' if MODELS['tomato'] else '❌ 로드 실패'}")
    logger.info(f"   상추 질병 모델 (Roboflow): {'✅ 준비됨' if MODELS['lettuce_disease'] else '❌ 사용 불가'}")
    logger.info(f"   상추 성장 모델: {'✅ 로드됨' if MODELS['lettuce_growth'] else '❌ 로드 실패'}")
    logger.info("=" * 60)
    logger.info("📊 신뢰도 임계값 (AI 성능 기반 최적화):")
    logger.info(f"   - 토마토:     {CONFIDENCE_THRESHOLDS['tomato']}")
    logger.info(f"   - 상추 질병:  {CONFIDENCE_THRESHOLDS['lettuce_disease']} (Roboflow mAP 69%)")
    logger.info(f"   - 상추 성장:  {CONFIDENCE_THRESHOLDS['lettuce_growth']} (자체 AI mAP 87%)")
    logger.info("=" * 60)
    logger.info("🔄 분석 순서: 성장 단계 → 질병 → i don't know")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)