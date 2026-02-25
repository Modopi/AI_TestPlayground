# 프로젝트 01: 이미지 분류기 (Image Classifier)

## 프로젝트 설명

EfficientNetB0 기반의 전이학습(Transfer Learning) 이미지 분류기입니다.
사용자 지정 데이터셋에 대해 2단계 학습(Feature Extraction → Fine-Tuning)을 수행하고,
학습된 모델을 TFLite로 변환하여 모바일/엣지 디바이스에 배포할 수 있습니다.

## 사용 기술

| 구성 요소 | 기술 |
|-----------|------|
| 백본 모델 | EfficientNetB0 (ImageNet 사전학습) |
| 데이터 파이프라인 | `tf.data` (cache, prefetch, parallel map) |
| 데이터 증강 | RandomFlip, RandomRotation |
| 모델 저장 | `.keras` 형식 |
| 모바일 배포 | TFLite (Dynamic Range Quantization) |
| 시각화 | TensorBoard |

## 디렉토리 구조

```
project01_image_classifier/
├── README.md           # 이 파일
├── train.py            # 모델 학습 스크립트
├── predict.py          # 단일 이미지 추론 스크립트
├── data/               # 학습 데이터 (사용자가 준비)
│   ├── class_A/
│   ├── class_B/
│   └── class_C/
├── model/              # 저장된 모델 (학습 후 생성)
│   └── classifier.keras
└── logs/               # TensorBoard 로그 (학습 중 생성)
```

### 데이터 디렉토리 구조

`data/` 아래 클래스별 서브폴더에 이미지를 배치합니다.

```
data/
├── 고양이/
│   ├── cat001.jpg
│   └── cat002.jpg
├── 강아지/
│   ├── dog001.jpg
│   └── dog002.jpg
└── 새/
    └── bird001.jpg
```

## 설치 및 실행 방법

### 1. 환경 활성화

```bash
conda activate tf_study
```

### 2. 모델 학습

```bash
python train.py --data_dir ./data
```

**주요 옵션:**

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | (필수) | 학습 데이터 루트 디렉토리 |
| `--epochs` | 50 | 전체 학습 에포크 수 |
| `--batch_size` | 32 | 배치 크기 |
| `--img_size` | 224 | 입력 이미지 크기 (정방형) |
| `--save_path` | `model/classifier.keras` | 모델 저장 경로 |

**예시:**

```bash
# 기본 설정으로 학습
conda activate tf_study && python train.py --data_dir ./data

# 커스텀 설정으로 학습
conda activate tf_study && python train.py \
    --data_dir ./data \
    --epochs 100 \
    --batch_size 64 \
    --img_size 224 \
    --save_path model/my_classifier.keras
```

### 3. TensorBoard 시각화

학습 중/후에 별도 터미널에서 실행:

```bash
conda activate tf_study && tensorboard --logdir ./logs
```

브라우저에서 `http://localhost:6006` 접속

### 4. 단일 이미지 추론

```bash
conda activate tf_study && python predict.py \
    --model_path model/classifier.keras \
    --image_path ./test_image.jpg \
    --class_names 고양이 강아지 새
```

## 학습 과정 설명

### Phase 1: Feature Extraction (에포크 1 ~ epochs/2)
- EfficientNetB0 가중치 고정 (`base.trainable = False`)
- 분류기 헤드만 학습 (GlobalAveragePooling → Dropout → Dense)
- 학습률: Adam 기본값 (0.001)

### Phase 2: Fine-Tuning (에포크 epochs/2 ~ epochs)
- 전체 네트워크 파인튜닝 (`base.trainable = True`)
- 낮은 학습률 사용 (1e-5) - 사전학습 가중치 손상 방지
- ModelCheckpoint로 최고 검증 정확도 모델 자동 저장

## 주의 사항

- 클래스당 최소 50장 이상의 이미지를 권장합니다
- GPU 메모리 부족 시 `--batch_size` 값을 줄이세요 (16 또는 8)
- Fine-Tuning 단계에서 학습률이 너무 크면 사전학습 가중치가 손상될 수 있습니다
