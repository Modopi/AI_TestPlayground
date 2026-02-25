# 프로젝트 02: 감성 분석기 (Sentiment Analysis)

## 프로젝트 설명

BERT(Bidirectional Encoder Representations from Transformers) 기반의 감성 분석 모델입니다.
영화 리뷰, 상품 리뷰, SNS 텍스트 등의 감성(긍정/부정/중립)을 분류합니다.
Hugging Face `transformers` 라이브러리와 TensorFlow를 함께 사용합니다.

## 사용 기술

| 구성 요소 | 기술 |
|-----------|------|
| 기반 모델 | BERT (`bert-base-uncased` 또는 한국어: `klue/bert-base`) |
| 프레임워크 | TensorFlow + Hugging Face Transformers |
| 토크나이저 | BERT Tokenizer (WordPiece) |
| 파인튜닝 | 분류 헤드 추가 후 전체 네트워크 미세 조정 |
| 모델 저장 | SavedModel 형식 |

## 디렉토리 구조

```
project02_sentiment_analysis/
├── README.md           # 이 파일
├── train.py            # BERT 파인튜닝 학습 스크립트
├── predict.py          # 감성 분석 추론 스크립트
├── data/               # 학습 데이터 (사용자가 준비)
│   ├── train.csv       # 컬럼: text, label (0=부정, 1=긍정)
│   └── test.csv
└── saved_model/        # 저장된 모델 (학습 후 생성)
```

### 데이터 형식

`data/train.csv` 예시:

```csv
text,label
"이 영화 정말 재미있었어요!",1
"기대 이하였습니다. 실망스럽네요.",0
"그냥 그랬어요. 평범한 영화.",2
```

## 설치

```bash
conda activate tf_study
pip install transformers datasets
```

## 설치 및 실행 방법

### 1. 환경 설정

```bash
conda activate tf_study
pip install transformers datasets
```

### 2. 모델 학습

```bash
python train.py --data_dir ./data --model_name bert-base-uncased
```

**주요 옵션:**

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | (필수) | 학습 데이터 디렉토리 |
| `--model_name` | `bert-base-uncased` | Hugging Face 모델 이름 |
| `--num_labels` | 2 | 분류 클래스 수 |
| `--max_length` | 128 | 최대 토큰 길이 |
| `--epochs` | 3 | 학습 에포크 수 |
| `--batch_size` | 16 | 배치 크기 |
| `--learning_rate` | 2e-5 | 학습률 (BERT 파인튜닝 권장값) |
| `--save_path` | `saved_model/` | 모델 저장 경로 |

**예시:**

```bash
# 영어 감성 분석 (2클래스: 긍정/부정)
conda activate tf_study && python train.py \
    --data_dir ./data \
    --model_name bert-base-uncased \
    --num_labels 2

# 한국어 감성 분석
conda activate tf_study && python train.py \
    --data_dir ./data \
    --model_name klue/bert-base \
    --num_labels 2 \
    --max_length 64
```

### 3. 추론

```bash
python predict.py \
    --model_path saved_model/ \
    --text "이 제품 정말 좋아요! 강력 추천합니다."
```

## BERT 파인튜닝 주의사항

- **학습률**: BERT는 2e-5 ~ 5e-5 범위를 권장. 너무 크면 파국적 망각 발생
- **에포크**: 3~5 에포크로 충분. 과적합 주의
- **배치 크기**: GPU 메모리에 따라 조정 (16 또는 32)
- **최대 길이**: 512가 최대이나 128~256이 속도/성능 균형에 좋음
- **워밍업**: 전체 스텝의 10% 정도 선형 학습률 워밍업 권장
