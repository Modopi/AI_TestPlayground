# 프로젝트 03: 한국어 텍스트 분류 (Korean NLP)

## 프로젝트 설명

BiLSTM(Bidirectional Long Short-Term Memory) 기반의 한국어 텍스트 분류 모델입니다.
형태소 분석기(KoNLPy)로 한국어를 전처리하고, 임베딩 + BiLSTM으로 텍스트를 분류합니다.
뉴스 카테고리 분류, 주제 분류, 의도 분류 등 다양한 한국어 NLP 태스크에 활용할 수 있습니다.

## 사용 기술

| 구성 요소 | 기술 |
|-----------|------|
| 형태소 분석 | KoNLPy (Okt 분석기) |
| 임베딩 | Embedding 레이어 (Word2Vec 방식, 학습 가능) |
| 모델 구조 | Bidirectional LSTM |
| 정규화 | Dropout + BatchNormalization |
| 프레임워크 | TensorFlow / Keras |
| 모델 저장 | `.keras` 형식 |

## 디렉토리 구조

```
project03_korean_nlp/
├── README.md           # 이 파일
├── train.py            # BiLSTM 학습 스크립트
├── predict.py          # 한국어 텍스트 분류 추론 스크립트
├── data/               # 학습 데이터 (사용자가 준비)
│   ├── train.csv       # 컬럼: text, label
│   └── test.csv
└── model/              # 저장된 모델 (학습 후 생성)
    ├── bilstm.keras    # 학습된 모델
    └── tokenizer.json  # 어휘사전 (Tokenizer 설정)
```

### 데이터 형식

`data/train.csv` 예시:

```csv
text,label
"오늘 날씨가 정말 맑고 좋네요",날씨
"최신 스마트폰 출시 소식입니다",IT
"올해 경제 성장률 전망 발표",경제
"대표팀 월드컵 예선 통과",스포츠
```

## 설치

```bash
conda activate tf_study

# KoNLPy 및 Java 런타임 필요
pip install konlpy

# macOS
brew install openjdk
export JAVA_HOME=/usr/local/opt/openjdk

# Ubuntu
sudo apt-get install default-jdk
```

## 설치 및 실행 방법

### 1. 환경 설정 및 의존성 설치

```bash
conda activate tf_study
pip install konlpy
```

### 2. 모델 학습

```bash
python train.py --data_dir ./data --num_classes 5
```

**주요 옵션:**

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | (필수) | 학습 데이터 디렉토리 |
| `--num_classes` | (필수) | 분류 클래스 수 |
| `--vocab_size` | 20000 | 어휘사전 크기 (상위 N개 단어 사용) |
| `--max_length` | 100 | 최대 시퀀스 길이 (형태소 단위) |
| `--embed_dim` | 128 | 임베딩 차원 수 |
| `--lstm_units` | 64 | LSTM 유닛 수 (단방향 기준) |
| `--epochs` | 20 | 학습 에포크 수 |
| `--batch_size` | 64 | 배치 크기 |
| `--save_dir` | `model/` | 모델 저장 디렉토리 |

**예시:**

```bash
# 뉴스 5개 카테고리 분류
conda activate tf_study && python train.py \
    --data_dir ./data \
    --num_classes 5 \
    --vocab_size 30000 \
    --max_length 150

# 짧은 텍스트 분류 (댓글, 메시지)
conda activate tf_study && python train.py \
    --data_dir ./data \
    --num_classes 3 \
    --max_length 50 \
    --lstm_units 32
```

### 3. 추론

```bash
python predict.py \
    --model_dir model/ \
    --text "오늘 코스피가 큰 폭으로 상승했습니다" \
    --class_names 경제 IT 스포츠 날씨 정치
```

## 모델 구조 설명

```
입력 텍스트 → Okt 형태소 분석 → 정수 인코딩 → 패딩
    ↓
Embedding 레이어 (vocab_size × embed_dim)
    ↓
Bidirectional LSTM (순방향 + 역방향)
    ↓
Dropout
    ↓
Dense (num_classes, softmax)
    ↓
분류 결과
```

## 한국어 NLP 특이사항

- 한국어는 교착어로 형태소 분석이 필수 (영어의 공백 분리와 다름)
- OKT (Open Korean Text) 분석기 사용: 속도와 정확도의 균형
- 자주 등장하는 조사, 어미는 어휘사전에서 제외 가능 (filter_words 설정)
- 학습 데이터가 적을 때 vocab_size를 줄이면 과적합 방지에 도움
