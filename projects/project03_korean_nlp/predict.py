"""한국어 텍스트 분류 BiLSTM 추론 스크립트

학습된 BiLSTM 모델을 사용하여 한국어 텍스트를 분류합니다.
KoNLPy Okt 형태소 분석기와 저장된 어휘사전을 사용합니다.

사용법:
    python predict.py --model_dir model/ \
        --text "코스피가 오늘 큰 폭으로 상승했습니다" \
        --class_names 경제 IT 스포츠 날씨 정치
"""
import argparse
import json
import os
import numpy as np
import tensorflow as tf

# KoNLPy 형태소 분석기 임포트
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False
    print("경고: konlpy가 설치되지 않았습니다.")
    print("설치 명령어: pip install konlpy")


def parse_args():
    """명령줄 인수를 파싱하여 반환한다."""
    parser = argparse.ArgumentParser(description='한국어 BiLSTM 텍스트 분류 추론')
    parser.add_argument('--model_dir',   required=True,
                        help='모델 저장 디렉토리 (bilstm.keras + tokenizer.json 포함)')
    parser.add_argument('--text',        required=True,
                        help='분류할 한국어 텍스트')
    parser.add_argument('--class_names', nargs='+', default=None,
                        help='클래스 이름 목록 (순서대로)')
    return parser.parse_args()


def load_tokenizer_config(model_dir):
    """저장된 어휘사전과 설정을 로드한다.

    Args:
        model_dir: tokenizer.json이 있는 디렉토리

    Returns:
        vocab (dict), label2idx (dict), max_length (int)
    """
    config_path = os.path.join(model_dir, 'tokenizer.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['vocab'], config['label2idx'], config['max_length']


def preprocess_text(text, okt, vocab, max_length):
    """한국어 텍스트를 모델 입력 형태로 전처리한다.

    처리 순서:
    1. Okt 형태소 분석
    2. 어휘사전으로 정수 변환 (미등록 형태소 → UNK)
    3. max_length에 맞게 잘림/패딩

    Args:
        text: 입력 텍스트 문자열
        okt: Okt 인스턴스
        vocab: {형태소: 인덱스} 딕셔너리
        max_length: 최대 시퀀스 길이

    Returns:
        (1, max_length) 형태의 NumPy 배열
    """
    # 1. 형태소 분석 (정규화 포함)
    morphs = okt.morphs(text, norm=True, stem=False)

    # 2. 정수 변환 (사전에 없는 형태소는 UNK(1) 처리)
    seq = [vocab.get(m, 1) for m in morphs]

    # 3. 길이 조정 (잘림 또는 후미 패딩)
    if len(seq) > max_length:
        seq = seq[:max_length]
    else:
        seq = seq + [0] * (max_length - len(seq))

    # 배치 차원 추가: (max_length,) → (1, max_length)
    return np.array([seq], dtype=np.int32)


def main():
    """모델 및 토크나이저 로드 → 추론 → 결과 출력."""
    if not KONLPY_AVAILABLE:
        print("konlpy를 먼저 설치하세요: pip install konlpy")
        return

    args = parse_args()

    # 어휘사전 및 설정 로드
    print(f"어휘사전 로드 중: {args.model_dir}")
    vocab, label2idx, max_length = load_tokenizer_config(args.model_dir)
    print(f"어휘사전 크기: {len(vocab)} | 최대 길이: {max_length}")

    # 인덱스 → 레이블 역매핑
    idx2label = {v: k for k, v in label2idx.items()}

    # 모델 로드
    model_path = os.path.join(args.model_dir, 'bilstm.keras')
    print(f"모델 로드 중: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("로드 완료")

    # Okt 초기화
    print("형태소 분석기 초기화 중...")
    okt = Okt()

    # 텍스트 전처리
    print(f"\n입력 텍스트: \"{args.text}\"")

    # 형태소 분석 결과 출력 (디버깅 목적)
    morphs = okt.morphs(args.text, norm=True)
    print(f"형태소 분석: {morphs}")

    # 어휘사전에 있는 형태소만 표시
    known = [m for m in morphs if m in vocab]
    unk   = [m for m in morphs if m not in vocab]
    print(f"사전 등록: {known}")
    if unk:
        print(f"미등록(UNK): {unk}")

    # 모델 입력으로 변환
    X = preprocess_text(args.text, okt, vocab, max_length)

    # 추론 실행
    predictions = model.predict(X, verbose=0)
    probs = predictions[0]

    # 최고 확률 클래스 선택
    pred_idx   = np.argmax(probs)
    confidence = probs[pred_idx] * 100

    # 클래스 이름 결정
    # 우선순위: 1) args.class_names, 2) label2idx 역매핑, 3) 인덱스
    if args.class_names and pred_idx < len(args.class_names):
        label = args.class_names[pred_idx]
        name_source = args.class_names
    elif pred_idx in idx2label:
        label = idx2label[pred_idx]
        name_source = [idx2label.get(i, str(i)) for i in range(len(probs))]
    else:
        label = str(pred_idx)
        name_source = None

    # 결과 출력
    print(f"\n분류 결과: {label} ({confidence:.1f}%)")

    # 전체 클래스 확률 출력
    print("\n클래스별 확률:")
    sorted_indices = np.argsort(probs)[::-1]
    for idx in sorted_indices:
        if name_source:
            name = name_source[idx] if idx < len(name_source) else str(idx)
        else:
            name = str(idx)
        bar = '█' * int(probs[idx] * 25)
        print(f"  {name:<12} {probs[idx]*100:5.1f}% {bar}")


if __name__ == '__main__':
    main()
