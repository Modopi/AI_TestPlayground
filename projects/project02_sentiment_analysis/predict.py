"""감성 분석 BERT 추론 스크립트

학습된 BERT 모델을 사용하여 텍스트의 감성을 분류합니다.

사용법:
    python predict.py --model_path saved_model/final_model \
        --text "이 제품 정말 좋아요!"
    python predict.py --model_path saved_model/final_model \
        --text "완전 실망입니다." \
        --class_names 부정 긍정
"""
import argparse
import numpy as np

# Hugging Face 라이브러리 임포트
try:
    from transformers import BertTokenizer, TFBertForSequenceClassification
    import tensorflow as tf
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("경고: transformers 라이브러리가 설치되지 않았습니다.")
    print("설치 명령어: pip install transformers")


def parse_args():
    """명령줄 인수를 파싱하여 반환한다."""
    parser = argparse.ArgumentParser(description='BERT 감성 분석 추론')
    parser.add_argument('--model_path',  required=True,
                        help='저장된 모델 디렉토리 경로')
    parser.add_argument('--text',        required=True,
                        help='감성 분석할 텍스트')
    parser.add_argument('--class_names', nargs='+', default=None,
                        help='클래스 이름 목록 (예: 부정 긍정)')
    parser.add_argument('--max_length',  type=int, default=128,
                        help='최대 토큰 길이 (학습 시와 동일하게 설정)')
    return parser.parse_args()


def predict_sentiment(model, tokenizer, text, max_length, class_names=None):
    """단일 텍스트의 감성을 예측한다.

    Args:
        model: TFBertForSequenceClassification 인스턴스
        tokenizer: BertTokenizer 인스턴스
        text: 분석할 텍스트 문자열
        max_length: 최대 토큰 길이
        class_names: 클래스 이름 리스트 (None이면 인덱스 사용)

    Returns:
        (predicted_class, confidence, all_probs) 튜플
    """
    # 텍스트 토크나이징
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    # BERT 추론 (logits 반환)
    outputs = model(inputs, training=False)
    logits = outputs.logits  # shape: (1, num_labels)

    # Softmax로 확률 변환
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

    # 최고 확률 클래스 선택
    pred_idx   = np.argmax(probs)
    confidence = probs[pred_idx] * 100

    # 클래스 이름 결정
    if class_names and pred_idx < len(class_names):
        label = class_names[pred_idx]
    else:
        label = str(pred_idx)

    return label, confidence, probs


def main():
    """모델 로드 → 텍스트 추론 → 결과 출력."""
    if not HF_AVAILABLE:
        print("transformers 라이브러리를 먼저 설치하세요: pip install transformers")
        return

    args = parse_args()

    # 모델 및 토크나이저 로드
    print(f"모델 로드 중: {args.model_path}")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model     = TFBertForSequenceClassification.from_pretrained(args.model_path)
    print("로드 완료")

    # 감성 예측
    print(f"\n입력 텍스트: \"{args.text}\"")
    label, confidence, probs = predict_sentiment(
        model, tokenizer,
        args.text,
        args.max_length,
        args.class_names
    )

    # 결과 출력
    print(f"\n감성 분석 결과: {label} ({confidence:.1f}%)")

    # 전체 클래스 확률 출력
    if args.class_names:
        print("\n클래스별 확률:")
        for i, prob in enumerate(probs):
            name = args.class_names[i] if i < len(args.class_names) else str(i)
            bar  = '█' * int(prob * 30)  # 간단한 막대 그래프
            print(f"  {name:<10} {prob*100:5.1f}% {bar}")
    else:
        print("\n클래스별 확률:")
        for i, prob in enumerate(probs):
            print(f"  클래스 {i}: {prob*100:.1f}%")


if __name__ == '__main__':
    main()
