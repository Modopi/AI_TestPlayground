"""이미지 분류 추론 스크립트

저장된 Keras 모델을 사용하여 단일 이미지를 분류합니다.

사용법:
    python predict.py --model_path model/classifier.keras --image_path ./test.jpg
    python predict.py --model_path model/classifier.keras --image_path ./test.jpg \
        --class_names 고양이 강아지 새
"""
import argparse
import tensorflow as tf
import numpy as np


def parse_args():
    """명령줄 인수를 파싱하여 반환한다."""
    parser = argparse.ArgumentParser(description='이미지 분류 추론')
    parser.add_argument('--model_path',  required=True,
                        help='학습된 모델 경로 (.keras 또는 SavedModel)')
    parser.add_argument('--image_path',  required=True,
                        help='추론할 이미지 파일 경로')
    parser.add_argument('--class_names', nargs='+', default=None,
                        help='클래스 이름 목록 (순서대로, 미지정 시 인덱스 출력)')
    parser.add_argument('--img_size',    type=int, default=224,
                        help='입력 이미지 크기 (학습 시와 동일하게 설정, 기본값: 224)')
    return parser.parse_args()


def main():
    """모델 로드 → 이미지 전처리 → 추론 → 결과 출력."""
    args = parse_args()

    # 모델 로드
    print(f"모델 로드 중: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    print("모델 로드 완료")

    # 이미지 로드 및 전처리
    # target_size: 학습 시와 동일한 크기로 리사이즈
    img = tf.keras.utils.load_img(
        args.image_path,
        target_size=(args.img_size, args.img_size)
    )
    # PIL Image → NumPy 배열 (H, W, C)
    img_array = tf.keras.utils.img_to_array(img)

    # 배치 차원 추가: (H, W, C) → (1, H, W, C)
    img_array = tf.expand_dims(img_array, 0)

    # 추론 실행
    print(f"이미지 추론 중: {args.image_path}")
    predictions = model.predict(img_array, verbose=0)

    # 가장 높은 확률의 클래스 선택
    pred_idx   = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx] * 100

    # 클래스 이름 결정 (지정된 경우 이름 사용, 아니면 인덱스)
    if args.class_names:
        if pred_idx < len(args.class_names):
            label = args.class_names[pred_idx]
        else:
            label = f"클래스_{pred_idx} (이름 미지정)"
    else:
        label = str(pred_idx)

    # 결과 출력
    print(f"\n예측 결과: {label} ({confidence:.1f}%)")

    # 상위 3개 예측 결과 출력 (클래스 이름이 있는 경우)
    if args.class_names:
        print("\n상위 예측 결과:")
        top_indices = np.argsort(predictions[0])[::-1][:3]
        for rank, idx in enumerate(top_indices, 1):
            name = args.class_names[idx] if idx < len(args.class_names) else str(idx)
            prob = predictions[0][idx] * 100
            print(f"  {rank}위: {name:<15} {prob:.1f}%")


if __name__ == '__main__':
    main()
