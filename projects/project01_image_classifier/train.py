"""이미지 분류기 학습 스크립트

EfficientNetB0 기반 전이학습 모델을 2단계로 학습합니다.
  Phase 1: Feature Extraction  - 백본 고정, 헤드만 학습
  Phase 2: Fine-Tuning         - 전체 네트워크 미세 조정

사용법:
    python train.py --data_dir ./data
    python train.py --data_dir ./data --epochs 100 --batch_size 64
"""
import argparse
import tensorflow as tf
from pathlib import Path


def parse_args():
    """명령줄 인수를 파싱하여 반환한다."""
    parser = argparse.ArgumentParser(description='이미지 분류기 학습')
    parser.add_argument('--data_dir',   type=str, required=True,
                        help='학습 데이터 루트 디렉토리 (클래스별 서브폴더 포함)')
    parser.add_argument('--epochs',     type=int, default=50,
                        help='전체 학습 에포크 수 (기본값: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기 (기본값: 32)')
    parser.add_argument('--img_size',   type=int, default=224,
                        help='입력 이미지 크기 (정방형, 기본값: 224)')
    parser.add_argument('--save_path',  type=str, default='model/classifier.keras',
                        help='모델 저장 경로 (기본값: model/classifier.keras)')
    return parser.parse_args()


def build_model(num_classes, img_size):
    """EfficientNetB0 기반 전이학습 모델을 생성한다.

    Args:
        num_classes: 분류 클래스 수
        img_size: 입력 이미지 크기 (정방형)

    Returns:
        컴파일되지 않은 tf.keras.Model
    """
    # EfficientNetB0 기반 전이학습 모델
    # include_top=False: ImageNet 분류 헤드 제외
    # weights='imagenet': ImageNet 사전학습 가중치 사용
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    # Feature Extraction 단계: 백본 가중치 고정
    base.trainable = False

    # 함수형 API로 모델 구성
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    # EfficientNet 전용 전처리 (0-255 → 정규화)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    # 백본 통과 (training=False: BatchNorm 고정)
    x = base(x, training=False)

    # 분류기 헤드
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # 공간 차원 평균 풀링
    x = tf.keras.layers.Dropout(0.2)(x)               # 과적합 방지
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


def main():
    """메인 학습 루프."""
    args = parse_args()

    # 데이터 로드 (train/validation 분리)
    print(f"데이터 로드 중: {args.data_dir}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset='training',
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset='validation',
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )

    # 클래스 정보 출력
    num_classes = len(train_ds.class_names)
    print(f"클래스 수: {num_classes}")
    print(f"클래스 목록: {train_ds.class_names}")

    # 전처리 파이프라인 최적화
    AUTOTUNE = tf.data.AUTOTUNE

    # 데이터 증강 레이어 (학습 시에만 적용)
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),   # 좌우 반전
        tf.keras.layers.RandomRotation(0.1),         # 최대 ±36도 회전
    ])

    # 학습 데이터: 증강 + 캐시 + 프리페치
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    ).cache().prefetch(AUTOTUNE)

    # 검증 데이터: 캐시 + 프리페치만 적용 (증강 없음)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # 모델 구성 및 Phase 1 컴파일
    print("\n모델 구성 중...")
    model = build_model(num_classes, args.img_size)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 저장 디렉토리 생성
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    # 공통 콜백 설정
    callbacks = [
        # 검증 정확도 기준 최고 모델 자동 저장
        tf.keras.callbacks.ModelCheckpoint(
            args.save_path,
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),
        # 검증 손실 개선 없으면 조기 종료
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor='val_loss'
        ),
        # TensorBoard 로깅
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=0  # 히스토그램 비활성화 (속도 우선)
        ),
    ]

    # Phase 1: Feature Extraction (백본 고정)
    print("\n" + "=" * 50)
    print("Phase 1: Feature Extraction")
    print("EfficientNetB0 백본 고정, 분류기 헤드만 학습")
    print("=" * 50)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs // 2,
        callbacks=callbacks
    )

    # Phase 2: Fine-Tuning (전체 네트워크 미세 조정)
    print("\n" + "=" * 50)
    print("Phase 2: Fine-Tuning")
    print("전체 네트워크 파인튜닝 (낮은 학습률)")
    print("=" * 50)

    # base_model (레이어 인덱스 2)의 trainable 해제
    # 주의: model.layers 인덱스는 모델 구조에 따라 다를 수 있음
    model.layers[2].trainable = True  # base_model

    # Fine-Tuning 시 낮은 학습률 사용 (사전학습 가중치 보호)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        initial_epoch=args.epochs // 2,  # Phase 1 이어서 진행
        callbacks=callbacks
    )

    print(f"\n학습 완료! 최고 모델 저장 위치: {args.save_path}")


if __name__ == '__main__':
    main()
