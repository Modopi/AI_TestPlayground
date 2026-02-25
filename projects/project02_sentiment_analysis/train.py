"""감성 분석 BERT 파인튜닝 학습 스크립트

BERT 모델을 사용자 정의 감성 분석 데이터셋에 파인튜닝합니다.
Hugging Face transformers 라이브러리와 TensorFlow를 함께 사용합니다.

사전 요구사항:
    pip install transformers datasets

사용법:
    python train.py --data_dir ./data --model_name bert-base-uncased
    python train.py --data_dir ./data --model_name klue/bert-base --num_labels 2
"""
import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf

# Hugging Face 라이브러리 임포트
try:
    from transformers import (
        BertTokenizer,
        TFBertForSequenceClassification,
        create_optimizer,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("경고: transformers 라이브러리가 설치되지 않았습니다.")
    print("설치 명령어: pip install transformers")


def parse_args():
    """명령줄 인수를 파싱하여 반환한다."""
    parser = argparse.ArgumentParser(description='BERT 감성 분석 파인튜닝')
    parser.add_argument('--data_dir',      type=str, required=True,
                        help='학습 데이터 디렉토리 (train.csv, test.csv 포함)')
    parser.add_argument('--model_name',    type=str, default='bert-base-uncased',
                        help='Hugging Face 모델 이름 (기본값: bert-base-uncased)')
    parser.add_argument('--num_labels',    type=int, default=2,
                        help='분류 클래스 수 (기본값: 2 - 긍정/부정)')
    parser.add_argument('--max_length',    type=int, default=128,
                        help='최대 입력 토큰 길이 (기본값: 128)')
    parser.add_argument('--epochs',        type=int, default=3,
                        help='학습 에포크 수 (기본값: 3)')
    parser.add_argument('--batch_size',    type=int, default=16,
                        help='배치 크기 (기본값: 16)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='초기 학습률 (기본값: 2e-5, BERT 파인튜닝 권장)')
    parser.add_argument('--warmup_ratio',  type=float, default=0.1,
                        help='워밍업 스텝 비율 (기본값: 0.1)')
    parser.add_argument('--save_path',     type=str, default='saved_model/',
                        help='모델 저장 경로 (기본값: saved_model/)')
    return parser.parse_args()


def load_csv_data(data_dir):
    """CSV 파일에서 텍스트와 레이블을 로드한다.

    Args:
        data_dir: train.csv와 test.csv가 있는 디렉토리

    Returns:
        (train_texts, train_labels, test_texts, test_labels) 튜플
    """
    train_path = os.path.join(data_dir, 'train.csv')
    test_path  = os.path.join(data_dir, 'test.csv')

    print(f"학습 데이터 로드: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"학습 샘플 수: {len(train_df)}")

    print(f"테스트 데이터 로드: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"테스트 샘플 수: {len(test_df)}")

    # 레이블 분포 출력
    print(f"\n레이블 분포 (학습):\n{train_df['label'].value_counts().to_string()}")

    return (
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        test_df['text'].tolist(),
        test_df['label'].tolist()
    )


def tokenize_texts(tokenizer, texts, max_length):
    """텍스트 리스트를 BERT 입력 형식으로 토크나이징한다.

    Args:
        tokenizer: BertTokenizer 인스턴스
        texts: 텍스트 문자열 리스트
        max_length: 최대 토큰 길이 (초과 시 잘림, 부족 시 패딩)

    Returns:
        {'input_ids': ..., 'attention_mask': ..., 'token_type_ids': ...} 딕셔너리
    """
    return tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',    # 최대 길이까지 패딩
        truncation=True,         # 최대 길이 초과 시 잘림
        return_tensors='tf'      # TensorFlow 텐서로 반환
    )


def build_tf_dataset(encodings, labels, batch_size, shuffle=False):
    """토크나이징된 데이터를 tf.data.Dataset으로 변환한다.

    Args:
        encodings: tokenizer 출력 딕셔너리
        labels: 정수 레이블 리스트
        batch_size: 배치 크기
        shuffle: 셔플 여부 (학습 데이터에만 True)

    Returns:
        배치 처리된 tf.data.Dataset
    """
    # 입력 딕셔너리와 레이블을 묶어 데이터셋 생성
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids':      encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'token_type_ids': encodings['token_type_ids'],
        },
        labels
    ))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels))

    # 프리페치로 데이터 로딩 병목 최소화
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    """BERT 파인튜닝 메인 학습 루프."""
    if not HF_AVAILABLE:
        print("transformers 라이브러리를 먼저 설치하세요: pip install transformers")
        return

    args = parse_args()

    print(f"사용 모델: {args.model_name}")
    print(f"클래스 수: {args.num_labels}")
    print(f"최대 토큰 길이: {args.max_length}")

    # 데이터 로드
    train_texts, train_labels, test_texts, test_labels = load_csv_data(args.data_dir)

    # BERT 토크나이저 로드 (첫 실행 시 자동 다운로드)
    print(f"\n토크나이저 로드 중: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # 텍스트 토크나이징
    print("텍스트 토크나이징 중...")
    train_encodings = tokenize_texts(tokenizer, train_texts, args.max_length)
    test_encodings  = tokenize_texts(tokenizer, test_texts,  args.max_length)

    # tf.data 데이터셋 생성
    train_dataset = build_tf_dataset(train_encodings, train_labels,
                                     args.batch_size, shuffle=True)
    test_dataset  = build_tf_dataset(test_encodings,  test_labels,
                                     args.batch_size, shuffle=False)

    # 학습 스텝 수 계산 (워밍업 스케줄러 설정용)
    num_train_steps = len(train_texts) // args.batch_size * args.epochs
    num_warmup_steps = int(num_train_steps * args.warmup_ratio)
    print(f"\n전체 학습 스텝: {num_train_steps}")
    print(f"워밍업 스텝: {num_warmup_steps}")

    # BERT 분류 모델 로드 (첫 실행 시 자동 다운로드)
    print(f"\nBERT 모델 로드 중: {args.model_name}")
    model = TFBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels
    )

    # AdamW 옵티마이저 + 선형 학습률 스케줄 (워밍업 포함)
    # BERT 파인튜닝 표준 설정: 선형 감쇠 + 워밍업
    optimizer, lr_schedule = create_optimizer(
        init_lr=args.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        weight_decay_rate=0.01    # L2 정규화 (가중치 감쇠)
    )

    # 모델 컴파일
    # from_logits=True: BERT 출력은 logit (softmax 미적용)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # 콜백 설정
    os.makedirs(args.save_path, exist_ok=True)
    callbacks = [
        # 검증 손실 기준 최고 모델 저장
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.save_path, 'best_model'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        # 조기 종료 (BERT는 과적합에 민감)
        tf.keras.callbacks.EarlyStopping(
            patience=2,
            restore_best_weights=True,
            monitor='val_loss'
        ),
        # TensorBoard 학습 로그
        tf.keras.callbacks.TensorBoard(log_dir='logs/bert_finetune'),
    ]

    # 모델 학습
    print("\n파인튜닝 시작...")
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # 최종 평가
    print("\n최종 테스트 평가...")
    test_results = model.evaluate(test_dataset, verbose=1)
    print(f"테스트 손실: {test_results[0]:.4f}")
    print(f"테스트 정확도: {test_results[1]:.4f}")

    # 모델 및 토크나이저 저장
    final_save_path = os.path.join(args.save_path, 'final_model')
    model.save_pretrained(final_save_path)       # BERT 가중치 저장
    tokenizer.save_pretrained(final_save_path)   # 토크나이저 저장
    print(f"\n모델 저장 완료: {final_save_path}")


if __name__ == '__main__':
    main()
