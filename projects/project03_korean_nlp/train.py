"""한국어 텍스트 분류 BiLSTM 학습 스크립트

KoNLPy Okt 형태소 분석기로 한국어를 전처리하고
Bidirectional LSTM 모델로 텍스트를 분류합니다.

사전 요구사항:
    pip install konlpy
    Java 런타임 환경 (JRE 또는 JDK)

사용법:
    python train.py --data_dir ./data --num_classes 5
    python train.py --data_dir ./data --num_classes 3 --max_length 50
"""
import argparse
import os
import json
import pandas as pd
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
    parser = argparse.ArgumentParser(description='한국어 BiLSTM 텍스트 분류')
    parser.add_argument('--data_dir',    type=str, required=True,
                        help='학습 데이터 디렉토리 (train.csv, test.csv 포함)')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='분류 클래스 수')
    parser.add_argument('--vocab_size',  type=int, default=20000,
                        help='어휘사전 크기 (상위 N개 형태소 사용, 기본값: 20000)')
    parser.add_argument('--max_length',  type=int, default=100,
                        help='최대 시퀀스 길이 (형태소 단위, 기본값: 100)')
    parser.add_argument('--embed_dim',   type=int, default=128,
                        help='임베딩 벡터 차원 (기본값: 128)')
    parser.add_argument('--lstm_units',  type=int, default=64,
                        help='LSTM 단위 수 - 단방향 기준 (기본값: 64)')
    parser.add_argument('--dropout',     type=float, default=0.3,
                        help='드롭아웃 비율 (기본값: 0.3)')
    parser.add_argument('--epochs',      type=int, default=20,
                        help='학습 에포크 수 (기본값: 20)')
    parser.add_argument('--batch_size',  type=int, default=64,
                        help='배치 크기 (기본값: 64)')
    parser.add_argument('--save_dir',    type=str, default='model/',
                        help='모델 저장 디렉토리 (기본값: model/)')
    return parser.parse_args()


def tokenize_korean(texts, okt, norm=True, stem=False):
    """KoNLPy Okt로 한국어 텍스트를 형태소 단위로 분리한다.

    Args:
        texts: 텍스트 문자열 리스트
        okt: Okt 인스턴스
        norm: 정규화 여부 (오탈자 교정 등)
        stem: 어간 추출 여부 (동사/형용사 기본형으로 변환)

    Returns:
        형태소 리스트의 리스트
    """
    tokenized = []
    for i, text in enumerate(texts):
        # morphs: 형태소 단위 분리 (품사 정보 없음, 빠른 처리)
        morphs = okt.morphs(str(text), norm=norm, stem=stem)
        tokenized.append(morphs)
        # 진행 상황 출력 (1000개 단위)
        if (i + 1) % 1000 == 0:
            print(f"  토크나이징 진행: {i+1}/{len(texts)}")
    return tokenized


def build_vocabulary(tokenized_texts, vocab_size):
    """형태소 빈도 기반으로 어휘사전을 구축한다.

    Args:
        tokenized_texts: 형태소 리스트의 리스트
        vocab_size: 사전에 포함할 최대 어휘 수 (빈도 상위 N개)

    Returns:
        {형태소: 정수 인덱스} 딕셔너리
        (0: 패딩, 1: 미등록 토큰 <UNK>)
    """
    from collections import Counter

    # 전체 형태소 빈도 집계
    all_morphs = [morph for text in tokenized_texts for morph in text]
    counter    = Counter(all_morphs)

    # 빈도 상위 vocab_size개만 선택
    most_common = counter.most_common(vocab_size - 2)  # 0(PAD), 1(UNK) 예약

    # 인덱스 할당: 2부터 시작 (0=PAD, 1=UNK)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for morph, _ in most_common:
        vocab[morph] = len(vocab)

    print(f"어휘사전 크기: {len(vocab)} (요청: {vocab_size})")
    return vocab


def encode_and_pad(tokenized_texts, vocab, max_length):
    """형태소 시퀀스를 정수 인덱스로 변환하고 패딩을 적용한다.

    Args:
        tokenized_texts: 형태소 리스트의 리스트
        vocab: 어휘사전 딕셔너리
        max_length: 패딩/잘림 길이

    Returns:
        (N, max_length) 형태의 NumPy 배열
    """
    encoded = []
    for morphs in tokenized_texts:
        # 형태소를 정수로 변환 (미등록 형태소 → 1: UNK)
        seq = [vocab.get(m, 1) for m in morphs]
        # max_length 기준으로 잘림 또는 패딩 (후미 0으로 패딩)
        if len(seq) > max_length:
            seq = seq[:max_length]
        else:
            seq = seq + [0] * (max_length - len(seq))
        encoded.append(seq)
    return np.array(encoded, dtype=np.int32)


def build_bilstm_model(vocab_size, embed_dim, lstm_units, num_classes, dropout_rate):
    """BiLSTM 텍스트 분류 모델을 생성한다.

    Args:
        vocab_size: 어휘사전 크기 (임베딩 입력 차원)
        embed_dim: 임베딩 차원
        lstm_units: 단방향 LSTM 유닛 수 (BiLSTM 출력은 2배)
        num_classes: 분류 클래스 수
        dropout_rate: 드롭아웃 비율

    Returns:
        컴파일되지 않은 tf.keras.Model
    """
    model = tf.keras.Sequential([
        # 임베딩 레이어: 정수 시퀀스 → 밀집 벡터
        # mask_zero=True: 패딩(0) 위치는 연산에서 제외
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,
            name='embedding'
        ),
        # 드롭아웃: 임베딩 레이어 이후 정규화
        tf.keras.layers.Dropout(dropout_rate, name='embed_dropout'),

        # Bidirectional LSTM: 순방향 + 역방향으로 문맥 정보 포착
        # return_sequences=False: 마지막 타임스텝 출력만 반환
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, name='lstm'),
            name='bilstm'
        ),
        # BatchNormalization: 내부 공변량 이동 방지, 학습 안정화
        tf.keras.layers.BatchNormalization(name='batch_norm'),
        # 드롭아웃: 과적합 방지
        tf.keras.layers.Dropout(dropout_rate, name='lstm_dropout'),

        # Dense 은닉층 (선택적)
        tf.keras.layers.Dense(64, activation='relu', name='dense_hidden'),
        tf.keras.layers.Dropout(dropout_rate / 2, name='dense_dropout'),

        # 출력층: 클래스 수만큼 뉴런
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
    ], name='korean_bilstm')
    return model


def main():
    """한국어 BiLSTM 메인 학습 루프."""
    if not KONLPY_AVAILABLE:
        print("konlpy를 먼저 설치하세요: pip install konlpy")
        return

    args = parse_args()

    # 데이터 로드
    train_path = os.path.join(args.data_dir, 'train.csv')
    test_path  = os.path.join(args.data_dir, 'test.csv')

    print(f"데이터 로드 중...")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    print(f"학습: {len(train_df)}개 | 테스트: {len(test_df)}개")

    # 레이블을 정수 인덱스로 변환
    # 레이블이 문자열인 경우 숫자로 매핑
    label_names = sorted(train_df['label'].unique().tolist())
    label2idx   = {name: idx for idx, name in enumerate(label_names)}
    print(f"레이블 매핑: {label2idx}")

    train_labels = train_df['label'].map(label2idx).values
    test_labels  = test_df['label'].map(label2idx).values

    # KoNLPy Okt 형태소 분석기 초기화
    print("\nOkt 형태소 분석기 초기화 중...")
    okt = Okt()

    # 형태소 분석 (시간이 소요됨)
    print("학습 데이터 토크나이징...")
    train_tokenized = tokenize_korean(train_df['text'].tolist(), okt)
    print("테스트 데이터 토크나이징...")
    test_tokenized  = tokenize_korean(test_df['text'].tolist(), okt)

    # 어휘사전 구축 (학습 데이터 기준)
    print("\n어휘사전 구축 중...")
    vocab = build_vocabulary(train_tokenized, args.vocab_size)
    actual_vocab_size = len(vocab)

    # 어휘사전 저장 (추론 시 필요)
    os.makedirs(args.save_dir, exist_ok=True)
    vocab_path = os.path.join(args.save_dir, 'tokenizer.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'vocab': vocab,
            'label2idx': label2idx,
            'max_length': args.max_length
        }, f, ensure_ascii=False, indent=2)
    print(f"어휘사전 저장: {vocab_path}")

    # 정수 인코딩 + 패딩
    print("시퀀스 인코딩 및 패딩...")
    X_train = encode_and_pad(train_tokenized, vocab, args.max_length)
    X_test  = encode_and_pad(test_tokenized,  vocab, args.max_length)
    print(f"학습 데이터 형상: {X_train.shape}")

    # BiLSTM 모델 구성
    print("\nBiLSTM 모델 구성 중...")
    model = build_bilstm_model(
        vocab_size=actual_vocab_size,
        embed_dim=args.embed_dim,
        lstm_units=args.lstm_units,
        num_classes=args.num_classes,
        dropout_rate=args.dropout
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 콜백 설정
    model_save_path = os.path.join(args.save_dir, 'bilstm.keras')
    callbacks = [
        # 검증 정확도 최고 모델 저장
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),
        # 학습률 감소: 검증 손실 개선 없으면 0.5배 감소
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # 조기 종료
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        # TensorBoard 로깅
        tf.keras.callbacks.TensorBoard(log_dir='logs/bilstm'),
    ]

    # 모델 학습
    print("\n학습 시작...")
    history = model.fit(
        X_train, train_labels,
        validation_data=(X_test, test_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 최종 평가
    test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=0)
    print(f"\n최종 테스트 정확도: {test_acc:.4f}")
    print(f"모델 저장 완료: {model_save_path}")


if __name__ == '__main__':
    main()
