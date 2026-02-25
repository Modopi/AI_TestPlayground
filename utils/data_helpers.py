"""
공용 데이터 로딩·전처리 헬퍼 함수
모든 챕터 노트북에서 import하여 사용:
    import sys; sys.path.append('..')
    from utils.data_helpers import load_mnist, load_fashion_mnist, load_cifar10, normalize_images
"""

import numpy as np
import tensorflow as tf


# ─────────────────────────────────────────────
# 데이터셋 로딩
# ─────────────────────────────────────────────

def load_mnist(normalize=True, flatten=False):
    """
    MNIST 손글씨 숫자 데이터셋 로드.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        x: shape (N, 28, 28) float32  [normalize=True 시 0~1]
              shape (N, 784)  float32  [flatten=True 시]
        y: shape (N,) int32
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if normalize:
        x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    if flatten:
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)
    print(f"[MNIST] 훈련: {x_train.shape}, 테스트: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def load_fashion_mnist(normalize=True, flatten=False):
    """
    Fashion MNIST 데이터셋 로드.

    Returns
    -------
    (x_train, y_train), (x_test, y_test), class_names
        class_names: list[str], 10개 클래스 이름
    """
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    if normalize:
        x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    if flatten:
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)
    print(f"[Fashion MNIST] 훈련: {x_train.shape}, 테스트: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test), class_names


def load_cifar10(normalize=True):
    """
    CIFAR-10 컬러 이미지 데이터셋 로드.

    Returns
    -------
    (x_train, y_train), (x_test, y_test), class_names
        x: shape (N, 32, 32, 3) float32
        y: shape (N, 1) uint8  → .squeeze()로 (N,) 변환 가능
    """
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if normalize:
        x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    print(f"[CIFAR-10] 훈련: {x_train.shape}, 테스트: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test), class_names


# ─────────────────────────────────────────────
# 전처리 유틸리티
# ─────────────────────────────────────────────

def normalize_images(images, mode="minmax"):
    """
    이미지 배열을 정규화한다.

    Parameters
    ----------
    images : np.ndarray
    mode : str
        'minmax' → [0, 1]
        'zscore' → 평균 0, 표준편차 1
        '255'    → /255.0

    Returns
    -------
    np.ndarray, float32
    """
    images = images.astype("float32")
    if mode == "minmax":
        mn, mx = images.min(), images.max()
        return (images - mn) / (mx - mn + 1e-8)
    elif mode == "zscore":
        mu, sigma = images.mean(), images.std()
        return (images - mu) / (sigma + 1e-8)
    elif mode == "255":
        return images / 255.0
    else:
        raise ValueError(f"알 수 없는 mode: {mode!r}. 'minmax', 'zscore', '255' 중 선택.")


def train_val_split(x, y, val_ratio=0.2, seed=42):
    """
    (x, y) 배열을 훈련/검증 세트로 분리한다.

    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray
    val_ratio : float
    seed : int

    Returns
    -------
    (x_train, y_train), (x_val, y_val)
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    indices = rng.permutation(n)
    val_size = int(n * val_ratio)
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx])
