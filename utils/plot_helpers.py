"""
공용 시각화 헬퍼 함수
모든 챕터 노트북에서 import하여 사용:
    import sys; sys.path.append('..')
    from utils.plot_helpers import plot_history, plot_confusion_matrix, plot_sample_images
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 한국어 폰트 설정 (macOS)
matplotlib.rcParams['font.family'] = ['AppleGothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_history(history, metrics=None, title="학습 곡선"):
    """
    model.fit() 반환값(history)으로 손실·지표 곡선을 그린다.

    Parameters
    ----------
    history : keras.callbacks.History
    metrics : list[str], optional
        추가로 그릴 지표 이름 (예: ['accuracy']). None이면 자동 감지.
    title : str
    """
    if metrics is None:
        # 'val_'로 시작하지 않는 키 중 'loss' 외 나머지를 지표로 사용
        all_keys = list(history.history.keys())
        metrics = [k for k in all_keys if not k.startswith("val_") and k != "loss"]

    n_plots = 1 + len(metrics)  # loss + 각 지표
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- 손실 곡선 ---
    ax = axes[0]
    ax.plot(history.history["loss"], label="훈련 손실")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="검증 손실", linestyle="--")
    ax.set_title("손실 (Loss)")
    ax.set_xlabel("에포크")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- 지표 곡선 ---
    for i, metric in enumerate(metrics):
        ax = axes[i + 1]
        ax.plot(history.history[metric], label=f"훈련 {metric}")
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label=f"검증 {metric}", linestyle="--")
        ax.set_title(metric)
        ax.set_xlabel("에포크")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", cmap="Blues"):
    """
    혼동 행렬(Confusion Matrix)을 히트맵으로 시각화한다.

    Parameters
    ----------
    cm : array-like, shape (n_classes, n_classes)
        sklearn.metrics.confusion_matrix() 결과
    class_names : list[str]
    title : str
    cmap : str
    """
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # 셀 내부 숫자 표시
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("실제 레이블")
    ax.set_xlabel("예측 레이블")
    plt.tight_layout()
    plt.show()


def plot_sample_images(images, labels=None, class_names=None, n_cols=8, title="샘플 이미지"):
    """
    이미지 배열을 격자로 시각화한다.

    Parameters
    ----------
    images : array-like, shape (N, H, W) or (N, H, W, C)
    labels : array-like, optional
        정수 레이블 배열
    class_names : list[str], optional
        레이블 인덱스 → 이름 매핑
    n_cols : int
        열 수
    title : str
    """
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = np.array(axes).reshape(-1)  # 1D로 평탄화

    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]
            # 픽셀값 범위 정규화 (0~1)
            if img.max() > 1.0:
                img = img / 255.0
            if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
                ax.imshow(img.squeeze(), cmap="gray")
            else:
                ax.imshow(img)
            if labels is not None:
                label = int(labels[i])
                name = class_names[label] if class_names else str(label)
                ax.set_title(name, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_activation_functions():
    """
    주요 활성화 함수 (ReLU, Sigmoid, Tanh, Softplus, ELU)를 한 그래프에 비교한다.
    """
    x = np.linspace(-5, 5, 300)

    activations = {
        "ReLU": np.maximum(0, x),
        "Sigmoid": 1 / (1 + np.exp(-x)),
        "Tanh": np.tanh(x),
        "ELU (α=1)": np.where(x >= 0, x, np.exp(x) - 1),
        "Softplus": np.log1p(np.exp(x)),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, y in activations.items():
        ax.plot(x, y, label=name, linewidth=2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 5)
    ax.set_title("활성화 함수 비교", fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
