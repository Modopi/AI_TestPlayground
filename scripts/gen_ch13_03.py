"""Generate Chapter 13-03: UNet for Diffusion."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helper import md, code, create_notebook

cells = [
# ─── Cell 0: 헤더 ───
md(r"""# Chapter 13: 생성 AI 심화 — 확산 모델용 UNet 아키텍처

## 학습 목표
- **Sinusoidal Time Embedding**의 수식과 구현을 이해하고 시간 정보가 모델에 주입되는 원리를 파악한다
- **잔차 블록(Residual Block)**에 시간 조건을 결합하는 방법을 구현한다
- **Cross-Attention** 메커니즘이 텍스트/클래스 조건 정보를 UNet에 전달하는 과정을 이해한다
- MNIST(28×28)용 **간단한 UNet 아키텍처**를 밑바닥부터 TensorFlow로 구현한다
- UNet의 **인코더-디코더 구조**와 **Skip Connection**이 노이즈 예측에 중요한 이유를 분석한다

## 목차
1. [수학적 기초: 시간 임베딩과 Cross-Attention](#1.-수학적-기초)
2. [Sinusoidal Time Embedding 구현](#2.-시간-임베딩)
3. [잔차 블록과 시간 조건 주입](#3.-잔차-블록)
4. [28×28 MNIST용 UNet 구현](#4.-UNet-구현)
5. [UNet 특징 맵 개념 시각화](#5.-특징-맵-시각화)
6. [정리](#6.-정리)"""),

# ─── Cell 1: 수학적 기초 ───
md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### Sinusoidal Time Embedding

Transformer의 위치 인코딩(Vaswani et al., 2017)을 시간 $t$에 적용합니다:

$$PE(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \quad PE(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

- $t$: 현재 확산 타임스텝 (정수)
- $d$: 임베딩 차원 (보통 128 또는 256)
- $i = 0, 1, \ldots, d/2 - 1$: 주파수 인덱스
- $10000^{2i/d}$: 주파수가 $i$에 따라 기하급수적으로 증가 → 다양한 스케일의 시간 정보 인코딩

**왜 사인/코사인인가?**

| 특성 | 설명 |
|------|------|
| 유일성 | 모든 $t$에 대해 고유한 벡터 생성 |
| 연속성 | 인접 $t$ 값의 임베딩이 유사 |
| 주기성 | 저차원: 빠른 진동 (미세 시간), 고차원: 느린 진동 (거시 시간) |
| 외삽 | 학습 시 보지 못한 $t$ 값에도 의미 있는 임베딩 제공 |

### Cross-Attention (조건 주입)

텍스트/클래스 등 외부 조건 $c$를 UNet 중간 특징에 주입합니다:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Self-Attention vs Cross-Attention:**

| 구분 | Q 출처 | K, V 출처 |
|------|--------|-----------|
| Self-Attention | 이미지 특징 $h$ | 이미지 특징 $h$ |
| Cross-Attention | 이미지 특징 $h$ | 조건 임베딩 $c$ (텍스트 등) |

$$Q = hW^Q, \quad K = cW^K, \quad V = cW^V$$

- $h \in \mathbb{R}^{HW \times d}$: UNet 중간층 특징 맵 (flatten)
- $c \in \mathbb{R}^{L \times d_c}$: 조건 시퀀스 (텍스트 토큰 임베딩 등)
- $W^Q \in \mathbb{R}^{d \times d_k}$, $W^K \in \mathbb{R}^{d_c \times d_k}$, $W^V \in \mathbb{R}^{d_c \times d_v}$

### DDPM UNet 전체 구조

```
입력 x_t (noisy image) + t (timestep)
    │
    ▼
[Sinusoidal Embedding(t)] ──────────────────┐
    │                                        │ (시간 조건)
    ▼                                        │
[Encoder Block 1] ─── skip ───────┐         │
    │ ↓ Downsample                │         │
[Encoder Block 2] ─── skip ──┐   │         │
    │ ↓ Downsample            │   │         │
[Bottleneck (Attention)]      │   │    t_emb│
    │ ↑ Upsample              │   │         │
[Decoder Block 2] ← concat ──┘   │         │
    │ ↑ Upsample                  │         │
[Decoder Block 1] ← concat ──────┘         │
    │                                        │
    ▼                                        │
[출력 Conv] → ε_θ(x_t, t) (예측 노이즈)    │
```

**요약 표:**

| 구성 요소 | 수식/역할 | MNIST UNet 설정 |
|-----------|-----------|----------------|
| Time Embedding | $\sin/\cos$ + MLP | $d=128$ |
| Residual Block | $h + F(h, t_{emb})$ | Conv3×3 + GroupNorm |
| Skip Connection | Encoder → Decoder concat | 채널 축 연결 |
| Cross-Attention | $\text{softmax}(QK^T/\sqrt{d})V$ | (옵션) 조건부 생성 시 |
| Down/Upsample | MaxPool / UpSampling2D | 2× 스케일 |"""),

# ─── Cell 2: 🐣 친절 설명 ───
md(r"""---

### 🐣 초등학생을 위한 UNet 친절 설명!

#### 🔢 UNet이 뭔가요?

> 💡 **비유**: UNet은 **모래시계** 모양의 신경망이에요!
> - **위쪽 (인코더)**: 그림을 점점 **작게 줄이면서** 핵심 정보를 추출해요 (압축)
> - **가운데 (병목)**: 가장 중요한 정보만 남아요
> - **아래쪽 (디코더)**: 다시 **크게 키우면서** 세부 사항을 복원해요
> - **건너뛰기 연결**: 위쪽에서 아래쪽으로 **지름길**을 만들어, 세부 정보가 사라지지 않게 해요!

#### ⏰ 시간 임베딩은 왜 필요한가요?

| 질문 | 답변 |
|------|------|
| 왜? | UNet이 "지금 $t=100$이야" vs "$t=900$이야"를 구분해야 해요 |
| 어떻게? | 시간 $t$를 사인/코사인 파동 패턴으로 바꿔서 모든 층에 알려줘요 |
| 비유 | 시험지에 **몇 교시인지** 적혀 있어야 어떤 과목인지 아는 것과 같아요! |"""),

# ─── Cell 3: 📝 연습 문제 ───
md(r"""---

### 📝 연습 문제

#### 문제 1: Sinusoidal Embedding 계산

$t=50$, $d=4$일 때, 시간 임베딩 벡터를 수동으로 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$d=4$이므로 $i = 0, 1$ (주파수 인덱스 2개)

$$PE(50, 0) = \sin\!\left(\frac{50}{10000^{0/4}}\right) = \sin(50) \approx -0.2624$$

$$PE(50, 1) = \cos\!\left(\frac{50}{10000^{0/4}}\right) = \cos(50) \approx 0.9649$$

$$PE(50, 2) = \sin\!\left(\frac{50}{10000^{2/4}}\right) = \sin(0.5) \approx 0.4794$$

$$PE(50, 3) = \cos\!\left(\frac{50}{10000^{2/4}}\right) = \cos(0.5) \approx 0.8776$$

임베딩 벡터: $[-0.2624,\, 0.9649,\, 0.4794,\, 0.8776]$
</details>

#### 문제 2: Skip Connection의 역할

UNet에서 Skip Connection을 제거하면 어떤 문제가 발생할까요?

<details>
<summary>💡 풀이 확인</summary>

Skip Connection 없이는:
1. 디코더가 인코더에서 손실된 **고해상도 세부 정보**(엣지, 텍스처)를 복원하기 어려움
2. 노이즈 예측 $\epsilon_\theta$의 **공간적 정밀도** 저하
3. 생성된 이미지가 **흐릿하고 디테일 부족**

Skip Connection은 인코더의 고해상도 특징을 디코더에 직접 전달하여, **위치 정밀도**를 유지합니다.
이는 원래 의료 영상 분할(Ronneberger et al., 2015)을 위해 설계된 UNet의 핵심 기여입니다.
</details>"""),

# ─── Cell 4: 임포트 ───
code("""# ── 라이브러리 임포트 ──────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")"""),

# ─── Cell 5: Section 2 - 시간 임베딩 ───
md(r"""## 2. Sinusoidal Time Embedding 구현 <a name='2.-시간-임베딩'></a>

각 확산 타임스텝 $t$를 고차원 벡터로 변환합니다. Transformer 위치 인코딩과 동일한 원리입니다."""),

# ─── Cell 6: 시간 임베딩 코드 ───
code(r"""# ── Sinusoidal Time Embedding 구현 ──────────────────────────────────
def sinusoidal_embedding(t, dim):
    # t: (batch,) 정수 텐서, dim: 임베딩 차원
    half_dim = dim // 2
    freqs = tf.exp(
        -tf.math.log(10000.0) * tf.range(half_dim, dtype=tf.float32) / half_dim
    )
    # t를 float로 변환하여 외적 계산
    t_float = tf.cast(t, tf.float32)
    args = t_float[:, None] * freqs[None, :]  # (batch, half_dim)
    embedding = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)  # (batch, dim)
    return embedding

# 테스트
test_t = tf.constant([0, 50, 100, 500, 999])
test_emb = sinusoidal_embedding(test_t, dim=128)
print(f"입력 타임스텝: {test_t.numpy()}")
print(f"임베딩 shape: {test_emb.shape}")
print(f"\nt=0   임베딩 (처음 8개): {test_emb[0, :8].numpy().round(4)}")
print(f"t=50  임베딩 (처음 8개): {test_emb[1, :8].numpy().round(4)}")
print(f"t=500 임베딩 (처음 8개): {test_emb[3, :8].numpy().round(4)}")
print(f"t=999 임베딩 (처음 8개): {test_emb[4, :8].numpy().round(4)}")

# 임베딩 간 코사인 유사도
def cosine_sim(a, b):
    return float(tf.reduce_sum(a * b) / (tf.norm(a) * tf.norm(b)))

print(f"\n코사인 유사도:")
print(f"  sim(t=0,   t=50)  = {cosine_sim(test_emb[0], test_emb[1]):.4f}")
print(f"  sim(t=50,  t=100) = {cosine_sim(test_emb[1], test_emb[2]):.4f}")
print(f"  sim(t=0,   t=999) = {cosine_sim(test_emb[0], test_emb[4]):.4f}")
print(f"  sim(t=500, t=999) = {cosine_sim(test_emb[3], test_emb[4]):.4f}")"""),

# ─── Cell 7: 시간 임베딩 시각화 ───
code(r"""# ── 시간 임베딩 히트맵 시각화 ──────────────────────────────────────
all_t = tf.range(0, 1000)
all_emb = sinusoidal_embedding(all_t, dim=128).numpy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) 히트맵: 전체 임베딩
ax1 = axes[0]
im = ax1.imshow(all_emb.T, aspect='auto', cmap='RdBu_r',
                extent=[0, 1000, 128, 0], vmin=-1, vmax=1)
ax1.set_xlabel('Timestep $t$', fontsize=11)
ax1.set_ylabel('Embedding Dimension', fontsize=11)
ax1.set_title('Sinusoidal Time Embedding (d=128)', fontweight='bold')
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

# (2) 개별 차원의 파동
ax2 = axes[1]
dims_to_show = [0, 1, 16, 32, 63]
t_range = np.arange(1000)
for d_idx in dims_to_show:
    ax2.plot(t_range, all_emb[:, d_idx], lw=1.5, label=f'dim {d_idx}', alpha=0.8)
ax2.set_xlabel('Timestep $t$', fontsize=11)
ax2.set_ylabel('Value', fontsize=11)
ax2.set_title('Individual Dimension Waveforms', fontweight='bold')
ax2.legend(fontsize=9, ncol=2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/time_embedding.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/time_embedding.png")
print(f"저차원(dim 0): 빠른 진동 → 미세 시간 구분")
print(f"고차원(dim 63): 느린 진동 → 거시 시간 구분")"""),

# ─── Cell 8: Section 3 - 잔차 블록 ───
md(r"""## 3. 잔차 블록과 시간 조건 주입 <a name='3.-잔차-블록'></a>

DDPM UNet의 기본 구성 단위인 **Residual Block**을 구현합니다. 시간 임베딩 $t_{emb}$가 블록 내부에 주입됩니다:

$$h' = \text{Conv}(\text{GN}(\text{SiLU}(h))) + W_t \cdot t_{emb}$$
$$\text{output} = h + \text{Conv}(\text{GN}(\text{SiLU}(h')))$$

- GN: Group Normalization (배치 크기에 독립적)
- SiLU: $\text{SiLU}(x) = x \cdot \sigma(x)$ (Swish 활성화)"""),

# ─── Cell 9: 잔차 블록 코드 ───
code(r"""# ── 잔차 블록 + 시간 조건 주입 구현 ─────────────────────────────────
class ResidualBlock(layers.Layer):
    # Conv → GroupNorm → SiLU → Conv, with time embedding injection
    def __init__(self, out_channels, n_groups=8, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.norm1 = layers.GroupNormalization(groups=n_groups)
        self.conv1 = layers.Conv2D(out_channels, 3, padding='same')
        self.time_proj = layers.Dense(out_channels)
        self.norm2 = layers.GroupNormalization(groups=n_groups)
        self.conv2 = layers.Conv2D(out_channels, 3, padding='same')
        self.skip_conv = None

    def build(self, input_shape):
        if input_shape[-1] != self.out_channels:
            self.skip_conv = layers.Conv2D(self.out_channels, 1)
        super().build(input_shape)

    def call(self, x, t_emb):
        residual = x

        h = self.norm1(x)
        h = tf.nn.silu(h)
        h = self.conv1(h)

        # 시간 임베딩 주입: (batch, dim) → (batch, 1, 1, channels)
        t_proj = tf.nn.silu(t_emb)
        t_proj = self.time_proj(t_proj)[:, None, None, :]
        h = h + t_proj

        h = self.norm2(h)
        h = tf.nn.silu(h)
        h = self.conv2(h)

        if self.skip_conv is not None:
            residual = self.skip_conv(residual)

        return h + residual

# 테스트
test_input = tf.random.normal([2, 28, 28, 1])
test_t_emb = sinusoidal_embedding(tf.constant([100, 500]), dim=128)

res_block = ResidualBlock(64)
output = res_block(test_input, test_t_emb)
print(f"ResidualBlock 테스트:")
print(f"  입력 shape:  {test_input.shape}")
print(f"  t_emb shape: {test_t_emb.shape}")
print(f"  출력 shape:  {output.shape}")
print(f"  파라미터 수: {sum(np.prod(v.shape) for v in res_block.trainable_variables):,}")
print(f"  출력 통계: 평균={float(tf.reduce_mean(output)):.4f}, 표준편차={float(tf.math.reduce_std(output)):.4f}")"""),

# ─── Cell 10: Section 4 - UNet 구현 ───
md(r"""## 4. 28×28 MNIST용 UNet 구현 <a name='4.-UNet-구현'></a>

MNIST(28×28×1)에 적합한 **소형 UNet**을 구현합니다. 구조:

| 단계 | 해상도 | 채널 수 |
|------|--------|---------|
| 입력 | 28×28 | 1 |
| Encoder 1 | 28×28 → 14×14 | 32 |
| Encoder 2 | 14×14 → 7×7 | 64 |
| Bottleneck | 7×7 | 128 |
| Decoder 2 | 7×7 → 14×14 | 64 |
| Decoder 1 | 14×14 → 28×28 | 32 |
| 출력 | 28×28 | 1 |"""),

# ─── Cell 11: UNet 코드 ───
code(r"""# ── MNIST 28x28 UNet 아키텍처 구현 ──────────────────────────────────
class SimpleUNet(keras.Model):
    # Encoder-Bottleneck-Decoder with skip connections and time conditioning
    def __init__(self, time_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.time_dim = time_dim

        # 시간 임베딩 MLP
        self.time_mlp = keras.Sequential([
            layers.Dense(time_dim, activation='swish'),
            layers.Dense(time_dim),
        ])

        # 입력 프로젝션
        self.input_conv = layers.Conv2D(32, 3, padding='same')

        # 인코더
        self.enc1 = ResidualBlock(32)
        self.down1 = layers.MaxPooling2D(2)   # 28→14
        self.enc2 = ResidualBlock(64)
        self.down2 = layers.MaxPooling2D(2)   # 14→7

        # 병목
        self.bottleneck1 = ResidualBlock(128)
        self.bottleneck2 = ResidualBlock(128)

        # 디코더
        self.up2 = layers.UpSampling2D(2)     # 7→14
        self.dec2 = ResidualBlock(64)
        self.up1 = layers.UpSampling2D(2)     # 14→28
        self.dec1 = ResidualBlock(32)

        # 출력
        self.output_norm = layers.GroupNormalization(groups=8)
        self.output_conv = layers.Conv2D(1, 1)

    def call(self, x, t):
        # 시간 임베딩
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # 입력
        h = self.input_conv(x)

        # 인코더
        h1 = self.enc1(h, t_emb)          # 28×28×32
        h = self.down1(h1)                  # 14×14×32
        h2 = self.enc2(h, t_emb)           # 14×14×64
        h = self.down2(h2)                  # 7×7×64

        # 병목
        h = self.bottleneck1(h, t_emb)     # 7×7×128
        h = self.bottleneck2(h, t_emb)     # 7×7×128

        # 디코더 + Skip Connection
        h = self.up2(h)                     # 14×14×128
        h = tf.concat([h, h2], axis=-1)    # 14×14×192
        h = self.dec2(h, t_emb)            # 14×14×64
        h = self.up1(h)                     # 28×28×64
        h = tf.concat([h, h1], axis=-1)    # 28×28×96
        h = self.dec1(h, t_emb)            # 28×28×32

        # 출력
        h = self.output_norm(h)
        h = tf.nn.silu(h)
        return self.output_conv(h)          # 28×28×1

# 모델 생성 및 테스트
model = SimpleUNet(time_dim=128)

# Forward pass 테스트
dummy_x = tf.random.normal([4, 28, 28, 1])
dummy_t = tf.constant([0, 250, 500, 999])
dummy_out = model(dummy_x, dummy_t)

print(f"SimpleUNet 아키텍처 테스트:")
print(f"  입력 x shape: {dummy_x.shape}")
print(f"  입력 t:       {dummy_t.numpy()}")
print(f"  출력 shape:   {dummy_out.shape}")
print(f"  총 파라미터 수: {sum(np.prod(v.shape) for v in model.trainable_variables):,}")
print(f"  출력 통계: 평균={float(tf.reduce_mean(dummy_out)):.4f}, 표준편차={float(tf.math.reduce_std(dummy_out)):.4f}")

# 레이어별 파라미터 수
print(f"\n레이어별 파라미터 수:")
layer_params = {}
for v in model.trainable_variables:
    layer_name = v.name.split('/')[0]
    cnt = int(np.prod(v.shape))
    layer_params[layer_name] = layer_params.get(layer_name, 0) + cnt

for name, cnt in sorted(layer_params.items()):
    print(f"  {name:<30s}: {cnt:>8,}")"""),

# ─── Cell 12: Section 5 - 특징 맵 시각화 ───
md(r"""## 5. UNet 특징 맵 개념 시각화 <a name='5.-특징-맵-시각화'></a>

UNet의 인코더가 서로 다른 해상도에서 어떤 특징을 추출하는지, 그리고 시간 임베딩에 따라 출력이 어떻게 변하는지를 시각화합니다."""),

# ─── Cell 13: 특징 맵 시각화 코드 ───
code(r"""# ── UNet 출력: 시간 조건에 따른 노이즈 예측 변화 ─────────────────
# 간단한 테스트 이미지 (대각선 패턴)
test_img = np.zeros((1, 28, 28, 1), dtype=np.float32)
for i in range(28):
    test_img[0, i, i, 0] = 1.0
    if i + 1 < 28:
        test_img[0, i, i+1, 0] = 0.5
    if i - 1 >= 0:
        test_img[0, i, i-1, 0] = 0.5

# 다양한 타임스텝에서의 UNet 출력 비교
timesteps_test = [0, 100, 300, 500, 700, 999]

fig, axes = plt.subplots(2, 6, figsize=(18, 6))

# 첫 번째 행: 입력 (시간에 따라 노이즈 추가된 이미지)
T_test = 1000
beta_test = np.linspace(1e-4, 0.02, T_test)
alpha_test = 1.0 - beta_test
abar_test = np.cumprod(alpha_test)

for idx, t in enumerate(timesteps_test):
    # Forward process로 노이즈 추가
    if t == 0:
        noisy = test_img.copy()
    else:
        abar = abar_test[t-1]
        eps = np.random.randn(*test_img.shape).astype(np.float32)
        noisy = np.sqrt(abar) * test_img + np.sqrt(1 - abar) * eps

    axes[0][idx].imshow(noisy[0, :, :, 0], cmap='gray', vmin=-2, vmax=2)
    axes[0][idx].set_title(f'입력: $t={t}$', fontsize=9, fontweight='bold')
    axes[0][idx].axis('off')

    # UNet 예측
    t_tensor = tf.constant([t])
    pred = model(tf.constant(noisy), t_tensor).numpy()
    axes[1][idx].imshow(pred[0, :, :, 0], cmap='RdBu_r')
    axes[1][idx].set_title(f'UNet 출력: $t={t}$', fontsize=9, fontweight='bold')
    axes[1][idx].axis('off')

axes[0][0].set_ylabel('노이즈 입력', fontsize=11)
axes[1][0].set_ylabel('예측 노이즈', fontsize=11)
plt.suptitle('UNet: 타임스텝에 따른 노이즈 예측 (초기화 직후, 학습 전)', fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/unet_feature_maps.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/unet_feature_maps.png")
print("참고: 학습 전이므로 UNet 출력은 무작위에 가깝습니다.")
print("학습 후에는 입력 노이즈를 정확하게 예측하게 됩니다.")

# UNet 출력 통계
print(f"\n타임스텝별 UNet 출력 통계 (학습 전):")
print(f"{'시점':>6} | {'평균':>10} | {'표준편차':>10} | {'최소':>10} | {'최대':>10}")
print("-" * 55)
for t in timesteps_test:
    if t == 0:
        noisy_t = test_img
    else:
        abar = abar_test[t-1]
        eps = np.random.randn(*test_img.shape).astype(np.float32)
        noisy_t = np.sqrt(abar) * test_img + np.sqrt(1 - abar) * eps
    pred = model(tf.constant(noisy_t), tf.constant([t])).numpy()
    print(f"  t={t:4d} | {pred.mean():>10.4f} | {pred.std():>10.4f} | {pred.min():>10.4f} | {pred.max():>10.4f}")"""),

# ─── Cell 14: 정리 ───
md(r"""## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| Sinusoidal Embedding | $PE(t,2i) = \sin(t/10000^{2i/d})$ — 시간 정보를 고차원 벡터로 변환 | ⭐⭐⭐ |
| Residual Block | $h + F(h, t_{emb})$ — 시간 조건이 주입된 잔차 학습 | ⭐⭐⭐ |
| Skip Connection | 인코더 → 디코더 직접 연결 — 고해상도 정보 보존 | ⭐⭐⭐ |
| Cross-Attention | $\text{softmax}(QK^T/\sqrt{d})V$ — 텍스트/클래스 조건 주입 | ⭐⭐⭐ |
| GroupNorm | 배치 크기 독립적 정규화 — 소배치에서도 안정적 | ⭐⭐ |
| UNet 구조 | 인코더-병목-디코더 with skip — 노이즈 예측의 표준 아키텍처 | ⭐⭐⭐ |

### 핵심 수식

$$PE(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \quad PE(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

$$\text{Cross-Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 다음 챕터 예고
**04_conditional_diffusion_cfg** — Classifier-Free Guidance(CFG)를 통해 클래스/텍스트 조건부 생성을 구현하고, 가이던스 스케일에 따른 품질-다양성 트레이드오프를 분석합니다."""),
]

if __name__ == '__main__':
    create_notebook(cells, 'chapter13_genai_diffusion/03_unet_for_diffusion.ipynb')
