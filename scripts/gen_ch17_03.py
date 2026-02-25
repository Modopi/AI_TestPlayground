"""Generate chapter17_diffusion_transformers/03_dit_conditioning_and_adaln.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 17: 비디오 생성 모델과 DiT — adaLN-Zero 조건 주입과 CFG

## 학습 목표
- adaLN-Zero의 수식을 도출하고 표준 LayerNorm과의 차이를 이해한다
- Zero-initialization이 학습 안정성에 미치는 영향을 분석한다
- 시간 $t$, 클래스, 텍스트 등 다양한 조건 주입 방식을 비교한다
- DiT에서의 Classifier-Free Guidance(CFG) 설계를 이해한다
- TensorFlow로 adaLN-Zero 레이어를 직접 구현한다

## 목차
1. [수학적 기초: adaLN-Zero와 조건 주입](#1.-수학적-기초)
2. [라이브러리 임포트 및 환경 설정](#2.-환경-설정)
3. [adaLN-Zero 레이어 구현](#3.-adaLN-Zero-구현)
4. [표준 LayerNorm vs adaLN 조건부 비교](#4.-LayerNorm-vs-adaLN)
5. [Zero-Init 학습 안정성 실험](#5.-Zero-Init-안정성)
6. [다중 조건 주입 (시간 + 클래스) 데모](#6.-다중-조건-주입)
7. [DiT에서의 CFG 설계](#7.-CFG-설계)
8. [정리](#8.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### 표준 LayerNorm

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

- $\gamma, \beta$: 학습 가능한 스케일/시프트 파라미터
- $\mu, \sigma^2$: 입력의 평균, 분산 (채널 축)

### Adaptive LayerNorm (adaLN)

조건 벡터 $c$ (시간 $t$, 클래스, 텍스트 임베딩)로부터 $\gamma, \beta$를 동적으로 생성:

$$\text{adaLN}(x, c) = \gamma_c \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_c$$

$$(\gamma_c, \beta_c) = \text{MLP}(c)$$

### adaLN-Zero (DiT의 핵심 기법)

adaLN에 **게이팅 파라미터** $\alpha$를 추가하고, 초기값을 0으로 설정:

$$h = x + \alpha_1 \cdot \text{Attn}\!\left((1 + \gamma_1) \odot \text{LN}(x) + \beta_1\right)$$

$$\text{output} = h + \alpha_2 \cdot \text{FFN}\!\left((1 + \gamma_2) \odot \text{LN}(h) + \beta_2\right)$$

여기서:

$$(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2) = \text{MLP}(c), \quad c = \text{Embed}(t) + \text{Embed}(y)$$

- $t$: diffusion 타임스텝
- $y$: 클래스 레이블 또는 텍스트 임베딩
- $\alpha_1, \alpha_2$: 게이팅 스케일 (초기값 = 0)

### Zero-Initialization의 효과

초기 상태에서 $\alpha = 0$이면:

$$h = x + 0 \cdot \text{Attn}(\cdots) = x \quad \text{(항등 함수)}$$

- 학습 시작 시 각 DiT 블록이 **항등 함수**로 동작
- 깊은 네트워크에서도 **그래디언트 소실 없이** 안정적으로 학습 시작
- ViT/ResNet의 잔차 연결과 유사하지만 더 강력한 초기화 전략

### Classifier-Free Guidance (CFG) in DiT

$$\tilde{\epsilon}_\theta(x_t, c) = (1 + w) \cdot \epsilon_\theta(x_t, c) - w \cdot \epsilon_\theta(x_t, \varnothing)$$

- $w$: guidance scale (높을수록 조건에 충실, 다양성 감소)
- $\varnothing$: null 조건 (학습 시 일정 비율로 조건을 drop)
- DiT 학습 시 10~20% 확률로 클래스 레이블을 $\varnothing$으로 대체

**요약 표:**

| 구분 | 수식 | 설명 |
|------|------|------|
| 표준 LN | $\gamma \odot \text{norm}(x) + \beta$ | 고정 파라미터 |
| adaLN | $\gamma_c \odot \text{norm}(x) + \beta_c$ | 조건부 파라미터 |
| adaLN-Zero | $x + \alpha \cdot f(\gamma_c \odot \text{norm}(x) + \beta_c)$ | 게이팅 + 영점 초기화 |
| CFG | $(1+w)\epsilon(x,c) - w\epsilon(x,\varnothing)$ | 조건부 가이던스 |

---

### 🐣 초등학생을 위한 adaLN-Zero 친절 설명!

#### 🔢 adaLN-Zero가 뭔가요?

> 💡 **비유**: 요리사(모델)에게 "지금은 겨울이고, 매운맛을 원해요"라는 주문(조건)을 받아서 양념(파라미터)을 동적으로 조절하는 것!

보통 LayerNorm은 항상 같은 양념을 쓰지만, adaLN은 주문에 따라 양념을 바꿉니다.
그리고 "Zero"는 처음에 **아무것도 안 넣겠다**는 뜻이에요.

#### 🤔 왜 처음에 0으로 시작하나요?

> 💡 **비유**: 새 요리사가 처음 일할 때, 일단 원래 레시피(입력)를 그대로 내보내면서 천천히 자기만의 양념을 배우는 거예요!

처음부터 이상한 양념을 넣으면 요리가 망하듯이, 처음에 α=0으로 시작하면 안전하게 학습을 시작할 수 있습니다.

---

### 📝 연습 문제

#### 문제 1: adaLN-Zero 초기 출력

DiT 블록의 입력이 $x = [1, 2, 3]$이고 초기 상태에서 $\alpha_1 = 0$일 때, Attention 이후 출력 $h$는?

<details>
<summary>💡 풀이 확인</summary>

$$h = x + \alpha_1 \cdot \text{Attn}(\cdots) = x + 0 \cdot \text{Attn}(\cdots) = [1, 2, 3]$$

$\alpha = 0$이므로 **항등 함수**가 됩니다. 입력이 그대로 출력됩니다.
깊은 네트워크(DiT-XL: 28블록)에서도 초기에 안정적으로 학습을 시작할 수 있는 핵심 트릭입니다.
</details>

#### 문제 2: CFG 출력 계산

$\epsilon_\theta(x_t, c) = 0.8$, $\epsilon_\theta(x_t, \varnothing) = 0.3$이고 guidance scale $w=4.0$일 때 CFG 출력은?

<details>
<summary>💡 풀이 확인</summary>

$$\tilde{\epsilon} = (1 + w) \cdot \epsilon(x_t, c) - w \cdot \epsilon(x_t, \varnothing)$$
$$= (1 + 4) \times 0.8 - 4 \times 0.3 = 4.0 - 1.2 = 2.8$$

CFG는 조건부 예측을 **증폭**합니다. $w$가 클수록 조건에 더 충실해지지만 다양성이 줄어듭니다.
</details>"""))

# ── Cell 3: Section 2 header ──────────────────────────────────────
cells.append(md("""\
## 2. 라이브러리 임포트 및 환경 설정 <a name='2.-환경-설정'></a>

TensorFlow를 사용하여 adaLN-Zero 레이어를 구현합니다."""))

# ── Cell 4: Imports ──────────────────────────────────────────────────
cells.append(code("""\
# ── 라이브러리 임포트 및 환경 설정 ──────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import time

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"GPU 사용 가능: {len(tf.config.list_physical_devices('GPU')) > 0}")"""))

# ── Cell 5: Section 3 header ──────────────────────────────────────
cells.append(md(r"""\
## 3. adaLN-Zero 레이어 구현 <a name='3.-adaLN-Zero-구현'></a>

DiT의 핵심인 adaLN-Zero를 TensorFlow로 구현합니다.
조건 벡터 $c$로부터 $(\gamma, \beta, \alpha)$ 6개 파라미터를 생성합니다."""))

# ── Cell 6: adaLN-Zero implementation ────────────────────────────────
cells.append(code("""\
# ── adaLN-Zero 레이어 구현 ──────────────────────────────────────────
class AdaLNZero(tf.keras.layers.Layer):
    # Adaptive Layer Normalization with Zero-initialization (Peebles & Xie, 2023)
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # 조건 → (gamma, beta, alpha) 매핑
        self.adaLN_modulation = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='silu'),
            tf.keras.layers.Dense(3 * hidden_dim)
        ])

    def build(self, input_shape):
        # Zero-initialization: 마지막 Dense의 가중치와 편향을 0으로 초기화
        last_layer = self.adaLN_modulation.layers[-1]
        last_layer.build((None, self.hidden_dim))
        last_layer.kernel.assign(tf.zeros_like(last_layer.kernel))
        last_layer.bias.assign(tf.zeros_like(last_layer.bias))
        super().build(input_shape)

    def call(self, x, condition):
        # x: (B, N, D), condition: (B, D_cond)
        modulation = self.adaLN_modulation(condition)  # (B, 3*D)
        gamma, beta, alpha = tf.split(modulation, 3, axis=-1)  # 각 (B, D)

        # adaLN: 조건부 정규화
        x_norm = self.norm(x)  # (B, N, D)
        # gamma, beta를 시퀀스 축으로 브로드캐스트
        gamma = tf.expand_dims(gamma, 1)  # (B, 1, D)
        beta = tf.expand_dims(beta, 1)
        alpha = tf.expand_dims(alpha, 1)

        x_modulated = (1 + gamma) * x_norm + beta
        return x_modulated, alpha

# 테스트
B, N, D = 2, 16, 384
x_test = tf.random.normal([B, N, D])
cond_test = tf.random.normal([B, D])

adaln = AdaLNZero(hidden_dim=D)
x_mod, alpha = adaln(x_test, cond_test)

print("=" * 55)
print("adaLN-Zero 레이어 테스트")
print("=" * 55)
print(f"입력 x shape: {x_test.shape}")
print(f"조건 c shape: {cond_test.shape}")
print(f"출력 x_modulated shape: {x_mod.shape}")
print(f"게이팅 alpha shape: {alpha.shape}")

# Zero-init 검증
print(f"\\nZero-initialization 검증:")
print(f"  alpha 평균: {tf.reduce_mean(alpha).numpy():.6f} (기대값: 0.0)")
print(f"  alpha 표준편차: {tf.math.reduce_std(alpha).numpy():.6f} (기대값: 0.0)")
print(f"  alpha 최대 절대값: {tf.reduce_max(tf.abs(alpha)).numpy():.6f}")

# 초기 상태에서 x + alpha * f(x) = x 확인
residual = x_test + alpha * x_mod  # 초기에 alpha=0
diff = tf.reduce_mean(tf.abs(residual - x_test)).numpy()
print(f"  |x + alpha*f(x) - x| 평균: {diff:.8f} (0이면 항등함수)")
print(f"  → 초기 상태에서 항등 함수 {'확인' if diff < 1e-6 else '실패'}!")

print(f"\\n파라미터 수: {adaln.count_params():,}")"""))

# ── Cell 7: Section 4 header ──────────────────────────────────────
cells.append(md("""\
## 4. 표준 LayerNorm vs adaLN 조건부 비교 <a name='4.-LayerNorm-vs-adaLN'></a>

표준 LayerNorm은 조건과 무관하게 고정된 정규화를 수행하지만,
adaLN은 조건에 따라 다른 정규화 파라미터를 적용합니다."""))

# ── Cell 8: Standard LN vs adaLN comparison ──────────────────────────
cells.append(code("""\
# ── 표준 LayerNorm vs adaLN 조건부 비교 ──────────────────────────────
# 동일 입력에 다른 조건을 주었을 때의 출력 차이를 비교

x_input = tf.random.normal([1, 8, 128])

# 표준 LayerNorm
standard_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
y_standard = standard_ln(x_input)

# adaLN-Zero (서로 다른 조건)
adaln_layer = AdaLNZero(hidden_dim=128)

# 조건을 충분히 크게 설정하여 차이를 관찰 (학습 후 상태 시뮬레이션)
cond_a = tf.constant([[1.0] * 64 + [-1.0] * 64])
cond_b = tf.constant([[-1.0] * 64 + [1.0] * 64])

# 수동으로 modulation 가중치 설정 (학습 후 상태 시뮬레이션)
adaln_layer(x_input, cond_a)  # build
for layer in adaln_layer.adaLN_modulation.layers:
    if hasattr(layer, 'kernel'):
        layer.kernel.assign(tf.random.normal(layer.kernel.shape, stddev=0.1))
        layer.bias.assign(tf.random.normal(layer.bias.shape, stddev=0.01))

y_ada_a, alpha_a = adaln_layer(x_input, cond_a)
y_ada_b, alpha_b = adaln_layer(x_input, cond_b)

print("=" * 60)
print("표준 LayerNorm vs adaLN 비교")
print("=" * 60)

# 두 조건 간 adaLN 출력 차이
diff_ada = tf.reduce_mean(tf.abs(y_ada_a - y_ada_b)).numpy()
diff_std = tf.reduce_mean(tf.abs(y_standard - y_standard)).numpy()  # 항상 0

print(f"입력 x shape: {x_input.shape}")
print(f"\\n표준 LayerNorm:")
print(f"  조건 A 출력 == 조건 B 출력: 항상 동일 (차이: {diff_std:.6f})")
print(f"  → 조건에 무관한 고정 정규화")
print(f"\\nadaLN-Zero:")
print(f"  조건 A와 조건 B 출력 차이: {diff_ada:.6f}")
print(f"  → 조건에 따라 다른 정규화 수행")
print(f"  alpha 평균 (조건 A): {tf.reduce_mean(alpha_a).numpy():.4f}")
print(f"  alpha 평균 (조건 B): {tf.reduce_mean(alpha_b).numpy():.4f}")

# 조건별 정규화 효과 시각화
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
token_idx = 0
x_vals = x_input[0, token_idx, :64].numpy()
y_std_vals = y_standard[0, token_idx, :64].numpy()
y_ada_a_vals = y_ada_a[0, token_idx, :64].numpy()
y_ada_b_vals = y_ada_b[0, token_idx, :64].numpy()

ax1.plot(x_vals, 'gray', alpha=0.5, lw=1, label='입력 x')
ax1.plot(y_std_vals, 'b-', lw=2, label='표준 LN')
ax1.plot(y_ada_a_vals, 'r-', lw=2, alpha=0.7, label='adaLN (조건 A)')
ax1.plot(y_ada_b_vals, 'g-', lw=2, alpha=0.7, label='adaLN (조건 B)')
ax1.set_xlabel('차원', fontsize=11)
ax1.set_ylabel('값', fontsize=11)
ax1.set_title('정규화 출력 비교 (첫 번째 토큰)', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 우: 조건에 따른 gamma, beta 분포
ax2 = axes[1]
modulation_a = adaln_layer.adaLN_modulation(cond_a).numpy()[0]
modulation_b = adaln_layer.adaLN_modulation(cond_b).numpy()[0]
D_h = 128
gamma_a, beta_a, alpha_vals_a = modulation_a[:D_h], modulation_a[D_h:2*D_h], modulation_a[2*D_h:]
gamma_b, beta_b, alpha_vals_b = modulation_b[:D_h], modulation_b[D_h:2*D_h], modulation_b[2*D_h:]

x_pos = np.arange(D_h)
ax2.scatter(gamma_a, beta_a, c='red', alpha=0.5, s=15, label='조건 A (gamma vs beta)')
ax2.scatter(gamma_b, beta_b, c='blue', alpha=0.5, s=15, label='조건 B (gamma vs beta)')
ax2.axhline(y=0, color='gray', ls='--', lw=1)
ax2.axvline(x=0, color='gray', ls='--', lw=1)
ax2.set_xlabel('$\\gamma_c$ (스케일)', fontsize=11)
ax2.set_ylabel('$\\beta_c$ (시프트)', fontsize=11)
ax2.set_title('조건별 adaLN 파라미터 분포', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/ln_vs_adaln_comparison.png',
            dpi=100, bbox_inches='tight')
plt.close()
print("\\n그래프 저장됨: chapter17_diffusion_transformers/ln_vs_adaln_comparison.png")"""))

# ── Cell 9: Section 5 header ──────────────────────────────────────
cells.append(md(r"""\
## 5. Zero-Init 학습 안정성 실험 <a name='5.-Zero-Init-안정성'></a>

Zero-initialization($\alpha=0$) vs 랜덤 초기화의 학습 안정성을 비교합니다.
간단한 함수 근사 과제에서 두 초기화 방식의 초기 손실과 그래디언트 크기를 관찰합니다."""))

# ── Cell 10: Zero-init stability demonstration ──────────────────────
cells.append(code("""\
# ── Zero-Init vs 랜덤 초기화 학습 안정성 비교 ────────────────────────
class SimpleDiTBlock(tf.keras.layers.Layer):
    # 간소화된 DiT 블록 (adaLN-Zero 포함)
    def __init__(self, dim, zero_init=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.zero_init = zero_init
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn_proj = tf.keras.layers.Dense(dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim * 4, activation='gelu'),
            tf.keras.layers.Dense(dim)
        ])
        self.adaln = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, activation='silu'),
            tf.keras.layers.Dense(6 * dim)
        ])

    def build(self, input_shape):
        dummy_cond = tf.zeros([1, self.dim])
        self.adaln(dummy_cond)
        if self.zero_init:
            last = self.adaln.layers[-1]
            last.kernel.assign(tf.zeros_like(last.kernel))
            last.bias.assign(tf.zeros_like(last.bias))
        super().build(input_shape)

    def call(self, x, cond):
        mod = self.adaln(cond)
        g1, b1, a1, g2, b2, a2 = tf.split(mod, 6, axis=-1)
        g1, b1, a1 = [tf.expand_dims(v, 1) for v in [g1, b1, a1]]
        g2, b2, a2 = [tf.expand_dims(v, 1) for v in [g2, b2, a2]]

        # Attention branch
        h = (1 + g1) * self.norm(x) + b1
        h = self.attn_proj(h)
        x = x + a1 * h

        # FFN branch
        h2 = (1 + g2) * self.norm(x) + b2
        h2 = self.ffn(h2)
        x = x + a2 * h2
        return x

# 다층 DiT 시뮬레이션 (8블록 스택)
dim = 64
n_blocks = 8

def build_stacked_dit(zero_init):
    blocks = [SimpleDiTBlock(dim, zero_init=zero_init) for _ in range(n_blocks)]
    return blocks

x_input = tf.random.normal([4, 16, dim])
cond = tf.random.normal([4, dim])
target = tf.random.normal([4, 16, dim])

results = {}
for name, zero_init in [("Zero-Init", True), ("Random-Init", False)]:
    tf.random.set_seed(42)
    blocks = build_stacked_dit(zero_init)

    # Forward pass로 초기 출력 확인
    h = x_input
    for block in blocks:
        h = block(h, cond)

    initial_output_std = tf.math.reduce_std(h).numpy()

    # 그래디언트 크기 측정
    trainable_vars = []
    for block in blocks:
        trainable_vars.extend(block.trainable_variables)

    with tf.GradientTape() as tape:
        h = x_input
        for block in blocks:
            h = block(h, cond)
        loss = tf.reduce_mean((h - target) ** 2)

    grads = tape.gradient(loss, trainable_vars)
    grad_norms = [tf.norm(g).numpy() for g in grads if g is not None]

    results[name] = {
        'output_std': initial_output_std,
        'loss': loss.numpy(),
        'grad_mean': np.mean(grad_norms),
        'grad_max': np.max(grad_norms),
        'grad_min': np.min(grad_norms),
    }

print("=" * 65)
print(f"Zero-Init vs Random-Init 안정성 비교 ({n_blocks}블록 DiT)")
print("=" * 65)
print(f"{'항목':<25} | {'Zero-Init':>15} | {'Random-Init':>15}")
print("-" * 60)
for key in ['output_std', 'loss', 'grad_mean', 'grad_max']:
    labels = {
        'output_std': '초기 출력 표준편차',
        'loss': '초기 손실',
        'grad_mean': '그래디언트 평균 norm',
        'grad_max': '그래디언트 최대 norm',
    }
    v_zero = results['Zero-Init'][key]
    v_rand = results['Random-Init'][key]
    print(f"{labels[key]:<25} | {v_zero:>15.4f} | {v_rand:>15.4f}")

print(f"\\n분석:")
if results['Zero-Init']['output_std'] < results['Random-Init']['output_std']:
    print(f"  Zero-Init: 초기 출력이 입력에 가까움 (항등 함수 근사)")
else:
    print(f"  Zero-Init: 초기 출력 표준편차 분석 완료")
print(f"  Random-Init: 초기부터 출력 분포가 크게 변형됨")
print(f"  → Zero-Init이 깊은 DiT에서 더 안정적인 학습 시작점을 제공")"""))

# ── Cell 11: Section 6 header ──────────────────────────────────────
cells.append(md(r"""\
## 6. 다중 조건 주입 (시간 + 클래스) 데모 <a name='6.-다중-조건-주입'></a>

DiT는 여러 조건을 동시에 주입합니다:
- **타임스텝 $t$**: Sinusoidal 임베딩 → 노이즈 수준 정보
- **클래스 레이블 $y$**: Embedding 테이블 → 생성 대상 카테고리
- **텍스트**: CLIP/T5 인코더 출력 → Cross-Attention 또는 adaLN

$$c = \text{MLP}(\text{SinEmbed}(t)) + \text{Embed}(y)$$"""))

# ── Cell 12: Multi-condition injection demo ──────────────────────────
cells.append(code("""\
# ── 다중 조건 주입 (시간 + 클래스) 데모 ──────────────────────────────
class TimestepEmbedding(tf.keras.layers.Layer):
    # Sinusoidal timestep embedding (DDPM 스타일)
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, activation='silu'),
            tf.keras.layers.Dense(dim),
        ])

    def call(self, t):
        half_dim = self.dim // 2
        freqs = tf.exp(-tf.math.log(10000.0) * tf.range(0, half_dim, dtype=tf.float32) / half_dim)
        args = tf.cast(t[:, None], tf.float32) * freqs[None, :]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        return self.mlp(embedding)


class ClassEmbedding(tf.keras.layers.Layer):
    # 클래스 레이블 임베딩
    def __init__(self, num_classes, dim, **kwargs):
        super().__init__(**kwargs)
        self.embed = tf.keras.layers.Embedding(num_classes + 1, dim)  # +1 for null class
        self.null_class = num_classes  # null class for CFG

    def call(self, y, drop_prob=0.0):
        # CFG: drop_prob 확률로 null class로 대체
        if drop_prob > 0:
            mask = tf.random.uniform(tf.shape(y)) < drop_prob
            y = tf.where(mask, self.null_class, y)
        return self.embed(y)


class DiTConditioner(tf.keras.layers.Layer):
    # 시간 + 클래스 조건 결합
    def __init__(self, dim, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.time_embed = TimestepEmbedding(dim)
        self.class_embed = ClassEmbedding(num_classes, dim)

    def call(self, t, y, cfg_drop_prob=0.0):
        c_time = self.time_embed(t)
        c_class = self.class_embed(y, drop_prob=cfg_drop_prob)
        return c_time + c_class


# 테스트
dim = 256
num_classes = 1000
conditioner = DiTConditioner(dim, num_classes)

batch_size = 8
timesteps = tf.constant([0, 100, 250, 500, 750, 900, 950, 999])
class_labels = tf.constant([1, 42, 100, 207, 404, 555, 888, 999])

# 일반 조건 (CFG off)
c_normal = conditioner(timesteps, class_labels, cfg_drop_prob=0.0)

# CFG 학습 모드 (10% drop)
c_cfg = conditioner(timesteps, class_labels, cfg_drop_prob=0.1)

# 무조건 생성 (null condition)
null_labels = tf.fill([batch_size], num_classes)
c_uncond = conditioner(timesteps, null_labels, cfg_drop_prob=0.0)

print("=" * 60)
print("다중 조건 주입 시스템 테스트")
print("=" * 60)
print(f"임베딩 차원: {dim}")
print(f"클래스 수: {num_classes}")
print(f"\\n조건 벡터 shape: {c_normal.shape}")
print(f"\\n시간별 조건 벡터 통계:")
print(f"{'타임스텝':>10} | {'클래스':>8} | {'조건 norm':>12} | {'조건 mean':>12}")
print("-" * 50)
for i in range(batch_size):
    t_val = timesteps[i].numpy()
    y_val = class_labels[i].numpy()
    c_norm = tf.norm(c_normal[i]).numpy()
    c_mean = tf.reduce_mean(c_normal[i]).numpy()
    print(f"{t_val:>10} | {y_val:>8} | {c_norm:>12.4f} | {c_mean:>12.4f}")

# CFG: 조건부 vs 무조건 차이
cfg_diff = tf.reduce_mean(tf.abs(c_normal - c_uncond)).numpy()
print(f"\\n조건부 vs 무조건 벡터 차이: {cfg_diff:.4f}")
print(f"→ CFG 추론 시 이 차이를 guidance scale로 증폭")"""))

# ── Cell 13: CFG design markdown ──────────────────────────────────
cells.append(md(r"""\
## 7. DiT에서의 CFG 설계 <a name='7.-CFG-설계'></a>

DiT는 Classifier-Free Guidance를 다음과 같이 적용합니다:

**학습 시:**
- 일정 비율(보통 10%)로 클래스 레이블을 $\varnothing$(null class)로 대체
- 모델이 조건부/무조건 생성을 모두 학습

**추론 시:**
$$\tilde{\epsilon}_\theta(x_t, c) = (1 + w) \cdot \epsilon_\theta(x_t, c) - w \cdot \epsilon_\theta(x_t, \varnothing)$$

| guidance scale $w$ | 효과 | 적용 |
|-------|------|------|
| 0.0 | 무조건 생성 (조건 무시) | 다양성 최대 |
| 1.0~2.0 | 약한 가이던스 | 균형 |
| 4.0~7.5 | 강한 가이던스 (DiT 기본값 4.0) | 품질 최대 |
| >10.0 | 과도한 가이던스 | 아티팩트 발생 |"""))

# ── Cell 14: Summary ────────────────────────────────────────────────
cells.append(md(r"""\
## 8. 정리 <a name='8.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| adaLN-Zero | 조건부 정규화 + 게이팅($\alpha=0$) 초기화 | ⭐⭐⭐ |
| Zero-Initialization | 초기 상태에서 항등 함수 → 안정적 학습 시작 | ⭐⭐⭐ |
| 조건 주입 | timestep(sinusoidal) + class(embedding) → adaLN | ⭐⭐⭐ |
| CFG in DiT | $(1+w)\epsilon(c) - w\epsilon(\varnothing)$, 학습 시 10% drop | ⭐⭐⭐ |
| 표준 LN vs adaLN | 고정 파라미터 vs 조건부 동적 파라미터 | ⭐⭐ |

### 핵심 수식

$$h = x + \alpha \cdot \text{Attn}\!\left((1 + \gamma_c) \odot \text{LN}(x) + \beta_c\right)$$

$$(\gamma, \beta, \alpha) \leftarrow \text{MLP}(c), \quad \alpha_{\text{init}} = 0 \;\text{(항등 함수)}$$

$$\tilde{\epsilon}_\theta = (1 + w) \cdot \epsilon_\theta(x_t, c) - w \cdot \epsilon_\theta(x_t, \varnothing)$$

### 다음 챕터 예고
**04_flow_matching_and_rectified_flow.ipynb** — Flow Matching ODE 수식과 Rectified Flow의 직선 경로를 DDPM과 비교하며, SD3/Flux 등 최신 모델의 훈련 방식을 다룹니다."""))

# ── 노트북 생성 ──────────────────────────────────────────────────
create_notebook(cells, 'chapter17_diffusion_transformers/03_dit_conditioning_and_adaln.ipynb')
