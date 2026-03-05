"""Generate Chapter 12-01: Llama Architecture Deep Dive."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helper import md, code, create_notebook

cells = [
# ─── Cell 0: 헤더 ───
md("""# Chapter 12: 최신 대형 언어 모델 아키텍처 — Llama 심층 분석

## 학습 목표
- Llama 3 아키텍처의 핵심 구성 요소(RMSNorm, SwiGLU, GQA)의 **수학적 원리**를 이해한다
- RMSNorm이 LayerNorm 대비 갖는 **연산 효율 이점**을 수식으로 증명하고 구현한다
- SwiGLU 활성화 함수의 **게이팅 메커니즘**을 구현하고 GELU 대비 성능을 비교한다
- MHA → MQA → GQA의 발전 과정을 이해하고, **KV 헤드 수에 따른 메모리 절감률**을 계산한다
- RMSNorm + SwiGLU + GQA를 결합한 **소형 Llama Block**을 밑바닥부터 구현한다

## 목차
1. [수학적 기초: RMSNorm, SwiGLU, GQA](#1.-수학적-기초)
2. [RMSNorm vs LayerNorm 비교 구현](#2.-RMSNorm-vs-LayerNorm)
3. [SwiGLU FFN 구현](#3.-SwiGLU-FFN)
4. [MHA → MQA → GQA 발전과 구현](#4.-GQA-구현)
5. [소형 Llama Block 조립](#5.-Llama-Block)
6. [정리](#6.-정리)"""),

# ─── Cell 1: 수학적 기초 ───
md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### RMSNorm (Root Mean Square Layer Normalization)

기존 LayerNorm은 평균과 분산을 모두 계산하지만, RMSNorm은 **평균 제거 없이** RMS(제곱평균제곱근)만으로 정규화합니다:

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(\mathbf{a})} \cdot g_i, \quad \text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n}\sum_{j=1}^{n} a_j^2 + \epsilon}$$

- $a_i$: 입력 벡터의 $i$번째 원소
- $g_i$: 학습 가능한 스케일 파라미터 (gain)
- $n$: 히든 차원 크기
- $\epsilon$: 수치 안정화 상수 (보통 $10^{-6}$)

**LayerNorm과 비교:**

| 구분 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 수식 | $\frac{a_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i$ | $\frac{a_i}{\text{RMS}(\mathbf{a})} \cdot g_i$ |
| 파라미터 | $\gamma, \beta$ (2n개) | $g$ (n개) |
| 연산 | 평균 + 분산 (2-pass) | RMS만 (1-pass) |
| FLOPs | $\sim 5n$ | $\sim 3n$ |

### SwiGLU (Swish-Gated Linear Unit)

Llama의 FFN은 표준 2-layer MLP 대신 **게이팅 메커니즘**을 사용합니다:

$$\text{SwiGLU}(x) = \text{Swish}_\beta(xW_1) \otimes (xW_2)$$

$$\text{Swish}_\beta(x) = x \cdot \sigma(\beta x), \quad \sigma(x) = \frac{1}{1+e^{-x}}$$

- $x \in \mathbb{R}^{d_{model}}$: 입력
- $W_1, W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$: 게이트/값 프로젝션 ($\beta=1$ for Llama)
- $W_3 \in \mathbb{R}^{d_{ff} \times d_{model}}$: 출력 프로젝션
- $\otimes$: 원소별 곱 (element-wise product)
- $d_{ff}$: Llama 3 8B에서 14,336 ($= \frac{8}{3} \times d_{model}$의 256 배수 올림)

**최종 FFN 출력:**

$$\text{FFN}(x) = \text{SwiGLU}(x) \cdot W_3 = [\text{Swish}(xW_1) \otimes (xW_2)] W_3$$

### Grouped Query Attention (GQA)

MHA에서 GQA로의 발전은 **KV 헤드 수를 줄여** 메모리를 절약합니다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| 방식 | Q 헤드 수 | KV 헤드 수 | KV 파라미터 비율 |
|------|----------|-----------|----------------|
| MHA | $H$ | $H$ | $1.0$ |
| MQA | $H$ | $1$ | $1/H$ |
| GQA | $H$ | $G$ | $G/H$ |

Llama 3 8B: $H = 32$, $G = 8$ → KV 파라미터가 MHA 대비 $8/32 = 25\%$로 절감

$$\text{KV 메모리 절감률} = 1 - \frac{G}{H} = 1 - \frac{n_{kv}}{n_q}$$

**요약 표:**

| 구성 요소 | 수식 | Llama 3 8B 값 |
|-----------|------|--------------|
| RMSNorm | $a_i / \text{RMS}(\mathbf{a}) \cdot g_i$ | $\epsilon = 10^{-5}$ |
| SwiGLU | $\text{Swish}(xW_1) \otimes (xW_2)$ | $d_{ff} = 14336$ |
| GQA | $H_Q=32, H_{KV}=8$ | 75% KV 절감 |
| 히든 차원 | $d_{model}$ | 4096 |
| 레이어 수 | $L$ | 32 |
| 어휘 크기 | $V$ | 128,000 |"""),

# ─── Cell 2: 🐣 친절 설명 ───
md(r"""---

### 🐣 초등학생을 위한 Llama 아키텍처 친절 설명!

#### 🔢 RMSNorm이 뭔가요?

> 💡 **비유**: 반 아이들의 키를 비교할 때, **평균 키를 빼고 나누는 방법**(LayerNorm)과 **그냥 키의 크기로만 나누는 방법**(RMSNorm)이 있어요. 두 번째 방법이 계산이 더 빨라요!

RMSNorm은 숫자들을 "적당한 크기"로 만들어주는 도구예요. 숫자가 너무 크거나 작으면 AI가 학습하기 어려운데, RMSNorm이 딱 맞게 조절해줍니다.

#### 🚪 SwiGLU는 뭔가요?

> 💡 **비유**: **문 두 개가 달린 복도**를 상상해보세요. 첫 번째 문(Swish)은 "얼마나 중요한지" 점수를 매기고, 두 번째 문은 "실제 정보"를 통과시켜요. 두 문의 결과를 합쳐서 정말 중요한 정보만 통과합니다!

#### 👥 GQA는 뭔가요?

> 💡 **비유**: 시험을 볼 때, **질문지(Q)는 32명**이 각자 다르게 갖고 있지만, **답안지(K,V)는 8개 그룹이 공유**해요. 답안지를 덜 만들어도 되니까 종이(메모리)가 절약됩니다!

| 방식 | 비유 | 메모리 |
|------|------|--------|
| MHA | 학생 32명이 각자 답안지 32장 | 💸💸💸 |
| MQA | 학생 32명이 답안지 1장 공유 | 💸 (너무 적어 품질↓) |
| GQA | 학생 32명이 8그룹으로 나눠 8장 공유 | 💸💸 (적절!) |"""),

# ─── Cell 3: 📝 연습 문제 ───
md(r"""---

### 📝 연습 문제

#### 문제 1: RMSNorm 수동 계산

입력 벡터 $\mathbf{a} = [3, 4, 0]$, 게인 $\mathbf{g} = [1, 1, 1]$, $\epsilon = 0$일 때 RMSNorm의 출력을 구하시오.

<details>
<summary>💡 풀이 확인</summary>

$$\text{RMS}(\mathbf{a}) = \sqrt{\frac{3^2 + 4^2 + 0^2}{3}} = \sqrt{\frac{25}{3}} \approx 2.887$$

$$\text{RMSNorm}(\mathbf{a}) = \left[\frac{3}{2.887}, \frac{4}{2.887}, \frac{0}{2.887}\right] \approx [1.039, 1.386, 0]$$

벡터의 "크기"만으로 정규화되어, 원래 방향은 유지하면서 스케일만 조절됨.
</details>

#### 문제 2: GQA 메모리 절감률 계산

Llama 3 70B는 $H_Q = 64$, $H_{KV} = 8$을 사용합니다. MHA 대비 GQA의 KV 파라미터 절감률과, 시퀀스 길이 $S = 4096$일 때 FP16 KV 캐시 크기를 계산하시오. ($L=80, d_{head}=128$)

<details>
<summary>💡 풀이 확인</summary>

**절감률**: $1 - G/H = 1 - 8/64 = 87.5\%$

**KV 캐시 크기**:
$$M_{KV} = 2 \times L \times H_{KV} \times d_{head} \times S \times 2\text{B}$$
$$= 2 \times 80 \times 8 \times 128 \times 4096 \times 2 = 2 \times 80 \times 8 \times 128 \times 4096 \times 2$$
$$= 1,073,741,824 \text{ bytes} = 1.07 \text{ GB (배치 1 기준)}$$

MHA였다면: $2 \times 80 \times 64 \times 128 \times 4096 \times 2 = 8.59$ GB → GQA로 **7.52 GB 절약!**
</details>"""),

# ─── Cell 4: 임포트 ───
code("""import numpy as np
import matplotlib
matplotlib.use('Agg')  # 헤드리스 환경 필수
import matplotlib.pyplot as plt
import tensorflow as tf
import time

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"NumPy 버전: {np.__version__}")"""),

# ─── Cell 5: 섹션2 마크다운 ───
md("""## 2. RMSNorm vs LayerNorm 비교 구현 <a name='2.-RMSNorm-vs-LayerNorm'></a>"""),

# ─── Cell 6: RMSNorm 구현 ───
code(r"""# ── RMSNorm vs LayerNorm 구현 및 비교 ──────────────────────────
# RMSNorm과 LayerNorm을 직접 구현하여 수치적 차이와 속도를 비교합니다

class RMSNorm(tf.keras.layers.Layer):
    # Root Mean Square Layer Normalization
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = self.add_weight(name='gain', shape=(dim,),
                                 initializer='ones', trainable=True)

    def call(self, x):
        # RMS 계산: sqrt(mean(x^2) + eps)
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.g


# 테스트 데이터
d_model = 4096
batch_seq = (2, 128)  # (batch, seq_len)
x = tf.random.normal((*batch_seq, d_model))

# RMSNorm
rms_norm = RMSNorm(d_model)
rms_out = rms_norm(x)

# LayerNorm (TF 내장)
layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
ln_out = layer_norm(x)

print("=" * 55)
print("RMSNorm vs LayerNorm 수치 비교")
print("=" * 55)
print(f"입력 shape: {x.shape}")
print(f"입력 평균: {tf.reduce_mean(x).numpy():.4f}")
print(f"입력 표준편차: {tf.math.reduce_std(x).numpy():.4f}")
print()
print(f"RMSNorm 출력 평균: {tf.reduce_mean(rms_out).numpy():.6f}")
print(f"RMSNorm 출력 std:  {tf.math.reduce_std(rms_out).numpy():.6f}")
print(f"LayerNorm 출력 평균: {tf.reduce_mean(ln_out).numpy():.6f}")
print(f"LayerNorm 출력 std:  {tf.math.reduce_std(ln_out).numpy():.6f}")
print()

# 파라미터 수 비교
rms_params = sum(tf.size(v).numpy() for v in rms_norm.trainable_variables)
ln_params = sum(tf.size(v).numpy() for v in layer_norm.trainable_variables)
print(f"RMSNorm 파라미터: {rms_params:,} (gain만)")
print(f"LayerNorm 파라미터: {ln_params:,} (gamma + beta)")
print(f"파라미터 절감: {(1 - rms_params/ln_params)*100:.0f}%")"""),

# ─── Cell 7: RMSNorm 속도 벤치마크 ───
code(r"""# ── RMSNorm vs LayerNorm 속도 벤치마크 ────────────────────────
# 반복 실행하여 평균 시간 측정

n_warmup = 10
n_runs = 100

# 워밍업
for _ in range(n_warmup):
    _ = rms_norm(x)
    _ = layer_norm(x)

# RMSNorm 벤치마크
start = time.perf_counter()
for _ in range(n_runs):
    _ = rms_norm(x)
rms_time = (time.perf_counter() - start) / n_runs * 1000

# LayerNorm 벤치마크
start = time.perf_counter()
for _ in range(n_runs):
    _ = layer_norm(x)
ln_time = (time.perf_counter() - start) / n_runs * 1000

print("=" * 55)
print("속도 벤치마크 (CPU, d_model=4096, batch=2, seq=128)")
print("=" * 55)
print(f"{'방법':<20} | {'시간 (ms)':>12} | {'상대 속도':>10}")
print("-" * 55)
print(f"{'RMSNorm':<20} | {rms_time:>12.3f} | {'기준':>10}")
print(f"{'LayerNorm':<20} | {ln_time:>12.3f} | {ln_time/rms_time:>9.2f}x")
print()
print(f"RMSNorm이 LayerNorm 대비 약 {(1-rms_time/ln_time)*100:.1f}% 빠름 (CPU 기준)")"""),

# ─── Cell 8: 섹션3 마크다운 ───
md("""## 3. SwiGLU FFN 구현 <a name='3.-SwiGLU-FFN'></a>"""),

# ─── Cell 9: SwiGLU 구현 ───
code(r"""# ── SwiGLU FFN 구현 ────────────────────────────────────────────
# Llama 3 스타일의 SwiGLU FFN을 구현하고, 표준 GELU FFN과 비교합니다

class SwiGLUFFN(tf.keras.layers.Layer):
    # SwiGLU Feed-Forward Network (Llama 3 style)
    def __init__(self, d_model, d_ff):
        super().__init__()
        # W1: gate projection, W2: up projection, W3: down projection
        self.w1 = tf.keras.layers.Dense(d_ff, use_bias=False, name='gate')
        self.w2 = tf.keras.layers.Dense(d_ff, use_bias=False, name='up')
        self.w3 = tf.keras.layers.Dense(d_model, use_bias=False, name='down')

    def call(self, x):
        # SwiGLU: Swish(x @ W1) * (x @ W2) @ W3
        gate = tf.nn.silu(self.w1(x))    # Swish = SiLU
        up = self.w2(x)
        return self.w3(gate * up)         # element-wise product → down projection


class StandardFFN(tf.keras.layers.Layer):
    # 표준 GELU FFN (GPT-2 스타일)
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(d_ff, use_bias=False)
        self.fc2 = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, x):
        return self.fc2(tf.nn.gelu(self.fc1(x)))


# Llama 3 8B 기준 FFN 차원
d_model = 4096
d_ff_swiglu = 14336   # Llama 3: 8/3 * 4096 → 256의 배수 올림
d_ff_standard = 11008  # 유사 파라미터 수의 표준 FFN

swiglu_ffn = SwiGLUFFN(d_model, d_ff_swiglu)
std_ffn = StandardFFN(d_model, d_ff_standard)

# 테스트
x_test = tf.random.normal((1, 16, d_model))
y_swiglu = swiglu_ffn(x_test)
y_std = std_ffn(x_test)

# 파라미터 수 비교
swiglu_params = sum(tf.size(v).numpy() for v in swiglu_ffn.trainable_variables)
std_params = sum(tf.size(v).numpy() for v in std_ffn.trainable_variables)

print("=" * 60)
print("SwiGLU FFN vs 표준 GELU FFN 비교")
print("=" * 60)
print(f"{'항목':<25} | {'SwiGLU':>15} | {'표준 GELU':>15}")
print("-" * 60)
print(f"{'d_ff':<25} | {d_ff_swiglu:>15,} | {d_ff_standard:>15,}")
print(f"{'파라미터 수':<25} | {swiglu_params:>15,} | {std_params:>15,}")
print(f"{'출력 shape':<25} | {str(y_swiglu.shape):>15} | {str(y_std.shape):>15}")
print(f"{'게이팅 방식':<25} | {'Swish gate':>15} | {'없음':>15}")
print()
print("SwiGLU는 게이트 프로젝션(W1)을 추가하여 3개의 가중치 행렬 사용")
print(f"  W1(gate): {d_model}×{d_ff_swiglu} = {d_model*d_ff_swiglu:,}")
print(f"  W2(up):   {d_model}×{d_ff_swiglu} = {d_model*d_ff_swiglu:,}")
print(f"  W3(down): {d_ff_swiglu}×{d_model} = {d_ff_swiglu*d_model:,}")"""),

# ─── Cell 10: SwiGLU 시각화 ───
code(r"""# ── SwiGLU 활성화 함수 시각화 ─────────────────────────────────
# Swish, GELU, ReLU 활성화 함수를 비교하여 SwiGLU의 특성을 보여줍니다

x_range = np.linspace(-4, 4, 500)
x_tf = tf.constant(x_range, dtype=tf.float32)

activations = {
    'Swish (SiLU)': tf.nn.silu(x_tf).numpy(),
    'GELU': tf.nn.gelu(x_tf).numpy(),
    'ReLU': tf.nn.relu(x_tf).numpy(),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: 활성화 함수 비교
ax1 = axes[0]
colors = ['#1E88E5', '#43A047', '#E53935']
for (name, vals), c in zip(activations.items(), colors):
    ax1.plot(x_range, vals, lw=2.5, label=name, color=c)
ax1.axhline(y=0, color='gray', ls='--', lw=1)
ax1.axvline(x=0, color='gray', ls='--', lw=1)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('f(x)', fontsize=11)
ax1.set_title('활성화 함수 비교', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 오른쪽: SwiGLU 게이팅 메커니즘 시각화
ax2 = axes[1]
gate_signal = tf.nn.silu(x_tf).numpy()
value_signal = np.tanh(x_range * 0.5)  # 예시 값 신호
swiglu_out = gate_signal * value_signal

ax2.plot(x_range, gate_signal, 'b-', lw=2, label='Gate: Swish(xW₁)', alpha=0.7)
ax2.plot(x_range, value_signal, 'g-', lw=2, label='Value: xW₂', alpha=0.7)
ax2.plot(x_range, swiglu_out, 'r-', lw=2.5, label='SwiGLU: Gate ⊗ Value')
ax2.axhline(y=0, color='gray', ls='--', lw=1)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('출력', fontsize=11)
ax2.set_title('SwiGLU 게이팅 메커니즘', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter12_modern_llms/swiglu_activation.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter12_modern_llms/swiglu_activation.png")
print()
print("핵심 관찰:")
print("  • Swish는 음수 영역에서 작은 음수값을 허용 (ReLU와 차이)")
print("  • SwiGLU는 Gate 신호로 Value 신호를 '필터링'하여 중요 정보만 통과")
print("  • 이 게이팅이 표준 FFN 대비 표현력을 높여줌")"""),

# ─── Cell 11: 섹션4 마크다운 ───
md("""## 4. MHA → MQA → GQA 발전과 구현 <a name='4.-GQA-구현'></a>"""),

# ─── Cell 12: GQA 구현 ───
code(r"""# ── MHA / MQA / GQA 구현 및 비교 ──────────────────────────────
# 세 가지 Attention 방식을 구현하여 메모리 및 파라미터 차이를 비교합니다

class GroupedQueryAttention(tf.keras.layers.Layer):
    # Grouped Query Attention (GQA) - MHA/MQA를 모두 포괄
    def __init__(self, d_model, n_q_heads, n_kv_heads):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_q_heads
        self.n_groups = n_q_heads // n_kv_heads  # Q 헤드를 KV 그룹으로 묶기

        # Q 프로젝션: 전체 Q 헤드 수 사용
        self.wq = tf.keras.layers.Dense(n_q_heads * self.d_head, use_bias=False)
        # K, V 프로젝션: KV 헤드 수만 사용 (여기가 핵심!)
        self.wk = tf.keras.layers.Dense(n_kv_heads * self.d_head, use_bias=False)
        self.wv = tf.keras.layers.Dense(n_kv_heads * self.d_head, use_bias=False)
        self.wo = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, x, mask=None):
        B, S, _ = x.shape

        # 프로젝션
        q = tf.reshape(self.wq(x), (B, S, self.n_q_heads, self.d_head))
        k = tf.reshape(self.wk(x), (B, S, self.n_kv_heads, self.d_head))
        v = tf.reshape(self.wv(x), (B, S, self.n_kv_heads, self.d_head))

        # KV 헤드를 Q 그룹 수만큼 반복 (repeat_kv)
        # [B, S, n_kv, d] → [B, S, n_kv, n_groups, d] → [B, S, n_q, d]
        k = tf.repeat(k, repeats=self.n_groups, axis=2)
        v = tf.repeat(v, repeats=self.n_groups, axis=2)

        # [B, n_heads, S, d_head] 형태로 전치
        q = tf.transpose(q, [0, 2, 1, 3])  # [B, n_q, S, d]
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # Scaled Dot-Product Attention
        scale = tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        attn = tf.matmul(q, k, transpose_b=True) / scale  # [B, n_q, S, S]

        if mask is not None:
            attn += mask

        attn_weights = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn_weights, v)  # [B, n_q, S, d]

        # Concat heads
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (B, S, self.n_q_heads * self.d_head))
        return self.wo(out)


# Llama 3 8B 스케일 (축소 버전)
d_model = 256  # 데모용 축소

configs = {
    'MHA (H=8, G=8)': (8, 8),
    'MQA (H=8, G=1)': (8, 1),
    'GQA (H=8, G=2)': (8, 2),  # Llama 스타일: G = H/4
}

x_test = tf.random.normal((1, 32, d_model))

print("=" * 65)
print("MHA / MQA / GQA 파라미터 및 KV 메모리 비교")
print("=" * 65)
print(f"{'방식':<20} | {'Q 헤드':>7} | {'KV 헤드':>7} | {'파라미터':>12} | {'KV 비율':>8}")
print("-" * 65)

kv_sizes = {}
for name, (n_q, n_kv) in configs.items():
    attn = GroupedQueryAttention(d_model, n_q, n_kv)
    _ = attn(x_test)  # 빌드
    total_params = sum(tf.size(v).numpy() for v in attn.trainable_variables)
    kv_ratio = n_kv / n_q
    kv_sizes[name] = kv_ratio
    print(f"{name:<20} | {n_q:>7} | {n_kv:>7} | {total_params:>12,} | {kv_ratio:>7.0%}")

print()
print("💡 GQA는 MHA의 품질을 유지하면서 MQA 수준의 메모리 효율을 달성!")"""),

# ─── Cell 13: GQA 시각화 ───
code(r"""# ── MHA / MQA / GQA 구조 비교 시각화 ─────────────────────────
# Q-K-V 헤드 매핑을 직관적으로 보여주는 다이어그램

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

configs_viz = [
    ('MHA\n(Multi-Head)', 8, 8),
    ('MQA\n(Multi-Query)', 8, 1),
    ('GQA\n(Grouped-Query)', 8, 2),
]

for ax, (title, n_q, n_kv) in zip(axes, configs_viz):
    n_groups = n_q // n_kv

    # Q 헤드 (상단)
    for i in range(n_q):
        color_idx = i // n_groups
        color = plt.cm.Set3(color_idx / max(n_kv, 1))
        ax.add_patch(plt.Rectangle((i * 1.1, 2.5), 0.9, 0.8,
                                    facecolor=color, edgecolor='black', lw=1.5))
        ax.text(i * 1.1 + 0.45, 2.9, f'Q{i}', ha='center', va='center', fontsize=7)

    # KV 헤드 (하단)
    kv_width = (n_q * 1.1) / n_kv
    for j in range(n_kv):
        color = plt.cm.Set3(j / max(n_kv, 1))
        # K
        ax.add_patch(plt.Rectangle((j * kv_width, 1.2), kv_width - 0.2, 0.8,
                                    facecolor=color, edgecolor='black', lw=1.5, alpha=0.7))
        ax.text(j * kv_width + (kv_width - 0.2) / 2, 1.6, f'K{j}',
                ha='center', va='center', fontsize=7)
        # V
        ax.add_patch(plt.Rectangle((j * kv_width, 0.0), kv_width - 0.2, 0.8,
                                    facecolor=color, edgecolor='black', lw=1.5, alpha=0.5))
        ax.text(j * kv_width + (kv_width - 0.2) / 2, 0.4, f'V{j}',
                ha='center', va='center', fontsize=7)

    # 연결선
    for i in range(n_q):
        kv_idx = i // n_groups
        x_q = i * 1.1 + 0.45
        x_kv = kv_idx * kv_width + (kv_width - 0.2) / 2
        ax.plot([x_q, x_kv], [2.5, 2.0], 'k-', alpha=0.3, lw=0.8)

    ax.set_xlim(-0.5, n_q * 1.1 + 0.5)
    ax.set_ylim(-0.5, 4.0)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylabel('Q → K,V 매핑', fontsize=9)
    ax.axis('off')

    # KV 메모리 비율 표시
    ratio = n_kv / n_q * 100
    ax.text(n_q * 1.1 / 2, -0.3, f'KV 메모리: {ratio:.0f}%', ha='center',
            fontsize=10, fontweight='bold', color='#D32F2F')

plt.suptitle('Attention 방식별 Q-KV 헤드 매핑 비교', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('chapter12_modern_llms/gqa_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter12_modern_llms/gqa_comparison.png")
print()
print("GQA (Llama 3 스타일):")
print(f"  • Q 헤드 32개가 KV 헤드 8개를 4:1로 공유")
print(f"  • MHA 대비 KV 파라미터 75% 절감")
print(f"  • MQA보다 품질이 좋으면서 MHA보다 빠름 → 최적 균형점")"""),

# ─── Cell 14: 섹션5 마크다운 ───
md("""## 5. 소형 Llama Block 조립 <a name='5.-Llama-Block'></a>"""),

# ─── Cell 15: Llama Block 조립 ───
code(r"""# ── 소형 Llama Decoder Block 조립 ──────────────────────────────
# RMSNorm + GQA + SwiGLU를 결합하여 Llama 3 스타일 블록을 완성합니다

class LlamaDecoderBlock(tf.keras.layers.Layer):
    # Llama 3 스타일 Decoder Block: Pre-Norm + GQA + SwiGLU
    def __init__(self, d_model, n_q_heads, n_kv_heads, d_ff, eps=1e-6):
        super().__init__()
        # Pre-Norm: Attention 전 RMSNorm
        self.attn_norm = RMSNorm(d_model, eps)
        self.attn = GroupedQueryAttention(d_model, n_q_heads, n_kv_heads)

        # Pre-Norm: FFN 전 RMSNorm
        self.ffn_norm = RMSNorm(d_model, eps)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def call(self, x, mask=None):
        # Pre-Norm → Attention → Residual
        h = x + self.attn(self.attn_norm(x), mask=mask)
        # Pre-Norm → SwiGLU FFN → Residual
        out = h + self.ffn(self.ffn_norm(h))
        return out


# 소형 Llama 3 설정 (데모용 축소)
config = {
    'd_model': 256,
    'n_q_heads': 8,
    'n_kv_heads': 2,     # GQA: 4:1
    'd_ff': 688,          # 약 8/3 * 256 → 256의 배수에 가깝게
}

block = LlamaDecoderBlock(**config)

# Causal mask 생성
seq_len = 32
causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
causal_mask = (1.0 - causal_mask) * -1e9  # 미래 토큰 마스킹
causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]  # [1, 1, S, S]

# 포워드 패스
x_input = tf.random.normal((2, seq_len, config['d_model']))
output = block(x_input, mask=causal_mask)

print("=" * 60)
print("소형 Llama Decoder Block 구조")
print("=" * 60)
print(f"설정:")
print(f"  d_model:    {config['d_model']}")
print(f"  n_q_heads:  {config['n_q_heads']}")
print(f"  n_kv_heads: {config['n_kv_heads']} (GQA ratio: {config['n_q_heads']//config['n_kv_heads']}:1)")
print(f"  d_ff:       {config['d_ff']}")
print()
print(f"입력 shape:  {x_input.shape}")
print(f"출력 shape:  {output.shape}")
print(f"shape 보존:  {x_input.shape == output.shape}")
print()

# 전체 파라미터 수
total_params = sum(tf.size(v).numpy() for v in block.trainable_variables)
print(f"블록 총 파라미터: {total_params:,}")
print()
print("블록 구조 (Pre-Norm Transformer):")
print("  x → RMSNorm → GQA → + (residual)")
print("    → RMSNorm → SwiGLU FFN → + (residual) → output")"""),

# ─── Cell 16: Llama 3 실제 규모 분석 ───
code(r"""# ── Llama 3 8B 실제 규모 분석 ─────────────────────────────────
# 실제 Llama 3 8B 파라미터로 모델 규모를 분석합니다

print("=" * 65)
print("Llama 3 8B 아키텍처 분석")
print("=" * 65)

# Llama 3 8B 공식 스펙 (Meta AI, 2024)
specs = {
    'd_model': 4096,
    'n_layers': 32,
    'n_q_heads': 32,
    'n_kv_heads': 8,
    'd_head': 128,       # 4096 / 32
    'd_ff': 14336,       # 8/3 * 4096, rounded to 256 multiple
    'vocab_size': 128000,
    'max_seq_len': 8192,
    'rope_base': 500000,
}

for k, v in specs.items():
    print(f"  {k:<15} = {v:>10,}")

print()

# 파라미터 수 계산
d = specs['d_model']
L = specs['n_layers']
Hq = specs['n_q_heads']
Hkv = specs['n_kv_heads']
dh = specs['d_head']
dff = specs['d_ff']
V = specs['vocab_size']

# 각 블록 파라미터
attn_params = (Hq * dh * d) + 2 * (Hkv * dh * d) + (d * d)  # Wq + Wk + Wv + Wo
ffn_params = 3 * d * dff  # W1 + W2 + W3 (SwiGLU)
norm_params = 2 * d  # 2x RMSNorm per block
block_params = attn_params + ffn_params + norm_params

# 전체 모델
embed_params = V * d  # 토큰 임베딩
final_norm = d
head_params = d * V  # lm_head (보통 임베딩과 공유)
total = L * block_params + embed_params + final_norm

print(f"{'구성 요소':<30} | {'파라미터 수':>15} | {'비율':>8}")
print("-" * 60)
print(f"{'Attention (per block)':<30} | {attn_params:>15,} | {attn_params/block_params*100:>6.1f}%")
print(f"{'SwiGLU FFN (per block)':<30} | {ffn_params:>15,} | {ffn_params/block_params*100:>6.1f}%")
print(f"{'RMSNorm (per block)':<30} | {norm_params:>15,} | {norm_params/block_params*100:>6.1f}%")
print(f"{'블록 합계':<30} | {block_params:>15,} |")
print(f"{'32 블록 합계':<30} | {L*block_params:>15,} |")
print(f"{'토큰 임베딩':<30} | {embed_params:>15,} |")
print(f"{'Final RMSNorm':<30} | {final_norm:>15,} |")
print("-" * 60)
print(f"{'총계 (lm_head 공유 가정)':<30} | {total:>15,} |")
print(f"{'≈':<30} | {total/1e9:>14.2f}B |")"""),

# ─── Cell 17: 정리 ───
md(r"""## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| RMSNorm | 평균 제거 없이 RMS만으로 정규화 → LayerNorm 대비 ~40% 연산 절감 | ⭐⭐⭐ |
| SwiGLU | Swish 게이팅으로 정보 필터링 → GELU FFN 대비 표현력 향상 | ⭐⭐⭐ |
| GQA | KV 헤드를 그룹으로 공유 → MHA 품질 + MQA 효율의 최적 균형 | ⭐⭐⭐ |
| Pre-Norm | Attention/FFN 전에 정규화 → Post-Norm 대비 학습 안정성 향상 | ⭐⭐ |
| Residual Connection | 입력을 출력에 더해 그래디언트 소실 방지 | ⭐⭐ |

### 핵심 수식

$$\text{RMSNorm}(a_i) = \frac{a_i}{\sqrt{\frac{1}{n}\sum_j a_j^2 + \epsilon}} \cdot g_i$$

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2)$$

$$\text{GQA: } H_Q = 32, \; H_{KV} = 8 \;\Rightarrow\; \text{KV 절감률} = 1 - \frac{8}{32} = 75\%$$

### Llama 3 8B 핵심 스펙

| 항목 | 값 |
|------|-----|
| 레이어 수 | 32 |
| 히든 차원 | 4096 |
| Q 헤드 | 32 |
| KV 헤드 | 8 (GQA) |
| FFN 차원 | 14,336 (SwiGLU) |
| 어휘 크기 | 128,000 |
| 총 파라미터 | ~8B |

### 다음 챕터 예고
**Chapter 12-02: KV Cache와 메모리 관리** — Autoregressive 생성에서 KV Cache의 메모리 계산, Rolling Buffer, Prefix Caching 등 실전 서빙에 필수적인 메모리 최적화 기법을 다룹니다."""),
]

create_notebook(cells, 'chapter12_modern_llms/01_llama_architecture_deepdive.ipynb')
