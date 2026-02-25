"""Generate chapter16_sparse_attention/03_linear_attention_and_hybrids.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 16: 최신 거대 모델의 효율성 — Linear Attention과 Hybrid 구조

## 학습 목표
- 표준 Softmax Attention과 Linear Attention의 복잡도 차이를 수식으로 이해한다
- GLA(Gated Linear Attention)의 순환(recurrence) 구조를 구현한다
- RetNet과 Mamba의 핵심 아이디어를 비교 분석한다
- Qwen의 SWA+Full+Linear Hybrid 아키텍처를 이해한다
- 시퀀스 길이별 메모리/속도 스케일링을 실험적으로 확인한다

## 목차
1. [수학적 기초: Linear Attention과 커널 함수](#1.-수학적-기초)
2. [Standard vs Linear Attention 복잡도 비교](#2.-복잡도-비교)
3. [GLA 순환 시뮬레이션](#3.-GLA-순환-시뮬레이션)
4. [Qwen Hybrid 아키텍처](#4.-Qwen-Hybrid-아키텍처)
5. [시퀀스 길이별 메모리/속도 비교](#5.-메모리-속도-비교)
6. [정리](#6.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### 표준 Softmax Attention

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right) V$$

- 시간 복잡도: $O(N^2 d)$, 공간 복잡도: $O(N^2 + Nd)$
- $N$: 시퀀스 길이, $d$: 차원

### Linear Attention

커널 함수 $\phi$를 사용하여 softmax를 근사:

$$O_i = \frac{\phi(Q_i) \sum_{j=1}^{N} \phi(K_j)^T V_j}{\phi(Q_i) \sum_{j=1}^{N} \phi(K_j)^T}$$

**핵심 트릭**: 결합 법칙을 이용한 연산 순서 변경

$$O = \phi(Q) \underbrace{\left(\phi(K)^T V\right)}_{S \in \mathbb{R}^{d \times d}} \quad \text{vs} \quad O = \underbrace{\left(\phi(Q) \phi(K)^T\right)}_{A \in \mathbb{R}^{N \times N}} V$$

- 좌측: $O(Nd^2)$ — Linear Attention (시퀀스 길이에 선형)
- 우측: $O(N^2d)$ — Standard Attention (시퀀스 길이에 이차)

### Causal Linear Attention (순환 형태)

$$s_t = s_{t-1} + \phi(k_t)^T v_t, \quad o_t = \frac{\phi(q_t) s_t}{\phi(q_t) z_t}$$

- $s_t \in \mathbb{R}^{d \times d}$: 숨겨진 상태 (누적 KV)
- $z_t = z_{t-1} + \phi(k_t)$: 정규화 항
- 토큰당 메모리: $O(d^2)$ — 시퀀스 길이와 무관!

### Gated Linear Attention (GLA)

$$s_t = G_t \odot s_{t-1} + k_t^T v_t$$

$$o_t = q_t \cdot s_t$$

- $G_t \in \mathbb{R}^{d \times d}$: 게이트 행렬 (망각 메커니즘)
- $G_t$가 작으면 과거 정보 잊기, 크면 유지

### RetNet의 순환 공식

$$s_t = \gamma \cdot s_{t-1} + k_t^T v_t, \quad o_t = q_t \cdot s_t$$

- $\gamma \in (0, 1)$: 고정 감쇠 비율
- GLA와 유사하지만 게이트가 스칼라 상수

**요약 표:**

| 모델 | 순환 공식 | 복잡도 (추론) | 장점 |
|------|-----------|---------------|------|
| Standard Attention | $\text{softmax}(QK^T/\sqrt{d})V$ | $O(N^2d)$ | 높은 표현력 |
| Linear Attention | $\phi(Q)(\phi(K)^TV)$ | $O(Nd^2)$ | 긴 시퀀스 효율 |
| GLA | $s_t = G_t \odot s_{t-1} + k_t^T v_t$ | $O(d^2)$/토큰 | 선택적 망각 |
| RetNet | $s_t = \gamma s_{t-1} + k_t^T v_t$ | $O(d^2)$/토큰 | 단순한 감쇠 |
| Mamba | SSM 기반 선택적 스캔 | $O(d)$/토큰 | 입력 의존 게이팅 |

---"""))

# ── Cell 3: 🐣 friendly explanation ──────────────────────────────
cells.append(md("""\
### 🐣 초등학생을 위한 Linear Attention 친절 설명!

#### 🔢 왜 Linear Attention이 필요한가요?

> 💡 **비유**: 교실에서 모든 학생(N명)이 서로 대화하려면 $N \\times N$번 대화해야 해요. 
> 학생이 100명이면 10,000번! 하지만 **반장을 통해** 대화하면 각자 반장과만 대화하면 돼서 
> 200번이면 충분해요!

Linear Attention은 **반장(숨겨진 상태 $s_t$)** 을 두는 거예요:
- 표준 Attention: 모든 토큰이 서로 직접 대화 → $N^2$번 계산
- Linear Attention: 각 토큰이 "요약 노트($s_t$)"만 업데이트하고 읽기 → $N$번 계산

#### 🧠 GLA는 뭐가 다른가요?

> 💡 **비유**: "요약 노트"에 **지우개(게이트 $G_t$)** 가 있어서, 
> 중요하지 않은 과거 정보는 지울 수 있어요!

일반 Linear Attention은 모든 과거 정보를 계속 쌓기만 하지만, 
GLA는 선택적으로 잊을 수 있어서 **더 똑똑하게** 정보를 관리해요.

---"""))

# ── Cell 4: 📝 Exercises ──────────────────────────────────────────
cells.append(md(r"""\
### 📝 연습 문제

#### 문제 1: 복잡도 비교

시퀀스 길이 $N=4096$, 차원 $d=128$일 때, Standard Attention과 Linear Attention의 연산량(FLOPs)을 비교하세요.

<details>
<summary>💡 풀이 확인</summary>

**Standard Attention:**
$$\text{FLOPs} = O(N^2 d) = 4096^2 \times 128 \approx 2.15 \times 10^9$$

**Linear Attention:**
$$\text{FLOPs} = O(N d^2) = 4096 \times 128^2 \approx 6.71 \times 10^7$$

$$\text{비율} = \frac{N^2 d}{N d^2} = \frac{N}{d} = \frac{4096}{128} = 32$$

→ Linear Attention이 **32배** 더 빠릅니다! ($N > d$일 때 항상 유리)
</details>

#### 문제 2: GLA 상태 업데이트

$s_0 = \mathbf{0}$, $G_1 = 0.9I$, $k_1 = [1, 0]^T$, $v_1 = [2, 3]^T$일 때 $s_1$을 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$$s_1 = G_1 \odot s_0 + k_1^T v_1 = 0.9 \cdot \mathbf{0} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} [2, 3] = \begin{bmatrix} 2 & 3 \\ 0 & 0 \end{bmatrix}$$

→ 첫 번째 토큰의 KV 정보가 상태 행렬에 저장됩니다.
</details>

---"""))

# ── Cell 5: Import cell ──────────────────────────────────────────
cells.append(code("""\
# ── 라이브러리 임포트 ──────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import time

np.random.seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"NumPy 버전: {np.__version__}")"""))

# ── Cell 6: Section 2 - Complexity comparison ────────────────────
cells.append(md(r"""\
## 2. Standard vs Linear Attention 복잡도 비교 <a name='2.-복잡도-비교'></a>

두 방식의 시간/공간 복잡도를 이론적으로 비교하고, 실제 연산 시간을 측정합니다.

| 측면 | Standard Attention | Linear Attention |
|------|-------------------|------------------|
| 시간 복잡도 | $O(N^2 d)$ | $O(Nd^2)$ |
| 공간 복잡도 | $O(N^2)$ | $O(d^2)$ |
| 유리한 조건 | $N < d$ | $N > d$ (긴 시퀀스) |"""))

cells.append(code("""\
# ── Standard vs Linear Attention 구현 및 비교 ────────────────────
def standard_attention(Q, K, V):
    # Q, K, V: [batch, seq, dim]
    d = tf.cast(tf.shape(K)[-1], tf.float32)
    scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d)
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, V)

def linear_attention(Q, K, V, kernel_fn=None):
    # 커널 함수: elu(x) + 1 (양수 보장)
    if kernel_fn is None:
        kernel_fn = lambda x: tf.nn.elu(x) + 1

    Q_prime = kernel_fn(Q)  # [B, N, d]
    K_prime = kernel_fn(K)  # [B, N, d]

    # S = K'^T V : [B, d, d]
    S = tf.matmul(K_prime, V, transpose_a=True)

    # O = Q' S : [B, N, d]
    numerator = tf.matmul(Q_prime, S)

    # 정규화
    Z = tf.matmul(Q_prime, tf.reduce_sum(K_prime, axis=1, keepdims=True),
                  transpose_b=True)
    Z = tf.maximum(Z, 1e-6)

    return numerator / Z

# 다양한 시퀀스 길이에서 연산 시간 비교
dims = 64
batch = 4
seq_lengths_test = [64, 128, 256, 512, 1024, 2048]

std_times = []
lin_times = []

print(f"Standard vs Linear Attention 실행 시간 비교:")
print(f"  d = {dims}, batch = {batch}")
print(f"{'시퀀스 길이':>12} | {'Standard (ms)':>14} | {'Linear (ms)':>14} | {'속도 비율':>10}")
print("-" * 58)

for seq_len in seq_lengths_test:
    Q = tf.random.normal([batch, seq_len, dims])
    K = tf.random.normal([batch, seq_len, dims])
    V = tf.random.normal([batch, seq_len, dims])

    # 워밍업
    _ = standard_attention(Q, K, V)
    _ = linear_attention(Q, K, V)

    # 시간 측정
    n_runs = 5
    t0 = time.time()
    for _ in range(n_runs):
        _ = standard_attention(Q, K, V)
    std_time = (time.time() - t0) / n_runs * 1000

    t0 = time.time()
    for _ in range(n_runs):
        _ = linear_attention(Q, K, V)
    lin_time = (time.time() - t0) / n_runs * 1000

    std_times.append(std_time)
    lin_times.append(lin_time)
    ratio = std_time / max(lin_time, 1e-6)
    print(f"{seq_len:>12} | {std_time:>14.2f} | {lin_time:>14.2f} | {ratio:>9.2f}x")

print(f"\\n→ 시퀀스 길이가 길어질수록 Linear Attention의 이점이 커짐")"""))

# ── Cell 8: Section 3 - GLA recurrence ───────────────────────────
cells.append(md(r"""\
## 3. GLA 순환 시뮬레이션 <a name='3.-GLA-순환-시뮬레이션'></a>

GLA(Gated Linear Attention)의 순환 형태를 구현합니다:

$$s_t = G_t \odot s_{t-1} + k_t v_t^T$$

$$o_t = s_t^T q_t$$

여기서 $G_t = \sigma(W_g h_t)$는 입력에 따라 동적으로 결정되는 게이트입니다."""))

cells.append(code("""\
# ── GLA 순환(Recurrence) 구현 ────────────────────────────────────
class GLARecurrence(tf.keras.layers.Layer):
    # Gated Linear Attention - 순환 형태 구현

    def __init__(self, d_model, d_key, d_value):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value

        self.W_q = tf.keras.layers.Dense(d_key)
        self.W_k = tf.keras.layers.Dense(d_key)
        self.W_v = tf.keras.layers.Dense(d_value)
        self.W_g = tf.keras.layers.Dense(d_key * d_value)  # 게이트

    def call(self, x):
        batch, seq_len, _ = x.shape

        q = self.W_q(x)  # [B, N, d_k]
        k = self.W_k(x)  # [B, N, d_k]
        v = self.W_v(x)  # [B, N, d_v]
        g = tf.sigmoid(self.W_g(x))  # [B, N, d_k*d_v]
        g = tf.reshape(g, [batch, seq_len, self.d_key, self.d_value])

        outputs = []
        # 상태 행렬 초기화
        s = tf.zeros([batch, self.d_key, self.d_value])

        for t in range(seq_len):
            k_t = k[:, t, :]  # [B, d_k]
            v_t = v[:, t, :]  # [B, d_v]
            q_t = q[:, t, :]  # [B, d_k]
            g_t = g[:, t, :, :]  # [B, d_k, d_v]

            # 상태 업데이트: s_t = G_t * s_{t-1} + k_t^T v_t
            kv_outer = tf.einsum('bi,bj->bij', k_t, v_t)  # [B, d_k, d_v]
            s = g_t * s + kv_outer

            # 출력: o_t = q_t^T s_t
            o_t = tf.einsum('bi,bij->bj', q_t, s)  # [B, d_v]
            outputs.append(o_t)

        output = tf.stack(outputs, axis=1)  # [B, N, d_v]
        return output, s


# 테스트
d_model = 64
d_key = 32
d_value = 32
batch_size = 2
seq_len = 20

gla = GLARecurrence(d_model, d_key, d_value)
x_test = tf.random.normal([batch_size, seq_len, d_model])
output, final_state = gla(x_test)

print(f"GLA Recurrence 결과:")
print(f"  입력: {x_test.shape}")
print(f"  출력: {output.shape}")
print(f"  최종 상태: {final_state.shape}")
print(f"  상태 행렬 크기: {d_key} x {d_value} = {d_key * d_value} (시퀀스 길이와 무관!)")

# 게이트 분석
g_sample = tf.sigmoid(gla.W_g(x_test))
g_mean = tf.reduce_mean(g_sample).numpy()
g_std = tf.math.reduce_std(g_sample).numpy()
print(f"\\n게이트 통계:")
print(f"  평균: {g_mean:.4f} (0.5에 가까우면 중립적)")
print(f"  표준편차: {g_std:.4f}")
print(f"  → 게이트가 0에 가까우면 과거 정보 잊기, 1에 가까우면 유지")

# 메모리 비교
std_kv_cache = seq_len * 2 * d_key  # Standard: 모든 K,V 저장
gla_state = d_key * d_value  # GLA: 상태 행렬만 저장
print(f"\\n메모리 비교 (시퀀스 길이 = {seq_len}):")
print(f"  Standard Attention KV Cache: {std_kv_cache} 원소")
print(f"  GLA 상태 행렬: {gla_state} 원소")
print(f"  절감률: {(1 - gla_state/std_kv_cache)*100:.1f}%")"""))

# ── Cell 10: Section 4 - Qwen Hybrid ────────────────────────────
cells.append(md(r"""\
## 4. Qwen Hybrid 아키텍처 <a name='4.-Qwen-Hybrid-아키텍처'></a>

Qwen은 세 가지 Attention 방식을 레이어별로 혼합하는 Hybrid 구조를 사용합니다:

| 레이어 유형 | Attention 방식 | 용도 |
|-------------|---------------|------|
| Full Attention | 표준 Softmax | 전역 의존성 포착 |
| Sliding Window (SWA) | 윈도우 내 Softmax | 지역 패턴 포착 |
| Linear Attention | 커널 기반 선형 | 효율적 장거리 요약 |

이러한 구조의 장점:
1. Full Attention으로 중요한 전역 관계 유지
2. SWA로 계산량 절감 (대부분의 레이어)
3. Linear로 초장거리 컨텍스트 효율적 처리"""))

cells.append(code("""\
# ── Qwen Hybrid 아키텍처 시뮬레이션 ──────────────────────────────
def sliding_window_attention(Q, K, V, window_size=256):
    # 슬라이딩 윈도우 Attention (간소화 버전)
    batch, seq_len, d = Q.shape
    d_float = tf.cast(d, tf.float32)

    # 전체 attention 계산 후 윈도우 마스크 적용
    scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_float)

    # 슬라이딩 윈도우 마스크 생성
    positions = tf.range(seq_len)
    row_pos = tf.expand_dims(positions, 1)  # [N, 1]
    col_pos = tf.expand_dims(positions, 0)  # [1, N]
    mask = tf.cast(tf.abs(row_pos - col_pos) <= window_size // 2, tf.float32)

    # causal 마스크도 적용
    causal_mask = tf.cast(col_pos <= row_pos, tf.float32)
    combined_mask = mask * causal_mask

    scores = scores + (1.0 - combined_mask) * (-1e9)
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, V)

# Hybrid 구조: 레이어별 방식 배정
n_layers = 24
layer_types = []
for i in range(n_layers):
    if i % 6 == 0:
        layer_types.append('Full')
    elif i % 6 == 3:
        layer_types.append('Linear')
    else:
        layer_types.append('SWA')

print(f"Qwen Hybrid 아키텍처 ({n_layers} 레이어):")
print(f"{'레이어':>6} | {'유형':<8} | 역할")
print("-" * 40)
type_counts = {'Full': 0, 'SWA': 0, 'Linear': 0}
for i, lt in enumerate(layer_types):
    type_counts[lt] += 1
    role = {'Full': '전역 의존성', 'SWA': '지역 패턴', 'Linear': '장거리 요약'}[lt]
    if i < 12 or i >= n_layers - 2:
        print(f"  L{i:>3} | {lt:<8} | {role}")
    elif i == 12:
        print(f"   ... | ...      | ...")

print(f"\\n레이어 유형 분포:")
for lt, count in type_counts.items():
    pct = count / n_layers * 100
    bar = '█' * int(pct / 2)
    print(f"  {lt:<8}: {count:>2}개 ({pct:.0f}%) {bar}")

# 시퀀스 길이별 연산 비교
seq_len_test = 512
d_test = 64
batch_t = 2
window = 128

Q_t = tf.random.normal([batch_t, seq_len_test, d_test])
K_t = tf.random.normal([batch_t, seq_len_test, d_test])
V_t = tf.random.normal([batch_t, seq_len_test, d_test])

out_full = standard_attention(Q_t, K_t, V_t)
out_swa = sliding_window_attention(Q_t, K_t, V_t, window_size=window)
out_lin = linear_attention(Q_t, K_t, V_t)

print(f"\\n출력 shape 확인:")
print(f"  Full Attention: {out_full.shape}")
print(f"  SWA (w={window}): {out_swa.shape}")
print(f"  Linear Attention: {out_lin.shape}")

# Hybrid 총 FLOPs 추정
N = 4096
d = 128
full_flops = N * N * d * type_counts['Full']
swa_flops = N * window * d * type_counts['SWA']
linear_flops = N * d * d * type_counts['Linear']
total_hybrid = full_flops + swa_flops + linear_flops
total_all_full = N * N * d * n_layers

print(f"\\nFLOPs 비교 (N={N}, d={d}):")
print(f"  전체 Full Attention: {total_all_full/1e9:.2f} GFLOPs")
print(f"  Hybrid: {total_hybrid/1e9:.2f} GFLOPs")
print(f"  절감률: {(1-total_hybrid/total_all_full)*100:.1f}%")"""))

# ── Cell 12: Section 5 - Memory/Speed comparison ────────────────
cells.append(md(r"""\
## 5. 시퀀스 길이별 메모리/속도 비교 <a name='5.-메모리-속도-비교'></a>

시퀀스 길이가 증가할 때 각 Attention 방식의 이론적 메모리 사용량과 연산량을 비교합니다.

**이론적 복잡도:**
- Standard: 메모리 $O(N^2)$, 연산 $O(N^2 d)$
- SWA ($w$): 메모리 $O(Nw)$, 연산 $O(Nwd)$
- Linear: 메모리 $O(d^2)$, 연산 $O(Nd^2)$"""))

cells.append(code("""\
# ── 시퀀스 길이별 메모리/속도 스케일링 시각화 ─────────────────────
seq_lengths = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
d = 128
w = 512  # SWA 윈도우 크기

# 이론적 메모리 (Attention 행렬 크기)
mem_standard = seq_lengths ** 2  # O(N^2)
mem_swa = seq_lengths * w  # O(Nw)
mem_linear = np.full_like(seq_lengths, d * d, dtype=float)  # O(d^2)

# 이론적 FLOPs
flops_standard = seq_lengths ** 2 * d
flops_swa = seq_lengths * w * d
flops_linear = seq_lengths * d ** 2

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) 메모리 스케일링
ax1 = axes[0]
ax1.loglog(seq_lengths / 1000, mem_standard / 1e6, 'r-o', lw=2.5, ms=7, label='Standard ($O(N^2)$)')
ax1.loglog(seq_lengths / 1000, mem_swa / 1e6, 'b-s', lw=2, ms=7, label=f'SWA ($O(Nw), w={w}$)')
ax1.loglog(seq_lengths / 1000, mem_linear / 1e6, 'g-^', lw=2, ms=7, label='Linear ($O(d^2)$)')
ax1.set_xlabel('시퀀스 길이 (K 토큰)', fontsize=11)
ax1.set_ylabel('Attention 메모리 (M 원소)', fontsize=11)
ax1.set_title('시퀀스 길이별 메모리 스케일링', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) FLOPs 스케일링
ax2 = axes[1]
ax2.loglog(seq_lengths / 1000, flops_standard / 1e9, 'r-o', lw=2.5, ms=7, label='Standard ($O(N^2d)$)')
ax2.loglog(seq_lengths / 1000, flops_swa / 1e9, 'b-s', lw=2, ms=7, label=f'SWA ($O(Nwd)$)')
ax2.loglog(seq_lengths / 1000, flops_linear / 1e9, 'g-^', lw=2, ms=7, label='Linear ($O(Nd^2)$)')
ax2.set_xlabel('시퀀스 길이 (K 토큰)', fontsize=11)
ax2.set_ylabel('FLOPs (G)', fontsize=11)
ax2.set_title('시퀀스 길이별 연산량 스케일링', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/attention_scaling_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter16_sparse_attention/attention_scaling_comparison.png")

# 수치 비교 표
print(f"\\n시퀀스 길이별 FLOPs 비교 (d={d}):")
print(f"{'길이':>8} | {'Standard':>12} | {'SWA':>12} | {'Linear':>12} | {'Std/Lin 비율':>12}")
print("-" * 65)
for i, N in enumerate(seq_lengths):
    ratio = flops_standard[i] / flops_linear[i]
    print(f"{N:>8} | {flops_standard[i]/1e9:>10.2f}G | {flops_swa[i]/1e9:>10.2f}G | "
          f"{flops_linear[i]/1e9:>10.2f}G | {ratio:>11.1f}x")

print(f"\\n→ N=65K에서 Linear는 Standard보다 {flops_standard[-1]/flops_linear[-1]:.0f}배 효율적!")"""))

# ── Cell 14: Summary ─────────────────────────────────────────────
cells.append(md(r"""\
## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| Linear Attention | $\phi(Q)(\phi(K)^TV)$ — $O(Nd^2)$ 복잡도 | ⭐⭐⭐ |
| GLA 순환 | $s_t = G_t \odot s_{t-1} + k_t^T v_t$ — 선택적 망각 | ⭐⭐⭐ |
| RetNet | $\gamma$ 고정 감쇠 기반 순환 | ⭐⭐ |
| Mamba | SSM 기반 입력 의존적 선택 메커니즘 | ⭐⭐ |
| Qwen Hybrid | Full + SWA + Linear 레이어 혼합 | ⭐⭐⭐ |
| 커널 함수 | $\phi(x) = \text{elu}(x) + 1$ — 양수 보장 | ⭐⭐ |
| 상태 행렬 | $s_t \in \mathbb{R}^{d \times d}$ — 시퀀스 길이와 무관 | ⭐⭐⭐ |

### 핵심 수식

$$O_i = \frac{\phi(Q_i) \sum_{j} \phi(K_j)^T V_j}{\phi(Q_i) \sum_{j} \phi(K_j)^T}$$

$$s_t = G_t \odot s_{t-1} + k_t^T v_t, \quad o_t = q_t^T s_t$$

### 다음 챕터 예고
**04_long_context_and_sparse_attn.ipynb** — YaRN 컨텍스트 확장, DeepSeek Sparse Attention, Sliding Window 방법론을 통합하여 50% 이상의 비용 절감 기법을 분석합니다."""))

create_notebook(cells, 'chapter16_sparse_attention/03_linear_attention_and_hybrids.ipynb')
