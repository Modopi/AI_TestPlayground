"""Generate chapter16_sparse_attention/practice/ex02_linear_attention_layer.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# 실습 퀴즈: Linear Attention과 GLA 레이어 구현

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다

## 목차
- [Q1: Linear Attention 커널 함수](#q1)
- [Q2: Causal Linear Attention 순환](#q2)
- [Q3: GLA 게이트 메커니즘](#q3)
- [Q4: 메모리 비교 Standard vs Linear](#q4)
- [종합 도전: Full GLA 레이어 구현 + 시퀀스 길이 스케일링 테스트](#bonus)"""))

# ── Cell 2: Import ───────────────────────────────────────────────
cells.append(code("""\
# ── 라이브러리 임포트 ──────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import time

np.random.seed(42)
print(f"TensorFlow 버전: {tf.__version__}")"""))

# ── Cell 3: Q1 문제 ─────────────────────────────────────────────
cells.append(md(r"""\
## Q1: Linear Attention 커널 함수 <a name='q1'></a>

### 문제

Linear Attention에서 커널 함수 $\phi$는 음수가 아닌 값을 보장해야 합니다.
다음 세 가지 커널 함수를 구현하고 출력 범위를 비교하세요:

1. $\phi_1(x) = \text{elu}(x) + 1$
2. $\phi_2(x) = \text{ReLU}(x)$
3. $\phi_3(x) = 1 + x/\sqrt{d} + x^2/(2d)$ (2차 Taylor 근사)

**여러분의 예측:** 어떤 커널이 음수를 가장 잘 방지할까요? `elu+1 / ReLU / Taylor`"""))

# ── Cell 4: Q1 풀이 ─────────────────────────────────────────────
cells.append(code("""\
# ── Q1 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q1 풀이: Linear Attention 커널 함수")
print("=" * 45)

x = tf.random.normal([1000, 64])

# 커널 1: elu + 1
phi_elu = tf.nn.elu(x) + 1.0

# 커널 2: ReLU
phi_relu = tf.nn.relu(x)

# 커널 3: 2차 Taylor 근사
d = 64.0
phi_taylor = 1.0 + x / tf.sqrt(d) + x**2 / (2 * d)

kernels = {'elu+1': phi_elu, 'ReLU': phi_relu, 'Taylor(2차)': phi_taylor}

print(f"커널 함수 비교 (입력: 표준정규분포, d={int(d)}):")
print(f"{'커널':<14} | {'최솟값':>10} | {'최댓값':>10} | {'평균':>10} | {'음수 비율':>10}")
print("-" * 62)

for name, phi in kernels.items():
    min_val = tf.reduce_min(phi).numpy()
    max_val = tf.reduce_max(phi).numpy()
    mean_val = tf.reduce_mean(phi).numpy()
    neg_ratio = tf.reduce_mean(tf.cast(phi < 0, tf.float32)).numpy() * 100
    print(f"{name:<14} | {min_val:>10.4f} | {max_val:>10.4f} | {mean_val:>10.4f} | {neg_ratio:>8.1f}%")

print()
print("[해설]")
print("  elu+1: 최솟값 > 0 보장 (elu의 최솟값 = -1, +1하면 0이상)")
print("  ReLU: 음수 = 0이 되어 정보 손실, 하지만 음수는 없음")
print("  Taylor: 이론적으로 exp의 근사이지만, 음수가 나올 수 있음")
print("  → 실전에서는 elu+1이 가장 안정적으로 사용됨")"""))

# ── Cell 5: Q2 문제 ─────────────────────────────────────────────
cells.append(md(r"""\
## Q2: Causal Linear Attention 순환 <a name='q2'></a>

### 문제

Causal Linear Attention의 순환 형태를 구현하세요:

$$s_t = s_{t-1} + \phi(k_t)^T v_t$$
$$z_t = z_{t-1} + \phi(k_t)$$
$$o_t = \frac{\phi(q_t) \cdot s_t}{\phi(q_t) \cdot z_t + \epsilon}$$

시퀀스 [1, 2, 3, 4]에 대해 각 스텝의 $s_t$, $z_t$, $o_t$를 추적하세요.

**여러분의 예측:** $s_t$의 크기는 시간에 따라 `증가/감소/일정`할까요?"""))

# ── Cell 6: Q2 풀이 ─────────────────────────────────────────────
cells.append(code("""\
# ── Q2 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q2 풀이: Causal Linear Attention 순환")
print("=" * 45)

d_k = 4
d_v = 4
seq_len = 8

np.random.seed(42)
Q = np.random.randn(seq_len, d_k).astype(np.float32)
K = np.random.randn(seq_len, d_k).astype(np.float32)
V = np.random.randn(seq_len, d_v).astype(np.float32)

# 커널 함수: elu + 1
def phi(x):
    return np.where(x > 0, x + 1, np.exp(x))

Q_phi = phi(Q)
K_phi = phi(K)

# 순환 계산
s = np.zeros((d_k, d_v))  # 상태 행렬
z = np.zeros(d_k)  # 정규화 벡터
eps = 1e-6

print(f"Causal Linear Attention 순환 (d_k={d_k}, d_v={d_v}, seq={seq_len})")
print()

outputs = []
for t in range(seq_len):
    # 상태 업데이트
    s = s + np.outer(K_phi[t], V[t])  # [d_k, d_v]
    z = z + K_phi[t]  # [d_k]

    # 출력 계산
    numerator = Q_phi[t] @ s  # [d_v]
    denominator = Q_phi[t] @ z + eps  # scalar
    o_t = numerator / denominator  # [d_v]
    outputs.append(o_t)

    s_norm = np.linalg.norm(s)
    z_norm = np.linalg.norm(z)
    o_norm = np.linalg.norm(o_t)
    print(f"  t={t}: ||s_t||={s_norm:.4f}, ||z_t||={z_norm:.4f}, ||o_t||={o_norm:.4f}")

print()
print("[해설]")
print("  s_t의 노름은 시간에 따라 증가합니다 (누적)")
print("  이것이 Linear Attention의 한계: 과거 정보가 계속 쌓임")
print("  → GLA는 게이트로 이 문제를 해결합니다")
print(f"  상태 행렬 크기: {d_k}x{d_v} = {d_k*d_v} (시퀀스 길이와 무관!)")"""))

# ── Cell 7: Q3 문제 ─────────────────────────────────────────────
cells.append(md(r"""\
## Q3: GLA 게이트 메커니즘 <a name='q3'></a>

### 문제

GLA의 게이트가 있는 순환을 구현하고, 게이트 값에 따른 동작을 확인하세요:

$$s_t = G_t \odot s_{t-1} + k_t^T v_t$$

실험:
1. $G_t = 0.0$ (모든 과거 정보 제거)
2. $G_t = 0.5$ (절반 유지)
3. $G_t = 0.99$ (거의 모든 과거 유지)

**여러분의 예측:** 게이트 값이 클수록 상태 노름이 `더 빠르게/더 느리게/비슷하게` 증가할까요?"""))

# ── Cell 8: Q3 풀이 ─────────────────────────────────────────────
cells.append(code("""\
# ── Q3 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q3 풀이: GLA 게이트 메커니즘")
print("=" * 45)

d_k = 8
d_v = 8
seq_len = 50

np.random.seed(42)
K_test = np.random.randn(seq_len, d_k).astype(np.float32)
V_test = np.random.randn(seq_len, d_v).astype(np.float32)
Q_test = np.random.randn(seq_len, d_k).astype(np.float32)

gate_values = [0.0, 0.5, 0.9, 0.99]
all_norms = {}

for g_val in gate_values:
    s = np.zeros((d_k, d_v))
    norms = []
    for t in range(seq_len):
        kv = np.outer(K_test[t], V_test[t])
        s = g_val * s + kv
        norms.append(np.linalg.norm(s))
    all_norms[g_val] = norms

# 결과 출력
print(f"게이트 값별 최종 상태 노름 (t={seq_len-1}):")
print(f"{'Gate':>6} | {'최종 ||s||':>12} | {'최대 ||s||':>12} | {'동작':>20}")
print("-" * 58)
for g_val in gate_values:
    final = all_norms[g_val][-1]
    peak = max(all_norms[g_val])
    if g_val == 0.0:
        behavior = "즉시 망각 (최근만)"
    elif g_val < 0.9:
        behavior = "점진적 감쇠"
    elif g_val < 1.0:
        behavior = "느린 감쇠 (장기 기억)"
    else:
        behavior = "완전 누적"
    print(f"{g_val:>6.2f} | {final:>12.4f} | {peak:>12.4f} | {behavior:>20}")

# 시각화
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for g_val in gate_values:
    ax.plot(all_norms[g_val], lw=2, label=f'Gate = {g_val}')

ax.set_xlabel('시간 스텝', fontsize=11)
ax.set_ylabel('상태 노름 $||s_t||$', fontsize=11)
ax.set_title('GLA 게이트 값에 따른 상태 행렬 크기 변화', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/practice_gla_gate_effect.png', dpi=100, bbox_inches='tight')
plt.close()
print("\\n그래프 저장됨: chapter16_sparse_attention/practice_gla_gate_effect.png")

print()
print("[해설]")
print("  Gate=0.0: 이전 상태 완전 제거 → 최근 토큰만 기억")
print("  Gate=0.5: 기하급수적 감쇠 → 짧은 범위 의존성")
print("  Gate=0.99: 느린 감쇠 → 장기 의존성 유지")
print("  GLA는 입력에 따라 Gate를 동적으로 조절하여 최적 기억 전략 학습")"""))

# ── Cell 9: Q4 문제 ─────────────────────────────────────────────
cells.append(md(r"""\
## Q4: 메모리 비교 Standard vs Linear <a name='q4'></a>

### 문제

시퀀스 길이 $N$, 차원 $d$일 때 다음을 계산하세요:

| 측면 | Standard Attention | Linear Attention |
|------|-------------------|------------------|
| Attention 행렬 | $N \times N$ | 없음 |
| KV 저장 (추론) | $2Nd$ | $d^2$ |
| FLOPs | $2N^2d$ | $2Nd^2$ |

$d=128$에서 Standard보다 Linear가 메모리/연산 효율적인 $N$의 최소값은?

**여러분의 예측:** $N \geq$ `?`"""))

# ── Cell 10: Q4 풀이 ─────────────────────────────────────────────
cells.append(code("""\
# ── Q4 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q4 풀이: 메모리 비교 Standard vs Linear")
print("=" * 45)

d = 128

# FLOPs 교차점: 2N^2d = 2Nd^2 → N = d
print(f"FLOPs 교차점 분석 (d={d}):")
print(f"  Standard: 2N²d")
print(f"  Linear:   2Nd²")
print(f"  교차점: 2N²d = 2Nd² → N = d = {d}")
print(f"  → N > {d}이면 Linear가 FLOPs 효율적")

# KV 메모리 교차점: 2Nd = d^2 → N = d/2
print(f"\\nKV 메모리 교차점:")
print(f"  Standard: 2Nd")
print(f"  Linear:   d²")
print(f"  교차점: 2Nd = d² → N = d/2 = {d//2}")
print(f"  → N > {d//2}이면 Linear가 KV 메모리 효율적")

# 수치 비교 표
N_values = [32, 64, 128, 256, 512, 1024, 4096, 16384]
print(f"\\n상세 비교 (d={d}):")
print(f"{'N':>8} | {'Std FLOPs':>12} | {'Lin FLOPs':>12} | {'비율':>8} | {'Std KV':>10} | {'Lin KV':>10}")
print("-" * 72)

for N in N_values:
    std_flops = 2 * N * N * d
    lin_flops = 2 * N * d * d
    ratio = std_flops / lin_flops

    std_kv = 2 * N * d
    lin_kv = d * d

    marker = " ✅" if ratio > 1 else ""
    print(f"{N:>8} | {std_flops:>12,} | {lin_flops:>12,} | {ratio:>7.1f}x | "
          f"{std_kv:>10,} | {lin_kv:>10,}{marker}")

print()
print("[해설]")
print(f"  N = d = {d}이 교차점: 이 이상에서 Linear가 유리")
print(f"  N = 4096: Standard는 Linear보다 {4096//d}배 더 많은 FLOPs 필요")
print(f"  N = 16384: {16384//d}배 차이 → 긴 시퀀스에서 압도적 우위")"""))

# ── Cell 11: 종합 도전 문제 ──────────────────────────────────────
cells.append(md("""\
## 종합 도전: Full GLA 레이어 구현 + 시퀀스 길이 스케일링 테스트 <a name='bonus'></a>

### 문제

완전한 GLA(Gated Linear Attention) 레이어를 TF 클래스로 구현하고:
1. Standard Attention과 출력 비교
2. 시퀀스 길이 [64, 128, 256, 512, 1024]에서 실행 시간 비교
3. 스케일링 그래프 그리기"""))

# ── Cell 12: 종합 도전 풀이 ──────────────────────────────────────
cells.append(code("""\
# ── 종합 도전 풀이: Full GLA 레이어 구현 ─────────────────────────
print("=" * 45)
print("종합 도전: Full GLA 레이어 구현")
print("=" * 45)

class GLALayer(tf.keras.layers.Layer):
    # Gated Linear Attention 레이어

    def __init__(self, d_model, d_key, d_value):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value

        self.W_q = tf.keras.layers.Dense(d_key)
        self.W_k = tf.keras.layers.Dense(d_key)
        self.W_v = tf.keras.layers.Dense(d_value)
        self.W_g = tf.keras.layers.Dense(d_key * d_value)
        self.W_o = tf.keras.layers.Dense(d_model)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        residual = x
        x = self.layer_norm(x)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        q = self.W_q(x)
        k = tf.nn.elu(self.W_k(x)) + 1.0  # 커널 함수 적용
        v = self.W_v(x)
        g = tf.sigmoid(self.W_g(x))
        g = tf.reshape(g, [batch_size, seq_len, self.d_key, self.d_value])

        outputs = tf.TensorArray(dtype=tf.float32, size=seq_len)
        s = tf.zeros([batch_size, self.d_key, self.d_value])

        for t in tf.range(seq_len):
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            q_t = q[:, t, :]
            g_t = g[:, t, :, :]

            kv = tf.einsum('bi,bj->bij', k_t, v_t)
            s = g_t * s + kv
            o_t = tf.einsum('bi,bij->bj', q_t, s)
            outputs = outputs.write(t, o_t)

        output = tf.transpose(outputs.stack(), [1, 0, 2])
        output = self.W_o(output) + residual
        return output


class StandardAttnLayer(tf.keras.layers.Layer):
    # 표준 Attention 레이어 (비교 기준)

    def __init__(self, d_model, d_key, d_value):
        super().__init__()
        self.d_key = d_key
        self.W_q = tf.keras.layers.Dense(d_key)
        self.W_k = tf.keras.layers.Dense(d_key)
        self.W_v = tf.keras.layers.Dense(d_value)
        self.W_o = tf.keras.layers.Dense(d_model)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        residual = x
        x = self.layer_norm(x)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        d = tf.cast(self.d_key, tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(d)
        seq_len = tf.shape(x)[1]
        mask = tf.linalg.band_part(tf.ones([seq_len, seq_len]), -1, 0)
        scores = scores + (1.0 - mask) * (-1e9)
        weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(weights, v)
        return self.W_o(output) + residual


# 모델 생성
d_model = 64
d_key = 32
d_value = 32

gla_layer = GLALayer(d_model, d_key, d_value)
std_layer = StandardAttnLayer(d_model, d_key, d_value)

# 시퀀스 길이별 실행 시간 비교
seq_lengths = [64, 128, 256, 512]
gla_times = []
std_times = []
batch = 2

print(f"시퀀스 길이별 실행 시간 비교 (d_model={d_model}, d_k={d_key}):")
print(f"{'시퀀스':>8} | {'Standard (ms)':>14} | {'GLA (ms)':>14} | {'비율':>8}")
print("-" * 52)

for sl in seq_lengths:
    x = tf.random.normal([batch, sl, d_model])

    # 워밍업
    _ = std_layer(x)
    _ = gla_layer(x)

    n_runs = 3
    t0 = time.time()
    for _ in range(n_runs):
        _ = std_layer(x)
    std_time = (time.time() - t0) / n_runs * 1000

    t0 = time.time()
    for _ in range(n_runs):
        _ = gla_layer(x)
    gla_time = (time.time() - t0) / n_runs * 1000

    std_times.append(std_time)
    gla_times.append(gla_time)
    ratio = std_time / max(gla_time, 0.01)
    print(f"{sl:>8} | {std_time:>14.2f} | {gla_time:>14.2f} | {ratio:>7.2f}x")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) 실행 시간
ax1 = axes[0]
ax1.plot(seq_lengths, std_times, 'r-o', lw=2.5, ms=8, label='Standard Attention')
ax1.plot(seq_lengths, gla_times, 'g-s', lw=2.5, ms=8, label='GLA (순환)')
ax1.set_xlabel('시퀀스 길이', fontsize=11)
ax1.set_ylabel('실행 시간 (ms)', fontsize=11)
ax1.set_title('Standard vs GLA 실행 시간', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) 이론적 메모리 비교
ax2 = axes[1]
std_mem = [s**2 * 4 / 1024 for s in seq_lengths]  # Attention matrix (KB)
gla_mem = [d_key * d_value * 4 / 1024 for _ in seq_lengths]  # State matrix (KB)
ax2.plot(seq_lengths, std_mem, 'r-o', lw=2.5, ms=8, label='Standard ($N^2$)')
ax2.plot(seq_lengths, gla_mem, 'g-s', lw=2.5, ms=8, label='GLA ($d^2$)')
ax2.set_xlabel('시퀀스 길이', fontsize=11)
ax2.set_ylabel('Attention 메모리 (KB)', fontsize=11)
ax2.set_title('메모리 사용량 비교', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/practice_gla_scaling.png', dpi=100, bbox_inches='tight')
plt.close()
print("\\n그래프 저장됨: chapter16_sparse_attention/practice_gla_scaling.png")

print(f"\\n핵심 결론:")
print(f"  GLA 상태 메모리: {d_key*d_value*4/1024:.2f} KB (시퀀스 길이와 무관!)")
print(f"  Standard N=512 메모리: {512**2*4/1024:.0f} KB")
print(f"  GLA는 긴 시퀀스에서 메모리 측면의 압도적 우위를 가짐")
print(f"  (단, 순환 형태의 Python 루프로 인해 속도는 최적화된 Standard보다 느릴 수 있음)")
print(f"  실전에서는 chunk-wise 병렬 + 하드웨어 커널 최적화를 적용합니다")"""))

create_notebook(cells, 'chapter16_sparse_attention/practice/ex02_linear_attention_layer.ipynb')
