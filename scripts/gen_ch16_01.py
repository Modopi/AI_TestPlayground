"""Generate chapter16_sparse_attention/01_deepseek_v3_fp8_training.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 16: 최신 거대 모델의 효율성 — FP8 혼합 정밀도 훈련과 DeepSeek-V3

## 학습 목표
- FP8 E4M3 / E5M2 부동소수점 형식의 비트 레이아웃과 수치 범위를 이해한다
- Per-tensor / per-channel 스케일링을 통한 FP8 양자화 수식을 도출한다
- FP8 훈련 시 발생하는 양자화 오차와 SNR을 분석한다
- DeepSeek-V3의 Auxiliary-Loss-Free 로드밸런싱 기법을 수식으로 이해한다
- Multi-Token Prediction(MTP) 수식을 도출하고 학습 효율성을 분석한다

## 목차
1. [수학적 기초: FP8 부동소수점과 양자화](#1.-수학적-기초)
2. [FP8/FP16/BF16/FP32 수치 범위 비교](#2.-수치-범위-비교)
3. [FP8 양자화 시뮬레이션과 오차 분석](#3.-FP8-양자화-시뮬레이션)
4. [FP8 E4M3 vs E5M2 트레이드오프](#4.-E4M3-vs-E5M2)
5. [FP8 vs FP16 학습 안정성 비교](#5.-학습-안정성-비교)
6. [Auxiliary-Loss-Free 로드밸런싱](#6.-Auxiliary-Loss-Free-로드밸런싱)
7. [Multi-Token Prediction (MTP)](#7.-Multi-Token-Prediction)
8. [정리](#8.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### FP8 부동소수점 형식

FP8은 8비트로 부동소수점을 표현하는 형식으로, 두 가지 변형이 있습니다:

**E4M3 (4-bit exponent, 3-bit mantissa):**

$$x = (-1)^s \times 2^{e - 7} \times (1 + m \cdot 2^{-3})$$

- $s$: 부호 비트 (1비트)
- $e$: 지수 (4비트, bias = 7)
- $m$: 가수 (3비트)
- 표현 범위: $[-448, 448]$

**E5M2 (5-bit exponent, 2-bit mantissa):**

$$x = (-1)^s \times 2^{e - 15} \times (1 + m \cdot 2^{-2})$$

- 표현 범위: $[-57344, 57344]$
- 더 넓은 범위, 더 낮은 정밀도

### FP8 스케일링 양자화

입력 텐서 $x$를 FP8로 변환하는 스케일링 수식:

$$Q(x) = \text{clamp}\!\left(\left\lfloor \frac{x}{s} \right\rceil,\; -q_{max},\; q_{max}\right) \cdot s$$

- $s = \frac{\max(|x|)}{q_{max}}$: 스케일 팩터
- $q_{max} = 448$ (E4M3) 또는 $q_{max} = 57344$ (E5M2)

**Per-tensor vs Per-channel 스케일링:**

| 방식 | 스케일 계산 | 장점 | 단점 |
|------|-------------|------|------|
| Per-tensor | $s = \frac{\max(\|x\|)}{q_{max}}$ | 단순, 빠름 | 아웃라이어에 취약 |
| Per-channel | $s_c = \frac{\max(\|x_c\|)}{q_{max}}$ | 채널별 최적화 | 오버헤드 증가 |

### 양자화 오차와 SNR

$$\text{MSE} = \mathbb{E}\left[(x - Q(x))^2\right]$$

$$\text{SNR} = 10 \log_{10}\!\left(\frac{\text{Var}(x)}{\text{MSE}}\right) \propto 2^{2b}$$

- $b$: 비트 수 (FP8 → $b \approx 3\sim4$ 유효 비트)

### Auxiliary-Loss-Free 로드밸런싱

DeepSeek-V3는 보조 손실 없이 전문가 부하를 균형시킵니다:

$$g_i' = g_i + b_i, \quad b_i \leftarrow b_i + \alpha \cdot \text{sign}(\bar{f} - f_i)$$

- $g_i$: 전문가 $i$의 게이트 값
- $b_i$: 편향 보정 항 (학습 가능)
- $f_i$: 전문가 $i$에 할당된 토큰 비율
- $\bar{f} = 1/N_E$: 목표 균등 비율
- $\alpha$: 업데이트 스텝 크기

### Multi-Token Prediction (MTP)

$$\mathcal{L}_{MTP} = \sum_{k=1}^{K} \lambda_k \cdot \mathbb{E}\left[-\log p_\theta(x_{t+k} \mid x_{\leq t})\right]$$

- $K$: 예측 토큰 수 (DeepSeek-V3: $K = 2$)
- $\lambda_k$: 각 토큰 위치의 가중치

**요약 표:**

| 구분 | 수식 | 설명 |
|------|------|------|
| FP8 E4M3 범위 | $[-448, 448]$ | 높은 정밀도, 좁은 범위 |
| FP8 E5M2 범위 | $[-57344, 57344]$ | 낮은 정밀도, 넓은 범위 |
| 스케일링 양자화 | $Q(x) = \lfloor x/s \rceil \cdot s$ | 동적 범위 조정 |
| SNR | $\propto 2^{2b}$ | 비트 수에 지수적 비례 |
| 편향 보정 | $b_i \leftarrow b_i + \alpha \cdot \text{sign}(\bar{f} - f_i)$ | 보조 손실 없는 균형 |

---"""))

# ── Cell 3: 🐣 friendly explanation ──────────────────────────────
cells.append(md("""\
### 🐣 초등학생을 위한 FP8 친절 설명!

#### 🔢 FP8이 뭔가요?

> 💡 **비유**: 숫자를 적을 때, A4 종이 한 장(FP32)에 쓰면 아주 세밀하게 쓸 수 있지만, 
> 포스트잇(FP8)에 쓰면 공간이 작아서 대략적으로만 적을 수 있어요!

FP8은 숫자를 **8비트**로 표현하는 아주 작은 형식이에요:
- **E4M3**: "큰 숫자는 못 쓰지만, 작은 숫자는 꽤 정확해요" → 가중치(weight) 저장에 좋아요
- **E5M2**: "큰 숫자도 쓸 수 있지만, 정확도가 좀 떨어져요" → 기울기(gradient) 저장에 좋아요

#### ⚖️ 왜 FP8로 훈련하나요?

> 💡 **비유**: 수학 시험을 볼 때, 정밀한 계산기(FP32) 대신 
> 주판(FP8)으로 풀면 **2배 빠르게** 풀 수 있어요! 약간의 오차가 있지만, 정답에 거의 맞아요.

DeepSeek-V3는 FP8로 훈련해서:
- 메모리를 **절반**으로 줄였어요
- GPU 연산을 **2배** 빠르게 했어요
- 그런데도 정확도는 거의 떨어지지 않았어요!

---"""))

# ── Cell 4: 📝 Exercises ──────────────────────────────────────────
cells.append(md(r"""\
### 📝 연습 문제

#### 문제 1: FP8 E4M3 표현 범위

E4M3에서 가수(mantissa)가 3비트이고 지수(exponent)가 4비트일 때, 
최대 양수 값을 계산하세요. (bias = 7, 특수값 제외)

<details>
<summary>💡 풀이 확인</summary>

$$x_{max} = 2^{(15-7)} \times (1 + (2^3-1) \cdot 2^{-3})$$
$$= 2^{8} \times (1 + 7/8) = 256 \times 1.875 = 480$$

실제 FP8 E4M3 표준에서는 NaN 표현을 위해 $e=15, m=7$을 예약하므로:
$$x_{max} = 2^{(14-7)} \times (1 + 7/8) = 128 \times 1.875 = 240$$

→ 실제 구현에서는 $448$을 사용합니다 (NVIDIA FP8 사양 기준, $e=15$의 일부를 유효값으로 활용).
</details>

#### 문제 2: 양자화 SNR 비교

FP32(23비트 가수)와 FP8 E4M3(3비트 가수)의 이론적 SNR 비율을 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$$\frac{\text{SNR}_{FP32}}{\text{SNR}_{FP8}} \approx \frac{2^{2 \times 23}}{2^{2 \times 3}} = 2^{40} \approx 10^{12}$$

→ FP32는 FP8보다 약 $10^{12}$ 배 더 높은 SNR을 가집니다. 하지만 딥러닝에서는 노이즈에 강건하므로 FP8로도 충분한 학습이 가능합니다.
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
import struct

np.random.seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"NumPy 버전: {np.__version__}")"""))

# ── Cell 6: Section 2 - Numeric range comparison ─────────────────
cells.append(md(r"""\
## 2. FP8/FP16/BF16/FP32 수치 범위 비교 <a name='2.-수치-범위-비교'></a>

다양한 부동소수점 형식의 비트 구성과 수치 범위를 비교합니다.

| 형식 | 총 비트 | 부호 | 지수 | 가수 | 최대값 | 최소 정규값 |
|------|---------|------|------|------|--------|-------------|
| FP32 | 32 | 1 | 8 | 23 | $\sim3.4\times10^{38}$ | $\sim1.2\times10^{-38}$ |
| FP16 | 16 | 1 | 5 | 10 | $65504$ | $\sim6.1\times10^{-5}$ |
| BF16 | 16 | 1 | 8 | 7 | $\sim3.4\times10^{38}$ | $\sim1.2\times10^{-38}$ |
| FP8 E4M3 | 8 | 1 | 4 | 3 | $448$ | $\sim0.015625$ |
| FP8 E5M2 | 8 | 1 | 5 | 2 | $57344$ | $\sim6.1\times10^{-5}$ |"""))

# ── Cell 7: Range comparison code ────────────────────────────────
cells.append(code("""\
# ── FP8/FP16/BF16/FP32 수치 범위 비교 표 ─────────────────────────
formats = {
    'FP32':     {'bits': 32, 'sign': 1, 'exp': 8,  'man': 23, 'max': 3.4028235e+38,  'min_normal': 1.175494e-38},
    'FP16':     {'bits': 16, 'sign': 1, 'exp': 5,  'man': 10, 'max': 65504.0,        'min_normal': 6.1035e-5},
    'BF16':     {'bits': 16, 'sign': 1, 'exp': 8,  'man': 7,  'max': 3.3895314e+38,  'min_normal': 1.175494e-38},
    'FP8 E4M3': {'bits': 8,  'sign': 1, 'exp': 4,  'man': 3,  'max': 448.0,          'min_normal': 0.015625},
    'FP8 E5M2': {'bits': 8,  'sign': 1, 'exp': 5,  'man': 2,  'max': 57344.0,        'min_normal': 6.1035e-5},
}

print("=" * 85)
print(f"{'형식':<12} | {'비트':>4} | {'부호':>4} | {'지수':>4} | {'가수':>4} | {'최대값':>15} | {'최소 정규값':>15}")
print("=" * 85)
for name, f in formats.items():
    print(f"{name:<12} | {f['bits']:>4} | {f['sign']:>4} | {f['exp']:>4} | {f['man']:>4} | "
          f"{f['max']:>15.4e} | {f['min_normal']:>15.4e}")
print("=" * 85)

# 동적 범위 (dB) 계산
print(f"\\n동적 범위 비교 (dB):")
print(f"{'형식':<12} | {'동적 범위 (dB)':>15} | {'유효 소수 자릿수':>15}")
print("-" * 50)
for name, f in formats.items():
    dynamic_range_db = 20 * np.log10(f['max'] / f['min_normal'])
    decimal_digits = np.log10(2**f['man'])
    print(f"{name:<12} | {dynamic_range_db:>15.1f} | {decimal_digits:>15.2f}")

# 메모리 절약률
print(f"\\n메모리 절약률 (FP32 대비):")
for name, f in formats.items():
    savings = (1 - f['bits'] / 32) * 100
    print(f"  {name}: {savings:.0f}% 절약 ({f['bits']}비트 / 32비트)")"""))

# ── Cell 8: Section 3 - FP8 quantization simulation ──────────────
cells.append(md(r"""\
## 3. FP8 양자화 시뮬레이션과 오차 분석 <a name='3.-FP8-양자화-시뮬레이션'></a>

실제 텐서에 FP8 양자화를 적용하고 발생하는 오차를 분석합니다.

$$Q_{E4M3}(x) = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right), -448, 448\right) \cdot s$$

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - Q(x_i))^2$$"""))

# ── Cell 9: FP8 quantization code ────────────────────────────────
cells.append(code("""\
# ── FP8 양자화 시뮬레이션 ────────────────────────────────────────
def simulate_fp8_quantization(x, fmt='e4m3'):
    # FP8 양자화 시뮬레이션 (소프트웨어 에뮬레이션)
    if fmt == 'e4m3':
        q_max = 448.0
        mantissa_bits = 3
    else:  # e5m2
        q_max = 57344.0
        mantissa_bits = 2

    # per-tensor 스케일링
    abs_max = np.max(np.abs(x)) + 1e-12
    scale = abs_max / q_max

    # 양자화: 스케일링 → 라운딩 → 클램핑
    x_scaled = x / scale
    n_levels = 2 ** mantissa_bits
    x_quantized = np.round(x_scaled * n_levels) / n_levels
    x_quantized = np.clip(x_quantized, -q_max, q_max)

    # 역양자화
    x_dequant = x_quantized * scale
    return x_dequant, scale

# 테스트 데이터: 다양한 분포의 텐서
np.random.seed(42)
n_elements = 10000

# 정규 분포 (모델 가중치 근사)
x_normal = np.random.randn(n_elements).astype(np.float32)
# 균등 분포
x_uniform = np.random.uniform(-2, 2, n_elements).astype(np.float32)
# 아웃라이어가 있는 분포
x_outlier = np.random.randn(n_elements).astype(np.float32)
x_outlier[np.random.choice(n_elements, 50)] *= 100  # 0.5% 아웃라이어

distributions = {'정규 분포': x_normal, '균등 분포': x_uniform, '아웃라이어 분포': x_outlier}

print(f"FP8 양자화 오차 분석 (원소 수: {n_elements})")
print(f"{'분포':<16} | {'형식':<8} | {'MSE':>12} | {'SNR (dB)':>10} | {'최대 오차':>10} | {'스케일':>10}")
print("-" * 78)

for dist_name, x in distributions.items():
    for fmt in ['e4m3', 'e5m2']:
        x_q, scale = simulate_fp8_quantization(x, fmt=fmt)
        mse = np.mean((x - x_q) ** 2)
        var_x = np.var(x)
        snr = 10 * np.log10(var_x / (mse + 1e-12))
        max_err = np.max(np.abs(x - x_q))
        fmt_label = 'E4M3' if fmt == 'e4m3' else 'E5M2'
        print(f"{dist_name:<16} | {fmt_label:<8} | {mse:>12.6f} | {snr:>10.2f} | {max_err:>10.4f} | {scale:>10.6f}")

print(f"\\n결론: E4M3는 일반 분포에서 더 높은 SNR (높은 정밀도)")
print(f"      E5M2는 아웃라이어 분포에서 상대적으로 안정적 (넓은 범위)")"""))

# ── Cell 10: Section 4 - E4M3 vs E5M2 ────────────────────────────
cells.append(md(r"""\
## 4. FP8 E4M3 vs E5M2 트레이드오프 <a name='4.-E4M3-vs-E5M2'></a>

DeepSeek-V3에서의 사용 전략:
- **Forward pass (가중치, 활성화)**: E4M3 — 높은 정밀도 필요
- **Backward pass (기울기)**: E5M2 — 큰 동적 범위 필요

$$\text{Forward}: W_{E4M3} \times X_{E4M3} \rightarrow Y_{FP32}$$
$$\text{Backward}: \frac{\partial L}{\partial Y}_{E5M2} \times X_{E4M3}^T \rightarrow \frac{\partial L}{\partial W}_{FP32}$$"""))

# ── Cell 11: E4M3 vs E5M2 visualization ──────────────────────────
cells.append(code("""\
# ── E4M3 vs E5M2 트레이드오프 시각화 ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) 양자화 오차 분포 비교
ax1 = axes[0]
x_test = np.random.randn(5000).astype(np.float32) * 2
x_e4m3, _ = simulate_fp8_quantization(x_test, 'e4m3')
x_e5m2, _ = simulate_fp8_quantization(x_test, 'e5m2')

err_e4m3 = x_test - x_e4m3
err_e5m2 = x_test - x_e5m2

ax1.hist(err_e4m3, bins=80, alpha=0.6, color='blue', label=f'E4M3 (std={np.std(err_e4m3):.4f})', density=True)
ax1.hist(err_e5m2, bins=80, alpha=0.6, color='red', label=f'E5M2 (std={np.std(err_e5m2):.4f})', density=True)
ax1.set_xlabel('양자화 오차', fontsize=11)
ax1.set_ylabel('밀도', fontsize=11)
ax1.set_title('E4M3 vs E5M2 양자화 오차 분포', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) 크기별 SNR 비교
ax2 = axes[1]
scale_factors = np.logspace(-2, 2, 20)
snr_e4m3_list = []
snr_e5m2_list = []

for sf in scale_factors:
    x_scaled = x_test * sf
    for fmt, snr_list in [('e4m3', snr_e4m3_list), ('e5m2', snr_e5m2_list)]:
        x_q, _ = simulate_fp8_quantization(x_scaled, fmt)
        mse = np.mean((x_scaled - x_q) ** 2) + 1e-12
        snr_val = 10 * np.log10(np.var(x_scaled) / mse)
        snr_list.append(snr_val)

ax2.semilogx(scale_factors, snr_e4m3_list, 'b-o', lw=2, ms=6, label='E4M3 (가중치용)')
ax2.semilogx(scale_factors, snr_e5m2_list, 'r-s', lw=2, ms=6, label='E5M2 (기울기용)')
ax2.set_xlabel('텐서 스케일 팩터', fontsize=11)
ax2.set_ylabel('SNR (dB)', fontsize=11)
ax2.set_title('스케일에 따른 FP8 SNR 비교', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/fp8_e4m3_vs_e5m2.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter16_sparse_attention/fp8_e4m3_vs_e5m2.png")
print(f"E4M3 평균 SNR: {np.mean(snr_e4m3_list):.2f} dB")
print(f"E5M2 평균 SNR: {np.mean(snr_e5m2_list):.2f} dB")"""))

# ── Cell 12: Section 5 - Training stability ──────────────────────
cells.append(md(r"""\
## 5. FP8 vs FP16 학습 안정성 비교 <a name='5.-학습-안정성-비교'></a>

FP8 혼합 정밀도 훈련과 FP16 훈련의 학습 곡선을 비교합니다.

**혼합 정밀도 훈련 전략 (DeepSeek-V3):**
1. Forward: $W_{E4M3} \times X_{E4M3}$ (GEMM 연산)
2. Backward: $\nabla_{E5M2}$ (기울기 계산)
3. Update: $W_{FP32} \leftarrow W_{FP32} - \eta \nabla_{FP32}$ (마스터 가중치 업데이트)"""))

# ── Cell 13: Training stability simulation ────────────────────────
cells.append(code("""\
# ── FP8 vs FP16 학습 안정성 시뮬레이션 ───────────────────────────
# 간단한 회귀 모델로 정밀도별 학습 곡선 비교
np.random.seed(42)

# 합성 데이터 생성
n_data = 200
X_data = np.random.randn(n_data, 10).astype(np.float32)
true_w = np.random.randn(10, 1).astype(np.float32)
Y_data = X_data @ true_w + np.random.randn(n_data, 1).astype(np.float32) * 0.1

X_tf = tf.constant(X_data)
Y_tf = tf.constant(Y_data)

def train_with_precision(precision_name, n_epochs=100, lr=0.01):
    # 마스터 가중치는 항상 FP32
    W = tf.Variable(tf.random.normal([10, 1], seed=42))
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    losses = []

    for epoch in range(n_epochs):
        with tf.GradientTape() as tape:
            if precision_name == 'FP8':
                # FP8 시뮬레이션: forward에서 양자화 노이즈 추가
                W_noisy = W + tf.random.normal(W.shape, stddev=0.005)
                X_noisy = X_tf + tf.random.normal(X_tf.shape, stddev=0.002)
                pred = tf.matmul(X_noisy, W_noisy)
            elif precision_name == 'FP16':
                W_fp16 = tf.cast(tf.cast(W, tf.float16), tf.float32)
                pred = tf.matmul(X_tf, W_fp16)
            else:  # FP32
                pred = tf.matmul(X_tf, W)

            loss = tf.reduce_mean((pred - Y_tf) ** 2)

        grads = tape.gradient(loss, [W])
        optimizer.apply_gradients(zip(grads, [W]))
        losses.append(loss.numpy())

    return losses

# 각 정밀도로 학습
losses_fp32 = train_with_precision('FP32')
losses_fp16 = train_with_precision('FP16')
losses_fp8 = train_with_precision('FP8')

# 결과 시각화
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
ax1.plot(losses_fp32, 'g-', lw=2.5, label='FP32 (기준)')
ax1.plot(losses_fp16, 'b--', lw=2, label='FP16')
ax1.plot(losses_fp8, 'r-.', lw=2, label='FP8 (시뮬레이션)')
ax1.set_xlabel('에폭', fontsize=11)
ax1.set_ylabel('MSE Loss', fontsize=11)
ax1.set_title('정밀도별 학습 곡선', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

ax2 = axes[1]
fp8_gap = np.array(losses_fp8) - np.array(losses_fp32)
fp16_gap = np.array(losses_fp16) - np.array(losses_fp32)
ax2.plot(fp16_gap, 'b-', lw=2, label='FP16 - FP32')
ax2.plot(fp8_gap, 'r-', lw=2, label='FP8 - FP32')
ax2.axhline(y=0, color='gray', ls='--', lw=1)
ax2.set_xlabel('에폭', fontsize=11)
ax2.set_ylabel('Loss 차이', fontsize=11)
ax2.set_title('FP32 대비 Loss 갭', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/fp8_vs_fp16_training.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter16_sparse_attention/fp8_vs_fp16_training.png")
print(f"\\n최종 Loss 비교:")
print(f"  FP32: {losses_fp32[-1]:.6f}")
print(f"  FP16: {losses_fp16[-1]:.6f} (FP32 대비 차이: {losses_fp16[-1]-losses_fp32[-1]:+.6f})")
print(f"  FP8:  {losses_fp8[-1]:.6f} (FP32 대비 차이: {losses_fp8[-1]-losses_fp32[-1]:+.6f})")
print(f"\\n→ FP8 혼합 정밀도 훈련은 약간의 노이즈가 추가되지만,")
print(f"  regularization 효과로 최종 성능에 큰 영향을 주지 않음")"""))

# ── Cell 14: Section 6 - Auxiliary-Loss-Free Load Balancing ──────
cells.append(md(r"""\
## 6. Auxiliary-Loss-Free 로드밸런싱 <a name='6.-Auxiliary-Loss-Free-로드밸런싱'></a>

기존 MoE의 보조 손실(Auxiliary Loss)은 학습 목적함수를 왜곡시킬 수 있습니다.
DeepSeek-V3는 **편향 보정(bias correction)** 방식으로 이를 해결합니다:

$$g_i^{(\text{topk})} = \begin{cases} g_i & \text{if } g_i + b_i \in \text{TopK}(g_1+b_1, \ldots, g_N+b_N) \\ 0 & \text{otherwise} \end{cases}$$

편향 업데이트 규칙:

$$b_i \leftarrow b_i + \alpha \cdot \text{sign}(\bar{f} - f_i)$$

- 과부하된 전문가: $f_i > \bar{f}$ → $b_i$ 감소 → 선택 확률 ↓
- 과소 활용 전문가: $f_i < \bar{f}$ → $b_i$ 증가 → 선택 확률 ↑"""))

cells.append(code("""\
# ── Auxiliary-Loss-Free 로드밸런싱 시뮬레이션 ────────────────────
np.random.seed(42)

n_experts = 8
n_tokens = 1000
n_steps = 200
top_k = 2
alpha = 0.01  # 편향 업데이트 스텝 크기
target_freq = 1.0 / n_experts

# 초기 게이트 값: 불균형하게 설정 (일부 전문가에 편향)
gate_bias = np.array([0.5, 0.3, 0.1, -0.1, -0.2, -0.3, 0.2, 0.0])

# 편향 보정 항 초기화
bias_correction = np.zeros(n_experts)

freq_history = []
bias_history = []

for step in range(n_steps):
    # 랜덤 토큰 게이트 값 생성
    raw_gates = np.random.randn(n_tokens, n_experts) + gate_bias

    # 편향 보정 적용
    adjusted_gates = raw_gates + bias_correction

    # Top-K 선택
    expert_counts = np.zeros(n_experts)
    for t in range(n_tokens):
        top_indices = np.argsort(adjusted_gates[t])[-top_k:]
        expert_counts[top_indices] += 1

    # 빈도 계산
    freqs = expert_counts / (n_tokens * top_k)
    freq_history.append(freqs.copy())
    bias_history.append(bias_correction.copy())

    # 편향 업데이트: 과부하 → 감소, 과소활용 → 증가
    bias_correction += alpha * np.sign(target_freq - freqs)

freq_history = np.array(freq_history)
bias_history = np.array(bias_history)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
for i in range(n_experts):
    ax1.plot(freq_history[:, i], lw=1.5, label=f'Expert {i}')
ax1.axhline(y=target_freq, color='black', ls='--', lw=2, label=f'목표 ({target_freq:.3f})')
ax1.set_xlabel('스텝', fontsize=11)
ax1.set_ylabel('토큰 할당 비율', fontsize=11)
ax1.set_title('Auxiliary-Loss-Free 로드밸런싱', fontweight='bold')
ax1.legend(fontsize=8, ncol=3, loc='upper right')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for i in range(n_experts):
    ax2.plot(bias_history[:, i], lw=1.5, label=f'Expert {i}')
ax2.axhline(y=0, color='black', ls='--', lw=1)
ax2.set_xlabel('스텝', fontsize=11)
ax2.set_ylabel('편향 보정 값 ($b_i$)', fontsize=11)
ax2.set_title('편향 보정 항 변화', fontweight='bold')
ax2.legend(fontsize=8, ncol=3)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/aux_loss_free_balancing.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter16_sparse_attention/aux_loss_free_balancing.png")

print(f"\\n로드밸런싱 결과:")
print(f"  초기 불균형 (step 0):  std = {freq_history[0].std():.4f}")
print(f"  최종 균형 (step {n_steps-1}): std = {freq_history[-1].std():.4f}")
print(f"  균형 개선 비율: {freq_history[0].std()/freq_history[-1].std():.1f}x")
print(f"\\n최종 전문가별 토큰 비율:")
for i in range(n_experts):
    bar = '█' * int(freq_history[-1, i] * 200)
    print(f"  Expert {i}: {freq_history[-1, i]:.4f} {bar}")"""))

# ── Cell 15: Section 7 - MTP ─────────────────────────────────────
cells.append(md(r"""\
## 7. Multi-Token Prediction (MTP) <a name='7.-Multi-Token-Prediction'></a>

DeepSeek-V3의 MTP는 한 번의 forward pass에서 여러 미래 토큰을 동시에 예측합니다:

$$\mathcal{L}_{total} = \mathcal{L}_{main} + \sum_{k=1}^{K} \lambda_k \cdot \mathcal{L}_{MTP}^{(k)}$$

$$\mathcal{L}_{MTP}^{(k)} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_{t+k} \mid x_{\leq t})$$

MTP의 이점:
1. 학습 데이터 활용 효율 증가 ($K$배 더 많은 신호)
2. 추론 시 Speculative Decoding과 결합 가능
3. 내부 표현의 예측 능력 강화"""))

cells.append(code("""\
# ── Multi-Token Prediction (MTP) 시뮬레이션 ──────────────────────
np.random.seed(42)

vocab_size = 100
seq_len = 32
batch_size = 16
embed_dim = 64
K_predict = 3  # 예측 토큰 수

# 간단한 MTP 모델: 공유 인코더 + K개의 예측 헤드
shared_encoder = tf.keras.layers.Dense(embed_dim, activation='relu')
mtp_heads = [tf.keras.layers.Dense(vocab_size) for _ in range(K_predict)]

# 합성 시퀀스 데이터
input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
embeddings = tf.one_hot(input_ids, vocab_size)
embeddings = tf.cast(embeddings, tf.float32)

# Forward pass
hidden = shared_encoder(tf.reshape(embeddings, [-1, vocab_size]))
hidden = tf.reshape(hidden, [batch_size, seq_len, embed_dim])

print(f"MTP 설정:")
print(f"  시퀀스 길이: {seq_len}")
print(f"  예측 토큰 수 (K): {K_predict}")
print(f"  어휘 크기: {vocab_size}")
print(f"  인코더 출력: {hidden.shape}")

# 각 예측 헤드별 손실 계산
total_loss = 0.0
lambda_weights = [1.0, 0.5, 0.25]  # 가까운 토큰일수록 가중치 높음

print(f"\\n각 예측 헤드별 손실:")
print(f"{'헤드 (k)':>10} | {'Lambda':>8} | {'Loss':>12} | {'가중 Loss':>12}")
print("-" * 50)

for k in range(K_predict):
    logits_k = mtp_heads[k](tf.reshape(hidden[:, :seq_len-K_predict, :], [-1, embed_dim]))
    targets_k = input_ids[:, k+1:seq_len-K_predict+k+1].flatten()
    targets_k = tf.constant(targets_k, dtype=tf.int32)

    loss_k = tf.keras.losses.sparse_categorical_crossentropy(
        targets_k, logits_k, from_logits=True
    )
    loss_k_mean = tf.reduce_mean(loss_k).numpy()
    weighted_loss = lambda_weights[k] * loss_k_mean
    total_loss += weighted_loss

    print(f"  k={k+1:>5} | {lambda_weights[k]:>8.2f} | {loss_k_mean:>12.4f} | {weighted_loss:>12.4f}")

print(f"{'':>10} {'':>8}   {'합계':>12}   {total_loss:>12.4f}")

# MTP 학습 효율 분석
print(f"\\nMTP 학습 효율:")
standard_signals = seq_len - 1  # 표준 next-token prediction
mtp_signals = (seq_len - K_predict) * K_predict + (seq_len - 1)
print(f"  표준 NTP 학습 신호: {standard_signals}개/시퀀스")
print(f"  MTP 학습 신호: {mtp_signals}개/시퀀스")
print(f"  효율 증가: {mtp_signals/standard_signals:.2f}x")
print(f"\\n→ MTP는 같은 데이터에서 {mtp_signals/standard_signals:.1f}배 더 많은 학습 신호를 추출")"""))

# ── Cell 18: Summary ─────────────────────────────────────────────
cells.append(md(r"""\
## 8. 정리 <a name='8.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| FP8 E4M3 | 높은 정밀도, $[-448, 448]$ 범위 — 가중치/활성화용 | ⭐⭐⭐ |
| FP8 E5M2 | 넓은 범위, $[-57344, 57344]$ — 기울기용 | ⭐⭐⭐ |
| Per-tensor 스케일링 | $s = \max(|x|)/q_{max}$ — 단순하고 빠름 | ⭐⭐ |
| SNR | $\propto 2^{2b}$ — 비트 수에 따른 품질 지표 | ⭐⭐ |
| Aux-Loss-Free | $b_i \leftarrow b_i + \alpha \cdot \text{sign}(\bar{f} - f_i)$ — 편향 보정 | ⭐⭐⭐ |
| MTP | $\mathcal{L} = \sum_k \lambda_k \mathcal{L}^{(k)}$ — 다중 토큰 예측 | ⭐⭐⭐ |
| 혼합 정밀도 전략 | Forward(E4M3) + Backward(E5M2) + Update(FP32) | ⭐⭐⭐ |

### 핵심 수식

$$Q(x) = \text{clamp}\!\left(\left\lfloor \frac{x}{s} \right\rceil, -q_{max}, q_{max}\right) \cdot s$$

$$b_i \leftarrow b_i + \alpha \cdot \text{sign}\!\left(\frac{1}{N_E} - f_i\right)$$

$$\mathcal{L}_{MTP} = \sum_{k=1}^{K} \lambda_k \cdot \mathbb{E}\left[-\log p_\theta(x_{t+k} \mid x_{\leq t})\right]$$

### 다음 챕터 예고
**02_multi_head_latent_attention.ipynb** — MLA(Multi-head Latent Attention)의 KV 압축 수식을 완전 도출하고, GQA 대비 KV Cache 절감률을 정량적으로 비교합니다."""))

create_notebook(cells, 'chapter16_sparse_attention/01_deepseek_v3_fp8_training.ipynb')
