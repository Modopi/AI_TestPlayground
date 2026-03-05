#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ─────────────────────────────────────────────
cells.append(md(r"""# 실습 퀴즈: AWQ 양자화 파이프라인 시뮬레이션

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다

## 목차
- [Q1: 균일 INT8 양자화](#q1)
- [Q2: 양자화 오차 측정](#q2)
- [Q3: Salient Channel 탐지](#q3)
- [Q4: AWQ 스케일 팩터 최적화](#q4)
- [종합 도전: AWQ 양자화 파이프라인](#bonus)"""))

# ── Cell 2: Import ─────────────────────────────────────────────
cells.append(code(r"""# ── 환경 설정 ──────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
print(f"TensorFlow 버전: {tf.__version__}")"""))

# ── Cell 3: Q1 Problem ────────────────────────────────────────
cells.append(md(r"""## Q1: 균일 INT8 양자화 <a name='q1'></a>

### 문제

가중치 벡터 $w = [-1.5, -0.3, 0.0, 0.7, 1.2]$를 INT8($b=8$)로 균일 양자화합니다.

$$\Delta = \frac{x_{max} - x_{min}}{2^b - 1}, \quad x_q = \text{round}\left(\frac{x - x_{min}}{\Delta}\right)$$

1. 양자화 스텝 $\Delta = ?$
2. 각 값의 양자화 정수 $x_q = ?$
3. 역양자화 후 복원값 $\hat{x} = ?$
4. 최대 양자화 오차 $= ?$

**여러분의 예측:**
- $\Delta = ?$
- $x_q[-1.5] = ?$, $x_q[0.7] = ?$"""))

# ── Cell 4: Q1 Solution ───────────────────────────────────────
cells.append(code(r"""# ── Q1 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q1 풀이: 균일 INT8 양자화")
print("=" * 45)

w = np.array([-1.5, -0.3, 0.0, 0.7, 1.2])
bits = 8
n_levels = 2**bits - 1

x_min, x_max = w.min(), w.max()
delta = (x_max - x_min) / n_levels

x_q = np.round((w - x_min) / delta).astype(int)
x_q = np.clip(x_q, 0, n_levels)

x_hat = x_q * delta + x_min
error = np.abs(w - x_hat)

print(f"\n원본 가중치: {w}")
print(f"범위: [{x_min}, {x_max}]")
print(f"비트폭: {bits}, 레벨 수: {n_levels + 1}")
print(f"양자화 스텝 Δ = ({x_max} - ({x_min})) / {n_levels} = {delta:.6f}")

print(f"\n{'원본 w':>10} | {'x_q':>6} | {'복원 x_hat':>12} | {'오차':>10}")
print("-" * 48)
for i in range(len(w)):
    print(f"{w[i]:>10.4f} | {x_q[i]:>6} | {x_hat[i]:>12.6f} | {error[i]:>10.6f}")

print(f"\n최대 양자화 오차: {max(error):.6f}")
print(f"이론적 최대 오차 (Δ/2): {delta/2:.6f}")

print(f"\n[해설]")
print(f"  INT8은 256개 레벨로 나누므로 스텝이 매우 작습니다.")
print(f"  Δ = {delta:.6f} → 최대 오차 ≈ Δ/2 = {delta/2:.6f}")
print(f"  실용적으로 INT8은 FP16과 거의 동일한 정밀도를 제공합니다.")"""))

# ── Cell 5: Q2 Problem ────────────────────────────────────────
cells.append(md(r"""## Q2: 양자화 오차 측정 <a name='q2'></a>

### 문제

정규분포 $\mathcal{N}(0, 0.02^2)$를 따르는 가중치 행렬 $W \in \mathbb{R}^{256 \times 256}$에 대해:

1. INT8과 INT4 양자화를 각각 적용하세요
2. 각각의 MSE, SNR(dB), Cosine Similarity를 측정하세요
3. INT4→INT8 전환 시 SNR이 이론값(약 24dB)에 근접하는지 확인하세요

**여러분의 예측:** INT8 SNR `?` dB, INT4 SNR `?` dB"""))

# ── Cell 6: Q2 Solution ───────────────────────────────────────
cells.append(code(r"""# ── Q2 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q2 풀이: 양자화 오차 측정")
print("=" * 45)

np.random.seed(42)
W = np.random.randn(256, 256) * 0.02

def uniform_quantize(x, num_bits):
    x_min, x_max = np.min(x), np.max(x)
    n_levels = 2**num_bits - 1
    delta = (x_max - x_min) / n_levels
    x_q = np.round((x - x_min) / delta).astype(int)
    x_q = np.clip(x_q, 0, n_levels)
    x_hat = x_q * delta + x_min
    return x_hat, delta

signal_power = np.var(W)
W_flat = W.flatten()

print(f"가중치 행렬: {W.shape}, 표준편차={np.std(W):.4f}")
print(f"신호 전력: {signal_power:.8f}")

print(f"\n{'비트폭':>8} | {'Δ':>12} | {'MSE':>14} | {'SNR(dB)':>10} | {'Cos Sim':>10}")
print("-" * 65)

snr_values = {}
for bits in [2, 4, 6, 8, 16]:
    W_hat, delta = uniform_quantize(W, bits)
    mse = np.mean((W - W_hat)**2)
    snr_db = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')
    cos_sim = np.dot(W_flat, W_hat.flatten()) / (np.linalg.norm(W_flat) * np.linalg.norm(W_hat.flatten()))
    snr_values[bits] = snr_db
    print(f"{bits:>8} | {delta:>12.8f} | {mse:>14.10f} | {snr_db:>10.2f} | {cos_sim:>10.8f}")

print(f"\nSNR 차이 검증:")
snr_diff = snr_values[8] - snr_values[4]
print(f"  INT4→INT8: {snr_diff:.2f} dB (이론: {6.02*4:.2f} dB)")
print(f"  이론 대비 오차: {abs(snr_diff - 6.02*4):.2f} dB")

print(f"\n[해설]")
print(f"  비트당 약 6dB 증가 법칙이 근사적으로 성립합니다.")
print(f"  완벽한 균일분포가 아닌 정규분포이므로 약간의 차이가 있습니다.")"""))

# ── Cell 7: Q3 Problem ────────────────────────────────────────
cells.append(md(r"""## Q3: Salient Channel 탐지 <a name='q3'></a>

### 문제

Llama와 유사한 설정에서 활성화 기반으로 중요 채널을 탐지합니다:

- 히든 차원: 512
- 캘리브레이션 데이터: 32 시퀀스 × 512 차원
- 5개 채널에 인위적으로 큰 활성화 삽입

1. 상위 1% 활성화 크기를 기준으로 salient channel을 탐지하세요
2. 탐지된 채널이 실제 salient channel과 일치하는지 확인하세요
3. Salient channel만 INT8로, 나머지는 INT4로 양자화하면 정밀도가 어떻게 되나요?

**여러분의 예측:** salient channel 탐지 정확도 `?`%"""))

# ── Cell 8: Q3 Solution ───────────────────────────────────────
cells.append(code(r"""# ── Q3 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q3 풀이: Salient Channel 탐지")
print("=" * 45)

np.random.seed(42)
hidden = 512
cal_seqs = 32
W = np.random.randn(hidden, hidden) * 0.02
X = np.random.randn(cal_seqs, hidden) * 0.5

true_salient = [25, 100, 250, 400, 480]
for ch in true_salient:
    X[:, ch] *= 20.0

act_magnitude = np.max(np.abs(X), axis=0)
threshold_1pct = np.percentile(act_magnitude, 99)
detected = np.where(act_magnitude > threshold_1pct)[0]

precision = len(set(detected) & set(true_salient)) / len(detected) if len(detected) > 0 else 0
recall = len(set(detected) & set(true_salient)) / len(true_salient)

print(f"히든 차원: {hidden}")
print(f"99th percentile 임계값: {threshold_1pct:.4f}")
print(f"탐지된 salient 채널: {detected.tolist()}")
print(f"실제 salient 채널: {true_salient}")
print(f"Precision: {precision:.1%}, Recall: {recall:.1%}")

# 혼합 정밀도 양자화 실험
Y_ref = X @ W.T

# 방법 1: 전체 INT4
W_q4, _ = uniform_quantize(W, 4)
Y_int4 = X @ W_q4.T
mse_int4 = np.mean((Y_ref - Y_int4)**2)

# 방법 2: 전체 INT8
W_q8, _ = uniform_quantize(W, 8)
Y_int8 = X @ W_q8.T
mse_int8 = np.mean((Y_ref - Y_int8)**2)

# 방법 3: 혼합 (salient=INT8, 나머지=INT4)
W_mixed = np.zeros_like(W)
salient_mask = np.zeros(hidden, dtype=bool)
salient_mask[detected] = True

for j in range(hidden):
    if salient_mask[j]:
        col_q, _ = uniform_quantize(W[:, j], 8)
    else:
        col_q, _ = uniform_quantize(W[:, j], 4)
    W_mixed[:, j] = col_q

Y_mixed = X @ W_mixed.T
mse_mixed = np.mean((Y_ref - Y_mixed)**2)

print(f"\n{'방법':<25} | {'MSE':>15} | {'상대 오차':>12}")
print("-" * 58)
print(f"{'전체 INT4':<25} | {mse_int4:>15.10f} | {1.0:>12.4f}x")
print(f"{'혼합 (salient=8, else=4)':<25} | {mse_mixed:>15.10f} | {mse_mixed/mse_int4:>12.4f}x")
print(f"{'전체 INT8':<25} | {mse_int8:>15.10f} | {mse_int8/mse_int4:>12.4f}x")

print(f"\n[해설]")
print(f"  Salient channel은 전체의 {len(detected)/hidden:.1%}에 불과하지만")
print(f"  이들만 INT8로 보호해도 MSE가 {(1-mse_mixed/mse_int4)*100:.1f}% 감소합니다.")
print(f"  AWQ는 이 원리를 스케일링으로 더 우아하게 구현합니다.")"""))

# ── Cell 9: Q4 Problem ────────────────────────────────────────
cells.append(md(r"""## Q4: AWQ 스케일 팩터 최적화 <a name='q4'></a>

### 문제

AWQ의 채널별 스케일 팩터 $S_j$를 최적화합니다:

$$S_j = \left(\frac{\max|X_j|}{\max|W_j|}\right)^\alpha, \quad \alpha \in [0, 1]$$

1. $\alpha = 0, 0.25, 0.5, 0.75, 1.0$에 대해 양자화 후 출력 MSE를 측정하세요
2. 최적 $\alpha$를 찾으세요
3. $\alpha = 0$과 $\alpha = 1$의 의미를 설명하세요

**여러분의 예측:** 최적 $\alpha \approx ?$"""))

# ── Cell 10: Q4 Solution ──────────────────────────────────────
cells.append(code(r"""# ── Q4 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q4 풀이: AWQ 스케일 팩터 최적화")
print("=" * 45)

np.random.seed(42)
hidden = 256
W = np.random.randn(hidden, hidden) * 0.02
X = np.random.randn(64, hidden) * 0.5

salient_chs = [15, 80, 150, 200]
for ch in salient_chs:
    X[:, ch] *= 15.0

Y_ref = X @ W.T

act_mag = np.max(np.abs(X), axis=0)
w_mag = np.max(np.abs(W), axis=0)

alphas = np.linspace(0, 1, 21)
mse_per_alpha = []

for alpha in alphas:
    S = np.where(w_mag > 1e-10, (act_mag / np.maximum(w_mag, 1e-10))**alpha, 1.0)
    W_scaled = W * S[np.newaxis, :]
    W_q, _ = uniform_quantize(W_scaled, 4)
    X_descaled = X / S[np.newaxis, :]
    Y_awq = X_descaled @ W_q.T
    mse = np.mean((Y_ref - Y_awq)**2)
    mse_per_alpha.append(mse)

best_idx = np.argmin(mse_per_alpha)
best_alpha = alphas[best_idx]
best_mse = mse_per_alpha[best_idx]

# Naive 양자화 (alpha=0 → S=1)
W_q_naive, _ = uniform_quantize(W, 4)
Y_naive = X @ W_q_naive.T
mse_naive = np.mean((Y_ref - Y_naive)**2)

print(f"\n{'alpha':>8} | {'MSE':>15} | {'상대 오차':>12}")
print("-" * 42)
for alpha_show in [0.0, 0.25, 0.5, 0.75, 1.0]:
    idx = int(alpha_show * 20)
    mse_val = mse_per_alpha[idx]
    print(f"{alpha_show:>8.2f} | {mse_val:>15.10f} | {mse_val/mse_naive:>12.4f}x")

print(f"\n최적 alpha: {best_alpha:.2f} (MSE = {best_mse:.10f})")
print(f"개선률: {(1 - best_mse/mse_naive)*100:.1f}% (Naive INT4 대비)")

print(f"\n[해설]")
print(f"  α=0: S_j=1 → 스케일링 없음 (Naive 양자화와 동일)")
print(f"  α=1: S_j = max|X_j|/max|W_j| → 활성화에만 의존")
print(f"  최적 α≈{best_alpha:.2f}: 활성화와 가중치를 균형 있게 고려")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(alphas, mse_per_alpha, 'b-o', lw=2, ms=5, label='AWQ INT4 MSE')
ax.axhline(y=mse_naive, color='red', ls='--', lw=2, label=f'Naive INT4 (α=0): {mse_naive:.6f}')
ax.axvline(x=best_alpha, color='green', ls=':', lw=2, label=f'최적 α={best_alpha:.2f}: {best_mse:.6f}')
ax.scatter([best_alpha], [best_mse], color='green', s=100, zorder=5)
ax.set_xlabel('α (스케일 팩터 지수)', fontsize=11)
ax.set_ylabel('출력 MSE', fontsize=11)
ax.set_title('AWQ α 하이퍼파라미터 vs 출력 MSE', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chapter14_extreme_inference/practice/q4_alpha_optimization.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n그래프 저장됨: chapter14_extreme_inference/practice/q4_alpha_optimization.png")"""))

# ── Cell 11: Bonus Problem ────────────────────────────────────
cells.append(md(r"""## 종합 도전: AWQ 양자화 파이프라인 시뮬레이션 <a name='bonus'></a>

### 미니 프로젝트

Llama-like 모델의 가중치를 AWQ 방식으로 W4A16 양자화하고, Perplexity 및 속도를 평가하는 전체 파이프라인을 구현하세요:

1. 캘리브레이션 데이터로 채널별 활성화 통계 수집
2. Salient channel 탐지 및 스케일 팩터 계산
3. 스케일 적용 후 INT4 양자화
4. Perplexity 시뮬레이션 (출력 분포 왜곡 측정)
5. 메모리 절약 및 속도 향상 추정"""))

# ── Cell 12: Bonus Solution ───────────────────────────────────
cells.append(code(r"""# ── 종합 도전 풀이: AWQ 양자화 파이프라인 ──────────────────────
print("=" * 50)
print("종합 도전 풀이: AWQ W4A16 양자화 파이프라인")
print("=" * 50)

np.random.seed(42)

# 1. 모델 설정 (소형 Transformer 레이어)
hidden_dim = 512
ffn_dim = 2048
num_layers = 4

class MiniTransformerLayer:
    def __init__(self, hidden, ffn):
        self.W_q = np.random.randn(hidden, hidden) * 0.02
        self.W_k = np.random.randn(hidden, hidden) * 0.02
        self.W_v = np.random.randn(hidden, hidden) * 0.02
        self.W_o = np.random.randn(hidden, hidden) * 0.02
        self.W_up = np.random.randn(ffn, hidden) * 0.02
        self.W_down = np.random.randn(hidden, ffn) * 0.02
        self.weight_names = ['W_q', 'W_k', 'W_v', 'W_o', 'W_up', 'W_down']
        self.weights = [self.W_q, self.W_k, self.W_v, self.W_o, self.W_up, self.W_down]

layers = [MiniTransformerLayer(hidden_dim, ffn_dim) for _ in range(num_layers)]

total_params = sum(w.size for layer in layers for w in layer.weights)
print(f"\n1. 모델 설정")
print(f"  히든: {hidden_dim}, FFN: {ffn_dim}, 레이어: {num_layers}")
print(f"  총 파라미터: {total_params:,} ({total_params * 2 / 1e6:.1f} MB @ FP16)")

# 2. 캘리브레이션 데이터
cal_data = np.random.randn(32, hidden_dim) * 0.5
salient_real = [30, 100, 250, 400, 470]
for ch in salient_real:
    cal_data[:, ch] *= 20.0

act_stats = np.max(np.abs(cal_data), axis=0)
threshold = np.percentile(act_stats, 99)
detected_salient = np.where(act_stats > threshold)[0]

print(f"\n2. 캘리브레이션 분석")
print(f"  샘플: {cal_data.shape[0]}개, 차원: {cal_data.shape[1]}")
print(f"  탐지된 salient 채널: {len(detected_salient)}개")

# 3. AWQ 양자화 적용
alpha_optimal = 0.5
results_by_method = {"FP16": [], "Naive INT4": [], "AWQ INT4": []}

for layer_idx, layer in enumerate(layers):
    for w_name, W in zip(layer.weight_names, layer.weights):
        # 입력 차원에 대한 스케일 계산
        in_dim = W.shape[1]
        if in_dim == hidden_dim:
            act_for_scale = act_stats
        else:
            act_for_scale = np.random.rand(in_dim) * 2

        w_mag = np.max(np.abs(W), axis=0)
        S = np.where(w_mag > 1e-10, (act_for_scale[:in_dim] / np.maximum(w_mag, 1e-10))**alpha_optimal, 1.0)

        # FP16 기준
        Y_ref = cal_data[:, :in_dim] @ W.T if in_dim == hidden_dim else np.random.randn(32, W.shape[0])

        # Naive INT4
        W_q_naive, _ = uniform_quantize(W, 4)
        Y_naive = cal_data[:, :in_dim] @ W_q_naive.T if in_dim == hidden_dim else np.random.randn(32, W.shape[0])
        mse_naive = np.mean((Y_ref - Y_naive)**2) if in_dim == hidden_dim else 0

        # AWQ INT4
        W_scaled = W * S[np.newaxis, :]
        W_q_awq, _ = uniform_quantize(W_scaled, 4)
        if in_dim == hidden_dim:
            X_descaled = cal_data[:, :in_dim] / S[np.newaxis, :]
            Y_awq = X_descaled @ W_q_awq.T
            mse_awq = np.mean((Y_ref - Y_awq)**2)
        else:
            mse_awq = 0

        results_by_method["Naive INT4"].append(mse_naive)
        results_by_method["AWQ INT4"].append(mse_awq)

avg_mse_naive = np.mean(results_by_method["Naive INT4"])
avg_mse_awq = np.mean(results_by_method["AWQ INT4"])

print(f"\n3. 양자화 결과")
print(f"  Naive INT4 평균 MSE: {avg_mse_naive:.8f}")
print(f"  AWQ INT4 평균 MSE:   {avg_mse_awq:.8f}")
print(f"  AWQ 개선률: {(1 - avg_mse_awq/avg_mse_naive)*100:.1f}%")

# 4. Perplexity 시뮬레이션
ppl_fp16 = 5.50
ppl_naive_int4 = ppl_fp16 * (1 + avg_mse_naive * 1e4)
ppl_awq_int4 = ppl_fp16 * (1 + avg_mse_awq * 1e4)

print(f"\n4. Perplexity 추정 (시뮬레이션)")
print(f"  FP16:       {ppl_fp16:.2f}")
print(f"  Naive INT4: {ppl_naive_int4:.2f} (+{ppl_naive_int4-ppl_fp16:.2f})")
print(f"  AWQ INT4:   {ppl_awq_int4:.2f} (+{ppl_awq_int4-ppl_fp16:.2f})")

# 5. 메모리 및 속도 추정
mem_fp16 = total_params * 2 / 1e9
mem_int4 = total_params * 0.5 / 1e9
speedup_est = mem_fp16 / mem_int4

print(f"\n5. 메모리 및 속도")
print(f"  FP16 메모리:  {mem_fp16:.3f} GB")
print(f"  INT4 메모리:  {mem_int4:.3f} GB")
print(f"  압축률: {mem_fp16/mem_int4:.1f}x")
print(f"  추론 속도 향상 (메모리 바운드): ~{speedup_est:.1f}x")

# 종합 비교 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Perplexity 비교
ax1 = axes[0]
methods = ['FP16', 'Naive\nINT4', 'AWQ\nINT4']
ppls = [ppl_fp16, ppl_naive_int4, ppl_awq_int4]
colors = ['#4C72B0', '#C44E52', '#55A868']
bars = ax1.bar(methods, ppls, color=colors, edgecolor='black', width=0.5)
ax1.set_ylabel('Perplexity', fontsize=11)
ax1.set_title('Perplexity 비교', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for bar, ppl in zip(bars, ppls):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{ppl:.2f}', ha='center', fontsize=10, fontweight='bold')

# 메모리 비교
ax2 = axes[1]
mems = [mem_fp16, mem_int4, mem_int4]
bars2 = ax2.bar(methods, mems, color=colors, edgecolor='black', width=0.5)
ax2.set_ylabel('메모리 (GB)', fontsize=11)
ax2.set_title('메모리 사용량', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for bar, m in zip(bars2, mems):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{m:.3f}', ha='center', fontsize=10, fontweight='bold')

# 레이어별 MSE 분포
ax3 = axes[2]
ax3.boxplot([results_by_method["Naive INT4"], results_by_method["AWQ INT4"]],
            labels=['Naive INT4', 'AWQ INT4'])
ax3.set_ylabel('레이어 MSE', fontsize=11)
ax3.set_title('레이어별 MSE 분포', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chapter14_extreme_inference/practice/bonus_awq_pipeline.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n그래프 저장됨: chapter14_extreme_inference/practice/bonus_awq_pipeline.png")
print("\nAWQ 양자화 파이프라인 시뮬레이션 완료!")"""))

path = '/workspace/chapter14_extreme_inference/practice/ex02_awq_quantization_eval.ipynb'
create_notebook(cells, path)
