"""Generate Chapter 13-02: Noise Schedules and Samplers."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helper import md, code, create_notebook

cells = [
# ─── Cell 0: 헤더 ───
md(r"""# Chapter 13: 생성 AI 심화 — 노이즈 스케줄과 샘플러

## 학습 목표
- **Linear, Cosine, EDM** 세 가지 노이즈 스케줄의 수학적 차이를 이해하고 $\bar{\alpha}_t$ 곡선을 비교한다
- Cosine 스케줄이 Linear 대비 **낮은 타임스텝에서 정보 보존을 개선**하는 원리를 파악한다
- DDIM(비마르코프 역방향)의 수식을 유도하고 **DDPM 대비 가속 샘플링**이 가능한 이유를 이해한다
- DPM-Solver++의 **고차(higher-order) ODE 솔버** 원리를 개념적으로 파악한다
- 샘플링 **스텝 수와 생성 품질** 사이의 트레이드오프를 실험적으로 확인한다

## 목차
1. [수학적 기초: 노이즈 스케줄과 DDIM](#1.-수학적-기초)
2. [3가지 스케줄 비교 구현](#2.-스케줄-비교)
3. [DDPM vs DDIM 샘플링 경로 비교](#3.-샘플링-경로-비교)
4. [스텝 수 vs 품질 트레이드오프 분석](#4.-스텝-수-분석)
5. [정리](#5.-정리)"""),

# ─── Cell 1: 수학적 기초 ───
md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### Linear 스케줄

DDPM 원논문(Ho et al., 2020)의 기본 스케줄입니다:

$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$$

- $\beta_{\min} = 10^{-4}$, $\beta_{\max} = 0.02$ (원논문 설정)
- $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$

**문제점**: $t$가 작을 때 $\bar{\alpha}_t$가 너무 빠르게 감소하여 저해상도 구조 정보가 일찍 소실됩니다.

### Cosine 스케줄

Nichol & Dhariwal (2021)이 제안한 개선된 스케줄입니다:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

- $s = 0.008$: 오프셋 상수 ($t=0$에서 $\bar{\alpha}_0 \approx 1$이 되도록)
- 코사인 함수의 특성상 **초기에는 천천히, 후반에 급격하게** $\bar{\alpha}_t$가 감소
- $\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$로 역산, $\beta_t$를 0.999 이하로 클리핑

**핵심 장점**: 저해상도 구조 정보(얼굴 윤곽, 물체 형태)를 더 오래 보존 → 생성 품질 향상

### EDM (Elucidating the Design Space) 스케줄

Karras et al. (2022)의 통합 프레임워크에서 제안한 노이즈 레벨:

$$\sigma(t) = t, \quad t \in [\sigma_{\min}, \sigma_{\max}]$$

EDM에서는 $\bar{\alpha}_t$와 $\sigma_t$ 대신 **연속적 노이즈 레벨** $\sigma$를 직접 사용합니다:

$$\bar{\alpha}_t^{EDM} = \frac{1}{1 + \sigma_t^2}, \quad \sigma_t = \sqrt{\frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}}$$

로그 균일 분포로 $\sigma$를 샘플링: $\ln\sigma \sim \mathcal{U}[\ln\sigma_{\min}, \ln\sigma_{\max}]$

### DDIM (Denoising Diffusion Implicit Models)

Song et al. (2020; arxiv 2010.02502)의 **비마르코프 역방향 과정**:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t, t) + \sigma_t\,\epsilon$$

여기서 $\hat{x}_0$는 **예측된 원본**:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

| 파라미터 | $\sigma_t$ 설정 | 특성 |
|----------|-----------------|------|
| DDPM | $\sigma_t = \sqrt{\tilde{\beta}_t}$ | 확률적, 다양한 샘플 |
| DDIM ($\eta=0$) | $\sigma_t = 0$ | **결정론적**, 같은 $x_T$ → 같은 $x_0$ |
| DDIM ($\eta=1$) | $\sigma_t = \sqrt{\tilde{\beta}_t}$ | DDPM과 동일 |

**DDIM의 핵심 장점**: $\sigma_t = 0$일 때 **임의의 서브시퀀스**로 건너뛸 수 있음 → 1000스텝 → 50스텝으로 가속!

### DPM-Solver++ 개요

Lu et al. (2022)의 고속 ODE 솔버:

$$x_{t_{i-1}} = \frac{\alpha_{t_{i-1}}}{\alpha_{t_i}} x_{t_i} - \alpha_{t_{i-1}} \sum_{n=0}^{k-1} h_n^{(k)}(\lambda_{t_{i-1}}, \lambda_{t_i}) \epsilon_\theta^{(n)}$$

- $k$차 다중 스텝 방법(1차=Euler, 2차=Midpoint)
- 10~25 스텝으로 DDIM 50스텝에 필적하는 품질 달성

**요약 표:**

| 스케줄/샘플러 | 수식 핵심 | 장점 |
|--------------|-----------|------|
| Linear | $\beta_t$ 선형 증가 | 구현 단순 |
| Cosine | $\bar{\alpha}_t = \cos^2(\cdot)$ | 초기 정보 보존 |
| EDM | $\sigma$로 통합 매개변수화 | 이론적 최적 |
| DDIM | $\sigma_t=0$ 결정론적 | 10~50x 가속 |
| DPM-Solver++ | $k$차 ODE 솔버 | 10~25 스텝 고품질 |"""),

# ─── Cell 2: 🐣 친절 설명 ───
md(r"""---

### 🐣 초등학생을 위한 노이즈 스케줄과 샘플러 친절 설명!

#### 🔢 노이즈 스케줄이 뭔가요?

> 💡 **비유**: 수채화를 물에 담가서 점점 번지게 만든다고 생각해 보세요.
> - **Linear 스케줄**: 처음부터 물에 **일정한 속도로** 담가요 → 초반에 그림이 너무 빨리 사라져요 😢
> - **Cosine 스케줄**: 처음에는 **살짝만** 담그고, 나중에 **확** 담가요 → 그림의 큰 형태가 오래 남아서 나중에 복원하기 쉬워요! ✨

#### ⚡ DDIM은 왜 빠른가요?

| 방법 | 비유 | 스텝 수 |
|------|------|---------|
| DDPM | 계단을 **한 칸씩** 1000번 내려가기 | 1000 |
| DDIM | 엘리베이터로 **10층씩** 건너뛰기 | 50~100 |
| DPM-Solver++ | 더 똑똑한 엘리베이터 (**곡선** 예측) | 10~25 |

DDIM은 "중간 층을 건너뛰어도 목적지가 같다"는 수학적 성질을 이용해요!"""),

# ─── Cell 3: 📝 연습 문제 ───
md(r"""---

### 📝 연습 문제

#### 문제 1: Cosine 스케줄 계산

$T=1000$, $s=0.008$일 때, $t=0$과 $t=500$에서 $f(t)$와 $\bar{\alpha}_t$를 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$$f(0) = \cos^2\!\left(\frac{0/1000 + 0.008}{1.008} \cdot \frac{\pi}{2}\right) = \cos^2(0.00794 \cdot \frac{\pi}{2}) = \cos^2(0.01247) \approx 0.99984$$

$$f(500) = \cos^2\!\left(\frac{0.5 + 0.008}{1.008} \cdot \frac{\pi}{2}\right) = \cos^2(0.5040 \cdot \frac{\pi}{2}) = \cos^2(0.7917) \approx 0.4988$$

$$\bar{\alpha}_{500} = \frac{f(500)}{f(0)} = \frac{0.4988}{0.99984} \approx 0.499$$

중간 시점에서 원본이 약 **50% 보존** — Linear보다 훨씬 많이 남아 있습니다!
</details>

#### 문제 2: DDIM 가속 원리

DDIM에서 $\sigma_t=0$이면 왜 서브시퀀스로 건너뛸 수 있나요?

<details>
<summary>💡 풀이 확인</summary>

$\sigma_t=0$이면 DDIM 업데이트가 **결정론적**이 됩니다:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\,\epsilon_\theta(x_t, t)$$

이 식에서 $t-1$은 **반드시 연속일 필요가 없습니다**. $t$에서 임의의 $t'$($t' < t$)로 직접 점프 가능합니다.
이는 DDIM이 마르코프 가정을 사용하지 않기 때문입니다. $\hat{x}_0$를 통해 항상 **원본 예측**을 경유하므로, 
중간 시점을 건너뛰어도 수학적으로 유효합니다.
</details>"""),

# ─── Cell 4: 임포트 ───
code("""# ── 라이브러리 임포트 ──────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")"""),

# ─── Cell 5: Section 2 - 스케줄 비교 ───
md(r"""## 2. 3가지 스케줄 비교 구현 <a name='2.-스케줄-비교'></a>

Linear, Cosine, EDM 스케줄의 $\bar{\alpha}_t$ 곡선과 $\beta_t$ 분포를 비교합니다."""),

# ─── Cell 6: 스케줄 비교 코드 ───
code(r"""# ── Linear / Cosine / EDM 스케줄 비교 ──────────────────────────────
T = 1000

# --- Linear Schedule ---
beta_lin = np.linspace(1e-4, 0.02, T)
alpha_lin = 1.0 - beta_lin
abar_lin = np.cumprod(alpha_lin)

# --- Cosine Schedule ---
s = 0.008
steps = np.arange(T + 1, dtype=np.float64)
f_t = np.cos(((steps / T) + s) / (1 + s) * (np.pi / 2)) ** 2
abar_cos = f_t[1:] / f_t[0]
abar_cos = np.clip(abar_cos, 1e-8, 1.0)
beta_cos = 1.0 - abar_cos / np.concatenate([[1.0], abar_cos[:-1]])
beta_cos = np.clip(beta_cos, 0, 0.999)

# --- EDM Schedule (log-uniform sigma → alpha_bar) ---
sigma_min, sigma_max = 0.002, 80.0
log_sigmas = np.linspace(np.log(sigma_min), np.log(sigma_max), T)
sigmas_edm = np.exp(log_sigmas)
abar_edm = 1.0 / (1.0 + sigmas_edm ** 2)

# 수치 비교
print(f"{'시점':>6} | {'Linear ᾱ':>12} | {'Cosine ᾱ':>12} | {'EDM ᾱ':>12}")
print("-" * 55)
for t in [1, 100, 250, 500, 750, 1000]:
    idx = t - 1
    print(f"  t={t:4d} | {abar_lin[idx]:>12.6f} | {abar_cos[idx]:>12.6f} | {abar_edm[idx]:>12.6f}")

print(f"\n초기 정보 보존 (t=100):")
print(f"  Linear: {abar_lin[99]*100:.2f}%")
print(f"  Cosine: {abar_cos[99]*100:.2f}%")
print(f"  EDM:    {abar_edm[99]*100:.2f}%")"""),

# ─── Cell 7: 스케줄 시각화 ───
code(r"""# ── 스케줄 비교 시각화 ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
t_axis = np.arange(1, T + 1)

# (1) alpha_bar 비교
ax1 = axes[0]
ax1.plot(t_axis, abar_lin, 'b-', lw=2, label='Linear')
ax1.plot(t_axis, abar_cos, 'r-', lw=2, label='Cosine')
ax1.plot(t_axis, abar_edm, 'g-', lw=2, label='EDM')
ax1.set_xlabel('Timestep $t$', fontsize=11)
ax1.set_ylabel(r'$\bar{\alpha}_t$', fontsize=11)
ax1.set_title(r'$\bar{\alpha}_t$ Comparison', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) beta_t 비교
ax2 = axes[1]
ax2.plot(t_axis, beta_lin, 'b-', lw=1.5, alpha=0.7, label='Linear')
ax2.plot(t_axis, beta_cos, 'r-', lw=1.5, alpha=0.7, label='Cosine')
ax2.set_xlabel('Timestep $t$', fontsize=11)
ax2.set_ylabel(r'$\beta_t$', fontsize=11)
ax2.set_title(r'$\beta_t$ Comparison', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# (3) SNR 비교 (로그 스케일)
snr_lin = abar_lin / (1 - abar_lin + 1e-10)
snr_cos = abar_cos / (1 - abar_cos + 1e-10)
snr_edm = abar_edm / (1 - abar_edm + 1e-10)

ax3 = axes[2]
ax3.semilogy(t_axis, snr_lin, 'b-', lw=2, label='Linear')
ax3.semilogy(t_axis, snr_cos, 'r-', lw=2, label='Cosine')
ax3.semilogy(t_axis, snr_edm, 'g-', lw=2, label='EDM')
ax3.set_xlabel('Timestep $t$', fontsize=11)
ax3.set_ylabel('SNR (log scale)', fontsize=11)
ax3.set_title('Signal-to-Noise Ratio', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/schedule_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/schedule_comparison.png")"""),

# ─── Cell 8: Section 3 - 샘플링 경로 비교 ───
md(r"""## 3. DDPM vs DDIM 샘플링 경로 비교 <a name='3.-샘플링-경로-비교'></a>

동일한 학습된 노이즈 예측기를 가정하고, **DDPM(확률적)** vs **DDIM(결정론적)** 역방향 경로를 시각화합니다.

**시뮬레이션 설정**: 사전 학습된 모델 대신, 실제 노이즈 $\epsilon$을 아는 "오라클 모델"로 역방향 과정을 시연합니다.

$$\text{DDPM}: x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right) + \sqrt{\beta_t}\,z$$

$$\text{DDIM}: x_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat{x}_0 + \sqrt{1-\bar\alpha_{t-1}}\epsilon_\theta$$"""),

# ─── Cell 9: 샘플링 경로 코드 ───
code(r"""# ── DDPM vs DDIM 샘플링 경로 시각화 ────────────────────────────────
T_demo = 100
beta_demo = np.linspace(1e-4, 0.02, T_demo).astype(np.float64)
alpha_demo = 1.0 - beta_demo
abar_demo = np.cumprod(alpha_demo)

# 원본 데이터 (1D)
x0_true = 3.0
eps_true = np.random.randn()

# Forward: x_T 생성
x_T = np.sqrt(abar_demo[-1]) * x0_true + np.sqrt(1 - abar_demo[-1]) * eps_true

# ─── DDPM Reverse (확률적) ───
n_trials = 5
ddpm_paths = []
for trial in range(n_trials):
    path = [x_T]
    x = x_T
    for t in reversed(range(T_demo)):
        abar_t = abar_demo[t]
        abar_prev = abar_demo[t-1] if t > 0 else 1.0
        a_t = alpha_demo[t]
        b_t = beta_demo[t]

        # 오라클: 실제 eps 사용
        x0_pred = (x - np.sqrt(1 - abar_t) * eps_true) / np.sqrt(abar_t)
        mu = (1 / np.sqrt(a_t)) * (x - (b_t / np.sqrt(1 - abar_t)) * eps_true)

        if t > 0:
            z = np.random.randn()
            x = mu + np.sqrt(b_t) * z
        else:
            x = mu
        path.append(x)
    ddpm_paths.append(path)

# ─── DDIM Reverse (결정론적, eta=0) ───
ddim_path = [x_T]
x = x_T
for t in reversed(range(T_demo)):
    abar_t = abar_demo[t]
    abar_prev = abar_demo[t-1] if t > 0 else 1.0

    # 오라클: x0 예측
    x0_pred = (x - np.sqrt(1 - abar_t) * eps_true) / np.sqrt(abar_t)
    # DDIM (sigma=0)
    x = np.sqrt(abar_prev) * x0_pred + np.sqrt(1 - abar_prev) * eps_true
    ddim_path.append(x)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
t_plot = list(range(T_demo, -1, -1))

# (1) DDPM (여러 경로)
ax1 = axes[0]
for i, path in enumerate(ddpm_paths):
    alpha_val = 0.3 if i > 0 else 0.8
    lw = 1 if i > 0 else 2
    ax1.plot(t_plot, path, lw=lw, alpha=alpha_val, label=f'Trial {i+1}' if i < 3 else None)
ax1.axhline(y=x0_true, color='red', ls='--', lw=2, label=f'$x_0$ = {x0_true:.1f}')
ax1.set_xlabel('Reverse Timestep (T → 0)', fontsize=11)
ax1.set_ylabel('$x$ value', fontsize=11)
ax1.set_title('DDPM (확률적) — 매번 다른 경로', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) DDIM (단일 결정론적 경로)
ax2 = axes[1]
ax2.plot(t_plot, ddim_path, 'g-', lw=2.5, label='DDIM ($\\eta=0$)')
ax2.axhline(y=x0_true, color='red', ls='--', lw=2, label=f'$x_0$ = {x0_true:.1f}')
ax2.set_xlabel('Reverse Timestep (T → 0)', fontsize=11)
ax2.set_ylabel('$x$ value', fontsize=11)
ax2.set_title('DDIM (결정론적) — 항상 같은 경로', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/ddpm_vs_ddim_path.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/ddpm_vs_ddim_path.png")

# 수치 비교
print(f"\n원본 x_0 = {x0_true:.4f}")
print(f"시작 x_T = {x_T:.4f}")
print(f"\nDDPM 최종값 (5 trials):")
for i, path in enumerate(ddpm_paths):
    print(f"  Trial {i+1}: {path[-1]:.4f} (오차: {abs(path[-1]-x0_true):.4f})")
print(f"\nDDIM 최종값: {ddim_path[-1]:.4f} (오차: {abs(ddim_path[-1]-x0_true):.6f})")"""),

# ─── Cell 10: Section 4 - 스텝 수 분석 ───
md(r"""## 4. 스텝 수 vs 품질 트레이드오프 분석 <a name='4.-스텝-수-분석'></a>

DDIM의 핵심 장점인 **서브시퀀스 샘플링**을 구현합니다. 전체 $T=1000$ 스텝 대신 균일하게 선택된 서브셋을 사용합니다.

$$\text{서브시퀀스 } \tau = (\tau_1, \tau_2, \ldots, \tau_S), \quad S \ll T$$"""),

# ─── Cell 11: 스텝 수 분석 코드 ───
code(r"""# ── DDIM 서브시퀀스 샘플링: 스텝 수별 복원 정확도 비교 ────────────
T_full = 1000
beta_full = np.linspace(1e-4, 0.02, T_full).astype(np.float64)
alpha_full = 1.0 - beta_full
abar_full = np.cumprod(alpha_full)

# 1D 가우시안 혼합에서 원본 데이터 생성
n_test = 500
np.random.seed(123)
x0_data = np.concatenate([
    np.random.randn(n_test // 2) * 0.5 + 3.0,
    np.random.randn(n_test // 2) * 0.5 - 3.0
])
eps_data = np.random.randn(n_test)
x_T_data = np.sqrt(abar_full[-1]) * x0_data + np.sqrt(1 - abar_full[-1]) * eps_data

# DDIM 서브시퀀스 역방향 (오라클 모델)
def ddim_reverse_subsequence(x_T, x0, eps, abar, steps):
    # 균일 서브시퀀스 생성
    if steps >= len(abar):
        indices = list(range(len(abar)))
    else:
        indices = np.linspace(0, len(abar) - 1, steps, dtype=int).tolist()

    x = x_T.copy()
    for i in reversed(range(len(indices))):
        t_idx = indices[i]
        abar_t = abar[t_idx]
        abar_prev = abar[indices[i-1]] if i > 0 else 1.0

        x0_pred = (x - np.sqrt(1 - abar_t) * eps) / np.sqrt(abar_t)
        x = np.sqrt(abar_prev) * x0_pred + np.sqrt(1 - abar_prev) * eps

    return x

# 다양한 스텝 수에서 실험
step_counts = [5, 10, 25, 50, 100, 250, 500, 1000]
mse_results = {}

for n_steps in step_counts:
    x_gen = ddim_reverse_subsequence(x_T_data, x0_data, eps_data, abar_full, n_steps)
    mse = np.mean((x_gen - x0_data) ** 2)
    mse_results[n_steps] = mse

print(f"DDIM 서브시퀀스 샘플링 — 스텝 수별 복원 MSE (오라클 모델)")
print(f"{'스텝 수':>8} | {'MSE':>12} | {'RMSE':>12} | {'상대 품질':>10}")
print("-" * 50)
base_mse = mse_results[1000]
for n_steps, mse in mse_results.items():
    rmse = np.sqrt(mse)
    rel = mse / (base_mse + 1e-15)
    print(f"  {n_steps:6d} | {mse:12.8f} | {rmse:12.6f} | {rel:10.4f}x")"""),

# ─── Cell 12: 스텝 수 시각화 ───
code(r"""# ── 스텝 수 vs 품질 시각화 ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

steps_list = list(mse_results.keys())
mse_list = list(mse_results.values())

# (1) MSE vs 스텝 수
ax1 = axes[0]
ax1.plot(steps_list, mse_list, 'bo-', lw=2, ms=8)
ax1.set_xscale('log')
ax1.set_xlabel('Sampling Steps (log scale)', fontsize=11)
ax1.set_ylabel('MSE', fontsize=11)
ax1.set_title('DDIM: Steps vs Reconstruction MSE', fontweight='bold')
ax1.grid(True, alpha=0.3)
for s, m in zip(steps_list, mse_list):
    ax1.annotate(f'{s}', (s, m), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=8)

# (2) 속도-품질 트레이드오프 (스텝 수 대비 상대 속도)
ax2 = axes[1]
speedup = [1000 / s for s in steps_list]
quality = [1.0 / (m + 1e-15) for m in mse_list]
quality_norm = [q / max(quality) for q in quality]

ax2.plot(speedup, quality_norm, 'rs-', lw=2, ms=8)
ax2.set_xlabel('Speedup (vs 1000 steps)', fontsize=11)
ax2.set_ylabel('Relative Quality (normalized)', fontsize=11)
ax2.set_title('Speed-Quality Tradeoff', fontweight='bold')
ax2.grid(True, alpha=0.3)
for sp, q, s in zip(speedup, quality_norm, steps_list):
    ax2.annotate(f'{s} steps', (sp, q), textcoords="offset points",
                xytext=(5, 5), fontsize=8)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/steps_vs_quality.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/steps_vs_quality.png")

print(f"\n핵심 결론:")
print(f"  50 스텝: 1000 스텝 대비 20x 빠르면서 MSE = {mse_results[50]:.8f}")
print(f"  25 스텝: 1000 스텝 대비 40x 빠르면서 MSE = {mse_results[25]:.8f}")
print(f"  10 스텝: 1000 스텝 대비 100x 빠르면서 MSE = {mse_results[10]:.8f}")"""),

# ─── Cell 13: 정리 ───
md(r"""## 5. 정리 <a name='5.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| Linear 스케줄 | $\beta_t$ 선형 증가 — 초기 정보 소실 빠름 | ⭐⭐ |
| Cosine 스케줄 | $\bar{\alpha}_t = \cos^2(\cdot)$ — 초기 정보 보존 우수 | ⭐⭐⭐ |
| EDM 스케줄 | 연속 $\sigma$ 매개변수화 — 통합 프레임워크 | ⭐⭐ |
| DDIM | 비마르코프 결정론적 역방향 — 10~50x 가속 | ⭐⭐⭐ |
| DPM-Solver++ | 고차 ODE 솔버 — 10~25 스텝 고품질 | ⭐⭐⭐ |
| 스텝-품질 트레이드오프 | 적은 스텝으로도 충분한 품질 달성 가능 | ⭐⭐⭐ |

### 핵심 수식

$$\text{Cosine}: \bar{\alpha}_t = \frac{\cos^2\!\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right)}{\cos^2\!\left(\frac{s}{1+s}\cdot\frac{\pi}{2}\right)}$$

$$\text{DDIM}: x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\epsilon_\theta + \sigma_t\epsilon$$

### 다음 챕터 예고
**03_unet_for_diffusion** — DDPM에서 노이즈를 예측하는 핵심 아키텍처인 UNet의 잔차 블록, 시간 임베딩, Cross-Attention 구조를 구현합니다."""),
]

if __name__ == '__main__':
    create_notebook(cells, 'chapter13_genai_diffusion/02_noise_schedules_and_samplers.ipynb')
