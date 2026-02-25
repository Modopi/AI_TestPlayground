#!/usr/bin/env python3
"""Generate chapter13_genai_diffusion/practice/ex01_ddpm_forward_reverse.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ─────────────────────────────────────────────
cells.append(md(r"""# 실습 퀴즈: DDPM Forward/Reverse 시뮬레이션

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다
- Linear/Cosine 스케줄로 노이즈를 입히고(Forward), DDIM Sampler로 제거하는(Reverse) 시뮬레이션을 비교합니다

## 목차
- [Q1: $\bar\alpha_t$ 스케줄 계산](#q1)
- [Q2: Forward Process 노이즈 시각화](#q2)
- [Q3: DDPM Reverse Step 구현](#q3)
- [Q4: DDIM Reverse Step 구현](#q4)
- [종합 도전: Full Forward + Reverse Pipeline](#bonus)"""))

# ── Cell 2: Imports ──────────────────────────────────────────────
cells.append(code(r"""# ── 라이브러리 임포트 ──────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"NumPy 버전: {np.__version__}")"""))

# ── Cell 3: Q1 Problem ──────────────────────────────────────────
cells.append(md(r"""---

## Q1: $\bar\alpha_t$ 스케줄 계산 <a name='q1'></a>

### 문제

**Linear 스케줄**과 **Cosine 스케줄**에서 $\bar\alpha_t$ (누적 신호 보존률)를 계산하세요.

**Linear 스케줄:**
$$\beta_t = \beta_{min} + \frac{t}{T}(\beta_{max} - \beta_{min}), \quad \alpha_t = 1 - \beta_t, \quad \bar\alpha_t = \prod_{s=1}^{t} \alpha_s$$

- $\beta_{min} = 0.0001$, $\beta_{max} = 0.02$, $T = 1000$

**Cosine 스케줄 (Nichol & Dhariwal, 2021):**
$$\bar\alpha_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right), \quad s = 0.008$$

**여러분의 예측:**
- $t = 250$일 때, Linear $\bar\alpha_{250}$은 약 `?`이고, Cosine $\bar\alpha_{250}$은 약 `?`입니다
- 어느 스케줄이 중간 구간에서 더 많은 신호를 보존할까요? `?`"""))

# ── Cell 4: Q1 Solution ──────────────────────────────────────────
cells.append(code(r"""# ── Q1 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q1 풀이: alpha_bar 스케줄 계산")
print("=" * 45)

T = 1000

# Linear 스케줄
beta_min, beta_max = 0.0001, 0.02
betas_linear = np.linspace(beta_min, beta_max, T)
alphas_linear = 1.0 - betas_linear
alpha_bars_linear = np.cumprod(alphas_linear)

# Cosine 스케줄
s = 0.008
t_steps = np.arange(T + 1)
f_t = np.cos(((t_steps / T) + s) / (1 + s) * np.pi / 2) ** 2
alpha_bars_cosine = f_t[1:] / f_t[0]
alpha_bars_cosine = np.clip(alpha_bars_cosine, 1e-5, 1.0)

# 비교 출력
print(f"\n{'시점 t':<10} | {'Linear ᾱ_t':>14} | {'Cosine ᾱ_t':>14} | {'차이':>10}")
print("-" * 55)
for t_check in [0, 100, 250, 500, 750, 999]:
    lin_val = alpha_bars_linear[t_check]
    cos_val = alpha_bars_cosine[t_check]
    print(f"t = {t_check:<5} | {lin_val:>14.6f} | {cos_val:>14.6f} | {cos_val - lin_val:>+10.4f}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(range(T), alpha_bars_linear, 'b-', lw=2.5, label='Linear')
ax1.plot(range(T), alpha_bars_cosine, 'r-', lw=2.5, label='Cosine')
ax1.axhline(y=0.5, color='gray', ls='--', lw=1, alpha=0.5)
ax1.set_xlabel('Timestep t', fontsize=11)
ax1.set_ylabel(r'$\bar\alpha_t$', fontsize=11)
ax1.set_title(r'$\bar\alpha_t$ 비교 (신호 보존률)', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
snr_linear = alpha_bars_linear / (1 - alpha_bars_linear + 1e-8)
snr_cosine = alpha_bars_cosine / (1 - alpha_bars_cosine + 1e-8)
ax2.plot(range(T), np.log10(snr_linear + 1e-8), 'b-', lw=2.5, label='Linear')
ax2.plot(range(T), np.log10(snr_cosine + 1e-8), 'r-', lw=2.5, label='Cosine')
ax2.set_xlabel('Timestep t', fontsize=11)
ax2.set_ylabel(r'$\log_{10}$ SNR', fontsize=11)
ax2.set_title('SNR 비교 (높을수록 신호 우세)', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/practice/q1_alpha_bar_schedules.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n그래프 저장됨: chapter13_genai_diffusion/practice/q1_alpha_bar_schedules.png")

print("\n[해설]")
print(f"  t=250: Linear ᾱ={alpha_bars_linear[250]:.4f}, Cosine ᾱ={alpha_bars_cosine[250]:.4f}")
print("  Cosine 스케줄이 중간 구간에서 더 많은 신호를 보존합니다.")
print("  Linear는 초기에 급격히 감쇄 → 정보 손실이 빠름")
print("  Cosine은 완만하게 감쇄 → 학습 안정성 우수")"""))

# ── Cell 5: Q2 Problem ──────────────────────────────────────────
cells.append(md(r"""---

## Q2: Forward Process 노이즈 시각화 <a name='q2'></a>

### 문제

2D 가우시안 데이터 $x_0 \sim \mathcal{N}(\mu, \sigma^2 I)$에 Forward process를 적용합니다:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

$t = [0, 250, 500, 750, 999]$에서의 데이터 분포를 시각화하세요.

**여러분의 예측:**
- $t = 500$에서 데이터의 원래 구조가 남아있을까요? `?`
- Linear와 Cosine 스케줄 중 어느 것이 $t=500$에서 구조를 더 보존할까요? `?`"""))

# ── Cell 6: Q2 Solution ──────────────────────────────────────────
cells.append(code(r"""# ── Q2 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q2 풀이: Forward Process 노이즈 시각화")
print("=" * 45)

# 2D 데이터 생성 (원 형태)
n_samples = 500
np.random.seed(42)
theta = np.random.uniform(0, 2 * np.pi, n_samples)
radius = 2.0 + np.random.randn(n_samples) * 0.15
x0 = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1).astype(np.float32)

timesteps_to_show = [0, 250, 500, 750, 999]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for row, (schedule_name, alpha_bars) in enumerate([
    ('Linear', alpha_bars_linear),
    ('Cosine', alpha_bars_cosine)
]):
    for col, t in enumerate(timesteps_to_show):
        ax = axes[row, col]
        if t == 0:
            x_t = x0
        else:
            ab = alpha_bars[t]
            eps = np.random.randn(*x0.shape).astype(np.float32)
            x_t = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps

        ax.scatter(x_t[:, 0], x_t[:, 1], s=3, alpha=0.4, c='blue')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ab_val = alpha_bars[t] if t > 0 else 1.0
        ax.set_title(f't={t}\n' + r'$\bar\alpha$=' + f'{ab_val:.4f}', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(f'{schedule_name}', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

plt.suptitle('Forward Process: Linear vs Cosine 스케줄', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/practice/q2_forward_noise.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/practice/q2_forward_noise.png")

print("\n[해설]")
print("  t=500에서:")
print(f"    Linear ᾱ = {alpha_bars_linear[500]:.4f} → 원래 구조 거의 없음")
print(f"    Cosine ᾱ = {alpha_bars_cosine[500]:.4f} → 약간의 구조 남음")
print("  Cosine 스케줄이 중간 시점에서 더 많은 정보를 보존합니다.")"""))

# ── Cell 7: Q3 Problem ──────────────────────────────────────────
cells.append(md(r"""---

## Q3: DDPM Reverse Step 구현 <a name='q3'></a>

### 문제

DDPM의 reverse step을 구현하세요:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

여기서 $\sigma_t = \sqrt{\beta_t}$이고 $z \sim \mathcal{N}(0, I)$ (마지막 스텝 제외).

간단한 MLP 노이즈 예측 모델을 사용하여 구현하세요.

**여러분의 예측:** DDPM reverse는 $T = 1000$ 스텝이 모두 필요한가요? `?`"""))

# ── Cell 8: Q3 Solution ──────────────────────────────────────────
cells.append(code(r"""# ── Q3 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q3 풀이: DDPM Reverse Step 구현")
print("=" * 45)

# 1D 데이터에서 DDPM reverse 시뮬레이션
T_ddpm = 200
beta_min_ddpm, beta_max_ddpm = 1e-4, 0.02
betas_ddpm = np.linspace(beta_min_ddpm, beta_max_ddpm, T_ddpm).astype(np.float32)
alphas_ddpm = 1.0 - betas_ddpm
alpha_bars_ddpm = np.cumprod(alphas_ddpm)

# 타겟 분포: 가우시안 혼합
target_means = np.array([-2.0, 2.0])
target_stds = np.array([0.4, 0.4])

# Oracle 노이즈 예측 (정답을 아는 경우)
def oracle_noise_predictor(x_t, t_idx, alpha_bars):
    ab_t = alpha_bars[t_idx]
    weighted_score = np.zeros_like(x_t)
    for mu, std in zip(target_means, target_stds):
        effective_mu = np.sqrt(ab_t) * mu
        effective_var = ab_t * std**2 + (1 - ab_t)
        weighted_score += np.exp(-0.5 * (x_t - effective_mu)**2 / effective_var) / np.sqrt(effective_var)
    total = weighted_score + 1e-10
    score = np.zeros_like(x_t)
    for mu, std in zip(target_means, target_stds):
        effective_mu = np.sqrt(ab_t) * mu
        effective_var = ab_t * std**2 + (1 - ab_t)
        w = np.exp(-0.5 * (x_t - effective_mu)**2 / effective_var) / np.sqrt(effective_var) / total
        score += w * (-(x_t - effective_mu) / effective_var)
    eps = -np.sqrt(1 - ab_t) * score
    return eps

# DDPM Reverse Process
def ddpm_reverse(n_samples, betas, alphas, alpha_bars, noise_pred_fn):
    T = len(betas)
    x = np.random.randn(n_samples).astype(np.float32)
    trajectory = [x.copy()]
    for t in reversed(range(T)):
        eps_pred = noise_pred_fn(x, t, alpha_bars)
        mean = (1.0 / np.sqrt(alphas[t])) * (x - betas[t] / np.sqrt(1 - alpha_bars[t]) * eps_pred)
        if t > 0:
            sigma = np.sqrt(betas[t])
            z = np.random.randn(n_samples).astype(np.float32)
            x = mean + sigma * z
        else:
            x = mean
        if t % (T // 10) == 0 or t == 0:
            trajectory.append(x.copy())
    return x, trajectory

np.random.seed(42)
n_test = 2000
x_final, traj = ddpm_reverse(n_test, betas_ddpm, alphas_ddpm, alpha_bars_ddpm, oracle_noise_predictor)

# 시각화
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(x_final, bins=80, density=True, alpha=0.6, color='blue', label='DDPM 생성 샘플')
x_plot = np.linspace(-5, 5, 300)
for mu, std in zip(target_means, target_stds):
    p = 0.5 * np.exp(-0.5 * ((x_plot - mu) / std)**2) / (std * np.sqrt(2 * np.pi))
    ax.plot(x_plot, p, 'r-', lw=2)
ax.plot([], [], 'r-', lw=2, label='타겟 분포')
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('밀도', fontsize=11)
ax.set_title(f'DDPM Reverse ({T_ddpm} steps)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/practice/q3_ddpm_reverse.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/practice/q3_ddpm_reverse.png")

print(f"\n생성 샘플 통계:")
print(f"  평균: {np.mean(x_final):.3f}")
print(f"  표준편차: {np.std(x_final):.3f}")
print(f"  왼쪽 모드(x<0) 비율: {np.mean(x_final < 0) * 100:.1f}%")
print(f"  오른쪽 모드(x>0) 비율: {np.mean(x_final > 0) * 100:.1f}%")
print("\n[해설]")
print(f"  DDPM은 {T_ddpm} 스텝 전체가 필요합니다 (각 스텝에서 약간의 노이즈 제거).")
print("  DDIM은 동일한 결과를 더 적은 스텝으로 달성할 수 있습니다.")"""))

# ── Cell 9: Q4 Problem ──────────────────────────────────────────
cells.append(md(r"""---

## Q4: DDIM Reverse Step 구현 <a name='q4'></a>

### 문제

DDIM sampler를 구현하세요 ($\sigma_t = 0$, 결정론적):

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar\alpha_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

여기서 $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\cdot\epsilon_\theta}{\sqrt{\bar\alpha_t}}$

**DDIM의 핵심**: 서브시퀀스 $\tau = [0, 50, 100, \ldots]$만 사용하여 **스텝 수를 크게 줄일 수 있습니다**.

**여러분의 예측:** 200스텝 DDPM과 20스텝 DDIM의 결과 차이는? `?`"""))

# ── Cell 10: Q4 Solution ─────────────────────────────────────────
cells.append(code(r"""# ── Q4 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q4 풀이: DDIM Reverse Step 구현")
print("=" * 45)

def ddim_reverse(n_samples, alpha_bars_full, noise_pred_fn, n_ddim_steps=20):
    T_full = len(alpha_bars_full)
    # 서브시퀀스 생성
    step_indices = np.linspace(0, T_full - 1, n_ddim_steps + 1, dtype=int)
    step_indices = step_indices[::-1]

    x = np.random.randn(n_samples).astype(np.float32)
    trajectory = [x.copy()]

    for i in range(len(step_indices) - 1):
        t_curr = step_indices[i]
        t_prev = step_indices[i + 1]

        ab_curr = alpha_bars_full[t_curr]
        ab_prev = alpha_bars_full[t_prev]

        eps_pred = noise_pred_fn(x, t_curr, alpha_bars_full)

        # x0 예측
        x0_pred = (x - np.sqrt(1 - ab_curr) * eps_pred) / np.sqrt(ab_curr)
        x0_pred = np.clip(x0_pred, -5, 5)

        # DDIM (sigma=0, 결정론적)
        x = np.sqrt(ab_prev) * x0_pred + np.sqrt(1 - ab_prev) * eps_pred

        trajectory.append(x.copy())

    return x, trajectory

# 다양한 스텝 수에서 DDIM 실행
ddim_step_counts = [10, 20, 50, 100, 200]
results = {}

for n_steps in ddim_step_counts:
    np.random.seed(42)
    x_ddim, _ = ddim_reverse(n_test, alpha_bars_ddpm, oracle_noise_predictor, n_ddim_steps=n_steps)
    results[n_steps] = x_ddim

# 비교 시각화
fig, axes = plt.subplots(1, len(ddim_step_counts), figsize=(20, 4))

for idx, n_steps in enumerate(ddim_step_counts):
    ax = axes[idx]
    ax.hist(results[n_steps], bins=60, density=True, alpha=0.6, color='green')
    for mu, std in zip(target_means, target_stds):
        p = 0.5 * np.exp(-0.5 * ((x_plot - mu) / std)**2) / (std * np.sqrt(2 * np.pi))
        ax.plot(x_plot, p, 'r-', lw=2)
    ax.set_title(f'DDIM {n_steps} steps', fontweight='bold', fontsize=11)
    ax.set_xlim(-5, 5)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.set_ylabel('밀도', fontsize=11)

plt.suptitle('DDIM: 스텝 수에 따른 생성 품질', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/practice/q4_ddim_steps.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/practice/q4_ddim_steps.png")

# 정량적 비교
print(f"\n{'스텝 수':<10} | {'평균':>8} | {'표준편차':>8} | {'좌측모드%':>10} | {'우측모드%':>10}")
print("-" * 55)
for n_steps in ddim_step_counts:
    data = results[n_steps]
    print(f"{n_steps:<10} | {np.mean(data):>8.3f} | {np.std(data):>8.3f} | {np.mean(data<0)*100:>9.1f}% | {np.mean(data>0)*100:>9.1f}%")

print("\n[해설]")
print("  DDIM 20스텝으로도 200스텝 DDPM과 비슷한 품질 달성!")
print("  DDIM은 결정론적(σ=0)이므로 같은 초기 노이즈 → 같은 결과")
print("  10x 스피드업이 가능한 것이 DDIM의 핵심 장점")"""))

# ── Cell 11: Bonus Problem ───────────────────────────────────────
cells.append(md(r"""---

## 종합 도전: Full Forward + Reverse Pipeline <a name='bonus'></a>

### 미니 구현

2D 원형 데이터에 대해:
1. **Forward**: Linear & Cosine 스케줄로 노이즈 추가
2. **Reverse**: DDPM(전체 스텝) vs DDIM(20스텝)으로 복원
3. 4가지 조합의 결과를 시각화하고 비교하세요"""))

# ── Cell 12: Bonus Solution ──────────────────────────────────────
cells.append(code(r"""# ── 종합 도전 풀이 ──────────────────────────────────────────────
print("=" * 45)
print("종합 도전: Forward + Reverse 파이프라인")
print("=" * 45)

# 2D oracle score/noise predictor
def oracle_2d_noise(x_t, t_idx, alpha_bars, target_data):
    ab_t = alpha_bars[t_idx]
    mean_x0 = np.mean(target_data, axis=0)
    cov_x0 = np.cov(target_data.T)
    eff_mu = np.sqrt(ab_t) * mean_x0
    eff_cov = ab_t * cov_x0 + (1 - ab_t) * np.eye(2)
    inv_cov = np.linalg.inv(eff_cov)
    diff = x_t - eff_mu
    score = -diff @ inv_cov.T
    eps = -np.sqrt(1 - ab_t) * score
    return eps

def ddpm_reverse_2d(n_samples, betas, alphas, alpha_bars, noise_fn, target_data):
    T = len(betas)
    x = np.random.randn(n_samples, 2).astype(np.float32)
    for t in reversed(range(T)):
        eps = noise_fn(x, t, alpha_bars, target_data)
        mean = (1.0 / np.sqrt(alphas[t])) * (x - betas[t] / np.sqrt(1 - alpha_bars[t]) * eps)
        if t > 0:
            x = mean + np.sqrt(betas[t]) * np.random.randn(n_samples, 2).astype(np.float32)
        else:
            x = mean
    return x

def ddim_reverse_2d(n_samples, alpha_bars_full, noise_fn, target_data, n_steps=20):
    T_full = len(alpha_bars_full)
    indices = np.linspace(0, T_full - 1, n_steps + 1, dtype=int)[::-1]
    x = np.random.randn(n_samples, 2).astype(np.float32)
    for i in range(len(indices) - 1):
        tc, tp = indices[i], indices[i+1]
        abc, abp = alpha_bars_full[tc], alpha_bars_full[tp]
        eps = noise_fn(x, tc, alpha_bars_full, target_data)
        x0_hat = (x - np.sqrt(1 - abc) * eps) / np.sqrt(abc)
        x0_hat = np.clip(x0_hat, -6, 6)
        x = np.sqrt(abp) * x0_hat + np.sqrt(1 - abp) * eps
    return x

# 2D 원형 데이터
n_data = 300
theta_data = np.random.uniform(0, 2 * np.pi, n_data)
r_data = 2.0 + np.random.randn(n_data) * 0.15
x0_2d = np.stack([r_data * np.cos(theta_data), r_data * np.sin(theta_data)], axis=1).astype(np.float32)

# 스케줄 계산 (T=200)
T_test = 200
betas_lin = np.linspace(1e-4, 0.02, T_test).astype(np.float32)
alphas_lin = 1.0 - betas_lin
ab_lin = np.cumprod(alphas_lin)

s_cos = 0.008
t_cos = np.arange(T_test + 1)
f_cos = np.cos(((t_cos / T_test) + s_cos) / (1 + s_cos) * np.pi / 2) ** 2
ab_cos = (f_cos[1:] / f_cos[0]).astype(np.float32)
ab_cos = np.clip(ab_cos, 1e-5, 1.0)
betas_cos = np.clip(1.0 - ab_cos / np.concatenate([[1.0], ab_cos[:-1]]), 1e-5, 0.999).astype(np.float32)
alphas_cos = 1.0 - betas_cos

# 4가지 조합
configs = [
    ('Linear + DDPM (200)', 'ddpm', betas_lin, alphas_lin, ab_lin),
    ('Linear + DDIM (20)', 'ddim', betas_lin, alphas_lin, ab_lin),
    ('Cosine + DDPM (200)', 'ddpm', betas_cos, alphas_cos, ab_cos),
    ('Cosine + DDIM (20)', 'ddim', betas_cos, alphas_cos, ab_cos),
]

fig, axes = plt.subplots(1, 5, figsize=(22, 4))

# 원본 데이터
axes[0].scatter(x0_2d[:, 0], x0_2d[:, 1], s=5, alpha=0.5, c='blue')
axes[0].set_title('원본 데이터', fontweight='bold')
axes[0].set_xlim(-4, 4)
axes[0].set_ylim(-4, 4)
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

for idx, (name, method, betas_cfg, alphas_cfg, ab_cfg) in enumerate(configs):
    np.random.seed(42)
    if method == 'ddpm':
        x_gen = ddpm_reverse_2d(n_data, betas_cfg, alphas_cfg, ab_cfg, oracle_2d_noise, x0_2d)
    else:
        x_gen = ddim_reverse_2d(n_data, ab_cfg, oracle_2d_noise, x0_2d, n_steps=20)

    ax = axes[idx + 1]
    ax.scatter(x_gen[:, 0], x_gen[:, 1], s=5, alpha=0.5, c='green')
    ax.set_title(name, fontweight='bold', fontsize=9)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle('Forward→Reverse 파이프라인 비교', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/practice/bonus_full_pipeline.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/practice/bonus_full_pipeline.png")

print("\n[결론]")
print("  1. Cosine 스케줄이 Linear보다 안정적인 생성 결과를 보여줍니다")
print("  2. DDIM 20스텝은 DDPM 200스텝과 유사한 품질 (10x 빠름)")
print("  3. 실제 Stable Diffusion은 DDIM/DPM-Solver++를 기본 사용합니다")"""))

# ── Create notebook ──────────────────────────────────────────────
path = '/workspace/chapter13_genai_diffusion/practice/ex01_ddpm_forward_reverse.ipynb'
create_notebook(cells, path)
