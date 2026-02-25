#!/usr/bin/env python3
"""Generate chapter13_genai_diffusion/05_score_matching_and_sde.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ─────────────────────────────────────────────
cells.append(md(r"""# Chapter 13: 생성 AI 심화 — Score Matching과 SDE

## 학습 목표
- **Score function** $\nabla_x \log p(x)$의 정의와 기하학적 의미를 이해한다
- **Denoising Score Matching**을 통해 스코어 함수를 효율적으로 학습하는 방법을 유도한다
- **Langevin Dynamics**로 스코어 함수를 사용한 샘플링 원리를 구현한다
- 확산 모델을 **연속 시간 SDE/ODE** 프레임워크(Song et al., 2021)로 통합하여 이해한다
- **VE-SDE**와 **VP-SDE**의 차이점을 수식과 코드로 비교한다

## 목차
1. [수학적 기초: Score Function, Langevin Dynamics, SDE](#1.-수학적-기초)
2. [2D Score Field 시각화](#2.-Score-Field-시각화)
3. [Langevin Dynamics 샘플링](#3.-Langevin-Dynamics-샘플링)
4. [VE-SDE vs VP-SDE 비교](#4.-VE-SDE-vs-VP-SDE)
5. [Probability Flow ODE](#5.-Probability-Flow-ODE)
6. [정리](#6.-정리)"""))

# ── Cell 2: Math Section ─────────────────────────────────────────
cells.append(md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### Score Function (스코어 함수)

데이터 분포 $p(x)$의 **스코어 함수**는 로그 확률 밀도의 그래디언트입니다:

$$\boxed{s(x) = \nabla_x \log p(x)}$$

- $s(x) \in \mathbb{R}^d$: 데이터 공간에서의 벡터장 (각 점에서 확률이 높은 방향을 가리킴)
- $\nabla_x$: 데이터 $x$에 대한 그래디언트
- $p(x)$를 직접 알 필요 없이 **스코어만으로** 샘플링 가능

**기하학적 의미**: 스코어 벡터는 데이터의 **확률 밀도가 높아지는 방향**을 가리킵니다.

### Score Matching (Hyvärinen, 2005)

분포 $p(x)$를 모르는 상태에서 스코어를 학습하는 목적함수:

$$J_{SM}(\theta) = \mathbb{E}_{p(x)}\!\left[\frac{1}{2}\|s_\theta(x) - \nabla_x \log p(x)\|^2\right]$$

이를 부분적분으로 변환하면 ($p(x)$ 없이 계산 가능):

$$J_{SM}(\theta) = \mathbb{E}_{p(x)}\!\left[\text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2}\|s_\theta(x)\|^2\right] + C$$

### Denoising Score Matching (DSM)

실제로는 **노이즈가 추가된 데이터**에서 스코어를 학습합니다:

$$q_\sigma(x) = \int p(y)\,\mathcal{N}(x;\, y,\, \sigma^2 I)\,dy$$

$$J_{DSM}(\theta) = \mathbb{E}_{p(y)}\mathbb{E}_{q_\sigma(x|y)}\!\left[\left\|s_\theta(x) - \nabla_x \log q_\sigma(x|y)\right\|^2\right]$$

가우시안 노이즈의 경우:

$$\nabla_x \log q_\sigma(x|y) = -\frac{x - y}{\sigma^2} = -\frac{\epsilon}{\sigma}$$

따라서 **DDPM의 Simple Loss와 본질적으로 동일**합니다!

### Langevin Dynamics (랑주뱅 역학)

스코어 함수를 사용하여 $p(x)$에서 샘플을 추출하는 반복적 과정:

$$\boxed{x_{t+1} = x_t + \frac{\delta}{2}\nabla_x \log p(x_t) + \sqrt{\delta}\,\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)}$$

- $\delta > 0$: 스텝 크기 (step size)
- $\frac{\delta}{2}\nabla_x \log p(x_t)$: 확률이 높은 방향으로의 드리프트
- $\sqrt{\delta}\,\epsilon_t$: 확률적 탐색을 위한 노이즈
- $\delta \to 0$, $t \to \infty$일 때 $x_t \sim p(x)$로 수렴

### SDE 프레임워크 (Song et al., 2021)

확산 모델을 **확률 미분 방정식(SDE)**으로 통합:

**Forward SDE** (데이터 → 노이즈):

$$\boxed{dx = f(x, t)\,dt + g(t)\,dw}$$

- $f(x, t)$: 드리프트 계수 (drift coefficient)
- $g(t)$: 확산 계수 (diffusion coefficient)
- $w$: 표준 위너 프로세스 (Brownian motion)

**Reverse SDE** (노이즈 → 데이터):

$$dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\,d\bar{w}$$

| SDE 유형 | $f(x,t)$ | $g(t)$ | 대응 모델 |
|----------|----------|--------|-----------|
| VP-SDE | $-\frac{1}{2}\beta(t)x$ | $\sqrt{\beta(t)}$ | DDPM |
| VE-SDE | $0$ | $\sqrt{\frac{d[\sigma^2(t)]}{dt}}$ | NCSN/SMLD |
| sub-VP SDE | $-\frac{1}{2}\beta(t)x$ | $\sqrt{\beta(t)(1-e^{-2\int_0^t\beta(s)ds})}$ | 개선된 VP |

### Probability Flow ODE

SDE의 **결정론적(deterministic)** 버전으로, 동일한 주변 분포를 생성합니다:

$$\boxed{\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)}$$

- 결정론적 → 같은 초기값에서 항상 같은 결과
- 정확한 로그 우도 계산 가능 (연속 정규화 흐름과 연결)
- DDIM은 이 ODE의 이산화로 해석 가능

**요약 표:**

| 개념 | 수식 | 핵심 |
|------|------|------|
| Score Function | $s(x) = \nabla_x \log p(x)$ | 확률 증가 방향 |
| Langevin Dynamics | $x' = x + \frac{\delta}{2}s(x) + \sqrt{\delta}\epsilon$ | 스코어 기반 샘플링 |
| Forward SDE | $dx = f\,dt + g\,dw$ | 데이터 → 노이즈 |
| Reverse SDE | $dx = [f - g^2 s_\theta]\,dt + g\,d\bar{w}$ | 노이즈 → 데이터 |
| Prob. Flow ODE | $dx/dt = f - \frac{1}{2}g^2 s_\theta$ | 결정론적 역방향 |"""))

# ── Cell 3: 🐣 Friendly Explanation ──────────────────────────────
cells.append(md(r"""---

### 🐣 초등학생을 위한 Score Function과 SDE 친절 설명!

#### 🧭 Score Function이 뭔가요?

> 💡 **비유**: 산에서 내려오는 물의 흐름을 생각해보세요!

산 위에 물을 뿌리면, 물은 **가장 낮은 곳(골짜기)**을 향해 흘러갑니다. Score function은 바로 이 **물이 흘러가는 방향**을 알려주는 화살표입니다.

- 🏔️ **높은 확률 = 깊은 골짜기**: 데이터가 많이 모여있는 곳
- ➡️ **Score 화살표**: 골짜기(데이터)를 향해 가리킴
- 🎲 **Langevin Dynamics**: 공을 굴리면서 약간 흔들어서 골짜기를 찾기

| 과정 | 비유 | 수학 |
|------|------|------|
| Score Function | 물이 흘러가는 방향 | $\nabla_x \log p(x)$ |
| Drift (드리프트) | 골짜기로 끌어당기기 | $\frac{\delta}{2}s(x)$ |
| Noise (노이즈) | 살짝 흔들어서 탐색 | $\sqrt{\delta}\epsilon$ |
| SDE | 물줄기가 갈라졌다 합치기 | $dx = f\,dt + g\,dw$ |

#### 🔄 SDE와 ODE의 차이는?

- **SDE** = 물에 바람이 불어서 방향이 **랜덤하게 흔들림** (확률적)
- **ODE** = 바람 없이 물이 **한 방향으로만** 흘러감 (결정론적)

둘 다 같은 곳(데이터 분포)에 도착하지만, ODE는 매번 같은 경로로!"""))

# ── Cell 4: 📝 Practice Problems ──────────────────────────────────
cells.append(md(r"""---

### 📝 연습 문제

#### 문제 1: Score Function 계산

1D 가우시안 분포 $p(x) = \mathcal{N}(x;\, \mu, \sigma^2)$의 스코어 함수를 구하세요.

<details>
<summary>💡 풀이 확인</summary>

$$\log p(x) = -\frac{(x-\mu)^2}{2\sigma^2} + C$$

$$s(x) = \nabla_x \log p(x) = -\frac{x - \mu}{\sigma^2}$$

**해석**: 스코어 벡터는 평균 $\mu$를 향하는 방향이며, 평균에서 멀수록 크기가 큽니다. 분산 $\sigma^2$가 작으면 더 강하게 끌어당깁니다.
</details>

#### 문제 2: Langevin Dynamics 수렴

$p(x) = \mathcal{N}(0, 1)$에 대해 Langevin dynamics를 적용합니다. $x_0 = 5$이고 $\delta = 0.1$일 때, 첫 스텝의 드리프트 항 $\frac{\delta}{2}s(x_0)$를 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$$s(x_0) = -\frac{x_0 - 0}{1} = -5$$

$$\frac{\delta}{2}s(x_0) = \frac{0.1}{2} \times (-5) = -0.25$$

$x_0 = 5$에서 평균 $0$ 방향으로 $-0.25$만큼 이동합니다. 노이즈 항 $\sqrt{0.1}\epsilon \approx 0.316\epsilon$이 추가되어 최종 위치가 결정됩니다.
</details>"""))

# ── Cell 5: Imports ──────────────────────────────────────────────
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

# ── Cell 6: 2D Score Field Visualization ─────────────────────────
cells.append(md(r"""## 2. 2D Score Field 시각화 <a name='2.-Score-Field-시각화'></a>

가우시안 혼합 분포(GMM)의 스코어 필드를 2D 벡터장으로 시각화합니다. 각 화살표는 확률이 높은 방향을 가리킵니다."""))

cells.append(code(r"""# ── 2D Score Field 시각화 ──────────────────────────────────────────
# 2D 가우시안 혼합 분포의 스코어 필드

def gmm_log_prob(x, means, covs, weights):
    # 가우시안 혼합 분포의 로그 확률 계산
    log_probs = []
    for mu, cov, w in zip(means, covs, weights):
        diff = x - mu
        inv_cov = np.linalg.inv(cov)
        log_p = -0.5 * np.sum(diff @ inv_cov * diff, axis=-1)
        log_p += np.log(w) - 0.5 * np.log(np.linalg.det(2 * np.pi * cov))
        log_probs.append(log_p)
    log_probs = np.stack(log_probs, axis=-1)
    return np.log(np.sum(np.exp(log_probs - log_probs.max(axis=-1, keepdims=True)), axis=-1)) + log_probs.max(axis=-1)

def gmm_score(x, means, covs, weights):
    # 수치적으로 스코어 계산 (유한 차분)
    eps = 1e-4
    score = np.zeros_like(x)
    for d in range(x.shape[-1]):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[..., d] += eps
        x_minus[..., d] -= eps
        score[..., d] = (gmm_log_prob(x_plus, means, covs, weights) -
                         gmm_log_prob(x_minus, means, covs, weights)) / (2 * eps)
    return score

# GMM 파라미터 (3개 모드)
means = [np.array([-2, -1]), np.array([2, 1]), np.array([0, 3])]
covs = [np.eye(2) * 0.5, np.eye(2) * 0.4, np.eye(2) * 0.6]
weights = [0.4, 0.35, 0.25]

# 그리드 생성
grid_size = 25
x_range = np.linspace(-5, 5, grid_size)
y_range = np.linspace(-4, 6, grid_size)
X, Y = np.meshgrid(x_range, y_range)
grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)

# 스코어 계산
scores = gmm_score(grid_points, means, covs, weights)
log_probs = gmm_log_prob(grid_points, means, covs, weights)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 확률 밀도
ax1 = axes[0]
prob_grid = np.exp(log_probs.reshape(grid_size, grid_size))
c1 = ax1.contourf(X, Y, prob_grid, levels=30, cmap='viridis')
plt.colorbar(c1, ax=ax1, label='p(x)')
for mu in means:
    ax1.scatter(*mu, c='red', s=100, marker='*', zorder=5, edgecolors='white')
ax1.set_title('확률 밀도 p(x)', fontweight='bold')
ax1.set_xlabel('$x_1$', fontsize=11)
ax1.set_ylabel('$x_2$', fontsize=11)
ax1.grid(True, alpha=0.3)

# 스코어 벡터장
ax2 = axes[1]
S_x = scores[:, 0].reshape(grid_size, grid_size)
S_y = scores[:, 1].reshape(grid_size, grid_size)
magnitude = np.sqrt(S_x**2 + S_y**2)
ax2.contourf(X, Y, prob_grid, levels=15, cmap='viridis', alpha=0.3)
ax2.quiver(X[::2, ::2], Y[::2, ::2], S_x[::2, ::2], S_y[::2, ::2],
           magnitude[::2, ::2], cmap='Reds', scale=30, width=0.004, alpha=0.8)
for mu in means:
    ax2.scatter(*mu, c='red', s=100, marker='*', zorder=5, edgecolors='white')
ax2.set_title(r'Score Field $\nabla_x \log p(x)$', fontweight='bold')
ax2.set_xlabel('$x_1$', fontsize=11)
ax2.set_ylabel('$x_2$', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/score_field_2d.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/score_field_2d.png")
print("스코어 벡터가 각 가우시안 중심(빨간 별)을 향하는 것을 관찰하세요")
print(f"스코어 최대 크기: {np.max(magnitude):.2f}")
print(f"스코어 평균 크기: {np.mean(magnitude):.2f}")"""))

# ── Cell 7: Langevin Dynamics Sampling ───────────────────────────
cells.append(md(r"""## 3. Langevin Dynamics 샘플링 <a name='3.-Langevin-Dynamics-샘플링'></a>

랜덤 노이즈에서 시작하여 Langevin dynamics로 GMM 분포에서 샘플을 생성합니다:

$$x_{t+1} = x_t + \frac{\delta}{2}\nabla_x \log p(x_t) + \sqrt{\delta}\,\epsilon_t$$"""))

cells.append(code(r"""# ── Langevin Dynamics 샘플링 ──────────────────────────────────────
# GMM에서 Langevin dynamics로 샘플 생성

def langevin_dynamics(score_fn, n_samples, n_steps, step_size, init_std=4.0):
    # Langevin dynamics 샘플러
    x = np.random.randn(n_samples, 2) * init_std
    trajectory = [x.copy()]

    for t in range(n_steps):
        score = score_fn(x)
        noise = np.random.randn(*x.shape)
        x = x + 0.5 * step_size * score + np.sqrt(step_size) * noise
        if t % (n_steps // 20) == 0 or t == n_steps - 1:
            trajectory.append(x.copy())

    return x, trajectory

score_fn = lambda x: gmm_score(x, means, covs, weights)

n_samples = 500
n_steps = 1000
step_size = 0.01

final_samples, trajectories = langevin_dynamics(score_fn, n_samples, n_steps, step_size)

# 시각화: 궤적과 최종 샘플
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 초기 분포
ax1 = axes[0]
ax1.scatter(trajectories[0][:, 0], trajectories[0][:, 1],
            c='blue', s=5, alpha=0.3, label='초기 샘플')
ax1.set_title('Step 0 (랜덤 초기화)', fontweight='bold')
ax1.set_xlim(-6, 6)
ax1.set_ylim(-5, 8)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('$x_1$', fontsize=11)
ax1.set_ylabel('$x_2$', fontsize=11)

# 중간 과정
mid_idx = len(trajectories) // 2
ax2 = axes[1]
ax2.scatter(trajectories[mid_idx][:, 0], trajectories[mid_idx][:, 1],
            c='orange', s=5, alpha=0.3, label=f'Step {n_steps//2}')
for mu in means:
    ax2.scatter(*mu, c='red', s=100, marker='*', zorder=5)
ax2.set_title(f'Step {n_steps//2} (수렴 중)', fontweight='bold')
ax2.set_xlim(-6, 6)
ax2.set_ylim(-5, 8)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('$x_1$', fontsize=11)
ax2.set_ylabel('$x_2$', fontsize=11)

# 최종 분포
ax3 = axes[2]
ax3.scatter(final_samples[:, 0], final_samples[:, 1],
            c='green', s=5, alpha=0.3, label=f'Step {n_steps}')
for mu in means:
    ax3.scatter(*mu, c='red', s=100, marker='*', zorder=5)
ax3.set_title(f'Step {n_steps} (수렴)', fontweight='bold')
ax3.set_xlim(-6, 6)
ax3.set_ylim(-5, 8)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('$x_1$', fontsize=11)
ax3.set_ylabel('$x_2$', fontsize=11)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/langevin_dynamics_sampling.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/langevin_dynamics_sampling.png")

# 통계
for i, mu in enumerate(means):
    dists = np.linalg.norm(final_samples - mu, axis=1)
    near = np.sum(dists < 1.5)
    print(f"모드 {i+1} (μ={mu}) 근처 샘플 수: {near}/{n_samples} ({near/n_samples*100:.1f}%)")"""))

# ── Cell 8: VE-SDE vs VP-SDE ────────────────────────────────────
cells.append(md(r"""## 4. VE-SDE vs VP-SDE 비교 <a name='4.-VE-SDE-vs-VP-SDE'></a>

두 가지 주요 SDE 유형의 노이즈 스케줄과 Forward process를 비교합니다.

| 유형 | Forward SDE | 노이즈 추가 방식 | 대응 모델 |
|------|-------------|-----------------|-----------|
| **VP-SDE** | $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dw$ | 신호 감쇠 + 노이즈 증가 | DDPM |
| **VE-SDE** | $dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\,dw$ | 신호 유지 + 노이즈만 증가 | NCSN/SMLD |"""))

cells.append(code(r"""# ── VE-SDE vs VP-SDE Forward Process 비교 ─────────────────────────
# 1D 데이터에 대한 Forward process 시뮬레이션

T_continuous = 1.0
n_steps_sde = 200
dt = T_continuous / n_steps_sde
t_values = np.linspace(0, T_continuous, n_steps_sde + 1)

x0 = 3.0
n_trajectories = 50

# VP-SDE: dx = -0.5*beta(t)*x*dt + sqrt(beta(t))*dw
def vp_sde_forward(x0, n_traj, beta_min=0.1, beta_max=20.0):
    trajectories = np.zeros((n_traj, n_steps_sde + 1))
    trajectories[:, 0] = x0

    for i in range(n_steps_sde):
        t = t_values[i]
        beta_t = beta_min + t * (beta_max - beta_min)
        drift = -0.5 * beta_t * trajectories[:, i]
        diffusion = np.sqrt(beta_t)
        dw = np.random.randn(n_traj) * np.sqrt(dt)
        trajectories[:, i+1] = trajectories[:, i] + drift * dt + diffusion * dw

    return trajectories

# VE-SDE: dx = sqrt(d[sigma^2]/dt) * dw
def ve_sde_forward(x0, n_traj, sigma_min=0.01, sigma_max=50.0):
    trajectories = np.zeros((n_traj, n_steps_sde + 1))
    trajectories[:, 0] = x0

    for i in range(n_steps_sde):
        t = t_values[i]
        sigma_t = sigma_min * (sigma_max / sigma_min) ** t
        dsigma2_dt = 2 * sigma_t * np.log(sigma_max / sigma_min) * sigma_t
        diffusion = np.sqrt(np.abs(dsigma2_dt))
        dw = np.random.randn(n_traj) * np.sqrt(dt)
        trajectories[:, i+1] = trajectories[:, i] + diffusion * dw

    return trajectories

np.random.seed(42)
vp_trajs = vp_sde_forward(x0, n_trajectories)
np.random.seed(42)
ve_trajs = ve_sde_forward(x0, n_trajectories)

# VP-SDE 해석적 분포 파라미터
beta_min, beta_max = 0.1, 20.0
vp_mean = np.zeros_like(t_values)
vp_std = np.zeros_like(t_values)
for i, t in enumerate(t_values):
    integral_beta = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
    vp_mean[i] = x0 * np.exp(-0.5 * integral_beta)
    vp_std[i] = np.sqrt(1 - np.exp(-integral_beta))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# VP-SDE
ax1 = axes[0]
for j in range(min(20, n_trajectories)):
    ax1.plot(t_values, vp_trajs[j], 'b-', alpha=0.1, lw=0.5)
ax1.plot(t_values, vp_mean, 'r-', lw=2.5, label=r'$E[x_t]$ (해석적)')
ax1.fill_between(t_values, vp_mean - 2*vp_std, vp_mean + 2*vp_std,
                 alpha=0.15, color='red', label=r'$\pm 2\sigma$ 구간')
ax1.axhline(y=0, color='gray', ls='--', lw=1)
ax1.set_title('VP-SDE (Variance Preserving)', fontweight='bold')
ax1.set_xlabel('t', fontsize=11)
ax1.set_ylabel('$x_t$', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# VE-SDE
ax2 = axes[1]
for j in range(min(20, n_trajectories)):
    ax2.plot(t_values, ve_trajs[j], 'g-', alpha=0.1, lw=0.5)
ve_empirical_mean = np.mean(ve_trajs, axis=0)
ve_empirical_std = np.std(ve_trajs, axis=0)
ax2.plot(t_values, ve_empirical_mean, 'r-', lw=2.5, label=r'$E[x_t]$ (경험적)')
ax2.fill_between(t_values, ve_empirical_mean - 2*ve_empirical_std,
                 ve_empirical_mean + 2*ve_empirical_std,
                 alpha=0.15, color='red', label=r'$\pm 2\sigma$ 구간')
ax2.axhline(y=x0, color='gray', ls='--', lw=1, label=f'$x_0 = {x0}$')
ax2.set_title('VE-SDE (Variance Exploding)', fontweight='bold')
ax2.set_xlabel('t', fontsize=11)
ax2.set_ylabel('$x_t$', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/ve_vs_vp_sde.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/ve_vs_vp_sde.png")

print(f"\n{'SDE 유형':<15} | {'최종 E[x_T]':>12} | {'최종 Std[x_T]':>12} | {'신호 보존':>10}")
print("-" * 58)
print(f"{'VP-SDE':<15} | {np.mean(vp_trajs[:, -1]):>12.3f} | {np.std(vp_trajs[:, -1]):>12.3f} | {'아니오':>10}")
print(f"{'VE-SDE':<15} | {np.mean(ve_trajs[:, -1]):>12.3f} | {np.std(ve_trajs[:, -1]):>12.3f} | {'예 (평균)':>10}")"""))

# ── Cell 9: Probability Flow ODE ────────────────────────────────
cells.append(md(r"""## 5. Probability Flow ODE <a name='5.-Probability-Flow-ODE'></a>

SDE와 동일한 주변 분포를 갖는 **결정론적 ODE**:

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

SDE(확률적) 경로와 ODE(결정론적) 경로를 비교합니다."""))

cells.append(code(r"""# ── SDE vs Probability Flow ODE 경로 비교 ─────────────────────────
# 1D 가우시안에서의 SDE vs ODE 역방향 경로

def reverse_vp_sde(x_T, score_fn, n_steps=200, beta_min=0.1, beta_max=20.0):
    # VP-SDE 역방향 (확률적)
    dt = 1.0 / n_steps
    x = x_T.copy()
    trajectory = [x.copy()]

    for i in range(n_steps):
        t = 1.0 - i * dt
        beta_t = beta_min + t * (beta_max - beta_min)
        drift = -0.5 * beta_t * x
        diffusion = np.sqrt(beta_t)
        score = score_fn(x, t)
        dx = (drift - diffusion**2 * score) * (-dt) + diffusion * np.random.randn(*x.shape) * np.sqrt(dt)
        x = x + dx
        if i % (n_steps // 50) == 0:
            trajectory.append(x.copy())

    trajectory.append(x.copy())
    return x, trajectory

def reverse_probability_flow_ode(x_T, score_fn, n_steps=200, beta_min=0.1, beta_max=20.0):
    # Probability Flow ODE (결정론적)
    dt = 1.0 / n_steps
    x = x_T.copy()
    trajectory = [x.copy()]

    for i in range(n_steps):
        t = 1.0 - i * dt
        beta_t = beta_min + t * (beta_max - beta_min)
        drift = -0.5 * beta_t * x
        diffusion = np.sqrt(beta_t)
        score = score_fn(x, t)
        dx = (drift - 0.5 * diffusion**2 * score) * (-dt)
        x = x + dx
        if i % (n_steps // 50) == 0:
            trajectory.append(x.copy())

    trajectory.append(x.copy())
    return x, trajectory

# 간단한 스코어 함수 (1D 가우시안 타겟 mu=2, sigma=0.5)
target_mu, target_sigma = 2.0, 0.5

def simple_score(x, t):
    beta_min, beta_max = 0.1, 20.0
    integral_beta = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
    alpha_bar_t = np.exp(-integral_beta)
    effective_mu = np.sqrt(alpha_bar_t) * target_mu
    effective_var = alpha_bar_t * target_sigma**2 + (1 - alpha_bar_t)
    return -(x - effective_mu) / effective_var

# 다수의 경로 생성
np.random.seed(42)
n_paths = 8
x_T_init = np.random.randn(n_paths) * 1.0

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# SDE 경로 (확률적)
ax1 = axes[0]
for i in range(n_paths):
    np.random.seed(i + 100)
    _, traj_sde = reverse_vp_sde(np.array([x_T_init[i]]), simple_score, n_steps=200)
    traj_arr = np.array([t[0] for t in traj_sde])
    ax1.plot(np.linspace(1, 0, len(traj_arr)), traj_arr, alpha=0.6, lw=1.5)
ax1.axhline(y=target_mu, color='red', ls='--', lw=2, label=f'target μ = {target_mu}')
ax1.set_title('Reverse SDE (확률적 경로)', fontweight='bold')
ax1.set_xlabel('t (역방향: 1→0)', fontsize=11)
ax1.set_ylabel('$x_t$', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ODE 경로 (결정론적)
ax2 = axes[1]
for i in range(n_paths):
    _, traj_ode = reverse_probability_flow_ode(np.array([x_T_init[i]]), simple_score, n_steps=200)
    traj_arr = np.array([t[0] for t in traj_ode])
    ax2.plot(np.linspace(1, 0, len(traj_arr)), traj_arr, alpha=0.6, lw=1.5)
ax2.axhline(y=target_mu, color='red', ls='--', lw=2, label=f'target μ = {target_mu}')
ax2.set_title('Probability Flow ODE (결정론적 경로)', fontweight='bold')
ax2.set_xlabel('t (역방향: 1→0)', fontsize=11)
ax2.set_ylabel('$x_t$', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/sde_vs_ode_paths.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/sde_vs_ode_paths.png")
print("\nSDE: 같은 초기값에서도 매번 다른 경로 (확률적 노이즈)")
print("ODE: 같은 초기값이면 항상 같은 경로 (결정론적)")
print(f"ODE의 장점: 정확한 로그 우도 계산 가능, DDIM은 ODE의 이산화")"""))

# ── Cell 10: Summary ─────────────────────────────────────────────
cells.append(md(r"""## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| Score Function | 로그 확률 밀도의 그래디언트 $\nabla_x \log p(x)$ | ⭐⭐⭐ |
| Denoising Score Matching | 노이즈 데이터에서 스코어 학습 (DDPM Loss와 동일) | ⭐⭐⭐ |
| Langevin Dynamics | 스코어 기반 반복적 샘플링 | ⭐⭐⭐ |
| VP-SDE | 신호 감쇠 + 노이즈 증가 (DDPM에 대응) | ⭐⭐⭐ |
| VE-SDE | 신호 유지 + 노이즈 폭발 (NCSN에 대응) | ⭐⭐ |
| Probability Flow ODE | 결정론적 역방향, 정확한 우도 계산 가능 | ⭐⭐⭐ |

### 핵심 수식

$$s(x) = \nabla_x \log p(x) \quad \text{(Score Function)}$$

$$x_{t+1} = x_t + \frac{\delta}{2}s(x_t) + \sqrt{\delta}\,\epsilon_t \quad \text{(Langevin Dynamics)}$$

$$dx = f(x,t)\,dt + g(t)\,dw \quad \text{(Forward SDE)}$$

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 s_\theta(x,t) \quad \text{(Probability Flow ODE)}$$

### DDPM ↔ Score 연결

$$\epsilon_\theta(x_t, t) = -\sqrt{1 - \bar\alpha_t} \cdot s_\theta(x_t, t)$$

노이즈 예측($\epsilon$-prediction)과 스코어 예측($s$-prediction)은 **스케일링 계수**만 다른 **동일한 학습**입니다.

### 다음 챕터 예고
**Chapter 14: 극한 추론 최적화** — FlashAttention, Speculative Decoding, PagedAttention, 양자화를 학습합니다."""))

# ── Create notebook ──────────────────────────────────────────────
path = '/workspace/chapter13_genai_diffusion/05_score_matching_and_sde.ipynb'
create_notebook(cells, path)
