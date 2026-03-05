import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──
cells.append(md(r"""# Chapter 17: 비디오 생성 모델과 DiT — Flow Matching과 Rectified Flow

## 학습 목표
- Flow Matching의 ODE 기반 생성 프레임워크를 수식 수준에서 이해한다
- Rectified Flow의 직선 경로(straight-line path) 설계를 도출하고 시각화한다
- DDPM의 곡선 노이즈 경로와 Flow Matching의 직선 경로를 정량적으로 비교한다
- Flow Matching Loss를 TensorFlow로 구현하고 학습 과정을 실험한다
- Euler ODE Solver를 이용한 샘플링 과정을 단계별로 구현한다
- SD3·Flux와 DDPM의 훈련 방식 차이를 체계적으로 비교한다

## 목차
1. [수학적 기초: Flow Matching과 Rectified Flow](#1.-수학적-기초)
2. [라이브러리 임포트 및 환경 설정](#2.-환경-설정)
3. [Rectified Flow 경로 시각화 (직선 vs DDPM 곡선)](#3.-경로-시각화)
4. [Flow Matching Loss 구현](#4.-FM-Loss-구현)
5. [Euler ODE Solver 샘플링](#5.-Euler-샘플링)
6. [SD3/Flux vs DDPM 훈련 방식 비교](#6.-훈련-비교)
7. [정리](#7.-정리)"""))

# ── Cell 2: Math Section ──
cells.append(md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### Flow Matching ODE

연속 정규화 흐름(Continuous Normalizing Flow)에서 데이터를 생성하는 ODE:

$$\frac{dx}{dt} = v_\theta(x, t), \quad t \in [0, 1]$$

- $x_0 \sim p_0$: 노이즈 분포 (예: $\mathcal{N}(0, I)$)
- $x_1 \sim p_1$: 데이터 분포
- $v_\theta$: 신경망이 학습하는 **속도장(velocity field)**
- $t=0$에서 $t=1$로 적분하면 노이즈 → 데이터 변환

### Rectified Flow (직선 경로)

가장 간단한 보간(interpolation) 경로를 정의합니다:

$$x_t = (1-t)x_0 + tx_1$$

이때 최적 속도장(Optimal Transport)은:

$$v^{OT}(x_t, t) = x_1 - x_0$$

- $x_0$: 노이즈 샘플
- $x_1$: 데이터 샘플
- 경로가 **직선**이므로 속도가 시간에 무관(상수)
- 이론적으로 1-step 생성이 가능 (실제로는 수 스텝 필요)

### Flow Matching Loss

$$\mathcal{L}_{FM} = \mathbb{E}_{t \sim U[0,1],\, x_0 \sim p_0,\, x_1 \sim p_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

- $v_\theta(x_t, t)$: 신경망의 속도 예측
- $(x_1 - x_0)$: GT(ground truth) 직선 속도
- MSE 손실로 속도장을 회귀(regression) 학습

### DDPM vs Flow Matching 비교

| 구분 | DDPM | Flow Matching (Rectified Flow) |
|------|------|------|
| 경로 | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ (곡선) | $x_t = (1-t)x_0 + tx_1$ (직선) |
| 예측 대상 | 노이즈 $\epsilon$ | 속도 $v = x_1 - x_0$ |
| 샘플링 | $T$ 스텝 역방향 SDE/ODE | Euler ODE (소수 스텝 가능) |
| 이론 기반 | 마르코프 체인 + 변분 추론 | 연속 정규화 흐름 + OT |

**요약 표:**

| 구분 | 수식 | 설명 |
|------|------|------|
| FM ODE | $dx/dt = v_\theta(x,t)$ | 속도장 기반 생성 |
| Rectified 보간 | $x_t = (1-t)x_0 + tx_1$ | 직선 경로 |
| 최적 속도 | $v^{OT} = x_1 - x_0$ | 시간 무관 상수 |
| FM Loss | $\mathbb{E}[\|v_\theta - v^{OT}\|^2]$ | MSE 회귀 손실 |
| Euler 스텝 | $x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$ | ODE 이산 적분 |

---

### 🐣 초등학생을 위한 Flow Matching 친절 설명!

#### 🔢 Flow Matching이 뭔가요?

> 💡 **비유**: 지도 앱에서 출발지(노이즈)에서 목적지(이미지)까지 "가장 빠른 길"을 찾는 것!

DDPM은 구불구불한 골목길로 돌아가는 방법이에요. 여러 번 방향을 바꿔야 하죠 (1000 스텝!).
Flow Matching은 **고속도로처럼 직선**으로 바로 가는 방법이에요. 훨씬 적은 스텝으로 도착할 수 있죠!

- 🛣️ **Rectified Flow**: "직선 도로"를 그어놓고, AI에게 "이 도로 위에서 속도를 맞춰봐"라고 학습시킴
- 🧭 **속도장**: 각 위치에서 "어느 방향으로 얼마나 빨리 가야 하는지" 알려주는 나침반
- 📏 **ODE**: 나침반 방향대로 한 걸음씩 걸으면 목적지에 도착! (걸음 수 = 스텝 수)

---

### 📝 연습 문제

#### 문제 1: Rectified Flow 보간 계산

$x_0 = [1, 0]$, $x_1 = [0, 1]$일 때, $t=0.3$에서의 $x_t$와 속도 $v^{OT}$를 구하세요.

<details>
<summary>💡 풀이 확인</summary>

$$x_t = (1-0.3)[1,0] + 0.3[0,1] = [0.7, 0.3]$$

$$v^{OT} = x_1 - x_0 = [0,1] - [1,0] = [-1, 1]$$

검증: $x_t + (1-t) \cdot v^{OT} = [0.7, 0.3] + 0.7 \cdot [-1, 1] = [0, 1] = x_1$ ✓
</details>

#### 문제 2: Euler 스텝 수와 오차

2스텝 Euler ($\Delta t = 0.5$)로 $x_0 = [1, 0]$에서 시작하여 $x_1 = [0, 1]$을 복원하세요 (속도가 상수 $v = [-1, 1]$일 때).

<details>
<summary>💡 풀이 확인</summary>

**스텝 1**: $x_{0.5} = x_0 + 0.5 \cdot v = [1,0] + 0.5[-1,1] = [0.5, 0.5]$

**스텝 2**: $x_1 = x_{0.5} + 0.5 \cdot v = [0.5,0.5] + 0.5[-1,1] = [0, 1]$

직선 경로에서 상수 속도면 **Euler 적분이 정확합니다** (오차 = 0)!
이것이 Rectified Flow의 핵심 장점입니다.
</details>"""))

# ── Cell 3: Section 2 MD ──
cells.append(md(r"""## 2. 라이브러리 임포트 및 환경 설정 <a name='2.-환경-설정'></a>"""))

# ── Cell 4: Imports ──
cells.append(code(r"""# ── 라이브러리 임포트 및 환경 설정 ──────────────────────────────────
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

# ── Cell 4: MD for section 3 ──
cells.append(md(r"""## 3. Rectified Flow 경로 시각화 (직선 vs DDPM 곡선) <a name='3.-경로-시각화'></a>

DDPM의 노이즈 확산 경로와 Rectified Flow의 직선 경로를 2D 공간에서 비교합니다.

- **DDPM**: $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$ (비선형 곡선 경로)
- **Rectified Flow**: $x_t = (1-t)x_0 + tx_1$ (직선 경로)"""))

# ── Cell 5: Path Visualization ──
cells.append(code(r"""# ── Rectified Flow vs DDPM 경로 시각화 ─────────────────────────────
# 2D 공간에서 노이즈(x0) → 데이터(x1)로의 경로 비교

x0 = np.array([2.0, -1.0])
x1 = np.array([-1.0, 2.0])
epsilon = x0.copy()

T = 50
t_values = np.linspace(0, 1, T)

# DDPM 경로: cosine schedule 모사
betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)

ddpm_path = np.zeros((T, 2))
rf_path = np.zeros((T, 2))

for i, t in enumerate(t_values):
    idx = min(i, len(alpha_bar) - 1)
    sqrt_ab = np.sqrt(alpha_bar[idx])
    sqrt_1_ab = np.sqrt(1 - alpha_bar[idx])
    ddpm_path[i] = sqrt_ab * x1 + sqrt_1_ab * epsilon
    rf_path[i] = (1 - t) * x0 + t * x1

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
ax1.plot(ddpm_path[:, 0], ddpm_path[:, 1], 'r-o', lw=2, ms=3, alpha=0.7, label='DDPM 경로 (곡선)')
ax1.plot(rf_path[:, 0], rf_path[:, 1], 'b-s', lw=2, ms=3, alpha=0.7, label='Rectified Flow (직선)')
ax1.plot(*x0, 'ko', ms=12, zorder=5)
ax1.annotate('$x_0$ (노이즈)', xy=x0, xytext=(x0[0]+0.3, x0[1]-0.5), fontsize=10)
ax1.plot(*x1, 'k*', ms=15, zorder=5)
ax1.annotate('$x_1$ (데이터)', xy=x1, xytext=(x1[0]+0.3, x1[1]+0.3), fontsize=10)
ax1.set_xlabel('Dimension 1', fontsize=11)
ax1.set_ylabel('Dimension 2', fontsize=11)
ax1.set_title('경로 비교: DDPM vs Rectified Flow', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ddpm_dist = np.sqrt(np.sum(np.diff(ddpm_path, axis=0)**2, axis=1))
rf_dist = np.sqrt(np.sum(np.diff(rf_path, axis=0)**2, axis=1))
ddpm_cumlen = np.concatenate([[0], np.cumsum(ddpm_dist)])
rf_cumlen = np.concatenate([[0], np.cumsum(rf_dist)])
ax2.plot(t_values, ddpm_cumlen, 'r-', lw=2.5, label=f'DDPM (총 길이: {ddpm_cumlen[-1]:.2f})')
ax2.plot(t_values, rf_cumlen, 'b-', lw=2.5, label=f'RF (총 길이: {rf_cumlen[-1]:.2f})')
ax2.set_xlabel('시간 t', fontsize=11)
ax2.set_ylabel('누적 경로 길이', fontsize=11)
ax2.set_title('누적 경로 길이 비교', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/flow_matching_paths.png', dpi=100, bbox_inches='tight')
plt.close()

straight_line_dist = np.linalg.norm(x1 - x0)
print(f"직선 거리 (최적): {straight_line_dist:.4f}")
print(f"DDPM 경로 총 길이: {ddpm_cumlen[-1]:.4f}")
print(f"Rectified Flow 경로 총 길이: {rf_cumlen[-1]:.4f}")
print(f"DDPM 경로 비효율: {ddpm_cumlen[-1]/straight_line_dist:.2f}x")
print(f"RF 경로 비효율: {rf_cumlen[-1]/straight_line_dist:.2f}x")
print("그래프 저장됨: chapter17_diffusion_transformers/flow_matching_paths.png")"""))

# ── Cell 6: MD for FM Loss ──
cells.append(md(r"""## 4. Flow Matching Loss 구현 <a name='4.-FM-Loss-구현'></a>

Flow Matching의 학습 목표는 속도장 $v_\theta$가 직선 속도 $v^{OT} = x_1 - x_0$을 예측하는 것입니다:

$$\mathcal{L}_{FM} = \mathbb{E}_{t,x_0,x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

간단한 2D 데이터에서 MLP 속도 모델을 학습합니다."""))

# ── Cell 7: FM Loss Implementation ──
cells.append(code(r"""# ── Flow Matching Loss 구현 및 학습 ──────────────────────────────────
# 간단한 2D 스위스롤 데이터에서 FM 학습

def make_swiss_roll_2d(n=1000):
    t = np.random.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    x = t * np.cos(t) * 0.05
    y = t * np.sin(t) * 0.05
    return np.stack([x, y], axis=-1).astype(np.float32)

data = make_swiss_roll_2d(2000)
print(f"학습 데이터 shape: {data.shape}, 범위: [{data.min():.3f}, {data.max():.3f}]")

class VelocityMLP(tf.keras.Model):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='silu'),
            tf.keras.layers.Dense(hidden_dim, activation='silu'),
            tf.keras.layers.Dense(hidden_dim, activation='silu'),
            tf.keras.layers.Dense(2)
        ])

    def call(self, x_t, t):
        t_embed = tf.concat([tf.sin(t * np.pi * 2), tf.cos(t * np.pi * 2)], axis=-1)
        inp = tf.concat([x_t, t_embed], axis=-1)
        return self.net(inp)

model = VelocityMLP(hidden_dim=128)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(x1_batch):
    batch_size = tf.shape(x1_batch)[0]
    t = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    x0 = tf.random.normal([batch_size, 2])
    x_t = (1.0 - t) * x0 + t * x1_batch
    v_target = x1_batch - x0

    with tf.GradientTape() as tape:
        v_pred = model(x_t, t)
        loss = tf.reduce_mean(tf.square(v_pred - v_target))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(2000).batch(256).repeat()
iterator = iter(dataset)

losses = []
for step in range(800):
    batch = next(iterator)
    loss = train_step(batch)
    losses.append(float(loss))
    if (step + 1) % 200 == 0:
        print(f"스텝 {step+1:4d} | FM Loss: {loss:.6f}")

print(f"\n최종 FM Loss: {losses[-1]:.6f}")
print(f"모델 파라미터 수: {sum(p.numpy().size for p in model.trainable_variables):,}")"""))

# ── Cell 8: MD for Euler Sampling ──
cells.append(md(r"""## 5. Euler ODE Solver 샘플링 <a name='5.-Euler-샘플링'></a>

학습된 속도장 $v_\theta$를 사용하여 Euler 방법으로 ODE를 적분합니다:

$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$$

$t=0$ (노이즈)에서 시작하여 $t=1$ (데이터)까지 $N$ 스텝으로 이동합니다."""))

# ── Cell 9: Euler Sampling ──
cells.append(code(r"""# ── Euler ODE Solver 샘플링 ───────────────────────────────────────
def euler_sample(model, n_samples=500, n_steps=50):
    x = tf.random.normal([n_samples, 2])
    dt = 1.0 / n_steps
    trajectory = [x.numpy()]

    for i in range(n_steps):
        t_val = i * dt
        t = tf.fill([n_samples, 1], t_val)
        v = model(x, t)
        x = x + dt * v
        if i % (n_steps // 5) == 0:
            trajectory.append(x.numpy())
    trajectory.append(x.numpy())
    return x.numpy(), trajectory

step_counts = [5, 20, 50]
fig, axes = plt.subplots(1, len(step_counts) + 1, figsize=(16, 4))

ax0 = axes[0]
ax0.scatter(data[:, 0], data[:, 1], s=2, alpha=0.5, c='green')
ax0.set_title('원본 데이터 (GT)', fontweight='bold')
ax0.set_xlim(-1.0, 1.0)
ax0.set_ylim(-1.0, 1.0)
ax0.grid(True, alpha=0.3)

for idx, n_steps in enumerate(step_counts):
    samples, _ = euler_sample(model, n_samples=500, n_steps=n_steps)
    ax = axes[idx + 1]
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5, c='blue')
    ax.set_title(f'Euler {n_steps} 스텝', fontweight='bold')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/euler_sampling_steps.png', dpi=100, bbox_inches='tight')
plt.close()

print("스텝 수에 따른 샘플링 품질:")
for n in step_counts:
    samples, _ = euler_sample(model, n_samples=1000, n_steps=n)
    mean_val = np.mean(samples, axis=0)
    std_val = np.std(samples, axis=0)
    print(f"  {n:3d} 스텝 | 평균: [{mean_val[0]:+.4f}, {mean_val[1]:+.4f}] | 표준편차: [{std_val[0]:.4f}, {std_val[1]:.4f}]")

gt_mean = np.mean(data, axis=0)
gt_std = np.std(data, axis=0)
print(f"  GT 통계 | 평균: [{gt_mean[0]:+.4f}, {gt_mean[1]:+.4f}] | 표준편차: [{gt_std[0]:.4f}, {gt_std[1]:.4f}]")
print("\n그래프 저장됨: chapter17_diffusion_transformers/euler_sampling_steps.png")"""))

# ── Cell 10: MD for Training Comparison ──
cells.append(md(r"""## 6. SD3/Flux vs DDPM 훈련 방식 비교 <a name='6.-훈련-비교'></a>

Stable Diffusion 3(SD3)와 Flux는 Flow Matching (Rectified Flow)을 채택하여 DDPM과 근본적으로 다른 훈련 방식을 사용합니다.

| 특성 | DDPM/DDIM | SD3/Flux (Flow Matching) |
|------|-----------|--------------------------|
| 이론 기반 | 변분 추론 + 마르코프 체인 | ODE + Optimal Transport |
| 노이즈 스케줄 | $\beta$ 스케줄 (linear/cosine) | 시간 $t \in [0,1]$ 균일 샘플링 |
| 보간 | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | $x_t = (1-t)x_0 + tx_1$ |
| 예측 대상 | $\epsilon$ (노이즈) | $v$ (속도) |
| 아키텍처 | U-Net | DiT (MM-DiT in SD3) |"""))

# ── Cell 11: Training Comparison Code ──
cells.append(code(r"""# ── SD3/Flux vs DDPM 훈련 방식 정량 비교 ─────────────────────────────
# 동일 데이터에 대해 두 패러다임의 학습 효율 비교

# DDPM-style: epsilon prediction
class EpsilonMLP(tf.keras.Model):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='silu'),
            tf.keras.layers.Dense(hidden_dim, activation='silu'),
            tf.keras.layers.Dense(hidden_dim, activation='silu'),
            tf.keras.layers.Dense(2)
        ])

    def call(self, x_t, t):
        t_embed = tf.concat([tf.sin(t * np.pi * 2), tf.cos(t * np.pi * 2)], axis=-1)
        inp = tf.concat([x_t, t_embed], axis=-1)
        return self.net(inp)

ddpm_model = EpsilonMLP(128)
ddpm_opt = tf.keras.optimizers.Adam(1e-3)

T_ddpm = 100
betas_tf = tf.cast(tf.linspace(1e-4, 0.02, T_ddpm), tf.float32)
alphas_tf = 1.0 - betas_tf
alpha_bar_tf = tf.math.cumprod(alphas_tf)

@tf.function
def ddpm_train_step(x0_batch):
    bs = tf.shape(x0_batch)[0]
    t_idx = tf.random.uniform([bs], 0, T_ddpm, dtype=tf.int32)
    ab = tf.gather(alpha_bar_tf, t_idx)
    ab = tf.reshape(ab, [-1, 1])
    eps = tf.random.normal(tf.shape(x0_batch))
    x_t = tf.sqrt(ab) * x0_batch + tf.sqrt(1.0 - ab) * eps
    t_norm = tf.cast(t_idx, tf.float32) / float(T_ddpm)
    t_norm = tf.reshape(t_norm, [-1, 1])

    with tf.GradientTape() as tape:
        eps_pred = ddpm_model(x_t, t_norm)
        loss = tf.reduce_mean(tf.square(eps_pred - eps))

    grads = tape.gradient(loss, ddpm_model.trainable_variables)
    ddpm_opt.apply_gradients(zip(grads, ddpm_model.trainable_variables))
    return loss

fm_model2 = VelocityMLP(128)
fm_opt2 = tf.keras.optimizers.Adam(1e-3)

@tf.function
def fm_train_step2(x1_batch):
    bs = tf.shape(x1_batch)[0]
    t = tf.random.uniform([bs, 1], 0.0, 1.0)
    x0 = tf.random.normal(tf.shape(x1_batch))
    x_t = (1.0 - t) * x0 + t * x1_batch
    v_target = x1_batch - x0

    with tf.GradientTape() as tape:
        v_pred = fm_model2(x_t, t)
        loss = tf.reduce_mean(tf.square(v_pred - v_target))

    grads = tape.gradient(loss, fm_model2.trainable_variables)
    fm_opt2.apply_gradients(zip(grads, fm_model2.trainable_variables))
    return loss

ddpm_losses, fm_losses = [], []
n_compare_steps = 500
dataset2 = tf.data.Dataset.from_tensor_slices(data).shuffle(2000).batch(256).repeat()
it2 = iter(dataset2)

for step in range(n_compare_steps):
    batch = next(it2)
    dl = ddpm_train_step(batch)
    fl = fm_train_step2(batch)
    ddpm_losses.append(float(dl))
    fm_losses.append(float(fl))

fig, ax = plt.subplots(figsize=(10, 5))
window = 20
ddpm_smooth = np.convolve(ddpm_losses, np.ones(window)/window, mode='valid')
fm_smooth = np.convolve(fm_losses, np.ones(window)/window, mode='valid')
ax.plot(ddpm_smooth, 'r-', lw=2, label='DDPM (epsilon prediction)', alpha=0.8)
ax.plot(fm_smooth, 'b-', lw=2, label='Flow Matching (velocity prediction)', alpha=0.8)
ax.set_xlabel('학습 스텝', fontsize=11)
ax.set_ylabel('Loss (이동 평균)', fontsize=11)
ax.set_title('DDPM vs Flow Matching 학습 곡선 비교', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/ddpm_vs_fm_training.png', dpi=100, bbox_inches='tight')
plt.close()

print(f"{'방법':<25} | {'최종 Loss':>12} | {'수렴 속도':>12}")
print("-" * 55)
print(f"{'DDPM (eps prediction)':<25} | {ddpm_losses[-1]:>12.6f} | {np.mean(ddpm_losses[-50:]):>12.6f}")
print(f"{'FM (velocity prediction)':<25} | {fm_losses[-1]:>12.6f} | {np.mean(fm_losses[-50:]):>12.6f}")
print()

print("SD3/Flux vs DDPM 핵심 차이점:")
headers = ['특성', 'DDPM', 'SD3/Flux (FM)']
rows = [
    ['보간 경로', '곡선 (sqrt(alpha_bar))', '직선 ((1-t)x0 + tx1)'],
    ['예측 대상', '노이즈 epsilon', '속도 v = x1 - x0'],
    ['샘플러', 'DDPM/DDIM (50~1000 스텝)', 'Euler ODE (5~50 스텝)'],
    ['아키텍처', 'U-Net', 'MM-DiT (Transformer)'],
    ['SD3 특징', '-', 'Rectified Flow + 3-text encoder'],
    ['Flux 특징', '-', 'Rectified Flow + rotary PE'],
]
print(f"{headers[0]:<22} | {headers[1]:<28} | {headers[2]:<28}")
print("-" * 84)
for r in rows:
    print(f"{r[0]:<22} | {r[1]:<28} | {r[2]:<28}")

print("\n그래프 저장됨: chapter17_diffusion_transformers/ddpm_vs_fm_training.png")"""))

# ── Cell 12: Loss Landscape Visualization ──
cells.append(code(r"""# ── Flow Matching 속도장 시각화 ────────────────────────────────────
# 학습된 속도장을 2D 벡터 필드로 시각화

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
time_points = [0.0, 0.5, 0.9]

for idx, t_val in enumerate(time_points):
    ax = axes[idx]
    grid_x = np.linspace(-1.5, 1.5, 15)
    grid_y = np.linspace(-1.5, 1.5, 15)
    X, Y = np.meshgrid(grid_x, grid_y)
    points = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)

    t_input = tf.fill([points.shape[0], 1], t_val)
    v_pred = model(tf.constant(points), t_input).numpy()

    ax.quiver(X, Y, v_pred[:, 0].reshape(X.shape), v_pred[:, 1].reshape(Y.shape),
              color='blue', alpha=0.6, scale=20)
    ax.scatter(data[:300, 0], data[:300, 1], s=3, alpha=0.3, c='green', label='데이터')
    ax.set_title(f't = {t_val:.1f}', fontweight='bold', fontsize=12)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=9)

plt.suptitle('학습된 속도장 $v_\\theta(x, t)$ (시간에 따른 변화)', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/velocity_field.png', dpi=100, bbox_inches='tight')
plt.close()
print("속도장 시각화 저장됨: chapter17_diffusion_transformers/velocity_field.png")
print(f"t=0.0: 노이즈 분포에서 데이터 방향으로 강한 속도")
print(f"t=0.5: 중간 지점에서 수렴하는 흐름")
print(f"t=0.9: 데이터 근처에서 미세 조정 속도")"""))

# ── Cell 13: Summary ──
cells.append(md(r"""## 7. 정리 <a name='7.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| Flow Matching ODE | $dx/dt = v_\theta(x,t)$로 노이즈→데이터 변환 | ⭐⭐⭐ |
| Rectified Flow | $(1-t)x_0 + tx_1$ 직선 보간, 속도 $= x_1 - x_0$ | ⭐⭐⭐ |
| FM Loss | 속도장 MSE 회귀: $\|v_\theta - (x_1-x_0)\|^2$ | ⭐⭐⭐ |
| Euler Solver | $x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta$ | ⭐⭐ |
| DDPM vs FM 경로 | 곡선(마르코프) vs 직선(OT) | ⭐⭐⭐ |
| SD3/Flux | Rectified Flow + MM-DiT 아키텍처 | ⭐⭐ |

### 핵심 수식

$$\frac{dx}{dt} = v_\theta(x, t)$$

$$x_t = (1-t)x_0 + tx_1, \quad v^{OT} = x_1 - x_0$$

$$\mathcal{L}_{FM} = \mathbb{E}_{t,x_0,x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

### 다음 챕터 예고
**05_sora_and_hunyuan_architecture** — Sora의 NaViT 가변 해상도 기법과 HunyuanVideo의 Dual/Single-stream 멀티모달 퓨전 아키텍처를 비교 분석합니다."""))

path = '/workspace/chapter17_diffusion_transformers/04_flow_matching_and_rectified_flow.ipynb'
create_notebook(cells, path)
