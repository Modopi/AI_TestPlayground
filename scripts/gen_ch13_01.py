"""Generate Chapter 13-01: DDPM Theory and Math."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helper import md, code, create_notebook

cells = [
# ─── Cell 0: 헤더 ───
md(r"""# Chapter 13: 생성 AI 심화 — DDPM 이론과 수학

## 학습 목표
- 확산 모델(Diffusion Model)의 **마르코프 체인 기반 Forward/Reverse 과정**을 수학적으로 이해한다
- Reparameterization trick을 사용하여 **임의 시점 $t$의 노이즈 샘플** $x_t$를 직접 계산하는 방법을 유도한다
- ELBO(Evidence Lower Bound)에서 **단순화된 손실 함수** $\|\epsilon - \epsilon_\theta\|^2$이 도출되는 과정을 전개한다
- TensorFlow로 **선형 베타 스케줄**과 Forward Process를 구현하고 시각화한다
- **1D 데이터에 대한 간단한 DDPM 학습 루프**를 직접 구현하여 역방향 생성 과정을 체험한다

## 목차
1. [수학적 기초: Forward Process, Reverse Process, ELBO](#1.-수학적-기초)
2. [베타 스케줄과 알파 누적곱 구현](#2.-베타-스케줄-구현)
3. [Forward Noising 과정 시각화](#3.-Forward-Noising-시각화)
4. [다양한 타임스텝에서의 노이즈 시각화](#4.-타임스텝별-노이즈)
5. [1D DDPM 학습 루프 구현](#5.-1D-DDPM-학습)
6. [정리](#6.-정리)"""),

# ─── Cell 1: 수학적 기초 ───
md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### Forward Process (확산 과정)

DDPM의 Forward Process는 깨끗한 데이터 $x_0$에 점진적으로 가우시안 노이즈를 추가하는 **마르코프 체인**입니다:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I\right)$$

- $x_t$: 시점 $t$에서의 노이즈가 추가된 데이터
- $\beta_t \in (0, 1)$: 시점 $t$의 노이즈 스케줄 (분산 크기)
- $I$: 단위 행렬 (각 차원에 독립적인 노이즈)
- $T$: 총 확산 단계 수 (보통 1000)

**핵심 정의:**

$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \prod_{s=1}^{t}(1 - \beta_s)$$

- $\alpha_t$: 시점 $t$에서 원본 신호의 보존 비율
- $\bar{\alpha}_t$: 시점 $0$부터 $t$까지의 누적 신호 보존 비율 (monotonically decreasing)

### Reparameterization Trick

$T$번의 순차적 노이즈 추가 없이, **한 번에** $x_0$에서 $x_t$를 샘플링할 수 있습니다:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\,x_0,\, (1-\bar{\alpha}_t)I\right)$$

$$\boxed{x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)}$$

| 시점 | $\bar{\alpha}_t$ | $\sqrt{\bar{\alpha}_t}$ (신호 계수) | $\sqrt{1-\bar{\alpha}_t}$ (노이즈 계수) | 의미 |
|------|------------------|-----------------------------------|-----------------------------------------|------|
| $t=0$ | $1.0$ | $1.0$ | $0.0$ | 원본 데이터 |
| $t=T/2$ | $\approx 0.1$ | $\approx 0.32$ | $\approx 0.95$ | 대부분 노이즈 |
| $t=T$ | $\approx 0$ | $\approx 0$ | $\approx 1.0$ | 순수 노이즈 |

### Reverse Process (역방향 생성)

신경망 $\epsilon_\theta$가 노이즈를 예측하여, 노이즈에서 깨끗한 데이터를 복원합니다:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I\right)$$

여기서 평균 $\mu_\theta$는 다음과 같이 계산됩니다:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

- $\epsilon_\theta(x_t, t)$: 신경망이 예측한 노이즈
- $\sigma_t^2 = \beta_t$ 또는 $\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$

### ELBO와 Simple Loss

DDPM의 최적화 목표는 **ELBO(Evidence Lower Bound)**에서 유도됩니다:

$$\mathcal{L}_{VLB} = \mathbb{E}_q\!\left[\underbrace{D_{KL}(q(x_T|x_0)\|p(x_T))}_{L_T} + \sum_{t=2}^{T}\underbrace{D_{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0}\right]$$

Ho et al. (2020)은 이를 **Simple Loss**로 단순화했습니다:

$$\boxed{\mathcal{L}_{simple}(\theta) = \mathbb{E}_{t \sim U[1,T],\, x_0,\, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\, t)\|^2\right]}$$

- $\epsilon \sim \mathcal{N}(0, I)$: 실제 추가된 노이즈
- $\epsilon_\theta(\cdot, t)$: 신경망이 예측한 노이즈
- 시점 $t$는 $\{1, 2, \ldots, T\}$에서 균일하게 샘플링

**요약 표:**

| 과정 | 수식 | 역할 |
|------|------|------|
| Forward | $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$ | 데이터에 노이즈 추가 |
| Reverse | $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta) + \sigma_t z$ | 노이즈 제거 (생성) |
| Simple Loss | $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ | 노이즈 예측 학습 |"""),

# ─── Cell 2: 🐣 친절 설명 ───
md(r"""---

### 🐣 초등학생을 위한 확산 모델 친절 설명!

#### 🔢 확산 모델이 뭔가요?

> 💡 **비유**: 깨끗한 그림 위에 **모래를 조금씩 뿌려서** 결국 아무것도 안 보이게 만드는 과정(Forward)을 상상해 보세요. 확산 모델은 이 과정을 **거꾸로** 배워서, 모래더미에서 그림을 복원하는 마법사예요!

1. **Forward Process (모래 뿌리기)**: 예쁜 고양이 사진에 모래를 1000번 뿌리면 결국 회색 노이즈만 남아요
2. **Reverse Process (모래 치우기)**: AI가 "이 모래더미에서 모래를 한 움큼 치우면 어떤 모양이 나올까?"를 학습해요
3. **생성**: 완전히 랜덤한 모래더미에서 시작해서, 한 움큼씩 치우다 보면... 새로운 고양이 사진이 나타나요! 🐱

#### 🎲 왜 $\bar{\alpha}_t$가 중요한가요?

| 단계 | 비유 | 수학 |
|------|------|------|
| $t=0$ (처음) | 깨끗한 그림 | $\bar{\alpha}_0 = 1$ → 원본 100% |
| $t=500$ (중간) | 그림이 흐릿 | $\bar{\alpha}_{500} \approx 0.1$ → 원본 10% |
| $t=1000$ (끝) | 완전 노이즈 | $\bar{\alpha}_{1000} \approx 0$ → 원본 0% |

$\bar{\alpha}_t$는 "**원본 그림이 얼마나 남아있는지**"를 알려주는 숫자예요!"""),

# ─── Cell 3: 📝 연습 문제 ───
md(r"""---

### 📝 연습 문제

#### 문제 1: Reparameterization 계산

$\beta_1 = 0.0001$, $\beta_2 = 0.0002$일 때, $\bar{\alpha}_2$를 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$$\alpha_1 = 1 - 0.0001 = 0.9999$$
$$\alpha_2 = 1 - 0.0002 = 0.9998$$
$$\bar{\alpha}_2 = \alpha_1 \cdot \alpha_2 = 0.9999 \times 0.9998 = 0.9997$$

$t=2$에서는 원본 신호가 **99.97%** 보존됩니다. 초기에는 노이즈가 매우 천천히 추가됩니다.
</details>

#### 문제 2: 노이즈 계수 비교

$\bar{\alpha}_t = 0.5$일 때, 신호 계수 $\sqrt{\bar{\alpha}_t}$와 노이즈 계수 $\sqrt{1-\bar{\alpha}_t}$를 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$$\sqrt{\bar{\alpha}_t} = \sqrt{0.5} \approx 0.707$$
$$\sqrt{1 - \bar{\alpha}_t} = \sqrt{0.5} \approx 0.707$$

신호와 노이즈가 **정확히 같은 비율**! 이 시점에서 원본과 노이즈가 절반씩 섞여 있습니다.

**참고**: $\sqrt{\bar{\alpha}_t}^2 + \sqrt{1-\bar{\alpha}_t}^2 = \bar{\alpha}_t + (1-\bar{\alpha}_t) = 1$ → 분산이 항상 보존됩니다.
</details>

#### 문제 3: Simple Loss 해석

Simple Loss $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$에서, 완벽한 모델($\epsilon_\theta = \epsilon$)일 때 손실값은?

<details>
<summary>💡 풀이 확인</summary>

$$\mathcal{L} = \|\epsilon - \epsilon\|^2 = \|0\|^2 = 0$$

손실이 0이면 모델이 노이즈를 **완벽하게 예측**하는 것입니다. 실제로는 0에 가까워질수록 좋은 모델이며, 학습 초기에는 값이 크고 점점 줄어듭니다.
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
print(f"TensorFlow 버전: {tf.__version__}")
print(f"GPU 사용 가능: {len(tf.config.list_physical_devices('GPU')) > 0}")"""),

# ─── Cell 5: 베타 스케줄 구현 (Section 2) ───
md(r"""## 2. 베타 스케줄과 알파 누적곱 구현 <a name='2.-베타-스케줄-구현'></a>

DDPM 원논문(Ho et al., 2020; arxiv 2006.11239)에서 사용한 **선형(linear) 베타 스케줄**을 구현합니다:

$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min}), \quad t = 1, \ldots, T$$

원논문 설정: $\beta_{\min} = 10^{-4}$, $\beta_{\max} = 0.02$, $T = 1000$"""),

# ─── Cell 6: 베타 스케줄 코드 ───
code(r"""# ── 선형 베타 스케줄 및 알파 누적곱 계산 ──────────────────────────────
T = 1000
beta_min = 1e-4
beta_max = 0.02

# 선형 스케줄: beta_1 = beta_min, beta_T = beta_max
betas = np.linspace(beta_min, beta_max, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

print(f"베타 스케줄 (선형)")
print(f"  β_1   = {betas[0]:.6f}")
print(f"  β_500 = {betas[499]:.6f}")
print(f"  β_T   = {betas[-1]:.6f}")
print()
print(f"알파 누적곱 (ᾱ_t)")
print(f"  ᾱ_1   = {alpha_bars[0]:.6f}  (원본 {alpha_bars[0]*100:.2f}% 보존)")
print(f"  ᾱ_250 = {alpha_bars[249]:.6f}  (원본 {alpha_bars[249]*100:.2f}% 보존)")
print(f"  ᾱ_500 = {alpha_bars[499]:.6f}  (원본 {alpha_bars[499]*100:.4f}% 보존)")
print(f"  ᾱ_750 = {alpha_bars[749]:.6f}  (원본 {alpha_bars[749]*100:.6f}% 보존)")
print(f"  ᾱ_T   = {alpha_bars[-1]:.8f}  (원본 거의 0%)")
print()
print(f"신호 계수 √ᾱ_T   = {np.sqrt(alpha_bars[-1]):.6f}")
print(f"노이즈 계수 √(1-ᾱ_T) = {np.sqrt(1-alpha_bars[-1]):.6f}")"""),

# ─── Cell 7: 알파바 시각화 ───
code(r"""# ── 베타 스케줄 및 알파 누적곱 시각화 ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (1) 베타 스케줄
ax1 = axes[0]
ax1.plot(range(1, T+1), betas, 'b-', lw=2)
ax1.set_xlabel('Timestep $t$', fontsize=11)
ax1.set_ylabel(r'$\beta_t$', fontsize=11)
ax1.set_title(r'Linear $\beta$ Schedule', fontweight='bold')
ax1.grid(True, alpha=0.3)

# (2) 알파 누적곱
ax2 = axes[1]
ax2.plot(range(1, T+1), alpha_bars, 'r-', lw=2)
ax2.set_xlabel('Timestep $t$', fontsize=11)
ax2.set_ylabel(r'$\bar{\alpha}_t$', fontsize=11)
ax2.set_title(r'Cumulative $\bar{\alpha}_t$', fontweight='bold')
ax2.axhline(y=0.5, color='gray', ls='--', lw=1, label=r'$\bar{\alpha}_t = 0.5$')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# (3) 신호/노이즈 계수
ax3 = axes[2]
ax3.plot(range(1, T+1), np.sqrt(alpha_bars), 'g-', lw=2, label=r'Signal: $\sqrt{\bar{\alpha}_t}$')
ax3.plot(range(1, T+1), np.sqrt(1 - alpha_bars), 'm-', lw=2, label=r'Noise: $\sqrt{1-\bar{\alpha}_t}$')
ax3.set_xlabel('Timestep $t$', fontsize=11)
ax3.set_ylabel('Coefficient', fontsize=11)
ax3.set_title('Signal vs Noise Coefficients', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/beta_schedule.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/beta_schedule.png")"""),

# ─── Cell 8: Forward Noising 시각화 (Section 3) ───
md(r"""## 3. Forward Noising 과정 시각화 <a name='3.-Forward-Noising-시각화'></a>

2D 가우시안 분포에서 시작하여 Forward Process가 어떻게 데이터를 순수 노이즈로 변환하는지 시각화합니다.

$$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$"""),

# ─── Cell 9: Forward Noising 코드 ───
code(r"""# ── Forward Process: 2D 가우시안에 노이즈 추가 ──────────────────────
# 2D 가우시안 혼합 모델 (원본 데이터)
n_samples = 2000

# 3개의 가우시안 클러스터로 "데이터 분포" 생성
centers = np.array([[2.0, 2.0], [-2.0, 2.0], [0.0, -2.0]])
x0_list = []
for c in centers:
    samples = np.random.randn(n_samples // 3, 2) * 0.4 + c
    x0_list.append(samples)
x0 = np.concatenate(x0_list, axis=0).astype(np.float32)

# Forward Process 적용: 여러 시점에서의 x_t
timesteps_to_show = [0, 50, 200, 500, 800, 999]
noised_samples = {}

for t in timesteps_to_show:
    if t == 0:
        noised_samples[t] = x0.copy()
    else:
        abar = alpha_bars[t - 1]
        eps = np.random.randn(*x0.shape).astype(np.float32)
        x_t = np.sqrt(abar) * x0 + np.sqrt(1 - abar) * eps
        noised_samples[t] = x_t

# 시각화
fig, axes = plt.subplots(1, 6, figsize=(18, 3))

for idx, t in enumerate(timesteps_to_show):
    ax = axes[idx]
    data = noised_samples[t]
    ax.scatter(data[:, 0], data[:, 1], s=1, alpha=0.3, c='steelblue')
    abar_val = 1.0 if t == 0 else alpha_bars[t - 1]
    ax.set_title(f'$t={t}$\n' + r'$\bar{\alpha}=$' + f'{abar_val:.4f}', fontweight='bold', fontsize=9)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle('Forward Diffusion Process on 2D Gaussian Mixture', fontweight='bold', fontsize=12, y=1.05)
plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/forward_noising.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/forward_noising.png")
print(f"\n시점별 데이터 통계:")
for t in timesteps_to_show:
    d = noised_samples[t]
    print(f"  t={t:4d} | 평균: ({d[:,0].mean():.3f}, {d[:,1].mean():.3f}) | 표준편차: ({d[:,0].std():.3f}, {d[:,1].std():.3f})")"""),

# ─── Cell 10: 타임스텝별 노이즈 시각화 (Section 4) ───
md(r"""## 4. 다양한 타임스텝에서의 노이즈 시각화 <a name='4.-타임스텝별-노이즈'></a>

1D 신호에 대해 Forward Process를 적용하여, 시점별 **신호 대 노이즈 비율(SNR)**을 시각화합니다.

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$"""),

# ─── Cell 11: 1D 노이즈 시각화 코드 ───
code(r"""# ── 1D 신호에 대한 타임스텝별 노이즈 시각화 ──────────────────────────
# 1D 원본 신호: 사인파
x_axis = np.linspace(0, 4 * np.pi, 200)
signal = np.sin(x_axis).astype(np.float32)

timesteps_demo = [0, 100, 300, 500, 700, 999]

fig, axes = plt.subplots(2, 3, figsize=(15, 7))

for idx, t in enumerate(timesteps_demo):
    row, col = divmod(idx, 3)
    ax = axes[row][col]

    if t == 0:
        noised = signal.copy()
        abar = 1.0
    else:
        abar = alpha_bars[t - 1]
        eps = np.random.randn(*signal.shape).astype(np.float32)
        noised = np.sqrt(abar) * signal + np.sqrt(1 - abar) * eps

    ax.plot(x_axis, signal, 'b-', alpha=0.3, lw=1, label='원본 신호')
    ax.plot(x_axis, noised, 'r-', lw=1.5, label=f'$x_t$ (t={t})')

    snr = abar / (1 - abar + 1e-10)
    ax.set_title(f't={t} | ' + r'$\bar{\alpha}$=' + f'{abar:.4f} | SNR={snr:.2f}', fontweight='bold', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-3.5, 3.5)

plt.suptitle('Forward Process: 1D 사인파에 대한 타임스텝별 노이즈', fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/timestep_noise.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/timestep_noise.png")

# SNR 분석
print("\n타임스텝별 SNR (Signal-to-Noise Ratio):")
print(f"{'시점':>6} | {'ᾱ_t':>10} | {'SNR':>10} | {'SNR(dB)':>10}")
print("-" * 48)
for t in [1, 100, 250, 500, 750, 1000]:
    abar = alpha_bars[t-1]
    snr = abar / (1 - abar + 1e-10)
    snr_db = 10 * np.log10(snr + 1e-10)
    print(f"  t={t:4d} | {abar:10.6f} | {snr:10.4f} | {snr_db:10.2f} dB")"""),

# ─── Cell 12: 1D DDPM 학습 (Section 5) ───
md(r"""## 5. 1D DDPM 학습 루프 구현 <a name='5.-1D-DDPM-학습'></a>

간단한 1D 데이터 분포(가우시안 혼합)에 대해 DDPM의 학습과 샘플링을 구현합니다.

**학습 알고리즘 (Algorithm 1 from Ho et al.):**
1. $x_0 \sim q(x_0)$에서 데이터 샘플링
2. $t \sim \text{Uniform}\{1, \ldots, T\}$에서 시점 샘플링
3. $\epsilon \sim \mathcal{N}(0, I)$에서 노이즈 샘플링
4. 그래디언트 스텝: $\nabla_\theta \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\, t)\|^2$"""),

# ─── Cell 13: 1D DDPM 학습 코드 ───
code(r"""# ── 1D DDPM 학습 및 샘플링 ──────────────────────────────────────────
T_steps = 200
beta_min_1d = 1e-4
beta_max_1d = 0.02
betas_1d = np.linspace(beta_min_1d, beta_max_1d, T_steps).astype(np.float32)
alphas_1d = 1.0 - betas_1d
alpha_bars_1d = np.cumprod(alphas_1d).astype(np.float32)

# 1D 데이터 분포: 2개의 가우시안 혼합
def sample_data(n):
    mix = np.random.choice([0, 1], size=n)
    samples = np.where(mix == 0,
                       np.random.randn(n) * 0.5 + 3.0,
                       np.random.randn(n) * 0.5 - 3.0)
    return samples.astype(np.float32)

# 노이즈 예측 모델 (간단한 MLP)
class NoisePredictor(tf.keras.Model):
    # 1D 노이즈 예측 MLP: (x_t, t_embedding) -> epsilon
    def __init__(self):
        super().__init__()
        self.time_embed = tf.keras.layers.Dense(64, activation='swish')
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='swish'),
            tf.keras.layers.Dense(128, activation='swish'),
            tf.keras.layers.Dense(64, activation='swish'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, x_t, t):
        t_emb = self.time_embed(tf.cast(t[:, None], tf.float32) / T_steps)
        x_input = tf.concat([x_t[:, None], t_emb], axis=-1)
        return self.net(x_input)[:, 0]

model = NoisePredictor()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

alpha_bars_tf = tf.constant(alpha_bars_1d)

# 학습 루프
losses = []
n_epochs = 300
batch_size = 512

for epoch in range(n_epochs):
    x0_batch = sample_data(batch_size)
    t_batch = np.random.randint(0, T_steps, size=batch_size)
    eps_batch = np.random.randn(batch_size).astype(np.float32)

    abar_t = alpha_bars_1d[t_batch]
    x_t = np.sqrt(abar_t) * x0_batch + np.sqrt(1 - abar_t) * eps_batch

    with tf.GradientTape() as tape:
        eps_pred = model(tf.constant(x_t), tf.constant(t_batch))
        loss = tf.reduce_mean((tf.constant(eps_batch) - eps_pred) ** 2)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    losses.append(float(loss))

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d}/{n_epochs} | Loss: {float(loss):.4f}")

print(f"\n최종 학습 Loss: {losses[-1]:.4f}")
print(f"모델 파라미터 수: {sum(np.prod(v.shape) for v in model.trainable_variables):,}")"""),

# ─── Cell 14: DDPM 샘플링 및 결과 시각화 ───
code(r"""# ── DDPM 샘플링 (역방향 과정) 및 결과 비교 ─────────────────────────
# Reverse Process: x_T ~ N(0,1) → x_0
n_gen = 3000
x_t = np.random.randn(n_gen).astype(np.float32)

# 역방향 샘플링
for t in reversed(range(T_steps)):
    t_tensor = tf.constant(np.full(n_gen, t, dtype=np.int32))
    eps_pred = model(tf.constant(x_t), t_tensor).numpy()

    alpha_t = alphas_1d[t]
    abar_t = alpha_bars_1d[t]
    beta_t = betas_1d[t]

    # 평균 계산: mu_theta
    mu = (1.0 / np.sqrt(alpha_t)) * (x_t - (beta_t / np.sqrt(1 - abar_t)) * eps_pred)

    if t > 0:
        z = np.random.randn(n_gen).astype(np.float32)
        sigma = np.sqrt(beta_t)
        x_t = mu + sigma * z
    else:
        x_t = mu

generated = x_t

# 시각화: 원본 분포 vs 생성 분포
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (1) 학습 Loss 곡선
ax1 = axes[0]
ax1.plot(losses, 'b-', lw=1, alpha=0.5)
window = 20
smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
ax1.plot(range(window-1, len(losses)), smoothed, 'r-', lw=2, label=f'{window}-에폭 이동평균')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('학습 Loss 곡선', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) 원본 vs 생성 분포
real_data = sample_data(n_gen)
ax2 = axes[1]
ax2.hist(real_data, bins=80, density=True, alpha=0.6, color='steelblue', label='원본 분포')
ax2.hist(generated, bins=80, density=True, alpha=0.6, color='coral', label='생성 분포')
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('원본 vs DDPM 생성 분포', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# (3) 학습 곡선 (로그 스케일)
ax3 = axes[2]
ax3.semilogy(smoothed, 'r-', lw=2)
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Loss (log)', fontsize=11)
ax3.set_title('Loss 곡선 (로그 스케일)', fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/ddpm_1d_result.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/ddpm_1d_result.png")

# 통계 비교
print(f"\n분포 통계 비교:")
print(f"{'':>12} | {'평균':>8} | {'표준편차':>8} | {'최소':>8} | {'최대':>8}")
print("-" * 55)
print(f"{'원본 분포':>12} | {real_data.mean():>8.3f} | {real_data.std():>8.3f} | {real_data.min():>8.3f} | {real_data.max():>8.3f}")
print(f"{'생성 분포':>12} | {generated.mean():>8.3f} | {generated.std():>8.3f} | {generated.min():>8.3f} | {generated.max():>8.3f}")"""),

# ─── Cell 15: 정리 (Section 6) ───
md(r"""## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| Forward Process | $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$ — 데이터에 점진적 노이즈 추가 | ⭐⭐⭐ |
| Reparameterization | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ — 한 번에 $x_t$ 계산 | ⭐⭐⭐ |
| Reverse Process | $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta)$ — 노이즈 제거 | ⭐⭐⭐ |
| Simple Loss | $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ — ELBO에서 유도된 단순화 목적함수 | ⭐⭐⭐ |
| 베타 스케줄 | $\beta_t$의 스케줄이 생성 품질에 결정적 영향 | ⭐⭐ |
| $\bar\alpha_t$ 누적곱 | Forward Process의 핵심 — 원본 신호 보존 비율 | ⭐⭐⭐ |

### 핵심 수식

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar\alpha_t}\,x_0,\, (1-\bar\alpha_t)I\right)$$

$$\mathcal{L}_{simple} = \mathbb{E}_{t,x_0,\epsilon}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\, t)\|^2\right]$$

### 다음 챕터 예고
**02_noise_schedules_and_samplers** — Linear/Cosine/EDM 노이즈 스케줄 비교와 DDIM 가속 샘플링을 다룹니다."""),
]

if __name__ == '__main__':
    create_notebook(cells, 'chapter13_genai_diffusion/01_ddpm_theory_and_math.ipynb')
