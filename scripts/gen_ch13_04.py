#!/usr/bin/env python3
"""Generate chapter13_genai_diffusion/04_conditional_diffusion_cfg.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ─────────────────────────────────────────────
cells.append(md(r"""# Chapter 13: 생성 AI 심화 — Classifier-Free Guidance (CFG)

## 학습 목표
- **Classifier Guidance**와 **Classifier-Free Guidance**의 차이를 수학적으로 이해한다
- CFG 수식 $\tilde\epsilon_\theta(x_t,c) = \epsilon_\theta(x_t) + w[\epsilon_\theta(x_t,c) - \epsilon_\theta(x_t)]$를 유도한다
- **Guidance Scale $w$**가 생성 품질(Quality)과 다양성(Diversity)에 미치는 트레이드오프를 실험한다
- **ControlNet** 조건부 제어 아키텍처의 핵심 원리를 이해한다
- TensorFlow로 CFG 노이즈 예측 시뮬레이션과 Guidance Scale 실험을 구현한다

## 목차
1. [수학적 기초: Classifier-Free Guidance](#1.-수학적-기초)
2. [CFG 노이즈 예측 시뮬레이션](#2.-CFG-시뮬레이션)
3. [Guidance Scale Sweep 시각화](#3.-Guidance-Scale-Sweep)
4. [품질 vs 다양성 트레이드오프](#4.-품질-vs-다양성)
5. [ControlNet 아키텍처 개요](#5.-ControlNet-개요)
6. [정리](#6.-정리)"""))

# ── Cell 2: Math Section ─────────────────────────────────────────
cells.append(md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### Classifier Guidance (Dhariwal & Nichol, 2021)

조건부 생성의 첫 번째 접근은 별도의 **분류기(classifier)** $p_\phi(c|x_t)$를 활용하는 것입니다:

$$\nabla_{x_t} \log p(x_t | c) = \nabla_{x_t} \log p(x_t) + \gamma \nabla_{x_t} \log p_\phi(c | x_t)$$

- $\nabla_{x_t} \log p(x_t)$: 비조건부 스코어 함수
- $\nabla_{x_t} \log p_\phi(c | x_t)$: 분류기의 그래디언트
- $\gamma$: guidance 강도

**한계**: 별도의 노이즈-강건 분류기를 학습시켜야 하며, 분류기 학습이 불안정할 수 있습니다.

### Classifier-Free Guidance (Ho & Salimans, 2022)

분류기 없이 **하나의 모델**로 조건부/비조건부 예측을 모두 수행합니다:

$$\boxed{\tilde\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + w \cdot \left[\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\right]}$$

- $\epsilon_\theta(x_t, c)$: 조건 $c$를 받은 노이즈 예측
- $\epsilon_\theta(x_t, \varnothing)$: 빈 조건(null condition)으로의 비조건부 노이즈 예측
- $w$: **guidance scale** (가이던스 스케일)
- $\varnothing$: 학습 시 일정 확률(보통 10~20%)로 조건을 드롭하여 학습

이를 정리하면:

$$\tilde\epsilon_\theta = (1 - w) \cdot \epsilon_\theta(x_t, \varnothing) + w \cdot \epsilon_\theta(x_t, c)$$

| $w$ 값 | 의미 | 효과 |
|---------|------|------|
| $w = 0$ | 비조건부 생성 | 최대 다양성, 조건 무시 |
| $w = 1$ | 표준 조건부 생성 | 조건 반영, 보통 품질 |
| $w > 1$ | 과도한 조건 강조 | 높은 품질(FID↓), 낮은 다양성 |
| $w = 7.5$ | Stable Diffusion 기본값 | 품질-다양성 균형점 |

### 수학적 해석: Score Function 관점

CFG는 스코어 함수 관점에서 다음과 같이 해석됩니다:

$$\nabla_{x_t} \log p_w(x_t | c) = (1-w)\nabla_{x_t}\log p(x_t) + w\nabla_{x_t}\log p(x_t|c)$$

$$= \nabla_{x_t}\log p(x_t) + w\nabla_{x_t}\log p(c|x_t)$$

이는 **implicit classifier** $p(c|x_t)$의 그래디언트를 $w$배 강화하는 것과 동일합니다.

**요약 표:**

| 구분 | 수식 | 핵심 |
|------|------|------|
| Classifier Guidance | $\nabla \log p(x_t) + \gamma \nabla \log p_\phi(c\|x_t)$ | 별도 분류기 필요 |
| Classifier-Free Guidance | $\epsilon_\theta(\varnothing) + w[\epsilon_\theta(c) - \epsilon_\theta(\varnothing)]$ | 단일 모델, 조건 드롭 |
| Score 해석 | $\nabla\log p(x_t) + w\nabla\log p(c\|x_t)$ | 암시적 분류기 강화 |"""))

# ── Cell 3: 🐣 Friendly Explanation ──────────────────────────────
cells.append(md(r"""---

### 🐣 초등학생을 위한 CFG 친절 설명!

#### 🎨 Classifier-Free Guidance가 뭔가요?

> 💡 **비유**: 그림 그리기 대회를 상상해보세요!

선생님(모델)에게 "고양이를 그려주세요"라고 말하면, 선생님은 두 가지 그림을 머릿속에 떠올립니다:

1. 🎲 **아무거나 그린 그림** (비조건부): 그냥 자유롭게 그린 그림
2. 🐱 **고양이 그림** (조건부): "고양이"라는 조건을 듣고 그린 그림

**Guidance Scale $w$**는 "고양이스러움"을 **얼마나 강조할지** 정하는 다이얼입니다:

| 다이얼 | 결과 |
|--------|------|
| $w = 0$ | 그냥 자유화 → 고양이 아닐 수도! |
| $w = 1$ | 보통 고양이 그림 |
| $w = 7$ | 아주 고양이스러운 그림! 🐱✨ |
| $w = 20$ | 너무 고양이만 강조 → 부자연스러운 그림 😵 |

$$\text{최종 그림} = \text{자유화} + w \times (\text{고양이 그림} - \text{자유화})$$

즉, **"고양이 느낌"만 뽑아서 $w$배로 강화**하는 겁니다!"""))

# ── Cell 4: 📝 Practice Problems ──────────────────────────────────
cells.append(md(r"""---

### 📝 연습 문제

#### 문제 1: CFG 출력 계산

모델이 다음 노이즈를 예측했습니다:
- 비조건부: $\epsilon_\theta(x_t, \varnothing) = [0.3, -0.5, 0.8]$
- 조건부: $\epsilon_\theta(x_t, c) = [0.1, -0.2, 1.2]$

Guidance scale $w = 3.0$일 때, CFG 결합 노이즈 $\tilde\epsilon$을 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$$\tilde\epsilon = \epsilon_\theta(\varnothing) + w[\epsilon_\theta(c) - \epsilon_\theta(\varnothing)]$$

$$= [0.3, -0.5, 0.8] + 3.0 \times ([0.1, -0.2, 1.2] - [0.3, -0.5, 0.8])$$

$$= [0.3, -0.5, 0.8] + 3.0 \times [-0.2, 0.3, 0.4]$$

$$= [0.3, -0.5, 0.8] + [-0.6, 0.9, 1.2]$$

$$= [-0.3, 0.4, 2.0]$$

$w > 1$이므로 조건부 방향이 **과강조**되어, 비조건부 예측보다 더 큰 폭으로 이동합니다.
</details>

#### 문제 2: Guidance Scale 효과

$w = 1$과 $w = 0$에서 각각 $\tilde\epsilon$은 무엇과 같아지는지 서술하세요.

<details>
<summary>💡 풀이 확인</summary>

- $w = 1$: $\tilde\epsilon = \epsilon_\theta(\varnothing) + 1 \cdot [\epsilon_\theta(c) - \epsilon_\theta(\varnothing)] = \epsilon_\theta(c)$
  → **표준 조건부 예측**과 동일

- $w = 0$: $\tilde\epsilon = \epsilon_\theta(\varnothing) + 0 \cdot [\cdots] = \epsilon_\theta(\varnothing)$
  → **비조건부 예측**과 동일

즉, $w$는 비조건부($w=0$)와 조건부($w=1$) 사이를 **연속적으로 보간**하며, $w > 1$은 조건 방향으로의 **외삽(extrapolation)**입니다.
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

# ── Cell 6: CFG Noise Prediction Simulation ──────────────────────
cells.append(md(r"""## 2. CFG 노이즈 예측 시뮬레이션 <a name='2.-CFG-시뮬레이션'></a>

간단한 신경망으로 조건부/비조건부 노이즈 예측을 시뮬레이션하고, CFG 공식을 적용합니다."""))

cells.append(code(r"""# ── CFG 노이즈 예측 시뮬레이션 ────────────────────────────────────
# 2D 가우시안 혼합 데이터로 CFG 시뮬레이션

data_dim = 2
n_timesteps = 50

# 베타 스케줄
betas = np.linspace(1e-4, 0.02, n_timesteps).astype(np.float32)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

# 간단한 조건부/비조건부 노이즈 예측 모델
class SimpleCFGModel(tf.keras.Model):
    # CFG 시뮬레이션을 위한 간단한 MLP 모델
    def __init__(self, dim=2, cond_dim=4):
        super().__init__()
        self.uncond_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(dim)
        ])
        self.cond_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(dim)
        ])
        self.time_embed = tf.keras.layers.Dense(16, activation='relu')

    def call(self, x_t, t_emb, condition=None):
        t_feat = self.time_embed(t_emb)
        if condition is None:
            inp = tf.concat([x_t, t_feat], axis=-1)
            return self.uncond_net(inp)
        else:
            inp = tf.concat([x_t, t_feat, condition], axis=-1)
            return self.cond_net(inp)

model = SimpleCFGModel(dim=data_dim, cond_dim=4)

# 더미 입력으로 모델 빌드
dummy_x = tf.zeros((1, data_dim))
dummy_t = tf.zeros((1, 1))
dummy_c = tf.zeros((1, 4))
_ = model(dummy_x, dummy_t, condition=None)
_ = model(dummy_x, dummy_t, condition=dummy_c)

print(f"모델 파라미터 수 (비조건부): {sum(np.prod(w.shape) for w in model.uncond_net.trainable_weights)}")
print(f"모델 파라미터 수 (조건부):   {sum(np.prod(w.shape) for w in model.cond_net.trainable_weights)}")

# CFG 적용 함수
def apply_cfg(model, x_t, t_emb, condition, guidance_scale):
    # 비조건부 예측
    eps_uncond = model(x_t, t_emb, condition=None)
    # 조건부 예측
    eps_cond = model(x_t, t_emb, condition=condition)
    # CFG 공식
    eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    return eps_cfg, eps_uncond, eps_cond

# 테스트
test_x = tf.random.normal((5, data_dim))
test_t = tf.constant([[0.5]] * 5)
test_c = tf.one_hot([0, 1, 2, 3, 0], depth=4)

for w in [0.0, 1.0, 3.0, 7.5]:
    eps_cfg, eps_u, eps_c = apply_cfg(model, test_x, test_t, test_c, w)
    magnitude = tf.reduce_mean(tf.norm(eps_cfg, axis=-1)).numpy()
    print(f"  w = {w:>4.1f} → CFG 노이즈 평균 크기: {magnitude:.4f}")"""))

# ── Cell 7: Guidance Scale Sweep ─────────────────────────────────
cells.append(md(r"""## 3. Guidance Scale Sweep 시각화 <a name='3.-Guidance-Scale-Sweep'></a>

다양한 guidance scale $w$에서 CFG 노이즈 방향이 어떻게 변하는지 2D로 시각화합니다."""))

cells.append(code(r"""# ── Guidance Scale Sweep 시각화 ────────────────────────────────────
# 2D에서 비조건부 vs 조건부 vs CFG 노이즈 방향 비교

n_points = 200
np.random.seed(42)

# 2D 포인트 생성
x_points = np.random.randn(n_points, 2).astype(np.float32) * 1.5

# 시뮬레이션: 비조건부 = 원점 방향, 조건부 = 특정 클래스 중심 방향
target_center = np.array([3.0, 2.0])
eps_uncond = -x_points * 0.3 + np.random.randn(n_points, 2).astype(np.float32) * 0.2
eps_cond = (target_center - x_points) * 0.3 + np.random.randn(n_points, 2).astype(np.float32) * 0.1

guidance_scales = [0.0, 1.0, 3.0, 7.5]
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for idx, w in enumerate(guidance_scales):
    ax = axes[idx]
    eps_cfg = eps_uncond + w * (eps_cond - eps_uncond)

    ax.scatter(x_points[:, 0], x_points[:, 1], c='gray', s=10, alpha=0.5, label='현재 위치')
    scale = 0.8
    ax.quiver(x_points[::5, 0], x_points[::5, 1],
              eps_cfg[::5, 0] * scale, eps_cfg[::5, 1] * scale,
              color='blue', alpha=0.6, scale=15, width=0.004)
    ax.scatter(*target_center, c='red', s=150, marker='*', zorder=5, label='조건 타겟')
    ax.set_xlim(-5, 7)
    ax.set_ylim(-5, 7)
    ax.set_title(f'w = {w}', fontweight='bold', fontsize=13)
    ax.set_xlabel('$x_1$', fontsize=11)
    ax.set_ylabel('$x_2$', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/cfg_guidance_scale_sweep.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/cfg_guidance_scale_sweep.png")
print(f"Guidance scale 값: {guidance_scales}")
print("w=0: 비조건부 (발산), w=1: 표준 조건부, w=7.5: 강한 조건 방향")"""))

# ── Cell 8: Quality vs Diversity Tradeoff ────────────────────────
cells.append(md(r"""## 4. 품질 vs 다양성 트레이드오프 <a name='4.-품질-vs-다양성'></a>

Guidance scale $w$를 높이면:
- **FID(품질) ↓** — 조건에 더 잘 맞는 샘플 생성 (좋음)
- **다양성 ↓** — 모드 붕괴(mode collapse)에 가까워짐 (나쁨)

이 트레이드오프를 시뮬레이션합니다."""))

cells.append(code(r"""# ── 품질 vs 다양성 트레이드오프 곡선 ──────────────────────────────
# 시뮬레이션: w 증가 → 품질↑(FID↓), 다양성↓

w_values = np.linspace(0, 20, 100)

# FID 시뮬레이션 (낮을수록 좋음): w=0에서 높고, w=7~8에서 최저, 이후 다시 상승
fid_sim = 50 * np.exp(-0.5 * w_values) + 5 + 0.3 * (w_values - 7.5)**2
fid_sim = np.clip(fid_sim, 3, 100)

# 다양성 시뮬레이션 (높을수록 좋음): w 증가하면 감소
diversity_sim = 1.0 / (1.0 + 0.15 * w_values**1.5)

# CLIP Score 시뮬레이션 (높을수록 좋음): w 증가하면 상승 후 포화
clip_score_sim = 0.35 * (1 - np.exp(-0.4 * w_values)) + 0.15

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# FID
ax1 = axes[0]
ax1.plot(w_values, fid_sim, 'b-', lw=2.5, label='FID (↓ 좋음)')
ax1.axvline(x=7.5, color='red', ls='--', lw=1.5, alpha=0.7, label='w=7.5 (SD 기본)')
best_fid_w = w_values[np.argmin(fid_sim)]
ax1.axvline(x=best_fid_w, color='green', ls=':', lw=1.5, label=f'최적 w≈{best_fid_w:.1f}')
ax1.set_xlabel('Guidance Scale (w)', fontsize=11)
ax1.set_ylabel('FID', fontsize=11)
ax1.set_title('FID vs Guidance Scale', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 다양성
ax2 = axes[1]
ax2.plot(w_values, diversity_sim, 'g-', lw=2.5, label='다양성 (↑ 좋음)')
ax2.axvline(x=7.5, color='red', ls='--', lw=1.5, alpha=0.7, label='w=7.5')
ax2.fill_between(w_values, diversity_sim, alpha=0.1, color='green')
ax2.set_xlabel('Guidance Scale (w)', fontsize=11)
ax2.set_ylabel('Diversity (정규화)', fontsize=11)
ax2.set_title('Diversity vs Guidance Scale', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# CLIP Score
ax3 = axes[2]
ax3.plot(w_values, clip_score_sim, 'r-', lw=2.5, label='CLIP Score (↑ 좋음)')
ax3.axvline(x=7.5, color='red', ls='--', lw=1.5, alpha=0.7, label='w=7.5')
ax3.set_xlabel('Guidance Scale (w)', fontsize=11)
ax3.set_ylabel('CLIP Score', fontsize=11)
ax3.set_title('CLIP Score vs Guidance Scale', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/cfg_quality_diversity_tradeoff.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/cfg_quality_diversity_tradeoff.png")

# 수치 비교 표
print(f"\n{'Guidance Scale w':<20} | {'FID (↓)':>10} | {'Diversity':>10} | {'CLIP (↑)':>10}")
print("-" * 58)
for w_test in [0, 1, 3, 5, 7.5, 10, 15, 20]:
    idx = np.argmin(np.abs(w_values - w_test))
    print(f"w = {w_test:<14.1f} | {fid_sim[idx]:>10.1f} | {diversity_sim[idx]:>10.3f} | {clip_score_sim[idx]:>10.3f}")"""))

# ── Cell 9: Pareto Curve ─────────────────────────────────────────
cells.append(code(r"""# ── 품질-다양성 파레토 프론티어 ────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

scatter = ax.scatter(diversity_sim, fid_sim, c=w_values, cmap='viridis', s=30, alpha=0.8)
cbar = plt.colorbar(scatter, ax=ax, label='Guidance Scale (w)')

# 특정 w 값 강조
for w_mark in [0, 1, 3, 7.5, 15]:
    idx = np.argmin(np.abs(w_values - w_mark))
    ax.annotate(f'w={w_mark}', (diversity_sim[idx], fid_sim[idx]),
                fontsize=10, fontweight='bold',
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red'))

ax.set_xlabel('Diversity (↑ 좋음)', fontsize=11)
ax.set_ylabel('FID (↓ 좋음)', fontsize=11)
ax.set_title('Quality-Diversity Pareto Front (CFG)', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/cfg_pareto_front.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/cfg_pareto_front.png")
print("파레토 프론트: w를 조절하면 품질-다양성 곡선 위를 이동합니다")
print(f"최적 FID 지점: w ≈ {w_values[np.argmin(fid_sim)]:.1f} (FID = {np.min(fid_sim):.1f})")"""))

# ── Cell 10: ControlNet Section ──────────────────────────────────
cells.append(md(r"""## 5. ControlNet 아키텍처 개요 <a name='5.-ControlNet-개요'></a>

### ControlNet (Zhang et al., 2023)

ControlNet은 **추가 조건(에지맵, 깊이맵, 포즈 등)**으로 확산 모델을 제어하는 아키텍처입니다.

#### 핵심 수식

기존 U-Net의 가중치 $\Theta$를 복제한 학습 가능한 사본 $\Theta_c$를 만들고, **zero convolution**으로 연결합니다:

$$y_c = F(x;\, \Theta) + \mathcal{Z}(F(x + \mathcal{Z}(c;\, \Theta_{z1});\, \Theta_c);\, \Theta_{z2})$$

- $F(\cdot;\, \Theta)$: 원본 U-Net 블록 (가중치 동결)
- $F(\cdot;\, \Theta_c)$: 복제된 학습 가능한 U-Net 블록
- $\mathcal{Z}(\cdot;\, \Theta_z)$: **Zero Convolution** — $1 \times 1$ Conv, 가중치/바이어스 모두 $0$으로 초기화
- $c$: 추가 조건 입력 (에지맵, 깊이맵, 세그멘테이션 등)

#### 학습 안정성의 비밀: Zero Convolution

$$\mathcal{Z}(x;\, \{W, b\}) = W \cdot x + b, \quad W = 0,\; b = 0$$

학습 시작 시 $\mathcal{Z}$의 출력이 $0$이므로, 원본 모델의 출력에 영향을 주지 않아 **안정적으로 학습**이 시작됩니다.

| 구성 요소 | 역할 | 학습 여부 |
|-----------|------|-----------|
| 원본 U-Net | 기존 생성 능력 유지 | ❄️ 동결 |
| 복제 U-Net | 추가 조건 학습 | 🔥 학습 가능 |
| Zero Conv | 안전한 연결 | 🔥 학습 가능 |
| 조건 인코더 | 조건 입력 처리 | 🔥 학습 가능 |"""))

# ── Cell 11: ControlNet Architecture Diagram ─────────────────────
cells.append(code(r"""# ── ControlNet 아키텍처 다이어그램 ─────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('ControlNet Architecture Overview', fontweight='bold', fontsize=14, pad=15)

# 색상 정의
frozen_color = '#4A90D9'
trainable_color = '#E8744F'
zero_color = '#50C878'
arrow_color = '#333333'

box_style = dict(boxstyle='round,pad=0.5', facecolor=frozen_color, edgecolor='black', alpha=0.8)
train_style = dict(boxstyle='round,pad=0.5', facecolor=trainable_color, edgecolor='black', alpha=0.8)
zero_style = dict(boxstyle='round,pad=0.3', facecolor=zero_color, edgecolor='black', alpha=0.8)

# 원본 U-Net (상단, 동결)
ax.text(4, 8.5, '원본 U-Net Encoder\n(❄️ 동결)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white', bbox=box_style)
ax.text(10, 8.5, '원본 U-Net Decoder\n(❄️ 동결)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white', bbox=box_style)

# 화살표: 인코더 → 디코더
ax.annotate('', xy=(7.5, 8.5), xytext=(6.3, 8.5),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
ax.text(7.0, 9.0, 'skip connection', ha='center', fontsize=8, style='italic')

# 복제 U-Net (하단, 학습 가능)
ax.text(4, 4.5, 'ControlNet 복제 블록\n(🔥 학습 가능)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white', bbox=train_style)

# 조건 입력
ax.text(1, 2, '조건 입력 c\n(에지맵/깊이맵)', ha='center', va='center',
        fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='#FFD700', alpha=0.8))

# Zero Conv 1
ax.text(2.5, 3.5, 'Z₁\n(zero)', ha='center', va='center',
        fontsize=9, fontweight='bold', color='white', bbox=zero_style)
ax.annotate('', xy=(2.5, 4.0), xytext=(1.5, 2.5),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

# 입력 x_t
ax.text(1, 6.5, 'x_t\n(노이즈 입력)', ha='center', va='center',
        fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='#D3D3D3', alpha=0.8))
ax.annotate('', xy=(2.5, 8.2), xytext=(1.5, 7.0),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
ax.annotate('', xy=(2.5, 5.0), xytext=(1.5, 6.5),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

# Zero Conv 2
ax.text(7, 4.5, 'Z₂\n(zero)', ha='center', va='center',
        fontsize=9, fontweight='bold', color='white', bbox=zero_style)
ax.annotate('', xy=(6.5, 4.5), xytext=(5.8, 4.5),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

# 결합 화살표
ax.annotate('', xy=(8.5, 7.5), xytext=(7.5, 5.0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5, ls='--'))
ax.text(8.5, 6.0, '+ (합산)', ha='center', fontsize=10, fontweight='bold', color='red')

# 출력
ax.text(12.5, 6.5, '최종 출력\nε_θ(x_t, t, c)', ha='center', va='center',
        fontsize=10, fontweight='bold', bbox=dict(boxstyle='round', facecolor='#E8E8E8', alpha=0.8))
ax.annotate('', xy=(11.5, 7.0), xytext=(11.0, 8.0),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

# 범례
legend_y = 1.0
ax.add_patch(plt.Rectangle((8, legend_y-0.15), 0.4, 0.3, fc=frozen_color, ec='black', alpha=0.8))
ax.text(8.6, legend_y, '동결 (Frozen)', fontsize=9, va='center')
ax.add_patch(plt.Rectangle((10, legend_y-0.15), 0.4, 0.3, fc=trainable_color, ec='black', alpha=0.8))
ax.text(10.6, legend_y, '학습 가능', fontsize=9, va='center')
ax.add_patch(plt.Rectangle((12, legend_y-0.15), 0.4, 0.3, fc=zero_color, ec='black', alpha=0.8))
ax.text(12.6, legend_y, 'Zero Conv', fontsize=9, va='center')

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/controlnet_architecture.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/controlnet_architecture.png")
print("\nControlNet 핵심:")
print("  1. 원본 U-Net 가중치는 동결 → 기존 생성 능력 보존")
print("  2. 복제 블록이 추가 조건을 학습")
print("  3. Zero Convolution으로 안정적 학습 시작 (초기 출력 = 0)")"""))

# ── Cell 12: Zero Conv Demo ──────────────────────────────────────
cells.append(code(r"""# ── Zero Convolution 동작 시뮬레이션 ──────────────────────────────
# Zero Convolution의 학습 과정 시뮬레이션

class ZeroConv(tf.keras.layers.Layer):
    # 가중치와 바이어스를 0으로 초기화하는 1x1 Conv
    def __init__(self, out_channels):
        super().__init__()
        self.conv = tf.keras.layers.Dense(out_channels,
                                           kernel_initializer='zeros',
                                           bias_initializer='zeros')

    def call(self, x):
        return self.conv(x)

zero_conv = ZeroConv(out_channels=8)

# 초기 상태: 출력이 0
test_input = tf.random.normal((4, 16))
initial_output = zero_conv(test_input)

print("Zero Convolution 초기 상태:")
print(f"  입력 크기: {test_input.shape}")
print(f"  출력 크기: {initial_output.shape}")
print(f"  출력 최대값: {tf.reduce_max(tf.abs(initial_output)).numpy():.6f}")
print(f"  출력이 0인가? {tf.reduce_all(initial_output == 0).numpy()}")

# 학습 시뮬레이션: 그래디언트 업데이트 후
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
target = tf.random.normal((4, 8))

for step in range(5):
    with tf.GradientTape() as tape:
        out = zero_conv(test_input)
        loss = tf.reduce_mean((out - target) ** 2)
    grads = tape.gradient(loss, zero_conv.trainable_variables)
    optimizer.apply_gradients(zip(grads, zero_conv.trainable_variables))

updated_output = zero_conv(test_input)
print(f"\n5 스텝 학습 후:")
print(f"  출력 최대값: {tf.reduce_max(tf.abs(updated_output)).numpy():.4f}")
print(f"  손실: {loss.numpy():.4f}")
print("  → 학습이 진행되면서 점진적으로 조건 정보를 주입!")"""))

# ── Cell 13: Summary ─────────────────────────────────────────────
cells.append(md(r"""## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| Classifier Guidance | 별도 분류기 그래디언트로 조건부 생성 | ⭐⭐ |
| Classifier-Free Guidance | 단일 모델, 조건 드롭으로 조건부/비조건부 학습 | ⭐⭐⭐ |
| Guidance Scale $w$ | 품질-다양성 트레이드오프 제어 | ⭐⭐⭐ |
| ControlNet | Zero Conv로 추가 조건 제어 | ⭐⭐⭐ |
| Zero Convolution | 0 초기화로 안정적 학습 시작 | ⭐⭐ |

### 핵심 수식

$$\tilde\epsilon_\theta(x_t, c) = \underbrace{\epsilon_\theta(x_t, \varnothing)}_{\text{비조건부}} + w \cdot \underbrace{\left[\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\right]}_{\text{조건 방향}}$$

$$y_c = F(x;\, \Theta) + \mathcal{Z}\!\left(F\!\left(x + \mathcal{Z}(c;\, \Theta_{z1});\, \Theta_c\right);\, \Theta_{z2}\right)$$

### 다음 챕터 예고
**Chapter 13-05: Score Matching과 SDE** — Score function, Langevin dynamics, 연속 시간 SDE/ODE 통합 프레임워크를 학습합니다."""))

# ── Create notebook ──────────────────────────────────────────────
path = '/workspace/chapter13_genai_diffusion/04_conditional_diffusion_cfg.ipynb'
create_notebook(cells, path)
