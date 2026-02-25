"""Generate chapter15_alignment_rlhf/03_rlhf_pipeline_overview.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 15: AI 얼라인먼트와 강화학습 — RLHF 파이프라인 개요

## 학습 목표
- InstructGPT의 SFT → Reward Model → PPO 3단계 아키텍처를 이해한다
- Bradley-Terry 선호 모델의 수식을 도출하고 구현한다
- Reward Model의 학습 손실 함수를 유도하고 훈련 과정을 구현한다
- RLHF의 PPO 단계에서 KL 페널티가 하는 역할을 수식으로 설명한다
- 전체 RLHF 파이프라인을 시뮬레이션으로 구현한다

## 목차
1. [수학적 기초: Bradley-Terry 모델과 RLHF 수식](#1.-수학적-기초)
2. [SFT → RM → PPO 파이프라인 시뮬레이션](#2.-RLHF-파이프라인)
3. [Bradley-Terry 선호 모델 구현](#3.-Bradley-Terry-모델)
4. [Reward Model 학습](#4.-Reward-Model-학습)
5. [KL Divergence 페널티 시각화](#5.-KL-Divergence-페널티)
6. [정리](#6.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### Bradley-Terry 선호 모델

두 응답 $y_w$(선호), $y_l$(비선호)에 대한 선호 확률:

$$P(y_w \succ y_l \mid x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

- $\sigma(z) = \frac{1}{1 + e^{-z}}$: 시그모이드 함수
- $r(x, y)$: 보상 모델이 출력하는 스칼라 점수
- $y_w \succ y_l$: $y_w$가 $y_l$보다 선호됨

### Reward Model 학습 손실

$$\mathcal{L}_{RM}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]$$

- 선호 응답의 보상을 비선호 응답보다 **높이는** 방향으로 학습

### RLHF 목적함수 (PPO 단계)

$$\max_\theta \; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot|x)} \left[r_\phi(x, y)\right] - \beta \, D_{KL}\left[\pi_\theta(\cdot|x) \;\|\; \pi_{ref}(\cdot|x)\right]$$

- $r_\phi(x, y)$: 학습된 Reward Model의 점수
- $\pi_{ref}$: SFT 모델 (기준 정책)
- $\beta$: KL 페널티 계수

### InstructGPT 3단계

| 단계 | 입력 | 출력 | 손실함수 |
|------|------|------|---------|
| **Step 1: SFT** | $(x, y^*)$ 시범 데이터 | $\pi_{SFT}$ | $-\log P(y^* \mid x)$ (교차 엔트로피) |
| **Step 2: RM** | $(x, y_w, y_l)$ 비교 쌍 | $r_\theta$ | $-\log\sigma(r_\theta(y_w) - r_\theta(y_l))$ |
| **Step 3: PPO** | $x$ 프롬프트 | $\pi_\theta$ | $\mathbb{E}[r_\phi(y)] - \beta D_{KL}$ |

**요약 표:**

| 구분 | 수식 | 설명 |
|------|------|------|
| Bradley-Terry | $P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$ | 쌍별 비교 선호 모델 |
| RM Loss | $-\mathbb{E}[\log\sigma(\Delta r)]$ | 선호 점수 차이 최대화 |
| RLHF Objective | $\mathbb{E}[r(y)] - \beta D_{KL}$ | 보상 최대화 + 안정성 |

---"""))

# ── Cell 3: 🐣 friendly explanation ──────────────────────────────
cells.append(md("""\
### 🐣 초등학생을 위한 RLHF 친절 설명!

#### 🔢 RLHF가 뭔가요?

> 💡 **비유**: AI를 강아지 훈련시키는 것과 비슷해요!

**1단계 - SFT (따라하기 훈련):** 먼저 강아지에게 "앉아"를 여러 번 보여주면서 따라하게 해요.
이것은 사람이 쓴 좋은 답변을 AI에게 보여주고 따라쓰게 하는 거예요.

**2단계 - Reward Model (점수판 만들기):** 강아지가 한 행동 중에서 "이게 더 좋아!"라고 
사람이 골라주면, AI가 "무엇이 좋은 행동인지" 배워요. 이것이 보상 모델이에요.

**3단계 - PPO (자유 연습):** 강아지가 자유롭게 행동하면서, 점수판(보상 모델)을 보고
스스로 더 좋은 행동을 찾아가요. 단, 너무 엉뚱한 행동은 안 되도록 제한해요(KL 페널티).

#### 🏆 Bradley-Terry가 뭔가요?

> 💡 **비유**: 축구 팀 랭킹을 생각해 보세요!

두 팀이 시합하면 점수가 높은 팀이 이길 확률이 높죠? Bradley-Terry 모델은:
- "A팀 점수가 100이고 B팀 점수가 80이면, A가 이길 확률은 얼마?"를 계산해요
- 마찬가지로 "답변 A의 점수가 높고 답변 B의 점수가 낮으면, A가 선호될 확률은?"을 계산해요

---"""))

# ── Cell 4: 📝 Exercises ──────────────────────────────────────────
cells.append(md(r"""\
### 📝 연습 문제

#### 문제 1: Bradley-Terry 확률 계산

보상 모델이 두 응답에 대해 $r(x, y_w) = 2.0$, $r(x, y_l) = 0.5$를 출력했습니다.
$y_w$가 선호될 확률 $P(y_w \succ y_l)$을 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$$P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l)) = \sigma(2.0 - 0.5) = \sigma(1.5)$$

$$= \frac{1}{1 + e^{-1.5}} = \frac{1}{1 + 0.2231} = \frac{1}{1.2231} \approx 0.8176$$

→ 약 81.8%의 확률로 $y_w$가 선호됩니다. 보상 차이가 클수록 확률이 1에 가까워집니다.
</details>

#### 문제 2: Reward Model Loss 계산

$r_\theta(y_w) = 1.0$, $r_\theta(y_l) = 0.8$일 때 손실값은?

<details>
<summary>💡 풀이 확인</summary>

$$\mathcal{L} = -\log\sigma(r_\theta(y_w) - r_\theta(y_l)) = -\log\sigma(1.0 - 0.8) = -\log\sigma(0.2)$$

$$\sigma(0.2) = \frac{1}{1+e^{-0.2}} = \frac{1}{1.8187} \approx 0.5498$$

$$\mathcal{L} = -\log(0.5498) \approx 0.5981$$

→ 보상 차이가 작아서 모델의 확신도가 낮고 손실이 비교적 큽니다. 학습을 통해 차이를 더 벌려야 합니다.
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

np.random.seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"NumPy 버전: {np.__version__}")"""))

# ── Cell 6: Section 2 header ─────────────────────────────────────
cells.append(md("""\
## 2. SFT → RM → PPO 파이프라인 시뮬레이션 <a name='2.-RLHF-파이프라인'></a>

InstructGPT의 3단계 RLHF 파이프라인을 간소화된 형태로 시뮬레이션합니다.
실제 LLM 대신 작은 MLP 네트워크를 사용하여 핵심 원리를 확인합니다.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Step 1: SFT │ ──→ │  Step 2: RM  │ ──→ │  Step 3: PPO│
│  지도학습     │     │  선호 학습    │     │  강화학습     │
│  (x, y*) 쌍  │     │  (x, y_w>y_l)│     │  r(x,y) 최대화│
└─────────────┘     └──────────────┘     └─────────────┘
```"""))

# ── Cell 7: RLHF pipeline simulation ────────────────────────────
cells.append(code("""\
# ── RLHF 3단계 파이프라인 시뮬레이션 ─────────────────────────────
# 간소화: 입력 x는 5차원 벡터, 출력 y는 3차원 벡터

input_dim = 5
output_dim = 3
hidden_dim = 16
n_samples = 200

# 시범 데이터 생성 (SFT용)
np.random.seed(42)
X_demo = np.random.randn(n_samples, input_dim).astype(np.float32)
# "좋은" 응답: 입력의 비선형 변환
true_transform = lambda x: np.tanh(x @ np.random.randn(input_dim, output_dim) * 0.5)
np.random.seed(42)
Y_demo = true_transform(X_demo).astype(np.float32)

# ============================================================
# Step 1: SFT (Supervised Fine-Tuning)
# ============================================================
print("=" * 55)
print("  Step 1: SFT (지도학습 미세조정)")
print("=" * 55)

sft_model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(hidden_dim, activation='relu'),
    tf.keras.layers.Dense(output_dim)
])
sft_model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')

sft_history = sft_model.fit(X_demo, Y_demo, epochs=50, batch_size=32,
                             verbose=0, validation_split=0.2)

sft_loss = sft_history.history['loss'][-1]
sft_val_loss = sft_history.history['val_loss'][-1]
print(f"  SFT 최종 손실: {sft_loss:.4f}")
print(f"  SFT 검증 손실: {sft_val_loss:.4f}")

# SFT 모델 가중치 저장 (기준 정책)
ref_weights = [w.numpy().copy() for w in sft_model.trainable_variables]

# 예측 예시
sample_x = X_demo[:3]
sample_pred = sft_model.predict(sample_x, verbose=0)
print(f"\\n  예측 예시 (입력 3개):")
for i in range(3):
    print(f"    y_pred[{i}] = [{', '.join(f'{v:.3f}' for v in sample_pred[i])}]")
    print(f"    y_true[{i}] = [{', '.join(f'{v:.3f}' for v in Y_demo[i])}]")"""))

# ── Cell 8: Bradley-Terry section ────────────────────────────────
cells.append(md(r"""\
## 3. Bradley-Terry 선호 모델 구현 <a name='3.-Bradley-Terry-모델'></a>

선호 쌍 데이터 $(x, y_w, y_l)$을 생성하고, Bradley-Terry 확률 모델을 적용합니다.

$$P(y_w \succ y_l \mid x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

보상 차이에 따른 선호 확률 분포를 시각화합니다."""))

# ── Cell 9: Bradley-Terry implementation ─────────────────────────
cells.append(code("""\
# ── Bradley-Terry 선호 모델 구현 ─────────────────────────────────
# 선호 쌍 데이터 생성
n_pairs = 500
np.random.seed(42)

X_pairs = np.random.randn(n_pairs, input_dim).astype(np.float32)

# 선호 응답: SFT 모델의 출력 + 약간의 개선
Y_preferred = sft_model.predict(X_pairs, verbose=0) + np.random.randn(n_pairs, output_dim).astype(np.float32) * 0.1
# 비선호 응답: 랜덤 노이즈가 큰 출력
Y_rejected = sft_model.predict(X_pairs, verbose=0) + np.random.randn(n_pairs, output_dim).astype(np.float32) * 0.8

# Bradley-Terry 확률 시각화
reward_diffs = np.linspace(-5, 5, 200)
bt_probs = 1.0 / (1.0 + np.exp(-reward_diffs))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) Bradley-Terry 확률 곡선
ax1 = axes[0]
ax1.plot(reward_diffs, bt_probs, 'b-', lw=2.5, label=r'$\sigma(\Delta r)$')
ax1.axhline(y=0.5, color='gray', ls='--', lw=1.5, alpha=0.5)
ax1.axvline(x=0, color='gray', ls='--', lw=1.5, alpha=0.5)
ax1.fill_between(reward_diffs, bt_probs, 0.5,
                 where=(reward_diffs > 0), alpha=0.15, color='green',
                 label=r'$r(y_w) > r(y_l)$')
ax1.fill_between(reward_diffs, bt_probs, 0.5,
                 where=(reward_diffs < 0), alpha=0.15, color='red',
                 label=r'$r(y_w) < r(y_l)$')
ax1.set_xlabel(r'보상 차이 $r(y_w) - r(y_l)$', fontsize=11)
ax1.set_ylabel(r'$P(y_w \succ y_l)$', fontsize=11)
ax1.set_title('Bradley-Terry 선호 확률', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) Reward Model Loss 곡선
ax2 = axes[1]
rm_loss_vals = -np.log(bt_probs + 1e-8)
ax2.plot(reward_diffs, rm_loss_vals, 'r-', lw=2.5, label=r'$-\log\sigma(\Delta r)$')
ax2.set_xlabel(r'보상 차이 $\Delta r$', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Reward Model 손실 함수', fontweight='bold')
ax2.set_ylim(0, 5)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/bradley_terry_model.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/bradley_terry_model.png")

# 수치 예시
print(f"\\nBradley-Terry 확률 예시:")
print(f"{'Δr':>6} | {'P(y_w ≻ y_l)':>14} | {'RM Loss':>10}")
print(f"{'-'*36}")
for dr in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0]:
    p = 1 / (1 + np.exp(-dr))
    loss = -np.log(p)
    print(f"{dr:>+6.1f} | {p:>14.4f} | {loss:>10.4f}")"""))

# ── Cell 10: Reward Model training section ───────────────────────
cells.append(md(r"""\
## 4. Reward Model 학습 <a name='4.-Reward-Model-학습'></a>

Reward Model은 입력 $(x, y)$를 받아 스칼라 점수 $r_\theta(x, y)$를 출력합니다.

손실 함수:

$$\mathcal{L}_{RM}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log \sigma\left(r_\theta(x_i, y_w^i) - r_\theta(x_i, y_l^i)\right)$$"""))

# ── Cell 11: Reward Model training ───────────────────────────────
cells.append(code("""\
# ── Reward Model 학습 구현 ───────────────────────────────────────
# Reward Model: (x, y)를 concat하여 스칼라 보상 출력
reward_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu',
                          input_shape=(input_dim + output_dim,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # 스칼라 보상
])

optimizer_rm = tf.keras.optimizers.Adam(learning_rate=0.001)

# 학습 데이터 준비
X_w = np.concatenate([X_pairs, Y_preferred], axis=1)
X_l = np.concatenate([X_pairs, Y_rejected], axis=1)

n_epochs = 50
batch_size = 64
rm_losses = []
rm_accuracies = []

for epoch in range(n_epochs):
    epoch_losses = []
    epoch_correct = 0

    indices = np.random.permutation(n_pairs)
    for start in range(0, n_pairs, batch_size):
        batch_idx = indices[start:start + batch_size]

        with tf.GradientTape() as tape:
            r_w = reward_model(X_w[batch_idx], training=True)
            r_l = reward_model(X_l[batch_idx], training=True)

            # Bradley-Terry loss
            diff = r_w - r_l
            loss = -tf.reduce_mean(tf.math.log_sigmoid(diff))

        grads = tape.gradient(loss, reward_model.trainable_variables)
        optimizer_rm.apply_gradients(zip(grads, reward_model.trainable_variables))

        epoch_losses.append(loss.numpy())
        epoch_correct += tf.reduce_sum(
            tf.cast(diff > 0, tf.float32)).numpy()

    avg_loss = np.mean(epoch_losses)
    accuracy = epoch_correct / n_pairs
    rm_losses.append(avg_loss)
    rm_accuracies.append(accuracy)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, "
              f"선호 정확도={accuracy:.4f}")

print(f"\\nReward Model 학습 완료")
print(f"  최종 손실: {rm_losses[-1]:.4f}")
print(f"  최종 선호 정확도: {rm_accuracies[-1]:.4f}")

# 보상 분포 확인
r_w_all = reward_model(X_w).numpy().flatten()
r_l_all = reward_model(X_l).numpy().flatten()
print(f"\\n보상 분포:")
print(f"  선호 응답 r(y_w): 평균={r_w_all.mean():.3f}, 표준편차={r_w_all.std():.3f}")
print(f"  비선호 응답 r(y_l): 평균={r_l_all.mean():.3f}, 표준편차={r_l_all.std():.3f}")
print(f"  평균 보상 차이: {(r_w_all - r_l_all).mean():.3f}")"""))

# ── Cell 12: RM training visualization ───────────────────────────
cells.append(code("""\
# ── Reward Model 학습 과정 시각화 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (1) 학습 손실
ax1 = axes[0]
ax1.plot(rm_losses, 'b-', lw=2, label='RM Loss')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Reward Model 학습 손실', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) 선호 정확도
ax2 = axes[1]
ax2.plot(rm_accuracies, 'g-', lw=2, label='선호 정확도')
ax2.axhline(y=0.5, color='red', ls='--', lw=1.5, label='랜덤 기준선')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('선호 쌍 분류 정확도', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.4, 1.05)

# (3) 보상 분포 히스토그램
ax3 = axes[2]
ax3.hist(r_w_all, bins=30, alpha=0.6, color='green', label=r'$r(y_w)$ 선호', density=True)
ax3.hist(r_l_all, bins=30, alpha=0.6, color='red', label=r'$r(y_l)$ 비선호', density=True)
ax3.axvline(x=r_w_all.mean(), color='darkgreen', ls='--', lw=2)
ax3.axvline(x=r_l_all.mean(), color='darkred', ls='--', lw=2)
ax3.set_xlabel('보상 점수', fontsize=11)
ax3.set_ylabel('밀도', fontsize=11)
ax3.set_title('학습된 보상 분포', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/reward_model_training.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/reward_model_training.png")
print(f"선호 응답(녹색)과 비선호 응답(빨간색)의 보상 분포가 잘 분리되었습니다.")"""))

# ── Cell 13: KL divergence section ───────────────────────────────
cells.append(md(r"""\
## 5. KL Divergence 페널티 시각화 <a name='5.-KL-Divergence-페널티'></a>

RLHF의 PPO 단계에서 KL 페널티는 학습 정책이 기준 정책(SFT)에서 너무 멀어지지 않도록 합니다:

$$\max_\theta \; \mathbb{E}_{y \sim \pi_\theta}\left[r_\phi(x, y)\right] - \beta \, D_{KL}\left[\pi_\theta \| \pi_{ref}\right]$$

$$D_{KL}\left[\pi_\theta \| \pi_{ref}\right] = \sum_y \pi_\theta(y|x) \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

$\beta$가 너무 작으면 → Reward Hacking (보상 모델의 허점을 악용)  
$\beta$가 너무 크면 → SFT 모델에서 거의 벗어나지 못함"""))

# ── Cell 14: KL divergence visualization ─────────────────────────
cells.append(code("""\
# ── KL Divergence 페널티 시각화 ──────────────────────────────────
# 다양한 β 값에 따른 RLHF 목적함수 분석

# 시뮬레이션: 정책이 기준에서 점진적으로 벗어나는 경우
n_vocab = 20
ref_logits = np.random.randn(n_vocab) * 0.5
ref_probs = np.exp(ref_logits) / np.exp(ref_logits).sum()

# 보상이 높은 토큰으로 정책이 이동하는 시나리오
high_reward_tokens = [3, 7, 12]
shift_amounts = np.linspace(0, 3, 50)

betas = [0.01, 0.05, 0.1, 0.3, 0.5]
results = {b: {'kl': [], 'reward': [], 'objective': []} for b in betas}

for shift in shift_amounts:
    # 정책 이동: 선호 토큰에 보너스
    shifted_logits = ref_logits.copy()
    for t in high_reward_tokens:
        shifted_logits[t] += shift
    shifted_probs = np.exp(shifted_logits) / np.exp(shifted_logits).sum()

    # KL divergence 계산
    kl = np.sum(shifted_probs * np.log(shifted_probs / (ref_probs + 1e-10) + 1e-10))

    # 보상 계산 (선호 토큰 확률 합)
    reward = sum(shifted_probs[t] for t in high_reward_tokens)

    for beta in betas:
        results[beta]['kl'].append(kl)
        results[beta]['reward'].append(reward)
        results[beta]['objective'].append(reward - beta * kl)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (1) 보상 vs KL 트레이드오프
ax1 = axes[0]
kls = results[0.01]['kl']
rewards = results[0.01]['reward']
ax1.plot(kls, rewards, 'b-o', lw=2, ms=3, label='보상 vs KL')
ax1.set_xlabel(r'$D_{KL}[\pi_\theta \| \pi_{ref}]$', fontsize=11)
ax1.set_ylabel('Expected Reward', fontsize=11)
ax1.set_title('보상-KL 트레이드오프', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) 다양한 β에 따른 목적함수
ax2 = axes[1]
colors_beta = ['#2196F3', '#FF9800', '#E91E63', '#9C27B0', '#4CAF50']
for beta, color in zip(betas, colors_beta):
    ax2.plot(shift_amounts, results[beta]['objective'],
             lw=2, color=color, label=f'β={beta}')
    # 최적점 표시
    opt_idx = np.argmax(results[beta]['objective'])
    ax2.plot(shift_amounts[opt_idx], results[beta]['objective'][opt_idx],
             '*', ms=15, color=color)
ax2.set_xlabel('정책 이동량', fontsize=11)
ax2.set_ylabel(r'$\mathbb{E}[r] - \beta D_{KL}$', fontsize=11)
ax2.set_title(r'RLHF 목적함수 ($\beta$ 변화)', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# (3) KL 발산 곡선
ax3 = axes[2]
ax3.plot(shift_amounts, results[0.01]['kl'], 'r-', lw=2.5,
         label=r'$D_{KL}[\pi_\theta \| \pi_{ref}]$')
ax3.fill_between(shift_amounts, 0, results[0.01]['kl'],
                 alpha=0.1, color='red')
ax3.set_xlabel('정책 이동량', fontsize=11)
ax3.set_ylabel(r'$D_{KL}$', fontsize=11)
ax3.set_title('KL Divergence 증가', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/kl_divergence_penalty.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/kl_divergence_penalty.png")

# 최적점 분석
print(f"\\nβ별 최적 정책 이동 지점:")
print(f"{'β':>6} | {'최적 이동량':>10} | {'보상':>8} | {'KL':>8} | {'목적함수':>10}")
print(f"{'-'*52}")
for beta in betas:
    opt_idx = np.argmax(results[beta]['objective'])
    print(f"{beta:>6.2f} | {shift_amounts[opt_idx]:>10.2f} | "
          f"{results[beta]['reward'][opt_idx]:>8.4f} | "
          f"{results[beta]['kl'][opt_idx]:>8.4f} | "
          f"{results[beta]['objective'][opt_idx]:>+10.4f}")
print(f"\\n  → β가 클수록 정책이 기준에 가까이 머무름 (보수적)")
print(f"  → β가 작을수록 보상 극대화에 집중 (공격적 → Reward Hacking 위험)")"""))

# ── Cell 15: Summary ─────────────────────────────────────────────
cells.append(md(r"""\
## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| InstructGPT 3단계 | SFT → Reward Model → PPO | ⭐⭐⭐ |
| Bradley-Terry | $P(y_w \succ y_l) = \sigma(\Delta r)$ — 쌍별 선호 확률 | ⭐⭐⭐ |
| RM Loss | $-\mathbb{E}[\log\sigma(r(y_w) - r(y_l))]$ | ⭐⭐⭐ |
| RLHF Objective | $\mathbb{E}[r(y)] - \beta D_{KL}[\pi_\theta \| \pi_{ref}]$ | ⭐⭐⭐ |
| KL 페널티 | $\beta D_{KL}$ — 기준 정책에서 이탈 제한 | ⭐⭐⭐ |
| Reward Hacking | β가 작을 때 보상 모델의 허점을 악용하는 현상 | ⭐⭐ |
| SFT | 시범 데이터로 지도학습 — RLHF의 첫 번째 단계 | ⭐⭐ |

### 핵심 수식

$$P(y_w \succ y_l \mid x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

$$\mathcal{L}_{RM}(\theta) = -\mathbb{E}\left[\log \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]$$

$$\max_\theta \; \mathbb{E}_{y \sim \pi_\theta}\left[r_\phi(x, y)\right] - \beta \, D_{KL}\left[\pi_\theta \| \pi_{ref}\right]$$

### 다음 챕터 예고
**04_dpo_and_preference_learning.ipynb** — Reward Model 없이도 선호 학습이 가능한 DPO(Direct Preference Optimization)의 베이즈 도출과 RLHF 대비 성능 비교를 다룹니다."""))

create_notebook(cells, 'chapter15_alignment_rlhf/03_rlhf_pipeline_overview.ipynb')
