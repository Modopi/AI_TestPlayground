"""Generate chapter15_alignment_rlhf/04_dpo_and_preference_learning.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 15: AI 얼라인먼트와 강화학습 — DPO와 선호 학습

## 학습 목표
- DPO(Direct Preference Optimization)의 수학적 도출 과정을 RLHF 목적함수로부터 유도한다
- DPO가 암묵적 보상 모델(implicit reward model)임을 수식으로 증명한다
- DPO 손실 함수를 구현하고 온도 파라미터 β의 효과를 분석한다
- DPO와 RLHF의 학습 곡선을 시뮬레이션으로 비교한다
- ORPO, KTO, SimPO 등 DPO 파생 기법의 핵심 차이를 이해한다

## 목차
1. [수학적 기초: DPO 도출과 암묵적 보상](#1.-수학적-기초)
2. [DPO 손실 함수 구현](#2.-DPO-손실-함수)
3. [β(온도) 효과 시각화](#3.-β-효과-시각화)
4. [DPO vs RLHF 학습 곡선 비교](#4.-DPO-vs-RLHF-비교)
5. [ORPO/KTO/SimPO 파생 기법](#5.-파생-기법)
6. [정리](#6.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### RLHF에서 DPO로의 유도

RLHF 목적함수의 최적 정책은 다음과 같이 닫힌 형태(closed-form)로 구해집니다:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r^*(y, x)\right)$$

- $Z(x) = \sum_y \pi_{ref}(y|x) \exp\!\left(\frac{1}{\beta}r^*(y,x)\right)$: 정규화 상수 (partition function)
- $\pi_{ref}$: 기준 정책 (SFT 모델)
- $\beta$: KL 페널티 계수 (온도 파라미터)

### 암묵적 보상 (Implicit Reward)

위 최적 정책을 보상에 대해 역으로 풀면:

$$r^*(y \mid x) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x)$$

- $r^*(y \mid x)$: 최적 정책이 내포하는 암묵적 보상
- 정책의 log-ratio가 곧 보상 함수 역할 → **별도의 Reward Model이 불필요**

### DPO 목적함수

Bradley-Terry 선호 모델에 암묵적 보상을 대입하면:

$$\mathcal{L}_{DPO}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)\right]$$

- $y_w$: 선호 응답 (chosen)
- $y_l$: 비선호 응답 (rejected)
- $\sigma$: 시그모이드 함수

### RLHF vs DPO 등가성

| 항목 | RLHF | DPO |
|------|------|-----|
| Reward Model | 명시적 학습 필요 ($r_\theta$) | **불필요** (정책이 암묵적 보상) |
| RL 알고리즘 | PPO (복잡한 강화학습) | **불필요** (지도학습과 동일) |
| 학습 목표 | $\mathbb{E}[r(y)] - \beta D_{KL}$ | $-\mathbb{E}[\log\sigma(\beta\Delta\log\pi)]$ |
| 안정성 | PPO 하이퍼파라미터 민감 | **안정적** (단순 cross-entropy 형태) |
| 계산 비용 | 4개 모델 (actor, critic, ref, RM) | **2개 모델** (actor, ref) |

**요약 표:**

| 구분 | 수식 | 설명 |
|------|------|------|
| 최적 정책 | $\pi^* \propto \pi_{ref}\exp(r/\beta)$ | KL-제약 하 최적해 |
| 암묵적 보상 | $r^* = \beta\log(\pi^*/\pi_{ref}) + \beta\log Z$ | 정책 = 보상 모델 |
| DPO Loss | $-\mathbb{E}[\log\sigma(\beta\Delta\log\pi)]$ | 선호 학습 직접 최적화 |

---"""))

# ── Cell 3: 🐣 friendly explanation ──────────────────────────────
cells.append(md("""\
### 🐣 초등학생을 위한 DPO 친절 설명!

#### 🔢 DPO가 뭔가요?

> 💡 **비유**: RLHF가 "시험 → 채점 → 공부 → 시험 → 채점 → ..."의 복잡한 반복이라면,
> DPO는 "정답과 오답을 보고 바로 공부하기"와 같아요!

**RLHF의 문제점:** 강아지 훈련에 비유하면, RLHF는
1. 먼저 "이것이 좋은 행동" 점수판(Reward Model)을 만들고
2. 그 점수판을 보면서 강아지를 따로 훈련(PPO)해야 해요
3. 점수판이 잘못되면 강아지가 엉뚱한 행동을 배울 수도 있어요!

**DPO의 해결:** DPO는 점수판을 따로 만들지 않아요!
- "이 행동이 저 행동보다 좋아"라는 비교 데이터만 있으면 돼요
- 강아지(모델)가 직접 "좋은 답변은 더 자주, 나쁜 답변은 덜 자주"로 배워요
- 수학적으로 RLHF와 **완전히 같은 결과**를 내지만, 훨씬 간단해요!

#### 🎯 β(베타)는 뭔가요?

> 💡 **비유**: β는 "얼마나 보수적으로 배울지" 결정하는 다이얼이에요!

- **β가 크면**: "원래 하던 대로 조금만 바꿔" → 안전하지만 느린 변화
- **β가 작으면**: "과감하게 바꿔!" → 빠르지만 위험할 수 있음

---"""))

# ── Cell 4: 📝 Exercises ──────────────────────────────────────────
cells.append(md(r"""\
### 📝 연습 문제

#### 문제 1: DPO 암묵적 보상 계산

정책의 log-ratio가 $\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} = 0.8$이고 
$\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} = -0.3$일 때, $\beta = 0.1$에서 DPO loss를 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$$\text{logit} = \beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) = 0.1 \times (0.8 - (-0.3)) = 0.1 \times 1.1 = 0.11$$

$$\mathcal{L}_{DPO} = -\log\sigma(0.11) = -\log\left(\frac{1}{1+e^{-0.11}}\right) = -\log(0.5275) \approx 0.6394$$

→ 선호 응답과 비선호 응답의 log-ratio 차이가 아직 작아서 손실이 비교적 큽니다.
</details>

#### 문제 2: β에 따른 최적 정책 변화

$\pi_{ref}(y|x) = 0.3$이고 $r^*(y|x) = 2.0$, $Z(x) = 1$일 때:
- $\beta = 0.5$에서 최적 정책 $\pi^*(y|x)$는?
- $\beta = 2.0$에서 최적 정책 $\pi^*(y|x)$는?

<details>
<summary>💡 풀이 확인</summary>

$$\pi^*(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)\exp\!\left(\frac{r^*(y|x)}{\beta}\right)$$

**β = 0.5:** $\pi^* = 0.3 \times \exp(2.0/0.5) = 0.3 \times e^4 = 0.3 \times 54.60 = 16.38$
(정규화 전 값이므로 Z로 나누면 1 미만이 됩니다)

**β = 2.0:** $\pi^* = 0.3 \times \exp(2.0/2.0) = 0.3 \times e^1 = 0.3 \times 2.718 = 0.816$

→ β가 작을수록 보상이 높은 응답에 **급격히** 확률을 집중시킵니다.
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

# ── Cell 6: DPO loss section header ──────────────────────────────
cells.append(md(r"""\
## 2. DPO 손실 함수 구현 <a name='2.-DPO-손실-함수'></a>

DPO 손실은 선호 응답과 비선호 응답의 log-ratio 차이에 시그모이드를 적용합니다:

$$\mathcal{L}_{DPO} = -\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)$$

실제 구현에서는 토큰 단위 log-probability를 시퀀스 길이로 평균하여 사용합니다."""))

# ── Cell 7: DPO loss implementation ──────────────────────────────
cells.append(code("""\
# ── DPO 손실 함수 구현 ──────────────────────────────────────────
# 실제 log-probability를 사용한 DPO loss 계산

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # log-ratio 계산
    chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratios = policy_rejected_logps - ref_rejected_logps

    # DPO logit
    logits = beta * (chosen_log_ratios - rejected_log_ratios)

    # DPO loss = -log(sigmoid(logits))
    losses = -tf.math.log_sigmoid(logits)

    # 암묵적 보상 추출
    chosen_rewards = beta * chosen_log_ratios
    rejected_rewards = beta * rejected_log_ratios

    return tf.reduce_mean(losses), chosen_rewards, rejected_rewards

# 시뮬레이션: 합성 log-probability 데이터 생성
np.random.seed(42)
n_samples = 500

# 기준 정책의 log-prob
ref_chosen_lp = np.random.normal(-2.0, 0.5, n_samples).astype(np.float32)
ref_rejected_lp = np.random.normal(-2.5, 0.5, n_samples).astype(np.float32)

# 학습 정책: 초기에는 ref와 유사, 학습 후 차이 발생
policy_chosen_lp = ref_chosen_lp + np.random.normal(0.3, 0.2, n_samples).astype(np.float32)
policy_rejected_lp = ref_rejected_lp + np.random.normal(-0.1, 0.2, n_samples).astype(np.float32)

# DPO loss 계산
loss_val, c_rewards, r_rewards = dpo_loss(
    tf.constant(policy_chosen_lp), tf.constant(policy_rejected_lp),
    tf.constant(ref_chosen_lp), tf.constant(ref_rejected_lp),
    beta=0.1
)

print("DPO 손실 함수 계산 결과")
print("=" * 50)
print(f"  DPO Loss: {loss_val.numpy():.4f}")
print(f"  선호 응답 암묵적 보상 평균: {tf.reduce_mean(c_rewards).numpy():.4f}")
print(f"  비선호 응답 암묵적 보상 평균: {tf.reduce_mean(r_rewards).numpy():.4f}")
print(f"  보상 마진 (chosen - rejected): {(tf.reduce_mean(c_rewards) - tf.reduce_mean(r_rewards)).numpy():.4f}")

# log-ratio 분포 확인
chosen_ratios = policy_chosen_lp - ref_chosen_lp
rejected_ratios = policy_rejected_lp - ref_rejected_lp
print(f"\\n  log-ratio 통계:")
print(f"  chosen log(pi/pi_ref)  평균={chosen_ratios.mean():.3f}, std={chosen_ratios.std():.3f}")
print(f"  rejected log(pi/pi_ref) 평균={rejected_ratios.mean():.3f}, std={rejected_ratios.std():.3f}")"""))

# ── Cell 8: Beta effect section ──────────────────────────────────
cells.append(md(r"""\
## 3. β(온도) 효과 시각화 <a name='3.-β-효과-시각화'></a>

$\beta$는 DPO에서 핵심 하이퍼파라미터로, 기준 정책으로부터의 이탈 정도를 제어합니다:

- **큰 β**: 기준 정책에 가까운 보수적 학습 → 안정적이나 느린 변화
- **작은 β**: 선호 데이터에 강하게 적합 → 빠르지만 과적합 위험"""))

# ── Cell 9: Beta effect visualization ────────────────────────────
cells.append(code("""\
# ── β(온도) 효과 시각화 ─────────────────────────────────────────
# 다양한 β에서 DPO loss의 gradient 강도 비교

log_ratio_diffs = np.linspace(-3, 3, 300)
betas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (1) DPO loss vs log-ratio 차이
ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(betas)))
for beta, color in zip(betas, colors):
    logits = beta * log_ratio_diffs
    loss_curve = -np.log(1 / (1 + np.exp(-logits)) + 1e-10)
    ax1.plot(log_ratio_diffs, loss_curve, lw=2, color=color,
             label=f'beta={beta}')
ax1.set_xlabel(r'$\Delta \log(\pi/\pi_{ref})$', fontsize=11)
ax1.set_ylabel('DPO Loss', fontsize=11)
ax1.set_title(r'$\beta$에 따른 DPO 손실', fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 4)

# (2) 그래디언트 강도 비교
ax2 = axes[1]
for beta, color in zip(betas, colors):
    logits = beta * log_ratio_diffs
    sigmoid_val = 1 / (1 + np.exp(-logits))
    gradient = -beta * (1 - sigmoid_val)
    ax2.plot(log_ratio_diffs, np.abs(gradient), lw=2, color=color,
             label=f'beta={beta}')
ax2.set_xlabel(r'$\Delta \log(\pi/\pi_{ref})$', fontsize=11)
ax2.set_ylabel(r'$|\nabla_\theta \mathcal{L}|$', fontsize=11)
ax2.set_title(r'$\beta$에 따른 그래디언트 크기', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# (3) β 범위별 accuracy 시뮬레이션
ax3 = axes[2]
beta_range = np.linspace(0.01, 1.0, 50)
accuracies = []
losses_by_beta = []
for b in beta_range:
    logits_sim = b * (chosen_ratios - rejected_ratios)
    acc = np.mean(logits_sim > 0)
    loss_sim = np.mean(-np.log(1 / (1 + np.exp(-logits_sim)) + 1e-10))
    accuracies.append(acc)
    losses_by_beta.append(loss_sim)

ax3.plot(beta_range, accuracies, 'b-', lw=2.5, label='선호 정확도')
ax3_twin = ax3.twinx()
ax3_twin.plot(beta_range, losses_by_beta, 'r--', lw=2, label='DPO Loss')
ax3.set_xlabel(r'$\beta$', fontsize=11)
ax3.set_ylabel('Accuracy', fontsize=11, color='blue')
ax3_twin.set_ylabel('Loss', fontsize=11, color='red')
ax3.set_title(r'$\beta$ 범위별 정확도와 손실', fontweight='bold')
ax3.legend(loc='center left', fontsize=9)
ax3_twin.legend(loc='center right', fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/dpo_beta_effect.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/dpo_beta_effect.png")

# 수치 비교
print(f"\\nbeta별 DPO 성능 비교:")
print(f"{'beta':>8} | {'Loss':>10} | {'Accuracy':>10}")
print("-" * 35)
for b in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
    logits_sim = b * (chosen_ratios - rejected_ratios)
    acc = np.mean(logits_sim > 0)
    loss_sim = np.mean(-np.log(1 / (1 + np.exp(-logits_sim)) + 1e-10))
    print(f"{b:>8.2f} | {loss_sim:>10.4f} | {acc:>10.4f}")"""))

# ── Cell 10: DPO vs RLHF section ─────────────────────────────────
cells.append(md(r"""\
## 4. DPO vs RLHF 학습 곡선 비교 <a name='4.-DPO-vs-RLHF-비교'></a>

DPO와 RLHF의 학습 과정을 시뮬레이션하여 수렴 속도와 안정성을 비교합니다.

**시뮬레이션 설정:**
- 정책 네트워크: 간단한 MLP (입력 → 출력 확률 분포)
- DPO: 선호 쌍 데이터로 직접 정책 학습
- RLHF: Reward Model + PPO 근사"""))

# ── Cell 11: DPO vs RLHF training comparison ─────────────────────
cells.append(code("""\
# ── DPO vs RLHF 학습 곡선 비교 시뮬레이션 ────────────────────────
np.random.seed(42)
n_steps = 200

# 합성 선호 데이터
input_dim = 8
n_data = 300
X = np.random.randn(n_data, input_dim).astype(np.float32)
true_reward = X @ np.random.randn(input_dim, 1).astype(np.float32) * 0.5
Y_chosen_logp = -1.5 + 0.3 * true_reward.flatten() + np.random.randn(n_data).astype(np.float32) * 0.2
Y_rejected_logp = -2.5 - 0.2 * true_reward.flatten() + np.random.randn(n_data).astype(np.float32) * 0.3

# DPO 학습
dpo_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2)
])
dpo_opt = tf.keras.optimizers.Adam(0.005)
ref_logps_chosen = Y_chosen_logp.copy()
ref_logps_rejected = Y_rejected_logp.copy()

dpo_losses = []
dpo_accs = []

for step in range(n_steps):
    idx = np.random.choice(n_data, 64, replace=False)
    with tf.GradientTape() as tape:
        out = dpo_model(X[idx], training=True)
        pi_chosen = out[:, 0]
        pi_rejected = out[:, 1]
        logits = 0.1 * ((pi_chosen - ref_logps_chosen[idx]) -
                         (pi_rejected - ref_logps_rejected[idx]))
        loss = -tf.reduce_mean(tf.math.log_sigmoid(logits))
    grads = tape.gradient(loss, dpo_model.trainable_variables)
    dpo_opt.apply_gradients(zip(grads, dpo_model.trainable_variables))
    acc = tf.reduce_mean(tf.cast(logits > 0, tf.float32)).numpy()
    dpo_losses.append(loss.numpy())
    dpo_accs.append(acc)

# RLHF 시뮬레이션 (Reward Model + PPO 근사)
rm_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
rm_opt = tf.keras.optimizers.Adam(0.005)

# RM 사전학습
for _ in range(80):
    idx = np.random.choice(n_data, 64, replace=False)
    with tf.GradientTape() as tape:
        r_w = rm_model(X[idx], training=True)[:, 0]
        r_l = rm_model(X[idx] + np.random.randn(64, input_dim).astype(np.float32) * 0.5,
                        training=True)[:, 0]
        rm_loss = -tf.reduce_mean(tf.math.log_sigmoid(r_w - r_l))
    grads = tape.gradient(rm_loss, rm_model.trainable_variables)
    rm_opt.apply_gradients(zip(grads, rm_model.trainable_variables))

rlhf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2)
])
rlhf_opt = tf.keras.optimizers.Adam(0.003)

rlhf_losses = []
rlhf_accs = []

for step in range(n_steps):
    idx = np.random.choice(n_data, 64, replace=False)
    with tf.GradientTape() as tape:
        out = rlhf_model(X[idx], training=True)
        reward_estimate = rm_model(X[idx], training=False)[:, 0]
        # PPO 근사: reward 기반 정책 경사
        chosen_logp = out[:, 0]
        kl_penalty = 0.1 * tf.reduce_mean(tf.square(chosen_logp - ref_logps_chosen[idx]))
        rlhf_loss = -tf.reduce_mean(reward_estimate * chosen_logp) + kl_penalty
        # PPO에 노이즈 추가 (현실적 불안정성 반영)
        noise = tf.random.normal([], stddev=0.05)
        rlhf_loss = rlhf_loss + noise

    grads = tape.gradient(rlhf_loss, rlhf_model.trainable_variables)
    rlhf_opt.apply_gradients(zip(grads, rlhf_model.trainable_variables))

    out_eval = rlhf_model(X[idx], training=False)
    logits_eval = 0.1 * ((out_eval[:, 0] - ref_logps_chosen[idx]) -
                          (out_eval[:, 1] - ref_logps_rejected[idx]))
    acc = tf.reduce_mean(tf.cast(logits_eval > 0, tf.float32)).numpy()
    rlhf_losses.append(rlhf_loss.numpy())
    rlhf_accs.append(acc)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) 학습 곡선 비교
ax1 = axes[0]
window = 10
dpo_smooth = np.convolve(dpo_losses, np.ones(window)/window, mode='valid')
rlhf_smooth = np.convolve(rlhf_losses, np.ones(window)/window, mode='valid')
ax1.plot(dpo_smooth, 'b-', lw=2.5, label='DPO Loss')
ax1.plot(rlhf_smooth, 'r-', lw=2.5, alpha=0.7, label='RLHF Loss')
ax1.set_xlabel('학습 스텝', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('DPO vs RLHF 학습 곡선', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) 정확도 비교
ax2 = axes[1]
dpo_acc_smooth = np.convolve(dpo_accs, np.ones(window)/window, mode='valid')
rlhf_acc_smooth = np.convolve(rlhf_accs, np.ones(window)/window, mode='valid')
ax2.plot(dpo_acc_smooth, 'b-', lw=2.5, label='DPO Accuracy')
ax2.plot(rlhf_acc_smooth, 'r-', lw=2.5, alpha=0.7, label='RLHF Accuracy')
ax2.axhline(y=0.5, color='gray', ls='--', lw=1.5, label='랜덤 기준선')
ax2.set_xlabel('학습 스텝', fontsize=11)
ax2.set_ylabel('선호 정확도', fontsize=11)
ax2.set_title('DPO vs RLHF 선호 정확도', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.3, 1.05)

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/dpo_vs_rlhf_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/dpo_vs_rlhf_comparison.png")

print(f"\\n최종 비교 결과:")
print(f"{'메트릭':<20} | {'DPO':>12} | {'RLHF':>12}")
print("-" * 50)
print(f"{'최종 Loss':<20} | {np.mean(dpo_losses[-20:]):>12.4f} | {np.mean(rlhf_losses[-20:]):>12.4f}")
print(f"{'최종 Accuracy':<20} | {np.mean(dpo_accs[-20:]):>12.4f} | {np.mean(rlhf_accs[-20:]):>12.4f}")
print(f"{'Loss 분산 (안정성)':<20} | {np.var(dpo_losses[-50:]):>12.6f} | {np.var(rlhf_losses[-50:]):>12.6f}")
print(f"\\n  → DPO: 안정적 수렴, 단순한 구조")
print(f"  → RLHF: 분산이 크고, RM + PPO의 복잡한 파이프라인")"""))

# ── Cell 12: Derived methods section ─────────────────────────────
cells.append(md(r"""\
## 5. ORPO/KTO/SimPO 파생 기법 <a name='5.-파생-기법'></a>

DPO의 성공 이후 다양한 파생 기법이 등장했습니다:

| 기법 | 핵심 아이디어 | 수식 특징 |
|------|-------------|----------|
| **ORPO** | SFT와 DPO를 한 번에 | $\mathcal{L} = \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR}$ |
| **KTO** | 쌍(pair) 데이터 불필요 | 개별 응답에 "좋다/나쁘다" 라벨만 필요 |
| **SimPO** | Reference-free DPO | $\pi_{ref}$ 없이 길이 정규화된 log-prob 사용 |
| **IPO** | σ-free 최적화 | 시그모이드 대신 제곱 손실 사용 |

### ORPO (Odds Ratio Preference Optimization)

$$\mathcal{L}_{ORPO} = \mathcal{L}_{NLL} + \lambda \cdot \mathcal{L}_{OR}$$

$$\mathcal{L}_{OR} = -\log\sigma\!\left(\log\frac{P(y_w|x)}{1-P(y_w|x)} - \log\frac{P(y_l|x)}{1-P(y_l|x)}\right)$$

### KTO (Kahneman-Tversky Optimization)

$$\mathcal{L}_{KTO} = \begin{cases} -\lambda_w \sigma(\beta r_w - z_{ref}) & \text{if } y \text{ is desirable} \\ -\lambda_l \sigma(z_{ref} - \beta r_l) & \text{if } y \text{ is undesirable}\end{cases}$$

### SimPO (Simple Preference Optimization)

$$\mathcal{L}_{SimPO} = -\log\sigma\!\left(\frac{\beta}{|y_w|}\log P(y_w|x) - \frac{\beta}{|y_l|}\log P(y_l|x) - \gamma\right)$$"""))

# ── Cell 13: ORPO/KTO/SimPO comparison code ──────────────────────
cells.append(code("""\
# ── ORPO/KTO/SimPO 비교 시뮬레이션 ──────────────────────────────
np.random.seed(42)
n_sim = 1000

# 합성 선호 데이터: log-probability
chosen_lp = np.random.normal(-1.5, 0.5, n_sim)
rejected_lp = np.random.normal(-2.5, 0.6, n_sim)
ref_chosen_lp_sim = np.random.normal(-1.8, 0.5, n_sim)
ref_rejected_lp_sim = np.random.normal(-2.3, 0.5, n_sim)
chosen_lengths = np.random.randint(10, 50, n_sim)
rejected_lengths = np.random.randint(10, 50, n_sim)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# DPO Loss
beta_sim = 0.1
dpo_logits = beta_sim * ((chosen_lp - ref_chosen_lp_sim) - (rejected_lp - ref_rejected_lp_sim))
dpo_loss_vals = -np.log(sigmoid(dpo_logits) + 1e-10)

# ORPO Loss (Odds Ratio)
chosen_prob = np.exp(chosen_lp)
rejected_prob = np.exp(rejected_lp)
odds_w = np.log(chosen_prob / (1 - chosen_prob + 1e-10) + 1e-10)
odds_l = np.log(rejected_prob / (1 - rejected_prob + 1e-10) + 1e-10)
orpo_logits = odds_w - odds_l
orpo_loss_vals = -np.log(sigmoid(orpo_logits) + 1e-10)

# KTO Loss (개별 응답)
z_ref = beta_sim * (ref_chosen_lp_sim - ref_rejected_lp_sim).mean()
kto_chosen_loss = 1 - sigmoid(beta_sim * (chosen_lp - ref_chosen_lp_sim) - z_ref)
kto_rejected_loss = 1 - sigmoid(z_ref - beta_sim * (rejected_lp - ref_rejected_lp_sim))
kto_loss_vals = 0.5 * kto_chosen_loss + 0.5 * kto_rejected_loss

# SimPO Loss (Reference-free, length-normalized)
gamma_sim = 0.5
simpo_logits = beta_sim * (chosen_lp / chosen_lengths - rejected_lp / rejected_lengths) - gamma_sim
simpo_loss_vals = -np.log(sigmoid(simpo_logits) + 1e-10)

# 비교 표
methods = ['DPO', 'ORPO', 'KTO', 'SimPO']
mean_losses = [dpo_loss_vals.mean(), orpo_loss_vals.mean(),
               kto_loss_vals.mean(), simpo_loss_vals.mean()]
std_losses = [dpo_loss_vals.std(), orpo_loss_vals.std(),
              kto_loss_vals.std(), simpo_loss_vals.std()]
accs = [np.mean(dpo_logits > 0), np.mean(orpo_logits > 0),
        np.mean(kto_chosen_loss < 0.5), np.mean(simpo_logits > 0)]

print("ORPO/KTO/SimPO 파생 기법 비교")
print("=" * 65)
print(f"{'기법':<8} | {'평균 Loss':>12} | {'Loss 표준편차':>14} | {'정확도':>8} | {'Ref 필요':>10}")
print("-" * 65)
for m, ml, sl, a in zip(methods, mean_losses, std_losses, accs):
    needs_ref = "필요" if m in ['DPO', 'KTO'] else "불필요"
    print(f"{m:<8} | {ml:>12.4f} | {sl:>14.4f} | {a:>8.4f} | {needs_ref:>10}")

print(f"\\n핵심 차이점:")
print(f"  ORPO: SFT + 선호학습 동시 → 학습 단계 절약")
print(f"  KTO:  쌍(pair) 데이터 없이 개별 '좋다/나쁘다' 라벨로 학습")
print(f"  SimPO: Reference model 불필요 → 메모리 절약, 길이 정규화")"""))

# ── Cell 14: Derived methods visualization ────────────────────────
cells.append(code("""\
# ── 파생 기법 손실 분포 시각화 ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) 손실 분포 비교
ax1 = axes[0]
all_losses = [dpo_loss_vals, orpo_loss_vals, kto_loss_vals, simpo_loss_vals]
colors_method = ['#2196F3', '#FF9800', '#E91E63', '#4CAF50']
positions = [1, 2, 3, 4]
bp = ax1.boxplot(all_losses, positions=positions, widths=0.6, patch_artist=True)
for patch, color in zip(bp['boxes'], colors_method):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax1.set_xticklabels(methods, fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('선호 학습 기법별 손실 분포', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# (2) 기법별 특성 레이더 차트 (바 차트로 표현)
ax2 = axes[1]
categories = ['학습 안정성', '메모리 효율', '데이터 효율', '구현 단순성']
# 5점 만점 상대 점수 (논문 기반 정성적 평가)
scores = {
    'DPO':   [4.5, 3.0, 3.5, 4.5],
    'ORPO':  [4.0, 4.0, 3.0, 3.5],
    'KTO':   [3.5, 3.5, 4.5, 3.0],
    'SimPO': [4.0, 4.5, 3.5, 4.0],
}

x_pos = np.arange(len(categories))
width = 0.18
for i, (method, sc) in enumerate(scores.items()):
    ax2.bar(x_pos + i * width, sc, width, label=method,
            color=colors_method[i], alpha=0.8)
ax2.set_xticks(x_pos + width * 1.5)
ax2.set_xticklabels(categories, fontsize=10)
ax2.set_ylabel('점수 (5점 만점)', fontsize=11)
ax2.set_title('기법별 특성 비교', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 5.5)

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/dpo_variants_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/dpo_variants_comparison.png")

# 정리 표
print(f"\\n기법별 요구사항 비교:")
print(f"{'기법':<8} | {'Reference Model':>16} | {'Pair 데이터':>12} | {'RM 필요':>10} | {'학습 단계':>10}")
print("-" * 65)
print(f"{'RLHF':<8} | {'필요':>16} | {'필요':>12} | {'필요':>10} | {'3단계':>10}")
print(f"{'DPO':<8} | {'필요':>16} | {'필요':>12} | {'불필요':>10} | {'1단계':>10}")
print(f"{'ORPO':<8} | {'불필요':>16} | {'필요':>12} | {'불필요':>10} | {'1단계':>10}")
print(f"{'KTO':<8} | {'필요':>16} | {'불필요':>12} | {'불필요':>10} | {'1단계':>10}")
print(f"{'SimPO':<8} | {'불필요':>16} | {'필요':>12} | {'불필요':>10} | {'1단계':>10}")"""))

# ── Cell 15: Summary ─────────────────────────────────────────────
cells.append(md(r"""\
## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| DPO 도출 | RLHF 최적해의 닫힌 형태 → BT 모델 대입 | ⭐⭐⭐ |
| 암묵적 보상 | $r^* = \beta\log(\pi^*/\pi_{ref}) + \beta\log Z$ — 정책이 곧 보상 | ⭐⭐⭐ |
| DPO Loss | $-\mathbb{E}[\log\sigma(\beta\Delta\log\pi)]$ — 별도 RM/PPO 불필요 | ⭐⭐⭐ |
| β 파라미터 | 큰 β = 보수적, 작은 β = 공격적 학습 | ⭐⭐⭐ |
| RLHF 등가성 | DPO = RLHF의 수학적으로 동등한 단순화 | ⭐⭐⭐ |
| ORPO | SFT + 선호학습 동시, Odds Ratio 사용 | ⭐⭐ |
| KTO | 쌍 데이터 불필요, 개별 선호/비선호 라벨 | ⭐⭐ |
| SimPO | Reference-free, 길이 정규화 | ⭐⭐ |

### 핵심 수식

$$\mathcal{L}_{DPO}(\pi_\theta) = -\mathbb{E}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

$$r^*(y|x) = \beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)$$

$$\pi^*(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)\exp\!\left(\frac{r^*(y|x)}{\beta}\right)$$

### 다음 챕터 예고
**05_constitutional_ai_and_rlaif.ipynb** — Anthropic의 Constitutional AI 원칙, AI-피드백(RLAIF) 자동화 파이프라인, Red Teaming과 Jailbreak 방어 기법을 다룹니다."""))

create_notebook(cells, 'chapter15_alignment_rlhf/04_dpo_and_preference_learning.ipynb')
