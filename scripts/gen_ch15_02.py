"""Generate chapter15_alignment_rlhf/02_actor_critic_and_ppo.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 15: AI 얼라인먼트와 강화학습 — Actor-Critic과 PPO

## 학습 목표
- Advantage Function의 수학적 의미와 분산 감소 효과를 이해한다
- Actor-Critic 구조에서 Critic이 Baseline 역할을 하는 원리를 이해한다
- PPO-Clip 목적함수의 3구간 동작을 수식으로 완전 전개하고 시각화한다
- KL 페널티 방식과 Clip 방식의 장단점을 비교하고 구현한다
- A2C에서 PPO로의 발전 과정과 LLM 정렬에서의 역할을 설명한다

## 목차
1. [수학적 기초: Advantage Function과 PPO](#1.-수학적-기초)
2. [Advantage Function 계산 데모](#2.-Advantage-Function-계산)
3. [PPO-Clip 목적함수 3구간 시각화](#3.-PPO-Clip-3구간-시각화)
4. [A2C vs PPO 비교 구현](#4.-A2C-vs-PPO-비교)
5. [KL 페널티 vs Clip 비교](#5.-KL-페널티-vs-Clip)
6. [정리](#6.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### Advantage Function

REINFORCE의 높은 분산 문제를 해결하기 위해 **Baseline** $b(s)$를 빼줍니다:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[(G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

최적의 Baseline은 상태 가치 함수 $V^\pi(s)$이며, 이때 $(G_t - V^\pi(s_t))$가 **Advantage**입니다:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

- $A^\pi(s, a) > 0$: 해당 행동이 **평균보다 나음** → 확률 증가
- $A^\pi(s, a) < 0$: 해당 행동이 **평균보다 못함** → 확률 감소
- $A^\pi(s, a) = 0$: 평균 수준

### GAE (Generalized Advantage Estimation)

TD 오차를 가중 합산하여 편향-분산 트레이드오프를 제어합니다:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t^{GAE} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

- $\lambda = 0$: 1-step TD (낮은 분산, 높은 편향)
- $\lambda = 1$: Monte Carlo (높은 분산, 낮은 편향)

### PPO-Clip 목적함수

확률 비율:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**PPO-Clip 목적함수:**

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

- $\epsilon$: 클리핑 범위 (보통 0.1 ~ 0.2)

**3구간 분석:**

| 구간 | 조건 | $\hat{A}_t > 0$ (좋은 행동) | $\hat{A}_t < 0$ (나쁜 행동) |
|------|------|-----|------|
| 좌측 | $r_t < 1 - \epsilon$ | 클리핑됨 (확률 감소 제한) | 원래 값 사용 |
| 중앙 | $1-\epsilon \leq r_t \leq 1+\epsilon$ | 원래 값 사용 | 원래 값 사용 |
| 우측 | $r_t > 1 + \epsilon$ | 원래 값 사용 | 클리핑됨 (확률 증가 제한) |

### KL 페널티 방식 (PPO-Penalty)

$$L^{KL}(\theta) = \mathbb{E}\left[r_t(\theta)\hat{A}_t - \beta D_{KL}[\pi_{\theta_{old}} \| \pi_\theta]\right]$$

- $\beta$: KL 페널티 계수 (적응적으로 조절 가능)

**요약 표:**

| 구분 | 수식 | 설명 |
|------|------|------|
| Advantage | $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ | 평균 대비 행동의 상대적 가치 |
| PPO-Clip | $\min(r_t \hat{A}_t, \text{clip}(r_t) \hat{A}_t)$ | 정책 업데이트 범위 제한 |
| 확률 비율 | $r_t = \pi_\theta / \pi_{\theta_{old}}$ | 새 정책 / 이전 정책 확률 비 |
| GAE | $\hat{A}_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$ | 편향-분산 균형 Advantage |

---"""))

# ── Cell 3: 🐣 friendly explanation ──────────────────────────────
cells.append(md("""\
### 🐣 초등학생을 위한 PPO 친절 설명!

#### 🔢 Advantage가 뭔가요?

> 💡 **비유**: 시험 점수를 생각해 보세요!

- 반 평균이 80점($V(s)$)인데 내가 95점($Q(s,a)$)을 받으면 → **Advantage = +15** (잘했다!)
- 반 평균이 80점인데 내가 60점을 받으면 → **Advantage = -20** (좀 더 노력해야!)
- 즉, Advantage는 "평균 대비 얼마나 잘했나?"를 알려주는 거예요.

#### 🔒 PPO-Clip이 뭔가요?

> 💡 **비유**: 자전거 보조 바퀴를 생각해 보세요!

PPO-Clip은 AI가 한 번에 너무 많이 변하지 않도록 **보조 바퀴**를 달아주는 거예요:
- 정책이 크게 바뀌면($r_t$가 1에서 멀어지면) → **"잠깐, 너무 급하게 바꾸지 마!"** 라고 제한
- 정책이 조금만 바뀌면($r_t \\approx 1$) → **"좋아, 그 정도면 괜찮아!"** 라고 허용
- 이렇게 하면 학습이 안정적으로 진행돼요!

---"""))

# ── Cell 4: 📝 Exercises ──────────────────────────────────────────
cells.append(md(r"""\
### 📝 연습 문제

#### 문제 1: Advantage 부호 해석

어떤 상태에서 $Q^\pi(s, \text{left}) = 8$, $Q^\pi(s, \text{right}) = 3$, $V^\pi(s) = 5$일 때:
- $A^\pi(s, \text{left})$과 $A^\pi(s, \text{right})$를 각각 구하세요.
- 각 행동의 확률은 어떻게 변해야 하나요?

<details>
<summary>💡 풀이 확인</summary>

$$A^\pi(s, \text{left}) = Q^\pi(s, \text{left}) - V^\pi(s) = 8 - 5 = +3$$

$$A^\pi(s, \text{right}) = Q^\pi(s, \text{right}) - V^\pi(s) = 3 - 5 = -2$$

- left 행동: $A > 0$ → 확률을 **증가**시켜야 합니다 (평균보다 좋은 행동)
- right 행동: $A < 0$ → 확률을 **감소**시켜야 합니다 (평균보다 나쁜 행동)
</details>

#### 문제 2: PPO-Clip 목적함수 값 계산

$r_t = 1.3$, $\hat{A}_t = 2.0$, $\epsilon = 0.2$일 때 $L^{CLIP}$의 값은?

<details>
<summary>💡 풀이 확인</summary>

$$\text{원래}: r_t \hat{A}_t = 1.3 \times 2.0 = 2.6$$

$$\text{클리핑}: \text{clip}(1.3, 0.8, 1.2) \times 2.0 = 1.2 \times 2.0 = 2.4$$

$$L^{CLIP} = \min(2.6, 2.4) = 2.4$$

→ $r_t > 1+\epsilon$이고 $\hat{A}_t > 0$이므로 클리핑되지 않습니다 (좋은 행동을 더 하려는 것은 막지 않음). 
실제로 $\min$을 취하면 2.4가 되어, 확률 증가폭이 자연스럽게 제한됩니다.
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
cells.append(md(r"""\
## 2. Advantage Function 계산 데모 <a name='2.-Advantage-Function-계산'></a>

간단한 에피소드를 생성하고 **TD 잔차, GAE Advantage**를 단계별로 계산합니다.

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \quad \hat{A}_t^{GAE} = \sum_{l=0}^{T-t}(\gamma\lambda)^l \delta_{t+l}$$"""))

# ── Cell 7: Advantage calculation ────────────────────────────────
cells.append(code("""\
# ── Advantage Function 계산 데모 ─────────────────────────────────
# 간단한 에피소드 (5 timestep)
rewards = np.array([1.0, 0.5, 2.0, -1.0, 3.0])
values  = np.array([2.0, 1.5, 3.0, 0.5, 2.5])  # V(s_t) 추정값
gamma = 0.99
lam = 0.95
T = len(rewards)
next_value = 0.0  # 에피소드 종료

print(f"에피소드 데이터:")
print(f"{'t':>4} | {'r_t':>6} | {'V(s_t)':>8} | {'V(s_t+1)':>10}")
print(f"{'-'*38}")
for t in range(T):
    v_next = values[t+1] if t < T-1 else next_value
    print(f"{t:>4} | {rewards[t]:>6.1f} | {values[t]:>8.2f} | {v_next:>10.2f}")

# TD 잔차 계산
deltas = np.zeros(T)
for t in range(T):
    v_next = values[t+1] if t < T-1 else next_value
    deltas[t] = rewards[t] + gamma * v_next - values[t]

print(f"\\nTD 잔차 δ_t:")
for t in range(T):
    print(f"  δ_{t} = r_{t} + γ·V(s_{t+1}) - V(s_{t}) = "
          f"{rewards[t]:.1f} + {gamma}×{values[t+1] if t<T-1 else next_value:.2f} - {values[t]:.2f} = {deltas[t]:+.4f}")

# GAE 계산 (역순 누적)
gae_advantages = np.zeros(T)
gae = 0
for t in reversed(range(T)):
    gae = deltas[t] + gamma * lam * gae
    gae_advantages[t] = gae

# Monte Carlo Advantage (비교용)
mc_returns = np.zeros(T)
G = next_value
for t in reversed(range(T)):
    G = rewards[t] + gamma * G
    mc_returns[t] = G
mc_advantages = mc_returns - values

print(f"\\n{'t':>4} | {'δ_t (TD)':>10} | {'GAE(λ={lam})':>12} | {'MC Advantage':>14}")
print(f"{'-'*50}")
for t in range(T):
    print(f"{t:>4} | {deltas[t]:>+10.4f} | {gae_advantages[t]:>+12.4f} | {mc_advantages[t]:>+14.4f}")

print(f"\\n분산 비교:")
print(f"  TD(λ=0) 분산:  {np.var(deltas):.4f}")
print(f"  GAE(λ={lam}) 분산: {np.var(gae_advantages):.4f}")
print(f"  MC(λ=1) 분산:  {np.var(mc_advantages):.4f}")"""))

# ── Cell 8: Section 3 header ─────────────────────────────────────
cells.append(md(r"""\
## 3. PPO-Clip 목적함수 3구간 시각화 <a name='3.-PPO-Clip-3구간-시각화'></a>

PPO-Clip의 핵심은 확률 비율 $r_t(\theta)$에 따라 목적함수가 **3구간으로 분리**되는 것입니다:

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

- **구간 1** ($r_t < 1-\epsilon$): $\hat{A}_t > 0$일 때 클리핑 → 좋은 행동의 확률 **감소 제한**
- **구간 2** ($1-\epsilon \leq r_t \leq 1+\epsilon$): 클리핑 없음 → 자유로운 업데이트
- **구간 3** ($r_t > 1+\epsilon$): $\hat{A}_t < 0$일 때 클리핑 → 나쁜 행동의 확률 **증가 제한**"""))

# ── Cell 9: PPO-Clip 3-region visualization (CRITICAL) ───────────
cells.append(code("""\
# ── PPO-Clip 3구간 시각화 (핵심!) ────────────────────────────────
epsilon = 0.2
r = np.linspace(0.3, 2.0, 500)

def ppo_clip_objective(r, A, eps):
    # 원래 목적
    surr1 = r * A
    # 클리핑된 목적
    r_clipped = np.clip(r, 1 - eps, 1 + eps)
    surr2 = r_clipped * A
    return np.minimum(surr1, surr2)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (1) A > 0 (좋은 행동)
ax1 = axes[0]
A_pos = 1.0
surr1_pos = r * A_pos
r_clip = np.clip(r, 1 - epsilon, 1 + epsilon)
surr2_pos = r_clip * A_pos
L_pos = np.minimum(surr1_pos, surr2_pos)

ax1.plot(r, surr1_pos, 'b--', lw=1.5, alpha=0.5, label=r'$r_t \hat{A}_t$ (원래)')
ax1.plot(r, surr2_pos, 'r--', lw=1.5, alpha=0.5, label=r'$\mathrm{clip}(r_t) \hat{A}_t$')
ax1.plot(r, L_pos, 'g-', lw=3, label=r'$L^{CLIP}$ (최종)')

# 3구간 배경 표시
ax1.axvspan(0.3, 1-epsilon, alpha=0.1, color='red', label=f'구간1: r < {1-epsilon}')
ax1.axvspan(1-epsilon, 1+epsilon, alpha=0.1, color='green', label=f'구간2: 클리핑 없음')
ax1.axvspan(1+epsilon, 2.0, alpha=0.1, color='blue', label=f'구간3: r > {1+epsilon}')

ax1.axvline(x=1.0, color='gray', ls=':', lw=1)
ax1.axvline(x=1-epsilon, color='red', ls='--', lw=1.5, alpha=0.7)
ax1.axvline(x=1+epsilon, color='blue', ls='--', lw=1.5, alpha=0.7)
ax1.set_xlabel(r'$r_t(\theta)$', fontsize=11)
ax1.set_ylabel(r'Objective', fontsize=11)
ax1.set_title(r'$\hat{A}_t > 0$ (좋은 행동)', fontweight='bold', fontsize=13)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.3, 2.0)

# (2) A < 0 (나쁜 행동)
ax2 = axes[1]
A_neg = -1.0
surr1_neg = r * A_neg
surr2_neg = r_clip * A_neg
L_neg = np.minimum(surr1_neg, surr2_neg)

ax2.plot(r, surr1_neg, 'b--', lw=1.5, alpha=0.5, label=r'$r_t \hat{A}_t$ (원래)')
ax2.plot(r, surr2_neg, 'r--', lw=1.5, alpha=0.5, label=r'$\mathrm{clip}(r_t) \hat{A}_t$')
ax2.plot(r, L_neg, 'g-', lw=3, label=r'$L^{CLIP}$ (최종)')

ax2.axvspan(0.3, 1-epsilon, alpha=0.1, color='red', label=f'구간1: r < {1-epsilon}')
ax2.axvspan(1-epsilon, 1+epsilon, alpha=0.1, color='green', label=f'구간2: 클리핑 없음')
ax2.axvspan(1+epsilon, 2.0, alpha=0.1, color='blue', label=f'구간3: r > {1+epsilon}')

ax2.axvline(x=1.0, color='gray', ls=':', lw=1)
ax2.axvline(x=1-epsilon, color='red', ls='--', lw=1.5, alpha=0.7)
ax2.axvline(x=1+epsilon, color='blue', ls='--', lw=1.5, alpha=0.7)
ax2.set_xlabel(r'$r_t(\theta)$', fontsize=11)
ax2.set_ylabel(r'Objective', fontsize=11)
ax2.set_title(r'$\hat{A}_t < 0$ (나쁜 행동)', fontweight='bold', fontsize=13)
ax2.legend(fontsize=8, loc='lower left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.3, 2.0)

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/ppo_clip_3regions.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/ppo_clip_3regions.png")

# 수치 분석
print(f"\\nPPO-Clip 3구간 수치 분석 (ε={epsilon}):")
print(f"{'':=<60}")
test_ratios = [0.5, 0.8, 1.0, 1.2, 1.5]
for rt in test_ratios:
    for A_val, A_label in [(1.0, "A>0"), (-1.0, "A<0")]:
        L_val = min(rt * A_val, np.clip(rt, 1-epsilon, 1+epsilon) * A_val)
        clipped = "✂️ 클리핑" if abs(L_val - rt * A_val) > 1e-6 else "✅ 원래값"
        print(f"  r_t={rt:.1f}, {A_label}: L_clip={L_val:+.2f} ({clipped})")"""))

# ── Cell 10: Section 4 header ────────────────────────────────────
cells.append(md("""\
## 4. A2C vs PPO 비교 구현 <a name='4.-A2C-vs-PPO-비교'></a>

간단한 연속 Bandit 환경에서 **A2C(Advantage Actor-Critic)**와 **PPO-Clip**의 학습 안정성을 비교합니다.

| 알고리즘 | 목적함수 | 특징 |
|---------|---------|------|
| A2C | $\\hat{A}_t \\nabla_\\theta \\log \\pi_\\theta$ | 단순하지만 큰 업데이트 가능 |
| PPO-Clip | $\\min(r_t \\hat{A}_t, \\text{clip}(r_t) \\hat{A}_t)$ | 업데이트 크기 제한 |"""))

# ── Cell 11: A2C vs PPO comparison code ──────────────────────────
cells.append(code("""\
# ── A2C vs PPO 비교 구현 ─────────────────────────────────────────
n_actions = 5
true_values = np.array([0.3, -0.2, 1.8, 0.5, -0.5])
n_episodes = 300
epsilon = 0.2

def run_a2c(lr=0.15):
    logits = tf.Variable(tf.zeros(n_actions))
    value_est = tf.Variable(0.0)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    rewards_hist = []

    for ep in range(n_episodes):
        with tf.GradientTape() as tape:
            probs = tf.nn.softmax(logits)
            action = tf.random.categorical(tf.math.log(probs[tf.newaxis, :]), 1)[0, 0]
            reward = true_values[action.numpy()] + np.random.randn() * 0.3

            # Advantage = r - V(s)
            advantage = reward - value_est
            # Actor loss
            log_prob = tf.math.log(probs[action] + 1e-8)
            actor_loss = -advantage * log_prob
            # Critic loss
            critic_loss = tf.square(advantage)
            loss = actor_loss + 0.5 * critic_loss

        grads = tape.gradient(loss, [logits, value_est])
        opt.apply_gradients(zip(grads, [logits, value_est]))
        rewards_hist.append(reward)

    return rewards_hist

def run_ppo(lr=0.15, eps=0.2, n_updates=3):
    logits = tf.Variable(tf.zeros(n_actions))
    value_est = tf.Variable(0.0)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    rewards_hist = []

    for ep in range(n_episodes):
        # 이전 정책으로 행동 선택
        old_probs = tf.nn.softmax(logits).numpy().copy()
        action_idx = np.random.choice(n_actions, p=old_probs)
        reward = true_values[action_idx] + np.random.randn() * 0.3
        advantage = reward - value_est.numpy()

        # PPO: 여러 번 업데이트
        for _ in range(n_updates):
            with tf.GradientTape() as tape:
                probs = tf.nn.softmax(logits)
                # 확률 비율
                ratio = probs[action_idx] / (old_probs[action_idx] + 1e-8)
                # PPO-Clip 목적
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1.0 - eps, 1.0 + eps) * advantage
                actor_loss = -tf.minimum(surr1, surr2)
                # Critic loss
                critic_loss = tf.square(reward - value_est)
                loss = actor_loss + 0.5 * critic_loss

            grads = tape.gradient(loss, [logits, value_est])
            opt.apply_gradients(zip(grads, [logits, value_est]))

        rewards_hist.append(reward)

    return rewards_hist

# 여러 번 실험하여 평균
n_runs = 5
a2c_all = []
ppo_all = []

for run in range(n_runs):
    np.random.seed(run * 100)
    tf.random.set_seed(run * 100)
    a2c_all.append(run_a2c())

    np.random.seed(run * 100)
    tf.random.set_seed(run * 100)
    ppo_all.append(run_ppo())

a2c_mean = np.mean(a2c_all, axis=0)
ppo_mean = np.mean(ppo_all, axis=0)
a2c_std = np.std(a2c_all, axis=0)
ppo_std = np.std(ppo_all, axis=0)

window = 20
a2c_smooth = np.convolve(a2c_mean, np.ones(window)/window, mode='valid')
ppo_smooth = np.convolve(ppo_mean, np.ones(window)/window, mode='valid')

print(f"A2C vs PPO 비교 ({n_runs}회 평균)")
print(f"{'':=<50}")
print(f"  최적 보상: {true_values.max():.1f} (Arm {np.argmax(true_values)})")
print(f"  A2C 최종 평균 보상 (마지막 50): {np.mean(a2c_mean[-50:]):.3f}")
print(f"  PPO 최종 평균 보상 (마지막 50): {np.mean(ppo_mean[-50:]):.3f}")
print(f"  A2C 보상 분산 (마지막 50):     {np.mean(a2c_std[-50:]):.3f}")
print(f"  PPO 보상 분산 (마지막 50):     {np.mean(ppo_std[-50:]):.3f}")"""))

# ── Cell 12: A2C vs PPO visualization ────────────────────────────
cells.append(code("""\
# ── A2C vs PPO 학습 곡선 시각화 ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) 학습 곡선 비교
ax1 = axes[0]
x_range = range(len(a2c_smooth))
ax1.plot(x_range, a2c_smooth, 'b-', lw=2, label='A2C', alpha=0.9)
ax1.plot(x_range, ppo_smooth, 'r-', lw=2, label='PPO-Clip', alpha=0.9)
ax1.axhline(y=true_values.max(), color='green', ls='--', lw=1.5,
            label=f'최적 보상 ({true_values.max():.1f})')
ax1.set_xlabel('에피소드', fontsize=11)
ax1.set_ylabel('평균 보상 (이동 평균)', fontsize=11)
ax1.set_title('A2C vs PPO 학습 곡선', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) 분산 비교 (롤링)
ax2 = axes[1]
a2c_rolling_var = np.array([np.var(a2c_mean[max(0,i-window):i+1])
                            for i in range(len(a2c_mean))])
ppo_rolling_var = np.array([np.var(ppo_mean[max(0,i-window):i+1])
                            for i in range(len(ppo_mean))])
ax2.plot(a2c_rolling_var, 'b-', lw=2, label='A2C 분산', alpha=0.9)
ax2.plot(ppo_rolling_var, 'r-', lw=2, label='PPO 분산', alpha=0.9)
ax2.set_xlabel('에피소드', fontsize=11)
ax2.set_ylabel('보상 분산 (롤링)', fontsize=11)
ax2.set_title('학습 안정성 비교', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/a2c_vs_ppo.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/a2c_vs_ppo.png")
print(f"PPO는 A2C 대비 더 안정적이고 일관된 학습 곡선을 보입니다.")"""))

# ── Cell 13: Section 5 header ────────────────────────────────────
cells.append(md(r"""\
## 5. KL 페널티 vs Clip 비교 <a name='5.-KL-페널티-vs-Clip'></a>

PPO에는 두 가지 변형이 있습니다:

**PPO-Clip** (실무에서 주로 사용):

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]$$

**PPO-Penalty** (KL 페널티):

$$L^{KL}(\theta) = \mathbb{E}\left[r_t \hat{A}_t\right] - \beta D_{KL}[\pi_{\theta_{old}} \| \pi_\theta]$$

| 특성 | PPO-Clip | PPO-Penalty |
|------|----------|-------------|
| 하이퍼파라미터 | $\epsilon$ (고정) | $\beta$ (적응 조절) |
| 구현 난이도 | 쉬움 | 중간 |
| 안정성 | 높음 | $\beta$ 조절에 민감 |
| 사용처 | 대부분의 RL | RLHF (InstructGPT) |"""))

# ── Cell 14: KL penalty vs Clip comparison ───────────────────────
cells.append(code("""\
# ── KL 페널티 vs Clip 비교 시각화 ────────────────────────────────
r = np.linspace(0.3, 2.0, 500)
epsilon = 0.2
A = 1.0

# PPO-Clip 목적
surr1 = r * A
r_clip = np.clip(r, 1 - epsilon, 1 + epsilon)
surr2 = r_clip * A
L_clip = np.minimum(surr1, surr2)

# PPO-Penalty: L = r*A - β * KL
# KL ≈ (r - 1) - log(r) (단일 행동 KL 근사)
kl_approx = (r - 1) - np.log(r + 1e-8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (1) PPO-Clip 목적
ax1 = axes[0]
ax1.plot(r, L_clip, 'g-', lw=3, label=r'$L^{CLIP}$')
ax1.plot(r, surr1, 'b--', lw=1.5, alpha=0.4, label=r'$r_t \hat{A}_t$')
ax1.axvline(x=1-epsilon, color='red', ls='--', lw=1, alpha=0.5)
ax1.axvline(x=1+epsilon, color='red', ls='--', lw=1, alpha=0.5)
ax1.axvline(x=1.0, color='gray', ls=':', lw=1)
ax1.set_xlabel(r'$r_t(\theta)$', fontsize=11)
ax1.set_ylabel('Objective', fontsize=11)
ax1.set_title('PPO-Clip', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) PPO-Penalty (다양한 β)
ax2 = axes[1]
betas = [0.01, 0.05, 0.1, 0.3]
colors = ['#2196F3', '#FF9800', '#E91E63', '#9C27B0']
for beta, color in zip(betas, colors):
    L_kl = r * A - beta * kl_approx
    ax2.plot(r, L_kl, lw=2, color=color, label=f'β={beta}')
ax2.plot(r, surr1, 'k--', lw=1, alpha=0.3, label=r'$r_t \hat{A}_t$ (원래)')
ax2.axvline(x=1.0, color='gray', ls=':', lw=1)
ax2.set_xlabel(r'$r_t(\theta)$', fontsize=11)
ax2.set_ylabel('Objective', fontsize=11)
ax2.set_title('PPO-Penalty (다양한 β)', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# (3) KL divergence 근사
ax3 = axes[2]
ax3.plot(r, kl_approx, 'r-', lw=2.5, label=r'$D_{KL} \approx (r-1) - \log r$')
ax3.fill_between(r, 0, kl_approx, alpha=0.1, color='red')
ax3.axvline(x=1.0, color='gray', ls=':', lw=1, label='r=1 (정책 동일)')
ax3.set_xlabel(r'$r_t(\theta)$', fontsize=11)
ax3.set_ylabel(r'$D_{KL}$', fontsize=11)
ax3.set_title('KL Divergence 근사', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/kl_penalty_vs_clip.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/kl_penalty_vs_clip.png")

# 정량적 비교
print(f"\\nPPO-Clip vs PPO-Penalty 비교:")
print(f"{'':=<55}")
print(f"{'r_t':>6} | {'L_clip':>8} | {'L_penalty(β=0.1)':>18} | {'KL':>8}")
print(f"{'-'*45}")
for rt in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8]:
    lc = min(rt * A, np.clip(rt, 1-epsilon, 1+epsilon) * A)
    kl = (rt - 1) - np.log(rt)
    lp = rt * A - 0.1 * kl
    print(f"{rt:>6.1f} | {lc:>+8.3f} | {lp:>+18.3f} | {kl:>8.3f}")"""))

# ── Cell 15: PPO in LLM context discussion ───────────────────────
cells.append(code("""\
# ── PPO의 LLM 정렬 적용 시뮬레이션 ──────────────────────────────
# RLHF에서 PPO가 사용되는 맥락을 간단히 시뮬레이션

vocab_size = 50
seq_len = 8
n_steps = 200
epsilon = 0.2
beta_kl = 0.05

# 기준 정책 (uniform에 가까운 소프트맥스)
ref_logits = tf.constant(np.random.randn(vocab_size).astype(np.float32) * 0.1)
ref_probs = tf.nn.softmax(ref_logits).numpy()

# 학습 정책
policy_logits = tf.Variable(tf.identity(ref_logits))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# 간단한 보상 모델: 특정 토큰(ID 10~15)을 선호
preferred_tokens = set(range(10, 16))

kl_history = []
reward_history = []

for step in range(n_steps):
    old_probs = tf.nn.softmax(policy_logits).numpy().copy()

    # 토큰 생성
    action = np.random.choice(vocab_size, p=old_probs)

    # 보상 모델 점수
    reward = 1.0 if action in preferred_tokens else -0.1

    with tf.GradientTape() as tape:
        probs = tf.nn.softmax(policy_logits)

        # PPO-Clip + KL 페널티 (RLHF 표준 방식)
        ratio = probs[action] / (old_probs[action] + 1e-8)
        advantage = reward - tf.reduce_sum(probs * tf.constant(
            [1.0 if i in preferred_tokens else -0.1 for i in range(vocab_size)]))

        surr1 = ratio * advantage
        surr2 = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
        clip_loss = -tf.minimum(surr1, surr2)

        # KL 페널티
        kl = tf.reduce_sum(probs * tf.math.log(probs / (ref_probs + 1e-8) + 1e-8))
        total_loss = clip_loss + beta_kl * kl

    grads = tape.gradient(total_loss, [policy_logits])
    optimizer.apply_gradients(zip(grads, [policy_logits]))

    kl_history.append(kl.numpy())
    reward_history.append(reward)

final_probs = tf.nn.softmax(policy_logits).numpy()
preferred_mass = sum(final_probs[i] for i in preferred_tokens)
print(f"RLHF 스타일 PPO 시뮬레이션 결과:")
print(f"{'':=<50}")
print(f"  선호 토큰(ID 10~15) 총 확률:")
print(f"    초기: {sum(ref_probs[i] for i in preferred_tokens):.4f}")
print(f"    최종: {preferred_mass:.4f}")
print(f"  최종 KL divergence: {kl_history[-1]:.4f}")
print(f"  평균 보상 (마지막 50): {np.mean(reward_history[-50:]):.3f}")
print(f"\\n  → KL 페널티 덕분에 기준 정책에서 크게 벗어나지 않으면서")
print(f"    선호 토큰의 확률이 점진적으로 증가합니다.")"""))

# ── Cell 16: Summary ─────────────────────────────────────────────
cells.append(md(r"""\
## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| Advantage Function | $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ — 평균 대비 행동 가치 | ⭐⭐⭐ |
| GAE | $\hat{A}_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$ — 편향/분산 조절 | ⭐⭐ |
| PPO-Clip | $\min(r_t\hat{A}_t, \text{clip}(r_t)\hat{A}_t)$ — 3구간 제한 | ⭐⭐⭐ |
| PPO-Penalty | $r_t\hat{A}_t - \beta D_{KL}$ — KL 기반 제한 | ⭐⭐ |
| 확률 비율 $r_t$ | $\pi_\theta / \pi_{\theta_{old}}$ — 정책 변화 측정 | ⭐⭐⭐ |
| Actor-Critic | Actor(정책) + Critic(가치) 동시 학습 | ⭐⭐ |
| Trust Region | 정책 업데이트 범위를 신뢰 영역 내로 제한 | ⭐⭐⭐ |

### 핵심 수식

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

### 다음 챕터 예고
**03_rlhf_pipeline_overview.ipynb** — InstructGPT의 SFT→Reward Model→PPO 3단계 RLHF 파이프라인을 구축하고, Bradley-Terry 선호 모델의 수식을 도출합니다."""))

create_notebook(cells, 'chapter15_alignment_rlhf/02_actor_critic_and_ppo.ipynb')
