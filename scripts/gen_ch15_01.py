"""Generate chapter15_alignment_rlhf/01_rl_fundamentals_mdp_policy.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 15: AI 얼라인먼트와 강화학습 — RL 기초와 MDP/Policy Gradient

## 학습 목표
- MDP(Markov Decision Process)의 수학적 정의와 구성 요소를 이해한다
- Bellman 방정식의 유도 과정을 수식으로 전개하고 Value Iteration을 구현한다
- REINFORCE(Policy Gradient) 알고리즘의 수식을 완전 도출하고 구현한다
- 리워드 신호(Reward Signal)가 NLP 정렬 문제에서 어떻게 설계되는지 이해한다
- 강화학습과 지도학습의 차이를 구별하고, LLM 정렬에서의 RL 필요성을 설명한다

## 목차
1. [수학적 기초: MDP와 Bellman 방정식](#1.-수학적-기초)
2. [GridWorld MDP 구현](#2.-GridWorld-MDP-구현)
3. [Value Iteration 시각화](#3.-Value-Iteration-시각화)
4. [REINFORCE Policy Gradient](#4.-REINFORCE-Policy-Gradient)
5. [NLP를 위한 리워드 신호 설계](#5.-NLP-리워드-신호-설계)
6. [정리](#6.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### MDP (Markov Decision Process)

MDP는 순차적 의사결정 문제를 수학적으로 정의하는 프레임워크입니다:

$$\mathcal{M} = (S, A, P, R, \gamma)$$

- $S$: 상태 공간 (State space)
- $A$: 행동 공간 (Action space)
- $P(s' | s, a)$: 상태 전이 확률 (Transition probability)
- $R(s, a)$: 보상 함수 (Reward function)
- $\gamma \in [0, 1)$: 할인 인자 (Discount factor)

### 정책과 가치 함수

**정책 (Policy):**

$$\pi(a | s) = P(A_t = a \mid S_t = s)$$

**상태 가치 함수 (State Value Function):**

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s\right]$$

**행동 가치 함수 (Action Value Function):**

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right]$$

### Bellman 방정식

$$V^\pi(s) = \sum_{a} \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\right]$$

**Bellman 최적 방정식:**

$$V^*(s) = \max_{a} \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')\right]$$

### REINFORCE (Policy Gradient)

목적 함수:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} R(s_t, a_t)\right]$$

**Policy Gradient Theorem:**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[G_t \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

- $G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k+1}$: 누적 보상 (Return)
- $\nabla_\theta \log \pi_\theta(a_t|s_t)$: Score function (로그 정책의 기울기)

**요약 표:**

| 구분 | 수식 | 설명 |
|------|------|------|
| MDP 5-튜플 | $(S, A, P, R, \gamma)$ | 순차적 의사결정 프레임워크 |
| Bellman 방정식 | $V^\pi(s) = \sum_a \pi(a|s)[R + \gamma \sum_{s'} PV^\pi]$ | 현재 가치 = 즉각 보상 + 미래 가치 |
| Policy Gradient | $\nabla_\theta J = \mathbb{E}[G_t \nabla_\theta \log \pi_\theta]$ | 높은 보상 → 해당 행동 확률 증가 |
| Return | $G_t = \sum_k \gamma^k R_{t+k+1}$ | 미래 보상의 할인 합 |

---"""))

# ── Cell 3: 🐣 friendly explanation ──────────────────────────────
cells.append(md("""\
### 🐣 초등학생을 위한 강화학습 친절 설명!

#### 🔢 MDP가 뭔가요?

> 💡 **비유**: 보드게임을 생각해 보세요! 주사위를 던져서(행동) 칸을 이동하고(상태 전이), 
> 어떤 칸에 도착하면 용돈을 받고(보상), 어떤 칸에 가면 벌금을 내요(음의 보상).

MDP는 이런 게임의 규칙을 수학으로 정리한 거예요:
- **상태(S)**: 지금 내가 어디에 있는지 (게임판 위의 내 위치)
- **행동(A)**: 내가 할 수 있는 선택 (위/아래/왼쪽/오른쪽 이동)
- **보상(R)**: 행동의 결과로 받는 점수 (용돈 또는 벌금)
- **할인인자(γ)**: 나중에 받을 보상은 조금 덜 중요해요 (지금 100원 > 내일 100원)

#### 🎯 Policy Gradient가 뭔가요?

> 💡 **비유**: 눈을 감고 다트를 던지는 연습을 한다고 해 봐요!

처음에는 아무 데나 던지지만, **과녁에 맞을 때마다 그때 던진 방법을 더 많이 써요**.
이것이 바로 Policy Gradient의 핵심이에요:
- 좋은 결과($G_t$가 큼) → 그 행동을 **더 자주** 하도록 정책 업데이트
- 나쁜 결과($G_t$가 작음) → 그 행동을 **덜 자주** 하도록 정책 업데이트

---"""))

# ── Cell 4: 📝 Exercises ──────────────────────────────────────────
cells.append(md(r"""\
### 📝 연습 문제

#### 문제 1: Bellman 방정식 계산

2개의 상태 $s_1, s_2$와 결정적 정책 $\pi(s_1) = a_R$ (오른쪽)이 있습니다.
- $R(s_1, a_R) = 5$, $P(s_2 | s_1, a_R) = 1.0$
- $R(s_2, a_R) = 10$ (종료 상태), $\gamma = 0.9$

$V^\pi(s_1)$을 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$$V^\pi(s_2) = 10 \quad \text{(종료 상태, 이후 보상 없음)}$$

$$V^\pi(s_1) = R(s_1, a_R) + \gamma \cdot P(s_2|s_1, a_R) \cdot V^\pi(s_2)$$

$$= 5 + 0.9 \times 1.0 \times 10 = 5 + 9 = 14$$

→ $V^\pi(s_1) = 14$. 현재 보상(5)과 할인된 미래 보상(9)의 합입니다.
</details>

#### 문제 2: Return 계산

보상 시퀀스 $R_1=1, R_2=2, R_3=3$이고 $\gamma=0.5$일 때 $G_0$은?

<details>
<summary>💡 풀이 확인</summary>

$$G_0 = R_1 + \gamma R_2 + \gamma^2 R_3 = 1 + 0.5 \times 2 + 0.25 \times 3 = 1 + 1 + 0.75 = 2.75$$

→ 할인 인자가 작을수록 근시안적(가까운 보상 선호), 클수록 원시안적(미래 보상도 중시)입니다.
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
## 2. GridWorld MDP 구현 <a name='2.-GridWorld-MDP-구현'></a>

4×4 격자 세계(GridWorld)를 MDP로 구현합니다. 에이전트는 상/하/좌/우로 이동하며, 
목표 지점에 도달하면 +1 보상, 함정에 빠지면 -1 보상을 받습니다.

| 구성 요소 | 설정 |
|-----------|------|
| 상태 공간 | 4×4 = 16개 상태 |
| 행동 공간 | {상, 하, 좌, 우} = 4개 |
| 보상 | 목표(+1), 함정(-1), 이동(-0.04) |
| 할인 인자 | $\\gamma = 0.99$ |"""))

# ── Cell 7: GridWorld MDP implementation ──────────────────────────
cells.append(code("""\
# ── GridWorld MDP 구현 ────────────────────────────────────────────
class GridWorldMDP:
    # 4x4 격자 세계 MDP 구현
    # 상태: 0~15 (4x4 격자), 행동: 0=상, 1=하, 2=좌, 3=우

    def __init__(self, size=4, gamma=0.99):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.gamma = gamma

        # 특수 상태 설정
        self.goal = 15       # 우하단 = 목표
        self.trap = 11       # 함정
        self.terminal = {self.goal, self.trap}

        # 보상 설정
        self.rewards = np.full(self.n_states, -0.04)  # 이동 비용
        self.rewards[self.goal] = 1.0   # 목표 도달
        self.rewards[self.trap] = -1.0  # 함정

        # 행동 방향: (행 변화, 열 변화)
        self.action_effects = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_names = ['↑', '↓', '←', '→']

    def _to_rc(self, s):
        return s // self.size, s % self.size

    def _to_s(self, r, c):
        return r * self.size + c

    def step(self, state, action):
        # 종료 상태면 그대로 반환
        if state in self.terminal:
            return state, 0.0

        r, c = self._to_rc(state)
        dr, dc = self.action_effects[action]
        nr, nc = r + dr, c + dc

        # 격자 벽 체크
        if 0 <= nr < self.size and 0 <= nc < self.size:
            next_state = self._to_s(nr, nc)
        else:
            next_state = state

        return next_state, self.rewards[next_state]

    def get_transition_prob(self, state, action):
        # 결정적 전이 (확률 1.0)
        next_state, reward = self.step(state, action)
        return [(next_state, reward, 1.0)]


env = GridWorldMDP()
print(f"GridWorld MDP 생성 완료")
print(f"  상태 공간: {env.n_states}개 (4×4 격자)")
print(f"  행동 공간: {env.n_actions}개 ({', '.join(env.action_names)})")
print(f"  할인 인자: γ = {env.gamma}")
print(f"  목표 상태: {env.goal} (보상: +{env.rewards[env.goal]})")
print(f"  함정 상태: {env.trap} (보상: {env.rewards[env.trap]})")
print(f"  이동 비용: {env.rewards[0]}")

# 격자 보상 맵 출력
print(f"\\n보상 맵 (4×4):")
reward_grid = env.rewards.reshape(4, 4)
for r in range(4):
    row_str = ""
    for c in range(4):
        s = r * 4 + c
        if s == env.goal:
            row_str += " [+1.0 G] "
        elif s == env.trap:
            row_str += " [-1.0 T] "
        else:
            row_str += f" [{env.rewards[s]:+.2f}] "
    print(f"  {row_str}")"""))

# ── Cell 8: Value Iteration section ──────────────────────────────
cells.append(md(r"""\
## 3. Value Iteration 시각화 <a name='3.-Value-Iteration-시각화'></a>

Bellman 최적 방정식을 반복적으로 적용하여 최적 가치 함수 $V^*$를 구합니다:

$$V_{k+1}(s) = \max_{a} \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s')\right]$$

수렴 조건: $\max_s |V_{k+1}(s) - V_k(s)| < \theta$"""))

# ── Cell 9: Value Iteration code ─────────────────────────────────
cells.append(code("""\
# ── Value Iteration 구현 ─────────────────────────────────────────
def value_iteration(env, theta=1e-6, max_iter=1000):
    V = np.zeros(env.n_states)
    history = [V.copy()]

    for iteration in range(max_iter):
        V_new = np.zeros(env.n_states)

        for s in range(env.n_states):
            if s in env.terminal:
                V_new[s] = env.rewards[s]
                continue

            q_values = []
            for a in range(env.n_actions):
                transitions = env.get_transition_prob(s, a)
                q = sum(prob * (reward + env.gamma * V[ns])
                        for ns, reward, prob in transitions)
                q_values.append(q)
            V_new[s] = max(q_values)

        delta = np.max(np.abs(V_new - V))
        V = V_new
        history.append(V.copy())

        if delta < theta:
            print(f"Value Iteration 수렴: {iteration + 1}회 반복")
            break

    # 최적 정책 추출
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        if s in env.terminal:
            continue
        q_values = []
        for a in range(env.n_actions):
            transitions = env.get_transition_prob(s, a)
            q = sum(prob * (reward + env.gamma * V[ns])
                    for ns, reward, prob in transitions)
            q_values.append(q)
        policy[s] = np.argmax(q_values)

    return V, policy, history


V_star, pi_star, vi_history = value_iteration(env)

print(f"\\n최적 가치 함수 V* (4×4):")
V_grid = V_star.reshape(4, 4)
for r in range(4):
    row_str = "  "
    for c in range(4):
        row_str += f"{V_grid[r, c]:+7.3f} "
    print(row_str)

print(f"\\n최적 정책 π* (4×4):")
for r in range(4):
    row_str = "  "
    for c in range(4):
        s = r * 4 + c
        if s == env.goal:
            row_str += "   G    "
        elif s == env.trap:
            row_str += "   T    "
        else:
            row_str += f"   {env.action_names[pi_star[s]]}    "
    print(row_str)"""))

# ── Cell 10: Value Iteration visualization ───────────────────────
cells.append(code("""\
# ── Value Iteration 수렴 시각화 ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (1) Value 수렴 과정
ax1 = axes[0]
iterations_to_show = [0, 1, 5, 10, 20, len(vi_history)-1]
for it in iterations_to_show:
    if it < len(vi_history):
        ax1.plot(range(env.n_states), vi_history[it],
                 'o-', ms=5, lw=1.5, label=f'iter={it}')
ax1.set_xlabel('State', fontsize=11)
ax1.set_ylabel('V(s)', fontsize=11)
ax1.set_title('Value Iteration 수렴 과정', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) 최적 가치 함수 히트맵
ax2 = axes[1]
V_grid = V_star.reshape(4, 4)
im = ax2.imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
for r in range(4):
    for c in range(4):
        s = r * 4 + c
        label = f'{V_grid[r,c]:.2f}'
        if s == env.goal:
            label += '\\n(G)'
        elif s == env.trap:
            label += '\\n(T)'
        ax2.text(c, r, label, ha='center', va='center', fontsize=9, fontweight='bold')
fig.colorbar(im, ax=ax2, shrink=0.8)
ax2.set_title('최적 가치 함수 V*', fontweight='bold')

# (3) 최적 정책 시각화
ax3 = axes[2]
arrow_map = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}
ax3.set_xlim(-0.5, 3.5)
ax3.set_ylim(3.5, -0.5)
for r in range(4):
    for c in range(4):
        s = r * 4 + c
        if s == env.goal:
            ax3.add_patch(plt.Rectangle((c-0.4, r-0.4), 0.8, 0.8,
                                        color='green', alpha=0.3))
            ax3.text(c, r, 'G', ha='center', va='center', fontsize=14, fontweight='bold')
        elif s == env.trap:
            ax3.add_patch(plt.Rectangle((c-0.4, r-0.4), 0.8, 0.8,
                                        color='red', alpha=0.3))
            ax3.text(c, r, 'T', ha='center', va='center', fontsize=14, fontweight='bold')
        else:
            dx, dy = arrow_map[pi_star[s]]
            ax3.annotate('', xy=(c+dx, r-dy), xytext=(c, r),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
ax3.set_title('최적 정책 π*', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(4))
ax3.set_yticks(range(4))

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/value_iteration_gridworld.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/value_iteration_gridworld.png")
print(f"총 수렴 반복 수: {len(vi_history) - 1}")"""))

# ── Cell 11: REINFORCE section header ────────────────────────────
cells.append(md(r"""\
## 4. REINFORCE Policy Gradient <a name='4.-REINFORCE-Policy-Gradient'></a>

REINFORCE 알고리즘은 정책을 직접 매개변수화하여 기울기 상승법으로 최적화합니다.

**알고리즘 단계:**
1. 현재 정책 $\pi_\theta$로 에피소드 생성: $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots)$
2. 각 시간 단계의 Return 계산: $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k+1}$
3. 정책 기울기 계산: $\nabla_\theta J(\theta) = \sum_t G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$
4. 파라미터 업데이트: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

간단한 **K-Armed Bandit** 문제에 REINFORCE를 적용해 봅니다."""))

# ── Cell 12: REINFORCE implementation ────────────────────────────
cells.append(code("""\
# ── REINFORCE: K-Armed Bandit 문제 ───────────────────────────────
# 5개의 슬롯머신(arm), 각각 다른 평균 보상을 가짐
n_arms = 5
true_rewards = np.array([0.2, -0.5, 1.5, 0.8, -0.2])

# TF로 정책 네트워크 구현 (Softmax 정책)
policy_logits = tf.Variable(tf.zeros(n_arms), dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

n_episodes = 500
reward_history = []
action_probs_history = []

print(f"K-Armed Bandit REINFORCE")
print(f"  팔 개수: {n_arms}")
print(f"  실제 보상 평균: {true_rewards}")
print(f"  최적 팔: arm {np.argmax(true_rewards)} (보상={true_rewards.max()})")
print()

for ep in range(n_episodes):
    with tf.GradientTape() as tape:
        # 현재 정책으로 행동 선택
        probs = tf.nn.softmax(policy_logits)
        action = tf.random.categorical(tf.math.log(probs[tf.newaxis, :]), 1)[0, 0]

        # 보상 수집 (가우시안 노이즈 포함)
        reward = true_rewards[action.numpy()] + np.random.randn() * 0.5

        # REINFORCE loss: -G * log π(a|s)
        log_prob = tf.math.log(probs[action] + 1e-8)
        loss = -reward * log_prob

    grads = tape.gradient(loss, [policy_logits])
    optimizer.apply_gradients(zip(grads, [policy_logits]))

    reward_history.append(reward)
    action_probs_history.append(tf.nn.softmax(policy_logits).numpy().copy())

    if (ep + 1) % 100 == 0:
        curr_probs = tf.nn.softmax(policy_logits).numpy()
        avg_reward = np.mean(reward_history[-50:])
        best_arm = np.argmax(curr_probs)
        print(f"  에피소드 {ep+1:4d}: 평균 보상(최근 50)={avg_reward:+.3f}, "
              f"최선 arm={best_arm} (P={curr_probs[best_arm]:.3f})")

final_probs = tf.nn.softmax(policy_logits).numpy()
print(f"\\n최종 정책 분포:")
for i in range(n_arms):
    bar = '█' * int(final_probs[i] * 40)
    print(f"  Arm {i} (실제 r={true_rewards[i]:+.1f}): P={final_probs[i]:.4f} {bar}")"""))

# ── Cell 13: REINFORCE visualization ─────────────────────────────
cells.append(code("""\
# ── REINFORCE 학습 과정 시각화 ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) 보상 이동 평균
ax1 = axes[0]
window = 20
rewards_smooth = np.convolve(reward_history, np.ones(window)/window, mode='valid')
ax1.plot(range(len(rewards_smooth)), rewards_smooth, 'b-', lw=2, label='이동 평균 (w=20)')
ax1.axhline(y=true_rewards.max(), color='red', ls='--', lw=1.5,
            label=f'최적 보상 ({true_rewards.max()})')
ax1.fill_between(range(len(rewards_smooth)),
                 rewards_smooth - 0.3, rewards_smooth + 0.3,
                 alpha=0.1, color='blue')
ax1.set_xlabel('에피소드', fontsize=11)
ax1.set_ylabel('보상', fontsize=11)
ax1.set_title('REINFORCE 학습 곡선', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) 정책 확률 변화
ax2 = axes[1]
probs_arr = np.array(action_probs_history)
for i in range(n_arms):
    ax2.plot(probs_arr[:, i], lw=2, label=f'Arm {i} (r={true_rewards[i]:+.1f})')
ax2.set_xlabel('에피소드', fontsize=11)
ax2.set_ylabel('선택 확률', fontsize=11)
ax2.set_title('정책 확률 변화 (Softmax)', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/reinforce_bandit.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/reinforce_bandit.png")
print(f"최종 선택 확률: Arm {np.argmax(final_probs)} = {final_probs.max():.4f}")"""))

# ── Cell 14: NLP Reward Signal Design ────────────────────────────
cells.append(md("""\
## 5. NLP를 위한 리워드 신호 설계 <a name='5.-NLP-리워드-신호-설계'></a>

LLM 정렬(Alignment)에서 RL을 사용할 때, 리워드 신호를 어떻게 설계하는지가 핵심입니다.

| 리워드 유형 | 수식/설명 | 예시 |
|-------------|-----------|------|
| 인간 선호 기반 | $r(x, y) = \\text{RewardModel}(x, y)$ | RLHF |
| 규칙 기반 | $r = \\mathbb{1}[\\text{조건 충족}]$ | 길이 제한, 안전성 |
| 자동 메트릭 | $r = \\text{BLEU}(y, y^*)$ 또는 $\\text{ROUGE}$ | 번역 품질 |
| KL 페널티 | $r_{total} = r(x,y) - \\beta D_{KL}[\\pi_\\theta \\| \\pi_{ref}]$ | 정책 이탈 방지 |"""))

# ── Cell 15: NLP Reward Design code ──────────────────────────────
cells.append(code("""\
# ── NLP 리워드 신호 설계 예시 ────────────────────────────────────
# 간단한 시퀀스 생성 문제: 토큰 생성 시 다양한 리워드 신호를 적용

vocab_size = 100
seq_len = 10
n_samples = 1000

# 1. 길이 기반 리워드: 짧은 응답에 보너스
def length_reward(seq, target_len=5):
    actual_len = len(seq)
    return -abs(actual_len - target_len) / target_len

# 2. 다양성 리워드: 반복 토큰 패널티
def diversity_reward(seq):
    unique_ratio = len(set(seq)) / len(seq)
    return unique_ratio

# 3. 안전성 리워드: 금지 토큰(ID 0~9)이 없으면 보상
def safety_reward(seq, forbidden=set(range(10))):
    has_forbidden = any(t in forbidden for t in seq)
    return 0.0 if has_forbidden else 1.0

# 4. KL 페널티: 기준 정책에서 벗어나지 않도록
def kl_penalty(pi_logits, ref_logits, beta=0.1):
    pi_probs = tf.nn.softmax(pi_logits)
    ref_probs = tf.nn.softmax(ref_logits)
    kl = tf.reduce_sum(pi_probs * tf.math.log(pi_probs / (ref_probs + 1e-8) + 1e-8))
    return -beta * kl.numpy()

# 랜덤 시퀀스에 대해 리워드 분포 시각화
np.random.seed(42)
length_rewards = []
diversity_rewards = []
safety_rewards = []

for _ in range(n_samples):
    seq = np.random.randint(0, vocab_size, size=seq_len).tolist()
    length_rewards.append(length_reward(seq))
    diversity_rewards.append(diversity_reward(seq))
    safety_rewards.append(safety_reward(seq))

print(f"리워드 신호 통계 (랜덤 시퀀스 {n_samples}개):")
print(f"{'리워드 유형':<20} | {'평균':>8} | {'표준편차':>8} | {'최소':>8} | {'최대':>8}")
print(f"{'-'*62}")
for name, values in [("길이 기반", length_rewards),
                      ("다양성 기반", diversity_rewards),
                      ("안전성 기반", safety_rewards)]:
    vals = np.array(values)
    print(f"{name:<20} | {vals.mean():>8.3f} | {vals.std():>8.3f} | {vals.min():>8.3f} | {vals.max():>8.3f}")

# KL 페널티 예시
pi_logits = tf.constant(np.random.randn(vocab_size).astype(np.float32))
ref_logits = tf.constant(np.random.randn(vocab_size).astype(np.float32))
kl_val = kl_penalty(pi_logits, ref_logits, beta=0.1)
print(f"\\nKL 페널티 예시 (β=0.1): {kl_val:.4f}")
print(f"  → 정책이 기준 정책에서 멀어질수록 음의 보상이 커짐")
print(f"  → 이는 RLHF에서 모델이 reward hacking하는 것을 방지")"""))

# ── Cell 16: Summary ─────────────────────────────────────────────
cells.append(md(r"""\
## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| MDP | $(S, A, P, R, \gamma)$ — 순차적 의사결정 프레임워크 | ⭐⭐⭐ |
| Bellman 방정식 | $V^\pi(s) = \sum_a \pi(a|s)[R + \gamma \sum_{s'} PV^\pi]$ | ⭐⭐⭐ |
| Value Iteration | Bellman 최적 방정식 반복 → 최적 정책 | ⭐⭐ |
| REINFORCE | $\nabla_\theta J = \mathbb{E}[G_t \nabla_\theta \log \pi_\theta]$ | ⭐⭐⭐ |
| Return ($G_t$) | 미래 보상의 할인 합: $\sum_k \gamma^k R_{t+k+1}$ | ⭐⭐ |
| KL 페널티 | $-\beta D_{KL}[\pi_\theta \| \pi_{ref}]$ — 정책 이탈 방지 | ⭐⭐⭐ |
| 리워드 설계 | 인간 선호 / 규칙 / 메트릭 기반 보상 신호 | ⭐⭐ |

### 핵심 수식

$$V^\pi(s) = \sum_{a} \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')\right]$$

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[G_t \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

### 다음 챕터 예고
**02_actor_critic_and_ppo.ipynb** — Advantage Function을 도입하여 REINFORCE의 높은 분산 문제를 해결하고, A2C에서 PPO-Clip까지 수식을 완전 전개합니다."""))

create_notebook(cells, 'chapter15_alignment_rlhf/01_rl_fundamentals_mdp_policy.ipynb')
