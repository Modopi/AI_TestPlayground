"""Generate Chapter 12-04: MoE Routing and Load Balancing."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helper import md, code, create_notebook

cells = [
md("""# Chapter 12: 최신 LLM 아키텍처 — MoE 라우팅과 부하 균형

## 학습 목표
- Mixture of Experts(MoE)의 **Top-k 게이팅 수식**과 동작 원리를 이해한다
- **Softmax 게이팅**과 **Linear 게이팅**의 차이를 비교하고 구현한다
- **Auxiliary Loss (보조 손실)**의 수학적 도출과 부하 균형 효과를 검증한다
- **Expert Capacity Factor**가 MoE 성능에 미치는 영향을 실험한다

## 목차
1. [수학적 기초: MoE 게이팅과 Auxiliary Loss](#1.-수학적-기초)
2. [Top-k 라우터 구현](#2.-Top-k-라우터)
3. [Auxiliary Loss와 부하 균형](#3.-Auxiliary-Loss)
4. [Expert Capacity Factor 실험](#4.-Expert-Capacity)
5. [정리](#5.-정리)"""),

md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### MoE 게이팅 메커니즘

MoE 레이어는 $N$개의 전문가(Expert) 중 **Top-k개만 활성화**합니다:

$$y = \sum_{i \in \text{Top-k}} g_i \cdot E_i(x)$$

**게이팅 가중치 계산:**

$$\mathbf{h} = x \cdot W_g, \quad W_g \in \mathbb{R}^{d_{model} \times N}$$

$$g_i = \frac{e^{h_i}}{\sum_{j \in \text{Top-k}} e^{h_j}} \quad (\text{Top-k 선택 후 재정규화})$$

- $x$: 입력 토큰 히든 벡터
- $W_g$: 게이팅 가중치 (라우터)
- $E_i$: $i$번째 전문가 FFN
- $g_i$: $i$번째 전문가의 기여 가중치

### Auxiliary Loss (부하 균형 보조 손실)

전문가에게 토큰이 **균등하게 분배**되도록 유도하는 손실:

$$L_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

- $f_i = \frac{\text{expert } i\text{에 라우팅된 토큰 수}}{\text{전체 토큰 수}}$ (실제 분배 비율)
- $P_i = \frac{1}{T}\sum_{x \in \mathcal{B}} p_i(x)$ (라우팅 확률 평균, $p_i(x) = \text{softmax}(h)_i$)
- $\alpha$: 보조 손실 계수 (보통 $10^{-2}$)
- $N$: 전문가 수

**균등 분배 시 최솟값:** $f_i = P_i = 1/N$ → $L_{aux} = \alpha \cdot N \cdot N \cdot (1/N)^2 = \alpha$

**요약 표:**

| 항목 | 수식 | 설명 |
|------|------|------|
| 게이팅 | $g_i = \text{softmax}(\text{Top-k}(xW_g))_i$ | k개 전문가 선택 |
| Expert 출력 | $y = \sum_{i \in \text{Top-k}} g_i \cdot E_i(x)$ | 가중 합산 |
| Aux Loss | $L_{aux} = \alpha N \sum f_i P_i$ | 부하 균형 |
| Capacity Factor | $C = \frac{k \cdot T}{N}$ | 전문가당 최대 토큰 |

---

### 🐣 초등학생을 위한 MoE 친절 설명!

#### 🍳 MoE가 뭔가요?

> 💡 **비유**: **뷔페 레스토랑**을 상상해보세요!
> - 요리사(Expert) 8명이 각각 다른 요리를 전문으로 해요
> - 손님(Token)이 오면 **2명의 요리사만 골라서** 음식을 받아요 (Top-2)
> - 이렇게 하면 모든 요리사가 동시에 일하지 않아도 되니까 빠르고 효율적!

#### ⚖️ Auxiliary Loss는 뭔가요?

> 💡 **비유**: 만약 모든 손님이 **피자 요리사만 찾아가면** 다른 요리사는 놀고 있잖아요!
> Auxiliary Loss는 "손님을 골고루 나눠주세요!"라고 알려주는 **공정 분배 규칙**이에요.

| 상황 | 비유 | 결과 |
|------|------|------|
| Aux Loss 없음 | 인기 있는 요리사만 바쁨 | 불균형 → 성능↓ |
| Aux Loss 있음 | 손님을 골고루 배분 | 균형 → 효율↑ |"""),

md(r"""---

### 📝 연습 문제

#### 문제 1: Top-2 게이팅 계산

입력 $h = [2.0, 1.0, 0.5, 3.0]$ (전문가 4명)에서 Top-2를 선택하고 게이팅 가중치를 계산하시오.

<details>
<summary>💡 풀이 확인</summary>

Top-2 선택: Expert 3 ($h_3=3.0$), Expert 0 ($h_0=2.0$)

재정규화: $g_3 = \frac{e^{3.0}}{e^{3.0}+e^{2.0}} = \frac{20.09}{20.09+7.39} = 0.731$

$g_0 = \frac{e^{2.0}}{e^{3.0}+e^{2.0}} = \frac{7.39}{27.48} = 0.269$

출력: $y = 0.731 \cdot E_3(x) + 0.269 \cdot E_0(x)$
</details>

#### 문제 2: Auxiliary Loss 계산

전문가 4명, $\alpha=0.01$, 배치 내 분배: $f=[0.5, 0.1, 0.1, 0.3]$, $P=[0.4, 0.2, 0.1, 0.3]$일 때 $L_{aux}$를 계산하시오.

<details>
<summary>💡 풀이 확인</summary>

$$L_{aux} = 0.01 \times 4 \times (0.5 \times 0.4 + 0.1 \times 0.2 + 0.1 \times 0.1 + 0.3 \times 0.3)$$
$$= 0.04 \times (0.20 + 0.02 + 0.01 + 0.09) = 0.04 \times 0.32 = 0.0128$$

균등 분배 시: $0.01 \times 4 \times 4 \times (1/4)^2 = 0.01$ → 현재 불균형!
</details>"""),

code("""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")"""),

md("""## 2. Top-k 라우터 구현 <a name='2.-Top-k-라우터'></a>"""),

code(r"""# ── Top-k MoE 라우터 구현 ─────────────────────────────────────
# Top-k 게이팅으로 전문가를 선택하고 출력을 합산합니다

class MoERouter(tf.keras.layers.Layer):
    # Top-k MoE Router with gating
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = tf.keras.layers.Dense(n_experts, use_bias=False, name='gate')

    def call(self, x):
        # x: [B, S, d_model]
        logits = self.gate(x)  # [B, S, n_experts]
        
        # Top-k 선택
        top_k_logits, top_k_indices = tf.math.top_k(logits, k=self.top_k)
        
        # Top-k에 대해서만 softmax (재정규화)
        top_k_gates = tf.nn.softmax(top_k_logits, axis=-1)  # [B, S, k]
        
        # 라우팅 확률 (전체 softmax, Aux Loss 계산용)
        routing_probs = tf.nn.softmax(logits, axis=-1)  # [B, S, N]
        
        return top_k_gates, top_k_indices, routing_probs


class MoELayer(tf.keras.layers.Layer):
    # Mixture of Experts Layer
    def __init__(self, d_model, d_ff, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = MoERouter(d_model, n_experts, top_k)
        # 각 Expert는 간단한 FFN
        self.experts = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(d_ff, activation='relu'),
                tf.keras.layers.Dense(d_model)
            ], name=f'expert_{i}')
            for i in range(n_experts)
        ]

    def call(self, x):
        B, S, D = x.shape
        gates, indices, routing_probs = self.router(x)
        
        # 각 토큰에 대해 선택된 Expert 출력을 가중 합산
        # (실제 구현은 scatter/gather 최적화, 여기서는 명시적 루프)
        output = tf.zeros_like(x)
        
        for k_idx in range(self.top_k):
            expert_indices = indices[:, :, k_idx]   # [B, S]
            expert_gates = gates[:, :, k_idx:k_idx+1]  # [B, S, 1]
            
            for e_idx in range(self.n_experts):
                mask = tf.cast(tf.equal(expert_indices, e_idx), tf.float32)
                mask = mask[:, :, tf.newaxis]  # [B, S, 1]
                
                if tf.reduce_sum(mask) > 0:
                    expert_out = self.experts[e_idx](x)  # [B, S, D]
                    output += expert_out * mask * expert_gates
        
        return output, routing_probs


# 테스트
d_model, d_ff, n_experts, top_k = 256, 512, 8, 2
moe = MoELayer(d_model, d_ff, n_experts, top_k)

x_test = tf.random.normal((2, 32, d_model))
output, probs = moe(x_test)

print("=" * 60)
print(f"MoE Layer (N={n_experts} experts, Top-{top_k})")
print("=" * 60)
print(f"입력 shape:  {x_test.shape}")
print(f"출력 shape:  {output.shape}")
print()

# 라우팅 통계
gates, indices, _ = moe.router(x_test)
flat_indices = tf.reshape(indices, [-1]).numpy()

print("라우팅 분포:")
for i in range(n_experts):
    count = np.sum(flat_indices == i)
    pct = count / len(flat_indices) * 100
    bar = '#' * int(pct / 2)
    print(f"  Expert {i}: {count:>4} tokens ({pct:>5.1f}%) {bar}")"""),

code(r"""# ── 게이팅 가중치 분포 시각화 ─────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: Expert별 라우팅 빈도
ax1 = axes[0]
expert_counts = [np.sum(flat_indices == i) for i in range(n_experts)]
colors_exp = plt.cm.Set3(np.linspace(0, 1, n_experts))
bars = ax1.bar(range(n_experts), expert_counts, color=colors_exp, edgecolor='black', lw=1)
ax1.axhline(y=len(flat_indices) / n_experts, color='red', ls='--', lw=2, label='균등 분배')
ax1.set_xlabel('Expert 인덱스', fontsize=11)
ax1.set_ylabel('라우팅된 토큰 수', fontsize=11)
ax1.set_title(f'Expert별 라우팅 빈도 (Top-{top_k})', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# 오른쪽: 게이팅 가중치 분포
ax2 = axes[1]
gate_values = gates.numpy().flatten()
ax2.hist(gate_values, bins=30, color='#1E88E5', edgecolor='black', alpha=0.7)
ax2.axvline(x=0.5, color='red', ls='--', lw=2, label='균등 (0.5)')
ax2.set_xlabel('게이팅 가중치 g_i', fontsize=11)
ax2.set_ylabel('빈도', fontsize=11)
ax2.set_title('Top-2 게이팅 가중치 분포', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter12_modern_llms/moe_routing.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter12_modern_llms/moe_routing.png")"""),

md("""## 3. Auxiliary Loss와 부하 균형 <a name='3.-Auxiliary-Loss'></a>"""),

code(r"""# ── Auxiliary Loss 구현 및 효과 검증 ──────────────────────────
# Aux Loss가 Expert 부하 균형에 미치는 영향을 시뮬레이션합니다

def compute_aux_loss(routing_probs, indices, n_experts, alpha=0.01):
    # routing_probs: [B, S, N] - 전체 softmax 확률
    # indices: [B, S, k] - 선택된 expert 인덱스
    B, S, N = routing_probs.shape
    T = B * S  # 전체 토큰 수
    
    # f_i: expert i에 실제 라우팅된 토큰 비율
    flat_indices = tf.reshape(indices, [-1])
    f = tf.zeros(N)
    for i in range(N):
        f_i = tf.reduce_sum(tf.cast(tf.equal(flat_indices, i), tf.float32))
        f = tf.tensor_scatter_nd_update(f, [[i]], [f_i / tf.cast(T, tf.float32)])
    
    # P_i: expert i의 평균 라우팅 확률
    P = tf.reduce_mean(routing_probs, axis=[0, 1])  # [N]
    
    # L_aux = alpha * N * sum(f_i * P_i)
    loss = alpha * tf.cast(N, tf.float32) * tf.reduce_sum(f * P)
    return loss, f, P

# 불균형 vs 균형 시나리오 비교
print("=" * 65)
print("Auxiliary Loss 효과 분석")
print("=" * 65)

# 시나리오 1: Aux Loss 없이 학습 (불균형 발생)
# 인위적으로 불균형한 라우팅 생성
logits_biased = tf.constant([[
    [5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Expert 0 선호
    [4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [4.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
] * 8])  # [1, 32, 8]
probs_biased = tf.nn.softmax(logits_biased, axis=-1)
_, indices_biased = tf.math.top_k(logits_biased, k=2)
loss_biased, f_biased, P_biased = compute_aux_loss(probs_biased, indices_biased, n_experts=8)

# 시나리오 2: 균형 잡힌 라우팅
logits_balanced = tf.random.normal((1, 32, 8))
probs_balanced = tf.nn.softmax(logits_balanced, axis=-1)
_, indices_balanced = tf.math.top_k(logits_balanced, k=2)
loss_balanced, f_balanced, P_balanced = compute_aux_loss(probs_balanced, indices_balanced, n_experts=8)

print(f"{'시나리오':<20} | {'Aux Loss':>12} | {'f 최대/최소':>15} | {'불균형도':>10}")
print("-" * 65)
f_b = f_biased.numpy()
f_bl = f_balanced.numpy()
print(f"{'불균형 (편향)':<20} | {loss_biased.numpy():>12.4f} | {f_b.max():.3f}/{f_b.min():.3f} | {'높음':>10}")
print(f"{'균형 (랜덤)':<20} | {loss_balanced.numpy():>12.4f} | {f_bl.max():.3f}/{f_bl.min():.3f} | {'낮음':>10}")
print()
print("Aux Loss가 높을수록 불균형 → Loss를 최소화하면 자동으로 균형 유도!")
print()

# Expert별 분배 비교
print("Expert별 토큰 분배 비율 (f_i):")
print(f"  {'Expert':<10}", end='')
for i in range(8):
    print(f" | {i:>6}", end='')
print()
print(f"  {'불균형':<10}", end='')
for v in f_b:
    print(f" | {v:>5.1%}", end='')
print()
print(f"  {'균형':<10}", end='')
for v in f_bl:
    print(f" | {v:>5.1%}", end='')
print()"""),

md("""## 4. Expert Capacity Factor 실험 <a name='4.-Expert-Capacity'></a>"""),

code(r"""# ── Expert Capacity Factor 실험 ───────────────────────────────
# Capacity가 Expert 활용률과 토큰 드롭에 미치는 영향을 분석합니다

def simulate_capacity(n_tokens, n_experts, top_k, capacity_factor):
    # Capacity: 각 Expert가 처리할 수 있는 최대 토큰 수
    capacity = int(capacity_factor * top_k * n_tokens / n_experts)
    
    # 랜덤 라우팅 (약간의 편향 포함)
    logits = np.random.randn(n_tokens, n_experts)
    logits[:, 0] += 1.0  # Expert 0에 약간의 편향
    
    top_k_indices = np.argsort(-logits, axis=-1)[:, :top_k]
    
    # Expert별 할당 (Capacity 제한)
    expert_counts = np.zeros(n_experts, dtype=int)
    assigned = 0
    dropped = 0
    
    for token in range(n_tokens):
        for k in range(top_k):
            e = top_k_indices[token, k]
            if expert_counts[e] < capacity:
                expert_counts[e] += 1
                assigned += 1
            else:
                dropped += 1
    
    total_assignments = n_tokens * top_k
    utilization = assigned / total_assignments
    drop_rate = dropped / total_assignments
    
    return capacity, utilization, drop_rate, expert_counts

n_tokens = 1024
n_experts = 8
top_k = 2

capacity_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
results = []

print("=" * 70)
print(f"Expert Capacity Factor 실험 (T={n_tokens}, N={n_experts}, k={top_k})")
print("=" * 70)
print(f"{'CF':>5} | {'Capacity':>10} | {'활용률':>8} | {'드롭률':>8} | {'Expert 분포 편차':>15}")
print("-" * 70)

for cf in capacity_factors:
    cap, util, drop, counts = simulate_capacity(n_tokens, n_experts, top_k, cf)
    std = np.std(counts)
    results.append((cf, cap, util, drop, std))
    print(f"{cf:>5.2f} | {cap:>10} | {util:>7.1%} | {drop:>7.1%} | {std:>15.1f}")

print()
print("핵심 관찰:")
print("  • CF < 1.0: 용량 부족 → 토큰 드롭 발생 (정보 손실)")
print("  • CF = 1.0: 이론상 딱 맞지만, 불균형 시 일부 드롭")
print("  • CF > 1.0: 여유 있지만, 메모리/연산 낭비 증가")
print("  • 실전 권장: CF = 1.0~1.25")"""),

code(r"""# ── Capacity Factor 시각화 ────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: CF별 활용률/드롭률
ax1 = axes[0]
cfs = [r[0] for r in results]
utils = [r[2] for r in results]
drops = [r[3] for r in results]

ax1.plot(cfs, utils, 'b-o', lw=2.5, ms=8, label='활용률')
ax1.plot(cfs, drops, 'r-s', lw=2.5, ms=8, label='드롭률')
ax1.axvline(x=1.0, color='gray', ls='--', lw=1.5, label='CF=1.0')
ax1.set_xlabel('Capacity Factor', fontsize=11)
ax1.set_ylabel('비율', fontsize=11)
ax1.set_title('CF별 Expert 활용률 vs 드롭률', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 오른쪽: Expert별 토큰 분배 (CF=1.0 vs CF=1.5)
ax2 = axes[1]
_, _, _, counts_1 = simulate_capacity(n_tokens, n_experts, top_k, 1.0)
_, _, _, counts_15 = simulate_capacity(n_tokens, n_experts, top_k, 1.5)

x_pos = np.arange(n_experts)
width = 0.35
ax2.bar(x_pos - width/2, counts_1, width, label='CF=1.0', color='#1E88E5', edgecolor='black')
ax2.bar(x_pos + width/2, counts_15, width, label='CF=1.5', color='#43A047', edgecolor='black')
ax2.axhline(y=n_tokens * top_k / n_experts, color='red', ls='--', lw=2, label='이상적 균등')
ax2.set_xlabel('Expert 인덱스', fontsize=11)
ax2.set_ylabel('할당된 토큰 수', fontsize=11)
ax2.set_title('CF별 Expert 토큰 분배', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chapter12_modern_llms/moe_capacity.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter12_modern_llms/moe_capacity.png")"""),

md(r"""## 5. 정리 <a name='5.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| Top-k 게이팅 | $N$개 Expert 중 $k$개만 활성화 → 연산 효율 | ⭐⭐⭐ |
| Auxiliary Loss | $L_{aux} = \alpha N \sum f_i P_i$ → Expert 부하 균형 유도 | ⭐⭐⭐ |
| Capacity Factor | Expert당 최대 토큰 수 제한 → 메모리/드롭 트레이드오프 | ⭐⭐ |
| Softmax 게이팅 | Top-k 선택 후 재정규화 → 부드러운 가중치 | ⭐⭐ |

### 핵심 수식

$$y = \sum_{i \in \text{Top-k}} g_i \cdot E_i(x), \quad g_i = \frac{e^{h_i}}{\sum_{j \in \text{Top-k}} e^{h_j}}$$

$$L_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

### 다음 챕터 예고
**Chapter 12-05: DeepSeek-V3 MoE 아키텍처** — Shared Expert, Auxiliary-Loss-Free 로드밸런싱, Multi-Token Prediction 등 최신 MoE 혁신을 다룹니다."""),
]

create_notebook(cells, 'chapter12_modern_llms/04_moe_routing_and_load_balancing.ipynb')
