"""Generate Chapter 12-05: DeepSeek MoE Architecture."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helper import md, code, create_notebook

cells = [
md("""# Chapter 12: 최신 LLM 아키텍처 — DeepSeek-V3 MoE 아키텍처

## 학습 목표
- DeepSeek-V3의 **Shared Expert + Routed Expert** 분리 설계의 수학적 원리를 이해한다
- **Auxiliary-Loss-Free 로드밸런싱**(편향 보정) 메커니즘을 구현한다
- **Multi-Token Prediction(MTP)** 수식을 도출하고 시뮬레이션한다
- DeepSeek-V3의 실제 아키텍처 스펙(671B, 256 Experts)을 분석한다

## 목차
1. [수학적 기초: DeepSeekMoE와 MTP](#1.-수학적-기초)
2. [DeepSeekMoE 레이어 구현](#2.-DeepSeekMoE)
3. [Auxiliary-Loss-Free 로드밸런싱](#3.-Aux-Free-Balance)
4. [Multi-Token Prediction (MTP)](#4.-MTP)
5. [정리](#5.-정리)"""),

md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### DeepSeekMoE: Shared Expert + Routed Expert

DeepSeek-V3는 기존 MoE와 달리 **공유 전문가(Shared Expert)**를 도입합니다:

$$y = \underbrace{E_s(x)}_{\text{Shared Expert}} + \sum_{i \in \text{Top-k}} g_i \cdot E_i^r(x)$$

- $E_s$: Shared Expert — **모든 토큰**이 반드시 거치는 전문가 (공통 지식)
- $E_i^r$: Routed Expert — Top-k로 **선택된** 전문가 (전문 지식)
- $g_i$: 라우팅 게이트 가중치

**DeepSeek-V3 실제 스펙** (arxiv:2412.19437):

| 항목 | 값 |
|------|-----|
| 총 파라미터 | 671B |
| 활성 파라미터/토큰 | 37B |
| 레이어 수 | 61 |
| Shared Expert | 1개 |
| Routed Expert | 256개 |
| Top-k | 8 |
| 히든 차원 | 7,168 |

### Auxiliary-Loss-Free 로드밸런싱

기존 Aux Loss($L_{aux}$)는 모델 성능을 저하시키는 부작용이 있었습니다. DeepSeek-V3는 **편향(bias) 항**으로 대체:

$$g_i' = g_i + b_i$$

- $b_i$: Expert $i$의 편향 항 (학습하지 않음!)
- **업데이트 규칙**: Expert $i$가 과도하게 사용되면 $b_i$ 감소, 적게 사용되면 $b_i$ 증가
- 하이퍼파라미터 $\gamma$가 편향 업데이트 속도를 제어

$$b_i \leftarrow b_i + \gamma \cdot (\bar{f} - f_i)$$

여기서 $\bar{f} = 1/N$ (이상적 균등 비율), $f_i$ = 실제 비율

### Multi-Token Prediction (MTP)

기존 LLM은 다음 1개 토큰만 예측하지만, MTP는 **여러 미래 토큰을 동시 예측**:

$$\mathcal{L}_{MTP} = -\frac{1}{D} \sum_{k=1}^{D} \sum_{t=1}^{T-k} \log P_\theta(x_{t+k} | x_{\leq t})$$

- $D$: 예측 깊이 (depth) — DeepSeek-V3는 $D=1$ (2토큰 동시)
- 학습 시 추가 예측 헤드 사용, 추론 시 Speculative Decoding에 활용

**요약 표:**

| 혁신 | 수식 | 효과 |
|------|------|------|
| Shared Expert | $y = E_s(x) + \sum g_i E_i^r(x)$ | 공통 지식 보존 |
| Aux-Free Balance | $g_i' = g_i + b_i$ | 성능 저하 없는 균형 |
| MTP | $\mathcal{L} = \sum_{k=1}^{D} \mathcal{L}_k$ | 표현력 향상 + Spec. Decoding |

---

### 🐣 초등학생을 위한 DeepSeek MoE 친절 설명!

#### 🏫 Shared Expert가 뭔가요?

> 💡 **비유**: **학교 수업**을 생각해보세요!
> - **공통 수업**(국어, 수학) = Shared Expert → 모든 학생이 들어야 함
> - **선택 과목**(미술, 음악, 코딩) = Routed Expert → 각자 원하는 2개만 선택
>
> 공통 지식은 모두가 배우고, 전문 지식은 필요한 학생만!

#### ⚖️ Auxiliary-Loss-Free가 뭔가요?

> 💡 **비유**: 기존 방법은 "반장이 벌점을 줘서" 골고루 나눴는데, DeepSeek 방법은 "각 선택 과목의 인기도를 보고 자동으로 시간표를 조정"해요. 벌점(Loss)이 필요 없어서 학습이 더 잘 됩니다!"""),

md(r"""---

### 📝 연습 문제

#### 문제 1: DeepSeek-V3 활성 파라미터

DeepSeek-V3에서 1개 Shared Expert + Top-8 Routed Expert가 활성화됩니다. 전체 256개 Routed Expert 중 활성화 비율은?

<details>
<summary>💡 풀이 확인</summary>

총 활성 Expert: $1 (\text{shared}) + 8 (\text{routed}) = 9$

Routed Expert 활성 비율: $8 / 256 = 3.125\%$

전체 Expert 활성 비율: $9 / 257 = 3.5\%$

→ 전체 671B 중 37B만 활성화 = $37/671 = 5.5\%$

나머지 파라미터는 임베딩, Attention 등에 사용
</details>

#### 문제 2: 편향 보정 시뮬레이션

Expert 4개, 현재 분배 $f = [0.4, 0.3, 0.2, 0.1]$, $\gamma=0.1$일 때 편향 업데이트 후 새 $b$를 계산하시오 (초기 $b=[0,0,0,0]$).

<details>
<summary>💡 풀이 확인</summary>

$\bar{f} = 1/4 = 0.25$

$b_0 = 0 + 0.1 \times (0.25 - 0.4) = -0.015$ (과다 → 감소)

$b_1 = 0 + 0.1 \times (0.25 - 0.3) = -0.005$

$b_2 = 0 + 0.1 \times (0.25 - 0.2) = +0.005$ (과소 → 증가)

$b_3 = 0 + 0.1 \times (0.25 - 0.1) = +0.015$

편향으로 인해 Expert 0,1은 선택 확률 감소, Expert 2,3은 증가 → 균형으로 수렴!
</details>"""),

code("""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")"""),

md("""## 2. DeepSeekMoE 레이어 구현 <a name='2.-DeepSeekMoE'></a>"""),

code(r"""# ── DeepSeekMoE 레이어 구현 ───────────────────────────────────
# Shared Expert 1개 + Routed Expert N개의 DeepSeek 스타일 MoE

class DeepSeekMoELayer(tf.keras.layers.Layer):
    # DeepSeek-V3 스타일 MoE: Shared Expert + Routed Experts
    def __init__(self, d_model, d_ff, n_routed_experts, top_k=2, n_shared=1):
        super().__init__()
        self.n_routed = n_routed_experts
        self.n_shared = n_shared
        self.top_k = top_k
        
        # Shared Expert(s)
        self.shared_experts = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(d_ff, activation='relu'),
                tf.keras.layers.Dense(d_model)
            ], name=f'shared_{i}')
            for i in range(n_shared)
        ]
        
        # Routed Experts
        self.routed_experts = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(d_ff, activation='relu'),
                tf.keras.layers.Dense(d_model)
            ], name=f'routed_{i}')
            for i in range(n_routed_experts)
        ]
        
        # Router (Routed Experts에 대해서만)
        self.gate = tf.keras.layers.Dense(n_routed_experts, use_bias=False, name='router')
        
        # Bias terms for Aux-Free balancing (non-trainable)
        self.expert_bias = tf.Variable(
            tf.zeros(n_routed_experts), trainable=False, name='expert_bias'
        )
    
    def call(self, x, training=False):
        B, S, D = x.shape
        
        # 1. Shared Expert 출력 (모든 토큰)
        shared_out = tf.zeros_like(x)
        for expert in self.shared_experts:
            shared_out += expert(x)
        
        # 2. Router 로짓 + 편향
        logits = self.gate(x)  # [B, S, N_routed]
        if not training:
            logits_biased = logits + self.expert_bias
        else:
            logits_biased = logits + self.expert_bias
        
        # Top-k 선택
        top_k_logits, top_k_indices = tf.math.top_k(logits_biased, k=self.top_k)
        top_k_gates = tf.nn.softmax(top_k_logits, axis=-1)  # [B, S, k]
        
        # 3. Routed Expert 출력
        routed_out = tf.zeros_like(x)
        for k_idx in range(self.top_k):
            expert_indices = top_k_indices[:, :, k_idx]
            expert_gates = top_k_gates[:, :, k_idx:k_idx+1]
            
            for e_idx in range(self.n_routed):
                mask = tf.cast(tf.equal(expert_indices, e_idx), tf.float32)
                mask = mask[:, :, tf.newaxis]
                if tf.reduce_sum(mask) > 0:
                    e_out = self.routed_experts[e_idx](x)
                    routed_out += e_out * mask * expert_gates
        
        # 4. 합산: Shared + Routed
        output = shared_out + routed_out
        
        return output, top_k_indices, tf.nn.softmax(logits, axis=-1)


# 데모 (축소 버전)
d_model, d_ff = 256, 512
n_routed, top_k = 8, 2

moe = DeepSeekMoELayer(d_model, d_ff, n_routed, top_k, n_shared=1)
x_test = tf.random.normal((2, 16, d_model))
output, indices, probs = moe(x_test)

print("=" * 60)
print(f"DeepSeekMoE Layer")
print(f"  Shared Experts: 1, Routed Experts: {n_routed}, Top-{top_k}")
print("=" * 60)
print(f"입력 shape:  {x_test.shape}")
print(f"출력 shape:  {output.shape}")
print()

# 파라미터 분석
shared_params = sum(tf.size(v).numpy() for exp in moe.shared_experts for v in exp.trainable_variables)
routed_params = sum(tf.size(v).numpy() for exp in moe.routed_experts for v in exp.trainable_variables)
router_params = sum(tf.size(v).numpy() for v in moe.gate.trainable_variables)

print(f"{'구성 요소':<25} | {'파라미터':>12}")
print("-" * 42)
print(f"{'Shared Expert (1개)':<25} | {shared_params:>12,}")
print(f"{'Routed Experts (8개)':<25} | {routed_params:>12,}")
print(f"{'Router':<25} | {router_params:>12,}")
print(f"{'총계':<25} | {shared_params+routed_params+router_params:>12,}")
print(f"{'활성 파라미터 (1+2)':<25} | {shared_params + routed_params//n_routed*top_k + router_params:>12,}")"""),

md("""## 3. Auxiliary-Loss-Free 로드밸런싱 <a name='3.-Aux-Free-Balance'></a>"""),

code(r"""# ── Auxiliary-Loss-Free 로드밸런싱 시뮬레이션 ─────────────────
# 편향 보정 방식으로 Expert 부하를 균형시키는 과정을 시뮬레이션합니다

def simulate_aux_free_balancing(n_experts=8, n_tokens=256, n_steps=50, gamma=0.1, top_k=2):
    # 편향이 점진적으로 균형을 찾아가는 과정
    biases = np.zeros(n_experts)
    
    # Expert 0, 1에 편향된 초기 라우터 가중치
    router_w = np.random.randn(64, n_experts) * 0.1
    router_w[:, 0] += 0.5  # Expert 0 선호
    router_w[:, 1] += 0.3  # Expert 1 약간 선호
    
    history = {'step': [], 'balance': [], 'max_load': [], 'min_load': []}
    load_history = []
    
    for step in range(n_steps):
        # 토큰 라우팅 시뮬레이션
        x = np.random.randn(n_tokens, 64)
        logits = x @ router_w + biases  # 편향 추가
        
        # Top-k 선택
        top_k_idx = np.argsort(-logits, axis=-1)[:, :top_k]
        
        # Expert별 로드 계산
        loads = np.zeros(n_experts)
        for e in range(n_experts):
            loads[e] = np.sum(top_k_idx == e) / (n_tokens * top_k)
        
        # 편향 업데이트: f_bar - f_i
        f_bar = 1.0 / n_experts
        biases += gamma * (f_bar - loads)
        
        history['step'].append(step)
        history['balance'].append(np.std(loads))
        history['max_load'].append(np.max(loads))
        history['min_load'].append(np.min(loads))
        load_history.append(loads.copy())
    
    return history, load_history, biases

history, load_hist, final_biases = simulate_aux_free_balancing()

print("=" * 65)
print("Auxiliary-Loss-Free 로드밸런싱 시뮬레이션")
print("=" * 65)
print(f"{'Step':<8} | {'부하 표준편차':>12} | {'최대 로드':>10} | {'최소 로드':>10}")
print("-" * 50)
for i in [0, 5, 10, 20, 49]:
    print(f"{i:<8} | {history['balance'][i]:>12.4f} | "
          f"{history['max_load'][i]:>9.1%} | {history['min_load'][i]:>9.1%}")
print()
print("최종 편향 값:")
for i, b in enumerate(final_biases):
    print(f"  Expert {i}: b={b:+.4f}")"""),

code(r"""# ── Aux-Free 로드밸런싱 수렴 과정 시각화 ─────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: 부하 편차 수렴
ax1 = axes[0]
ax1.plot(history['step'], history['balance'], 'b-', lw=2.5, label='부하 표준편차')
ax1.fill_between(history['step'], history['min_load'], history['max_load'],
                  alpha=0.2, color='orange', label='최대-최소 로드 범위')
ax1.axhline(y=0, color='gray', ls='--', lw=1)
ax1.set_xlabel('학습 스텝', fontsize=11)
ax1.set_ylabel('부하 편차 / 로드', fontsize=11)
ax1.set_title('Aux-Free 로드밸런싱 수렴', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 오른쪽: Expert별 로드 변화
ax2 = axes[1]
load_arr = np.array(load_hist)
for e in range(8):
    ax2.plot(range(len(load_hist)), load_arr[:, e], lw=1.5, 
             label=f'Expert {e}', alpha=0.7)
ax2.axhline(y=1/8, color='red', ls='--', lw=2, label='이상적 (12.5%)')
ax2.set_xlabel('학습 스텝', fontsize=11)
ax2.set_ylabel('Expert 로드 비율', fontsize=11)
ax2.set_title('Expert별 부하 변화', fontweight='bold')
ax2.legend(fontsize=8, ncol=3)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter12_modern_llms/deepseek_auxfree.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter12_modern_llms/deepseek_auxfree.png")"""),

md("""## 4. Multi-Token Prediction (MTP) <a name='4.-MTP'></a>"""),

code(r"""# ── Multi-Token Prediction 시뮬레이션 ─────────────────────────
# 다음 1개 토큰 예측(NTP) vs 다중 토큰 예측(MTP)의 학습 신호 비교

# 간단한 시퀀스 예측 시나리오
vocab_size = 100
seq_len = 32
d_model = 128

# 랜덤 시퀀스
sequence = np.random.randint(0, vocab_size, size=seq_len)

# NTP: 각 위치에서 다음 1토큰 예측
ntp_targets = sequence[1:]  # T-1개의 타겟
ntp_loss_terms = len(ntp_targets)

# MTP (depth=2): 각 위치에서 다음 2토큰 예측
mtp_depth = 2
mtp_loss_terms = 0
for d in range(1, mtp_depth + 1):
    mtp_loss_terms += max(0, seq_len - d)

print("=" * 60)
print("Multi-Token Prediction (MTP) vs Next-Token Prediction (NTP)")
print("=" * 60)
print(f"시퀀스 길이: {seq_len}")
print()
print(f"{'방법':<20} | {'예측 깊이':>10} | {'Loss 항 수':>12} | {'학습 신호':>10}")
print("-" * 60)
print(f"{'NTP (기존)':<20} | {'D=1':>10} | {ntp_loss_terms:>12} | {'기준':>10}")
print(f"{'MTP (DeepSeek)':<20} | {f'D={mtp_depth}':>10} | {mtp_loss_terms:>12} | {f'{mtp_loss_terms/ntp_loss_terms:.1f}x':>10}")
print()

# MTP 구조 시각화 (텍스트)
print("MTP 동작 예시 (D=2):")
print("=" * 50)
example_seq = "나는 오늘 학교에 갔다".split()
for t in range(min(4, len(example_seq)-2)):
    context = " ".join(example_seq[:t+1])
    target1 = example_seq[t+1] if t+1 < len(example_seq) else "?"
    target2 = example_seq[t+2] if t+2 < len(example_seq) else "?"
    print(f"  입력: [{context}]")
    print(f"    → NTP 예측: {target1}")
    print(f"    → MTP 예측: {target1}, {target2}")
    print()

print("MTP 장점:")
print("  1. 학습 신호 증가 → 동일 데이터로 더 많은 학습")
print("  2. 추론 시 MTP 헤드를 Draft Model로 활용 → Speculative Decoding")
print("  3. DeepSeek-V3: D=1 (2토큰 동시 예측)으로 성능 향상 확인")"""),

md(r"""## 5. 정리 <a name='5.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| Shared Expert | 모든 토큰이 거치는 공유 전문가 → 공통 지식 보존 | ⭐⭐⭐ |
| Aux-Free Balance | 편향 항 $b_i$로 부하 균형 → 성능 저하 없음 | ⭐⭐⭐ |
| MTP | 다중 미래 토큰 동시 예측 → 학습 효율 + Speculative Decoding | ⭐⭐⭐ |
| DeepSeek-V3 스케일 | 671B (37B active), 256 Routed + 1 Shared, Top-8 | ⭐⭐ |

### 핵심 수식

$$y = E_s(x) + \sum_{i \in \text{Top-k}} g_i \cdot E_i^r(x)$$

$$b_i \leftarrow b_i + \gamma(\bar{f} - f_i) \quad \text{(Aux-Free Balance)}$$

$$\mathcal{L}_{MTP} = -\frac{1}{D}\sum_{k=1}^{D}\sum_{t} \log P(x_{t+k}|x_{\leq t})$$

### DeepSeek-V3 핵심 스펙

| 항목 | 값 |
|------|-----|
| 총 파라미터 | 671B |
| 활성/토큰 | 37B |
| Shared Expert | 1 |
| Routed Expert | 256 |
| Top-k | 8 |
| 학습 데이터 | 14.8T 토큰 |
| 학습 비용 | 2.788M H800 GPU-hours |
| 정밀도 | FP8 혼합 정밀도 |

### 다음 챕터 예고
**Chapter 13: 생성 AI 심화 — 확산 모델과 SDE** — DDPM의 수학적 기초부터 노이즈 스케줄, DDIM 샘플러, CFG, Score Matching까지 확산 모델의 전 이론 체계를 다룹니다."""),
]

create_notebook(cells, 'chapter12_modern_llms/05_deepseek_moe_architecture.ipynb')
