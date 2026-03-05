"""Generate chapter14_extreme_inference/03_speculative_decoding.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 14: 극단적 추론 최적화 — Speculative Decoding

## 학습 목표
- Speculative Decoding의 **Draft-Verify 패러다임**이 왜 Memory-bound Decode를 가속하는지 이해한다
- 수용률(acceptance rate) $\\beta$와 드래프트 길이 $k$로부터 **기대 수용 토큰 수를 유도**한다
- 검증(Verification) 단계의 **수학적 정확성 보장 메커니즘**을 이해한다
- Medusa, EAGLE 등 **다중 헤드 방식**의 아키텍처와 Trade-off를 비교한다
- TensorFlow로 Draft-Verify 시뮬레이션을 구현하고 **속도 향상을 실험적으로 검증**한다

## 목차
1. [수학적 기초: 수용률과 기대 토큰 수](#1.-수학적-기초)
2. [Draft-Verify 시뮬레이션](#2.-Draft-Verify-시뮬레이션)
3. [기대 속도 향상 분석](#3.-속도-향상-분석)
4. [Medusa vs EAGLE 아키텍처 비교](#4.-Medusa-vs-EAGLE)
5. [토큰 수용 시각화](#5.-토큰-수용-시각화)
6. [정리](#6.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### Speculative Decoding 핵심 아이디어

Decode 단계는 **Memory-bound**이므로, 작은 Draft 모델로 $k$개 토큰을 빠르게 생성한 후
큰 Target 모델로 **한 번의 Forward Pass로 병렬 검증**합니다.

### 수용 확률 (Acceptance Probability)

Draft 모델의 분포 $q(x)$, Target 모델의 분포 $p(x)$에 대해:

$$\alpha(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

- $\alpha(x)$: 토큰 $x$의 수용 확률
- $q(x) \leq p(x)$인 토큰은 항상 수용
- $q(x) > p(x)$인 토큰은 $p(x)/q(x)$ 확률로 수용

### 기대 수용 토큰 수

드래프트 길이 $k$, 평균 수용률 $\beta$일 때, 기대 수용 토큰 수:

$$E[\text{accepted}] = \frac{1 - \beta^{k+1}}{1 - \beta}$$

**유도:**
- 첫 번째 토큰 수용 확률: $\beta$
- $i$번째 토큰까지 **모두** 수용될 확률: $\beta^i$
- 기대 수용 수: $\sum_{i=0}^{k} \beta^i = \frac{1 - \beta^{k+1}}{1 - \beta}$ (등비급수)

거부 시 Target 모델의 수정된 분포에서 1개 토큰을 샘플링하므로, 최소 1개 토큰은 항상 생성됩니다.

### Speedup 공식

$$\text{Speedup} = \frac{E[\text{accepted}]}{c \cdot k + 1}$$

- $c$: Draft 모델의 상대적 비용 ($c = T_{draft} / T_{target}$, 보통 $c \approx 0.05 \sim 0.1$)
- $k$: 드래프트 길이
- 분모의 1: Target 모델 검증 1회

### 검증의 정확성

Speculative Decoding은 **Target 모델의 출력 분포를 정확히 보존**합니다:

거부 시 수정 분포: $p'(x) = \text{norm}\left(\max(0, p(x) - q(x))\right)$

$$\text{최종 분포} = \alpha \cdot q(x) + (1-\alpha) \cdot p'(x) = p(x)$$

**요약 표:**

| 지표 | 수식 | 의미 |
|------|------|------|
| 수용 확률 | $\min(1, p(x)/q(x))$ | 토큰별 수용/거부 결정 |
| 기대 토큰 수 | $(1-\beta^{k+1})/(1-\beta)$ | 한 라운드 평균 생성 수 |
| Speedup | $E[\text{acc}] / (ck + 1)$ | 이론적 속도 향상 |
| 수정 분포 | $\text{norm}(\max(0, p-q))$ | 분포 정확성 보장 |"""))

# ── Cell 3: 🐣 friendly explanation ──────────────────────────────
cells.append(md("""\
### 🐣 초등학생을 위한 Speculative Decoding 친절 설명!

#### 🔢 Draft-Verify가 뭔가요?

> 💡 **비유**: 수학 시험에서 **연습장에 빠르게 답을 쓰고**, 선생님이 한꺼번에 채점하는 것!

**기존 방식**: 선생님(큰 모델)이 한 문제씩 직접 풀어요. 정확하지만 느려요! 🐢

**Speculative Decoding**: 
1. 학생(작은 Draft 모델)이 연습장에 5문제를 빠르게 풀어요 ✏️
2. 선생님(큰 Target 모델)이 5문제를 **한 번에** 채점해요 ✅❌
3. 맞은 데까지 채택하고, 틀린 문제부터 선생님이 직접 답을 써요

선생님이 채점하는 시간 = 문제 1개 푸는 시간이니까, 학생이 잘 맞출수록 빨라져요!

#### 📊 수용률이 뭔가요?

> 💡 **비유**: 학생의 **정답률**이에요!

- 수용률 90% → 10문제 중 9문제를 맞춰요 → 거의 10배 빨라질 수 있어요!
- 수용률 50% → 10문제 중 5문제를 맞춰요 → 약 2배 빨라져요
- 수용률이 낮으면 오히려 학생한테 시키는 것이 손해예요

---"""))

# ── Cell 4: 📝 Exercises ──────────────────────────────────────────
cells.append(md(r"""\
### 📝 연습 문제

#### 문제 1: 기대 토큰 수 계산

$\beta = 0.8$, $k = 5$일 때 기대 수용 토큰 수는?

<details>
<summary>💡 풀이 확인</summary>

$$E = \frac{1 - 0.8^{5+1}}{1 - 0.8} = \frac{1 - 0.8^6}{0.2} = \frac{1 - 0.262144}{0.2} = \frac{0.737856}{0.2} = 3.689$$

→ 평균 **3.69개** 토큰이 수용됩니다. (최소 1개 보장이므로 실제로는 더 높을 수 있음)
</details>

#### 문제 2: 최적 드래프트 길이

$\beta = 0.7$, Draft 비용 $c = 0.05$일 때, Speedup을 최대화하는 $k$는?

<details>
<summary>💡 풀이 확인</summary>

$k$별 Speedup 계산:

| $k$ | $E[\text{acc}]$ | $ck+1$ | Speedup |
|-----|---------|--------|---------|
| 1 | $\frac{1-0.49}{0.3}=1.70$ | 1.05 | 1.62 |
| 3 | $\frac{1-0.7^4}{0.3}=2.37$ | 1.15 | 2.06 |
| 5 | $\frac{1-0.7^6}{0.3}=2.72$ | 1.25 | 2.17 |
| 7 | $\frac{1-0.7^8}{0.3}=2.89$ | 1.35 | 2.14 |
| 10 | $\frac{1-0.7^{11}}{0.3}=2.98$ | 1.50 | 1.99 |

→ $k=5$ 부근에서 Speedup이 최대 (**~2.17x**)! $k$가 너무 크면 Draft 비용이 증가하여 역효과.
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
import time

np.random.seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"NumPy 버전: {np.__version__}")"""))

# ── Cell 6: Section 2 header ─────────────────────────────────────
cells.append(md(r"""\
## 2. Draft-Verify 시뮬레이션 <a name='2.-Draft-Verify-시뮬레이션'></a>

Draft 모델과 Target 모델의 분포를 시뮬레이션하여 Speculative Decoding의 동작을 검증합니다.

검증 알고리즘:
1. Draft 모델에서 $k$개 토큰 $x_1, \ldots, x_k$를 자기회귀 생성
2. Target 모델에서 $x_1, \ldots, x_k$를 **한 번에** 검증
3. $\alpha_i = \min(1, p(x_i)/q(x_i))$로 각 토큰 수용/거부
4. 첫 번째 거부 위치에서 수정 분포 $p'$에서 새 토큰 샘플링"""))

# ── Cell 7: Draft-Verify simulation ──────────────────────────────
cells.append(code("""\
# ── Draft-Verify 시뮬레이션 ──────────────────────────────────────
vocab_size = 100

def create_model_distribution(vocab_size, temperature=1.0):
    # 시뮬레이션용: 랜덤 logit으로 분포 생성
    logits = np.random.randn(vocab_size) / temperature
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs

def speculative_decode_step(target_probs, draft_probs, k):
    # Draft 단계: k개 토큰 생성
    draft_tokens = []
    for _ in range(k):
        token = np.random.choice(len(draft_probs), p=draft_probs)
        draft_tokens.append(token)

    # Verify 단계: Target 모델로 검증
    accepted = []
    for i, token in enumerate(draft_tokens):
        p = target_probs[token]
        q = draft_probs[token]
        alpha = min(1.0, p / q)

        if np.random.random() < alpha:
            accepted.append(token)
        else:
            # 수정 분포에서 새 토큰 샘플링
            residual = np.maximum(0, target_probs - draft_probs)
            residual_sum = np.sum(residual)
            if residual_sum > 0:
                residual /= residual_sum
                bonus_token = np.random.choice(len(residual), p=residual)
            else:
                bonus_token = np.random.choice(len(target_probs), p=target_probs)
            accepted.append(bonus_token)
            break
    else:
        # 모두 수용된 경우: Target에서 추가 1개 생성
        bonus = np.random.choice(len(target_probs), p=target_probs)
        accepted.append(bonus)

    return accepted, len(draft_tokens)

# 다양한 수용률로 실험
draft_temps = [0.5, 1.0, 2.0, 5.0]
k_values = [3, 5, 7]
n_trials = 2000

print(f"{'':=<75}")
print(f"  Speculative Decoding 시뮬레이션 (vocab={vocab_size}, trials={n_trials})")
print(f"{'':=<75}")

target_probs = create_model_distribution(vocab_size, temperature=1.0)

for k in k_values:
    print(f"\\n  Draft 길이 k={k}:")
    print(f"  {'Draft Temp':>12} | {'평균 수용':>8} | {'수용률 β':>8} | {'이론 E[acc]':>12} | {'실제 E[acc]':>12}")
    print(f"  {'-'*62}")

    for temp in draft_temps:
        draft_probs = create_model_distribution(vocab_size, temperature=temp)

        total_accepted = 0
        total_drafted = 0

        for _ in range(n_trials):
            accepted, drafted = speculative_decode_step(target_probs, draft_probs, k)
            total_accepted += len(accepted)
            total_drafted += drafted

        avg_accepted = total_accepted / n_trials
        beta = 1.0 - (total_drafted - total_accepted + n_trials) / (total_drafted + n_trials)
        beta = max(0.01, min(0.99, beta))
        theoretical = (1 - beta**(k+1)) / (1 - beta)

        print(f"  {temp:>12.1f} | {avg_accepted:>8.2f} | {beta:>8.3f} | {theoretical:>12.2f} | {avg_accepted:>12.2f}")

print(f"\\n결론: Draft 모델이 Target에 가까울수록 (온도 유사) 수용률이 높아짐")"""))

# ── Cell 8: Section 3 header ─────────────────────────────────────
cells.append(md(r"""\
## 3. 기대 속도 향상 분석 <a name='3.-속도-향상-분석'></a>

수용률 $\beta$와 드래프트 길이 $k$에 따른 이론적 Speedup을 분석합니다.

$$\text{Speedup}(\beta, k) = \frac{(1-\beta^{k+1}) / (1-\beta)}{c \cdot k + 1}$$"""))

# ── Cell 9: Expected speedup vs acceptance rate ──────────────────
cells.append(code("""\
# ── 수용률 vs 속도 향상 분석 ──────────────────────────────────────
betas = np.linspace(0.01, 0.99, 200)
c = 0.05  # Draft 모델 비용 비율

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
for k in [1, 3, 5, 7, 10, 15]:
    E_acc = (1 - betas**(k+1)) / (1 - betas)
    speedup = E_acc / (c * k + 1)
    ax1.plot(betas, speedup, lw=2, label=f'k={k}')

ax1.axhline(y=1.0, color='gray', ls='--', lw=1.5, alpha=0.5)
ax1.set_xlabel('Acceptance Rate (β)', fontsize=11)
ax1.set_ylabel('Speedup', fontsize=11)
ax1.set_title('Speedup vs Acceptance Rate (c=0.05)', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 10)

# 최적 k 찾기
ax2 = axes[1]
k_range = np.arange(1, 21)
for beta in [0.5, 0.7, 0.8, 0.9, 0.95]:
    speedups = []
    for k in k_range:
        E_acc = (1 - beta**(k+1)) / (1 - beta)
        sp = E_acc / (c * k + 1)
        speedups.append(sp)
    optimal_k = k_range[np.argmax(speedups)]
    ax2.plot(k_range, speedups, '-o', lw=2, ms=4, label=f'β={beta} (optimal k={optimal_k})')

ax2.axhline(y=1.0, color='gray', ls='--', lw=1.5, alpha=0.5)
ax2.set_xlabel('Draft Length (k)', fontsize=11)
ax2.set_ylabel('Speedup', fontsize=11)
ax2.set_title('Speedup vs Draft Length (c=0.05)', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter14_extreme_inference/speculative_speedup.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter14_extreme_inference/speculative_speedup.png")

# 최적 k 요약표
print(f"\\n{'β':>6} | {'최적 k':>6} | {'최대 Speedup':>14}")
print(f"{'-'*32}")
for beta in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    best_sp = 0
    best_k = 1
    for k in range(1, 30):
        E_acc = (1 - beta**(k+1)) / (1 - beta)
        sp = E_acc / (c * k + 1)
        if sp > best_sp:
            best_sp = sp
            best_k = k
    print(f"{beta:>6.2f} | {best_k:>6} | {best_sp:>14.2f}x")"""))

# ── Cell 10: Section 4 header ────────────────────────────────────
cells.append(md("""\
## 4. Medusa vs EAGLE 아키텍처 비교 <a name='4.-Medusa-vs-EAGLE'></a>

기존 Speculative Decoding은 별도 Draft 모델이 필요하지만, **Medusa**와 **EAGLE**은 
Target 모델 자체에 추가 헤드를 부착하여 Draft 단계를 수행합니다.

| 방식 | 구조 | Draft 생성 |
|------|------|-----------|
| 전통 방식 | 별도 Draft 모델 | 자기회귀 |
| Medusa | Target + 다중 MLP 헤드 | 병렬 (비자기회귀) |
| EAGLE | Target + Feature 기반 헤드 | 자기회귀 (feature-level) |"""))

# ── Cell 11: Medusa vs EAGLE comparison ──────────────────────────
cells.append(code("""\
# ── Medusa vs EAGLE 비교표 ───────────────────────────────────────
print("=" * 95)
print("  Speculative Decoding 방식 비교: 전통 vs Medusa vs EAGLE")
print("=" * 95)

headers = ['항목', '전통 Spec. Decoding', 'Medusa (2024)', 'EAGLE (2024)']
rows = [
    ['논문', 'Leviathan et al. 2023', 'Cai et al. 2024', 'Li et al. 2024'],
    ['Draft 소스', '별도 소형 모델', 'Target + MLP 헤드', 'Target + Autoregressive 헤드'],
    ['Draft 방식', '자기회귀 (순차)', '비자기회귀 (병렬)', '자기회귀 (feature-level)'],
    ['토큰 트리', '선형 체인', 'Tree Attention', 'Tree Attention'],
    ['추가 파라미터', '별도 모델 전체', '~0.6% (MLP heads)', '~0.2-2% (feature head)'],
    ['학습 필요', 'Draft 모델 사전학습', 'MLP 헤드 학습', 'Feature 헤드 학습'],
    ['학습 데이터', '일반 코퍼스', 'Target 출력', 'Target hidden states'],
    ['분포 보존', '보장 (rejection)', '근사적', '보장 (rejection)'],
    ['속도 향상', '~2-3x', '~2-3x', '~2.5-4x'],
    ['메모리 추가', '큼 (Draft 모델)', '매우 작음', '작음'],
    ['호환성', '모델 쌍 필요', 'Target만 수정', 'Target만 수정'],
]

col_widths = [18, 22, 22, 25]
header_line = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
print(header_line)
print('-' * len(header_line))
for row in rows:
    line = ' | '.join(val.ljust(w) for val, w in zip(row, col_widths))
    print(line)

# 성능 비교 시각화
methods = ['Autoregressive\\n(Baseline)', 'Spec. Decoding\\n(70B+7B)', 'Medusa\\n(70B+heads)', 'EAGLE\\n(70B+head)']
speedups = [1.0, 2.2, 2.5, 3.5]
extra_params = [0, 100, 0.6, 1.5]  # % of base model

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = ['#E0E0E0', '#90CAF9', '#A5D6A7', '#FFB74D']

ax1 = axes[0]
bars = ax1.bar(methods, speedups, color=colors, edgecolor='black', lw=0.5)
for bar, val in zip(bars, speedups):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
             f'{val:.1f}x', ha='center', fontsize=11, fontweight='bold')
ax1.axhline(y=1.0, color='red', ls='--', lw=1.5, alpha=0.5)
ax1.set_ylabel('Speedup', fontsize=11)
ax1.set_title('Speculative Decoding 방식별 속도 향상', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
bars2 = ax2.bar(methods, extra_params, color=colors, edgecolor='black', lw=0.5)
for bar, val in zip(bars2, extra_params):
    label = f'{val:.1f}%' if val < 10 else f'{val:.0f}%'
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             label, ha='center', fontsize=11, fontweight='bold')
ax2.set_ylabel('Extra Parameters (% of Target)', fontsize=11)
ax2.set_title('추가 파라미터 오버헤드', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chapter14_extreme_inference/speculative_methods_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("\\n그래프 저장됨: chapter14_extreme_inference/speculative_methods_comparison.png")

print(f"\\n핵심 인사이트:")
print(f"  1. EAGLE: Feature-level 자기회귀 → 높은 수용률과 분포 보존")
print(f"  2. Medusa: 비자기회귀 병렬 → 빠른 Draft, 하지만 근사적 분포")
print(f"  3. 전통 방식: 별도 모델 필요 → 메모리 부담이 크지만 범용적")"""))

# ── Cell 12: Section 5 header ────────────────────────────────────
cells.append(md("""\
## 5. 토큰 수용 시각화 <a name='5.-토큰-수용-시각화'></a>

실제 Speculative Decoding에서 토큰이 어떻게 수용/거부되는지를 시각화합니다."""))

# ── Cell 13: Token acceptance visualization ──────────────────────
cells.append(code("""\
# ── 토큰 수용/거부 패턴 시각화 ───────────────────────────────────
np.random.seed(42)
vocab_size = 50
k = 7  # 드래프트 길이
n_rounds = 20

target_logits = np.random.randn(vocab_size) * 2
target_probs = np.exp(target_logits) / np.sum(np.exp(target_logits))

draft_logits = target_logits + np.random.randn(vocab_size) * 0.5
draft_probs = np.exp(draft_logits) / np.sum(np.exp(draft_logits))

acceptance_map = np.zeros((n_rounds, k + 1))
accepted_counts = []

for r in range(n_rounds):
    draft_tokens = [np.random.choice(vocab_size, p=draft_probs) for _ in range(k)]
    n_accepted = 0

    for i, token in enumerate(draft_tokens):
        p = target_probs[token]
        q = draft_probs[token]
        alpha = min(1.0, p / q)

        if np.random.random() < alpha:
            acceptance_map[r, i] = 1  # 수용
            n_accepted += 1
        else:
            acceptance_map[r, i] = -1  # 거부
            break
        if i < k - 1:
            for j in range(i + 2, k + 1):
                acceptance_map[r, j] = 0  # 미도달

    # 보너스 토큰 (항상 1개)
    if n_accepted == k:
        acceptance_map[r, k] = 2  # 보너스

    accepted_counts.append(n_accepted + 1)  # +1 보너스/수정 토큰

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# 히트맵
ax1 = axes[0]
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#BDBDBD', '#EF5350', '#66BB6A', '#42A5F5'])
# -1=거부(빨), 0=미도달(회), 1=수용(초), 2=보너스(파)
mapped = np.zeros_like(acceptance_map)
mapped[acceptance_map == 0] = 0
mapped[acceptance_map == -1] = 1
mapped[acceptance_map == 1] = 2
mapped[acceptance_map == 2] = 3

im = ax1.imshow(mapped, cmap=cmap, aspect='auto', interpolation='nearest')
ax1.set_xlabel('Token Position', fontsize=11)
ax1.set_ylabel('Round', fontsize=11)
ax1.set_title(f'Token Acceptance Pattern (k={k})', fontweight='bold')
ax1.set_xticks(range(k + 1))
ax1.set_xticklabels([f'Draft {i+1}' for i in range(k)] + ['Bonus'])

legend_labels = ['Not Reached', 'Rejected', 'Accepted', 'Bonus']
legend_colors = ['#BDBDBD', '#EF5350', '#66BB6A', '#42A5F5']
patches = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in legend_colors]
ax1.legend(patches, legend_labels, loc='lower right', fontsize=8)

# 수용 토큰 수 분포
ax2 = axes[1]
bins = range(1, k + 3)
ax2.hist(accepted_counts, bins=bins, color='#42A5F5', edgecolor='black',
         alpha=0.8, align='left', rwidth=0.8)
ax2.axvline(x=np.mean(accepted_counts), color='red', ls='--', lw=2,
            label=f'Mean = {np.mean(accepted_counts):.1f}')
ax2.set_xlabel('Accepted Tokens per Round', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Distribution of Accepted Tokens', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter14_extreme_inference/token_acceptance_pattern.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter14_extreme_inference/token_acceptance_pattern.png")

avg_beta = np.mean([c - 1 for c in accepted_counts]) / k
theoretical_E = (1 - avg_beta**(k+1)) / (1 - avg_beta)
print(f"\\n실험 결과:")
print(f"  평균 수용 토큰: {np.mean(accepted_counts):.2f}")
print(f"  추정 수용률 β: {avg_beta:.3f}")
print(f"  이론적 E[accepted]: {theoretical_E:.2f}")
print(f"  Draft 길이: {k}")
print(f"  예상 Speedup (c=0.05): {np.mean(accepted_counts) / (0.05 * k + 1):.2f}x")"""))

# ── Cell 14: Summary ─────────────────────────────────────────────
cells.append(md(r"""\
## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| Draft-Verify | 소형 모델 추측 + 대형 모델 병렬 검증 | ⭐⭐⭐ |
| 수용률 $\beta$ | $\min(1, p(x)/q(x))$ — Draft 품질 지표 | ⭐⭐⭐ |
| 기대 토큰 수 | $(1-\beta^{k+1})/(1-\beta)$ — 등비급수 | ⭐⭐⭐ |
| 분포 보존 | 수정 분포 $\max(0, p-q)$로 정확성 보장 | ⭐⭐⭐ |
| Medusa | 비자기회귀 다중 MLP 헤드 | ⭐⭐ |
| EAGLE | Feature-level 자기회귀 헤드, 최고 성능 | ⭐⭐⭐ |
| 최적 $k$ | $\beta$와 $c$에 따라 최적점 존재 | ⭐⭐ |

### 핵심 수식

$$E[\text{accepted}] = \frac{1 - \beta^{k+1}}{1 - \beta}, \quad \text{Speedup} = \frac{E[\text{acc}]}{c \cdot k + 1}$$

$$\alpha(x) = \min\left(1, \frac{p(x)}{q(x)}\right), \quad p'(x) = \text{norm}\left(\max(0, p(x) - q(x))\right)$$

### 다음 챕터 예고
**04_vllm_and_paged_attention.ipynb** — OS의 Page Table 개념을 KV Cache에 적용한 PagedAttention, Continuous Batching, 동적 KV Block 스케줄링을 학습합니다."""))

create_notebook(cells, 'chapter14_extreme_inference/03_speculative_decoding.ipynb')
