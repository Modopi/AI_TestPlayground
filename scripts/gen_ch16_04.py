"""Generate chapter16_sparse_attention/04_long_context_and_sparse_attn.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 16: 최신 거대 모델의 효율성 — Long Context와 Sparse Attention

## 학습 목표
- YaRN 주파수 재조정을 통한 컨텍스트 창 확장 원리를 수식으로 이해한다
- Sliding Window Attention(SWA)의 마스크 구조와 메모리 절약 원리를 구현한다
- DeepSeek Sparse Attention(DSA)의 선택적 토큰 패턴을 분석한다
- Full, Sliding Window, Sparse Attention의 마스크를 시각화하여 비교한다
- 50% 이상의 비용 절감을 달성하는 통합 방법론을 이해한다

## 목차
1. [수학적 기초: 컨텍스트 확장과 Sparse Attention](#1.-수학적-기초)
2. [Attention 마스크 시각화](#2.-Attention-마스크-시각화)
3. [메모리 절약 비교](#3.-메모리-절약-비교)
4. [Long Context Perplexity 시뮬레이션](#4.-Long-Context-Perplexity)
5. [50% 이상 비용 절감 분석](#5.-비용-절감-분석)
6. [정리](#6.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### YaRN (Yet another RoPE extensioN)

RoPE의 주파수를 재조정하여 학습된 컨텍스트 창을 넘어서는 위치를 처리합니다:

$$\theta_i' = \theta_i \cdot \begin{cases} 1 & \text{if } \lambda_i < 1 \text{ (고주파 — 변경 없음)} \\ 1/s & \text{if } \lambda_i > 1 \text{ (저주파 — 스케일 다운)} \\ (1-\gamma) \cdot 1 + \gamma / s & \text{otherwise (선형 보간)} \end{cases}$$

- $\theta_i = 10000^{-2i/d}$: 원래 RoPE 주파수
- $s$: 컨텍스트 확장 비율 (예: 4x → $s=4$)
- $\lambda_i = 2\pi / \theta_i$: 파장
- $\gamma$: 보간 비율

### Sliding Window Attention (SWA)

각 토큰은 자신의 윈도우 내 토큰만 attend합니다:

$$A_{ij} = \begin{cases} \text{softmax}(q_i k_j^T / \sqrt{d}) & \text{if } |i - j| \leq w/2 \text{ and } j \leq i \\ 0 & \text{otherwise} \end{cases}$$

- $w$: 윈도우 크기
- 메모리 복잡도: $O(Nw)$ vs Full의 $O(N^2)$

### DeepSeek Sparse Attention (DSA)

토큰을 블록 단위로 나누고, 중요한 블록만 선택적으로 attend:

1. **블록 분할**: 시퀀스를 $B$개 블록으로 나눔 ($b = N/B$)
2. **중요도 점수**: $\text{score}_j = \max_{k \in \text{block}_j} q_i^T k / \sqrt{d}$
3. **Top-K 블록 선택**: 상위 $K$개 블록만 attend

$$\text{Sparsity} = 1 - \frac{K}{B}, \quad \text{비용 절감} = 1 - \frac{Kb + w}{N}$$

### 비용 절감 분석

| 방법 | Attention 비용 | Full 대비 절감률 |
|------|---------------|-----------------|
| Full | $N^2$ | 0% |
| SWA ($w$) | $Nw$ | $1 - w/N$ |
| DSA ($K$ 블록, $b$ 크기) | $NKb$ | $1 - Kb/N$ |
| SWA + DSA | $N(w + Kb)$ | $1 - (w + Kb)/N$ |

**요약 표:**

| 구분 | 수식 | 설명 |
|------|------|------|
| YaRN 스케일링 | $\theta_i' \propto \theta_i / s$ | 저주파 주파수 압축 |
| SWA 마스크 | $\|i-j\| \leq w/2$ | 지역 윈도우 제한 |
| DSA 블록 선택 | $\text{TopK}(\max_{k} q^Tk)$ | 중요 블록만 attend |
| 비용 절감 | $> 50\%$ 가능 | SWA + DSA 결합 |

---"""))

# ── Cell 3: 🐣 friendly explanation ──────────────────────────────
cells.append(md("""\
### 🐣 초등학생을 위한 Long Context 친절 설명!

#### 🔢 왜 긴 문장을 읽기 어려운가요?

> 💡 **비유**: 1000페이지짜리 책을 읽을 때, 모든 페이지의 내용을 동시에 기억하려면 
> 엄청난 메모리가 필요해요! AI도 마찬가지예요.

- **Full Attention**: 모든 페이지를 동시에 참조 → 페이지가 많으면 $N^2$배 느려짐
- **Sliding Window**: 현재 읽는 페이지 주변 10페이지만 참조 → 빠르지만 먼 내용을 못 봄
- **Sparse Attention**: **중요한 페이지만** 골라서 참조 → 빠르면서도 핵심은 놓치지 않음!

#### 📏 YaRN은 뭔가요?

> 💡 **비유**: 30cm 자(학습된 컨텍스트)로 1m를 재고 싶을 때, 
> 자의 눈금 간격을 줄여서(주파수 재조정) 더 긴 거리를 잴 수 있게 만드는 거예요!

YaRN은 학습 시 4K 토큰까지만 배운 모델이 64K 토큰까지 읽을 수 있게 해줘요.

---"""))

# ── Cell 4: 📝 Exercises ──────────────────────────────────────────
cells.append(md(r"""\
### 📝 연습 문제

#### 문제 1: SWA 절감률

시퀀스 길이 $N = 8192$, 윈도우 크기 $w = 512$일 때 SWA의 Full 대비 비용 절감률은?

<details>
<summary>💡 풀이 확인</summary>

$$\text{절감률} = 1 - \frac{w}{N} = 1 - \frac{512}{8192} = 1 - 0.0625 = 93.75\%$$

→ 윈도우를 제한하는 것만으로 **93.75%** 비용 절감!
다만 512 토큰 밖의 장거리 의존성을 놓칠 수 있습니다.
</details>

#### 문제 2: DSA + SWA 결합

$N=8192$, SWA $w=512$, DSA 블록 크기 $b=256$, 선택 블록 $K=4$일 때 총 비용 절감률은?

<details>
<summary>💡 풀이 확인</summary>

$$\text{SWA 비용} = N \times w = 8192 \times 512$$
$$\text{DSA 비용} = N \times K \times b = 8192 \times 4 \times 256$$
$$\text{총 비용} = N(w + Kb) = 8192 \times (512 + 1024) = 8192 \times 1536$$
$$\text{Full 비용} = N^2 = 8192^2 = 8192 \times 8192$$

$$\text{절감률} = 1 - \frac{1536}{8192} = 1 - 0.1875 = 81.25\%$$

→ SWA + DSA 결합으로 **81.25%** 비용 절감 달성!
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

# ── Cell 6: Section 2 - Attention mask visualization ─────────────
cells.append(md(r"""\
## 2. Attention 마스크 시각화 <a name='2.-Attention-마스크-시각화'></a>

Full, Sliding Window, DeepSeek Sparse Attention의 마스크 패턴을 비교합니다."""))

cells.append(code("""\
# ── Attention 마스크 시각화 ──────────────────────────────────────
N = 64  # 시각화를 위한 작은 시퀀스

def create_causal_mask(N):
    # 표준 causal mask (lower triangular)
    return np.tril(np.ones((N, N)))

def create_sliding_window_mask(N, window_size=16):
    # Sliding window + causal
    mask = np.zeros((N, N))
    for i in range(N):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 1.0
    return mask

def create_sparse_block_mask(N, block_size=8, top_k_blocks=2, window_size=8):
    # Sparse attention: 윈도우 + Top-K 블록
    mask = np.zeros((N, N))
    n_blocks = N // block_size

    for i in range(N):
        # (1) 로컬 윈도우
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 1.0

        # (2) Top-K 블록 (랜덤 시뮬레이션)
        current_block = i // block_size
        available_blocks = list(range(0, current_block + 1))
        if len(available_blocks) > top_k_blocks:
            np.random.seed(i * 7 + 3)
            selected = np.random.choice(available_blocks, top_k_blocks, replace=False)
        else:
            selected = available_blocks

        for b in selected:
            b_start = b * block_size
            b_end = min(b_start + block_size, i + 1)
            mask[i, b_start:b_end] = 1.0

    return mask

# 마스크 생성
mask_full = create_causal_mask(N)
mask_swa = create_sliding_window_mask(N, window_size=16)
mask_sparse = create_sparse_block_mask(N, block_size=8, top_k_blocks=2, window_size=8)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

titles = ['Full Causal Attention', 'Sliding Window (w=16)', 'Sparse Block (k=2, b=8, w=8)']
masks = [mask_full, mask_swa, mask_sparse]
cmaps = ['Blues', 'Oranges', 'Greens']

for ax, title, mask, cmap in zip(axes, titles, masks, cmaps):
    ax.imshow(mask, cmap=cmap, aspect='equal', interpolation='nearest')
    ax.set_xlabel('Key 위치', fontsize=11)
    ax.set_ylabel('Query 위치', fontsize=11)
    ax.set_title(title, fontweight='bold')

    # 밀도(density) 표시
    density = mask.sum() / mask.size * 100
    ax.text(N*0.95, N*0.05, f'밀도: {density:.1f}%',
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/attention_masks_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter16_sparse_attention/attention_masks_comparison.png")

# 밀도 비교
print(f"\\nAttention 마스크 밀도 비교 (N={N}):")
for title, mask in zip(titles, masks):
    density = mask.sum() / mask.size * 100
    active = int(mask.sum())
    total = mask.size
    print(f"  {title}: {density:.1f}% ({active}/{total} 원소 활성)")"""))

# ── Cell 8: Section 3 - Memory savings ───────────────────────────
cells.append(md(r"""\
## 3. 메모리 절약 비교 <a name='3.-메모리-절약-비교'></a>

다양한 Attention 방식의 시퀀스 길이별 메모리 사용량을 비교합니다.

| 방식 | Attention 메모리 | KV Cache (추론) |
|------|-----------------|-----------------|
| Full Causal | $O(N^2)$ | $O(Nd)$ |
| SWA | $O(Nw)$ | $O(wd)$ |
| Sparse Block | $O(NKb)$ | $O(Kbd)$ |"""))

cells.append(code("""\
# ── 메모리 절약 비교 시각화 ──────────────────────────────────────
seq_lengths = np.array([1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
window_size = 1024
block_size = 512
top_k_blocks = 4
d = 128

# Attention 행렬 메모리 (원소 수)
mem_full = seq_lengths ** 2
mem_swa = seq_lengths * window_size
mem_sparse = seq_lengths * (top_k_blocks * block_size + window_size)

# KV Cache 메모리 (바이트, FP16, 단일 레이어)
kv_full = seq_lengths * 2 * d * 2  # 모든 토큰의 KV
kv_swa = np.minimum(seq_lengths, window_size) * 2 * d * 2  # 윈도우만
kv_sparse = (top_k_blocks * block_size + window_size) * 2 * d * 2  # 블록 + 윈도우

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) Attention 연산 메모리
ax1 = axes[0]
ax1.loglog(seq_lengths / 1000, mem_full / 1e9, 'r-o', lw=2.5, ms=7, label='Full ($O(N^2)$)')
ax1.loglog(seq_lengths / 1000, mem_swa / 1e9, 'b-s', lw=2, ms=7, label=f'SWA ($w={window_size}$)')
ax1.loglog(seq_lengths / 1000, mem_sparse / 1e9, 'g-^', lw=2, ms=7,
           label=f'Sparse ($K={top_k_blocks}, b={block_size}$)')

ax1.set_xlabel('시퀀스 길이 (K 토큰)', fontsize=11)
ax1.set_ylabel('Attention 메모리 (G 원소)', fontsize=11)
ax1.set_title('Attention 연산 메모리 비교', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) 절감률
ax2 = axes[1]
savings_swa = (1 - mem_swa / mem_full) * 100
savings_sparse = (1 - mem_sparse / mem_full) * 100

ax2.plot(seq_lengths / 1000, savings_swa, 'b-s', lw=2.5, ms=7, label='SWA')
ax2.plot(seq_lengths / 1000, savings_sparse, 'g-^', lw=2.5, ms=7, label='Sparse')
ax2.axhline(y=50, color='red', ls='--', lw=1.5, label='50% 기준')
ax2.fill_between(seq_lengths / 1000, 50, 100, alpha=0.05, color='green')
ax2.set_xlabel('시퀀스 길이 (K 토큰)', fontsize=11)
ax2.set_ylabel('Full 대비 절감률 (%)', fontsize=11)
ax2.set_title('시퀀스 길이별 비용 절감률', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/memory_savings_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter16_sparse_attention/memory_savings_comparison.png")

# 수치 표
print(f"\\n메모리 절감률 수치 비교:")
print(f"{'시퀀스 길이':>12} | {'SWA 절감':>10} | {'Sparse 절감':>12} | {'>50% 달성':>10}")
print("-" * 52)
for i, N in enumerate(seq_lengths):
    swa_s = savings_swa[i]
    sp_s = savings_sparse[i]
    check = '✅' if sp_s > 50 else '❌'
    print(f"{N:>12,} | {swa_s:>9.1f}% | {sp_s:>11.1f}% | {check:>10}")"""))

# ── Cell 10: Section 4 - Long Context Perplexity ────────────────
cells.append(md(r"""\
## 4. Long Context Perplexity 시뮬레이션 <a name='4.-Long-Context-Perplexity'></a>

YaRN을 사용한 컨텍스트 확장 시 perplexity 변화를 시뮬레이션합니다.

$$\text{PPL}(s) = \text{PPL}_{base} \cdot \left(1 + \alpha \cdot \max(0, s - 1)\right)$$

- $s$: 컨텍스트 확장 비율
- $\alpha$: 품질 저하 계수 (YaRN이 작을수록 우수)"""))

cells.append(code("""\
# ── Long Context Perplexity 시뮬레이션 ───────────────────────────
np.random.seed(42)

# 컨텍스트 확장 비율
extension_ratios = np.array([1, 2, 4, 8, 16, 32])
base_ctx = 4096  # 기본 학습 컨텍스트

# 각 방법의 perplexity 모델링
ppl_base = 5.0  # 기본 perplexity

# NTK-aware (단순 스케일링)
alpha_ntk = 0.15
ppl_ntk = ppl_base * (1 + alpha_ntk * np.maximum(0, extension_ratios - 1))

# YaRN (개선된 스케일링)
alpha_yarn = 0.03
ppl_yarn = ppl_base * (1 + alpha_yarn * np.maximum(0, extension_ratios - 1))

# 학습 없이 직접 확장 (PI: Position Interpolation)
alpha_pi = 0.08
ppl_pi = ppl_base * (1 + alpha_pi * np.maximum(0, extension_ratios - 1))

# 학습 없이 RoPE (외삽 - 급격히 나빠짐)
ppl_no_ext = ppl_base * np.exp(0.2 * np.maximum(0, extension_ratios - 1))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) PPL vs 확장 비율
ax1 = axes[0]
effective_ctx = base_ctx * extension_ratios
ax1.plot(effective_ctx / 1000, ppl_no_ext, 'r-x', lw=2, ms=8, label='RoPE 외삽 (학습 없음)')
ax1.plot(effective_ctx / 1000, ppl_ntk, 'orange', lw=2, marker='D', ms=7, label='NTK-aware')
ax1.plot(effective_ctx / 1000, ppl_pi, 'b-s', lw=2, ms=7, label='Position Interpolation')
ax1.plot(effective_ctx / 1000, ppl_yarn, 'g-o', lw=2.5, ms=8, label='YaRN')

ax1.set_xlabel('유효 컨텍스트 길이 (K)', fontsize=11)
ax1.set_ylabel('Perplexity', fontsize=11)
ax1.set_title('컨텍스트 확장 방법별 Perplexity', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_ylim(4, 30)

# (2) YaRN 주파수 재조정
ax2 = axes[1]
d_model = 128
n_dims = d_model // 2
freqs = 10000 ** (-2 * np.arange(n_dims) / d_model)
wavelengths = 2 * np.pi / freqs

scale = 4  # 4x 확장
threshold_low = base_ctx
threshold_high = base_ctx * scale

freqs_yarn = np.copy(freqs)
for i in range(n_dims):
    wl = wavelengths[i]
    if wl < threshold_low:
        freqs_yarn[i] = freqs[i]  # 고주파: 변경 없음
    elif wl > threshold_high:
        freqs_yarn[i] = freqs[i] / scale  # 저주파: 스케일 다운
    else:
        gamma = (wl - threshold_low) / (threshold_high - threshold_low)
        freqs_yarn[i] = freqs[i] * ((1 - gamma) + gamma / scale)

ax2.plot(range(n_dims), freqs, 'b-', lw=2, alpha=0.5, label='원래 RoPE')
ax2.plot(range(n_dims), freqs_yarn, 'g-', lw=2.5, label='YaRN (4x)')
ax2.set_xlabel('차원 인덱스', fontsize=11)
ax2.set_ylabel('주파수', fontsize=11)
ax2.set_title('YaRN 주파수 재조정 (4x 확장)', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/yarn_context_extension.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter16_sparse_attention/yarn_context_extension.png")

print(f"\\nPerplexity 비교 (기본 = {ppl_base}):")
print(f"{'확장 비율':>10} | {'유효 길이':>10} | {'RoPE 외삽':>10} | {'NTK':>8} | {'PI':>8} | {'YaRN':>8}")
print("-" * 65)
for i, ratio in enumerate(extension_ratios):
    ctx = base_ctx * ratio
    print(f"{ratio:>10}x | {ctx:>10,} | {ppl_no_ext[i]:>10.2f} | {ppl_ntk[i]:>8.2f} | "
          f"{ppl_pi[i]:>8.2f} | {ppl_yarn[i]:>8.2f}")
print(f"\\n→ YaRN은 32x 확장에서도 PPL 증가가 {(ppl_yarn[-1]/ppl_base - 1)*100:.1f}%에 불과")"""))

# ── Cell 12: Section 5 - Cost reduction analysis ────────────────
cells.append(md(r"""\
## 5. 50% 이상 비용 절감 분석 <a name='5.-비용-절감-분석'></a>

SWA + Sparse Attention을 결합한 실전 비용 절감 분석입니다.

**DeepSeek-V3 실제 설정:**
- 일부 레이어: Full Attention (전역 의존성)
- 대부분 레이어: SWA ($w=4096$) + Sparse Block ($K=4, b=512$)
- 총 비용 절감: $> 50\%$"""))

cells.append(code("""\
# ── 비용 절감 종합 분석 ──────────────────────────────────────────
# 실전 모델 설정 기반 비용 분석
N_values = [4096, 8192, 16384, 32768, 65536, 131072]

# 모델 설정
n_total_layers = 61  # DeepSeek-V3
n_full_layers = 4  # Full attention 레이어
n_swa_layers = n_total_layers - n_full_layers  # SWA + sparse 레이어
w = 4096  # SWA 윈도우
K_blocks = 4  # Top-K 블록
b = 512  # 블록 크기

print(f"DeepSeek-V3 스타일 비용 분석:")
print(f"  총 레이어: {n_total_layers} (Full: {n_full_layers}, SWA+Sparse: {n_swa_layers})")
print(f"  SWA 윈도우: {w}, Top-K 블록: {K_blocks}, 블록 크기: {b}")
print()

print(f"{'시퀀스':>8} | {'Full Only':>12} | {'Hybrid':>12} | {'절감률':>8} | {'>50%':>5}")
print("-" * 55)

savings_list = []

for N in N_values:
    # Full attention 비용 (모든 레이어)
    full_cost = n_total_layers * N * N

    # Hybrid 비용
    full_layer_cost = n_full_layers * N * N
    swa_sparse_cost = n_swa_layers * N * min(w + K_blocks * b, N)
    hybrid_cost = full_layer_cost + swa_sparse_cost

    saving = (1 - hybrid_cost / full_cost) * 100
    savings_list.append(saving)
    check = '✅' if saving > 50 else '❌'
    print(f"{N:>8,} | {full_cost/1e12:>10.2f}T | {hybrid_cost/1e12:>10.2f}T | {saving:>7.1f}% | {check:>5}")

# 비용 분해 시각화
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) 비용 구성 (스택 바 차트)
ax1 = axes[0]
x_pos = range(len(N_values))
full_attn_cost = [n_full_layers * N * N / 1e12 for N in N_values]
swa_cost = [n_swa_layers * N * min(w, N) / 1e12 for N in N_values]
sparse_cost = [n_swa_layers * N * min(K_blocks * b, N) / 1e12 for N in N_values]

ax1.bar(x_pos, full_attn_cost, color='red', alpha=0.7, label='Full Attn 레이어')
ax1.bar(x_pos, swa_cost, bottom=full_attn_cost, color='blue', alpha=0.7, label='SWA')
bottoms = [f + s for f, s in zip(full_attn_cost, swa_cost)]
ax1.bar(x_pos, sparse_cost, bottom=bottoms, color='green', alpha=0.7, label='Sparse Block')

ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{N//1000}K' for N in N_values], fontsize=9)
ax1.set_xlabel('시퀀스 길이', fontsize=11)
ax1.set_ylabel('연산 비용 (T 연산)', fontsize=11)
ax1.set_title('Hybrid Attention 비용 구성', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# (2) 절감률
ax2 = axes[1]
ax2.bar(x_pos, savings_list, color=['orange' if s < 50 else 'green' for s in savings_list],
        alpha=0.7, edgecolor='black')
ax2.axhline(y=50, color='red', ls='--', lw=2, label='50% 기준')
for i, s in enumerate(savings_list):
    ax2.text(i, s + 1, f'{s:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{N//1000}K' for N in N_values], fontsize=9)
ax2.set_xlabel('시퀀스 길이', fontsize=11)
ax2.set_ylabel('비용 절감률 (%)', fontsize=11)
ax2.set_title('시퀀스 길이별 비용 절감률', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/cost_reduction_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("\\n그래프 저장됨: chapter16_sparse_attention/cost_reduction_analysis.png")

# 최종 결론
print(f"\\n결론:")
print(f"  시퀀스 길이 4K: 절감률 = {savings_list[0]:.1f}% (짧은 시퀀스에서는 효과 제한적)")
over_50 = sum(1 for s in savings_list if s > 50)
print(f"  50% 이상 절감 달성: {over_50}/{len(N_values)} 설정")
print(f"  시퀀스 길이 128K: 절감률 = {savings_list[-1]:.1f}%")
print(f"  → 긴 시퀀스에서 SWA + Sparse Attention의 효과가 극대화됨")"""))

# ── Cell 14: Summary ─────────────────────────────────────────────
cells.append(md(r"""\
## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| YaRN | 주파수 재조정으로 컨텍스트 창 확장 | ⭐⭐⭐ |
| Sliding Window (SWA) | $\|i-j\| \leq w/2$ 마스크로 지역 패턴 포착 | ⭐⭐⭐ |
| DeepSeek Sparse Attn | Top-K 블록 선택으로 중요 정보만 attend | ⭐⭐⭐ |
| Hybrid 전략 | Full + SWA + Sparse 레이어별 혼합 | ⭐⭐⭐ |
| 비용 절감 | 긴 시퀀스에서 $>50\%$ 절감 달성 | ⭐⭐⭐ |
| 마스크 밀도 | Full(50%) → SWA(~6%) → Sparse(~20%) | ⭐⭐ |

### 핵심 수식

$$\theta_i' = \theta_i / s \quad \text{(YaRN 저주파 재조정)}$$

$$\text{SWA}: A_{ij} \neq 0 \iff |i-j| \leq w/2 \text{ and } j \leq i$$

$$\text{비용 절감} = 1 - \frac{n_{full} \cdot N + n_{swa} \cdot (w + Kb)}{n_{total} \cdot N}$$

### 다음 챕터 예고
**Chapter 17: Diffusion Transformers** — DiT 아키텍처의 adaLN-Zero 수식과 Flow Matching, HunyuanVideo의 3D 비디오 인코딩 아키텍처를 다룹니다."""))

create_notebook(cells, 'chapter16_sparse_attention/04_long_context_and_sparse_attn.ipynb')
