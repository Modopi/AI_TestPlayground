"""Generate chapter14_extreme_inference/02_flash_attention_deepdive.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 14: 극단적 추론 최적화 — FlashAttention 심층 분석

## 학습 목표
- 표준 Attention의 **IO 복잡도**를 분석하고 HBM 접근이 병목임을 수식으로 증명한다
- FlashAttention의 **Tiling + Online Softmax + Recomputation** 전략의 수학적 원리를 이해한다
- SRAM 크기 $M$에 따른 **IO 복잡도 $O(N^2d^2/M)$**을 유도한다
- FlashAttention v1 → v2 → v3의 **성능 개선 포인트**를 비교 분석한다
- 블록 단위 Attention 계산을 **TensorFlow로 직접 시뮬레이션**한다

## 목차
1. [수학적 기초: Attention IO 복잡도](#1.-수학적-기초)
2. [표준 Attention의 메모리 병목](#2.-표준-Attention)
3. [FlashAttention Tiling 시뮬레이션](#3.-Tiling-시뮬레이션)
4. [IO 복잡도 비교 시각화](#4.-IO-복잡도-시각화)
5. [v1 → v2 → v3 발전사](#5.-FlashAttention-발전사)
6. [정리](#6.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### 표준 Attention의 IO 복잡도

Attention 연산: $\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$

여기서 $Q, K, V \in \mathbb{R}^{N \times d}$이고, $N$은 시퀀스 길이, $d$는 헤드 차원입니다.

**표준 구현의 HBM 접근:**

| 단계 | 읽기 | 쓰기 | HBM 접근 |
|------|------|------|----------|
| $S = QK^T/\sqrt{d}$ | $Q, K$ ($2Nd$) | $S$ ($N^2$) | $O(Nd + N^2)$ |
| $P = \text{softmax}(S)$ | $S$ ($N^2$) | $P$ ($N^2$) | $O(N^2)$ |
| $O = PV$ | $P$ ($N^2$), $V$ ($Nd$) | $O$ ($Nd$) | $O(N^2 + Nd)$ |

$$\text{Total HBM I/O} = O(Nd + N^2) \quad \text{bytes (원소 단위)}$$

핵심 문제: **$N^2$ 크기의 중간 행렬 $S, P$가 HBM에 저장**됨!

### FlashAttention의 IO 복잡도

Tiling을 적용하면 $S, P$를 **SRAM에서만 처리**하고 HBM에 쓰지 않습니다:

$$\text{FlashAttention HBM I/O} = O\left(\frac{N^2 d^2}{M}\right)$$

- $M$: SRAM 크기 (A100: 192KB per SM, 전체 ~20MB)
- 조건: $d^2 \leq M$ (헤드 차원의 제곱이 SRAM에 들어가야 함)

### Tiling 전략

블록 크기: $B_r = \lceil M / (4d) \rceil$, $B_c = \min\left(\lceil M / (4d) \rceil, d\right)$

내부 루프에서 블록 $(i, j)$의 연산:

$$S_{ij} = Q_i K_j^T \in \mathbb{R}^{B_r \times B_c}$$

Online Softmax (Milakov & Gimelshein 2018):

$$m_i^{(j)} = \max(m_i^{(j-1)}, \text{rowmax}(S_{ij}))$$

$$\ell_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}} \ell_i^{(j-1)} + \text{rowsum}(e^{S_{ij} - m_i^{(j)}})$$

$$O_i^{(j)} = \text{diag}(e^{m_i^{(j-1)} - m_i^{(j)}}) O_i^{(j-1)} + e^{S_{ij} - m_i^{(j)}} V_j$$

**요약 표:**

| 방법 | HBM I/O | 메모리 사용 | 정확도 |
|------|---------|------------|--------|
| 표준 Attention | $O(Nd + N^2)$ | $O(N^2)$ | Exact |
| FlashAttention | $O(N^2d^2/M)$ | $O(N)$ | Exact |
| Sparse Attention | $O(N\sqrt{N})$ | $O(N\sqrt{N})$ | Approximate |"""))

# ── Cell 3: 🐣 friendly explanation ──────────────────────────────
cells.append(md("""\
### 🐣 초등학생을 위한 FlashAttention 친절 설명!

#### 🔢 Tiling(타일링)이 뭔가요?

> 💡 **비유**: 퍼즐을 풀 때 **전체 그림을 책상에 펼치는 것** vs **작은 조각씩 맞추는 것**을 비교해 봅시다!

**표준 Attention**: 시퀀스 길이가 4096이면, 4096×4096 = 1,677만 개의 숫자를 한꺼번에 
메모장(HBM)에 써야 해요. 메모장까지 왕복하느라 시간이 오래 걸려요! 📝

**FlashAttention**: 작은 블록(예: 64×64)씩 잘라서 **머릿속 칠판(SRAM)**에서 바로 계산해요.
칠판은 작지만 아주 빨라서, 메모장에 왕복하지 않아도 돼요! 🧠

#### 🔄 Online Softmax는 뭔가요?

> 💡 **비유**: 시험 점수의 평균을 구할 때, 모든 점수를 다 모아야 할까요?

아니요! 한 명씩 점수를 받으면서 "지금까지의 최대값"과 "누적 합"을 업데이트하면 돼요.
FlashAttention도 전체 행을 보지 않고, **블록별로 softmax를 점진적으로 계산**합니다!

---"""))

# ── Cell 4: 📝 Exercises ──────────────────────────────────────────
cells.append(md(r"""\
### 📝 연습 문제

#### 문제 1: HBM 접근량 비교

$N=2048$, $d=128$, SRAM $M = 100\text{KB} = 100 \times 1024 / 2 = 51200$ 원소 (FP16)일 때:

1. 표준 Attention의 HBM I/O (원소 수)
2. FlashAttention의 HBM I/O (원소 수)
3. 절감 비율

<details>
<summary>💡 풀이 확인</summary>

**표준 Attention:**
$$\text{I/O} = Nd + N^2 + N^2 + Nd + N^2 = 3N^2 + 2Nd$$
$$= 3 \times 2048^2 + 2 \times 2048 \times 128 = 12,582,912 + 524,288 = 13,107,200$$

**FlashAttention:**
$$\text{I/O} = O\left(\frac{N^2 d^2}{M}\right) = \frac{2048^2 \times 128^2}{51200} = \frac{68,719,476,736}{51200} \approx 1,342,177$$

**절감 비율:** $13,107,200 / 1,342,177 \approx 9.8\times$ → **약 10배 절감!**
</details>

#### 문제 2: 블록 크기 계산

SRAM $M = 192\text{KB}$, $d = 128$, FP16 (2 bytes)일 때 적절한 블록 크기 $B_r$은?

<details>
<summary>💡 풀이 확인</summary>

SRAM에 $Q_i$ ($B_r \times d$), $K_j$ ($B_c \times d$), $S_{ij}$ ($B_r \times B_c$), $O_i$ ($B_r \times d$)를 저장해야 합니다.

$$M \geq (B_r \cdot d + B_c \cdot d + B_r \cdot B_c + B_r \cdot d) \times 2\;\text{bytes}$$

$B_r = B_c$로 놓으면: $M \geq (3Bd + B^2) \times 2$

$$192 \times 1024 \geq (3B \times 128 + B^2) \times 2$$
$$98304 \geq 768B + 2B^2$$

$B = 64$: $768 \times 64 + 2 \times 4096 = 49152 + 8192 = 57344 < 98304$ ✓

$B = 128$: $768 \times 128 + 2 \times 16384 = 98304 + 32768 = 131072 > 98304$ ✗

→ $B_r = B_c = 64$가 적절합니다.
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
cells.append(md("""\
## 2. 표준 Attention의 메모리 병목 <a name='2.-표준-Attention'></a>

표준 Attention과 FlashAttention의 **HBM 접근 횟수**를 시퀀스 길이별로 비교합니다."""))

# ── Cell 7: Standard vs tiled memory access comparison ───────────
cells.append(code("""\
# ── 표준 vs FlashAttention HBM 접근량 비교 ──────────────────────
d = 128
M_sram_bytes = 192 * 1024  # 192KB SRAM (A100 per SM)
M_sram_elements = M_sram_bytes // 2  # FP16

seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]

print(f"SRAM 크기: {M_sram_bytes / 1024:.0f} KB ({M_sram_elements} FP16 원소)")
print(f"헤드 차원 d: {d}")
print(f"\\n{'N':>7} | {'Standard I/O':>14} | {'Flash I/O':>14} | {'절감 배율':>10} | {'Standard Mem':>14} | {'Flash Mem':>12}")
print(f"{'-'*82}")

standard_ios = []
flash_ios = []
standard_mems = []
flash_mems = []

for N in seq_lengths:
    # 표준 Attention
    std_io = 3 * N * N + 2 * N * d  # S=QK^T, P=softmax, O=PV
    std_mem = N * N  # S/P 행렬 저장

    # FlashAttention
    flash_io = (N * N * d * d) / M_sram_elements
    flash_mem = N  # O(N) 추가 메모리

    ratio = std_io / flash_io if flash_io > 0 else float('inf')

    standard_ios.append(std_io)
    flash_ios.append(flash_io)
    standard_mems.append(std_mem * 2 / 1e6)  # MB
    flash_mems.append(flash_mem * 2 / 1e6)

    print(f"{N:>7} | {std_io:>14.2e} | {flash_io:>14.2e} | {ratio:>9.1f}x | {std_mem * 2 / 1e6:>12.1f} MB | {flash_mem * 2 / 1e6:>10.4f} MB")

print(f"\\n핵심:")
print(f"  표준 Attention: O(N^2) 메모리 → N=16384에서 {standard_mems[-1]:.0f} MB!")
print(f"  FlashAttention: O(N) 메모리  → N=16384에서 {flash_mems[-1]:.4f} MB")"""))

# ── Cell 8: Section 3 header ─────────────────────────────────────
cells.append(md(r"""\
## 3. FlashAttention Tiling 시뮬레이션 <a name='3.-Tiling-시뮬레이션'></a>

블록 단위로 Attention을 계산하는 FlashAttention 알고리즘을 TensorFlow로 구현합니다.
핵심은 **Online Softmax**: 전체 행을 보지 않고도 정확한 softmax를 계산합니다.

$$O_i = \frac{\sum_j e^{S_{ij} - m_i} V_j}{\sum_j e^{S_{ij} - m_i}} \quad \text{where } m_i = \max_j S_{ij}$$"""))

# ── Cell 9: FlashAttention tiling simulation ─────────────────────
cells.append(code("""\
# ── FlashAttention Tiling 시뮬레이션 (블록 단위 계산) ──────────────
def standard_attention(Q, K, V):
    # 표준 Attention: 전체 S 행렬을 메모리에 생성
    d_k = tf.cast(tf.shape(Q)[-1], tf.float32)
    scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
    weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(weights, V)
    return output

def flash_attention_sim(Q, K, V, block_size=64):
    # FlashAttention 시뮬레이션: 블록 단위 + Online Softmax
    N = Q.shape[0]
    d = Q.shape[1]

    O = tf.zeros_like(Q)
    l = tf.zeros([N, 1])  # softmax 분모
    m = tf.fill([N, 1], -1e9)  # 행별 최대값

    n_blocks = (N + block_size - 1) // block_size
    hbm_reads = 0

    for j in range(n_blocks):
        kj_start = j * block_size
        kj_end = min((j + 1) * block_size, N)

        Kj = K[kj_start:kj_end]  # [Bc, d]
        Vj = V[kj_start:kj_end]  # [Bc, d]
        hbm_reads += (kj_end - kj_start) * d * 2  # K, V 읽기

        for i in range(n_blocks):
            qi_start = i * block_size
            qi_end = min((i + 1) * block_size, N)

            Qi = Q[qi_start:qi_end]  # [Br, d]
            hbm_reads += (qi_end - qi_start) * d  # Q 읽기

            Sij = tf.matmul(Qi, Kj, transpose_b=True) / tf.sqrt(tf.cast(d, tf.float32))

            m_old = m[qi_start:qi_end]
            m_new_block = tf.reduce_max(Sij, axis=-1, keepdims=True)
            m_new = tf.maximum(m_old, m_new_block)

            exp_old = tf.exp(m_old - m_new)
            exp_new = tf.exp(Sij - m_new)

            l_old = l[qi_start:qi_end]
            l_new = exp_old * l_old + tf.reduce_sum(exp_new, axis=-1, keepdims=True)

            O_old = O[qi_start:qi_end]
            O_new = exp_old * O_old + tf.matmul(exp_new, Vj)

            # 텐서 갱신 (scatter 대신 슬라이스)
            O = tf.concat([O[:qi_start], O_new, O[qi_end:]], axis=0)
            l = tf.concat([l[:qi_start], l_new, l[qi_end:]], axis=0)
            m = tf.concat([m[:qi_start], m_new, m[qi_end:]], axis=0)

    O = O / l
    return O, hbm_reads

# 테스트
N, d = 256, 64
Q = tf.random.normal([N, d])
K = tf.random.normal([N, d])
V = tf.random.normal([N, d])

std_out = standard_attention(Q, K, V)
flash_out, hbm_reads = flash_attention_sim(Q, K, V, block_size=64)

diff = tf.reduce_max(tf.abs(std_out - flash_out)).numpy()
print(f"시퀀스 길이: {N}, 헤드 차원: {d}, 블록 크기: 64")
print(f"표준 Attention vs FlashAttention 최대 오차: {diff:.2e}")
print(f"수치적으로 동일 여부: {diff < 1e-4}")
print(f"FlashAttention 시뮬레이션 HBM 읽기: {hbm_reads:,} 원소")
print(f"표준 Attention HBM 읽기 (이론): {3*N*N + 2*N*d:,} 원소")"""))

# ── Cell 10: Section 4 header ────────────────────────────────────
cells.append(md("""\
## 4. IO 복잡도 비교 시각화 <a name='4.-IO-복잡도-시각화'></a>

시퀀스 길이에 따른 HBM I/O량과 메모리 사용량을 시각화합니다."""))

# ── Cell 11: IO complexity visualization ─────────────────────────
cells.append(code("""\
# ── IO 복잡도 시각화 ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# HBM I/O 비교
ax1 = axes[0]
ax1.semilogy(seq_lengths, standard_ios, 'r-o', lw=2.5, ms=8, label='Standard: $O(Nd + N^2)$')
ax1.semilogy(seq_lengths, flash_ios, 'b-s', lw=2.5, ms=8, label='FlashAttn: $O(N^2d^2/M)$')
ax1.fill_between(seq_lengths, flash_ios, standard_ios, alpha=0.1, color='green')
ax1.set_xlabel('Sequence Length (N)', fontsize=11)
ax1.set_ylabel('HBM I/O (elements, log scale)', fontsize=11)
ax1.set_title('HBM I/O: Standard vs FlashAttention', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, which='both')

# 메모리 사용량 비교
ax2 = axes[1]
ax2.semilogy(seq_lengths, standard_mems, 'r-o', lw=2.5, ms=8, label='Standard: $O(N^2)$')
ax2.semilogy(seq_lengths, flash_mems, 'b-s', lw=2.5, ms=8, label='FlashAttn: $O(N)$')
ax2.axhline(y=80*1024, color='gray', ls='--', lw=1.5, label='A100 80GB')
ax2.set_xlabel('Sequence Length (N)', fontsize=11)
ax2.set_ylabel('Peak Memory (MB, log scale)', fontsize=11)
ax2.set_title('Memory Usage: Standard vs FlashAttention', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('chapter14_extreme_inference/flash_attention_io_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter14_extreme_inference/flash_attention_io_comparison.png")

# 절감 비율 표
print(f"\\n시퀀스 길이별 I/O 절감 비율:")
for i, N in enumerate(seq_lengths):
    ratio = standard_ios[i] / flash_ios[i]
    print(f"  N={N:>6}: {ratio:>6.1f}x 절감")"""))

# ── Cell 12: Section 5 header ────────────────────────────────────
cells.append(md("""\
## 5. FlashAttention v1 → v2 → v3 발전사 <a name='5.-FlashAttention-발전사'></a>

FlashAttention은 세 버전에 걸쳐 점진적으로 개선되었습니다. 각 버전의 핵심 기여를 정리합니다."""))

# ── Cell 13: v1→v2→v3 comparison ─────────────────────────────────
cells.append(code("""\
# ── FlashAttention v1 → v2 → v3 비교표 ──────────────────────────
print("=" * 90)
print("  FlashAttention Version Comparison")
print("=" * 90)

headers = ['항목', 'v1 (2022.05)', 'v2 (2023.07)', 'v3 (2024.07)']
rows = [
    ['논문', 'Dao et al.', 'Dao (단독)', 'Shah, Dao et al.'],
    ['타겟 GPU', 'A100', 'A100/H100', 'H100 (Hopper)'],
    ['IO 복잡도', 'O(N²d²/M)', 'O(N²d²/M) (동일)', 'O(N²d²/M) (동일)'],
    ['핵심 개선', 'Tiling+Online Softmax', '루프 순서 최적화', 'Warp 특화+비동기'],
    ['Forward 속도', '~2-4x vs PyTorch', '~2x vs v1', '~1.5-2x vs v2'],
    ['Backward 지원', 'O (Recomputation)', 'O (개선된 분할)', 'O (비동기 파이프라인)'],
    ['Causal Mask', '지원', '최적화됨 (~50% 절감)', '추가 최적화'],
    ['MQA/GQA', '미지원', '지원', '완전 지원'],
    ['FP8 지원', '미지원', '미지원', '지원 (H100)'],
    ['워프 병렬화', '배치+헤드', '+시퀀스 분할', '+warpgroup 특화'],
    ['비동기 연산', '미사용', '미사용', 'TMA+WGMMA 오버랩'],
    ['피크 TFLOPS', '~125 (A100)', '~230 (H100)', '~740 (H100, FP8)'],
]

col_widths = [18, 22, 22, 22]
header_line = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
print(header_line)
print('-' * len(header_line))

for row in rows:
    line = ' | '.join(val.ljust(w) for val, w in zip(row, col_widths))
    print(line)

print(f"\\n핵심 발전 요약:")
print(f"  v1: Tiling + Online Softmax → O(N^2) 메모리를 O(N)으로 절감")
print(f"  v2: 루프 순서 재배치 + 시퀀스 병렬화 → 2x 속도 향상")
print(f"  v3: Hopper 전용 최적화 (TMA, WGMMA) → FP8까지 확장, 740 TFLOPS")

# 성능 시각화
versions = ['v1\\n(A100)', 'v2\\n(A100)', 'v2\\n(H100)', 'v3\\n(H100 FP16)', 'v3\\n(H100 FP8)']
tflops = [125, 190, 230, 420, 740]
peak_util = [40, 61, 23, 43, 75]  # GPU utilization %

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = ['#90CAF9', '#42A5F5', '#1E88E5', '#1565C0', '#0D47A1']

ax1 = axes[0]
bars = ax1.bar(versions, tflops, color=colors, edgecolor='black', lw=0.5)
for bar, val in zip(bars, tflops):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             f'{val}', ha='center', fontsize=10, fontweight='bold')
ax1.set_ylabel('TFLOPS', fontsize=11)
ax1.set_title('FlashAttention 버전별 성능', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
bars2 = ax2.bar(versions, peak_util, color=colors, edgecolor='black', lw=0.5)
for bar, val in zip(bars2, peak_util):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{val}%', ha='center', fontsize=10, fontweight='bold')
ax2.set_ylabel('GPU Utilization (%)', fontsize=11)
ax2.set_title('FlashAttention 버전별 GPU 활용률', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chapter14_extreme_inference/flash_attention_versions.png', dpi=100, bbox_inches='tight')
plt.close()
print("\\n그래프 저장됨: chapter14_extreme_inference/flash_attention_versions.png")"""))

# ── Cell 14: Summary ─────────────────────────────────────────────
cells.append(md(r"""\
## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| 표준 Attention I/O | $O(Nd + N^2)$ HBM 접근, $O(N^2)$ 메모리 | ⭐⭐⭐ |
| FlashAttention I/O | $O(N^2d^2/M)$ HBM 접근, $O(N)$ 메모리 | ⭐⭐⭐ |
| Tiling | Q, K, V를 블록으로 분할하여 SRAM에서 처리 | ⭐⭐⭐ |
| Online Softmax | 블록별 점진적 softmax — 전체 행 없이도 정확 | ⭐⭐⭐ |
| Recomputation | Backward에서 S, P를 재계산 (저장 안 함) | ⭐⭐ |
| v1 → v2 개선 | 루프 순서 + 시퀀스 병렬화 | ⭐⭐ |
| v3 (Hopper) | TMA + WGMMA 비동기, FP8 지원 | ⭐⭐ |

### 핵심 수식

$$\text{Standard I/O} = O(Nd + N^2), \quad \text{Flash I/O} = O\left(\frac{N^2 d^2}{M}\right)$$

$$\text{Online Softmax: } m_i^{new} = \max(m_i^{old}, \max(S_{ij})), \quad \ell_i^{new} = e^{m_i^{old}-m_i^{new}}\ell_i^{old} + \sum e^{S_{ij}-m_i^{new}}$$

$$\text{Block size: } B_r = \left\lceil \frac{M}{4d} \right\rceil$$

### 다음 챕터 예고
**03_speculative_decoding.ipynb** — Draft-Verify 패러다임으로 Decode 단계를 가속하는 Speculative Decoding의 수학적 원리와 Medusa/EAGLE 등 최신 기법을 분석합니다."""))

create_notebook(cells, 'chapter14_extreme_inference/02_flash_attention_deepdive.ipynb')
