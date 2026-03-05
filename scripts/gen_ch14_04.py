#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ─────────────────────────────────────────────
cells.append(md(r"""# Chapter 14: 극단적 추론 최적화 — vLLM과 PagedAttention

## 학습 목표
- OS의 **가상 메모리(Page Table)** 개념이 LLM KV Cache 관리에 어떻게 적용되는지 이해한다
- **PagedAttention**의 블록 할당·해제·Copy-on-Write 메커니즘을 수식으로 도출한다
- **Continuous Batching**이 Static Batching 대비 처리량을 어떻게 향상시키는지 정량적으로 분석한다
- vLLM의 **동적 KV Block 스케줄링** 전략을 시뮬레이션으로 검증한다
- 메모리 단편화(Fragmentation)의 내부/외부 낭비를 시각화하고 PagedAttention의 해결 방식을 이해한다

## 목차
1. [수학적 기초: 페이지 테이블과 메모리 관리](#1.-수학적-기초)
2. [PagedAttention 블록 할당 시뮬레이터](#2.-PagedAttention-블록-할당)
3. [Continuous Batching vs Static Batching](#3.-Continuous-Batching)
4. [메모리 단편화 시각화](#4.-메모리-단편화)
5. [정리](#5.-정리)"""))

# ── Cell 2: Math Foundations ────────────────────────────────────
cells.append(md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### 가상 메모리와 페이지 테이블

OS는 물리 메모리를 고정 크기 **페이지(Page)**로 나누고, **페이지 테이블**로 가상 → 물리 주소를 매핑합니다.
PagedAttention은 이 아이디어를 KV Cache에 적용합니다:

$$\text{PageTable}: \text{VirtualBlock}[i] \;\rightarrow\; \text{PhysicalBlock}[j]$$

- 각 블록은 $B_{tok}$개의 토큰에 대한 KV 벡터를 저장
- 블록 크기: $B_{tok} \times 2 \times n_{kv} \times d_{head} \times \text{bytes}$ (K와 V 각각)

### KV Cache 메모리 — 전통 방식 vs PagedAttention

**전통 방식 (연속 할당):**

$$M_{KV}^{trad} = 2 \times L \times n_{kv} \times d_{head} \times S_{max} \times B \times \text{bytes\_per\_elem}$$

- $L$: 레이어 수, $n_{kv}$: KV 헤드 수, $d_{head}$: 헤드 차원
- $S_{max}$: **최대** 시퀀스 길이 (사용하지 않는 부분도 할당)
- $B$: 배치 크기

**PagedAttention (블록 단위 할당):**

$$M_{KV}^{paged} = N_{blocks}^{used} \times B_{tok} \times 2 \times n_{kv} \times d_{head} \times \text{bytes}$$

$$N_{blocks}^{used} = \sum_{r=1}^{R} \left\lceil \frac{S_r}{B_{tok}} \right\rceil$$

- $S_r$: 요청 $r$의 **실제** 시퀀스 길이
- $R$: 동시 요청 수

### 메모리 절약률

$$\text{Savings} = 1 - \frac{M_{KV}^{paged}}{M_{KV}^{trad}} = 1 - \frac{\sum_r \lceil S_r / B_{tok} \rceil \cdot B_{tok}}{R \cdot S_{max}}$$

### 메모리 단편화 분석

| 유형 | 정의 | 수식 |
|------|------|------|
| 내부 단편화 | 마지막 블록의 미사용 슬롯 | $W_{int} = \sum_r (B_{tok} - S_r \bmod B_{tok}) \bmod B_{tok}$ |
| 외부 단편화 | 연속 공간 부족으로 할당 불가 | PagedAttention에서는 **0** (비연속 할당) |

### Continuous Batching 처리량

$$\text{Throughput}_{static} = \frac{B}{\max_r T_r}, \quad \text{Throughput}_{cont} \approx \frac{\sum_r 1/T_r}{1} \cdot B_{eff}$$

- $T_r$: 요청 $r$의 처리 시간
- Static Batching은 가장 긴 요청이 끝날 때까지 GPU 자원이 낭비됨

**요약 표:**

| 개념 | 수식 | 설명 |
|------|------|------|
| 페이지 매핑 | $\text{VBlock}[i] \to \text{PBlock}[j]$ | 비연속 메모리 할당 |
| 블록 메모리 | $B_{tok} \times 2 n_{kv} d_h \times \text{bytes}$ | 블록당 KV 저장량 |
| 내부 단편화 | $(B_{tok} - S \bmod B_{tok}) \bmod B_{tok}$ | 마지막 블록 낭비 |
| 절약률 | $1 - M_{paged}/M_{trad}$ | 실제 대비 최대 할당 |"""))

# ── Cell 3: 🐣 Friendly Explanation ────────────────────────────
cells.append(md(r"""### 🐣 초등학생을 위한 PagedAttention 친절 설명!

#### 🔢 PagedAttention이 뭔가요?

> 💡 **비유**: 도서관에서 책을 빌리는 것과 같아요!

전통 방식은 마치 **"이 사람은 최대 100권까지 빌릴 수 있으니, 100칸짜리 책장을 통째로 예약해줘!"** 라고 하는 거예요.
실제로 10권만 빌려도 나머지 90칸은 텅 비어 있죠. 😢

PagedAttention은 **"책 5권이 들어가는 작은 바구니를 필요할 때마다 하나씩 빌려줄게!"** 라고 하는 거예요.
10권 빌리면 바구니 2개만 쓰면 되고, 바구니가 꽉 차면 새 바구니를 하나 더 주면 됩니다! 🎉

#### 🏗️ 가상 블록과 물리 블록

| 개념 | 비유 |
|------|------|
| 가상 블록 | 빌린 책의 **목록 번호** (1번, 2번, 3번...) |
| 물리 블록 | 실제 **책장 위치** (A-3선반, B-7선반...) |
| 페이지 테이블 | 목록 번호 → 실제 위치를 알려주는 **지도** |

바구니(블록)가 비연속적인 위치에 있어도 목록만 있으면 순서대로 찾을 수 있어요!"""))

# ── Cell 4: 📝 Practice Problems ───────────────────────────────
cells.append(md(r"""### 📝 연습 문제

#### 문제 1: 블록 수 계산

모델 설정: $B_{tok} = 16$, 요청 A의 시퀀스 길이 $S_A = 50$, 요청 B의 시퀀스 길이 $S_B = 30$.
각 요청에 필요한 블록 수와 총 내부 단편화를 구하세요.

<details>
<summary>💡 풀이 확인</summary>

$$N_A = \lceil 50 / 16 \rceil = 4 \text{ 블록}, \quad N_B = \lceil 30 / 16 \rceil = 2 \text{ 블록}$$

내부 단편화:
$$W_A = (16 - 50 \bmod 16) \bmod 16 = (16 - 2) = 14 \text{ 슬롯}$$
$$W_B = (16 - 30 \bmod 16) \bmod 16 = (16 - 14) = 2 \text{ 슬롯}$$
$$W_{total} = 14 + 2 = 16 \text{ 슬롯 낭비}$$
</details>

#### 문제 2: 메모리 절약률

전통 방식: $S_{max} = 2048$, 배치 2개 (실제 길이 50, 30).
PagedAttention($B_{tok}=16$) 대비 절약률을 구하세요.

<details>
<summary>💡 풀이 확인</summary>

$$M_{trad} \propto 2 \times 2048 = 4096 \text{ (슬롯 기준)}$$
$$M_{paged} \propto (4 + 2) \times 16 = 96 \text{ (슬롯 기준)}$$
$$\text{Savings} = 1 - \frac{96}{4096} = 1 - 0.0234 = 97.7\%$$
</details>"""))

# ── Cell 5: Import Cell ────────────────────────────────────────
cells.append(code(r"""# ── 환경 설정 및 임포트 ──────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import time

np.random.seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"NumPy 버전: {np.__version__}")"""))

# ── Cell 6: Section 2 Markdown ─────────────────────────────────
cells.append(md(r"""## 2. PagedAttention 블록 할당 시뮬레이터 <a name='2.-PagedAttention-블록-할당'></a>

### PagedAttention 블록 할당 다이어그램

아래는 3개의 요청이 물리 블록 풀에서 비연속적으로 블록을 할당받는 과정입니다:

```
┌─────────────────────────────────────────────────────┐
│                   Physical Block Pool                │
│  [PB0] [PB1] [PB2] [PB3] [PB4] [PB5] [PB6] [PB7]  │
└─────────────────────────────────────────────────────┘

Request A (seq_len=50, 4 blocks needed):
  Virtual:  [VB0] → [VB1] → [VB2] → [VB3]
  Physical: [PB0]   [PB1]   [PB4]   [PB6]   ← 비연속!

Request B (seq_len=30, 2 blocks needed):
  Virtual:  [VB0] → [VB1]
  Physical: [PB2]   [PB3]

Free Pool: [PB5] [PB7]

Page Table:
  Req A: {VB0→PB0, VB1→PB1, VB2→PB4, VB3→PB6}
  Req B: {VB0→PB2, VB1→PB3}
```

핵심: 물리 블록이 **비연속**이어도 페이지 테이블로 순서를 추적하므로 문제없습니다."""))

# ── Cell 7: PagedAttention Simulator Code ──────────────────────
cells.append(code(r"""# ── PagedAttention 블록 할당 시뮬레이터 ──────────────────────────
# 물리/가상 블록 매핑을 시뮬레이션합니다

class PagedAttentionAllocator:
    def __init__(self, num_physical_blocks, block_size):
        self.num_physical_blocks = num_physical_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_physical_blocks))
        self.page_tables = {}
        self.seq_lengths = {}

    def allocate(self, request_id, seq_len):
        num_blocks_needed = int(np.ceil(seq_len / self.block_size))
        if num_blocks_needed > len(self.free_blocks):
            print(f"  [ERROR] 요청 {request_id}: {num_blocks_needed}블록 필요, "
                  f"가용 {len(self.free_blocks)}블록 — 할당 실패!")
            return False
        allocated = []
        for _ in range(num_blocks_needed):
            pb = self.free_blocks.pop(0)
            allocated.append(pb)
        self.page_tables[request_id] = allocated
        self.seq_lengths[request_id] = seq_len
        internal_frag = (self.block_size - seq_len % self.block_size) % self.block_size
        print(f"  요청 {request_id}: seq_len={seq_len}, "
              f"블록 {num_blocks_needed}개 할당 → 물리블록 {allocated}")
        print(f"    내부 단편화: {internal_frag} 슬롯 "
              f"(마지막 블록 {seq_len % self.block_size or self.block_size}/{self.block_size} 사용)")
        return True

    def free(self, request_id):
        if request_id in self.page_tables:
            freed = self.page_tables.pop(request_id)
            self.free_blocks.extend(freed)
            self.free_blocks.sort()
            del self.seq_lengths[request_id]
            print(f"  요청 {request_id} 해제: 물리블록 {freed} 반환")
            return freed
        return []

    def status(self):
        used = sum(len(v) for v in self.page_tables.values())
        free = len(self.free_blocks)
        total_slots = self.num_physical_blocks * self.block_size
        used_slots = sum(self.seq_lengths.values())
        internal_frag = sum(
            (self.block_size - s % self.block_size) % self.block_size
            for s in self.seq_lengths.values()
        )
        print(f"\n{'='*55}")
        print(f"  블록 사용: {used}/{self.num_physical_blocks} "
              f"({used/self.num_physical_blocks*100:.1f}%)")
        print(f"  가용 블록: {free} → {self.free_blocks}")
        print(f"  실제 토큰: {used_slots}/{total_slots} 슬롯 "
              f"(활용률: {used_slots/total_slots*100:.1f}%)")
        print(f"  내부 단편화: {internal_frag} 슬롯")
        print(f"  외부 단편화: 0 (비연속 할당)")
        print(f"{'='*55}")

print("=== PagedAttention 블록 할당 시뮬레이션 ===\n")
allocator = PagedAttentionAllocator(num_physical_blocks=16, block_size=16)

print("[1단계] 3개 요청 동시 할당")
allocator.allocate("A", seq_len=50)
allocator.allocate("B", seq_len=30)
allocator.allocate("C", seq_len=75)
allocator.status()

print("\n[2단계] 요청 B 완료 → 블록 해제")
allocator.free("B")
allocator.status()

print("\n[3단계] 새 요청 D 할당 (해제된 블록 재사용)")
allocator.allocate("D", seq_len=40)
allocator.status()

print("\n[페이지 테이블 내용]")
for req_id, blocks in allocator.page_tables.items():
    vblocks = [f"VB{i}→PB{pb}" for i, pb in enumerate(blocks)]
    print(f"  요청 {req_id}: {{{', '.join(vblocks)}}}")"""))

# ── Cell 8: Section 3 Markdown ─────────────────────────────────
cells.append(md(r"""## 3. Continuous Batching vs Static Batching <a name='3.-Continuous-Batching'></a>

### Static Batching의 문제

Static Batching에서는 배치 내 **모든 요청이 끝날 때까지** 기다려야 합니다:

$$T_{static} = \max_{r \in \text{batch}} T_r$$

짧은 요청이 먼저 끝나도 해당 슬롯의 GPU 자원은 유휴 상태입니다.

### Continuous Batching의 해결

완료된 요청의 슬롯에 **즉시 새 요청을 삽입**합니다:

$$\text{GPU Utilization}_{cont} \approx 1 - \frac{\text{swap overhead}}{\text{total time}} \gg \text{GPU Util}_{static}$$"""))

# ── Cell 9: Continuous vs Static Batching Code ─────────────────
cells.append(code(r"""# ── Continuous Batching vs Static Batching 비교 ─────────────────
# 요청별 생성 토큰 수가 다른 상황을 시뮬레이션합니다

np.random.seed(42)

num_requests = 20
max_batch_size = 4
output_lengths = np.random.randint(10, 200, size=num_requests)
time_per_token = 0.01

# --- Static Batching ---
static_total_time = 0.0
static_tokens_generated = 0
idx = 0
while idx < num_requests:
    batch = output_lengths[idx:idx + max_batch_size]
    batch_time = max(batch) * time_per_token
    static_total_time += batch_time
    static_tokens_generated += sum(batch)
    idx += max_batch_size

static_throughput = static_tokens_generated / static_total_time

# --- Continuous Batching ---
cont_total_time = 0.0
cont_tokens_generated = 0
active_slots = np.zeros(max_batch_size, dtype=int)
waiting_queue = list(output_lengths)
step = 0

for slot_i in range(min(max_batch_size, len(waiting_queue))):
    active_slots[slot_i] = waiting_queue.pop(0)

while np.any(active_slots > 0) or waiting_queue:
    active_slots = np.maximum(active_slots - 1, 0)
    tokens_this_step = np.sum(active_slots > 0) + np.sum(active_slots == 0)
    cont_tokens_generated += int(np.sum(active_slots >= 0) - np.sum(active_slots == 0)) + np.sum(active_slots == 0)
    for slot_i in range(max_batch_size):
        if active_slots[slot_i] == 0 and waiting_queue:
            active_slots[slot_i] = waiting_queue.pop(0)
    cont_total_time += time_per_token
    step += 1
    if step > 10000:
        break

cont_tokens_generated = sum(output_lengths)
cont_throughput = cont_tokens_generated / cont_total_time

print("=" * 60)
print(f"{'배칭 방식':<25} | {'총 시간(s)':>10} | {'처리량(tok/s)':>14}")
print("-" * 60)
print(f"{'Static Batching':<25} | {static_total_time:>10.3f} | {static_throughput:>14.1f}")
print(f"{'Continuous Batching':<25} | {cont_total_time:>10.3f} | {cont_throughput:>14.1f}")
print("-" * 60)
speedup = cont_throughput / static_throughput
print(f"{'Continuous 처리량 향상':<25} | {'':>10} | {speedup:>13.2f}x")
print("=" * 60)

print(f"\n총 요청 수: {num_requests}")
print(f"총 생성 토큰: {sum(output_lengths)}")
print(f"출력 길이 범위: {min(output_lengths)} ~ {max(output_lengths)} 토큰")
print(f"\nStatic Batching은 배치 내 가장 긴 요청 완료까지 대기 → GPU 낭비 발생")
print(f"Continuous Batching은 완료 슬롯에 즉시 새 요청 삽입 → GPU 활용률 극대화")"""))

# ── Cell 10: Static vs Continuous Visualization ────────────────
cells.append(code(r"""# ── Static vs Continuous Batching 타임라인 시각화 ────────────────
np.random.seed(42)

batch_size = 4
requests = [
    {"id": "R1", "len": 30},
    {"id": "R2", "len": 80},
    {"id": "R3", "len": 15},
    {"id": "R4", "len": 60},
    {"id": "R5", "len": 25},
    {"id": "R6", "len": 45},
    {"id": "R7", "len": 70},
    {"id": "R8", "len": 20},
]

colors = plt.cm.Set3(np.linspace(0, 1, len(requests)))

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

# --- Static Batching Timeline ---
ax1 = axes[0]
ax1.set_title('Static Batching (배치 단위 대기)', fontweight='bold', fontsize=12)
time_offset = 0
for batch_start in range(0, len(requests), batch_size):
    batch = requests[batch_start:batch_start + batch_size]
    max_len = max(r["len"] for r in batch)
    for slot_i, r in enumerate(batch):
        ax1.barh(slot_i, r["len"], left=time_offset, height=0.6,
                 color=colors[batch_start + slot_i], edgecolor='black', linewidth=0.5)
        ax1.barh(slot_i, max_len - r["len"], left=time_offset + r["len"],
                 height=0.6, color='lightgray', edgecolor='black',
                 linewidth=0.5, alpha=0.4, hatch='//')
        ax1.text(time_offset + r["len"]/2, slot_i, r["id"],
                 ha='center', va='center', fontsize=8, fontweight='bold')
    time_offset += max_len
ax1.set_xlabel('토큰 생성 스텝', fontsize=11)
ax1.set_ylabel('GPU 슬롯', fontsize=11)
ax1.set_yticks(range(batch_size))
ax1.axvline(x=80, color='red', ls='--', lw=1, alpha=0.5)
ax1.legend(['유휴 구간 (회색)'], fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3, axis='x')

# --- Continuous Batching Timeline ---
ax2 = axes[1]
ax2.set_title('Continuous Batching (즉시 슬롯 재사용)', fontweight='bold', fontsize=12)
slots = [0] * batch_size
slot_assignments = []
queue = list(range(len(requests)))
time_cursor = [0] * batch_size

for slot_i in range(min(batch_size, len(queue))):
    req_idx = queue.pop(0)
    r = requests[req_idx]
    start = slots[slot_i]
    end = start + r["len"]
    slot_assignments.append((slot_i, start, r["len"], req_idx))
    slots[slot_i] = end

while queue:
    earliest_slot = int(np.argmin(slots))
    req_idx = queue.pop(0)
    r = requests[req_idx]
    start = slots[earliest_slot]
    end = start + r["len"]
    slot_assignments.append((earliest_slot, start, r["len"], req_idx))
    slots[earliest_slot] = end

for (slot_i, start, length, req_idx) in slot_assignments:
    ax2.barh(slot_i, length, left=start, height=0.6,
             color=colors[req_idx], edgecolor='black', linewidth=0.5)
    ax2.text(start + length/2, slot_i, requests[req_idx]["id"],
             ha='center', va='center', fontsize=8, fontweight='bold')

ax2.set_xlabel('토큰 생성 스텝', fontsize=11)
ax2.set_ylabel('GPU 슬롯', fontsize=11)
ax2.set_yticks(range(batch_size))
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('chapter14_extreme_inference/batching_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter14_extreme_inference/batching_comparison.png")

static_time = 0
for batch_start in range(0, len(requests), batch_size):
    batch = requests[batch_start:batch_start + batch_size]
    static_time += max(r["len"] for r in batch)
cont_time = max(slots)
total_tokens = sum(r["len"] for r in requests)

print(f"\nStatic 총 시간: {static_time} 스텝")
print(f"Continuous 총 시간: {cont_time} 스텝")
print(f"총 토큰: {total_tokens}")
print(f"처리량 향상: {static_time/cont_time:.2f}x")"""))

# ── Cell 11: Section 4 Markdown ────────────────────────────────
cells.append(md(r"""## 4. 메모리 단편화 시각화 <a name='4.-메모리-단편화'></a>

### 내부 단편화 (Internal Fragmentation)

블록 크기 $B_{tok}$로 나누어 떨어지지 않는 시퀀스에서 마지막 블록에 빈 슬롯이 발생합니다:

$$W_{internal}(S) = (B_{tok} - S \bmod B_{tok}) \bmod B_{tok}$$

### 외부 단편화 (External Fragmentation)

전통 연속 할당에서는 빈 공간이 있어도 **연속 공간**이 부족하면 할당에 실패합니다.
PagedAttention은 비연속 할당이므로 **외부 단편화가 0**입니다."""))

# ── Cell 12: Fragmentation Visualization Code ──────────────────
cells.append(code(r"""# ── 메모리 단편화 비교 시각화 ──────────────────────────────────────
block_sizes = [4, 8, 16, 32, 64]
np.random.seed(42)
num_samples = 500
seq_lengths = np.random.exponential(scale=80, size=num_samples).astype(int) + 1

results = {}
for bs in block_sizes:
    internal_waste = [(bs - s % bs) % bs for s in seq_lengths]
    waste_ratio = [w / bs for w in internal_waste]
    avg_waste = np.mean(internal_waste)
    avg_ratio = np.mean(waste_ratio)
    results[bs] = {
        "avg_waste": avg_waste,
        "avg_ratio": avg_ratio,
        "waste_dist": internal_waste,
    }

print(f"{'블록 크기':<12} | {'평균 낭비(슬롯)':>14} | {'평균 낭비 비율':>14} | {'최대 낭비':>10}")
print("-" * 58)
for bs in block_sizes:
    r = results[bs]
    print(f"{bs:<12} | {r['avg_waste']:>14.2f} | {r['avg_ratio']:>13.1%} | {bs-1:>10}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: 블록 크기별 내부 단편화
ax1 = axes[0]
avg_wastes = [results[bs]["avg_waste"] for bs in block_sizes]
avg_ratios = [results[bs]["avg_ratio"] * 100 for bs in block_sizes]
x = np.arange(len(block_sizes))
bars = ax1.bar(x, avg_wastes, color='coral', edgecolor='black', alpha=0.8, width=0.5)
ax1.set_xlabel('블록 크기 (tokens)', fontsize=11)
ax1.set_ylabel('평균 내부 단편화 (슬롯)', fontsize=11)
ax1.set_title('블록 크기별 내부 단편화', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(block_sizes)
ax1.grid(True, alpha=0.3, axis='y')
for i, (w, r) in enumerate(zip(avg_wastes, avg_ratios)):
    ax1.text(i, w + 0.5, f'{r:.1f}%', ha='center', fontsize=9, fontweight='bold')

# 오른쪽: Traditional vs PagedAttention 비교
ax2 = axes[1]
traditional_waste = []
paged_waste = []
max_seq = max(seq_lengths)

for n_req in [10, 50, 100, 200, 500]:
    sample = seq_lengths[:n_req]
    trad_total = n_req * max_seq
    trad_used = sum(sample)
    trad_waste_pct = (1 - trad_used / trad_total) * 100
    traditional_waste.append(trad_waste_pct)

    bs = 16
    paged_blocks = sum(int(np.ceil(s / bs)) for s in sample)
    paged_total = paged_blocks * bs
    paged_used = sum(sample)
    paged_waste_pct = (1 - paged_used / paged_total) * 100
    paged_waste.append(paged_waste_pct)

n_reqs = [10, 50, 100, 200, 500]
x2 = np.arange(len(n_reqs))
width = 0.35
ax2.bar(x2 - width/2, traditional_waste, width, label='Traditional (연속 할당)',
        color='salmon', edgecolor='black', alpha=0.8)
ax2.bar(x2 + width/2, paged_waste, width, label='PagedAttention (블록 할당)',
        color='skyblue', edgecolor='black', alpha=0.8)
ax2.set_xlabel('동시 요청 수', fontsize=11)
ax2.set_ylabel('메모리 낭비 비율 (%)', fontsize=11)
ax2.set_title('Traditional vs PagedAttention 메모리 낭비', fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(n_reqs)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chapter14_extreme_inference/fragmentation_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n그래프 저장됨: chapter14_extreme_inference/fragmentation_comparison.png")

print(f"\n전통 방식: max_seq={max_seq}로 고정 할당 → 평균 {np.mean(traditional_waste):.1f}% 낭비")
print(f"PagedAttention(B=16): 블록 단위 할당 → 평균 {np.mean(paged_waste):.1f}% 낭비")
print(f"절약 효과: {np.mean(traditional_waste) - np.mean(paged_waste):.1f}%p 메모리 절약")"""))

# ── Cell 13: KV Block Scheduling Simulation ────────────────────
cells.append(code(r"""# ── 동적 KV Block 스케줄링 시뮬레이션 ────────────────────────────
# vLLM의 동적 블록 할당/해제를 시간축으로 추적합니다

class VLLMScheduler:
    def __init__(self, total_blocks, block_size):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_count = total_blocks
        self.requests = {}
        self.history = []

    def step(self, time, arrivals=None, completions=None):
        if completions:
            for req_id in completions:
                if req_id in self.requests:
                    freed = self.requests.pop(req_id)
                    self.free_count += freed

        if arrivals:
            for req_id, seq_len in arrivals:
                needed = int(np.ceil(seq_len / self.block_size))
                if needed <= self.free_count:
                    self.requests[req_id] = needed
                    self.free_count -= needed

        self.history.append({
            "time": time,
            "used_blocks": self.total_blocks - self.free_count,
            "free_blocks": self.free_count,
            "active_requests": len(self.requests),
        })

scheduler = VLLMScheduler(total_blocks=64, block_size=16)

events = [
    (0,  [("R1", 120), ("R2", 80)], []),
    (1,  [("R3", 200)], []),
    (2,  [("R4", 50)], ["R2"]),
    (3,  [("R5", 150)], []),
    (4,  [], ["R1"]),
    (5,  [("R6", 100), ("R7", 60)], ["R4"]),
    (6,  [], ["R3"]),
    (7,  [("R8", 180)], []),
    (8,  [], ["R5", "R6"]),
    (9,  [], ["R7", "R8"]),
]

print("=== vLLM 동적 KV Block 스케줄링 ===\n")
print(f"{'시간':>4} | {'도착':>20} | {'완료':>15} | {'사용블록':>8} | {'가용블록':>8} | {'활성요청':>8}")
print("-" * 80)

for time, arrivals, completions in events:
    scheduler.step(time, arrivals, completions)
    h = scheduler.history[-1]
    arr_str = ",".join(f"{a[0]}({a[1]})" for a in arrivals) if arrivals else "-"
    comp_str = ",".join(completions) if completions else "-"
    print(f"{time:>4} | {arr_str:>20} | {comp_str:>15} | "
          f"{h['used_blocks']:>8} | {h['free_blocks']:>8} | {h['active_requests']:>8}")

times = [h["time"] for h in scheduler.history]
used = [h["used_blocks"] for h in scheduler.history]
free = [h["free_blocks"] for h in scheduler.history]
active = [h["active_requests"] for h in scheduler.history]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.fill_between(times, 0, used, alpha=0.6, color='coral', label='사용 블록')
ax1.fill_between(times, used, [u+f for u, f in zip(used, free)],
                 alpha=0.4, color='lightgreen', label='가용 블록')
ax1.set_xlabel('시간 스텝', fontsize=11)
ax1.set_ylabel('블록 수', fontsize=11)
ax1.set_title('KV Block 사용량 변화', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 70)

ax2 = axes[1]
ax2.plot(times, active, 'b-o', lw=2, ms=7, label='활성 요청 수')
ax2.fill_between(times, 0, active, alpha=0.15, color='blue')
ax2.set_xlabel('시간 스텝', fontsize=11)
ax2.set_ylabel('활성 요청 수', fontsize=11)
ax2.set_title('활성 요청 수 변화', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter14_extreme_inference/kv_block_scheduling.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n그래프 저장됨: chapter14_extreme_inference/kv_block_scheduling.png")
print(f"\n최대 블록 사용: {max(used)}/{scheduler.total_blocks} "
      f"({max(used)/scheduler.total_blocks*100:.1f}%)")
print(f"최대 활성 요청: {max(active)}개")"""))

# ── Cell 14: Summary ───────────────────────────────────────────
cells.append(md(r"""## 5. 정리 <a name='5.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| PagedAttention | OS 페이지 테이블 방식으로 KV Cache를 비연속 블록에 저장 | ⭐⭐⭐ |
| 가상/물리 블록 매핑 | VirtualBlock → PhysicalBlock 페이지 테이블 | ⭐⭐⭐ |
| 내부 단편화 | 마지막 블록의 미사용 슬롯 (유일한 낭비) | ⭐⭐ |
| 외부 단편화 제거 | 비연속 할당으로 외부 단편화 = 0 | ⭐⭐⭐ |
| Continuous Batching | 완료 슬롯에 즉시 새 요청 삽입 | ⭐⭐⭐ |
| 동적 KV 스케줄링 | 블록 할당/해제를 실시간으로 관리 | ⭐⭐ |
| Copy-on-Write | 공유 블록을 수정 시에만 복사 | ⭐⭐ |

### 핵심 수식

$$\text{PageTable}: \text{VBlock}[i] \rightarrow \text{PBlock}[j] \quad \text{(비연속 할당)}$$

$$W_{internal} = (B_{tok} - S \bmod B_{tok}) \bmod B_{tok} \quad \text{(내부 단편화)}$$

$$\text{Savings} = 1 - \frac{\sum_r \lceil S_r / B_{tok} \rceil \cdot B_{tok}}{R \cdot S_{max}} \quad \text{(메모리 절약률)}$$

### 참고 논문
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (arxiv 2309.06180)
- vLLM 공식 문서: https://docs.vllm.ai

### 다음 챕터 예고
**Chapter 14-05: 양자화 심화 (GPTQ & AWQ)** — PTQ 기초부터 Hessian 기반 GPTQ, Activation-aware AWQ까지 양자화 기법을 수식으로 도출하고 W4A16/INT8/FP8 벤치마크를 비교합니다."""))

path = '/workspace/chapter14_extreme_inference/04_vllm_and_paged_attention.ipynb'
create_notebook(cells, path)
