#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ─────────────────────────────────────────────
cells.append(md(r"""# 실습 퀴즈: PagedAttention 메모리 블록 할당 시뮬레이션

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다

## 목차
- [Q1: 물리/가상 블록 매핑](#q1)
- [Q2: 다중 요청 블록 할당](#q2)
- [Q3: Copy-on-Write 구현](#q3)
- [Q4: 메모리 단편화 계산](#q4)
- [종합 도전: 전체 PagedAttention 시뮬레이터](#bonus)"""))

# ── Cell 2: Import ─────────────────────────────────────────────
cells.append(code(r"""# ── 환경 설정 ──────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
print(f"TensorFlow 버전: {tf.__version__}")"""))

# ── Cell 3: Q1 Problem ────────────────────────────────────────
cells.append(md(r"""## Q1: 물리/가상 블록 매핑 <a name='q1'></a>

### 문제

물리 블록 풀에 8개의 블록(PB0~PB7)이 있고, 블록 크기는 $B_{tok} = 16$입니다.

요청 A의 시퀀스 길이가 $S_A = 45$일 때:

1. 필요한 블록 수 $N_A = \lceil S_A / B_{tok} \rceil = ?$
2. 마지막 블록의 내부 단편화 $W_{int} = (B_{tok} - S_A \bmod B_{tok}) \bmod B_{tok} = ?$
3. 메모리 활용률 $= S_A / (N_A \times B_{tok}) = ?$

**여러분의 예측:**
- 필요 블록 수: `?`개
- 내부 단편화: `?`슬롯
- 활용률: `?`%"""))

# ── Cell 4: Q1 Solution ───────────────────────────────────────
cells.append(code(r"""# ── Q1 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q1 풀이: 물리/가상 블록 매핑")
print("=" * 45)

block_size = 16
seq_len_A = 45

num_blocks_A = int(np.ceil(seq_len_A / block_size))
internal_frag = (block_size - seq_len_A % block_size) % block_size
utilization = seq_len_A / (num_blocks_A * block_size)

print(f"\n블록 크기: {block_size} tokens")
print(f"시퀀스 길이: {seq_len_A}")
print(f"\n1. 필요 블록 수: ceil({seq_len_A}/{block_size}) = {num_blocks_A}개")
print(f"2. 내부 단편화: ({block_size} - {seq_len_A}%{block_size}) % {block_size} "
      f"= ({block_size} - {seq_len_A % block_size}) % {block_size} = {internal_frag} 슬롯")
print(f"3. 활용률: {seq_len_A}/({num_blocks_A}×{block_size}) = {utilization:.2%}")

print(f"\n[해설]")
print(f"  마지막 블록(VB{num_blocks_A-1})에 {seq_len_A % block_size}개 토큰만 채워지고")
print(f"  나머지 {internal_frag}슬롯이 빈 상태 = 내부 단편화")
print(f"  하지만 외부 단편화는 0 (비연속 할당 가능)")

# 시각화
fig, ax = plt.subplots(1, 1, figsize=(12, 3))
for i in range(num_blocks_A):
    tokens_in_block = min(block_size, seq_len_A - i * block_size)
    ax.barh(0, tokens_in_block, left=i * block_size, height=0.5,
            color='steelblue', edgecolor='black', alpha=0.8)
    if tokens_in_block < block_size:
        ax.barh(0, block_size - tokens_in_block,
                left=i * block_size + tokens_in_block, height=0.5,
                color='lightcoral', edgecolor='black', alpha=0.4, hatch='//')
    ax.text(i * block_size + block_size/2, 0, f'VB{i}\n({tokens_in_block}/{block_size})',
            ha='center', va='center', fontsize=8)

ax.set_xlim(-1, num_blocks_A * block_size + 1)
ax.set_xlabel('토큰 슬롯', fontsize=11)
ax.set_title(f'요청 A: {seq_len_A} tokens → {num_blocks_A} blocks (파란=사용, 빨간=단편화)',
             fontweight='bold')
ax.set_yticks([])
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('chapter14_extreme_inference/practice/q1_block_mapping.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n그래프 저장됨: chapter14_extreme_inference/practice/q1_block_mapping.png")"""))

# ── Cell 5: Q2 Problem ────────────────────────────────────────
cells.append(md(r"""## Q2: 다중 요청 블록 할당 <a name='q2'></a>

### 문제

물리 블록 12개(PB0~PB11), 블록 크기 = 16인 시스템에 3개 요청이 도착합니다:

| 요청 | 시퀀스 길이 | 필요 블록 수 |
|------|------------|-------------|
| A | 50 | ? |
| B | 30 | ? |
| C | 75 | ? |

1. 각 요청에 필요한 블록 수를 계산하세요
2. 전통 방식($S_{max}=128$)과 PagedAttention의 메모리 사용량을 비교하세요
3. 요청 B가 완료된 후 가용 블록 수는?

**여러분의 예측:** 총 필요 블록 수 `?`개, 전통 방식 대비 절약률 `?`%"""))

# ── Cell 6: Q2 Solution ───────────────────────────────────────
cells.append(code(r"""# ── Q2 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q2 풀이: 다중 요청 블록 할당")
print("=" * 45)

block_size = 16
total_physical = 12
requests = {"A": 50, "B": 30, "C": 75}
S_max = 128

print(f"\n{'요청':>6} | {'시퀀스길이':>10} | {'필요블록':>8} | {'사용슬롯':>10} | {'내부단편화':>10}")
print("-" * 55)

total_blocks = 0
total_frag = 0
for req_id, seq_len in requests.items():
    n_blocks = int(np.ceil(seq_len / block_size))
    used_slots = n_blocks * block_size
    frag = (block_size - seq_len % block_size) % block_size
    total_blocks += n_blocks
    total_frag += frag
    print(f"{req_id:>6} | {seq_len:>10} | {n_blocks:>8} | {used_slots:>10} | {frag:>10}")

print(f"\n총 필요 블록: {total_blocks}/{total_physical}")
print(f"총 내부 단편화: {total_frag} 슬롯")

# 전통 방식 비교
trad_slots = len(requests) * S_max
paged_slots = total_blocks * block_size
actual_tokens = sum(requests.values())

print(f"\n[전통 방식 (S_max={S_max})]")
print(f"  할당 슬롯: {len(requests)} × {S_max} = {trad_slots}")
print(f"  실제 사용: {actual_tokens} → 활용률: {actual_tokens/trad_slots:.1%}")

print(f"\n[PagedAttention]")
print(f"  할당 슬롯: {total_blocks} × {block_size} = {paged_slots}")
print(f"  실제 사용: {actual_tokens} → 활용률: {actual_tokens/paged_slots:.1%}")
print(f"  절약률: {(1 - paged_slots/trad_slots)*100:.1f}%")

# 요청 B 완료 후
b_blocks = int(np.ceil(requests["B"] / block_size))
after_free = total_physical - total_blocks + b_blocks
print(f"\n[요청 B 완료 후]")
print(f"  해제 블록: {b_blocks}개")
print(f"  가용 블록: {total_physical} - {total_blocks} + {b_blocks} = {after_free}개")

print(f"\n[해설]")
print(f"  PagedAttention은 전통 방식 대비 {(1-paged_slots/trad_slots)*100:.0f}% 메모리 절약")
print(f"  해제된 블록은 즉시 다른 요청에 재할당 가능")"""))

# ── Cell 7: Q3 Problem ────────────────────────────────────────
cells.append(md(r"""## Q3: Copy-on-Write 구현 <a name='q3'></a>

### 문제

두 요청 A, B가 동일한 시스템 프롬프트(32 토큰)를 공유합니다.

```
요청 A: [시스템 프롬프트 32tok] + [사용자 입력 20tok]
요청 B: [시스템 프롬프트 32tok] + [사용자 입력 15tok]
```

블록 크기 = 16일 때:
1. 시스템 프롬프트 블록을 공유하면 몇 블록을 절약할 수 있나요?
2. 요청 B가 공유 블록의 내용을 수정하려면 어떻게 해야 하나요?

**여러분의 예측:** 절약 블록 수 `?`개"""))

# ── Cell 8: Q3 Solution ───────────────────────────────────────
cells.append(code(r"""# ── Q3 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q3 풀이: Copy-on-Write 구현")
print("=" * 45)

block_size = 16
system_prompt_len = 32
user_a_len = 20
user_b_len = 15

class CoWBlockManager:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.block_ref_count = {}
        self.page_tables = {}

    def alloc_block(self):
        if not self.free_blocks:
            return None
        pb = self.free_blocks.pop(0)
        self.block_ref_count[pb] = 1
        return pb

    def share_block(self, pb):
        self.block_ref_count[pb] = self.block_ref_count.get(pb, 0) + 1

    def copy_on_write(self, old_pb):
        if self.block_ref_count.get(old_pb, 0) <= 1:
            return old_pb
        new_pb = self.alloc_block()
        if new_pb is None:
            return None
        self.block_ref_count[old_pb] -= 1
        return new_pb

    def status(self):
        used = self.num_blocks - len(self.free_blocks)
        return used, len(self.free_blocks)

mgr = CoWBlockManager(num_blocks=16, block_size=16)

# 시스템 프롬프트 블록 할당 (공유)
shared_blocks = []
for i in range(int(np.ceil(system_prompt_len / block_size))):
    pb = mgr.alloc_block()
    shared_blocks.append(pb)

# 요청 A: 공유 블록 참조 + 사용자 입력 블록
page_table_a = list(shared_blocks)
for pb in shared_blocks:
    mgr.share_block(pb)
user_a_blocks = int(np.ceil(user_a_len / block_size))
for _ in range(user_a_blocks):
    page_table_a.append(mgr.alloc_block())

# 요청 B: 공유 블록 참조 + 사용자 입력 블록
page_table_b = list(shared_blocks)
for pb in shared_blocks:
    mgr.share_block(pb)
user_b_blocks = int(np.ceil(user_b_len / block_size))
for _ in range(user_b_blocks):
    page_table_b.append(mgr.alloc_block())

used, free = mgr.status()
without_sharing = (int(np.ceil(system_prompt_len/block_size)) * 2
                   + user_a_blocks + user_b_blocks)
saved = without_sharing - used

print(f"\n시스템 프롬프트: {system_prompt_len} tokens = {len(shared_blocks)} blocks (공유)")
print(f"요청 A 사용자 입력: {user_a_len} tokens = {user_a_blocks} blocks")
print(f"요청 B 사용자 입력: {user_b_len} tokens = {user_b_blocks} blocks")
print(f"\n공유 블록 참조 카운트: {[mgr.block_ref_count[pb] for pb in shared_blocks]}")
print(f"요청 A 페이지 테이블: {page_table_a}")
print(f"요청 B 페이지 테이블: {page_table_b}")
print(f"\n사용 블록: {used}, 가용 블록: {free}")
print(f"공유 없이 필요한 블록: {without_sharing}")
print(f"공유로 절약한 블록: {saved}개")

# Copy-on-Write 시연
print(f"\n--- Copy-on-Write 시연 ---")
cow_block = shared_blocks[0]
print(f"요청 B가 PB{cow_block} 수정 시도 (ref_count={mgr.block_ref_count[cow_block]})")
new_block = mgr.copy_on_write(cow_block)
old_idx = page_table_b.index(cow_block)
page_table_b[old_idx] = new_block
print(f"→ PB{cow_block} 복사 → PB{new_block} (새 블록 할당)")
print(f"→ PB{cow_block} ref_count: {mgr.block_ref_count[cow_block]}")
print(f"→ PB{new_block} ref_count: {mgr.block_ref_count[new_block]}")
print(f"요청 B 페이지 테이블 (업데이트): {page_table_b}")

print(f"\n[해설]")
print(f"  CoW로 공유 블록은 읽기 전용으로 유지됩니다.")
print(f"  수정이 필요할 때만 복사 → 메모리와 복사 비용 최소화")"""))

# ── Cell 9: Q4 Problem ────────────────────────────────────────
cells.append(md(r"""## Q4: 메모리 단편화 계산 <a name='q4'></a>

### 문제

1000개의 요청이 동시에 처리됩니다. 시퀀스 길이는 지수분포 $\text{Exp}(\lambda=1/80)$를 따릅니다.

블록 크기 $B_{tok}=16$, 전통 방식 $S_{max}=512$일 때:

1. 전통 방식의 평균 메모리 낭비율은?
2. PagedAttention의 평균 내부 단편화율은?
3. 블록 크기를 8로 줄이면 내부 단편화가 어떻게 변하나요?

**여러분의 예측:** 전통 방식 낭비 `?`%, PagedAttention 낭비 `?`%"""))

# ── Cell 10: Q4 Solution ──────────────────────────────────────
cells.append(code(r"""# ── Q4 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q4 풀이: 메모리 단편화 계산")
print("=" * 45)

np.random.seed(42)
n_requests = 1000
seq_lengths = np.random.exponential(scale=80, size=n_requests).astype(int) + 1
S_max = 512

print(f"\n요청 수: {n_requests}")
print(f"시퀀스 길이 통계: 평균={np.mean(seq_lengths):.1f}, "
      f"중앙값={np.median(seq_lengths):.1f}, "
      f"최대={np.max(seq_lengths)}, 최소={np.min(seq_lengths)}")

# 전통 방식
trad_total = n_requests * S_max
trad_used = sum(seq_lengths)
trad_waste = 1 - trad_used / trad_total

print(f"\n[전통 방식 (S_max={S_max})]")
print(f"  할당: {trad_total:,} 슬롯")
print(f"  사용: {trad_used:,} 슬롯")
print(f"  낭비율: {trad_waste:.1%}")

# PagedAttention (다양한 블록 크기)
for bs in [8, 16, 32, 64]:
    blocks = sum(int(np.ceil(s / bs)) for s in seq_lengths)
    paged_total = blocks * bs
    internal_frag = paged_total - trad_used
    frag_rate = internal_frag / paged_total

    print(f"\n[PagedAttention (B_tok={bs})]")
    print(f"  블록 수: {blocks:,}")
    print(f"  할당: {paged_total:,} 슬롯")
    print(f"  내부 단편화: {internal_frag:,} 슬롯 ({frag_rate:.1%})")
    print(f"  전통 방식 대비 절약: {(1 - paged_total/trad_total)*100:.1f}%")

print(f"\n[해설]")
print(f"  블록 크기가 작을수록 내부 단편화가 줄어듭니다.")
print(f"  하지만 블록이 작으면 페이지 테이블 관리 오버헤드가 증가합니다.")
print(f"  실제 vLLM은 B_tok=16을 기본값으로 사용합니다.")"""))

# ── Cell 11: Bonus Problem ────────────────────────────────────
cells.append(md(r"""## 종합 도전: 전체 PagedAttention 시뮬레이터 <a name='bonus'></a>

### 미니 프로젝트

아래 요구사항을 만족하는 **완전한 PagedAttention 시뮬레이터**를 구현하세요:

1. 물리 블록 풀 관리 (할당/해제)
2. 요청별 가상→물리 블록 페이지 테이블
3. Copy-on-Write 지원
4. 동적 블록 확장 (토큰 생성 시 블록 추가)
5. 블록 사용률 시각화"""))

# ── Cell 12: Bonus Solution ───────────────────────────────────
cells.append(code(r"""# ── 종합 도전 풀이: 전체 PagedAttention 시뮬레이터 ──────────────
print("=" * 45)
print("종합 도전 풀이: PagedAttention 시뮬레이터")
print("=" * 45)

class FullPagedAttentionSimulator:
    def __init__(self, num_physical_blocks, block_size):
        self.num_pb = num_physical_blocks
        self.block_size = block_size
        self.free_pool = list(range(num_physical_blocks))
        self.ref_counts = {}
        self.page_tables = {}
        self.seq_lens = {}
        self.history = []

    def _alloc(self):
        if not self.free_pool:
            return None
        pb = self.free_pool.pop(0)
        self.ref_counts[pb] = 1
        return pb

    def _free(self, pb):
        self.ref_counts[pb] -= 1
        if self.ref_counts[pb] <= 0:
            del self.ref_counts[pb]
            self.free_pool.append(pb)
            self.free_pool.sort()

    def new_request(self, req_id, initial_len, shared_prefix_from=None):
        self.page_tables[req_id] = []
        self.seq_lens[req_id] = 0
        if shared_prefix_from and shared_prefix_from in self.page_tables:
            src_pt = self.page_tables[shared_prefix_from]
            prefix_blocks = int(np.ceil(self.seq_lens.get(shared_prefix_from, 0) / self.block_size))
            for pb in src_pt[:prefix_blocks]:
                self.ref_counts[pb] = self.ref_counts.get(pb, 0) + 1
                self.page_tables[req_id].append(pb)
            self.seq_lens[req_id] = prefix_blocks * self.block_size
        remaining = initial_len - self.seq_lens[req_id]
        if remaining > 0:
            self.append_tokens(req_id, remaining)

    def append_tokens(self, req_id, num_tokens):
        if req_id not in self.page_tables:
            return
        for _ in range(num_tokens):
            self.seq_lens[req_id] += 1
            cur_len = self.seq_lens[req_id]
            if cur_len % self.block_size == 1 or cur_len == 1:
                pb = self._alloc()
                if pb is None:
                    print(f"  [OOM] 요청 {req_id}: 블록 부족!")
                    return
                self.page_tables[req_id].append(pb)

    def cow_write(self, req_id, block_idx):
        pt = self.page_tables[req_id]
        if block_idx >= len(pt):
            return
        old_pb = pt[block_idx]
        if self.ref_counts.get(old_pb, 0) > 1:
            new_pb = self._alloc()
            if new_pb is None:
                return
            self.ref_counts[old_pb] -= 1
            pt[block_idx] = new_pb

    def finish_request(self, req_id):
        if req_id not in self.page_tables:
            return
        for pb in self.page_tables[req_id]:
            self._free(pb)
        del self.page_tables[req_id]
        del self.seq_lens[req_id]

    def record(self, step):
        used = self.num_pb - len(self.free_pool)
        self.history.append({
            "step": step, "used": used, "free": len(self.free_pool),
            "requests": len(self.page_tables)
        })

    def summary(self):
        used = self.num_pb - len(self.free_pool)
        total_tokens = sum(self.seq_lens.values())
        total_slots = used * self.block_size
        util = total_tokens / total_slots if total_slots > 0 else 0
        print(f"  블록: {used}/{self.num_pb} 사용 ({used/self.num_pb:.1%})")
        print(f"  토큰: {total_tokens}/{total_slots} 슬롯 (활용률: {util:.1%})")
        print(f"  활성 요청: {len(self.page_tables)}")

sim = FullPagedAttentionSimulator(num_physical_blocks=32, block_size=16)

print("\n[Step 0] 요청 A: 프롬프트 48 토큰")
sim.new_request("A", 48)
sim.record(0)
sim.summary()

print("\n[Step 1] 요청 B: A와 프롬프트 공유 (prefix sharing)")
sim.new_request("B", 48, shared_prefix_from="A")
sim.record(1)
sim.summary()

print("\n[Step 2] 요청 B가 공유 블록 수정 → CoW")
sim.cow_write("B", 0)
sim.record(2)
sim.summary()

print("\n[Step 3] A, B 각각 토큰 생성 (32 토큰씩)")
sim.append_tokens("A", 32)
sim.append_tokens("B", 32)
sim.record(3)
sim.summary()

print("\n[Step 4] 요청 A 완료")
sim.finish_request("A")
sim.record(4)
sim.summary()

print("\n[Step 5] 새 요청 C, D 도착")
sim.new_request("C", 100)
sim.new_request("D", 60)
sim.record(5)
sim.summary()

print("\n[Step 6] B, C 완료")
sim.finish_request("B")
sim.finish_request("C")
sim.record(6)
sim.summary()

# 시각화
steps = [h["step"] for h in sim.history]
used_hist = [h["used"] for h in sim.history]
free_hist = [h["free"] for h in sim.history]
req_hist = [h["requests"] for h in sim.history]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
ax1.fill_between(steps, 0, used_hist, alpha=0.6, color='coral', label='사용 블록')
ax1.fill_between(steps, used_hist, [u+f for u, f in zip(used_hist, free_hist)],
                 alpha=0.4, color='lightgreen', label='가용 블록')
ax1.set_xlabel('시뮬레이션 스텝', fontsize=11)
ax1.set_ylabel('블록 수', fontsize=11)
ax1.set_title('블록 사용량 변화 (종합 시뮬레이션)', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(steps, req_hist, 'b-o', lw=2.5, ms=8)
ax2.fill_between(steps, 0, req_hist, alpha=0.15, color='blue')
ax2.set_xlabel('시뮬레이션 스텝', fontsize=11)
ax2.set_ylabel('활성 요청 수', fontsize=11)
ax2.set_title('활성 요청 수 변화', fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter14_extreme_inference/practice/bonus_simulation.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n그래프 저장됨: chapter14_extreme_inference/practice/bonus_simulation.png")
print("\n전체 시뮬레이션 완료!")"""))

path = '/workspace/chapter14_extreme_inference/practice/ex01_paged_attention_sim.ipynb'
create_notebook(cells, path)
