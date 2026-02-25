"""Generate Chapter 12-02: KV Cache and Memory."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helper import md, code, create_notebook

cells = [
# ─── Cell 0: 헤더 ───
md("""# Chapter 12: 최신 LLM 아키텍처 — KV Cache와 메모리 관리

## 학습 목표
- Autoregressive 생성에서 **KV Cache의 역할과 메모리 공식**을 정확히 이해하고 계산한다
- Llama 3 8B 실제 파라미터로 **배치/시퀀스별 KV 캐시 메모리**를 계산한다
- **Rolling Buffer**(Sliding Window) 방식으로 고정 메모리 KV 관리를 구현한다
- **Multi-Turn 대화**에서 KV 캐시 메모리 증가 패턴을 시각화하고 분석한다
- **Prefix Caching** 개요와 시스템 프롬프트 재사용 이점을 이해한다

## 목차
1. [수학적 기초: KV Cache 메모리 공식](#1.-수학적-기초)
2. [Autoregressive 생성과 KV Cache](#2.-KV-Cache-원리)
3. [Rolling Buffer (Sliding Window)](#3.-Rolling-Buffer)
4. [Multi-Turn 대화 메모리 분석](#4.-Multi-Turn-분석)
5. [Prefix Caching 시뮬레이션](#5.-Prefix-Caching)
6. [정리](#6.-정리)"""),

# ─── Cell 1: 수학적 기초 ───
md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### KV Cache 메모리 공식

Autoregressive 디코딩에서 매 토큰 생성 시 이전 토큰들의 K, V를 **재계산하지 않고 캐시**합니다:

$$M_{KV} = 2 \times L \times H_{kv} \times d_{head} \times S \times B \times \text{bytes\_per\_element}$$

- $2$: K와 V 두 텐서
- $L$: 레이어(블록) 수
- $H_{kv}$: KV 헤드 수 (GQA에서 Q 헤드보다 적음)
- $d_{head}$: 헤드 차원
- $S$: 현재까지 생성된 시퀀스 길이
- $B$: 배치 크기
- bytes: FP16=2, FP32=4, INT8=1

**Llama 3 8B 예시 ($L=32, H_{kv}=8, d_{head}=128$, FP16):**

| 시퀀스 길이 | 배치=1 | 배치=8 | 배치=32 |
|------------|--------|--------|---------|
| $S=512$ | 64 MB | 512 MB | 2.0 GB |
| $S=2048$ | 256 MB | 2.0 GB | 8.0 GB |
| $S=8192$ | 1.0 GB | 8.0 GB | 32 GB |

### KV Cache 없이 vs 있을 때 복잡도

| 항목 | Cache 없음 | Cache 있음 |
|------|-----------|-----------|
| 토큰당 K,V 계산 | $O(S \cdot d)$ 전체 재계산 | $O(d)$ 새 토큰만 |
| Attention FLOPs | $O(S^2 \cdot d)$ 매번 | $O(S \cdot d)$ 새 Q만 |
| 추가 메모리 | 없음 | $O(L \cdot H_{kv} \cdot d \cdot S)$ |
| 총 생성 FLOPs ($T$토큰) | $O(T \cdot S^2 \cdot d)$ | $O(T \cdot S \cdot d)$ |

---

### 🐣 초등학생을 위한 KV Cache 친절 설명!

#### 📝 KV Cache가 뭔가요?

> 💡 **비유**: 수학 시험에서 **앞 문제의 풀이를 메모장에 적어두는 것**과 같아요!
> 
> AI가 글을 쓸 때, "나는 오늘 학교에..."까지 쓰고 다음 단어를 예측하려면 앞 단어들의 정보(K,V)가 필요해요.
> **메모장 없으면**: 매번 앞 단어들을 처음부터 다시 계산 → 느림! 🐢
> **메모장 있으면**: 이전 계산을 기억하고 새 단어만 추가 계산 → 빠름! 🚀

| 상황 | 비유 | 속도 |
|------|------|------|
| Cache 없음 | 매번 1번 문제부터 다시 풀기 | 🐢🐢🐢 |
| Cache 있음 | 메모장 보고 마지막 문제만 풀기 | 🚀 |
| 메모장이 꽉 참 | 오래된 메모 지우기 (Rolling Buffer) | 🚀 (고정 메모리) |"""),

# ─── Cell 2: 📝 연습 문제 ───
md(r"""---

### 📝 연습 문제

#### 문제 1: KV Cache 메모리 계산

Llama 3 8B ($L=32, H_{kv}=8, d_{head}=128$)에서 배치 크기 $B=4$, 시퀀스 길이 $S=4096$일 때 FP16 KV 캐시 크기를 계산하시오.

<details>
<summary>💡 풀이 확인</summary>

$$M_{KV} = 2 \times 32 \times 8 \times 128 \times 4096 \times 4 \times 2 \text{ bytes}$$
$$= 2 \times 32 \times 8 \times 128 \times 4096 \times 4 \times 2$$
$$= 2,147,483,648 \text{ bytes} = 2.0 \text{ GB}$$

배치 1 기준 512 MB × 배치 4 = 2.0 GB
</details>

#### 문제 2: GQA vs MHA KV Cache 비교

동일 모델에서 MHA($H_{kv}=32$)를 사용했다면 KV Cache는 몇 GB인가? GQA 대비 몇 배인가?

<details>
<summary>💡 풀이 확인</summary>

$$M_{MHA} = 2 \times 32 \times 32 \times 128 \times 4096 \times 4 \times 2 = 8.0 \text{ GB}$$

$$\frac{M_{MHA}}{M_{GQA}} = \frac{32}{8} = 4\text{배}$$

GQA($H_{kv}=8$)가 MHA($H_{kv}=32$) 대비 **KV 메모리 75% 절감!**
</details>"""),

# ─── Cell 3: 임포트 ───
code("""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import time

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")"""),

# ─── Cell 4: 섹션2 ───
md("""## 2. Autoregressive 생성과 KV Cache <a name='2.-KV-Cache-원리'></a>"""),

# ─── Cell 5: KV Cache 원리 구현 ───
code(r"""# ── KV Cache 있/없이 Autoregressive 생성 비교 ─────────────────
# 간단한 Attention 연산에서 KV Cache의 효과를 측정합니다

d_model = 256
n_heads = 8
d_head = d_model // n_heads
seq_len = 64
gen_len = 32  # 생성할 토큰 수

# 랜덤 가중치 (데모용)
Wq = tf.random.normal((d_model, d_model))
Wk = tf.random.normal((d_model, d_model))
Wv = tf.random.normal((d_model, d_model))

# === 방법 1: KV Cache 없이 (매번 전체 재계산) ===
def generate_no_cache(prompt, gen_len):
    tokens = tf.identity(prompt)  # [1, prompt_len, d]
    flops_total = 0
    for step in range(gen_len):
        S = tokens.shape[1]
        Q = tokens @ Wq  # 전체 시퀀스 Q
        K = tokens @ Wk  # 전체 시퀀스 K (매번 재계산!)
        V = tokens @ Wv  # 전체 시퀀스 V (매번 재계산!)
        attn = tf.nn.softmax(Q @ tf.transpose(K, [0, 2, 1]) / tf.sqrt(float(d_model)))
        out = attn @ V
        new_token = out[:, -1:, :]  # 마지막 토큰의 출력
        tokens = tf.concat([tokens, new_token], axis=1)
        flops_total += S * S * d_model * 2  # 대략적 FLOPs
    return tokens, flops_total

# === 방법 2: KV Cache 사용 ===
def generate_with_cache(prompt, gen_len):
    B = prompt.shape[0]
    # 프롬프트 K,V 미리 계산 (Prefill)
    k_cache = prompt @ Wk  # [1, prompt_len, d]
    v_cache = prompt @ Wv
    tokens = tf.identity(prompt)
    flops_total = 0
    for step in range(gen_len):
        # 새 토큰의 Q만 계산
        q_new = tokens[:, -1:, :] @ Wq  # [1, 1, d]
        k_new = tokens[:, -1:, :] @ Wk  # [1, 1, d]
        v_new = tokens[:, -1:, :] @ Wv
        # 캐시에 추가
        k_cache = tf.concat([k_cache, k_new], axis=1)
        v_cache = tf.concat([v_cache, v_new], axis=1)
        S = k_cache.shape[1]
        # Attention: 새 Q × 전체 cached K
        attn = tf.nn.softmax(q_new @ tf.transpose(k_cache, [0, 2, 1]) / tf.sqrt(float(d_model)))
        out = attn @ v_cache  # [1, 1, d]
        tokens = tf.concat([tokens, out], axis=1)
        flops_total += S * d_model * 2  # S×d (캐시 방식: S^2 → S)
    return tokens, flops_total

prompt = tf.random.normal((1, seq_len, d_model))

# 워밍업
_, _ = generate_no_cache(prompt, 2)
_, _ = generate_with_cache(prompt, 2)

# 벤치마크
start = time.perf_counter()
_, flops_no = generate_no_cache(prompt, gen_len)
time_no = time.perf_counter() - start

start = time.perf_counter()
_, flops_with = generate_with_cache(prompt, gen_len)
time_with = time.perf_counter() - start

print("=" * 60)
print(f"Autoregressive 생성 비교 (프롬프트={seq_len}, 생성={gen_len}토큰)")
print("=" * 60)
print(f"{'방법':<25} | {'시간 (ms)':>12} | {'대략 FLOPs':>15}")
print("-" * 60)
print(f"{'KV Cache 없음':<25} | {time_no*1000:>12.1f} | {flops_no:>15,}")
print(f"{'KV Cache 있음':<25} | {time_with*1000:>12.1f} | {flops_with:>15,}")
print(f"{'속도 향상':<25} | {time_no/time_with:>11.1f}x | {flops_no/flops_with:>14.1f}x")
print()
print("KV Cache로 연산량 대폭 감소: 매 스텝 O(S^2) → O(S)")"""),

# ─── Cell 6: KV Cache 메모리 계산 ───
code(r"""# ── Llama 3 8B KV Cache 메모리 계산기 ─────────────────────────
# 실제 Llama 3 8B 파라미터로 다양한 시나리오의 메모리를 계산합니다

def kv_cache_memory_bytes(n_layers, n_kv_heads, d_head, seq_len, batch_size, dtype_bytes=2):
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch_size * dtype_bytes

# Llama 3 8B 파라미터
L, H_kv, d_h = 32, 8, 128

print("=" * 70)
print("Llama 3 8B KV Cache 메모리 계산 (FP16)")
print(f"  L={L}, H_kv={H_kv}, d_head={d_h}")
print("=" * 70)

seq_lengths = [512, 1024, 2048, 4096, 8192]
batch_sizes = [1, 4, 8, 16, 32]

# 표 헤더
header = f"{'S \\ B':<8}"
for b in batch_sizes:
    header += f" | {'B='+str(b):>8}"
print(header)
print("-" * (8 + 11 * len(batch_sizes)))

for s in seq_lengths:
    row = f"{s:<8}"
    for b in batch_sizes:
        mem = kv_cache_memory_bytes(L, H_kv, d_h, s, b)
        if mem >= 1e9:
            row += f" | {mem/1e9:>6.1f} GB"
        else:
            row += f" | {mem/1e6:>5.0f} MB "
        
    print(row)

print()
print("⚠️ 주의: 모델 가중치(~16GB FP16) + KV Cache + 활성화 메모리 합계가 GPU VRAM을 초과하면 OOM!")
print()

# 80GB A100 예시
model_weights_gb = 16  # Llama 3 8B FP16
vram_gb = 80
available_kv = vram_gb - model_weights_gb - 5  # 5GB 여유

print(f"A100 80GB 기준 가용 KV 메모리: ~{available_kv} GB")
max_batch_4096 = int(available_kv * 1e9 / kv_cache_memory_bytes(L, H_kv, d_h, 4096, 1))
print(f"  S=4096에서 최대 배치: ~{max_batch_4096}")
max_batch_8192 = int(available_kv * 1e9 / kv_cache_memory_bytes(L, H_kv, d_h, 8192, 1))
print(f"  S=8192에서 최대 배치: ~{max_batch_8192}")"""),

# ─── Cell 7: 섹션3 ───
md("""## 3. Rolling Buffer (Sliding Window) <a name='3.-Rolling-Buffer'></a>"""),

# ─── Cell 8: Rolling Buffer 구현 ───
code(r"""# ── Rolling Buffer KV Cache 구현 ──────────────────────────────
# 고정 크기 버퍼로 메모리를 제한하면서 최근 컨텍스트를 유지합니다

class RollingKVCache:
    # 고정 크기 Rolling Buffer KV Cache (Mistral 스타일)
    def __init__(self, max_size, d_model, n_layers=1):
        self.max_size = max_size
        self.d_model = d_model
        self.n_layers = n_layers
        # [n_layers, 2(K,V), max_size, d_model]
        self.buffer = np.zeros((n_layers, 2, max_size, d_model))
        self.write_pos = 0  # 다음 쓰기 위치 (circular)
        self.length = 0     # 현재 저장된 토큰 수

    def update(self, k_new, v_new, layer=0):
        # k_new, v_new: [1, d_model]
        pos = self.write_pos % self.max_size  # 순환 위치
        self.buffer[layer, 0, pos] = k_new
        self.buffer[layer, 1, pos] = v_new
        self.write_pos += 1
        self.length = min(self.length + 1, self.max_size)
        return pos

    def get_kv(self, layer=0):
        # 현재 저장된 K,V 반환 (순서 보정)
        if self.length < self.max_size:
            return self.buffer[layer, 0, :self.length], self.buffer[layer, 1, :self.length]
        else:
            # 순환 버퍼: 가장 오래된 것부터 순서대로
            start = self.write_pos % self.max_size
            indices = [(start + i) % self.max_size for i in range(self.max_size)]
            return self.buffer[layer, 0, indices], self.buffer[layer, 1, indices]

    def memory_bytes(self, dtype_bytes=2):
        return self.buffer.nbytes  # 실제 고정 메모리


# 시뮬레이션: Rolling Buffer vs 무제한 캐시
d = 128
window_size = 64  # 최근 64 토큰만 유지
total_tokens = 200

cache_rolling = RollingKVCache(max_size=window_size, d_model=d)
rolling_memory = []
unlimited_memory = []

for t in range(total_tokens):
    k_new = np.random.randn(d)
    v_new = np.random.randn(d)
    cache_rolling.update(k_new, v_new)
    
    rolling_memory.append(cache_rolling.length * d * 2 * 2)  # K+V, float16
    unlimited_memory.append((t + 1) * d * 2 * 2)

print("=" * 60)
print(f"Rolling Buffer KV Cache 시뮬레이션 (window={window_size})")
print("=" * 60)
print(f"총 생성 토큰: {total_tokens}")
print(f"Rolling 최종 메모리: {rolling_memory[-1]:,} bytes ({rolling_memory[-1]/1024:.1f} KB)")
print(f"무제한 최종 메모리: {unlimited_memory[-1]:,} bytes ({unlimited_memory[-1]/1024:.1f} KB)")
print(f"메모리 절감: {(1 - rolling_memory[-1]/unlimited_memory[-1])*100:.0f}%")
print()

# 저장된 토큰 확인
k_stored, v_stored = cache_rolling.get_kv()
print(f"Rolling Buffer 저장 토큰 수: {len(k_stored)} (최근 {window_size}개만)")"""),

# ─── Cell 9: Rolling Buffer 시각화 ───
code(r"""# ── Rolling Buffer vs 무제한 캐시 메모리 비교 시각화 ──────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: 메모리 사용량 비교
ax1 = axes[0]
tokens = np.arange(1, total_tokens + 1)
ax1.plot(tokens, np.array(unlimited_memory) / 1024, 'r-', lw=2.5, label='무제한 캐시')
ax1.plot(tokens, np.array(rolling_memory) / 1024, 'b-', lw=2.5, label=f'Rolling Buffer (W={window_size})')
ax1.axhline(y=window_size * d * 2 * 2 / 1024, color='blue', ls='--', lw=1.5, alpha=0.5)
ax1.fill_between(tokens, np.array(rolling_memory) / 1024, np.array(unlimited_memory) / 1024,
                  alpha=0.15, color='red', label='절약된 메모리')
ax1.set_xlabel('생성된 토큰 수', fontsize=11)
ax1.set_ylabel('KV Cache 메모리 (KB)', fontsize=11)
ax1.set_title('KV Cache 메모리 증가 패턴', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 오른쪽: Rolling Buffer 동작 원리 (circular)
ax2 = axes[1]
n_slots = 8  # 시각화용 작은 버퍼
positions = np.arange(n_slots)
current_write = 5  # 현재 쓰기 위치

colors_slot = ['#43A047' if i < current_write else '#E0E0E0' for i in range(n_slots)]
colors_slot[current_write % n_slots] = '#E53935'  # 다음 쓰기 위치

bars = ax2.bar(positions, [1]*n_slots, color=colors_slot, edgecolor='black', lw=1.5)

for i in range(n_slots):
    if i < current_write:
        age = current_write - i
        ax2.text(i, 0.5, f't-{age}', ha='center', va='center', fontsize=9, fontweight='bold')
    elif i == current_write:
        ax2.text(i, 0.5, '→next', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax2.set_xlabel('버퍼 슬롯', fontsize=11)
ax2.set_title('Rolling Buffer 순환 구조', fontweight='bold')
ax2.set_yticks([])
ax2.set_xticks(positions)

# 범례 패치
import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(facecolor='#43A047', label='캐시된 토큰'),
    mpatches.Patch(facecolor='#E53935', label='다음 쓰기 위치'),
    mpatches.Patch(facecolor='#E0E0E0', label='빈 슬롯'),
]
ax2.legend(handles=legend_elements, fontsize=9)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('chapter12_modern_llms/kv_cache_rolling_buffer.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter12_modern_llms/kv_cache_rolling_buffer.png")"""),

# ─── Cell 10: 섹션4 ───
md("""## 4. Multi-Turn 대화 메모리 분석 <a name='4.-Multi-Turn-분석'></a>"""),

# ─── Cell 11: Multi-Turn 시뮬레이션 ───
code(r"""# ── Multi-Turn 대화의 KV Cache 메모리 시뮬레이션 ──────────────
# 실제 대화 시나리오에서 턴마다 KV Cache가 어떻게 증가하는지 분석합니다

# Llama 3 8B 기준
L, H_kv, d_h = 32, 8, 128

def kv_memory_gb(seq_len, batch=1, dtype_bytes=2):
    return 2 * L * H_kv * d_h * seq_len * batch * dtype_bytes / 1e9

# 시나리오: 5턴 대화
turns = [
    ("시스템 프롬프트", 200),
    ("사용자 질문 1", 50),
    ("AI 응답 1", 300),
    ("사용자 질문 2", 80),
    ("AI 응답 2", 500),
    ("사용자 질문 3", 60),
    ("AI 응답 3", 800),
]

cumulative_tokens = 0
turn_data = []
print("=" * 70)
print("Multi-Turn 대화 KV Cache 메모리 분석 (Llama 3 8B, FP16, B=1)")
print("=" * 70)
print(f"{'턴':<25} | {'토큰':>6} | {'누적':>6} | {'KV 메모리':>10}")
print("-" * 70)

for name, tokens in turns:
    cumulative_tokens += tokens
    mem = kv_memory_gb(cumulative_tokens)
    turn_data.append((name, tokens, cumulative_tokens, mem))
    print(f"{name:<25} | {tokens:>6} | {cumulative_tokens:>6} | {mem*1000:>8.1f} MB")

print("-" * 70)
print(f"{'총계':<25} | {cumulative_tokens:>6} |        | {kv_memory_gb(cumulative_tokens)*1000:>8.1f} MB")
print()
print("배치 크기별 최종 메모리:")
for b in [1, 4, 8, 16]:
    mem = kv_memory_gb(cumulative_tokens, batch=b)
    print(f"  B={b:>2}: {mem:.3f} GB")"""),

# ─── Cell 12: Multi-Turn 시각화 ───
code(r"""# ── Multi-Turn KV Cache 메모리 증가 시각화 ─────────────────────

names = [d[0] for d in turn_data]
cumulative = [d[2] for d in turn_data]
memories = [d[3] * 1000 for d in turn_data]  # MB
tokens_per_turn = [d[1] for d in turn_data]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: 누적 KV Cache 메모리
ax1 = axes[0]
colors_turn = ['#1565C0', '#43A047', '#E53935', '#43A047', '#E53935', '#43A047', '#E53935']
labels_turn = ['System', 'User', 'AI', 'User', 'AI', 'User', 'AI']

ax1.bar(range(len(turn_data)), memories, color=colors_turn, edgecolor='black', lw=1)
ax1.set_xticks(range(len(turn_data)))
ax1.set_xticklabels([f'Turn {i}' for i in range(len(turn_data))], fontsize=8, rotation=30)
ax1.set_ylabel('KV Cache 메모리 (MB)', fontsize=11)
ax1.set_title('턴별 누적 KV Cache 메모리', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

for i, (m, l) in enumerate(zip(memories, labels_turn)):
    ax1.text(i, m + 2, f'{m:.0f}MB\n({l})', ha='center', fontsize=7)

# 오른쪽: 배치별 메모리 (최종 상태)
ax2 = axes[1]
batch_sizes = [1, 2, 4, 8, 16, 32]
final_seq = cumulative[-1]
batch_memories = [kv_memory_gb(final_seq, b) for b in batch_sizes]

ax2.plot(batch_sizes, batch_memories, 'r-o', lw=2.5, ms=8)
ax2.axhline(y=80, color='gray', ls='--', lw=1.5, label='A100 80GB VRAM')
ax2.axhline(y=24, color='orange', ls='--', lw=1.5, label='RTX 4090 24GB VRAM')
ax2.fill_between(batch_sizes, batch_memories, 0, alpha=0.1, color='red')
ax2.set_xlabel('배치 크기', fontsize=11)
ax2.set_ylabel('KV Cache 메모리 (GB)', fontsize=11)
ax2.set_title(f'배치별 KV Cache (S={final_seq})', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter12_modern_llms/kv_cache_multiturn.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter12_modern_llms/kv_cache_multiturn.png")"""),

# ─── Cell 13: 섹션5 ───
md("""## 5. Prefix Caching 시뮬레이션 <a name='5.-Prefix-Caching'></a>"""),

# ─── Cell 14: Prefix Caching ───
code(r"""# ── Prefix Caching 시뮬레이션 ─────────────────────────────────
# 동일한 시스템 프롬프트를 공유하는 다수의 요청에서 KV Cache를 재사용합니다

# 시나리오: 동일 시스템 프롬프트(500 토큰)로 100개의 서로 다른 사용자 요청 처리
system_prompt_tokens = 500
user_query_avg_tokens = 100
num_requests = 100

# Llama 3 8B 기준
def kv_prefill_flops(seq_len, d_model=4096, n_layers=32):
    # 대략적: 각 레이어에서 QKV projection + attention + FFN
    return n_layers * (3 * seq_len * d_model**2 + 2 * seq_len**2 * d_model)

# === 방법 1: Prefix Caching 없음 (매번 시스템 프롬프트 재계산) ===
flops_no_prefix = 0
for _ in range(num_requests):
    total_seq = system_prompt_tokens + user_query_avg_tokens
    flops_no_prefix += kv_prefill_flops(total_seq)

# === 방법 2: Prefix Caching (시스템 프롬프트 1회만 계산) ===
flops_prefix = kv_prefill_flops(system_prompt_tokens)  # 1회
for _ in range(num_requests):
    # 사용자 쿼리만 처리 (기존 prefix 캐시 재사용)
    flops_prefix += kv_prefill_flops(user_query_avg_tokens)

# KV 메모리도 비교
kv_per_request = kv_memory_gb(system_prompt_tokens + user_query_avg_tokens)
kv_shared_prefix = kv_memory_gb(system_prompt_tokens)  # 공유 부분
kv_per_query = kv_memory_gb(user_query_avg_tokens)

print("=" * 65)
print(f"Prefix Caching 효과 분석 (시스템 프롬프트 {system_prompt_tokens}토큰, {num_requests}개 요청)")
print("=" * 65)
print(f"{'항목':<30} | {'캐싱 없음':>15} | {'Prefix 캐싱':>15}")
print("-" * 65)
print(f"{'총 Prefill FLOPs':<30} | {flops_no_prefix:>15.2e} | {flops_prefix:>15.2e}")
print(f"{'FLOPs 절감':<30} | {'-':>15} | {(1-flops_prefix/flops_no_prefix)*100:>13.1f}%")
print(f"{'요청당 시스템프롬프트 재계산':<30} | {'매번':>15} | {'1회만':>15}")
print()
print("Prefix Caching 핵심 원리:")
print("  1. 시스템 프롬프트의 KV를 한 번 계산 후 GPU 메모리에 유지")
print("  2. 새 사용자 쿼리는 prefix KV를 복사(COW) 후 이어서 계산")
print("  3. vLLM, SGLang 등 서빙 프레임워크에서 자동 지원")"""),

# ─── Cell 15: 정리 ───
md(r"""## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| KV Cache | 이전 토큰의 K,V를 저장하여 재계산 방지 → 생성 속도 $O(S^2) \to O(S)$ | ⭐⭐⭐ |
| KV 메모리 공식 | $M_{KV} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot S \cdot B \cdot \text{bytes}$ | ⭐⭐⭐ |
| Rolling Buffer | 고정 크기 순환 버퍼로 메모리 상한 설정 → Mistral 스타일 | ⭐⭐ |
| Prefix Caching | 공통 접두사(시스템 프롬프트)의 KV를 공유 → 서빙 처리량 향상 | ⭐⭐⭐ |
| GQA + KV Cache | $H_{kv} \ll H_Q$로 캐시 메모리를 $G/H$배로 줄임 | ⭐⭐⭐ |

### 핵심 수식

$$M_{KV} = 2 \times L \times H_{kv} \times d_{head} \times S \times B \times \text{bytes}$$

Llama 3 8B ($L=32, H_{kv}=8, d_{head}=128$): $S=4096, B=1$ → **512 MB**

### 다음 챕터 예고
**Chapter 12-03: Rotary Position Embedding (RoPE)** — 복소수 평면에서의 위치 인코딩 수식 도출, 장거리 의존성 보존, YaRN을 이용한 컨텍스트 확장 원리를 다룹니다."""),
]

create_notebook(cells, 'chapter12_modern_llms/02_kv_cache_and_memory.ipynb')
