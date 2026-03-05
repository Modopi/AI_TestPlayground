"""Generate chapter14_extreme_inference/01_inference_bottlenecks.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 14: 극단적 추론 최적화 — 추론 병목 분석

## 학습 목표
- LLM 추론의 두 단계(Prefill / Decode)가 **물리적으로 다른 병목**을 갖는 원인을 이해한다
- Arithmetic Intensity(연산 강도)를 이용해 각 단계가 **Compute-bound / Memory-bound**인지 판별한다
- Roofline 모델을 LLM 추론에 적용하여 **하드웨어 활용률**을 시각화한다
- TTFT(Time To First Token)와 TPOT(Time Per Output Token)의 **수식을 도출**하고 측정한다
- 배치 크기(Batch Size)가 **처리량(Throughput)과 지연(Latency)**에 미치는 영향을 분석한다

## 목차
1. [수학적 기초: 연산 강도와 Roofline 모델](#1.-수학적-기초)
2. [Prefill vs Decode 단계 분석](#2.-Prefill-vs-Decode)
3. [Roofline 시각화](#3.-Roofline-시각화)
4. [TTFT / TPOT 측정 시뮬레이션](#4.-TTFT-/-TPOT)
5. [배치 크기와 처리량 관계](#5.-배치-크기와-처리량)
6. [정리](#6.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### Arithmetic Intensity (연산 강도)

하드웨어가 **연산 병목(Compute-bound)**인지 **메모리 병목(Memory-bound)**인지 판단하는 핵심 지표입니다:

$$\text{AI} = \frac{\text{FLOPs (연산량)}}{\text{Bytes Transferred (메모리 이동량)}}$$

- $\text{AI}$: Arithmetic Intensity (단위: FLOPs/Byte)
- $\text{FLOPs}$: 부동소수점 연산 횟수
- $\text{Bytes}$: HBM ↔ 연산 유닛 간 데이터 이동량

### Roofline 모델

GPU의 이론적 최대 성능을 두 가지 한계로 모델링합니다:

$$\text{Performance} = \min\left(\text{Peak FLOPs},\; \text{AI} \times \text{Memory Bandwidth}\right)$$

경계점(Ridge Point):

$$\text{AI}_{ridge} = \frac{\text{Peak FLOPs (FLOPS)}}{\text{Memory Bandwidth (Bytes/s)}}$$

- $\text{AI} < \text{AI}_{ridge}$: **Memory-bound** (메모리 대역폭이 병목)
- $\text{AI} > \text{AI}_{ridge}$: **Compute-bound** (연산 능력이 병목)

### Prefill vs Decode의 연산 강도

| 단계 | 연산 | AI | 병목 |
|------|------|-----|------|
| Prefill | $QK^T$ 행렬곱 ($S \times S$) | $\text{AI}_{prefill} \approx \frac{2 \cdot S \cdot d}{2d + 2S} \approx S$ | **Compute-bound** |
| Decode | 벡터-행렬 곱 ($1 \times S$) | $\text{AI}_{decode} \approx \frac{2d}{2d + 2S} \approx 1$ | **Memory-bound** |

- $S$: 시퀀스 길이
- $d$: 히든 차원

### TTFT와 TPOT

$$\text{TTFT} = \frac{2 \cdot P \cdot S_{input}}{\text{GPU FLOPs}} \quad \text{(Prefill 시간, Compute-bound)}$$

$$\text{TPOT} = \frac{2P \cdot \text{bytes\_per\_param}}{\text{Memory Bandwidth}} \quad \text{(Decode 한 토큰, Memory-bound)}$$

- $P$: 모델 파라미터 수
- $S_{input}$: 입력 시퀀스 길이
- $\text{bytes\_per\_param}$: 파라미터당 바이트 (FP16=2, INT8=1)

**요약 표:**

| 지표 | 수식 | 의미 |
|------|------|------|
| Arithmetic Intensity | $\text{FLOPs} / \text{Bytes}$ | 연산 대비 메모리 비율 |
| Ridge Point | $\text{Peak FLOPS} / \text{BW}$ | Compute↔Memory 경계 |
| TTFT | $2PS_{in} / \text{FLOPS}$ | 첫 토큰 생성 시간 |
| TPOT | $2P \cdot b / \text{BW}$ | 토큰 당 생성 시간 |"""))

# ── Cell 3: 🐣 friendly explanation ──────────────────────────────
cells.append(md("""\
### 🐣 초등학생을 위한 추론 병목 친절 설명!

#### 🔢 Prefill과 Decode가 뭔가요?

> 💡 **비유**: LLM이 답변하는 과정을 **시험**에 비유해 봅시다!

**Prefill(문제 읽기)**: 시험 문제를 쭉 읽는 단계예요. 문제가 길면 읽는 시간이 오래 걸리지만, 
눈(=GPU)은 계속 바쁘게 읽고 있어요. → **계산이 바쁜(Compute-bound)** 상태!

**Decode(답 쓰기)**: 한 글자씩 답을 쓰는 단계예요. 머릿속(=GPU)은 빠른데 
손(=메모리)이 느려서 한 글자씩만 쓸 수 있어요. → **메모리가 바쁜(Memory-bound)** 상태!

#### 🏔️ Roofline이 뭔가요?

> 💡 **비유**: 수도꼭지와 물통을 생각해 보세요!

- **수도꼭지** = 메모리 대역폭 (데이터를 가져오는 속도)
- **물통** = GPU 연산 유닛 (계산하는 속도)
- 물통이 아무리 커도, 수도꼭지가 좁으면 물(=데이터)이 천천히 차요 → Memory-bound
- 수도꼭지가 넓어도, 물통이 작으면 넘쳐요 → Compute-bound

---"""))

# ── Cell 4: 📝 Exercises ──────────────────────────────────────────
cells.append(md(r"""\
### 📝 연습 문제

#### 문제 1: Arithmetic Intensity 계산

행렬곱 $C = AB$에서 $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$일 때:
- FLOPs = $2MKN$
- Bytes = $2(MK + KN + MN)$ (FP16 기준)

$M=1, K=4096, N=4096$ (Decode 단계)일 때 AI를 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$$\text{FLOPs} = 2 \times 1 \times 4096 \times 4096 = 33,554,432$$

$$\text{Bytes} = 2(1 \times 4096 + 4096 \times 4096 + 1 \times 4096) = 2(4096 + 16,777,216 + 4096) = 33,570,816$$

$$\text{AI} = \frac{33,554,432}{33,570,816} \approx 1.0 \;\text{FLOPs/Byte}$$

→ AI ≈ 1로 **Memory-bound**! Decode 단계에서 가중치를 읽는 비용이 연산을 압도합니다.
</details>

#### 문제 2: TTFT 예측

Llama 3 8B ($P = 8 \times 10^9$), 입력 512 토큰, A100 GPU (312 TFLOPS FP16) 일 때 TTFT는?

<details>
<summary>💡 풀이 확인</summary>

$$\text{TTFT} = \frac{2 \times 8 \times 10^9 \times 512}{312 \times 10^{12}} = \frac{8.192 \times 10^{12}}{312 \times 10^{12}} \approx 26.3\text{ ms}$$

→ 약 26ms로, 입력 길이에 비례하여 증가합니다.
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
## 2. Prefill vs Decode 단계 분석 <a name='2.-Prefill-vs-Decode'></a>

Llama 3 8B의 실제 파라미터를 사용하여 각 단계의 FLOPs, 메모리 이동량, Arithmetic Intensity를 계산합니다.

| 파라미터 | 값 |
|---------|-----|
| 레이어 수 ($L$) | 32 |
| 히든 차원 ($d_{model}$) | 4096 |
| Q 헤드 수 ($n_q$) | 32 |
| KV 헤드 수 ($n_{kv}$) | 8 |
| 헤드 차원 ($d_{head}$) | 128 |
| FFN 차원 ($d_{ff}$) | 14336 |
| 총 파라미터 | ~8B |"""))

# ── Cell 7: Prefill vs Decode FLOPs/memory analysis ─────────────
cells.append(code("""\
# ── Prefill vs Decode FLOPs/메모리 분석 (Llama 3 8B) ─────────────
# Llama 3 8B 파라미터
L = 32          # 레이어 수
d_model = 4096  # 히든 차원
n_q = 32        # Q 헤드 수
n_kv = 8        # KV 헤드 수
d_head = 128    # 헤드 차원
d_ff = 14336    # FFN 차원
P = 8e9         # 총 파라미터

seq_lengths = [128, 256, 512, 1024, 2048, 4096]
batch_size = 1

print(f"{'':=<70}")
print(f"  Llama 3 8B: Prefill vs Decode 연산 분석")
print(f"{'':=<70}")
print(f"{'Seq Len':>8} | {'Prefill FLOPs':>15} | {'Decode FLOPs':>15} | {'AI(Prefill)':>12} | {'AI(Decode)':>11}")
print(f"{'-'*70}")

prefill_ais = []
decode_ais = []

for S in seq_lengths:
    # Prefill: 전체 시퀀스를 한 번에 처리 (행렬-행렬 곱)
    # Attention: Q*K^T + Attn*V → 2 * n_q * S * S * d_head per layer
    attn_flops_prefill = L * 2 * n_q * S * S * d_head * 2
    # Linear projections: QKV + Output → 각 d_model * d_model * S * 2
    qkv_flops = L * 2 * S * d_model * (d_model + 2 * n_kv * d_head) * 2
    # FFN: gate + up + down (SwiGLU)
    ffn_flops = L * 2 * S * d_model * d_ff * 3 * 2
    prefill_flops = attn_flops_prefill + qkv_flops + ffn_flops

    # 메모리: 모든 가중치 읽기 + 활성화
    weight_bytes = 2 * P * 2  # FP16, 읽기 1회
    activation_bytes = 2 * L * S * d_model * 2
    prefill_bytes = weight_bytes + activation_bytes

    # Decode: 토큰 1개씩 (벡터-행렬 곱)
    attn_flops_decode = L * 2 * n_q * 1 * S * d_head * 2
    qkv_flops_d = L * 2 * 1 * d_model * (d_model + 2 * n_kv * d_head) * 2
    ffn_flops_d = L * 2 * 1 * d_model * d_ff * 3 * 2
    decode_flops = attn_flops_decode + qkv_flops_d + ffn_flops_d

    # Decode 메모리: 가중치 전체 + KV cache 읽기
    kv_cache_bytes = 2 * L * n_kv * d_head * S * 2 * 2  # K,V * layers * FP16
    decode_bytes = weight_bytes + kv_cache_bytes

    ai_prefill = prefill_flops / prefill_bytes
    ai_decode = decode_flops / decode_bytes
    prefill_ais.append(ai_prefill)
    decode_ais.append(ai_decode)

    print(f"{S:>8} | {prefill_flops:>15.2e} | {decode_flops:>15.2e} | {ai_prefill:>12.1f} | {ai_decode:>11.1f}")

print(f"\\n결론:")
print(f"  Prefill AI: {min(prefill_ais):.0f} ~ {max(prefill_ais):.0f} (시퀀스 길이에 비례 증가)")
print(f"  Decode AI:  {min(decode_ais):.1f} ~ {max(decode_ais):.1f} (항상 낮음 → Memory-bound)")"""))

# ── Cell 8: Section 3 header ─────────────────────────────────────
cells.append(md("""\
## 3. Roofline 시각화 <a name='3.-Roofline-시각화'></a>

A100 GPU 스펙을 기준으로 Roofline 모델을 그리고, Prefill/Decode 연산 지점을 표시합니다.

| A100 스펙 | 값 |
|-----------|-----|
| FP16 Peak | 312 TFLOPS |
| HBM 대역폭 | 2.0 TB/s |
| Ridge Point | 156 FLOPs/Byte |"""))

# ── Cell 9: Roofline visualization ──────────────────────────────
cells.append(code("""\
# ── Roofline 모델 시각화 ─────────────────────────────────────────
peak_flops = 312e12   # A100 FP16: 312 TFLOPS
mem_bw = 2.0e12       # A100 HBM: 2.0 TB/s
ridge_point = peak_flops / mem_bw  # 156 FLOPs/Byte

ai_range = np.logspace(-1, 4, 500)
roofline = np.minimum(peak_flops, ai_range * mem_bw)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.loglog(ai_range, roofline / 1e12, 'b-', lw=3, label='Roofline (A100)')
ax.axvline(x=ridge_point, color='gray', ls='--', lw=1.5, alpha=0.7, label=f'Ridge Point = {ridge_point:.0f}')

seq_labels = [128, 512, 2048, 4096]
colors_p = ['#2196F3', '#1976D2', '#0D47A1', '#0A3069']
colors_d = ['#FF9800', '#F57C00', '#E65100', '#BF360C']

for i, S in enumerate(seq_labels):
    idx = seq_lengths.index(S)
    ai_p = prefill_ais[idx]
    perf_p = min(peak_flops, ai_p * mem_bw)
    ax.plot(ai_p, perf_p / 1e12, 'o', ms=12, color=colors_p[i],
            label=f'Prefill S={S}', zorder=5)

    ai_d = decode_ais[idx]
    perf_d = min(peak_flops, ai_d * mem_bw)
    ax.plot(ai_d, perf_d / 1e12, 's', ms=10, color=colors_d[i],
            label=f'Decode S={S}', zorder=5)

ax.fill_between([0.1, ridge_point], [0.0001, 0.0001], [1000, 1000],
                alpha=0.05, color='orange')
ax.fill_between([ridge_point, 10000], [0.0001, 0.0001], [1000, 1000],
                alpha=0.05, color='blue')
ax.text(2, 200, 'Memory\\nBound', fontsize=14, color='orange', fontweight='bold', alpha=0.6)
ax.text(1500, 200, 'Compute\\nBound', fontsize=14, color='blue', fontweight='bold', alpha=0.6)

ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=11)
ax.set_ylabel('Performance (TFLOPS)', fontsize=11)
ax.set_title('Roofline Model: LLM Prefill vs Decode (A100 GPU)', fontweight='bold')
ax.legend(fontsize=8, loc='lower right', ncol=2)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0.1, 10000)
ax.set_ylim(0.1, 500)

plt.tight_layout()
plt.savefig('chapter14_extreme_inference/roofline_prefill_decode.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter14_extreme_inference/roofline_prefill_decode.png")
print(f"\\nRidge Point: {ridge_point:.0f} FLOPs/Byte")
print(f"Prefill → Compute-bound 영역 (AI >> Ridge)")
print(f"Decode  → Memory-bound 영역 (AI << Ridge)")"""))

# ── Cell 10: Section 4 header ────────────────────────────────────
cells.append(md(r"""\
## 4. TTFT / TPOT 측정 시뮬레이션 <a name='4.-TTFT-/-TPOT'></a>

실제 GPU 없이도 **이론적 TTFT와 TPOT를 계산**하여 추론 시간을 예측합니다.

$$\text{TTFT} \approx \frac{2 \cdot P \cdot S_{input}}{\text{GPU FLOPS}}, \quad \text{TPOT} \approx \frac{2P \cdot \text{bytes\_per\_param}}{\text{Memory BW}}$$"""))

# ── Cell 11: TTFT/TPOT measurement simulation ───────────────────
cells.append(code("""\
# ── TTFT / TPOT 이론 계산 시뮬레이션 ────────────────────────────
# GPU 스펙
gpus = {
    'A100 (FP16)': {'flops': 312e12, 'bw': 2.0e12},
    'H100 (FP16)': {'flops': 989e12, 'bw': 3.35e12},
    'H200 (FP16)': {'flops': 989e12, 'bw': 4.8e12},
}

P = 8e9
bytes_per_param = 2  # FP16
input_lengths = [128, 256, 512, 1024, 2048]
output_length = 128

print(f"{'':=<80}")
print(f"  Llama 3 8B 추론 시간 예측 (TTFT + TPOT)")
print(f"{'':=<80}")

for gpu_name, specs in gpus.items():
    flops = specs['flops']
    bw = specs['bw']

    tpot = (2 * P * bytes_per_param) / bw * 1000  # ms

    print(f"\\n  GPU: {gpu_name}")
    print(f"  {'Input Len':>10} | {'TTFT (ms)':>10} | {'TPOT (ms)':>10} | {'Total 128tok (ms)':>18} | {'tok/s':>8}")
    print(f"  {'-'*65}")

    for S_in in input_lengths:
        ttft = (2 * P * S_in) / flops * 1000  # ms
        total = ttft + tpot * output_length
        tps = output_length / (total / 1000)

        print(f"  {S_in:>10} | {ttft:>10.1f} | {tpot:>10.2f} | {total:>18.1f} | {tps:>8.1f}")

print(f"\\n핵심 인사이트:")
print(f"  1. TTFT는 입력 길이에 비례 (Compute-bound)")
print(f"  2. TPOT는 입력 길이와 무관 (Memory-bound, 가중치 읽기 시간)")
print(f"  3. H200의 높은 대역폭(4.8TB/s)이 TPOT를 크게 개선")"""))

# ── Cell 12: Section 5 header ────────────────────────────────────
cells.append(md("""\
## 5. 배치 크기와 처리량 관계 <a name='5.-배치-크기와-처리량'></a>

배치 크기를 늘리면 **동일한 가중치 읽기**로 여러 요청을 처리하므로 처리량이 향상됩니다.
그러나 KV Cache 메모리가 배치에 비례하여 증가하므로 GPU 메모리가 한계점이 됩니다."""))

# ── Cell 13: Batch size effect on throughput ─────────────────────
cells.append(code("""\
# ── 배치 크기 vs 처리량/지연 분석 ────────────────────────────────
# A100 80GB 기준
gpu_mem = 80e9   # 80 GB
peak_flops = 312e12
mem_bw = 2.0e12
P = 8e9

L, n_kv, d_head = 32, 8, 128
S_max = 2048
bytes_per_param = 2

model_mem = P * bytes_per_param
print(f"모델 가중치 메모리: {model_mem / 1e9:.1f} GB")

batch_sizes = list(range(1, 65))
throughputs = []
latencies = []
kv_mems = []

for B in batch_sizes:
    kv_cache_per_token = 2 * L * n_kv * d_head * bytes_per_param
    kv_cache = B * S_max * kv_cache_per_token
    total_mem = model_mem + kv_cache

    if total_mem > gpu_mem:
        throughputs.append(None)
        latencies.append(None)
        kv_mems.append(kv_cache / 1e9)
        continue

    kv_mems.append(kv_cache / 1e9)

    # Decode: 가중치 1회 읽기 + KV cache 읽기로 B개 토큰 동시 생성
    weight_read_time = (P * bytes_per_param) / mem_bw
    kv_read_time = (kv_cache) / mem_bw
    compute_time = (2 * P * B) / peak_flops

    step_time = max(weight_read_time + kv_read_time, compute_time)
    latency = step_time * 1000  # ms per step
    throughput = B / step_time  # tokens/s

    throughputs.append(throughput)
    latencies.append(latency)

max_batch = max(i+1 for i, t in enumerate(throughputs) if t is not None)
print(f"최대 배치 크기 (A100 80GB, S={S_max}): {max_batch}")
print(f"\\n{'Batch':>6} | {'Throughput':>12} | {'Latency':>10} | {'KV Cache':>10} | {'Total Mem':>10}")
print(f"{'-'*58}")

for i, B in enumerate(batch_sizes):
    if B in [1, 2, 4, 8, 16, 32, max_batch]:
        if throughputs[i] is not None:
            total = model_mem / 1e9 + kv_mems[i]
            print(f"{B:>6} | {throughputs[i]:>10.0f} t/s | {latencies[i]:>8.2f} ms | {kv_mems[i]:>8.1f} GB | {total:>8.1f} GB")

print(f"\\n결론:")
print(f"  배치 증가 → 처리량 증가 (가중치 읽기를 공유)")
print(f"  배치 증가 → KV Cache 메모리 증가 → GPU 메모리 한계")"""))

# ── Cell 14: Batch throughput visualization ──────────────────────
cells.append(code("""\
# ── 배치 크기 vs 처리량 시각화 ───────────────────────────────────
valid_b = [b for b, t in zip(batch_sizes, throughputs) if t is not None]
valid_t = [t for t in throughputs if t is not None]
valid_l = [l for l in latencies if l is not None]
valid_kv = [kv_mems[i] for i, t in enumerate(throughputs) if t is not None]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax1 = axes[0]
ax1.plot(valid_b, valid_t, 'b-o', lw=2, ms=4, label='Throughput')
ax1.set_xlabel('Batch Size', fontsize=11)
ax1.set_ylabel('Throughput (tokens/s)', fontsize=11)
ax1.set_title('Batch Size vs Throughput', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

ax2 = axes[1]
ax2.plot(valid_b, valid_l, 'r-s', lw=2, ms=4, label='Latency per step')
ax2.set_xlabel('Batch Size', fontsize=11)
ax2.set_ylabel('Latency (ms)', fontsize=11)
ax2.set_title('Batch Size vs Latency', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

ax3 = axes[2]
total_mems = [model_mem / 1e9 + kv for kv in valid_kv]
ax3.bar(valid_b, [model_mem / 1e9] * len(valid_b), color='steelblue', label='Model Weights')
ax3.bar(valid_b, valid_kv, bottom=[model_mem / 1e9] * len(valid_b), color='coral', label='KV Cache')
ax3.axhline(y=80, color='red', ls='--', lw=2, label='A100 80GB Limit')
ax3.set_xlabel('Batch Size', fontsize=11)
ax3.set_ylabel('Memory (GB)', fontsize=11)
ax3.set_title('Memory Breakdown', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

plt.tight_layout()
plt.savefig('chapter14_extreme_inference/batch_throughput_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter14_extreme_inference/batch_throughput_analysis.png")
print(f"\\n최대 처리량: {max(valid_t):.0f} tokens/s (Batch={valid_b[valid_t.index(max(valid_t))]})") """))

# ── Cell 15: Summary ─────────────────────────────────────────────
cells.append(md(r"""\
## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| Arithmetic Intensity | FLOPs / Bytes — 연산 vs 메모리 병목 판별 | ⭐⭐⭐ |
| Prefill (Compute-bound) | 전체 시퀀스 행렬곱, AI ∝ S | ⭐⭐⭐ |
| Decode (Memory-bound) | 한 토큰씩 벡터-행렬곱, AI ≈ 1 | ⭐⭐⭐ |
| Roofline Model | Peak FLOPS와 Bandwidth로 성능 상한 모델링 | ⭐⭐⭐ |
| TTFT | 첫 토큰 시간, 입력 길이에 비례 | ⭐⭐ |
| TPOT | 토큰 당 시간, 메모리 대역폭에 반비례 | ⭐⭐ |
| Batch Size 효과 | 처리량↑, KV Cache 메모리↑ | ⭐⭐⭐ |

### 핵심 수식

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}, \quad \text{Performance} = \min(\text{Peak FLOPS}, \text{AI} \times \text{BW})$$

$$\text{TTFT} = \frac{2PS_{in}}{\text{GPU FLOPS}}, \quad \text{TPOT} = \frac{2P \cdot b}{\text{Memory BW}}$$

### 다음 챕터 예고
**02_flash_attention_deepdive.ipynb** — FlashAttention의 IO 복잡도 수식, Tiling + Recomputation 원리, v1→v2→v3 성능 발전사를 심층 분석합니다."""))

create_notebook(cells, 'chapter14_extreme_inference/01_inference_bottlenecks.ipynb')
