"""Generate chapter16_sparse_attention/02_multi_head_latent_attention.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 16: 최신 거대 모델의 효율성 — Multi-head Latent Attention (MLA)

## 학습 목표
- MLA의 KV 압축(down-projection)과 복원(up-projection) 수식을 완전 도출한다
- MHA, MQA, GQA, MLA의 KV Cache 크기를 정량적으로 비교한다
- MLA가 GQA 대비 달성하는 메모리 절감률을 수치로 분석한다
- 시퀀스 길이에 따른 KV Cache 메모리 변화를 시각화한다
- MLA의 attention quality 유지 메커니즘을 이해한다

## 목차
1. [수학적 기초: Attention과 KV 압축](#1.-수학적-기초)
2. [MLA Down/Up Projection 구현](#2.-MLA-Projection-구현)
3. [GQA vs MHA vs MLA 메모리 비교](#3.-메모리-비교)
4. [KV Cache 크기 계산](#4.-KV-Cache-크기-계산)
5. [Attention Quality 비교](#5.-Attention-Quality-비교)
6. [정리](#6.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### 표준 Multi-Head Attention (MHA) 복습

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_h}}\right)V$$

KV Cache 크기 (토큰당):
$$M_{KV}^{MHA} = 2 \times H \times d_h = 2 \times d_{model}$$

- $H$: 헤드 수, $d_h$: 헤드 차원, $d_{model} = H \times d_h$

### Grouped-Query Attention (GQA)

$$M_{KV}^{GQA} = 2 \times H_{kv} \times d_h$$

- $H_{kv} \ll H$: KV 헤드 수 (Q 헤드를 그룹으로 묶어 공유)
- 절감률: $H_{kv} / H$ (예: Llama 3 8B에서 $H=32, H_{kv}=8$ → 75% 절감)

### Multi-head Latent Attention (MLA) — DeepSeek-V2/V3

**Step 1: Down-projection (KV 압축)**

$$c_t^{KV} = W_d^{KV} h_t \in \mathbb{R}^{d_c}$$

- $W_d^{KV} \in \mathbb{R}^{d_c \times d_{model}}$: 압축 행렬
- $d_c \ll d_{model}$: 압축 차원 (예: DeepSeek-V2에서 $d_c = 512$, $d_{model} = 5120$)

**Step 2: Up-projection (KV 복원)**

$$[k_t^C;\; v_t^C] = W_u^{KV} c_t^{KV} \in \mathbb{R}^{2 \times H \times d_h}$$

- $W_u^{KV} \in \mathbb{R}^{(2Hd_h) \times d_c}$: 복원 행렬
- 복원된 $k_t^C, v_t^C$로 표준 attention 수행

**KV Cache 크기 (MLA, 토큰당):**

$$M_{KV}^{MLA} = d_c \quad \text{(오직 압축 벡터만 저장)}$$

**절감률:**

$$\text{절감률} = \frac{d_c}{2 \times H_{kv} \times d_h} \quad \text{(GQA 대비)}$$

**요약 표:**

| 방식 | KV Cache (토큰당) | 예시 ($d=5120, H=40, d_h=128$) |
|------|-------------------|--------------------------------|
| MHA | $2Hd_h = 2d_{model}$ | $10240$ |
| GQA ($H_{kv}=8$) | $2 H_{kv} d_h$ | $2048$ |
| MQA | $2d_h$ | $256$ |
| MLA ($d_c=512$) | $d_c$ | $512$ |

---"""))

# ── Cell 3: 🐣 friendly explanation ──────────────────────────────
cells.append(md("""\
### 🐣 초등학생을 위한 MLA 친절 설명!

#### 🔢 MLA가 뭔가요?

> 💡 **비유**: 도서관에 책(KV Cache)을 보관할 때를 생각해 보세요!

- **MHA**: 모든 책을 원본 그대로 보관 → 공간이 엄청나게 필요해요
- **GQA**: 비슷한 책끼리 묶어서 대표 한 권만 보관 → 공간 절약!
- **MLA**: 모든 책의 **요약본**(압축 벡터)만 보관하고, 필요할 때 원본을 복원해요 → 최고의 절약!

#### 📦 어떻게 압축하나요?

> 💡 **비유**: 5120개의 숫자를 512개로 압축하는 것은, 
> 긴 문장을 핵심 키워드로 요약하는 것과 같아요!

1. **압축(Down)**: $5120 \\rightarrow 512$ (10배 줄이기)
2. **저장**: 512개의 숫자만 KV Cache에 저장
3. **복원(Up)**: $512 \\rightarrow 5120$ (필요할 때 복원)

핵심은 **저장할 때만 작게, 사용할 때는 원래 크기로** 돌리는 거예요!

---"""))

# ── Cell 4: 📝 Exercises ──────────────────────────────────────────
cells.append(md(r"""\
### 📝 연습 문제

#### 문제 1: MLA 압축률 계산

DeepSeek-V2 기준: $d_{model}=5120$, $H=40$, $d_h=128$, $d_c=512$.
MLA의 MHA 대비 KV Cache 절감률(%)을 계산하세요.

<details>
<summary>💡 풀이 확인</summary>

$$M_{KV}^{MHA} = 2 \times H \times d_h = 2 \times 40 \times 128 = 10240$$

$$M_{KV}^{MLA} = d_c = 512$$

$$\text{절감률} = 1 - \frac{512}{10240} = 1 - 0.05 = 0.95 = 95\%$$

→ MLA는 MHA 대비 **95%** KV Cache 절감을 달성합니다!
</details>

#### 문제 2: GQA vs MLA 비교

$H_{kv}=8$ GQA와 $d_c=512$ MLA의 KV Cache 크기를 비교하세요 ($d_h=128$).

<details>
<summary>💡 풀이 확인</summary>

$$M_{KV}^{GQA} = 2 \times 8 \times 128 = 2048$$

$$M_{KV}^{MLA} = 512$$

$$\frac{M_{KV}^{MLA}}{M_{KV}^{GQA}} = \frac{512}{2048} = 0.25$$

→ MLA는 GQA 대비 **75% 추가 절감**을 달성합니다 (4배 더 작음).
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

# ── Cell 6: Section 2 - MLA Projection Implementation ───────────
cells.append(md(r"""\
## 2. MLA Down/Up Projection 구현 <a name='2.-MLA-Projection-구현'></a>

MLA의 핵심은 KV를 저차원 잠재 공간으로 압축하는 것입니다:

1. **Down-projection**: $c_t^{KV} = W_d^{KV} h_t$ ($d_{model} \rightarrow d_c$)
2. **Up-projection**: $[k_t^C; v_t^C] = W_u^{KV} c_t^{KV}$ ($d_c \rightarrow 2 H d_h$)

이를 TensorFlow로 구현합니다."""))

cells.append(code("""\
# ── MLA Down/Up Projection 구현 ──────────────────────────────────
class MLAProjection(tf.keras.layers.Layer):
    # Multi-head Latent Attention의 KV 압축/복원 레이어

    def __init__(self, d_model, n_heads, d_head, d_compress):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_compress = d_compress

        # Down-projection: d_model -> d_compress
        self.W_down = tf.keras.layers.Dense(d_compress, use_bias=False, name='kv_down')

        # Up-projection: d_compress -> 2 * n_heads * d_head (K + V)
        self.W_up = tf.keras.layers.Dense(2 * n_heads * d_head, use_bias=False, name='kv_up')

        # Q projection (별도)
        self.W_q = tf.keras.layers.Dense(n_heads * d_head, use_bias=False, name='q_proj')

    def call(self, h):
        batch, seq_len, _ = h.shape

        # Q projection
        q = self.W_q(h)
        q = tf.reshape(q, [batch, seq_len, self.n_heads, self.d_head])

        # KV Down-projection (압축)
        c_kv = self.W_down(h)  # [B, S, d_compress]

        # KV Up-projection (복원)
        kv = self.W_up(c_kv)  # [B, S, 2*H*d_h]
        kv = tf.reshape(kv, [batch, seq_len, 2, self.n_heads, self.d_head])
        k, v = kv[:, :, 0], kv[:, :, 1]

        return q, k, v, c_kv


# 파라미터 설정 (DeepSeek-V2 스케일 다운)
d_model = 512
n_heads = 8
d_head = 64
d_compress = 64  # 8배 압축

mla = MLAProjection(d_model, n_heads, d_head, d_compress)

# 테스트 입력
batch_size = 2
seq_len = 16
h = tf.random.normal([batch_size, seq_len, d_model])

q, k, v, c_kv = mla(h)

print(f"MLA Projection 결과:")
print(f"  입력 h: {h.shape}")
print(f"  Q: {q.shape}")
print(f"  K (복원): {k.shape}")
print(f"  V (복원): {v.shape}")
print(f"  압축 벡터 c_kv: {c_kv.shape}")
print(f"\\n압축률:")
print(f"  원본 KV 크기: {2 * n_heads * d_head} (= 2 x {n_heads} x {d_head})")
print(f"  압축 KV 크기: {d_compress}")
print(f"  압축률: {d_compress / (2 * n_heads * d_head):.2%} ({(2 * n_heads * d_head) / d_compress:.0f}x 압축)")

# 파라미터 수 비교
mla_params = d_model * d_compress + d_compress * (2 * n_heads * d_head)
mha_params = d_model * (2 * n_heads * d_head)  # K, V projection
print(f"\\n파라미터 수:")
print(f"  MLA (Down + Up): {mla_params:,}")
print(f"  MHA (K + V proj): {mha_params:,}")
print(f"  MLA 오버헤드: {mla_params/mha_params:.2f}x (추론 시 KV Cache는 {d_compress}만 저장)")"""))

# ── Cell 8: Section 3 - Memory Comparison ────────────────────────
cells.append(md(r"""\
## 3. GQA vs MHA vs MLA 메모리 비교 <a name='3.-메모리-비교'></a>

시퀀스 길이가 증가할 때 각 방식의 KV Cache 메모리를 비교합니다.

| 방식 | KV Cache 크기 (바이트/토큰/레이어) |
|------|----------------------------------|
| MHA | $2 \times H \times d_h \times \text{bytes}$ |
| GQA | $2 \times H_{kv} \times d_h \times \text{bytes}$ |
| MQA | $2 \times d_h \times \text{bytes}$ |
| MLA | $d_c \times \text{bytes}$ |"""))

cells.append(code("""\
# ── GQA vs MHA vs MLA 메모리 비교 (시퀀스 길이 축) ───────────────
# DeepSeek-V2 기준 파라미터
d_model_real = 5120
n_heads_real = 40
d_head_real = 128
n_layers = 60
bytes_per_elem = 2  # FP16

configs = {
    'MHA': {'kv_per_token': 2 * n_heads_real * d_head_real},
    'GQA (H_kv=8)': {'kv_per_token': 2 * 8 * d_head_real},
    'GQA (H_kv=4)': {'kv_per_token': 2 * 4 * d_head_real},
    'MQA': {'kv_per_token': 2 * d_head_real},
    'MLA (d_c=512)': {'kv_per_token': 512},
}

seq_lengths = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (1) 총 KV Cache 메모리 (GB) — 시퀀스 길이별
ax1 = axes[0]
colors = ['red', 'orange', 'blue', 'purple', 'green']
markers = ['o', 's', '^', 'D', 'v']

for (name, cfg), color, marker in zip(configs.items(), colors, markers):
    memory_gb = seq_lengths * n_layers * cfg['kv_per_token'] * bytes_per_elem / (1024**3)
    ax1.semilogy(seq_lengths / 1000, memory_gb, f'-{marker}',
                 color=color, lw=2, ms=7, label=name)

ax1.set_xlabel('시퀀스 길이 (K 토큰)', fontsize=11)
ax1.set_ylabel('KV Cache 메모리 (GB)', fontsize=11)
ax1.set_title('시퀀스 길이별 KV Cache 메모리', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=80, color='gray', ls=':', lw=1.5, alpha=0.7)
ax1.text(seq_lengths[-1]/1000*0.5, 85, 'H100 80GB', fontsize=9, color='gray')

# (2) MHA 대비 절감률
ax2 = axes[1]
mha_mem = configs['MHA']['kv_per_token']
for (name, cfg), color, marker in zip(list(configs.items())[1:], colors[1:], markers[1:]):
    savings = (1 - cfg['kv_per_token'] / mha_mem) * 100
    ax2.barh(name, savings, color=color, alpha=0.7, edgecolor='black')
    ax2.text(savings + 1, name, f'{savings:.1f}%', va='center', fontsize=10, fontweight='bold')

ax2.set_xlabel('MHA 대비 KV Cache 절감률 (%)', fontsize=11)
ax2.set_title('KV Cache 절감률 비교', fontweight='bold')
ax2.set_xlim(0, 105)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/mha_gqa_mla_memory.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter16_sparse_attention/mha_gqa_mla_memory.png")

# 수치 표
print(f"\\nKV Cache 메모리 비교 (시퀀스 길이 = 32K, {n_layers} 레이어, FP16):")
print(f"{'방식':<16} | {'토큰당 KV':>12} | {'메모리 (GB)':>12} | {'MHA 대비':>10}")
print("-" * 58)
for name, cfg in configs.items():
    mem_gb = 32768 * n_layers * cfg['kv_per_token'] * bytes_per_elem / (1024**3)
    ratio = cfg['kv_per_token'] / mha_mem
    print(f"{name:<16} | {cfg['kv_per_token']:>12} | {mem_gb:>12.2f} | {ratio:>9.2%}")"""))

# ── Cell 10: Section 4 - KV Cache calculation ───────────────────
cells.append(md(r"""\
## 4. KV Cache 크기 계산 <a name='4.-KV-Cache-크기-계산'></a>

실제 모델 파라미터를 사용한 KV Cache 크기 계산:

$$M_{KV} = B \times S \times L \times M_{per\_token\_per\_layer} \times \text{bytes}$$

- $B$: 배치 크기
- $S$: 시퀀스 길이
- $L$: 레이어 수"""))

cells.append(code("""\
# ── 실제 모델별 KV Cache 크기 계산기 ─────────────────────────────
models = {
    'Llama-3-8B': {
        'n_layers': 32, 'n_heads': 32, 'd_head': 128,
        'n_kv_heads': 8, 'method': 'GQA',
        'kv_per_token': lambda: 2 * 8 * 128
    },
    'Llama-3-70B': {
        'n_layers': 80, 'n_heads': 64, 'd_head': 128,
        'n_kv_heads': 8, 'method': 'GQA',
        'kv_per_token': lambda: 2 * 8 * 128
    },
    'DeepSeek-V2': {
        'n_layers': 60, 'n_heads': 128, 'd_head': 128,
        'd_compress': 512, 'method': 'MLA',
        'kv_per_token': lambda: 512
    },
    'DeepSeek-V3': {
        'n_layers': 61, 'n_heads': 128, 'd_head': 128,
        'd_compress': 512, 'method': 'MLA',
        'kv_per_token': lambda: 512
    },
}

batch_size = 1
seq_length = 4096
fp16_bytes = 2

print(f"모델별 KV Cache 크기 (B={batch_size}, S={seq_length}, FP16)")
print("=" * 80)
print(f"{'모델':<18} | {'방식':<6} | {'레이어':>6} | {'토큰당 KV':>10} | {'총 메모리':>12} | {'배치8':>12}")
print("-" * 80)

for name, cfg in models.items():
    kv = cfg['kv_per_token']()
    mem = batch_size * seq_length * cfg['n_layers'] * kv * fp16_bytes
    mem_gb = mem / (1024**3)
    mem_batch8 = mem_gb * 8
    print(f"{name:<18} | {cfg['method']:<6} | {cfg['n_layers']:>6} | {kv:>10} | {mem_gb:>10.3f} GB | {mem_batch8:>10.3f} GB")

# DeepSeek-V3 MLA vs 가상의 GQA 비교
print(f"\\nDeepSeek-V3: MLA vs 가상의 GQA 비교 (S={seq_length}):")
mla_kv = 512 * 61 * seq_length * fp16_bytes
gqa8_kv = (2 * 8 * 128) * 61 * seq_length * fp16_bytes
gqa4_kv = (2 * 4 * 128) * 61 * seq_length * fp16_bytes
print(f"  MLA (d_c=512):     {mla_kv / (1024**3):.3f} GB")
print(f"  GQA (H_kv=8):      {gqa8_kv / (1024**3):.3f} GB")
print(f"  GQA (H_kv=4):      {gqa4_kv / (1024**3):.3f} GB")
print(f"  MLA/GQA(8) 비율:   {mla_kv/gqa8_kv:.2%}")
print(f"  MLA/GQA(4) 비율:   {mla_kv/gqa4_kv:.2%}")"""))

# ── Cell 12: Section 5 - Attention quality ───────────────────────
cells.append(md(r"""\
## 5. Attention Quality 비교 <a name='5.-Attention-Quality-비교'></a>

MLA의 KV 압축이 attention 품질에 미치는 영향을 분석합니다.
압축 차원 $d_c$가 클수록 복원 품질이 높지만, 메모리 절약은 줄어듭니다.

$$\text{Reconstruction Error} = \frac{\|KV_{original} - W_u \cdot W_d \cdot h\|_F}{\|KV_{original}\|_F}$$"""))

cells.append(code("""\
# ── MLA 압축 차원별 Attention Quality 비교 ───────────────────────
d_model_test = 256
n_heads_test = 4
d_head_test = 64
seq_len_test = 64
batch_test = 4

# 원본 KV 생성 (MHA)
h_test = tf.random.normal([batch_test, seq_len_test, d_model_test])
W_kv_original = tf.keras.layers.Dense(2 * n_heads_test * d_head_test, use_bias=False)
kv_original = W_kv_original(h_test)

# 다양한 압축 차원에서 복원 품질 측정
compress_dims = [16, 32, 64, 128, 256, 512]
errors = []
savings = []

print(f"MLA 압축 차원별 복원 품질:")
print(f"{'d_c':>6} | {'복원 오차':>12} | {'KV Cache 절감률':>16} | {'Attn 유사도':>12}")
print("-" * 55)

for d_c in compress_dims:
    # Down-projection + Up-projection
    W_down = tf.keras.layers.Dense(d_c, use_bias=False)
    W_up = tf.keras.layers.Dense(2 * n_heads_test * d_head_test, use_bias=False)

    c_kv = W_down(h_test)
    kv_reconstructed = W_up(c_kv)

    # 복원 오차 (Frobenius norm)
    error = tf.norm(kv_original - kv_reconstructed) / tf.norm(kv_original)
    error_val = error.numpy()
    errors.append(error_val)

    # KV Cache 절감률
    original_size = 2 * n_heads_test * d_head_test
    saving = (1 - d_c / original_size) * 100
    savings.append(saving)

    # Attention 유사도 (코사인 유사도)
    kv_orig_flat = tf.reshape(kv_original, [-1, kv_original.shape[-1]])
    kv_recon_flat = tf.reshape(kv_reconstructed, [-1, kv_reconstructed.shape[-1]])
    cos_sim = tf.reduce_mean(
        tf.reduce_sum(kv_orig_flat * kv_recon_flat, axis=-1) /
        (tf.norm(kv_orig_flat, axis=-1) * tf.norm(kv_recon_flat, axis=-1) + 1e-8)
    ).numpy()

    print(f"{d_c:>6} | {error_val:>12.4f} | {saving:>14.1f}% | {cos_sim:>12.4f}")

# 시각화: 압축 차원 vs 복원 오차/절감률 트레이드오프
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

color1 = 'tab:blue'
ax1.set_xlabel('압축 차원 ($d_c$)', fontsize=11)
ax1.set_ylabel('복원 오차 (상대)', fontsize=11, color=color1)
ax1.plot(compress_dims, errors, 'b-o', lw=2.5, ms=8, label='복원 오차')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

ax2_twin = ax1.twinx()
color2 = 'tab:red'
ax2_twin.set_ylabel('KV Cache 절감률 (%)', fontsize=11, color=color2)
ax2_twin.plot(compress_dims, savings, 'r--s', lw=2, ms=7, label='절감률')
ax2_twin.tick_params(axis='y', labelcolor=color2)

ax1.set_title('MLA 압축 차원별 복원 품질 vs 메모리 절감', fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='center right')

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/mla_quality_tradeoff.png', dpi=100, bbox_inches='tight')
plt.close()
print("\\n그래프 저장됨: chapter16_sparse_attention/mla_quality_tradeoff.png")
print(f"\\n→ d_c가 커질수록 복원 오차 감소, 하지만 절감률도 감소")
print(f"→ DeepSeek-V2/V3은 d_c=512로 93.75% 절감과 높은 품질을 동시에 달성")"""))

# ── Cell 14: Summary ─────────────────────────────────────────────
cells.append(md(r"""\
## 6. 정리 <a name='6.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| MLA Down-projection | $c_t^{KV} = W_d^{KV} h_t$ — KV를 저차원으로 압축 | ⭐⭐⭐ |
| MLA Up-projection | $[k_t^C; v_t^C] = W_u^{KV} c_t^{KV}$ — KV 복원 | ⭐⭐⭐ |
| KV Cache 절감 | MLA: $d_c$ vs GQA: $2H_{kv}d_h$ | ⭐⭐⭐ |
| 압축 차원 선택 | $d_c$가 클수록 품질↑, 절감률↓ — 트레이드오프 | ⭐⭐ |
| MHA→GQA→MLA 진화 | 메모리 효율성의 단계적 발전 | ⭐⭐⭐ |
| DeepSeek-V2/V3 설정 | $d_c=512$, 95% MHA 대비 절감 | ⭐⭐ |

### 핵심 수식

$$c_t^{KV} = W_d^{KV} h_t \in \mathbb{R}^{d_c}, \quad d_c \ll d_{model}$$

$$[k_t^C;\; v_t^C] = W_u^{KV} c_t^{KV} \in \mathbb{R}^{2Hd_h}$$

$$\text{KV Cache 절감률} = 1 - \frac{d_c}{2 \times H \times d_h}$$

### 다음 챕터 예고
**03_linear_attention_and_hybrids.ipynb** — GLA, RetNet, Mamba 등 Linear Attention 계열 기법과 Qwen의 SWA+Full+Linear Hybrid 구조를 분석합니다."""))

create_notebook(cells, 'chapter16_sparse_attention/02_multi_head_latent_attention.ipynb')
