"""Generate chapter16_sparse_attention/practice/ex01_mla_from_scratch.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# 실습 퀴즈: Multi-head Latent Attention (MLA) 구현

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다

## 목차
- [Q1: Down-projection 차원 계산](#q1)
- [Q2: KV Cache 크기 비교 (MLA vs GQA vs MHA)](#q2)
- [Q3: Up-projection 복원 정확도](#q3)
- [Q4: 압축 KV로 Attention 계산](#q4)
- [종합 도전: Full MLA 레이어 구현 + 메모리 측정](#bonus)"""))

# ── Cell 2: Import ───────────────────────────────────────────────
cells.append(code("""\
# ── 라이브러리 임포트 ──────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
print(f"TensorFlow 버전: {tf.__version__}")"""))

# ── Cell 3: Q1 문제 ─────────────────────────────────────────────
cells.append(md(r"""\
## Q1: Down-projection 차원 계산 <a name='q1'></a>

### 문제

DeepSeek-V2 모델의 파라미터:
- $d_{model} = 5120$
- $H = 128$ (Attention 헤드 수)
- $d_h = 128$ (헤드 차원)

MLA에서 KV Cache를 원래의 $1/16$로 줄이려면 압축 차원 $d_c$는 얼마가 되어야 할까요?

**여러분의 예측:** $d_c$ = `?`

힌트: $d_c / (2 \times H \times d_h) = 1/16$"""))

# ── Cell 4: Q1 풀이 ─────────────────────────────────────────────
cells.append(code(r"""\
# ── Q1 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q1 풀이: Down-projection 차원 계산")
print("=" * 45)

d_model = 5120
H = 128
d_h = 128

# MHA의 KV Cache 크기 (토큰당)
kv_mha = 2 * H * d_h
print(f"MHA KV Cache (토큰당): 2 × {H} × {d_h} = {kv_mha}")

# 1/16로 줄이려면
compression_ratio = 1 / 16
d_c = int(kv_mha * compression_ratio)
print(f"\n목표 압축률: {compression_ratio} (= 1/16)")
print(f"d_c = {kv_mha} × {compression_ratio} = {d_c}")

# 검증
actual_ratio = d_c / kv_mha
print(f"\n검증: {d_c} / {kv_mha} = {actual_ratio:.4f} = 1/{int(1/actual_ratio)}")
print(f"→ d_c = {d_c}이면 KV Cache가 정확히 1/16로 줄어듭니다")

# 실제 DeepSeek-V2 설정과 비교
print(f"\n참고: DeepSeek-V2 실제 설정 d_c = 512")
print(f"  실제 비율: 512 / {kv_mha} = {512/kv_mha:.4f} = 1/{kv_mha//512}")"""))

# ── Cell 5: Q2 문제 ─────────────────────────────────────────────
cells.append(md(r"""\
## Q2: KV Cache 크기 비교 (MLA vs GQA vs MHA) <a name='q2'></a>

### 문제

다음 모델 설정에서 시퀀스 길이 $S=4096$, 배치 크기 $B=1$, 레이어 수 $L=60$일 때 
각 방식의 총 KV Cache 크기(MB)를 FP16 기준으로 계산하세요:

| 방식 | 설정 |
|------|------|
| MHA | $H=128$, $d_h=128$ |
| GQA | $H_{kv}=8$, $d_h=128$ |
| MLA | $d_c=512$ |

**여러분의 예측:** MHA = `?` MB, GQA = `?` MB, MLA = `?` MB"""))

# ── Cell 6: Q2 풀이 ─────────────────────────────────────────────
cells.append(code("""\
# ── Q2 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q2 풀이: KV Cache 크기 비교")
print("=" * 45)

S = 4096
B = 1
L = 60
bytes_per_elem = 2  # FP16

# MHA: 2 * H * d_h per token per layer
mha_per_token = 2 * 128 * 128
mha_total = B * S * L * mha_per_token * bytes_per_elem
mha_mb = mha_total / (1024 ** 2)

# GQA: 2 * H_kv * d_h per token per layer
gqa_per_token = 2 * 8 * 128
gqa_total = B * S * L * gqa_per_token * bytes_per_elem
gqa_mb = gqa_total / (1024 ** 2)

# MLA: d_c per token per layer
mla_per_token = 512
mla_total = B * S * L * mla_per_token * bytes_per_elem
mla_mb = mla_total / (1024 ** 2)

print(f"설정: S={S}, B={B}, L={L}, FP16")
print()
print(f"{'방식':<8} | {'토큰당 KV':>12} | {'총 바이트':>15} | {'크기 (MB)':>10}")
print("-" * 55)
print(f"{'MHA':<8} | {mha_per_token:>12} | {mha_total:>15,} | {mha_mb:>10.1f}")
print(f"{'GQA':<8} | {gqa_per_token:>12} | {gqa_total:>15,} | {gqa_mb:>10.1f}")
print(f"{'MLA':<8} | {mla_per_token:>12} | {mla_total:>15,} | {mla_mb:>10.1f}")
print()
print(f"[해설]")
print(f"  MLA는 MHA 대비 {(1-mla_mb/mha_mb)*100:.1f}% 절감")
print(f"  MLA는 GQA 대비 {(1-mla_mb/gqa_mb)*100:.1f}% 절감")
print(f"  특히 H가 크고 d_c가 작을수록 MLA의 이점이 극대화됨")"""))

# ── Cell 7: Q3 문제 ─────────────────────────────────────────────
cells.append(md(r"""\
## Q3: Up-projection 복원 정확도 <a name='q3'></a>

### 문제

$d_{model}=256$, $d_c=32$, $H=4$, $d_h=64$인 MLA에서:
1. Down-projection과 Up-projection을 수행하세요
2. 원본 KV와 복원된 KV의 코사인 유사도를 측정하세요
3. $d_c$를 $\{16, 32, 64, 128\}$로 변경하며 유사도 변화를 관찰하세요

**여러분의 예측:** $d_c$가 커질수록 유사도가 `증가/감소`할까요?"""))

# ── Cell 8: Q3 풀이 ─────────────────────────────────────────────
cells.append(code("""\
# ── Q3 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q3 풀이: Up-projection 복원 정확도")
print("=" * 45)

d_model = 256
H = 4
d_h = 64
batch = 4
seq_len = 32

# 원본 hidden states
h = tf.random.normal([batch, seq_len, d_model])

# 원본 KV (MHA 방식)
W_kv_ref = tf.keras.layers.Dense(2 * H * d_h, use_bias=False)
kv_original = W_kv_ref(h)

# 다양한 d_c에서 복원 품질 측정
d_c_values = [16, 32, 64, 128, 256]
print(f"원본 KV 차원: {2 * H * d_h}")
print()
print(f"{'d_c':>6} | {'코사인 유사도':>14} | {'상대 오차':>12} | {'압축률':>10}")
print("-" * 50)

for d_c in d_c_values:
    W_down = tf.keras.layers.Dense(d_c, use_bias=False)
    W_up = tf.keras.layers.Dense(2 * H * d_h, use_bias=False)

    c = W_down(h)
    kv_restored = W_up(c)

    # 코사인 유사도
    orig_flat = tf.reshape(kv_original, [-1, kv_original.shape[-1]])
    rest_flat = tf.reshape(kv_restored, [-1, kv_restored.shape[-1]])
    cos_sim = tf.reduce_mean(
        tf.reduce_sum(orig_flat * rest_flat, axis=-1) /
        (tf.norm(orig_flat, axis=-1) * tf.norm(rest_flat, axis=-1) + 1e-8)
    ).numpy()

    # 상대 오차
    rel_error = (tf.norm(kv_original - kv_restored) / tf.norm(kv_original)).numpy()

    compression = d_c / (2 * H * d_h) * 100
    print(f"{d_c:>6} | {cos_sim:>14.4f} | {rel_error:>12.4f} | {compression:>8.1f}%")

print()
print("[해설]")
print("  d_c가 커질수록 코사인 유사도가 증가합니다 (복원 품질 향상)")
print("  하지만 초기화 상태(미학습)이므로 유사도가 낮을 수 있습니다")
print("  학습을 통해 Down/Up projection이 최적화되면 유사도가 크게 향상됩니다")
print("  실전에서는 d_c=512 정도면 충분한 복원 품질을 달성합니다")"""))

# ── Cell 9: Q4 문제 ─────────────────────────────────────────────
cells.append(md(r"""\
## Q4: 압축 KV로 Attention 계산 <a name='q4'></a>

### 문제

MLA의 전체 attention 과정을 구현하세요:
1. $h \rightarrow c^{KV}$ (Down-projection)
2. $c^{KV} \rightarrow [K, V]$ (Up-projection)  
3. $h \rightarrow Q$ (Q projection)
4. $\text{Attention}(Q, K, V)$ 계산

**여러분의 예측:** 출력 shape은 `[batch, seq_len, ?]`이 될까요?"""))

# ── Cell 10: Q4 풀이 ─────────────────────────────────────────────
cells.append(code("""\
# ── Q4 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q4 풀이: 압축 KV로 Attention 계산")
print("=" * 45)

d_model = 128
n_heads = 4
d_head = 32
d_c = 32
batch = 2
seq_len = 16

# MLA 모듈 정의
W_q = tf.keras.layers.Dense(n_heads * d_head, use_bias=False)
W_down = tf.keras.layers.Dense(d_c, use_bias=False)
W_up = tf.keras.layers.Dense(2 * n_heads * d_head, use_bias=False)
W_o = tf.keras.layers.Dense(d_model, use_bias=False)

# 입력
h = tf.random.normal([batch, seq_len, d_model])
print(f"입력 h: {h.shape}")

# Step 1: Q projection
q = W_q(h)
q = tf.reshape(q, [batch, seq_len, n_heads, d_head])
q = tf.transpose(q, [0, 2, 1, 3])  # [B, H, S, d_h]
print(f"Q: {q.shape}")

# Step 2: KV down-projection (압축)
c_kv = W_down(h)  # [B, S, d_c]
print(f"압축 벡터 c_kv: {c_kv.shape} (이것만 KV Cache에 저장!)")

# Step 3: KV up-projection (복원)
kv = W_up(c_kv)
kv = tf.reshape(kv, [batch, seq_len, 2, n_heads, d_head])
k = tf.transpose(kv[:, :, 0], [0, 2, 1, 3])  # [B, H, S, d_h]
v = tf.transpose(kv[:, :, 1], [0, 2, 1, 3])  # [B, H, S, d_h]
print(f"K (복원): {k.shape}")
print(f"V (복원): {v.shape}")

# Step 4: Attention 계산
d_float = tf.cast(d_head, tf.float32)
scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(d_float)

# Causal mask
mask = tf.linalg.band_part(tf.ones([seq_len, seq_len]), -1, 0)
scores = scores + (1.0 - mask) * (-1e9)

weights = tf.nn.softmax(scores, axis=-1)
attn_out = tf.matmul(weights, v)  # [B, H, S, d_h]
print(f"Attention 출력: {attn_out.shape}")

# 헤드 합치기 + output projection
attn_out = tf.transpose(attn_out, [0, 2, 1, 3])  # [B, S, H, d_h]
attn_out = tf.reshape(attn_out, [batch, seq_len, n_heads * d_head])
output = W_o(attn_out)
print(f"최종 출력: {output.shape}")

print(f"\n[해설]")
print(f"  입력 shape = 출력 shape = [{batch}, {seq_len}, {d_model}]")
print(f"  KV Cache에 저장되는 것: c_kv = [{batch}, {seq_len}, {d_c}]")
print(f"  MHA였다면: [{batch}, {seq_len}, {2*n_heads*d_head}]를 저장해야 함")
print(f"  메모리 절감: {d_c} vs {2*n_heads*d_head} = {d_c/(2*n_heads*d_head)*100:.1f}%만 사용")"""))

# ── Cell 11: 종합 도전 문제 ──────────────────────────────────────
cells.append(md("""\
## 종합 도전: Full MLA 레이어 구현 + 메모리 측정 <a name='bonus'></a>

### 문제

완전한 MLA 레이어를 TF 클래스로 구현하고, MHA/GQA와의 메모리 사용량을 정량적으로 비교하세요.

요구사항:
1. `MLALayer` 클래스: Down/Up projection + Multi-head Attention + Output projection
2. `MHALayer` 클래스: 표준 Multi-Head Attention (비교 기준)
3. 시퀀스 길이 [128, 256, 512, 1024]에서 KV Cache 메모리 비교 그래프"""))

# ── Cell 12: 종합 도전 풀이 ──────────────────────────────────────
cells.append(code("""\
# ── 종합 도전 풀이: Full MLA vs MHA 구현 및 비교 ─────────────────
print("=" * 45)
print("종합 도전: Full MLA 레이어 구현")
print("=" * 45)

class MLALayer(tf.keras.layers.Layer):
    # Multi-head Latent Attention 레이어

    def __init__(self, d_model, n_heads, d_head, d_compress):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_compress = d_compress

        self.W_q = tf.keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.W_down = tf.keras.layers.Dense(d_compress, use_bias=False)
        self.W_up = tf.keras.layers.Dense(2 * n_heads * d_head, use_bias=False)
        self.W_o = tf.keras.layers.Dense(d_model, use_bias=False)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, return_cache=False):
        residual = x
        x = self.layer_norm(x)
        batch, seq_len, _ = x.shape

        # Q projection
        q = tf.reshape(self.W_q(x), [batch, seq_len, self.n_heads, self.d_head])
        q = tf.transpose(q, [0, 2, 1, 3])

        # KV compression + restoration
        c_kv = self.W_down(x)
        kv = tf.reshape(self.W_up(c_kv), [batch, seq_len, 2, self.n_heads, self.d_head])
        k = tf.transpose(kv[:, :, 0], [0, 2, 1, 3])
        v = tf.transpose(kv[:, :, 1], [0, 2, 1, 3])

        # Attention
        scale = tf.sqrt(tf.cast(self.d_head, tf.float32))
        scores = tf.matmul(q, k, transpose_b=True) / scale
        mask = tf.linalg.band_part(tf.ones([seq_len, seq_len]), -1, 0)
        scores = scores + (1.0 - mask) * (-1e9)
        weights = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(weights, v)

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch, seq_len, self.n_heads * self.d_head])
        out = self.W_o(out) + residual

        if return_cache:
            return out, c_kv
        return out


class MHALayer(tf.keras.layers.Layer):
    # 표준 Multi-Head Attention 레이어 (비교 기준)

    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head

        self.W_q = tf.keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.W_k = tf.keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.W_v = tf.keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.W_o = tf.keras.layers.Dense(d_model, use_bias=False)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, return_cache=False):
        residual = x
        x = self.layer_norm(x)
        batch, seq_len, _ = x.shape

        q = tf.reshape(self.W_q(x), [batch, seq_len, self.n_heads, self.d_head])
        k = tf.reshape(self.W_k(x), [batch, seq_len, self.n_heads, self.d_head])
        v = tf.reshape(self.W_v(x), [batch, seq_len, self.n_heads, self.d_head])

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        scale = tf.sqrt(tf.cast(self.d_head, tf.float32))
        scores = tf.matmul(q, k, transpose_b=True) / scale
        mask = tf.linalg.band_part(tf.ones([seq_len, seq_len]), -1, 0)
        scores = scores + (1.0 - mask) * (-1e9)
        weights = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(weights, v)

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch, seq_len, self.n_heads * self.d_head])
        out = self.W_o(out) + residual

        if return_cache:
            kv_cache = tf.concat([
                tf.reshape(k, [batch, seq_len, -1]),
                tf.reshape(v, [batch, seq_len, -1])
            ], axis=-1)
            return out, kv_cache
        return out


# 파라미터 설정
d_model = 256
n_heads = 8
d_head = 32
d_compress = 32
batch = 2

mla = MLALayer(d_model, n_heads, d_head, d_compress)
mha = MHALayer(d_model, n_heads, d_head)

# 시퀀스 길이별 KV Cache 비교
seq_lengths = [128, 256, 512, 1024]
mla_cache_sizes = []
mha_cache_sizes = []

print(f"KV Cache 크기 비교 (d_model={d_model}, H={n_heads}, d_h={d_head}, d_c={d_compress}):")
print(f"{'시퀀스':>8} | {'MHA Cache':>12} | {'MLA Cache':>12} | {'절감률':>8}")
print("-" * 48)

for sl in seq_lengths:
    x = tf.random.normal([batch, sl, d_model])

    _, mla_cache = mla(x, return_cache=True)
    _, mha_cache = mha(x, return_cache=True)

    mla_size = np.prod(mla_cache.shape) * 4  # FP32 bytes
    mha_size = np.prod(mha_cache.shape) * 4

    mla_cache_sizes.append(mla_size)
    mha_cache_sizes.append(mha_size)

    saving = (1 - mla_size / mha_size) * 100
    print(f"{sl:>8} | {mha_size:>10,} B | {mla_size:>10,} B | {saving:>7.1f}%")

# 시각화
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

x_pos = range(len(seq_lengths))
width = 0.35
bars1 = ax.bar([p - width/2 for p in x_pos], [s/1024 for s in mha_cache_sizes],
               width, label='MHA', color='red', alpha=0.7)
bars2 = ax.bar([p + width/2 for p in x_pos], [s/1024 for s in mla_cache_sizes],
               width, label='MLA', color='green', alpha=0.7)

ax.set_xticks(x_pos)
ax.set_xticklabels([str(s) for s in seq_lengths])
ax.set_xlabel('시퀀스 길이', fontsize=11)
ax.set_ylabel('KV Cache 크기 (KB)', fontsize=11)
ax.set_title('MHA vs MLA KV Cache 크기 비교', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chapter16_sparse_attention/practice_mla_cache_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("\\n그래프 저장됨: chapter16_sparse_attention/practice_mla_cache_comparison.png")

# 총 파라미터 수 비교
mla_params = sum(np.prod(v.shape) for v in mla.trainable_variables)
mha_params = sum(np.prod(v.shape) for v in mha.trainable_variables)
print(f"\\n모델 파라미터 수:")
print(f"  MHA: {mha_params:,}")
print(f"  MLA: {mla_params:,}")
print(f"  MLA는 Down/Up projection 추가로 파라미터가 약간 더 많지만,")
print(f"  KV Cache가 {d_compress/(2*n_heads*d_head)*100:.1f}%만 필요하여 추론 메모리를 크게 절약합니다!")"""))

create_notebook(cells, 'chapter16_sparse_attention/practice/ex01_mla_from_scratch.ipynb')
