import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 0: Header ──
cells.append(md(r"""# 실습 퀴즈: 시공간 3D DiT 패치 시퀀스 변환기

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다

## 목차
- [Q1: 3D 패치 추출](#q1)
- [Q2: 패치 시퀀스 길이 계산](#q2)
- [Q3: 3D RoPE 주파수 계산](#q3)
- [Q4: 시공간 위치 임베딩](#q4)
- [종합 도전: 완전한 시공간 패처 모듈](#bonus)"""))

# ── Cell 1: Imports ──
cells.append(code(r"""# ── 라이브러리 임포트 ──────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")"""))

# ── Cell 2: Q1 Problem ──
cells.append(md(r"""## Q1: 3D 패치 추출 <a name='q1'></a>

### 문제

비디오 텐서 $x \in \mathbb{R}^{B \times T \times H \times W \times C}$에서 3D 패치를 추출합니다.
패치 크기가 $(p_t, p_h, p_w) = (2, 4, 4)$이고, 입력이 $B=1, T=8, H=16, W=16, C=3$일 때:

1. 추출되는 패치 수는?
2. 각 패치의 벡터 크기(flatten 후)는?
3. `tf.extract_volume_patches` 또는 reshape로 패치를 추출하세요.

**여러분의 예측:**
- 패치 수 = `?`
- 패치 벡터 크기 = `?`"""))

# ── Cell 3: Q1 Solution ──
cells.append(code(r"""# ── Q1 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q1 풀이: 3D 패치 추출")
print("=" * 45)

B, T, H, W, C = 1, 8, 16, 16, 3
pt, ph, pw = 2, 4, 4

n_t = T // pt
n_h = H // ph
n_w = W // pw
n_patches = n_t * n_h * n_w
patch_dim = pt * ph * pw * C

print(f"\n입력 shape: ({B}, {T}, {H}, {W}, {C})")
print(f"패치 크기: ({pt}, {ph}, {pw})")
print(f"\n시간 패치 수: {T}/{pt} = {n_t}")
print(f"높이 패치 수: {H}/{ph} = {n_h}")
print(f"너비 패치 수: {W}/{pw} = {n_w}")
print(f"총 패치 수: {n_t} x {n_h} x {n_w} = {n_patches}")
print(f"패치 벡터 크기: {pt} x {ph} x {pw} x {C} = {patch_dim}")

video = tf.random.normal([B, T, H, W, C])

patches = tf.reshape(video, [B, n_t, pt, n_h, ph, n_w, pw, C])
patches = tf.transpose(patches, [0, 1, 3, 5, 2, 4, 6, 7])
patches = tf.reshape(patches, [B, n_patches, patch_dim])

print(f"\n추출된 패치 텐서 shape: {patches.shape}")
print(f"  → (배치, 시퀀스길이, 패치차원) = ({B}, {n_patches}, {patch_dim})")

print("\n[해설]")
print("  3D 패치 추출은 reshape + transpose로 구현합니다.")
print(f"  결과적으로 {n_patches}개 토큰의 시퀀스로 변환됩니다.")"""))

# ── Cell 4: Q2 Problem ──
cells.append(md(r"""## Q2: 패치 시퀀스 길이 계산 <a name='q2'></a>

### 문제

다양한 비디오 해상도에 대해 DiT 시퀀스 길이를 계산하세요.

패치 크기 $(p_t, p_h, p_w) = (2, 4, 4)$, VAE 압축 $M_t=4, M_h=8, M_w=8$일 때:

| 원본 비디오 | Latent 크기 | 시퀀스 길이 |
|------------|-------------|------------|
| 32×256×256 | ? | ? |
| 64×512×512 | ? | ? |
| 128×1024×1024 | ? | ? |

**여러분의 예측:** 시퀀스 길이 = `?`, `?`, `?`"""))

# ── Cell 5: Q2 Solution ──
cells.append(code(r"""# ── Q2 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q2 풀이: 패치 시퀀스 길이 계산")
print("=" * 45)

Mt, Mh, Mw = 4, 8, 8
pt, ph, pw = 2, 4, 4

videos = [
    (32, 256, 256),
    (64, 512, 512),
    (128, 1024, 1024),
]

print(f"\nVAE 압축: Mt={Mt}, Mh={Mh}, Mw={Mw}")
print(f"패치 크기: pt={pt}, ph={ph}, pw={pw}\n")

print(f"{'원본 비디오':>18} | {'Latent':>18} | {'시퀀스 길이':>12} | {'Self-Attn FLOPs':>16}")
print("-" * 72)

for T_orig, H_orig, W_orig in videos:
    lat_T = T_orig // Mt
    lat_H = H_orig // Mh
    lat_W = W_orig // Mw

    seq_len = (lat_T // pt) * (lat_H // ph) * (lat_W // pw)
    flops_attn = 2 * seq_len ** 2 * 128

    print(f"{T_orig:>4}x{H_orig:>4}x{W_orig:<4} | "
          f"{lat_T:>3}x{lat_H:>3}x{lat_W:<4}   | "
          f"{seq_len:>12,} | "
          f"{flops_attn/1e9:>13.2f} G")

print("\n[해설]")
print("  시퀀스 길이 = (T/Mt/pt) x (H/Mh/ph) x (W/Mw/pw)")
print("  해상도가 2배 → 시퀀스 길이 ~8배 → Self-Attention 비용 ~64배!")
print("  이것이 Flash Attention, Sparse Attention이 필수인 이유입니다.")"""))

# ── Cell 6: Q3 Problem ──
cells.append(md(r"""## Q3: 3D RoPE 주파수 계산 <a name='q3'></a>

### 문제

3D RoPE(Rotary Position Embedding)는 시간(t), 높이(h), 너비(w) 3개 축에 대해 독립적인 주파수를 계산합니다:

$$\theta_k^{(axis)} = \frac{1}{10000^{2k/d_{axis}}}, \quad k = 0, 1, \ldots, d_{axis}/2 - 1$$

$d_{model} = 12$ (3축 균등 배분: $d_t = d_h = d_w = 4$)일 때:

1. 각 축의 주파수 벡터를 계산하세요 ($k=0, 1$)
2. 위치 $(t=3, h=5, w=7)$에서의 회전 각도를 구하세요

**여러분의 예측:** $\theta_0^{(t)} = ?$, $\theta_1^{(t)} = ?$"""))

# ── Cell 7: Q3 Solution ──
cells.append(code(r"""# ── Q3 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q3 풀이: 3D RoPE 주파수 계산")
print("=" * 45)

d_model = 12
d_per_axis = d_model // 3
base = 10000.0

print(f"\nd_model = {d_model}, 축당 차원 = {d_per_axis}")
print(f"base = {base}\n")

for axis_name, d_axis in [('시간(t)', d_per_axis), ('높이(h)', d_per_axis), ('너비(w)', d_per_axis)]:
    freqs = []
    for k in range(d_axis // 2):
        theta = 1.0 / (base ** (2 * k / d_axis))
        freqs.append(theta)
        print(f"  {axis_name} θ_{k} = 1 / 10000^({2*k}/{d_axis}) = {theta:.6f}")
    print()

pos_t, pos_h, pos_w = 3, 5, 7

print("위치 (t=3, h=5, w=7)에서의 회전 각도:\n")
for axis_name, pos, d_axis in [('시간', pos_t, d_per_axis), ('높이', pos_h, d_per_axis), ('너비', pos_w, d_per_axis)]:
    angles = []
    for k in range(d_axis // 2):
        theta = 1.0 / (base ** (2 * k / d_axis))
        angle = pos * theta
        angles.append(angle)
    print(f"  {axis_name} 축 (pos={pos}): 각도 = {angles}")

def compute_3d_rope(positions, d_per_axis, base=10000.0):
    B = positions.shape[0]
    freqs_list = []
    for axis in range(3):
        pos = positions[:, axis:axis+1]
        k = tf.range(d_per_axis // 2, dtype=tf.float32)
        theta = 1.0 / tf.pow(base, 2.0 * k / float(d_per_axis))
        angles = tf.cast(pos, tf.float32) * theta[tf.newaxis, :]
        cos_vals = tf.cos(angles)
        sin_vals = tf.sin(angles)
        freqs_list.append(tf.stack([cos_vals, sin_vals], axis=-1))
    rope = tf.concat(freqs_list, axis=1)
    return rope

test_positions = tf.constant([[3, 5, 7], [0, 0, 0], [1, 1, 1]])
rope_result = compute_3d_rope(test_positions, d_per_axis)
print(f"\n3D RoPE 텐서 shape: {rope_result.shape}")
print(f"위치 (3,5,7)의 cos/sin 값:\n{rope_result[0].numpy()}")

print("\n[해설]")
print("  3D RoPE는 각 공간 축에 독립적인 주파수를 할당합니다.")
print("  차원을 3등분하여 시간/높이/너비 축에 각각 적용합니다.")"""))

# ── Cell 8: Q4 Problem ──
cells.append(md(r"""## Q4: 시공간 위치 임베딩 <a name='q4'></a>

### 문제

3D 패치 시퀀스에 위치 정보를 추가하는 두 가지 방법을 비교하세요:

1. **Learnable Position Embedding**: $z_i = z_i + \text{PE}[i]$ (학습 가능)
2. **3D RoPE**: 쿼리/키에 회전 적용 (상대 위치)

$T_{\text{lat}}=4, H_{\text{lat}}=8, W_{\text{lat}}=8$, 패치 크기 $(1,2,2)$, $d_{model}=64$일 때:
- 총 시퀀스 길이는?
- Learnable PE의 파라미터 수는?

**여러분의 예측:** 시퀀스 길이 = `?`, PE 파라미터 수 = `?`"""))

# ── Cell 9: Q4 Solution ──
cells.append(code(r"""# ── Q4 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q4 풀이: 시공간 위치 임베딩")
print("=" * 45)

T_lat, H_lat, W_lat = 4, 8, 8
pt, ph, pw = 1, 2, 2
d_model = 64

seq_len = (T_lat // pt) * (H_lat // ph) * (W_lat // pw)
learnable_params = seq_len * d_model

print(f"\nLatent 크기: T={T_lat}, H={H_lat}, W={W_lat}")
print(f"패치 크기: ({pt}, {ph}, {pw})")
print(f"시퀀스 길이: ({T_lat}/{pt}) x ({H_lat}/{ph}) x ({W_lat}/{pw}) = {seq_len}")
print(f"\n1. Learnable PE 파라미터 수: {seq_len} x {d_model} = {learnable_params:,}")

# Learnable PE 구현
pos_embed = tf.Variable(tf.random.normal([1, seq_len, d_model]) * 0.02)
x = tf.random.normal([2, seq_len, d_model])
x_with_pe = x + pos_embed
print(f"   입력 shape: {x.shape} → 출력 shape: {x_with_pe.shape}")

# 3D RoPE 구현 (Q, K에 적용)
print(f"\n2. 3D RoPE (파라미터 0개 — 학습 불필요)")

positions = []
for t in range(T_lat // pt):
    for h in range(H_lat // ph):
        for w in range(W_lat // pw):
            positions.append([t, h, w])
positions = tf.constant(positions)

d_per_axis = d_model // 3
rope_freqs = compute_3d_rope(positions, d_per_axis + (1 if d_model % 3 else 0))
print(f"   위치 좌표 shape: {positions.shape}")
print(f"   RoPE 주파수 shape: {rope_freqs.shape}")

print(f"\n비교 요약:")
print(f"{'방법':<20} | {'파라미터 수':>12} | {'해상도 일반화':>15}")
print("-" * 55)
print(f"{'Learnable PE':<20} | {learnable_params:>12,} | {'제한적 (고정 길이)':>15}")
print(f"{'3D RoPE':<20} | {'0':>12} | {'유연 (상대 위치)':>15}")

print("\n[해설]")
print("  3D RoPE는 파라미터 없이 상대 위치를 인코딩합니다.")
print("  학습 시와 다른 해상도에서도 자연스럽게 확장 가능합니다.")
print("  HunyuanVideo, SD3 등 최신 모델은 모두 RoPE를 사용합니다.")"""))

# ── Cell 10: Bonus Problem ──
cells.append(md(r"""## 종합 도전: 완전한 시공간 패처 모듈 <a name='bonus'></a>

### 문제

다음 기능을 모두 포함하는 `SpatiotemporalPatcher` 클래스를 구현하세요:

1. 3D 비디오 텐서 → 패치 추출 (reshape + transpose)
2. 패치 → 선형 임베딩 ($d_{model}$ 차원)
3. 3D 위치 좌표 생성 (t, h, w)
4. 3D RoPE 주파수 계산

입력: `(B, T, H, W, C)` → 출력: `(B, N, d_model)` + RoPE 주파수"""))

# ── Cell 11: Bonus Solution ──
cells.append(code(r"""# ── 종합 도전 풀이 ──────────────────────────────────────────────
print("=" * 45)
print("종합 도전 풀이: 시공간 패처 모듈")
print("=" * 45)

class SpatiotemporalPatcher(tf.keras.layers.Layer):
    def __init__(self, patch_size, d_model, rope_base=10000.0):
        super().__init__()
        self.pt, self.ph, self.pw = patch_size
        self.d_model = d_model
        self.rope_base = rope_base
        self.patch_dim = self.pt * self.ph * self.pw
        self.projection = None

    def build(self, input_shape):
        C = input_shape[-1]
        full_patch_dim = self.patch_dim * C
        self.projection = tf.keras.layers.Dense(self.d_model)
        super().build(input_shape)

    def extract_patches(self, x):
        B = tf.shape(x)[0]
        T, H, W, C = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        n_t = T // self.pt
        n_h = H // self.ph
        n_w = W // self.pw
        x = tf.reshape(x, [B, n_t, self.pt, n_h, self.ph, n_w, self.pw, C])
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
        N = n_t * n_h * n_w
        x = tf.reshape(x, [B, N, self.pt * self.ph * self.pw * C])
        return x, (n_t, n_h, n_w)

    def make_positions(self, n_t, n_h, n_w):
        positions = []
        for t in range(n_t):
            for h in range(n_h):
                for w in range(n_w):
                    positions.append([t, h, w])
        return tf.constant(positions, dtype=tf.float32)

    def compute_rope(self, positions, d_per_axis):
        freqs_list = []
        for axis in range(3):
            pos = positions[:, axis:axis+1]
            k = tf.range(d_per_axis // 2, dtype=tf.float32)
            theta = 1.0 / tf.pow(self.rope_base, 2.0 * k / tf.cast(d_per_axis, tf.float32))
            angles = pos * theta[tf.newaxis, :]
            cos_vals = tf.cos(angles)
            sin_vals = tf.sin(angles)
            freqs_list.append(tf.stack([cos_vals, sin_vals], axis=-1))
        return tf.concat(freqs_list, axis=1)

    def call(self, x):
        patches, (n_t, n_h, n_w) = self.extract_patches(x)
        embedded = self.projection(patches)
        positions = self.make_positions(n_t, n_h, n_w)
        d_per_axis = self.d_model // 3
        rope_freqs = self.compute_rope(positions, d_per_axis)
        return embedded, rope_freqs, positions

# 테스트
B, T, H, W, C = 2, 8, 32, 32, 3
patch_size = (2, 4, 4)
d_model = 96

patcher = SpatiotemporalPatcher(patch_size, d_model)
video = tf.random.normal([B, T, H, W, C])

embedded, rope_freqs, positions = patcher(video)

print(f"\n입력 비디오 shape: {video.shape}")
print(f"패치 크기: {patch_size}")
print(f"d_model: {d_model}")
print(f"\n출력:")
print(f"  임베딩 shape: {embedded.shape}")
print(f"  RoPE 주파수 shape: {rope_freqs.shape}")
print(f"  위치 좌표 shape: {positions.shape}")

expected_n = (T // patch_size[0]) * (H // patch_size[1]) * (W // patch_size[2])
print(f"\n검증:")
print(f"  예상 시퀀스 길이: {expected_n}")
print(f"  실제 시퀀스 길이: {embedded.shape[1]}")
print(f"  일치: {expected_n == embedded.shape[1]}")

n_params = sum(p.numpy().size for p in patcher.trainable_variables)
print(f"  총 파라미터 수: {n_params:,}")

print(f"\n위치 좌표 (처음 5개):")
for i in range(min(5, positions.shape[0])):
    t, h, w = positions[i].numpy()
    print(f"  토큰 {i}: (t={t:.0f}, h={h:.0f}, w={w:.0f})")

print("\n[해설]")
print("  SpatiotemporalPatcher는 비디오를 DiT가 처리할 수 있는")
print("  토큰 시퀀스로 변환하는 핵심 전처리 모듈입니다.")
print("  3D RoPE를 통해 시공간 위치 정보를 인코딩합니다.")"""))

path = '/workspace/chapter17_diffusion_transformers/practice/ex01_spatiotemporal_patcher.ipynb'
create_notebook(cells, path)
