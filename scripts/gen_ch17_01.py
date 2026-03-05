"""Generate chapter17_diffusion_transformers/01_from_unet_to_dit.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 17: 비디오 생성 모델과 DiT — U-Net에서 DiT로의 전환

## 학습 목표
- U-Net 기반 Diffusion 모델의 스케일링 한계를 수학적으로 이해한다
- DiT(Diffusion Transformer)의 패치 임베딩 수식을 도출하고 구현한다
- 2D(이미지)와 3D(비디오) 패치 토크나이징의 차이를 비교한다
- DiT 스케일링 법칙(FID vs Compute)을 시각화하고 분석한다
- 패치 크기가 시퀀스 길이·메모리·품질에 미치는 영향을 정량화한다

## 목차
1. [수학적 기초: 패치 임베딩과 스케일링 법칙](#1.-수학적-기초)
2. [라이브러리 임포트 및 환경 설정](#2.-환경-설정)
3. [2D 패치 임베딩 구현](#3.-2D-패치-임베딩)
4. [U-Net vs DiT 파라미터·FLOPs 비교](#4.-UNet-vs-DiT-비교)
5. [2D(이미지) vs 3D(비디오) 패치 비교](#5.-2D-vs-3D-패치)
6. [DiT 스케일링 법칙 시각화](#6.-스케일링-법칙)
7. [패치 크기에 따른 시퀀스 길이·메모리 분석](#7.-패치-크기-분석)
8. [정리](#8.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### 패치 임베딩 (Patch Embedding)

입력 이미지 $x \in \mathbb{R}^{H \times W \times C}$를 패치 크기 $p$로 분할하면:

$$x_{\text{patches}} \in \mathbb{R}^{N \times (p^2 \cdot C)}, \quad N = \frac{H}{p} \cdot \frac{W}{p}$$

- $H, W$: 이미지 높이, 너비
- $C$: 채널 수 (RGB=3, latent=4 등)
- $p$: 패치 크기 (예: 2, 4, 8)
- $N$: 시퀀스 길이 (Transformer의 토큰 수)

각 패치는 선형 변환으로 $d$차원 임베딩으로 투영됩니다:

$$z_i = W_E \cdot \text{flatten}(\text{patch}_i) + b_E, \quad W_E \in \mathbb{R}^{d \times (p^2 C)}$$

### 3D 패치 임베딩 (비디오)

비디오 입력 $x \in \mathbb{R}^{T \times H \times W \times C}$에 대해 시공간 패치 $(p_t, p_h, p_w)$를 적용하면:

$$N_{3D} = \frac{T}{p_t} \cdot \frac{H}{p_h} \cdot \frac{W}{p_w}, \quad \text{patch dim} = p_t \cdot p_h \cdot p_w \cdot C$$

### DiT 스케일링 법칙

Peebles & Xie (2023)이 발견한 핵심 관계:

$$\text{FID} \propto -a \cdot \log(\text{GFLOPs}) + b$$

- FID(Frechet Inception Distance)는 계산량의 로그에 비례하여 개선됩니다
- U-Net은 깊이를 늘려도 성능이 포화되지만, DiT는 ViT처럼 스케일링이 가능합니다

### U-Net vs DiT 파라미터 비교

| 모델 | 파라미터 수 | FLOPs (256×256) | FID-256 |
|------|------------|-----------------|---------|
| U-Net (ADM) | ~554M | ~1050 GFLOPs | 10.94 |
| DiT-XL/2 | ~675M | ~119 GFLOPs | 9.62 |
| DiT-XL/2 (cfg) | ~675M | ~238 GFLOPs | 2.27 |

**요약 표:**

| 구분 | 수식 | 설명 |
|------|------|------|
| 2D 시퀀스 길이 | $N = (H/p)(W/p)$ | 이미지 → 토큰 수 |
| 3D 시퀀스 길이 | $N = (T/p_t)(H/p_h)(W/p_w)$ | 비디오 → 토큰 수 |
| 패치 차원 | $d_{patch} = p^2 \cdot C$ (2D) | 각 패치의 벡터 크기 |
| Self-Attention 비용 | $O(N^2 \cdot d)$ | 시퀀스 길이에 2차 의존 |

---

### 🐣 초등학생을 위한 DiT 패치 임베딩 친절 설명!

#### 🔢 패치 임베딩이 뭔가요?

> 💡 **비유**: 큰 사진을 작은 퍼즐 조각으로 나누는 것을 생각해보세요!

큰 사진(이미지)을 한 번에 이해하기는 어렵습니다. 그래서 사진을 작은 정사각형 조각(패치)으로 잘라서,
각 조각을 숫자 리스트(벡터)로 바꿉니다. 이러면 Transformer가 각 조각을 "단어"처럼 처리할 수 있어요!

- 🖼️ **이미지 퍼즐**: 256×256 사진을 8×8 조각으로 나누면 → 32×32 = 1,024개의 퍼즐 조각
- 🎬 **비디오 퍼즐**: 비디오는 시간도 있으니 3차원으로 자릅니다 → 시간 조각 + 공간 조각

#### 🤔 왜 U-Net 대신 Transformer를 쓰나요?

> 💡 **비유**: U-Net은 "고정된 크기의 깔때기"이고, DiT는 "원하는 만큼 넓힐 수 있는 공장"입니다

U-Net은 구조가 고정되어 있어서 모델을 키우기 어렵지만,
Transformer(DiT)는 층을 쌓기만 하면 되니까 더 크고 더 좋은 모델을 쉽게 만들 수 있어요!

---

### 📝 연습 문제

#### 문제 1: 패치 시퀀스 길이 계산

512×512 이미지를 패치 크기 $p=16$으로 분할할 때 시퀀스 길이 $N$은?

$$N = \frac{H}{p} \cdot \frac{W}{p} = ?$$

<details>
<summary>💡 풀이 확인</summary>

$$N = \frac{512}{16} \times \frac{512}{16} = 32 \times 32 = 1024$$

시퀀스 길이는 **1,024 토큰**입니다. 비교: $p=8$이면 $N = 64 \times 64 = 4096$으로 4배 증가합니다.
</details>

#### 문제 2: 3D 비디오 패치

16프레임, 256×256 비디오를 $(p_t, p_h, p_w) = (4, 16, 16)$으로 패치할 때 토큰 수는?

<details>
<summary>💡 풀이 확인</summary>

$$N = \frac{16}{4} \times \frac{256}{16} \times \frac{256}{16} = 4 \times 16 \times 16 = 1024$$

3D 패치 결과도 **1,024 토큰**입니다. 시간 압축 $p_t$가 클수록 토큰 수가 줄어 효율적이지만, 시간 해상도가 감소합니다.
</details>"""))

# ── Cell 3: Imports ──────────────────────────────────────────────────
cells.append(code("""\
# ── 라이브러리 임포트 및 환경 설정 ──────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import time

np.random.seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"GPU 사용 가능: {len(tf.config.list_physical_devices('GPU')) > 0}")"""))

# ── Cell 4: Section 3 header ──────────────────────────────────────
cells.append(md("""\
## 3. 2D 패치 임베딩 구현 <a name='3.-2D-패치-임베딩'></a>

DiT의 핵심은 이미지를 패치 시퀀스로 변환하는 것입니다.
`tf.image.extract_patches` 또는 Conv2D를 활용하여 2D 패치 임베딩을 구현합니다."""))

# ── Cell 5: 2D patch embedding implementation ──────────────────────
cells.append(code("""\
# ── 2D 패치 임베딩 구현 ──────────────────────────────────────────
class PatchEmbedding2D(tf.keras.layers.Layer):
    # 이미지를 패치로 분할하고 임베딩하는 레이어
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = tf.keras.layers.Conv2D(
            embed_dim, kernel_size=patch_size, strides=patch_size,
            padding='valid', use_bias=True
        )

    def call(self, x):
        # x: (B, H, W, C)
        patches = self.proj(x)  # (B, H/p, W/p, embed_dim)
        B = tf.shape(patches)[0]
        patches = tf.reshape(patches, [B, -1, self.embed_dim])  # (B, N, D)
        return patches

# 테스트: 256x256 RGB 이미지, 패치 크기 8
image = tf.random.normal([2, 256, 256, 3])
patch_embed = PatchEmbedding2D(patch_size=8, embed_dim=384)
tokens = patch_embed(image)

H, W, p = 256, 256, 8
expected_N = (H // p) * (W // p)

print("=== 2D 패치 임베딩 결과 ===")
print(f"입력 이미지 shape: {image.shape}")
print(f"패치 크기: {p}x{p}")
print(f"출력 토큰 shape: {tokens.shape}")
print(f"시퀀스 길이 N = ({H}/{p}) x ({W}/{p}) = {expected_N}")
print(f"각 토큰 차원: {tokens.shape[-1]}")
print(f"파라미터 수: {patch_embed.count_params():,}")

# 다양한 패치 크기 비교
print("\\n=== 패치 크기별 시퀀스 길이 ===")
print(f"{'패치 크기':<12} | {'시퀀스 길이':>12} | {'패치 차원':>10}")
print("-" * 42)
for ps in [2, 4, 8, 16, 32]:
    seq_len = (256 // ps) * (256 // ps)
    patch_dim = ps * ps * 3
    print(f"p={ps:<10} | {seq_len:>12,} | {patch_dim:>10}")"""))

# ── Cell 6: Section 4 header ──────────────────────────────────────
cells.append(md("""\
## 4. U-Net vs DiT 파라미터·FLOPs 비교 <a name='4.-UNet-vs-DiT-비교'></a>

U-Net(ADM)과 DiT의 파라미터 수와 FLOPs를 동일 해상도에서 비교합니다.
DiT는 더 적은 FLOPs로 더 좋은 FID를 달성합니다."""))

# ── Cell 7: UNet vs DiT comparison table ────────────────────────────
cells.append(code("""\
# ── U-Net vs DiT 파라미터/FLOPs 비교 ────────────────────────────────
# Peebles & Xie (2023), "Scalable Diffusion Models with Transformers"

models = {
    'DiT-S/2': {'params': 33e6, 'gflops': 6.06, 'fid': 68.4, 'type': 'DiT'},
    'DiT-S/4': {'params': 33e6, 'gflops': 1.55, 'fid': 122.0, 'type': 'DiT'},
    'DiT-B/2': {'params': 130e6, 'gflops': 23.0, 'fid': 43.5, 'type': 'DiT'},
    'DiT-B/4': {'params': 130e6, 'gflops': 5.97, 'fid': 68.4, 'type': 'DiT'},
    'DiT-L/2': {'params': 458e6, 'gflops': 80.7, 'fid': 23.3, 'type': 'DiT'},
    'DiT-L/4': {'params': 458e6, 'gflops': 20.5, 'fid': 44.4, 'type': 'DiT'},
    'DiT-XL/2': {'params': 675e6, 'gflops': 119.0, 'fid': 9.62, 'type': 'DiT'},
    'DiT-XL/4': {'params': 675e6, 'gflops': 29.9, 'fid': 27.0, 'type': 'DiT'},
    'ADM (U-Net)': {'params': 554e6, 'gflops': 1050.0, 'fid': 10.94, 'type': 'UNet'},
    'ADM-U': {'params': 730e6, 'gflops': 742.0, 'fid': 7.49, 'type': 'UNet'},
    'LDM-4 (U-Net)': {'params': 400e6, 'gflops': 103.0, 'fid': 10.56, 'type': 'UNet'},
}

print("=" * 78)
print("U-Net vs DiT 모델 비교 (ImageNet 256x256, class-conditional)")
print("=" * 78)
print(f"{'모델':<18} | {'파라미터':>10} | {'GFLOPs':>10} | {'FID-256':>8} | {'타입':>6}")
print("-" * 78)
for name, m in models.items():
    p_str = f"{m['params']/1e6:.0f}M"
    print(f"{name:<18} | {p_str:>10} | {m['gflops']:>10.1f} | {m['fid']:>8.2f} | {m['type']:>6}")

# 핵심 비교 요약
dit_xl = models['DiT-XL/2']
adm = models['ADM (U-Net)']
flops_ratio = adm['gflops'] / dit_xl['gflops']
print(f"\\n핵심 비교: DiT-XL/2 vs ADM (U-Net)")
print(f"  FLOPs 절감: {flops_ratio:.1f}x 적은 연산량")
print(f"  FID 개선: {adm['fid']:.2f} → {dit_xl['fid']:.2f}")
print(f"  DiT는 {flops_ratio:.1f}배 적은 FLOPs로 더 좋은 FID를 달성!")"""))

# ── Cell 8: Section 5 header ──────────────────────────────────────
cells.append(md(r"""\
## 5. 2D(이미지) vs 3D(비디오) 패치 비교 <a name='5.-2D-vs-3D-패치'></a>

DiT를 비디오로 확장할 때 핵심은 **시공간 3D 패치**입니다.

| 구분 | 2D 패치 (이미지) | 3D 패치 (비디오) |
|------|-----------------|-----------------|
| 입력 | $\mathbb{R}^{H \times W \times C}$ | $\mathbb{R}^{T \times H \times W \times C}$ |
| 패치 크기 | $(p_h, p_w)$ | $(p_t, p_h, p_w)$ |
| 시퀀스 길이 | $(H/p_h)(W/p_w)$ | $(T/p_t)(H/p_h)(W/p_w)$ |
| 패치 차원 | $p_h \cdot p_w \cdot C$ | $p_t \cdot p_h \cdot p_w \cdot C$ |

Sora/HunyuanVideo 등 비디오 생성 모델은 3D 패치를 사용하여 시간 축까지 압축합니다."""))

# ── Cell 9: 2D vs 3D patch comparison code ──────────────────────────
cells.append(code("""\
# ── 2D(이미지) vs 3D(비디오) 패치 비교 ────────────────────────────
class PatchEmbedding3D(tf.keras.layers.Layer):
    # 비디오를 시공간 패치로 분할하고 임베딩하는 레이어
    def __init__(self, patch_size_t, patch_size_h, patch_size_w, embed_dim, in_channels=4, **kwargs):
        super().__init__(**kwargs)
        self.pt, self.ph, self.pw = patch_size_t, patch_size_h, patch_size_w
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        patch_dim = patch_size_t * patch_size_h * patch_size_w * in_channels
        self.linear_proj = tf.keras.layers.Dense(embed_dim)

    def call(self, x):
        # x: (B, T, H, W, C) - 비디오 텐서
        B = tf.shape(x)[0]
        T, H, W, C = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        nt, nh, nw = T // self.pt, H // self.ph, W // self.pw

        # 시공간 패치 추출 (reshape 기반)
        x = tf.reshape(x, [B, nt, self.pt, nh, self.ph, nw, self.pw, C])
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])  # (B, nt, nh, nw, pt, ph, pw, C)
        x = tf.reshape(x, [B, nt * nh * nw, self.pt * self.ph * self.pw * C])

        # 선형 투영
        tokens = self.linear_proj(x)  # (B, N, embed_dim)
        return tokens

# 2D 패치 테스트
image = tf.random.normal([1, 256, 256, 4])
patch_2d = PatchEmbedding2D(patch_size=2, embed_dim=1152)
tokens_2d = patch_2d(image)

# 3D 패치 테스트 (비디오: 16 프레임)
video = tf.random.normal([1, 16, 256, 256, 4])
patch_3d = PatchEmbedding3D(
    patch_size_t=2, patch_size_h=2, patch_size_w=2,
    embed_dim=1152, in_channels=4
)
tokens_3d = patch_3d(video)

print("=" * 65)
print("2D(이미지) vs 3D(비디오) 패치 임베딩 비교")
print("=" * 65)
print(f"{'구분':<18} | {'2D (이미지)':>20} | {'3D (비디오)':>20}")
print("-" * 65)
print(f"{'입력 shape':<18} | {str(image.shape):>20} | {str(video.shape):>20}")
print(f"{'패치 크기':<18} | {'(2, 2)':>20} | {'(2, 2, 2)':>20}")
print(f"{'시퀀스 길이 N':<18} | {tokens_2d.shape[1]:>20,} | {tokens_3d.shape[1]:>20,}")
print(f"{'토큰 차원 D':<18} | {tokens_2d.shape[2]:>20} | {tokens_3d.shape[2]:>20}")
mem_2d = tokens_2d.shape[1] * 1152 * 4 / 1024
mem_3d = tokens_3d.shape[1] * 1152 * 4 / 1024
print(f"{'메모리 (float32)':<18} | {mem_2d:>17.0f} KB | {mem_3d:>17.0f} KB")

# 다양한 3D 패치 크기 비교
print("\\n=== 3D 패치 크기별 시퀀스 길이 (16프레임, 256x256, 4ch) ===")
print(f"{'(pt, ph, pw)':<15} | {'토큰 수':>10} | {'패치 차원':>10} | {'비고':>20}")
print("-" * 62)
configs = [(1,2,2), (2,2,2), (4,4,4), (1,8,8), (2,4,4)]
for pt, ph, pw in configs:
    T, H, W, C = 16, 256, 256, 4
    n_tok = (T//pt) * (H//ph) * (W//pw)
    p_dim = pt * ph * pw * C
    note = ""
    if pt == 1:
        note = "시간 압축 없음"
    elif pt == 4:
        note = "높은 시간 압축"
    else:
        note = "시간+공간 균형"
    print(f"({pt},{ph},{pw})" + " " * (14-len(f"({pt},{ph},{pw})")) + f"| {n_tok:>10,} | {p_dim:>10} | {note:>20}")"""))

# ── Cell 10: Section 6 - Scaling law ──────────────────────────────
cells.append(md("""\
## 6. DiT 스케일링 법칙 시각화 <a name='6.-스케일링-법칙'></a>

Peebles & Xie (2023)은 DiT가 Transformer와 동일한 스케일링 법칙을 따름을 보였습니다.
FID는 GFLOPs의 로그에 비례하여 개선됩니다."""))

# ── Cell 11: Scaling law visualization ────────────────────────────
cells.append(code("""\
# ── DiT 스케일링 법칙 시각화 ──────────────────────────────────────
# 논문 Table 1, Figure 4 기반 데이터 (Peebles & Xie, 2023)

dit_data = {
    'DiT-S/8': (0.4, 177.0), 'DiT-S/4': (1.6, 122.0), 'DiT-S/2': (6.1, 68.4),
    'DiT-B/8': (1.4, 131.0), 'DiT-B/4': (6.0, 68.4), 'DiT-B/2': (23.0, 43.5),
    'DiT-L/8': (5.0, 99.4), 'DiT-L/4': (20.5, 44.4), 'DiT-L/2': (80.7, 23.3),
    'DiT-XL/8': (7.4, 80.1), 'DiT-XL/4': (29.9, 27.0), 'DiT-XL/2': (119.0, 9.62),
}

sizes = {'S': ('o', '#2196F3', 7), 'B': ('s', '#4CAF50', 8),
         'L': ('^', '#FF9800', 9), 'XL': ('D', '#F44336', 10)}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 좌: GFLOPs vs FID (모든 모델)
ax1 = axes[0]
for name, (gflops, fid) in dit_data.items():
    size_key = name.split('-')[1].split('/')[0]
    marker, color, ms = sizes[size_key]
    ax1.scatter(gflops, fid, marker=marker, color=color, s=ms**2,
                zorder=5, edgecolors='black', linewidth=0.5)

# 추세선 (로그 선형)
gflops_arr = np.array([v[0] for v in dit_data.values()])
fid_arr = np.array([v[1] for v in dit_data.values()])
log_gf = np.log10(gflops_arr)
coeffs = np.polyfit(log_gf, fid_arr, 1)
x_fit = np.linspace(np.log10(0.3), np.log10(150), 100)
y_fit = np.polyval(coeffs, x_fit)
ax1.plot(10**x_fit, y_fit, 'k--', lw=1.5, alpha=0.5, label='log-linear 추세')

ax1.set_xscale('log')
ax1.set_xlabel('GFLOPs (log scale)', fontsize=11)
ax1.set_ylabel('FID-50K ↓', fontsize=11)
ax1.set_title('DiT 스케일링 법칙: FID vs Compute', fontweight='bold')

# 범례 (크기별)
for size_key, (marker, color, ms) in sizes.items():
    ax1.scatter([], [], marker=marker, color=color, s=ms**2,
                label=f'DiT-{size_key}', edgecolors='black', linewidth=0.5)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)

# 우: 패치 크기별 FID 비교 (같은 모델 크기)
ax2 = axes[1]
patch_sizes = [8, 4, 2]
for size_key, (marker, color, ms) in sizes.items():
    fids = []
    for ps in patch_sizes:
        key = f'DiT-{size_key}/{ps}'
        if key in dit_data:
            fids.append(dit_data[key][1])
    if len(fids) == 3:
        ax2.plot(patch_sizes, fids, f'-{marker}', color=color, lw=2,
                 ms=ms, label=f'DiT-{size_key}')

ax2.set_xlabel('패치 크기 p', fontsize=11)
ax2.set_ylabel('FID-50K ↓', fontsize=11)
ax2.set_title('패치 크기가 작을수록 FID 개선', fontweight='bold')
ax2.set_xticks([2, 4, 8])
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/dit_scaling_law.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter17_diffusion_transformers/dit_scaling_law.png")
print(f"\\n스케일링 추세 계수: FID ≈ {coeffs[0]:.1f} * log10(GFLOPs) + {coeffs[1]:.1f}")
print("→ GFLOPs가 10배 증가할 때 FID가 약 {:.1f} 감소".format(abs(coeffs[0])))"""))

# ── Cell 12: Section 7 - Patch size analysis ──────────────────────
cells.append(md("""\
## 7. 패치 크기에 따른 시퀀스 길이·메모리 분석 <a name='7.-패치-크기-분석'></a>

패치 크기 $p$는 품질과 효율성의 트레이드오프를 결정합니다:
- $p$ 작으면: 높은 품질 (FID ↓), 긴 시퀀스, 높은 메모리 ($O(N^2)$)
- $p$ 크면: 낮은 품질 (FID ↑), 짧은 시퀀스, 낮은 메모리"""))

# ── Cell 13: Patch size analysis visualization ──────────────────────
cells.append(code("""\
# ── 패치 크기 → 시퀀스 길이 · 메모리 분석 ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

patch_sizes = np.array([1, 2, 4, 8, 16, 32])
H_img, W_img, C_img = 256, 256, 4
d_model = 1152

# 시퀀스 길이
seq_lens = (H_img // patch_sizes) * (W_img // patch_sizes)

# Self-Attention 메모리 (float32, 단일 헤드, 배치 1)
attn_mem_mb = (seq_lens.astype(np.float64) ** 2 * 4) / (1024**2)

# FLOPs (self-attention only)
flops_giga = 2 * seq_lens.astype(np.float64) ** 2 * d_model / 1e9

ax1 = axes[0]
ax1.bar(range(len(patch_sizes)), seq_lens, color='#2196F3', alpha=0.8, edgecolor='black')
ax1.set_xticks(range(len(patch_sizes)))
ax1.set_xticklabels([f'p={p}' for p in patch_sizes])
ax1.set_ylabel('시퀀스 길이 N', fontsize=11)
ax1.set_title('패치 크기별 시퀀스 길이', fontweight='bold')
ax1.set_yscale('log')
for i, v in enumerate(seq_lens):
    ax1.text(i, v * 1.3, f'{v:,}', ha='center', fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
ax2.bar(range(len(patch_sizes)), attn_mem_mb, color='#F44336', alpha=0.8, edgecolor='black')
ax2.set_xticks(range(len(patch_sizes)))
ax2.set_xticklabels([f'p={p}' for p in patch_sizes])
ax2.set_ylabel('Attention 메모리 (MB)', fontsize=11)
ax2.set_title('Self-Attention 메모리 사용량', fontweight='bold')
ax2.set_yscale('log')
for i, v in enumerate(attn_mem_mb):
    label = f'{v:.0f}MB' if v >= 1 else f'{v*1024:.0f}KB'
    ax2.text(i, v * 1.3, label, ha='center', fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

ax3 = axes[2]
ax3.bar(range(len(patch_sizes)), flops_giga, color='#4CAF50', alpha=0.8, edgecolor='black')
ax3.set_xticks(range(len(patch_sizes)))
ax3.set_xticklabels([f'p={p}' for p in patch_sizes])
ax3.set_ylabel('Self-Attention FLOPs (GFLOPs)', fontsize=11)
ax3.set_title('Attention 연산 비용', fontweight='bold')
ax3.set_yscale('log')
for i, v in enumerate(flops_giga):
    ax3.text(i, v * 1.3, f'{v:.1f}G', ha='center', fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/patch_size_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter17_diffusion_transformers/patch_size_analysis.png")

# 수치 요약 표
print(f"\\n{'패치 크기':<10} | {'시퀀스 길이':>12} | {'Attn 메모리':>12} | {'Attn FLOPs':>12}")
print("-" * 55)
for i, p in enumerate(patch_sizes):
    mem_str = f"{attn_mem_mb[i]:.1f} MB" if attn_mem_mb[i] >= 1 else f"{attn_mem_mb[i]*1024:.0f} KB"
    print(f"p={p:<8} | {seq_lens[i]:>12,} | {mem_str:>12} | {flops_giga[i]:>10.1f} G")

print("\\n결론: p=2 (DiT-XL/2)가 품질-효율 균형점이며, 대부분의 DiT 논문에서 채택")"""))

# ── Cell 14: Summary ────────────────────────────────────────────────
cells.append(md(r"""\
## 8. 정리 <a name='8.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| 2D 패치 임베딩 | 이미지 → $(H/p)(W/p)$ 토큰 시퀀스 | ⭐⭐⭐ |
| 3D 패치 임베딩 | 비디오 → $(T/p_t)(H/p_h)(W/p_w)$ 토큰 시퀀스 | ⭐⭐⭐ |
| DiT 스케일링 법칙 | FID ∝ −log(GFLOPs), U-Net 대비 효율적 스케일링 | ⭐⭐⭐ |
| 패치 크기 트레이드오프 | p 작을수록 품질 ↑, 메모리/연산 ↑ ($O(N^2)$) | ⭐⭐ |
| U-Net → DiT 전환 | 고정 구조 → Transformer 스케일링, ~9x FLOPs 절감 | ⭐⭐⭐ |

### 핵심 수식

$$N_{2D} = \frac{H}{p} \cdot \frac{W}{p}, \quad N_{3D} = \frac{T}{p_t} \cdot \frac{H}{p_h} \cdot \frac{W}{p_w}$$

$$\text{FID} \approx -a \cdot \log_{10}(\text{GFLOPs}) + b \quad \text{(DiT 스케일링 법칙)}$$

$$\text{Self-Attention Cost} = O(N^2 \cdot d) \quad \text{→ 패치 크기가 핵심}$$

### 다음 챕터 예고
**02_spatiotemporal_vae.ipynb** — 3D Causal VAE를 통한 비디오 잠재 공간 압축과 시간적 일관성 유지 원리를 다룹니다."""))

# ── 노트북 생성 ──────────────────────────────────────────────────
create_notebook(cells, 'chapter17_diffusion_transformers/01_from_unet_to_dit.ipynb')
