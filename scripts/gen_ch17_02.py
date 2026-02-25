"""Generate chapter17_diffusion_transformers/02_spatiotemporal_vae.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# Chapter 17: 비디오 생성 모델과 DiT — 시공간 VAE (Spatiotemporal VAE)

## 학습 목표
- 3D Causal VAE의 압축 수식과 잠재 공간 구조를 이해한다
- 비디오 VAE의 ELBO 목적함수를 도출하고 각 항의 역할을 분석한다
- 시간적 인과성(Temporal Causality) 제약 조건의 수학적 의미를 이해한다
- 공간/시간 압축 비율에 따른 품질 트레이드오프를 정량적으로 분석한다
- 3D 패치 토크나이징을 통해 비디오 텐서를 잠재 토큰으로 변환하는 과정을 구현한다

## 목차
1. [수학적 기초: 3D Causal VAE와 비디오 압축](#1.-수학적-기초)
2. [라이브러리 임포트 및 환경 설정](#2.-환경-설정)
3. [3D 패치 토크나이징 데모](#3.-3D-패치-토크나이징)
4. [압축 비율 vs 품질 트레이드오프 시각화](#4.-압축-비율-vs-품질)
5. [Causal vs Non-Causal 시간 컨볼루션 비교](#5.-Causal-vs-Non-Causal)
6. [비디오 잠재 공간 통계 분석](#6.-잠재-공간-통계)
7. [정리](#7.-정리)"""))

# ── Cell 2: Math foundations ──────────────────────────────────────
cells.append(md(r"""\
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### 3D Causal VAE 압축

비디오 입력 $x \in \mathbb{R}^{T \times H \times W \times 3}$을 잠재 공간으로 인코딩:

$$z \in \mathbb{R}^{C_z \times \lfloor T/M_t \rfloor \times \lfloor H/M_h \rfloor \times \lfloor W/M_w \rfloor}$$

- $M_t, M_h, M_w$: 시간, 높이, 너비 방향 압축 비율
- $C_z$: 잠재 채널 수 (일반적으로 4~16)
- 총 압축 비율: $R = M_t \times M_h \times M_w \times (3 / C_z)$

**주요 비디오 VAE 압축 설정:**

| 모델 | $M_t$ | $M_h$ | $M_w$ | $C_z$ | 총 압축 비율 |
|------|--------|--------|--------|--------|------------|
| HunyuanVideo | 4 | 8 | 8 | 16 | $4 \times 8 \times 8 \times 3/16 = 48$x |
| CogVideoX | 4 | 8 | 8 | 16 | 48x |
| Open-Sora | 4 | 8 | 8 | 4 | 192x |
| SD 이미지 VAE | 1 | 8 | 8 | 4 | 48x |

### 비디오 VAE ELBO 목적함수

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q(z|x)}\left[\|x - \hat{x}\|^2\right]}_{\text{재구성 손실}} + \underbrace{\beta \cdot D_{KL}\!\left(q(z|x) \;\|\; p(z)\right)}_{\text{KL 정규화}}$$

- $q(z|x)$: 인코더 (비디오 → 잠재 벡터 분포)
- $p(z) = \mathcal{N}(0, I)$: 사전 분포
- $\hat{x} = \text{Decoder}(z)$: 디코더 (잠재 벡터 → 비디오 복원)
- $\beta$: KL 가중치 (β-VAE에서 조절)

**KL Divergence (가우시안):**

$$D_{KL} = \frac{1}{2}\sum_{j=1}^{d_z}\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$$

### 시간적 인과성 제약 (Temporal Causality)

Causal Temporal Convolution에서는 미래 프레임이 현재에 영향을 주지 않도록 합니다:

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-k}, \quad \text{(과거 프레임만 참조)}$$

- Non-Causal: $y_t = \sum_{k=-K/2}^{K/2} w_k \cdot x_{t+k}$ (미래도 참조)
- Causal: 왼쪽 패딩으로 미래 정보 차단
- 인과성 보장 → 스트리밍 인코딩/디코딩 가능, 시간적 Flickering 억제

**요약 표:**

| 구분 | 수식 | 설명 |
|------|------|------|
| 잠재 공간 크기 | $C_z \times (T/M_t) \times (H/M_h) \times (W/M_w)$ | 압축된 비디오 표현 |
| 재구성 손실 | $\mathbb{E}[\|x - \hat{x}\|^2]$ | 원본과 복원 간 차이 |
| KL 정규화 | $\beta \cdot D_{KL}(q \| p)$ | 잠재 분포 정규화 |
| Causal Conv | $y_t = \sum_{k=0}^{K-1} w_k x_{t-k}$ | 미래 정보 차단 |

---

### 🐣 초등학생을 위한 비디오 VAE 친절 설명!

#### 🔢 VAE가 뭔가요?

> 💡 **비유**: 동영상을 아주 작은 "요약 노트"로 압축하고, 나중에 그 노트만 보고 다시 동영상을 그려내는 기계예요!

1. **인코더** (압축기): 큰 동영상 → 작은 숫자 뭉치 (잠재 벡터)
2. **디코더** (복원기): 작은 숫자 뭉치 → 동영상 복원

#### 🤔 왜 "Causal"(인과적)이어야 하나요?

> 💡 **비유**: 시험을 볼 때, 뒷장을 먼저 본 다음 앞장에 답을 쓰면 안 되죠!

비디오를 처리할 때 미래 프레임을 미리 보면 안 됩니다.
"지금까지 본 것"만으로 현재 프레임을 압축해야 나중에 실시간 생성이 가능합니다.

---

### 📝 연습 문제

#### 문제 1: 압축 비율 계산

720p 비디오 (32프레임, 1280×720, RGB)를 $M_t=4, M_h=8, M_w=8, C_z=16$으로 압축할 때:
1. 잠재 텐서의 shape은?
2. 압축 비율은?

<details>
<summary>💡 풀이 확인</summary>

1. 잠재 shape: $16 \times (32/4) \times (1280/8) \times (720/8) = 16 \times 8 \times 160 \times 90$
2. 원본 크기: $32 \times 1280 \times 720 \times 3 = 88,473,600$개 값
   잠재 크기: $16 \times 8 \times 160 \times 90 = 1,843,200$개 값
   압축 비율: $88,473,600 / 1,843,200 = 48$x

총 48배 압축됩니다. 원본 RGB 3채널이 잠재 16채널로 변환되며, 공간은 64배, 시간은 4배 압축됩니다.
</details>

#### 문제 2: KL Divergence

잠재 분포가 $q(z) = \mathcal{N}(\mu=0.5, \sigma^2=1.2)$일 때 표준 정규분포와의 KL Divergence는?

<details>
<summary>💡 풀이 확인</summary>

$$D_{KL} = \frac{1}{2}(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$$
$$= \frac{1}{2}(0.25 + 1.2 - \log 1.2 - 1) = \frac{1}{2}(0.25 + 1.2 - 0.1823 - 1) = \frac{0.2677}{2} \approx 0.134$$

$D_{KL} \approx 0.134$로, 잠재 분포가 표준 정규분포에서 약간 벗어난 상태입니다.
</details>"""))

# ── Cell 3: Section 2 header ──────────────────────────────────────
cells.append(md("""\
## 2. 라이브러리 임포트 및 환경 설정 <a name='2.-환경-설정'></a>

NumPy, TensorFlow, Matplotlib를 준비합니다. 3D 비디오 텐서 처리를 위한 기본 환경입니다."""))

# ── Cell 4: Imports ──────────────────────────────────────────────────
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
## 3. 3D 패치 토크나이징 데모 <a name='3.-3D-패치-토크나이징'></a>

비디오 텐서를 3D 시공간 패치로 분할한 뒤 선형 투영으로 토큰 시퀀스를 생성합니다.
이 과정은 DiT가 비디오를 처리하기 전의 전처리 단계입니다."""))

# ── Cell 5: 3D patch tokenization demo ──────────────────────────────
cells.append(code("""\
# ── 3D 패치 토크나이징 데모 (비디오 텐서 → 토큰) ──────────────────
class VideoTokenizer(tf.keras.layers.Layer):
    # 비디오 텐서를 시공간 패치로 분할하고 임베딩하는 토크나이저
    def __init__(self, patch_t, patch_h, patch_w, embed_dim, in_channels, **kwargs):
        super().__init__(**kwargs)
        self.pt, self.ph, self.pw = patch_t, patch_h, patch_w
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        patch_dim = patch_t * patch_h * patch_w * in_channels
        self.projection = tf.keras.layers.Dense(embed_dim)

    def call(self, x):
        B = tf.shape(x)[0]
        T, H, W, C = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        nt = T // self.pt
        nh = H // self.ph
        nw = W // self.pw

        x = tf.reshape(x, [B, nt, self.pt, nh, self.ph, nw, self.pw, C])
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
        x = tf.reshape(x, [B, nt * nh * nw, self.pt * self.ph * self.pw * C])
        tokens = self.projection(x)
        return tokens, (nt, nh, nw)

# 시뮬레이션: 잠재 공간 비디오 (VAE 출력)
# HunyuanVideo 스타일: 원본 (32, 512, 512, 3) → VAE 잠재 (8, 64, 64, 16)
latent_video = tf.random.normal([1, 8, 64, 64, 16])

tokenizer = VideoTokenizer(
    patch_t=2, patch_h=2, patch_w=2,
    embed_dim=1152, in_channels=16
)
tokens, (nt, nh, nw) = tokenizer(latent_video)

print("=" * 60)
print("3D 패치 토크나이징 결과")
print("=" * 60)
print(f"입력 잠재 비디오 shape: {latent_video.shape}")
print(f"  → (B, T_latent, H_latent, W_latent, C_latent)")
print(f"패치 크기: (pt=2, ph=2, pw=2)")
print(f"출력 토큰 shape: {tokens.shape}")
print(f"  → 시간 패치 수: {nt}")
print(f"  → 높이 패치 수: {nh}")
print(f"  → 너비 패치 수: {nw}")
print(f"  → 총 토큰 수: {nt * nh * nw:,}")
print(f"  → 각 토큰 차원: {tokens.shape[-1]}")
print(f"\\n원본 비디오 크기 (추정): 32 x 512 x 512 x 3")
print(f"  → 원본 원소 수: {32*512*512*3:,}")
print(f"  → 잠재 원소 수: {8*64*64*16:,}")
print(f"  → 토큰 원소 수: {tokens.shape[1]*tokens.shape[2]:,}")
print(f"  → 전체 압축 비율: {32*512*512*3 / (tokens.shape[1]*tokens.shape[2]):.1f}x")"""))

# ── Cell 6: Section 4 header ──────────────────────────────────────
cells.append(md(r"""\
## 4. 압축 비율 vs 품질 트레이드오프 시각화 <a name='4.-압축-비율-vs-품질'></a>

공간 압축($M_h, M_w$)과 시간 압축($M_t$)의 비율에 따라 복원 품질(PSNR)이 달라집니다.
압축을 많이 할수록 DiT의 토큰 수가 줄어 효율적이지만, 복원 품질이 저하됩니다."""))

# ── Cell 7: Compression ratio vs quality visualization ──────────────
cells.append(code("""\
# ── 압축 비율 vs 품질 트레이드오프 시각화 ────────────────────────────
# 시뮬레이션: 다양한 압축 비율에서의 PSNR 추정
# 실제 VAE 학습 없이 정보이론적 추정 + 실험적 데이터 기반

spatial_ratios = np.array([4, 8, 16, 32])
temporal_ratios = np.array([1, 2, 4, 8])

# PSNR은 압축 비율에 반비례 (로그 관계)
# 기준: SD VAE (8x8 공간, 시간 없음) → PSNR ~32-35 dB
base_psnr = 35.0
psnr_spatial = base_psnr - 4.5 * np.log2(spatial_ratios / 4)
psnr_temporal = base_psnr - 3.0 * np.log2(np.maximum(temporal_ratios, 1))
psnr_temporal[0] = base_psnr  # 시간 압축 없음

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 좌: 공간 압축 비율 vs PSNR
ax1 = axes[0]
colors_s = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
bars1 = ax1.bar(range(len(spatial_ratios)), psnr_spatial,
                color=colors_s, alpha=0.85, edgecolor='black', linewidth=0.8)
ax1.set_xticks(range(len(spatial_ratios)))
ax1.set_xticklabels([f'{r}x' for r in spatial_ratios])
ax1.set_xlabel('공간 압축 비율 ($M_h = M_w$)', fontsize=11)
ax1.set_ylabel('PSNR (dB) ↑', fontsize=11)
ax1.set_title('공간 압축 비율 vs 복원 품질', fontweight='bold')
ax1.set_ylim(20, 38)
ax1.axhline(y=30.0, color='red', ls='--', lw=1.5, alpha=0.5, label='최소 허용 품질')
for i, v in enumerate(psnr_spatial):
    ax1.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# 우: 시간 압축 비율 vs PSNR  
ax2 = axes[1]
colors_t = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
bars2 = ax2.bar(range(len(temporal_ratios)), psnr_temporal,
                color=colors_t, alpha=0.85, edgecolor='black', linewidth=0.8)
ax2.set_xticks(range(len(temporal_ratios)))
ax2.set_xticklabels([f'{r}x' for r in temporal_ratios])
ax2.set_xlabel('시간 압축 비율 ($M_t$)', fontsize=11)
ax2.set_ylabel('PSNR (dB) ↑', fontsize=11)
ax2.set_title('시간 압축 비율 vs 복원 품질', fontweight='bold')
ax2.set_ylim(20, 38)
ax2.axhline(y=30.0, color='red', ls='--', lw=1.5, alpha=0.5, label='최소 허용 품질')
for i, v in enumerate(psnr_temporal):
    ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/compression_quality_tradeoff.png',
            dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter17_diffusion_transformers/compression_quality_tradeoff.png")

# 수치 요약
print(f"\\n{'압축 설정':<25} | {'총 비율':>10} | {'PSNR':>8} | {'토큰 수(32f,512x512)':>20}")
print("-" * 72)
configs = [
    ('SD VAE (8x, 시간 없음)', 8, 8, 1, 4, 35.0),
    ('HunyuanVideo (8x, 4t)', 8, 8, 4, 16, 31.5),
    ('고압축 (16x, 4t)', 16, 16, 4, 4, 26.0),
    ('초고압축 (32x, 8t)', 32, 32, 8, 4, 22.0),
]
for name, mh, mw, mt, cz, psnr in configs:
    T, H, W = 32, 512, 512
    total_ratio = mh * mw * mt * 3 / cz
    tokens = (T // mt) * (H // mh) * (W // mw)
    print(f"{name:<25} | {total_ratio:>9.0f}x | {psnr:>6.1f}dB | {tokens:>20,}")"""))

# ── Cell 8: Section 5 header ──────────────────────────────────────
cells.append(md(r"""\
## 5. Causal vs Non-Causal 시간 컨볼루션 비교 <a name='5.-Causal-vs-Non-Causal'></a>

Causal 컨볼루션은 미래 프레임 정보를 차단합니다:
- **Non-Causal**: $y_t = \sum_{k=-K/2}^{K/2} w_k \cdot x_{t+k}$ (양방향)
- **Causal**: $y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-k}$ (단방향, 패딩 포함)

Causal 조건은 비디오 생성 시 **자기회귀적 시간 일관성**과 **스트리밍 생성**을 가능하게 합니다."""))

# ── Cell 9: Causal vs Non-Causal comparison code ──────────────────
cells.append(code("""\
# ── Causal vs Non-Causal 시간 컨볼루션 비교 ────────────────────────
class CausalTemporalConv1D(tf.keras.layers.Layer):
    # Causal Conv: 왼쪽 패딩으로 미래 정보 차단
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.pad_size = kernel_size - 1
        self.conv = tf.keras.layers.Conv1D(
            filters, kernel_size, padding='valid', use_bias=True
        )

    def call(self, x):
        # 왼쪽(과거) 방향으로만 패딩
        padded = tf.pad(x, [[0, 0], [self.pad_size, 0], [0, 0]])
        return self.conv(padded)


class NonCausalTemporalConv1D(tf.keras.layers.Layer):
    # Non-Causal Conv: 양방향 참조 (same 패딩)
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(
            filters, kernel_size, padding='same', use_bias=True
        )

    def call(self, x):
        return self.conv(x)


# 테스트: 시간 축 시퀀스 (8 프레임, 채널 4)
T_frames = 8
x_temporal = tf.random.normal([1, T_frames, 4])

causal_conv = CausalTemporalConv1D(filters=4, kernel_size=3)
noncausal_conv = NonCausalTemporalConv1D(filters=4, kernel_size=3)

y_causal = causal_conv(x_temporal)
y_noncausal = noncausal_conv(x_temporal)

print("=" * 60)
print("Causal vs Non-Causal 시간 컨볼루션 비교")
print("=" * 60)
print(f"입력 shape: {x_temporal.shape}  (B, T, C)")
print(f"커널 크기: 3")
print(f"Causal 출력 shape: {y_causal.shape}")
print(f"Non-Causal 출력 shape: {y_noncausal.shape}")

# 인과성 검증: t=0 프레임의 출력이 x[0]만 의존하는지 확인
print(f"\\n인과성 검증:")
x_test_a = tf.constant([[[1.0, 0, 0, 0],
                          [0.0, 0, 0, 0],
                          [0.0, 0, 0, 0],
                          [0.0, 0, 0, 0],
                          [0.0, 0, 0, 0],
                          [0.0, 0, 0, 0],
                          [0.0, 0, 0, 0],
                          [0.0, 0, 0, 0]]])
x_test_b = tf.constant([[[1.0, 0, 0, 0],
                          [9.9, 9, 9, 9],
                          [9.9, 9, 9, 9],
                          [0.0, 0, 0, 0],
                          [0.0, 0, 0, 0],
                          [0.0, 0, 0, 0],
                          [0.0, 0, 0, 0],
                          [0.0, 0, 0, 0]]])

y_a_causal = causal_conv(x_test_a)
y_b_causal = causal_conv(x_test_b)
y_a_noncausal = noncausal_conv(x_test_a)
y_b_noncausal = noncausal_conv(x_test_b)

diff_causal_t0 = tf.reduce_sum(tf.abs(y_a_causal[0, 0] - y_b_causal[0, 0])).numpy()
diff_noncausal_t0 = tf.reduce_sum(tf.abs(y_a_noncausal[0, 0] - y_b_noncausal[0, 0])).numpy()

print(f"  미래 프레임 변경 시 t=0 출력 변화 (Causal):     {diff_causal_t0:.6f}")
print(f"  미래 프레임 변경 시 t=0 출력 변화 (Non-Causal): {diff_noncausal_t0:.6f}")
print(f"  → Causal: 미래 변경에 영향 없음 ({'통과' if diff_causal_t0 < 1e-5 else '실패'})")
print(f"  → Non-Causal: 미래 변경에 영향 있음 ({'미래 의존 확인' if diff_noncausal_t0 > 1e-5 else '이상'})")

print(f"\\n비교 요약:")
print(f"{'항목':<25} | {'Causal':>15} | {'Non-Causal':>15}")
print("-" * 60)
print(f"{'미래 프레임 의존성':<25} | {'없음 (안전)':>15} | {'있음 (위험)':>15}")
print(f"{'패딩 방식':<25} | {'왼쪽(과거)만':>15} | {'양쪽(same)':>15}")
print(f"{'스트리밍 생성':<25} | {'가능':>15} | {'불가능':>15}")
print(f"{'시간 일관성':<25} | {'높음':>15} | {'Flickering 위험':>15}")"""))

# ── Cell 10: Section 6 header ──────────────────────────────────────
cells.append(md("""\
## 6. 비디오 잠재 공간 통계 분석 <a name='6.-잠재-공간-통계'></a>

VAE 인코더 출력의 잠재 벡터 분포를 분석합니다.
잘 학습된 VAE는 잠재 분포가 $\\mathcal{N}(0, I)$에 가까워야 합니다."""))

# ── Cell 11: Video latent space statistics ────────────────────────
cells.append(code("""\
# ── 비디오 잠재 공간 통계 시뮬레이션 ────────────────────────────────
# 가상 VAE 인코더 출력 시뮬레이션 (학습 전/후 비교)

# 학습 전: 정규화되지 않은 잠재 분포
z_untrained_mu = np.random.normal(loc=2.0, scale=1.5, size=(1000, 16))
z_untrained_logvar = np.random.normal(loc=0.5, scale=0.8, size=(1000, 16))
z_untrained = z_untrained_mu + np.exp(0.5 * z_untrained_logvar) * np.random.randn(1000, 16)

# 학습 후: N(0, I)에 가까운 잠재 분포
z_trained_mu = np.random.normal(loc=0.0, scale=0.1, size=(1000, 16))
z_trained_logvar = np.random.normal(loc=-0.05, scale=0.2, size=(1000, 16))
z_trained = z_trained_mu + np.exp(0.5 * z_trained_logvar) * np.random.randn(1000, 16)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 좌: 잠재 벡터 분포 (히스토그램)
ax1 = axes[0]
ax1.hist(z_untrained.flatten(), bins=60, alpha=0.6, color='#F44336',
         density=True, label='학습 전', edgecolor='black', linewidth=0.3)
ax1.hist(z_trained.flatten(), bins=60, alpha=0.6, color='#2196F3',
         density=True, label='학습 후', edgecolor='black', linewidth=0.3)
x_range = np.linspace(-4, 4, 200)
ax1.plot(x_range, 1/np.sqrt(2*np.pi) * np.exp(-x_range**2/2),
         'k--', lw=2, label='$\\mathcal{N}(0,1)$ 목표')
ax1.set_xlabel('잠재 값 z', fontsize=11)
ax1.set_ylabel('확률 밀도', fontsize=11)
ax1.set_title('잠재 벡터 분포', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 중: 채널별 평균
ax2 = axes[1]
ch_mu_untrained = z_untrained.mean(axis=0)
ch_mu_trained = z_trained.mean(axis=0)
x_ch = np.arange(16)
ax2.bar(x_ch - 0.2, ch_mu_untrained, 0.4, color='#F44336', alpha=0.7,
        label='학습 전', edgecolor='black', linewidth=0.5)
ax2.bar(x_ch + 0.2, ch_mu_trained, 0.4, color='#2196F3', alpha=0.7,
        label='학습 후', edgecolor='black', linewidth=0.5)
ax2.axhline(y=0, color='black', ls='-', lw=1)
ax2.set_xlabel('잠재 채널', fontsize=11)
ax2.set_ylabel('평균 $\\mu$', fontsize=11)
ax2.set_title('채널별 평균 (목표: 0)', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 우: 채널별 표준편차
ax3 = axes[2]
ch_std_untrained = z_untrained.std(axis=0)
ch_std_trained = z_trained.std(axis=0)
ax3.bar(x_ch - 0.2, ch_std_untrained, 0.4, color='#F44336', alpha=0.7,
        label='학습 전', edgecolor='black', linewidth=0.5)
ax3.bar(x_ch + 0.2, ch_std_trained, 0.4, color='#2196F3', alpha=0.7,
        label='학습 후', edgecolor='black', linewidth=0.5)
ax3.axhline(y=1.0, color='green', ls='--', lw=1.5, label='$\\sigma=1$ 목표')
ax3.set_xlabel('잠재 채널', fontsize=11)
ax3.set_ylabel('표준편차 $\\sigma$', fontsize=11)
ax3.set_title('채널별 표준편차 (목표: 1)', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/latent_space_statistics.png',
            dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter17_diffusion_transformers/latent_space_statistics.png")

# KL Divergence 계산
def compute_kl_divergence(mu, logvar):
    return 0.5 * np.mean(np.sum(mu**2 + np.exp(logvar) - logvar - 1, axis=1))

kl_untrained = compute_kl_divergence(z_untrained_mu, z_untrained_logvar)
kl_trained = compute_kl_divergence(z_trained_mu, z_trained_logvar)

print(f"\\n잠재 공간 통계 요약:")
print(f"{'항목':<20} | {'학습 전':>12} | {'학습 후':>12} | {'목표값':>10}")
print("-" * 60)
print(f"{'전체 평균':<20} | {z_untrained.mean():>12.4f} | {z_trained.mean():>12.4f} | {'0.0':>10}")
print(f"{'전체 표준편차':<20} | {z_untrained.std():>12.4f} | {z_trained.std():>12.4f} | {'1.0':>10}")
print(f"{'KL Divergence':<20} | {kl_untrained:>12.4f} | {kl_trained:>12.4f} | {'0.0':>10}")
print(f"\\n→ 학습 후 잠재 분포가 N(0,I)에 크게 근접 (KL: {kl_untrained:.2f} → {kl_trained:.2f})")"""))

# ── Cell 12: Additional analysis - Full pipeline ──────────────────
cells.append(code("""\
# ── 전체 VAE 파이프라인 시뮬레이션 ────────────────────────────────
# 비디오 → VAE 인코딩 → 잠재 공간 → 3D 패치 → DiT 토큰

print("=" * 65)
print("비디오 VAE → DiT 토큰 전체 파이프라인")
print("=" * 65)

# 다양한 비디오 해상도 시뮬레이션
configs = [
    ("480p 짧은 영상", 16, 640, 480, 3),
    ("720p 중간 영상", 32, 1280, 720, 3),
    ("1080p 긴 영상", 64, 1920, 1080, 3),
    ("4K 짧은 영상", 16, 3840, 2160, 3),
]

# VAE 압축 설정 (HunyuanVideo 스타일)
Mt, Mh, Mw, Cz = 4, 8, 8, 16
# DiT 패치 크기
pt, ph, pw = 1, 2, 2

print(f"\\nVAE 압축: Mt={Mt}, Mh={Mh}, Mw={Mw}, Cz={Cz}")
print(f"DiT 패치: pt={pt}, ph={ph}, pw={pw}")
print()
print(f"{'비디오 설정':<20} | {'원본 크기':>14} | {'잠재 크기':>14} | {'DiT 토큰':>10} | {'메모리(fp16)':>12}")
print("-" * 78)

for name, T, H, W, C in configs:
    original = T * H * W * C
    lat_T, lat_H, lat_W = T // Mt, H // Mh, W // Mw
    latent = Cz * lat_T * lat_H * lat_W
    n_tokens = (lat_T // pt) * (lat_H // ph) * (lat_W // pw)
    mem_fp16 = n_tokens * 1152 * 2 / (1024**2)

    orig_str = f"{original/1e6:.1f}M"
    lat_str = f"{latent/1e6:.2f}M"
    mem_str = f"{mem_fp16:.1f} MB"
    print(f"{name:<20} | {orig_str:>14} | {lat_str:>14} | {n_tokens:>10,} | {mem_str:>12}")

print(f"\\n핵심 인사이트:")
print(f"  1. VAE 압축으로 원본 대비 ~48x 데이터 감소")
print(f"  2. 추가 패치화로 토큰 수 추가 감소 → Transformer 처리 가능")
print(f"  3. 고해상도/긴 비디오 → 토큰 수 급증 → Sparse Attention 필요")"""))

# ── Cell 13: Summary ────────────────────────────────────────────────
cells.append(md(r"""\
## 7. 정리 <a name='7.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| 3D Causal VAE | 비디오를 시공간 압축하여 잠재 공간으로 인코딩 | ⭐⭐⭐ |
| 압축 비율 $(M_t, M_h, M_w)$ | 시간·공간 각 축의 다운샘플링 비율 | ⭐⭐⭐ |
| VAE ELBO | 재구성 손실 + β·KL 정규화 | ⭐⭐⭐ |
| Causal Conv | 미래 프레임 참조 차단 → 시간 일관성 보장 | ⭐⭐ |
| 잠재 분포 정규화 | $q(z|x) \approx \mathcal{N}(0, I)$ 목표 | ⭐⭐ |
| Flickering 억제 | Causal 구조 + 시간적 일관성 손실로 해결 | ⭐⭐ |

### 핵심 수식

$$z \in \mathbb{R}^{C_z \times (T/M_t) \times (H/M_h) \times (W/M_w)}$$

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}[\|x - \hat{x}\|^2] + \beta \cdot D_{KL}(q(z|x) \| p(z))$$

$$y_t^{\text{causal}} = \sum_{k=0}^{K-1} w_k \cdot x_{t-k} \quad \text{(미래 프레임 차단)}$$

### 다음 챕터 예고
**03_dit_conditioning_and_adaln.ipynb** — adaLN-Zero를 통한 조건 주입 방식과 DiT에서의 Classifier-Free Guidance 설계를 다룹니다."""))

# ── 노트북 생성 ──────────────────────────────────────────────────
create_notebook(cells, 'chapter17_diffusion_transformers/02_spatiotemporal_vae.ipynb')
