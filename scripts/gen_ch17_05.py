import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──
cells.append(md(r"""# Chapter 17: 비디오 생성 모델과 DiT — Sora와 HunyuanVideo 아키텍처

## 학습 목표
- Sora의 스케일링 철학과 NaViT 가변 해상도 패킹 기법을 이해한다
- HunyuanVideo의 Dual-stream → Single-stream 전환 구조를 수식으로 분석한다
- 3D Causal VAE의 시공간 압축 메커니즘을 정량적으로 파악한다
- 텍스트-비디오 멀티모달 퓨전 방식의 차이를 비교한다
- 비디오 생성 파이프라인의 전체 흐름을 구현 수준에서 이해한다

## 목차
1. [수학적 기초: NaViT와 멀티모달 퓨전](#1.-수학적-기초)
2. [라이브러리 임포트 및 환경 설정](#2.-환경-설정)
3. [NaViT 가변 해상도 패킹 시뮬레이션](#3.-NaViT-패킹)
4. [Dual-stream vs Single-stream 아키텍처 비교](#4.-Dual-vs-Single)
5. [HunyuanVideo 3D Causal VAE 압축 분석](#5.-압축-분석)
6. [비디오 생성 파이프라인 개요](#6.-파이프라인)
7. [정리](#7.-정리)"""))

# ── Cell 2: Math Section ──
cells.append(md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### NaViT 가변 해상도 패킹 (Variable Resolution Packing)

NaViT(Native Resolution ViT)는 서로 다른 해상도의 이미지/비디오를 **하나의 배치**로 패킹합니다:

$$\text{Batch} = \{x_1^{H_1 \times W_1},\; x_2^{H_2 \times W_2},\; \ldots,\; x_B^{H_B \times W_B}\}$$

각 샘플의 패치 수:

$$N_i = \frac{H_i}{p_h} \cdot \frac{W_i}{p_w}, \quad \text{총 토큰 수} = \sum_{i=1}^B N_i$$

- **패딩 없음**: 각 샘플을 원본 해상도 그대로 토큰화
- **Attention Mask**: 다른 샘플의 토큰에 attend하지 않도록 블록 대각 마스크 적용
- 핵심 장점: 정사각형 crop/resize 없이 원본 비율 유지 → 생성 품질 향상

### Dual-stream 아키텍처

텍스트와 비디오를 **분리된** Transformer 스트림으로 처리:

$$h_{\text{text}}^{(l)} = \text{SelfAttn}(h_{\text{text}}^{(l-1)}) + \text{CrossAttn}(h_{\text{text}}^{(l-1)}, h_{\text{video}}^{(l-1)})$$

$$h_{\text{video}}^{(l)} = \text{SelfAttn}(h_{\text{video}}^{(l-1)}) + \text{CrossAttn}(h_{\text{video}}^{(l-1)}, h_{\text{text}}^{(l-1)})$$

- 각 모달리티가 독자적인 Self-Attention 수행
- Cross-Attention으로 상호 정보 교환
- 파라미터가 두 배이지만, 각 모달리티의 특성을 보존

### Single-stream 아키텍처

텍스트와 비디오 토큰을 **연결(concatenate)** 후 단일 Transformer:

$$h^{(l)} = \text{SelfAttn}\!\left([h_{\text{text}}; h_{\text{video}}]^{(l-1)}\right)$$

- 시퀀스 길이: $N_{\text{text}} + N_{\text{video}}$
- 모든 토큰이 서로 attend → 자연스러운 정보 융합
- 파라미터 효율적이나 긴 시퀀스 문제

### HunyuanVideo: Dual → Single 전환

HunyuanVideo는 **앞쪽 레이어에서 Dual-stream**, **뒷쪽 레이어에서 Single-stream**을 사용합니다:

$$\text{Layer } 1 \sim L_d: \text{Dual-stream} \quad \rightarrow \quad \text{Layer } L_d+1 \sim L: \text{Single-stream}$$

- 초기: 각 모달리티가 독립적 특징 형성
- 후기: 깊은 멀티모달 융합으로 정밀한 텍스트-비디오 정합

### 3D Causal VAE 압축

$$z \in \mathbb{R}^{C_z \times (T/M_t) \times (H/M_h) \times (W/M_w)}$$

- $M_t, M_h, M_w$: 시간/공간 압축 비율
- HunyuanVideo: $M_t=4, M_h=8, M_w=8$
- 압축률: $M_t \times M_h \times M_w = 256$배

**요약 표:**

| 구분 | 수식 | 설명 |
|------|------|------|
| NaViT 패킹 | $N = \sum_i (H_i/p)(W_i/p)$ | 가변 해상도 총 토큰 수 |
| Dual-stream | 별도 Self-Attn + Cross-Attn | 모달리티 분리 처리 |
| Single-stream | $[h_{\text{text}}; h_{\text{video}}]$ 연결 | 통합 Self-Attn |
| 3D VAE 압축 | $M_t \times M_h \times M_w$ | 시공간 압축률 |

---

### 🐣 초등학생을 위한 Sora/HunyuanVideo 친절 설명!

#### 🔢 Sora가 뭔가요?

> 💡 **비유**: Sora는 "이야기를 듣고 영화를 만드는 AI 감독"이에요!

"강아지가 해변에서 뛰어노는 장면"이라고 말하면, AI가 진짜 같은 영상을 만들어줘요.
비결은 사진을 퍼즐 조각(패치)으로 나누고, Transformer가 조각들을 조합하는 거예요.

- 🧩 **NaViT**: 다양한 크기의 퍼즐을 한 상자에 담는 기술 (큰 사진, 작은 사진 모두 OK!)
- 🎬 **비디오 패치**: 시간 방향으로도 퍼즐을 나눔 (사진 → 동영상 확장)

#### 🤔 Dual-stream이랑 Single-stream은 뭐가 달라요?

> 💡 **비유**: Dual-stream은 "통역사가 있는 회의", Single-stream은 "같은 언어로 대화하는 회의"

Dual-stream: 글(텍스트)팀과 영상(비디오)팀이 따로 일한 뒤, 통역사(Cross-Attention)가 연결
Single-stream: 글과 영상을 한 테이블에 모아놓고 함께 토론

HunyuanVideo는 처음엔 Dual(따로), 나중엔 Single(함께)로 진행해요!

---

### 📝 연습 문제

#### 문제 1: NaViT 패킹 계산

패치 크기 $p=16$일 때, 해상도 $256 \times 256$, $512 \times 384$, $128 \times 128$ 세 이미지를 하나의 배치로 패킹하면 총 토큰 수는?

<details>
<summary>💡 풀이 확인</summary>

$$N_1 = \frac{256}{16} \times \frac{256}{16} = 16 \times 16 = 256$$

$$N_2 = \frac{512}{16} \times \frac{384}{16} = 32 \times 24 = 768$$

$$N_3 = \frac{128}{16} \times \frac{128}{16} = 8 \times 8 = 64$$

$$N_{\text{total}} = 256 + 768 + 64 = 1088 \text{ 토큰}$$

기존 방식(512로 리사이즈)이면 $3 \times 32 \times 32 = 3072$ 토큰 → NaViT가 **64.6% 절감**!
</details>

#### 문제 2: 3D VAE 압축률

720p 비디오 (1280×720, 30fps, 5초)를 HunyuanVideo의 3D Causal VAE ($M_t=4, M_h=8, M_w=8$, $C_z=16$)로 압축하면 latent 텐서 크기는?

<details>
<summary>💡 풀이 확인</summary>

원본: $T=150, H=720, W=1280, C=3$

$$\text{Latent}: C_z \times \frac{T}{M_t} \times \frac{H}{M_h} \times \frac{W}{M_w} = 16 \times \frac{150}{4} \times \frac{720}{8} \times \frac{1280}{8}$$
$$= 16 \times 37 \times 90 \times 160 \approx 8{,}524{,}800 \text{ 값}$$

원본 크기: $150 \times 720 \times 1280 \times 3 = 414{,}720{,}000$ 값

압축률: $414{,}720{,}000 / 8{,}524{,}800 \approx 48.6\times$
</details>"""))

# ── Cell 3: Section 2 MD ──
cells.append(md(r"""## 2. 라이브러리 임포트 및 환경 설정 <a name='2.-환경-설정'></a>"""))

# ── Cell 4: Imports ──
cells.append(code(r"""# ── 라이브러리 임포트 및 환경 설정 ──────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import time

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"GPU 사용 가능: {len(tf.config.list_physical_devices('GPU')) > 0}")"""))

# ── Cell 4: MD NaViT ──
cells.append(md(r"""## 3. NaViT 가변 해상도 패킹 시뮬레이션 <a name='3.-NaViT-패킹'></a>

NaViT는 다양한 해상도의 이미지를 하나의 배치로 패킹하여 계산 효율을 극대화합니다.
블록 대각 Attention Mask를 사용하여 서로 다른 샘플 간의 정보 누출을 방지합니다."""))

# ── Cell 5: NaViT Code ──
cells.append(code(r"""# ── NaViT 가변 해상도 패킹 시뮬레이션 ─────────────────────────────
patch_size = 16

resolutions = [
    ('A: 256x256', 256, 256),
    ('B: 512x384', 512, 384),
    ('C: 128x128', 128, 128),
    ('D: 384x256', 384, 256),
]

print("=== NaViT 패킹 분석 ===\n")
print(f"패치 크기: {patch_size}x{patch_size}\n")

total_tokens = 0
sample_sizes = []
print(f"{'샘플':<15} | {'해상도':>12} | {'패치 수':>10} | {'토큰 수':>10}")
print("-" * 55)

for name, h, w in resolutions:
    n_patches = (h // patch_size) * (w // patch_size)
    total_tokens += n_patches
    sample_sizes.append(n_patches)
    print(f"{name:<15} | {h:>4}x{w:<4}   | {(h//patch_size):>3}x{(w//patch_size):<3}  | {n_patches:>10}")

print(f"\n총 패킹 토큰 수: {total_tokens}")
max_res = max(h * w for _, h, w in resolutions)
max_side = int(np.sqrt(max_res))
naive_tokens = len(resolutions) * (max_side // patch_size) ** 2
print(f"Naive 패딩 방식 (모두 {max_side}x{max_side}로 리사이즈): {naive_tokens}")
print(f"NaViT 토큰 절감률: {(1 - total_tokens / naive_tokens) * 100:.1f}%")

# Attention Mask 시각화
mask = np.zeros((total_tokens, total_tokens))
offset = 0
for size in sample_sizes:
    mask[offset:offset+size, offset:offset+size] = 1.0
    offset += size

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
im = ax1.imshow(mask, cmap='Blues', interpolation='nearest')
offset = 0
for i, size in enumerate(sample_sizes):
    ax1.axhline(y=offset-0.5, color='red', lw=1.5)
    ax1.axvline(x=offset-0.5, color='red', lw=1.5)
    ax1.text(offset + size/2, offset + size/2, resolutions[i][0].split(':')[0],
             ha='center', va='center', fontsize=10, fontweight='bold', color='darkblue')
    offset += size
ax1.set_xlabel('Key/Value 토큰', fontsize=11)
ax1.set_ylabel('Query 토큰', fontsize=11)
ax1.set_title('NaViT 블록 대각 Attention Mask', fontweight='bold')

ax2 = axes[1]
names = [r[0].split(':')[0] for r in resolutions]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
bars = ax2.bar(names, sample_sizes, color=colors, edgecolor='white', lw=1.5)
for bar, val in zip(bars, sample_sizes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(val), ha='center', fontsize=10, fontweight='bold')
ax2.set_ylabel('토큰 수', fontsize=11)
ax2.set_title('샘플별 토큰 수 분포', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/navit_packing.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n그래프 저장됨: chapter17_diffusion_transformers/navit_packing.png")"""))

# ── Cell 6: MD Dual vs Single ──
cells.append(md(r"""## 4. Dual-stream vs Single-stream 아키텍처 비교 <a name='4.-Dual-vs-Single'></a>

HunyuanVideo는 앞쪽 레이어에서 Dual-stream (분리 처리 + Cross-Attention), 뒷쪽 레이어에서 Single-stream (연결 + Self-Attention)을 사용합니다.

두 방식의 계산량과 메모리 특성을 비교합니다."""))

# ── Cell 7: Dual vs Single Code ──
cells.append(code(r"""# ── Dual-stream vs Single-stream 비교 구현 ────────────────────────
d_model = 128
n_heads = 8
d_head = d_model // n_heads

n_text_tokens = 64
n_video_tokens = 256

# Dual-stream 블록 (간소화)
class DualStreamBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.text_self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads)
        self.video_self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads)
        self.text_cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads)
        self.video_cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads)
        self.text_ln = tf.keras.layers.LayerNormalization()
        self.video_ln = tf.keras.layers.LayerNormalization()

    def call(self, h_text, h_video):
        t_self = self.text_self_attn(h_text, h_text)
        v_self = self.video_self_attn(h_video, h_video)
        t_cross = self.text_cross_attn(h_text, h_video)
        v_cross = self.video_cross_attn(h_video, h_text)
        h_text = self.text_ln(h_text + t_self + t_cross)
        h_video = self.video_ln(h_video + v_self + v_cross)
        return h_text, h_video

# Single-stream 블록 (간소화)
class SingleStreamBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads)
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, h_text, h_video):
        h_concat = tf.concat([h_text, h_video], axis=1)
        h_out = self.self_attn(h_concat, h_concat)
        h_out = self.ln(h_concat + h_out)
        h_text_out = h_out[:, :tf.shape(h_text)[1], :]
        h_video_out = h_out[:, tf.shape(h_text)[1]:, :]
        return h_text_out, h_video_out

dual_block = DualStreamBlock(d_model, n_heads)
single_block = SingleStreamBlock(d_model, n_heads)

h_text = tf.random.normal([1, n_text_tokens, d_model])
h_video = tf.random.normal([1, n_video_tokens, d_model])

# Dual-stream forward
t0 = time.time()
for _ in range(10):
    dt_out, dv_out = dual_block(h_text, h_video)
dual_time = (time.time() - t0) / 10

# Single-stream forward
t0 = time.time()
for _ in range(10):
    st_out, sv_out = single_block(h_text, h_video)
single_time = (time.time() - t0) / 10

dual_params = sum(p.numpy().size for p in dual_block.trainable_variables)
single_params = sum(p.numpy().size for p in single_block.trainable_variables)

print("=== Dual-stream vs Single-stream 비교 ===\n")
print(f"{'항목':<20} | {'Dual-stream':>15} | {'Single-stream':>15}")
print("-" * 58)
print(f"{'텍스트 토큰 수':<20} | {n_text_tokens:>15} | {n_text_tokens:>15}")
print(f"{'비디오 토큰 수':<20} | {n_video_tokens:>15} | {n_video_tokens:>15}")
print(f"{'파라미터 수':<20} | {dual_params:>15,} | {single_params:>15,}")
print(f"{'추론 시간 (ms)':<20} | {dual_time*1000:>15.2f} | {single_time*1000:>15.2f}")
print(f"{'출력 text shape':<20} | {str(dt_out.shape):>15} | {str(st_out.shape):>15}")
print(f"{'출력 video shape':<20} | {str(dv_out.shape):>15} | {str(sv_out.shape):>15}")

print("\nHunyuanVideo 설계 전략:")
print("  - 전체 38개 블록 중 앞쪽 20개: Dual-stream (모달리티별 독립 특징 형성)")
print("  - 뒤쪽 18개: Single-stream (깊은 멀티모달 융합)")
print("  - Cross-Attention → Self-Attention 전환으로 파라미터 효율 + 성능 균형")"""))

# ── Cell 8: MD Compression ──
cells.append(md(r"""## 5. HunyuanVideo 3D Causal VAE 압축 분석 <a name='5.-압축-분석'></a>

HunyuanVideo의 3D Causal VAE는 비디오를 시간·공간 양방향으로 압축하여 DiT가 처리할 latent 공간을 생성합니다.

핵심 압축 파라미터:
- 시간 압축률: $M_t = 4$
- 공간 압축률: $M_h = M_w = 8$
- Latent 채널 수: $C_z = 16$"""))

# ── Cell 9: Compression Code ──
cells.append(code(r"""# ── HunyuanVideo 3D Causal VAE 압축 통계 ──────────────────────────
# 다양한 비디오 해상도에 대한 압축률 분석

Mt, Mh, Mw = 4, 8, 8
Cz = 16

videos = [
    ('480p-2s', 2, 24, 480, 854, 3),
    ('720p-5s', 5, 30, 720, 1280, 3),
    ('1080p-10s', 10, 30, 1080, 1920, 3),
    ('4K-5s', 5, 30, 2160, 3840, 3),
]

print("=== HunyuanVideo 3D Causal VAE 압축 분석 ===")
print(f"압축 비율: Mt={Mt}, Mh={Mh}, Mw={Mw}, Cz={Cz}\n")

headers = ['비디오', '원본 크기(MB)', 'Latent 크기(MB)', '압축률', 'DiT 토큰 수']
print(f"{'비디오':<14} | {'원본(MB)':>10} | {'Latent(MB)':>11} | {'압축률':>8} | {'토큰 수':>10}")
print("-" * 65)

patch_t, patch_h, patch_w = 1, 2, 2
token_counts = []
names = []

for name, dur, fps, H, W, C in videos:
    T = dur * fps
    original = T * H * W * C * 4
    lat_T = max(1, T // Mt)
    lat_H = H // Mh
    lat_W = W // Mw
    latent = Cz * lat_T * lat_H * lat_W * 4
    ratio = original / latent

    n_tokens = (lat_T // patch_t) * (lat_H // patch_h) * (lat_W // patch_w)
    token_counts.append(n_tokens)
    names.append(name)

    print(f"{name:<14} | {original/1e6:>10.1f} | {latent/1e6:>11.2f} | {ratio:>7.1f}x | {n_tokens:>10,}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
original_sizes = []
latent_sizes = []
for name, dur, fps, H, W, C in videos:
    T = dur * fps
    original_sizes.append(T * H * W * C * 4 / 1e6)
    lat_T = max(1, T // Mt)
    latent_sizes.append(Cz * lat_T * (H // Mh) * (W // Mw) * 4 / 1e6)

x_pos = np.arange(len(videos))
width = 0.35
bars1 = ax1.bar(x_pos - width/2, original_sizes, width, label='원본', color='#FF5722', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, latent_sizes, width, label='Latent', color='#2196F3', alpha=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(names, fontsize=9)
ax1.set_ylabel('크기 (MB, FP32)', fontsize=11)
ax1.set_title('원본 vs Latent 크기 비교', fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
bars = ax2.bar(names, token_counts, color=colors, edgecolor='white', lw=1.5)
for bar, val in zip(bars, token_counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
             f'{val:,}', ha='center', fontsize=9, fontweight='bold')
ax2.set_ylabel('DiT 입력 토큰 수', fontsize=11)
ax2.set_title('해상도별 DiT 시퀀스 길이', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/hunyuan_compression.png', dpi=100, bbox_inches='tight')
plt.close()

print("\n그래프 저장됨: chapter17_diffusion_transformers/hunyuan_compression.png")
print(f"\n핵심: 1080p-10s 비디오도 DiT가 처리 가능한 {token_counts[2]:,}개 토큰으로 압축!")"""))

# ── Cell 10: MD Pipeline ──
cells.append(md(r"""## 6. 비디오 생성 파이프라인 개요 <a name='6.-파이프라인'></a>

HunyuanVideo의 전체 비디오 생성 파이프라인:

1. **텍스트 인코딩**: MLLM (멀티모달 LLM) + CLIP으로 텍스트 임베딩 생성
2. **3D Causal VAE 인코딩**: (학습 시) 비디오 → latent 압축
3. **DiT Denoising**: Flow Matching으로 latent에서 노이즈 제거
4. **3D Causal VAE 디코딩**: latent → 비디오 복원"""))

# ── Cell 11: Pipeline Code ──
cells.append(code(r"""# ── 비디오 생성 파이프라인 단계별 시뮬레이션 ─────────────────────────
# HunyuanVideo 파이프라인의 각 단계를 텐서 shape으로 추적

print("=== HunyuanVideo 비디오 생성 파이프라인 ===\n")

# 설정
T_frames, H, W = 32, 256, 256
Mt, Mh, Mw, Cz = 4, 8, 8, 16
d_model = 128
n_text_tokens_pipe = 77
n_steps = 30

print("[ 입력 설정 ]")
print(f"  목표 비디오: {T_frames} frames x {H}x{W}")
print(f"  텍스트 프롬프트: '강아지가 해변에서 뛰어노는 장면'")
print()

# Step 1: Text Encoding
text_embed = tf.random.normal([1, n_text_tokens_pipe, d_model])
print("[ Step 1: 텍스트 인코딩 ]")
print(f"  MLLM + CLIP → text_embed: {text_embed.shape}")
print()

# Step 2: Latent shape 계산
lat_T = T_frames // Mt
lat_H = H // Mh
lat_W = W // Mw
print("[ Step 2: 3D Causal VAE Latent Space ]")
print(f"  원본 비디오: ({T_frames}, {H}, {W}, 3)")
print(f"  Latent 크기: ({Cz}, {lat_T}, {lat_H}, {lat_W})")
print(f"  시간 압축: {T_frames} → {lat_T} (/{Mt})")
print(f"  공간 압축: {H}x{W} → {lat_H}x{lat_W} (/{Mh}x{Mw})")
print()

# Step 3: DiT Denoising (Flow Matching)
n_video_patches = lat_T * (lat_H // 2) * (lat_W // 2)
print("[ Step 3: DiT Denoising (Flow Matching) ]")
print(f"  비디오 토큰 수: {n_video_patches}")
print(f"  텍스트 토큰 수: {n_text_tokens_pipe}")
print(f"  Euler ODE 스텝 수: {n_steps}")

# 간단한 Euler sampling 시뮬레이션
z = tf.random.normal([1, n_video_patches, d_model])
dt = 1.0 / n_steps
print(f"\n  시뮬레이션 (shape 추적):")
for step in [0, n_steps//3, 2*n_steps//3, n_steps-1]:
    t_val = step * dt
    noise_level = 1.0 - t_val
    print(f"    t={t_val:.2f} | 노이즈 수준: {noise_level:.2f} | z shape: {z.shape}")
print()

# Step 4: VAE Decode
print("[ Step 4: 3D Causal VAE 디코딩 ]")
print(f"  Latent ({Cz}, {lat_T}, {lat_H}, {lat_W}) → 비디오 ({T_frames}, {H}, {W}, 3)")
print()

# 전체 파이프라인 요약
print("=" * 60)
print("HunyuanVideo 아키텍처 사양 (arxiv 2412.17601)")
print("=" * 60)
specs = [
    ('DiT 파라미터', '13B'),
    ('Dual-stream 블록', '20개'),
    ('Single-stream 블록', '18개'),
    ('3D VAE 채널', '16'),
    ('시간 압축률 (Mt)', '4'),
    ('공간 압축률 (Mh, Mw)', '8, 8'),
    ('텍스트 인코더', 'MLLM + CLIP'),
    ('훈련 방식', 'Flow Matching (Rectified Flow)'),
    ('최대 해상도', '1280x720, 129 frames'),
]
for k, v in specs:
    print(f"  {k:<24}: {v}")"""))

# ── Cell 12: Architecture Diagram ──
cells.append(code(r"""# ── HunyuanVideo 아키텍처 다이어그램 ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

box_style = dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1976D2', lw=2)
box_style2 = dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#388E3C', lw=2)
box_style3 = dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#F57C00', lw=2)
box_style4 = dict(boxstyle='round,pad=0.4', facecolor='#F3E5F5', edgecolor='#7B1FA2', lw=2)

# Text Encoder
ax.text(1.5, 6.5, 'Text Encoder\n(MLLM + CLIP)', ha='center', va='center',
        fontsize=10, fontweight='bold', bbox=box_style2)

# 3D VAE Encoder
ax.text(5, 6.5, '3D Causal VAE\nEncoder', ha='center', va='center',
        fontsize=10, fontweight='bold', bbox=box_style3)

# Dual-stream
ax.text(5, 4.5, 'Dual-stream Blocks\n(20 layers)\nSelf + Cross-Attn', ha='center', va='center',
        fontsize=10, fontweight='bold', bbox=box_style)

# Single-stream
ax.text(5, 2.5, 'Single-stream Blocks\n(18 layers)\nConcat Self-Attn', ha='center', va='center',
        fontsize=10, fontweight='bold', bbox=box_style4)

# 3D VAE Decoder
ax.text(9.5, 2.5, '3D Causal VAE\nDecoder', ha='center', va='center',
        fontsize=10, fontweight='bold', bbox=box_style3)

# Output
ax.text(12.5, 2.5, 'Generated\nVideo', ha='center', va='center',
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFCDD2', edgecolor='#D32F2F', lw=2))

# Noise input
ax.text(9.5, 6.5, 'Gaussian\nNoise $z_T$', ha='center', va='center',
        fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='#616161', lw=2))

# Arrows
arrow_props = dict(arrowstyle='->', color='#333', lw=2)
ax.annotate('', xy=(3.5, 4.5), xytext=(1.5, 5.9), arrowprops=arrow_props)
ax.annotate('', xy=(5, 3.3), xytext=(5, 3.8), arrowprops=arrow_props)
ax.annotate('', xy=(5, 5.2), xytext=(5, 5.8), arrowprops=arrow_props)
ax.annotate('', xy=(8.3, 2.5), xytext=(6.8, 2.5), arrowprops=arrow_props)
ax.annotate('', xy=(11.2, 2.5), xytext=(10.8, 2.5), arrowprops=arrow_props)
ax.annotate('', xy=(7.5, 5.5), xytext=(9.5, 5.9), arrowprops=arrow_props)

# Flow Matching label
ax.text(7, 3.5, 'Flow Matching\n(Euler ODE)', ha='center', va='center',
        fontsize=9, fontstyle='italic', color='#1976D2')

ax.set_title('HunyuanVideo 전체 아키텍처 (Dual→Single Stream)', fontweight='bold', fontsize=14, pad=15)

plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/hunyuan_architecture.png', dpi=100, bbox_inches='tight')
plt.close()
print("아키텍처 다이어그램 저장됨: chapter17_diffusion_transformers/hunyuan_architecture.png")

# Sora vs HunyuanVideo 비교
print("\n=== Sora vs HunyuanVideo 비교 ===\n")
print(f"{'특성':<22} | {'Sora (OpenAI)':>22} | {'HunyuanVideo (Tencent)':>22}")
print("-" * 74)
comparisons = [
    ('공개 여부', '비공개 (기술 보고서만)', '오픈소스 (코드+가중치)'),
    ('아키텍처', 'DiT (추정)', 'Dual→Single DiT'),
    ('훈련 방식', 'Flow Matching (추정)', 'Flow Matching'),
    ('해상도 처리', 'NaViT 패킹', 'Multi-resolution 학습'),
    ('텍스트 인코더', '비공개', 'MLLM + CLIP'),
    ('VAE', '시공간 패치', '3D Causal VAE'),
    ('최대 길이', '~60초 (데모)', '~5초 (오픈소스)'),
    ('파라미터', '비공개', '13B'),
]
for feat, sora, hunyuan in comparisons:
    print(f"{feat:<22} | {sora:>22} | {hunyuan:>22}")"""))

# ── Cell 13: Summary ──
cells.append(md(r"""## 7. 정리 <a name='7.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| NaViT 패킹 | 가변 해상도를 패딩 없이 한 배치로 처리 | ⭐⭐⭐ |
| Dual-stream | 텍스트/비디오 분리 처리 + Cross-Attention | ⭐⭐⭐ |
| Single-stream | 토큰 연결 후 통합 Self-Attention | ⭐⭐⭐ |
| Dual→Single 전환 | HunyuanVideo의 핵심 설계 (20+18 블록) | ⭐⭐⭐ |
| 3D Causal VAE | 시공간 $4\times8\times8 = 256$배 압축 | ⭐⭐ |
| Flow Matching | Rectified Flow 기반 Euler ODE 샘플링 | ⭐⭐ |

### 핵심 수식

$$\text{NaViT 총 토큰} = \sum_{i=1}^B \frac{H_i}{p_h} \cdot \frac{W_i}{p_w}$$

$$h^{(l)}_{\text{single}} = \text{SelfAttn}\!\left([h_{\text{text}}; h_{\text{video}}]^{(l-1)}\right)$$

$$z \in \mathbb{R}^{C_z \times (T/M_t) \times (H/M_h) \times (W/M_w)}$$

### 다음 챕터 예고
이것으로 TensorFlow 강의 커리큘럼의 전체 이론 챕터가 마무리됩니다. **projects/** 디렉토리의 종합 실전 프로젝트로 넘어가세요!"""))

path = '/workspace/chapter17_diffusion_transformers/05_sora_and_hunyuan_architecture.ipynb'
create_notebook(cells, path)
