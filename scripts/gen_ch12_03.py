"""Generate Chapter 12-03: Rotary Position Embedding."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helper import md, code, create_notebook

cells = [
md("""# Chapter 12: 최신 LLM 아키텍처 — Rotary Position Embedding (RoPE)

## 학습 목표
- **복소수 평면**에서 RoPE의 수식을 도출하고, 회전 행렬과의 관계를 이해한다
- RoPE가 **상대 위치 정보**를 자연스럽게 인코딩하는 수학적 원리를 증명한다
- TensorFlow로 RoPE를 **두 가지 방식**(회전 행렬, 복소수)으로 구현한다
- 위치 간 **attention score 감쇠** 패턴을 시각화하고 분석한다
- **YaRN**을 통한 컨텍스트 창 확장 원리를 이해하고 시뮬레이션한다

## 목차
1. [수학적 기초: 복소수 평면과 RoPE](#1.-수학적-기초)
2. [RoPE 구현 (회전 행렬 vs 복소수)](#2.-RoPE-구현)
3. [위치별 Attention Score 감쇠 분석](#3.-Attention-감쇠)
4. [YaRN 컨텍스트 확장](#4.-YaRN)
5. [정리](#5.-정리)"""),

md(r"""## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### 복소수 평면에서의 위치 인코딩

RoPE는 쿼리/키 벡터를 **위치에 따라 회전**시켜 상대 위치 정보를 주입합니다.

2D 부분 공간 $(q_{2i}, q_{2i+1})$에 위치 $m$의 회전을 적용하면:

$$f(q, m) = \begin{pmatrix} q_{2i} \cos(m\theta_i) - q_{2i+1} \sin(m\theta_i) \\ q_{2i+1} \cos(m\theta_i) + q_{2i} \sin(m\theta_i) \end{pmatrix}$$

여기서 주파수 $\theta_i$는:

$$\theta_i = \text{base}^{-2i/d}, \quad i = 0, 1, \dots, d/2 - 1$$

- $\text{base}$: 기본 주파수 (Llama 3: $500{,}000$)
- $d$: 헤드 차원 ($d_{head} = 128$)
- $m$: 토큰의 절대 위치

### 핵심 성질: 상대 위치 인코딩

$$\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)$$

즉, RoPE가 적용된 Q와 K의 내적은 **상대 위치 $m - n$에만 의존**합니다!

**증명 스케치** (복소수 표현):

$$f(q, m) = q \cdot e^{im\theta}, \quad f(k, n) = k \cdot e^{in\theta}$$

$$\text{Re}[f(q,m) \cdot \overline{f(k,n)}] = \text{Re}[q \bar{k} \cdot e^{i(m-n)\theta}]$$

→ 내적이 $m - n$ (상대 거리)에만 의존!

### 회전 행렬 표현

전체 $d$-차원에 대한 블록 대각 회전 행렬:

$$R_{\Theta, m} = \begin{pmatrix}
\cos m\theta_0 & -\sin m\theta_0 & & \\
\sin m\theta_0 & \cos m\theta_0 & & \\
& & \ddots & \\
& & & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\
& & & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1}
\end{pmatrix}$$

**요약 표:**

| 항목 | Sinusoidal PE | RoPE |
|------|-------------|------|
| 적용 위치 | 임베딩에 더함 ($x + PE$) | Q, K에 회전 ($R \cdot q$) |
| 상대 위치 | 간접적 | 수학적으로 보장 |
| 학습 파라미터 | 없음 | 없음 |
| 확장성 | 고정 최대 길이 | base 조정으로 확장 (YaRN) |
| 사용 모델 | Transformer (원본) | Llama, Mistral, Qwen 등 |

---

### 🐣 초등학생을 위한 RoPE 친절 설명!

#### 🔄 RoPE가 뭔가요?

> 💡 **비유**: **시계 바늘**을 상상해보세요! 각 단어마다 시계 바늘을 다른 각도로 돌려요.
> - 1번째 단어: 10도 회전
> - 2번째 단어: 20도 회전
> - 3번째 단어: 30도 회전
>
> 이렇게 하면, 두 단어 사이의 **거리**(몇 칸 떨어져 있는지)가 자연스럽게 각도 차이로 표현됩니다!

| 위치 인코딩 | 비유 | 장점 |
|------------|------|------|
| 절대 위치 | 좌석 번호표 | 간단 |
| 상대 위치 | 두 사람 사이 거리 | 관계 파악 쉬움 |
| RoPE | 시계 바늘 각도 차이 | 절대+상대 동시에! |"""),

md(r"""---

### 📝 연습 문제

#### 문제 1: RoPE 주파수 계산

Llama 3의 $\text{base}=500{,}000$, $d_{head}=128$일 때, $\theta_0$과 $\theta_{63}$의 값을 구하시오.

<details>
<summary>💡 풀이 확인</summary>

$$\theta_0 = 500000^{-0/128} = 500000^0 = 1.0$$

$$\theta_{63} = 500000^{-126/128} = 500000^{-0.984} \approx 2.89 \times 10^{-6}$$

$\theta_0$은 가장 빠르게 회전 (고주파), $\theta_{63}$은 극히 느리게 회전 (저주파)
→ 다양한 스케일의 위치 정보를 동시에 인코딩!
</details>

#### 문제 2: 상대 위치 보존 증명

위치 $m=10$의 Q와 위치 $n=5$의 K에 RoPE를 적용한 후의 내적이, 위치 $m=100$의 Q와 $n=95$의 K에 RoPE를 적용한 후의 내적과 같음을 설명하시오 (동일 $q, k$ 가정).

<details>
<summary>💡 풀이 확인</summary>

두 경우 모두 상대 거리 $m - n = 5$이므로:

$$\text{Re}[q\bar{k} \cdot e^{i(m-n)\theta}] = \text{Re}[q\bar{k} \cdot e^{i \cdot 5 \cdot \theta}]$$

RoPE의 내적은 $m - n$에만 의존하므로, 절대 위치가 달라도 **상대 거리가 같으면 동일한 attention score**를 생성합니다. 이것이 RoPE의 핵심 성질입니다.
</details>"""),

code("""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")"""),

md("""## 2. RoPE 구현 (회전 행렬 vs 복소수) <a name='2.-RoPE-구현'></a>"""),

code(r"""# ── RoPE 구현: 방법 1 (회전 행렬) & 방법 2 (복소수) ───────────
# 두 가지 동치(equivalent)한 구현 방식을 비교합니다

def precompute_freqs(d_head, max_seq_len, base=500000.0):
    # theta_i = base^(-2i/d), i = 0, 1, ..., d/2-1
    freqs = 1.0 / (base ** (np.arange(0, d_head, 2, dtype=np.float32) / d_head))
    # 위치별 각도: [max_seq_len, d/2]
    positions = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)  # [S, d/2]
    return angles

def rope_rotation_matrix(x, angles):
    # 방법 1: 회전 행렬 방식
    # x: [B, S, d], angles: [S, d/2]
    B, S, d = x.shape
    x_pairs = tf.reshape(x, (B, S, d // 2, 2))  # [..., (x0, x1)]

    cos_a = tf.cast(tf.cos(angles[:S]), tf.float32)  # [S, d/2]
    sin_a = tf.cast(tf.sin(angles[:S]), tf.float32)

    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]  # [B, S, d/2]

    # 2D 회전: [x0 cos - x1 sin, x0 sin + x1 cos]
    rot_x0 = x0 * cos_a - x1 * sin_a
    rot_x1 = x0 * sin_a + x1 * cos_a

    rotated = tf.stack([rot_x0, rot_x1], axis=-1)  # [B, S, d/2, 2]
    return tf.reshape(rotated, (B, S, d))

def rope_complex(x, angles):
    # 방법 2: 복소수 방식 (간결한 구현)
    # x를 복소수로 취급: x_complex = x0 + i*x1
    B, S, d = x.shape
    x_pairs = tf.reshape(x, (B, S, d // 2, 2))
    x_complex = tf.complex(x_pairs[..., 0], x_pairs[..., 1])  # [B, S, d/2]

    # e^(i*angle) = cos + i*sin
    angles_s = tf.cast(angles[:S], tf.float32)
    freqs_complex = tf.complex(tf.cos(angles_s), tf.sin(angles_s))  # [S, d/2]

    # 복소수 곱 = 회전
    rotated_complex = x_complex * freqs_complex  # [B, S, d/2]

    # 실수부/허수부 분리
    rotated = tf.stack([tf.math.real(rotated_complex),
                        tf.math.imag(rotated_complex)], axis=-1)
    return tf.reshape(rotated, (B, S, d))


# 테스트
d_head = 128
max_seq = 512
angles = precompute_freqs(d_head, max_seq, base=500000.0)

x_test = tf.random.normal((2, 64, d_head))

out_rotation = rope_rotation_matrix(x_test, angles)
out_complex = rope_complex(x_test, angles)

# 두 방법의 결과가 동일한지 확인
diff = tf.reduce_max(tf.abs(out_rotation - out_complex)).numpy()

print("=" * 55)
print("RoPE 두 가지 구현 비교")
print("=" * 55)
print(f"입력 shape: {x_test.shape}")
print(f"d_head: {d_head}, base: 500,000")
print(f"회전 행렬 출력 shape: {out_rotation.shape}")
print(f"복소수 출력 shape:   {out_complex.shape}")
print(f"두 방법 최대 오차:   {diff:.2e}")
print(f"동치 여부: {'✅ 동일' if diff < 1e-5 else '❌ 차이 있음'}")
print()
print("주파수 범위:")
print(f"  theta_0 (고주파):   {1.0/500000**(0/d_head):.6f}")
print(f"  theta_63 (저주파): {1.0/500000**(126/d_head):.2e}")"""),

code(r"""# ── RoPE 주파수 및 회전 패턴 시각화 ──────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1) 주파수 분포
ax1 = axes[0]
dims = np.arange(0, d_head, 2)
freqs = 1.0 / (500000.0 ** (dims / d_head))
ax1.semilogy(dims, freqs, 'b-o', ms=3, lw=1.5)
ax1.set_xlabel('차원 인덱스 (2i)', fontsize=11)
ax1.set_ylabel('주파수 θ_i (log scale)', fontsize=11)
ax1.set_title('RoPE 주파수 분포', fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2) 위치별 cos/sin 패턴 (저/고 주파수)
ax2 = axes[1]
positions = np.arange(128)
ax2.plot(positions, np.cos(positions * freqs[0]), 'r-', lw=2, label=f'cos(m·θ₀), θ₀={freqs[0]:.2f}', alpha=0.8)
ax2.plot(positions, np.cos(positions * freqs[16]), 'b-', lw=2, label=f'cos(m·θ₁₆), θ₁₆={freqs[16]:.2e}', alpha=0.8)
ax2.plot(positions, np.cos(positions * freqs[63]), 'g-', lw=2, label=f'cos(m·θ₆₃), θ₆₃={freqs[63]:.2e}', alpha=0.8)
ax2.set_xlabel('위치 m', fontsize=11)
ax2.set_ylabel('cos(m·θ)', fontsize=11)
ax2.set_title('위치별 회전 패턴', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3) 2D 회전 시각화
ax3 = axes[2]
theta = freqs[0]  # 고주파
q_original = np.array([1.0, 0.0])
for m in range(0, 16, 2):
    angle = m * theta
    q_rot = np.array([q_original[0]*np.cos(angle) - q_original[1]*np.sin(angle),
                       q_original[0]*np.sin(angle) + q_original[1]*np.cos(angle)])
    ax3.annotate('', xy=q_rot, xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=plt.cm.viridis(m/16), lw=2))
    ax3.text(q_rot[0]*1.15, q_rot[1]*1.15, f'm={m}', fontsize=8, ha='center')

circle = plt.Circle((0, 0), 1, fill=False, color='gray', ls='--', lw=1)
ax3.add_patch(circle)
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_aspect('equal')
ax3.set_xlabel('q₀', fontsize=11)
ax3.set_ylabel('q₁', fontsize=11)
ax3.set_title('복소수 평면에서의 회전', fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter12_modern_llms/rope_visualization.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter12_modern_llms/rope_visualization.png")"""),

md("""## 3. 위치별 Attention Score 감쇠 분석 <a name='3.-Attention-감쇠'></a>"""),

code(r"""# ── RoPE 적용 후 상대 위치에 따른 Attention Score 감쇠 ─────
# 상대 거리가 멀어질수록 attention score가 어떻게 변하는지 분석

d_head = 128
max_pos = 256
angles = precompute_freqs(d_head, max_pos, base=500000.0)

# 고정된 Q, K 벡터 (위치 0)
q_base = tf.random.normal((1, 1, d_head))
k_base = tf.random.normal((1, 1, d_head))

# 다양한 상대 거리에 대해 attention score 계산
relative_distances = np.arange(0, max_pos)
scores = []

for dist in relative_distances:
    # Q: 위치 0, K: 위치 dist
    q_pos = rope_rotation_matrix(q_base, angles)  # 위치 0에서의 Q (angles[0])
    
    # K를 dist 위치에 배치
    k_shifted = tf.random.normal((1, 1, d_head))  # K 벡터
    # angles를 dist 위치로 설정
    k_angles = precompute_freqs(d_head, dist + 1, base=500000.0)
    k_at_pos = rope_rotation_matrix(k_base, k_angles)
    
    # Q(0) · K(dist) 내적
    score = tf.reduce_sum(q_pos * k_at_pos).numpy()
    scores.append(score)

scores = np.array(scores)
scale = np.sqrt(d_head)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: Attention Score vs 상대 거리
ax1 = axes[0]
ax1.plot(relative_distances, scores / scale, 'b-', lw=1.5, alpha=0.8)
ax1.axhline(y=0, color='gray', ls='--', lw=1)
ax1.set_xlabel('상대 거리 (|m - n|)', fontsize=11)
ax1.set_ylabel('Attention Score (scaled)', fontsize=11)
ax1.set_title('RoPE: 상대 거리별 Attention Score', fontweight='bold')
ax1.grid(True, alpha=0.3)

# 오른쪽: Attention Weight Heatmap (짧은 시퀀스)
ax2 = axes[1]
seq = 32
q_all = tf.random.normal((1, seq, d_head))
k_all = tf.random.normal((1, seq, d_head))

angles_short = precompute_freqs(d_head, seq, base=500000.0)
q_rope = rope_rotation_matrix(q_all, angles_short)
k_rope = rope_rotation_matrix(k_all, angles_short)

# Attention matrix
attn_scores = tf.matmul(q_rope, k_rope, transpose_b=True) / tf.sqrt(float(d_head))
attn_weights = tf.nn.softmax(attn_scores, axis=-1).numpy()[0]

im = ax2.imshow(attn_weights, cmap='Blues', aspect='auto')
ax2.set_xlabel('Key 위치', fontsize=11)
ax2.set_ylabel('Query 위치', fontsize=11)
ax2.set_title('RoPE Attention Weight Map', fontweight='bold')
plt.colorbar(im, ax=ax2, shrink=0.8)

plt.tight_layout()
plt.savefig('chapter12_modern_llms/rope_attention_decay.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter12_modern_llms/rope_attention_decay.png")
print()
print("관찰:")
print("  • RoPE는 가까운 토큰에 더 높은 attention을 부여하는 경향")
print("  • 저주파 차원은 먼 거리의 위치 정보를, 고주파 차원은 가까운 정보를 인코딩")"""),

md("""## 4. YaRN 컨텍스트 확장 <a name='4.-YaRN'></a>"""),

code(r"""# ── YaRN (Yet another RoPE extensioN) 컨텍스트 확장 ─────────
# 학습된 RoPE를 더 긴 시퀀스에 적용하기 위한 주파수 재조정 기법

def yarn_freqs(d_head, max_seq, base=500000.0, scale_factor=4.0, 
               beta_fast=32, beta_slow=1):
    # YaRN: 주파수별 차등 스케일링
    # 고주파(가까운 위치) → 그대로, 저주파(먼 위치) → 스케일 조정
    freqs = 1.0 / (base ** (np.arange(0, d_head, 2, dtype=np.float32) / d_head))
    
    # 주파수별 보간 비율 계산
    low_freq_factor = max_seq / (2 * np.pi / freqs)
    
    # 선형 보간: 고주파는 원본, 저주파는 스케일링
    alpha = np.clip((low_freq_factor - beta_slow) / (beta_fast - beta_slow), 0, 1)
    
    # NTK-aware 보간
    scaled_freqs = freqs / scale_factor
    yarn_freqs = alpha * freqs + (1 - alpha) * scaled_freqs
    
    positions = np.arange(max_seq * scale_factor, dtype=np.float32)
    angles = np.outer(positions, yarn_freqs)
    return angles, yarn_freqs

# 비교: 원본 vs NTK 스케일링 vs YaRN
original_max = 8192   # 원래 학습된 길이
extended_max = 32768   # 4배 확장 목표
scale = extended_max / original_max

# 원본 주파수
orig_freqs = 1.0 / (500000.0 ** (np.arange(0, d_head, 2) / d_head))
# NTK 스케일링: base를 키움
ntk_base = 500000.0 * scale ** (d_head / (d_head - 2))
ntk_freqs = 1.0 / (ntk_base ** (np.arange(0, d_head, 2) / d_head))
# YaRN
_, yarn_f = yarn_freqs(d_head, original_max, scale_factor=scale)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: 주파수 비교
ax1 = axes[0]
dims = np.arange(0, d_head, 2)
ax1.semilogy(dims, orig_freqs, 'b-o', ms=2, lw=1.5, label='원본 (8K)')
ax1.semilogy(dims, ntk_freqs, 'r-s', ms=2, lw=1.5, label='NTK 스케일링', alpha=0.7)
ax1.semilogy(dims, yarn_f, 'g-^', ms=2, lw=1.5, label='YaRN (32K)', alpha=0.7)
ax1.set_xlabel('차원 인덱스', fontsize=11)
ax1.set_ylabel('주파수 (log)', fontsize=11)
ax1.set_title('RoPE 확장 방법별 주파수 비교', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 오른쪽: 위치별 회전 각도 비교 (dim=0, 고주파)
ax2 = axes[1]
positions = np.arange(0, extended_max, 100)
ax2.plot(positions, np.cos(positions * orig_freqs[0]), 'b-', lw=1, label='원본', alpha=0.5)
ax2.plot(positions, np.cos(positions * yarn_f[0]), 'g-', lw=1.5, label='YaRN')
ax2.axvline(x=original_max, color='red', ls='--', lw=2, label=f'원래 최대 ({original_max})')
ax2.set_xlabel('위치', fontsize=11)
ax2.set_ylabel('cos(m·θ₀)', fontsize=11)
ax2.set_title('확장 영역에서의 회전 패턴', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter12_modern_llms/yarn_extension.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter12_modern_llms/yarn_extension.png")
print()
print("YaRN 컨텍스트 확장 원리:")
print(f"  원래 컨텍스트: {original_max:,} 토큰")
print(f"  확장 목표:     {extended_max:,} 토큰 ({scale:.0f}x)")
print()
print("  • 고주파 차원 (가까운 위치 인코딩): 그대로 유지")
print("  • 저주파 차원 (먼 위치 인코딩): 스케일 팩터로 조정")
print("  • 결과: 기존 짧은 범위 성능 유지 + 긴 범위 외삽 가능")"""),

md(r"""## 5. 정리 <a name='5.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| RoPE | 복소수 회전으로 상대 위치를 인코딩 → Q, K에 적용 | ⭐⭐⭐ |
| 주파수 설계 | $\theta_i = \text{base}^{-2i/d}$ → 다중 스케일 위치 정보 | ⭐⭐⭐ |
| 상대 위치 보존 | $\langle R_m q, R_n k \rangle = g(m-n)$ → 내적이 상대 거리에만 의존 | ⭐⭐⭐ |
| YaRN | 고/저주파 차등 스케일링으로 컨텍스트 확장 | ⭐⭐ |

### 핵심 수식

$$f(q, m) = q \cdot e^{im\theta}, \quad \theta_i = \text{base}^{-2i/d}$$

$$\text{RoPE 내적: } \langle f(q,m), f(k,n) \rangle = g(q, k, m-n)$$

### Llama 3 RoPE 설정

| 항목 | 값 |
|------|-----|
| base | 500,000 |
| $d_{head}$ | 128 |
| 적용 대상 | Q, K (V에는 미적용) |

### 다음 챕터 예고
**Chapter 12-04: MoE 라우팅과 부하 균형** — Top-k 게이팅, Softmax 라우팅, Auxiliary Loss를 통한 전문가 부하 균형 전략을 다룹니다."""),
]

create_notebook(cells, 'chapter12_modern_llms/03_rotary_position_embedding.ipynb')
