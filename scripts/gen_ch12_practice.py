"""Generate Chapter 12 practice files."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helper import md, code, create_notebook

# =================== ex01: Implement Llama Scratch ===================
ex01_cells = [
md("""# 실습 퀴즈: Llama 아키텍처 밑바닥 구현

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다

## 목차
- [Q1: RMSNorm 직접 구현](#q1)
- [Q2: RoPE 적용 후 Attention Score 계산](#q2)
- [Q3: SwiGLU FFN Forward Pass](#q3)
- [Q4: GQA Attention 구현](#q4)
- [종합 도전: 소형 Llama Block 모듈 조립](#bonus)

---"""),

code("""# 퀴즈 환경 설정
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print("퀴즈 환경 준비 완료!")
print("각 문제를 풀기 전에 먼저 답을 예측해보세요.")"""),

md(r"""---
## Q1: RMSNorm 직접 구현 <a name='q1'></a>

### 문제

다음 조건으로 RMSNorm을 구현하세요:
- 입력: $x = [1.0, 2.0, 3.0, 4.0]$
- 게인: $g = [1.0, 1.0, 1.0, 1.0]$
- $\epsilon = 10^{-6}$

$$\text{RMSNorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{n}\sum_j x_j^2 + \epsilon}} \cdot g_i$$

**여러분의 예측:** RMSNorm 출력의 L2 norm은 `?` 입니다."""),

code(r"""# ── Q1 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q1 풀이: RMSNorm 직접 구현")
print("=" * 45)

x = tf.constant([1.0, 2.0, 3.0, 4.0])
g = tf.constant([1.0, 1.0, 1.0, 1.0])
eps = 1e-6

# RMS 계산
rms = tf.sqrt(tf.reduce_mean(tf.square(x)) + eps)
print(f"입력 x: {x.numpy()}")
print(f"x^2: {tf.square(x).numpy()}")
print(f"mean(x^2): {tf.reduce_mean(tf.square(x)).numpy():.4f}")
print(f"RMS = sqrt(mean(x^2) + eps) = {rms.numpy():.6f}")

# 정규화
output = (x / rms) * g
print(f"\nRMSNorm 출력: {output.numpy()}")
print(f"출력 L2 norm: {tf.norm(output).numpy():.4f}")
print()

print("[해설]")
print(f"  RMS = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5) = {np.sqrt(7.5):.4f}")
print("  RMSNorm은 벡터를 sqrt(n) 크기로 정규화")
print(f"  출력 L2 norm ≈ sqrt(n) = sqrt(4) = 2.0")"""),

md(r"""---
## Q2: RoPE 적용 후 Attention Score <a name='q2'></a>

### 문제

2D 벡터 $q = [1, 0]$, $k = [1, 0]$에 RoPE($\theta = \pi/4$)를 적용합니다.
- Q 위치: $m = 0$
- K 위치: $n = 2$

RoPE 적용 후 $q \cdot k$ (내적)를 계산하세요.

**여러분의 예측:** 내적 값은 `?` 입니다."""),

code(r"""# ── Q2 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q2 풀이: RoPE 적용 후 Attention Score")
print("=" * 45)

theta = np.pi / 4  # 45도
q = np.array([1.0, 0.0])
k = np.array([1.0, 0.0])
m, n = 0, 2

# RoPE 회전 적용
def rotate_2d(vec, angle):
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([vec[0]*cos_a - vec[1]*sin_a,
                     vec[0]*sin_a + vec[1]*cos_a])

q_rot = rotate_2d(q, m * theta)  # m=0이므로 회전 없음
k_rot = rotate_2d(k, n * theta)  # n=2, 각도=pi/2

dot = np.dot(q_rot, k_rot)

print(f"q = {q}, m = {m}")
print(f"k = {k}, n = {n}")
print(f"theta = pi/4 = {theta:.4f}")
print()
print(f"q 회전 (m*theta = {m*theta:.2f}): {q_rot}")
print(f"k 회전 (n*theta = {n*theta:.2f}): {k_rot}")
print(f"내적 q_rot · k_rot = {dot:.4f}")
print()
print("[해설]")
print(f"  q는 0도 회전 (그대로): [1, 0]")
print(f"  k는 pi/2 회전: [cos(pi/2), sin(pi/2)] = [0, 1]")
print(f"  내적 = 1*0 + 0*1 = 0")
print(f"  상대 거리 |m-n|=2 → 각도 차이 = 2*pi/4 = pi/2 → 직교!")"""),

md(r"""---
## Q3: SwiGLU FFN Forward Pass <a name='q3'></a>

### 문제

SwiGLU FFN의 중간 출력을 계산하세요:
- $x = [1.0, -1.0]$, $d_{model} = 2$, $d_{ff} = 3$
- $W_1 = [[1, 0, -1], [0, 1, 0]]$ (gate)
- $W_2 = [[0, 1, 1], [1, 0, -1]]$ (up)
- Swish = SiLU: $x \cdot \sigma(x)$

**여러분의 예측:** $\text{Swish}(xW_1) \otimes (xW_2)$의 결과는 `?` 입니다."""),

code(r"""# ── Q3 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q3 풀이: SwiGLU FFN Forward Pass")
print("=" * 45)

x = np.array([[1.0, -1.0]])
W1 = np.array([[1, 0, -1], [0, 1, 0]])  # gate
W2 = np.array([[0, 1, 1], [1, 0, -1]])  # up

# Step 1: xW1 (gate projection)
gate_linear = x @ W1
print(f"x = {x[0]}")
print(f"xW1 (gate) = {gate_linear[0]}")

# Step 2: Swish(xW1)
def swish(z):
    return z * (1 / (1 + np.exp(-z)))

gate = swish(gate_linear)
print(f"Swish(xW1)  = {gate[0]}")

# Step 3: xW2 (up projection)
up = x @ W2
print(f"xW2 (up)    = {up[0]}")

# Step 4: element-wise product
swiglu_out = gate * up
print(f"Gate ⊗ Up   = {swiglu_out[0]}")
print()

print("[해설]")
print("  1. xW1 = [1*1+(-1)*0, 1*0+(-1)*1, 1*(-1)+(-1)*0] = [1, -1, -1]")
print(f"  2. Swish([1,-1,-1]) = [1*σ(1), -1*σ(-1), -1*σ(-1)]")
print(f"     = [{1*0.7311:.4f}, {-1*0.2689:.4f}, {-1*0.2689:.4f}]")
print("  3. xW2 = [0+1, 1+0, 1-(-1)] = [-1, 1, 2]")
print(f"  4. Gate ⊗ Up = element-wise product")"""),

md(r"""---
## Q4: GQA Attention 구현 <a name='q4'></a>

### 문제

$H_Q = 4, H_{KV} = 2, d_{head} = 4$일 때:
1. Q 그룹 수는?
2. Q 프로젝션 파라미터 수 vs KV 프로젝션 파라미터 수 비율은?

**여러분의 예측:** 그룹 수는 `?`, 파라미터 비율은 `?` 입니다."""),

code(r"""# ── Q4 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q4 풀이: GQA Attention 구현")
print("=" * 45)

H_Q, H_KV, d_head = 4, 2, 4
d_model = H_Q * d_head  # 16

n_groups = H_Q // H_KV
print(f"H_Q = {H_Q}, H_KV = {H_KV}, d_head = {d_head}")
print(f"d_model = H_Q * d_head = {d_model}")
print(f"Q 그룹 수 = H_Q / H_KV = {n_groups}")
print()

# 파라미터 수
wq_params = d_model * (H_Q * d_head)    # Q projection
wk_params = d_model * (H_KV * d_head)   # K projection
wv_params = d_model * (H_KV * d_head)   # V projection
wo_params = d_model * d_model            # O projection

print(f"Wq 파라미터: d_model × (H_Q × d_head) = {d_model} × {H_Q * d_head} = {wq_params}")
print(f"Wk 파라미터: d_model × (H_KV × d_head) = {d_model} × {H_KV * d_head} = {wk_params}")
print(f"Wv 파라미터: d_model × (H_KV × d_head) = {d_model} × {H_KV * d_head} = {wv_params}")
print(f"Wo 파라미터: d_model × d_model = {d_model} × {d_model} = {wo_params}")
print()
print(f"Q / KV 파라미터 비율: {wq_params} / {wk_params} = {wq_params/wk_params:.1f}x")
print(f"MHA 대비 KV 절감률: {(1 - H_KV/H_Q)*100:.0f}%")
print()

# 실제 GQA 연산 시뮬레이션
x = tf.random.normal((1, 8, d_model))
Wq = tf.random.normal((d_model, H_Q * d_head))
Wk = tf.random.normal((d_model, H_KV * d_head))

Q = tf.reshape(x @ Wq, (1, 8, H_Q, d_head))
K = tf.reshape(x @ Wk, (1, 8, H_KV, d_head))

# KV repeat
K_repeated = tf.repeat(K, repeats=n_groups, axis=2)

print(f"Q shape: {Q.shape} → [B, S, H_Q, d_head]")
print(f"K shape (원본): {K.shape} → [B, S, H_KV, d_head]")
print(f"K shape (반복): {K_repeated.shape} → [B, S, H_Q, d_head]")
print()
print("[해설]")
print(f"  GQA에서 {H_KV}개의 KV 헤드를 {n_groups}번 반복하여 {H_Q}개의 Q 헤드에 매칭")"""),

md(r"""---
## 종합 도전: 소형 Llama Block 모듈 조립 <a name='bonus'></a>

### 문제

다음 구성 요소를 모두 조합하여 **완전한 Llama Decoder Block**을 구현하세요:
1. Pre-Norm: RMSNorm
2. Attention: GQA (H_Q=8, H_KV=2)
3. FFN: SwiGLU (d_ff = 8/3 * d_model)
4. Residual Connection

설정: `d_model=64, seq_len=16, batch=2`"""),

code(r"""# ── 종합 도전 풀이: 소형 Llama Block 조립 ────────────────────
print("=" * 55)
print("종합 도전: 소형 Llama Decoder Block")
print("=" * 55)

# 구성 요소 정의
class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = self.add_weight('gain', shape=(dim,), initializer='ones')
    def call(self, x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.g

class GQAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_q_heads, n_kv_heads):
        super().__init__()
        self.n_q = n_q_heads
        self.n_kv = n_kv_heads
        self.d_h = d_model // n_q_heads
        self.groups = n_q_heads // n_kv_heads
        self.wq = tf.keras.layers.Dense(n_q_heads * self.d_h, use_bias=False)
        self.wk = tf.keras.layers.Dense(n_kv_heads * self.d_h, use_bias=False)
        self.wv = tf.keras.layers.Dense(n_kv_heads * self.d_h, use_bias=False)
        self.wo = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, x):
        B, S, _ = x.shape
        q = tf.reshape(self.wq(x), (B, S, self.n_q, self.d_h))
        k = tf.reshape(self.wk(x), (B, S, self.n_kv, self.d_h))
        v = tf.reshape(self.wv(x), (B, S, self.n_kv, self.d_h))
        k = tf.repeat(k, self.groups, axis=2)
        v = tf.repeat(v, self.groups, axis=2)
        q, k, v = [tf.transpose(t, [0, 2, 1, 3]) for t in [q, k, v]]
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(self.d_h)))
        return self.wo(tf.reshape(tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3]), (B, S, -1)))

class SwiGLUFFN(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = tf.keras.layers.Dense(d_ff, use_bias=False)
        self.w2 = tf.keras.layers.Dense(d_ff, use_bias=False)
        self.w3 = tf.keras.layers.Dense(d_model, use_bias=False)
    def call(self, x):
        return self.w3(tf.nn.silu(self.w1(x)) * self.w2(x))

class LlamaBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_q, n_kv, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GQAttention(d_model, n_q, n_kv)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)
    def call(self, x):
        x = x + self.attn(self.norm1(x))   # Pre-Norm + Residual
        x = x + self.ffn(self.norm2(x))     # Pre-Norm + Residual
        return x

# 조립 및 테스트
d_model = 64
block = LlamaBlock(d_model=d_model, n_q=8, n_kv=2, d_ff=int(8/3*d_model))
x = tf.random.normal((2, 16, d_model))
out = block(x)

total_params = sum(tf.size(v).numpy() for v in block.trainable_variables)

print(f"설정: d_model={d_model}, H_Q=8, H_KV=2, d_ff={int(8/3*d_model)}")
print(f"입력: {x.shape}")
print(f"출력: {out.shape}")
print(f"Shape 보존: {x.shape == out.shape}")
print(f"총 파라미터: {total_params:,}")
print()
print("아키텍처:")
print("  x → RMSNorm → GQA (H_Q=8, H_KV=2) → + residual")
print("    → RMSNorm → SwiGLU FFN → + residual → output")
print()
print("🎉 축하합니다! Llama 3 스타일 Decoder Block을 성공적으로 구현했습니다!")"""),
]

create_notebook(ex01_cells, 'chapter12_modern_llms/practice/ex01_implement_llama_scratch.ipynb')

# =================== ex02: Custom MoE Layer ===================
ex02_cells = [
md("""# 실습 퀴즈: MoE 레이어 커스텀 구현

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다

## 목차
- [Q1: Top-2 Router Softmax 게이팅](#q1)
- [Q2: Expert Dispatch와 Combine](#q2)
- [Q3: Auxiliary Loss 계산](#q3)
- [Q4: Shared Expert 추가](#q4)
- [종합 도전: DeepSeekMoE 스타일 완전한 MoE 레이어](#bonus)

---"""),

code("""# 퀴즈 환경 설정
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print("퀴즈 환경 준비 완료!")"""),

md(r"""---
## Q1: Top-2 Router Softmax 게이팅 <a name='q1'></a>

### 문제

4명의 Expert에 대한 라우터 로짓이 $h = [1.0, 3.0, 0.5, 2.0]$입니다.
Top-2를 선택하고 재정규화된 게이팅 가중치를 계산하세요.

**여러분의 예측:** 선택된 Expert 인덱스는 `?`, 게이팅 가중치는 `?` 입니다."""),

code(r"""# ── Q1 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q1 풀이: Top-2 Router Softmax 게이팅")
print("=" * 45)

h = tf.constant([1.0, 3.0, 0.5, 2.0])
top_k = 2

# Top-2 선택
values, indices = tf.math.top_k(h, k=top_k)
print(f"로짓 h: {h.numpy()}")
print(f"Top-{top_k} 인덱스: {indices.numpy()}")
print(f"Top-{top_k} 값: {values.numpy()}")

# 재정규화 (Top-k에 대해서만 softmax)
gates = tf.nn.softmax(values)
print(f"게이팅 가중치: {gates.numpy()}")
print()
print("[해설]")
print(f"  Top-2: Expert {indices[0].numpy()} (h={values[0].numpy():.1f}), "
      f"Expert {indices[1].numpy()} (h={values[1].numpy():.1f})")
print(f"  g_1 = exp(3) / (exp(3) + exp(2)) = {np.exp(3)/(np.exp(3)+np.exp(2)):.4f}")
print(f"  g_3 = exp(2) / (exp(3) + exp(2)) = {np.exp(2)/(np.exp(3)+np.exp(2)):.4f}")"""),

md(r"""---
## Q2: Expert Dispatch와 Combine <a name='q2'></a>

### 문제

3개의 토큰, 4명의 Expert, Top-2 기준으로:
- 토큰 0 → Expert [1, 3], 가중치 [0.7, 0.3]
- 토큰 1 → Expert [0, 2], 가중치 [0.6, 0.4]
- 토큰 2 → Expert [1, 0], 가중치 [0.5, 0.5]

Expert별 할당된 토큰 수와 가중합 출력을 계산하세요.

**여러분의 예측:** Expert 1에 할당된 토큰 수는 `?`개 입니다."""),

code(r"""# ── Q2 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q2 풀이: Expert Dispatch와 Combine")
print("=" * 45)

dispatch = {
    0: {'experts': [1, 3], 'gates': [0.7, 0.3]},
    1: {'experts': [0, 2], 'gates': [0.6, 0.4]},
    2: {'experts': [1, 0], 'gates': [0.5, 0.5]},
}

n_experts = 4
expert_loads = {i: [] for i in range(n_experts)}

for tok, info in dispatch.items():
    for exp, gate in zip(info['experts'], info['gates']):
        expert_loads[exp].append((tok, gate))

print("Expert별 할당:")
for exp, assignments in expert_loads.items():
    tokens = [f"토큰{t}(g={g})" for t, g in assignments]
    print(f"  Expert {exp}: {', '.join(tokens) if tokens else '없음'} → {len(assignments)}개 토큰")
print()
print("출력 계산 (가상 Expert 출력 사용):")
d = 4
expert_outputs = {i: np.random.randn(d) for i in range(n_experts)}
for tok, info in dispatch.items():
    combined = np.zeros(d)
    for exp, gate in zip(info['experts'], info['gates']):
        combined += gate * expert_outputs[exp]
    print(f"  토큰 {tok}: {combined[:2]}... (처음 2차원)")
print()
print("[해설]")
print("  Expert 1: 토큰 0, 2 → 2개 (가장 많음)")
print("  Expert 3: 토큰 0만 → 1개")"""),

md(r"""---
## Q3: Auxiliary Loss 계산 <a name='q3'></a>

### 문제

$N=4$, $\alpha=0.01$, 전체 토큰 16개:
- Expert별 토큰 수: [8, 4, 2, 2]
- 평균 라우팅 확률 $P = [0.4, 0.3, 0.15, 0.15]$

$L_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$를 계산하세요.

**여러분의 예측:** $L_{aux}$는 `?` 입니다."""),

code(r"""# ── Q3 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q3 풀이: Auxiliary Loss 계산")
print("=" * 45)

N = 4
alpha = 0.01
counts = np.array([8, 4, 2, 2], dtype=float)
T = counts.sum()
f = counts / T  # 실제 분배 비율
P = np.array([0.4, 0.3, 0.15, 0.15])  # 평균 라우팅 확률

L_aux = alpha * N * np.sum(f * P)

print(f"N = {N}, alpha = {alpha}")
print(f"토큰 수: {counts.astype(int)}, 총 T = {int(T)}")
print(f"f (실제 비율): {f}")
print(f"P (라우팅 확률): {P}")
print()
print(f"f · P = {f * P}")
print(f"sum(f · P) = {np.sum(f * P):.4f}")
print(f"L_aux = {alpha} × {N} × {np.sum(f * P):.4f} = {L_aux:.6f}")
print()

# 균등 분배 시 비교
L_uniform = alpha * N * N * (1/N)**2
print(f"균등 분배 시 L_aux = {L_uniform:.6f}")
print(f"현재 불균형도: L_aux / L_uniform = {L_aux/L_uniform:.2f}x")
print()
print("[해설]")
print(f"  현재 L_aux ({L_aux:.6f}) > 균등 L_aux ({L_uniform:.6f})")
print(f"  → 불균형 상태. Loss를 최소화하면 자동으로 균형 유도!")"""),

md(r"""---
## Q4: Shared Expert 추가 <a name='q4'></a>

### 문제

기존 MoE에 Shared Expert 1개를 추가합니다. 입력 $x$에 대해:
- Shared Expert 출력: $E_s(x)$
- Routed Expert 출력: $\sum g_i E_i^r(x)$ (Top-2)

최종 출력 공식과, Shared Expert 추가가 파라미터 수에 미치는 영향을 분석하세요.

**여러분의 예측:** Shared Expert 추가 시 활성 파라미터 증가량은 `?`% 입니다."""),

code(r"""# ── Q4 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q4 풀이: Shared Expert 추가 분석")
print("=" * 45)

d_model = 256
d_ff = 512
n_routed = 8
top_k = 2

# Expert당 파라미터 (FFN: up + down)
expert_params = d_model * d_ff + d_ff * d_model
shared_params = expert_params  # Shared도 같은 구조

# 활성 파라미터 (처리 시)
active_without_shared = top_k * expert_params
active_with_shared = (1 + top_k) * expert_params

print(f"설정: d_model={d_model}, d_ff={d_ff}, N_routed={n_routed}, Top-{top_k}")
print(f"Expert당 파라미터: {expert_params:,}")
print()
print(f"{'구성':<25} | {'총 파라미터':>12} | {'활성 파라미터':>12}")
print("-" * 55)
print(f"{'MoE (Shared 없음)':<25} | {n_routed*expert_params:>12,} | {active_without_shared:>12,}")
print(f"{'DeepSeekMoE (Shared 1)':<25} | {(n_routed+1)*expert_params:>12,} | {active_with_shared:>12,}")
print()
increase = (active_with_shared / active_without_shared - 1) * 100
print(f"Shared Expert 추가 시 활성 파라미터 증가: {increase:.0f}%")
print(f"  → Shared Expert가 '공통 지식'을 담당하므로 Top-k 감소 가능")
print()
print("최종 출력:")
print("  y = E_s(x) + sum(g_i * E_i^r(x)) for i in Top-k")
print("  Shared는 항상 활성 → 공통 패턴 학습")
print("  Routed는 선택적 → 전문 패턴 학습")"""),

md(r"""---
## 종합 도전: DeepSeekMoE 스타일 완전한 MoE 레이어 <a name='bonus'></a>

### 문제

다음 조건으로 **완전한 DeepSeekMoE 레이어**를 구현하세요:
1. Shared Expert 1개 + Routed Expert 4개
2. Top-2 라우팅 (Softmax 게이팅)
3. Auxiliary-Loss-Free 편향 보정
4. 배치 처리 지원"""),

code(r"""# ── 종합 도전 풀이: DeepSeekMoE 레이어 ────────────────────────
print("=" * 55)
print("종합 도전: DeepSeekMoE 스타일 MoE 레이어")
print("=" * 55)

class MiniDeepSeekMoE(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, n_routed=4, top_k=2):
        super().__init__()
        self.n_routed = n_routed
        self.top_k = top_k
        
        # Shared Expert
        self.shared = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        # Routed Experts
        self.experts = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(d_ff, activation='relu'),
                tf.keras.layers.Dense(d_model)
            ]) for _ in range(n_routed)
        ]
        
        # Router
        self.router = tf.keras.layers.Dense(n_routed, use_bias=False)
        
        # Aux-Free bias
        self.bias = tf.Variable(tf.zeros(n_routed), trainable=False)
    
    def call(self, x):
        # Shared output
        out = self.shared(x)
        
        # Routing
        logits = self.router(x) + self.bias
        top_vals, top_idx = tf.math.top_k(logits, k=self.top_k)
        gates = tf.nn.softmax(top_vals, axis=-1)
        
        # Routed output
        for k in range(self.top_k):
            idx = top_idx[:, :, k]
            g = gates[:, :, k:k+1]
            for e in range(self.n_routed):
                mask = tf.cast(tf.equal(idx, e), tf.float32)[:, :, tf.newaxis]
                if tf.reduce_sum(mask) > 0:
                    out += self.experts[e](x) * mask * g
        
        return out

# 테스트
d_model, d_ff = 64, 128
moe = MiniDeepSeekMoE(d_model, d_ff, n_routed=4, top_k=2)
x = tf.random.normal((2, 8, d_model))
y = moe(x)

total = sum(tf.size(v).numpy() for v in moe.trainable_variables)
print(f"입력: {x.shape}")
print(f"출력: {y.shape}")
print(f"총 파라미터: {total:,}")
print(f"활성 Expert: 1(shared) + 2(routed) = 3")
print()
print("🎉 DeepSeekMoE 스타일 레이어 구현 완료!")
print("   Shared Expert가 공통 지식을, Routed Expert가 전문 지식을 담당합니다.")"""),
]

create_notebook(ex02_cells, 'chapter12_modern_llms/practice/ex02_custom_moe_layer.ipynb')
