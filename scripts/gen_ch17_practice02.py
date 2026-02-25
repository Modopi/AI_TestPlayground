import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 0: Header ──
cells.append(md(r"""# 실습 퀴즈: adaLN-Zero DiT 블록과 Flow Matching

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다

## 목차
- [Q1: adaLN-Zero 단일 블록](#q1)
- [Q2: Flow Matching Loss 계산](#q2)
- [Q3: Euler ODE 샘플링 스텝](#q3)
- [Q4: DiT 블록 조립 (adaLN + Attention + FFN)](#q4)
- [종합 도전: 소형 DiT 학습](#bonus)"""))

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
cells.append(md(r"""## Q1: adaLN-Zero 단일 블록 <a name='q1'></a>

### 문제

adaLN-Zero의 핵심 수식을 구현하세요:

$$h = x + \alpha \cdot f\!\left((1 + \gamma) \odot \text{LN}(x) + \beta\right)$$

여기서 $(\gamma, \beta, \alpha)$는 조건 벡터 $c$로부터 MLP로 생성되며, **$\alpha$의 초기값은 0**입니다.

`d_model=32`, 입력 `x`의 shape이 `(1, 4, 32)`일 때:
1. MLP가 출력해야 하는 파라미터 수는? (γ, β, α 각각 d_model차원)
2. 초기 상태에서 출력 `h`와 입력 `x`의 차이는?

**여러분의 예측:**
- MLP 출력 차원 = `?`
- 초기 `h - x`의 norm = `?`"""))

# ── Cell 3: Q1 Solution ──
cells.append(code(r"""# ── Q1 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q1 풀이: adaLN-Zero 단일 블록")
print("=" * 45)

d_model = 32

class AdaLNZeroBlock(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.linear_f = tf.keras.layers.Dense(d_model, activation='silu')
        self.adaLN_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation='silu'),
            tf.keras.layers.Dense(d_model * 3)
        ])

    def build(self, input_shape):
        super().build(input_shape)
        last_layer = self.adaLN_mlp.layers[-1]
        last_layer.kernel.assign(tf.zeros_like(last_layer.kernel))
        last_layer.bias.assign(tf.zeros_like(last_layer.bias))

    def call(self, x, c):
        params = self.adaLN_mlp(c)
        gamma = params[..., :d_model]
        beta = params[..., d_model:2*d_model]
        alpha = params[..., 2*d_model:]

        if len(gamma.shape) == 2:
            gamma = gamma[:, tf.newaxis, :]
            beta = beta[:, tf.newaxis, :]
            alpha = alpha[:, tf.newaxis, :]

        normed = self.ln(x)
        modulated = (1.0 + gamma) * normed + beta
        h = x + alpha * self.linear_f(modulated)
        return h

block = AdaLNZeroBlock(d_model)
x = tf.random.normal([1, 4, d_model])
c = tf.random.normal([1, d_model])

h = block(x, c)
diff_norm = tf.reduce_mean(tf.abs(h - x)).numpy()

print(f"\nd_model = {d_model}")
print(f"MLP 출력 차원 = {d_model} x 3 (γ, β, α) = {d_model * 3}")
print(f"\n입력 x shape: {x.shape}")
print(f"조건 c shape: {c.shape}")
print(f"출력 h shape: {h.shape}")
print(f"\n초기 상태 |h - x| 평균: {diff_norm:.8f}")
print(f"(MLP 마지막 층을 0으로 초기화 → α=0 → h=x)")

n_params = sum(p.numpy().size for p in block.trainable_variables)
print(f"블록 파라미터 수: {n_params:,}")

print("\n[해설]")
print("  α=0이면 h = x + 0 * f(...) = x (항등 함수)")
print("  이것이 adaLN-Zero의 핵심: 초기에 블록이 아무 변화를 주지 않음")
print("  학습이 진행되며 서서히 α가 커지면서 변환이 활성화됩니다.")"""))

# ── Cell 4: Q2 Problem ──
cells.append(md(r"""## Q2: Flow Matching Loss 계산 <a name='q2'></a>

### 문제

Flow Matching Loss를 직접 계산하세요:

$$\mathcal{L}_{FM} = \mathbb{E}_{t,x_0,x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

$x_0 = [1.0, 0.0]$, $x_1 = [-1.0, 2.0]$, $t = 0.4$일 때:
1. $x_t = (1-t)x_0 + tx_1$ = ?
2. $v^{GT} = x_1 - x_0$ = ?
3. 모델이 $v_\theta = [-1.5, 1.8]$을 예측했다면 loss는?

**여러분의 예측:** loss = `?`"""))

# ── Cell 5: Q2 Solution ──
cells.append(code(r"""# ── Q2 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q2 풀이: Flow Matching Loss 계산")
print("=" * 45)

x0 = np.array([1.0, 0.0])
x1 = np.array([-1.0, 2.0])
t = 0.4

x_t = (1 - t) * x0 + t * x1
v_gt = x1 - x0
v_pred = np.array([-1.5, 1.8])

loss = np.mean((v_pred - v_gt) ** 2)

print(f"\nx_0 = {x0}")
print(f"x_1 = {x1}")
print(f"t = {t}")
print(f"\nx_t = (1-{t}){x0} + {t}{x1}")
print(f"    = {1-t}{x0} + {t}{x1}")
print(f"    = {(1-t)*x0} + {t*x1}")
print(f"    = {x_t}")
print(f"\nv_GT = x_1 - x_0 = {x1} - {x0} = {v_gt}")
print(f"v_pred = {v_pred}")
print(f"\nLoss = mean(|v_pred - v_GT|^2)")
print(f"     = mean(|{v_pred} - {v_gt}|^2)")
print(f"     = mean(|{v_pred - v_gt}|^2)")
print(f"     = mean({(v_pred - v_gt)**2})")
print(f"     = {loss:.4f}")

# TensorFlow로 검증
x0_tf = tf.constant([[1.0, 0.0]])
x1_tf = tf.constant([[-1.0, 2.0]])
t_tf = tf.constant([[0.4]])
xt_tf = (1 - t_tf) * x0_tf + t_tf * x1_tf
vgt_tf = x1_tf - x0_tf
vpred_tf = tf.constant([[-1.5, 1.8]])
loss_tf = tf.reduce_mean(tf.square(vpred_tf - vgt_tf))
print(f"\nTF 검증 Loss: {loss_tf.numpy():.4f}")

print("\n[해설]")
print("  FM Loss는 단순한 MSE입니다.")
print("  v_pred가 v_GT에 가까울수록 loss가 작아집니다.")
print(f"  이 경우 오차는 {v_pred - v_gt}로, 작은 예측 오차입니다.")"""))

# ── Cell 6: Q3 Problem ──
cells.append(md(r"""## Q3: Euler ODE 샘플링 스텝 <a name='q3'></a>

### 문제

Euler 방법으로 ODE를 적분하여 샘플링합니다:

$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$$

$x_0 = [2.0, -1.0]$에서 시작, 학습된 속도 모델이 항상 $v_\theta = [-1.0, 1.5]$을 출력한다고 가정합니다.

$\Delta t = 0.25$ (4스텝)로 $t=0 \to t=1$ 샘플링 경로를 구하세요.

**여러분의 예측:** $x_1 = ?$"""))

# ── Cell 7: Q3 Solution ──
cells.append(code(r"""# ── Q3 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q3 풀이: Euler ODE 샘플링 스텝")
print("=" * 45)

x = np.array([2.0, -1.0])
v_const = np.array([-1.0, 1.5])
dt = 0.25
n_steps = 4

print(f"\nx_0 = {x}")
print(f"v_θ = {v_const} (상수)")
print(f"Δt = {dt}, 스텝 수 = {n_steps}\n")

trajectory = [x.copy()]
for step in range(n_steps):
    t_val = step * dt
    x_new = x + dt * v_const
    print(f"스텝 {step+1}: t={t_val:.2f} → t={t_val+dt:.2f}")
    print(f"  x_{t_val+dt:.2f} = {x} + {dt} * {v_const} = {x_new}")
    x = x_new
    trajectory.append(x.copy())

trajectory = np.array(trajectory)
print(f"\n최종 x_1 = {x}")
expected = np.array([2.0, -1.0]) + 1.0 * v_const
print(f"해석적 해: x_0 + 1.0 * v = {expected}")
print(f"오차: {np.linalg.norm(x - expected):.10f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', lw=2.5, ms=10, label='Euler 경로')
ax.plot(trajectory[0, 0], trajectory[0, 1], 'ro', ms=14, zorder=5, label='$x_0$ (시작)')
ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'g*', ms=18, zorder=5, label='$x_1$ (도착)')

for i, (px, py) in enumerate(trajectory):
    ax.annotate(f't={i*dt:.2f}', (px, py), textcoords="offset points",
                xytext=(10, 10), fontsize=9)

ax.quiver(trajectory[:-1, 0], trajectory[:-1, 1],
          np.full(n_steps, v_const[0]*dt), np.full(n_steps, v_const[1]*dt),
          angles='xy', scale_units='xy', scale=1, color='orange', alpha=0.5, width=0.01)

ax.set_xlabel('Dimension 1', fontsize=11)
ax.set_ylabel('Dimension 2', fontsize=11)
ax.set_title('Euler ODE 샘플링 경로 (4 스텝)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/practice/euler_steps_q3.png', dpi=100, bbox_inches='tight')
plt.close()

print("\n그래프 저장됨: chapter17_diffusion_transformers/practice/euler_steps_q3.png")
print("\n[해설]")
print("  상수 속도에서 Euler 적분은 정확합니다 (오차 = 0).")
print("  이것이 Rectified Flow의 장점입니다!")"""))

# ── Cell 8: Q4 Problem ──
cells.append(md(r"""## Q4: DiT 블록 조립 (adaLN + Attention + FFN) <a name='q4'></a>

### 문제

완전한 DiT 블록을 조립하세요:

$$h = x + \alpha_1 \cdot \text{Attn}\!\left((1 + \gamma_1) \odot \text{LN}(x) + \beta_1\right)$$
$$\text{out} = h + \alpha_2 \cdot \text{FFN}\!\left((1 + \gamma_2) \odot \text{LN}(h) + \beta_2\right)$$

`d_model=64`, `n_heads=4`, `ffn_dim=256`으로 구현하세요.

**여러분의 예측:** 블록 파라미터 수 = `?`"""))

# ── Cell 9: Q4 Solution ──
cells.append(code(r"""# ── Q4 풀이 ──────────────────────────────────────────────────
print("=" * 45)
print("Q4 풀이: DiT 블록 조립")
print("=" * 45)

class DiTBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        self.d_model = d_model
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ffn_dim, activation='gelu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.adaLN_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation='silu'),
            tf.keras.layers.Dense(d_model * 6)
        ])

    def build(self, input_shape):
        super().build(input_shape)
        last_layer = self.adaLN_mlp.layers[-1]
        last_layer.kernel.assign(tf.zeros_like(last_layer.kernel))
        last_layer.bias.assign(tf.zeros_like(last_layer.bias))

    def call(self, x, c):
        params = self.adaLN_mlp(c)
        if len(params.shape) == 2:
            params = params[:, tf.newaxis, :]

        gamma1 = params[..., 0*self.d_model:1*self.d_model]
        beta1 = params[..., 1*self.d_model:2*self.d_model]
        alpha1 = params[..., 2*self.d_model:3*self.d_model]
        gamma2 = params[..., 3*self.d_model:4*self.d_model]
        beta2 = params[..., 4*self.d_model:5*self.d_model]
        alpha2 = params[..., 5*self.d_model:6*self.d_model]

        normed1 = (1.0 + gamma1) * self.ln1(x) + beta1
        h = x + alpha1 * self.attn(normed1, normed1)

        normed2 = (1.0 + gamma2) * self.ln2(h) + beta2
        out = h + alpha2 * self.ffn(normed2)
        return out

d_model, n_heads, ffn_dim = 64, 4, 256
block = DiTBlock(d_model, n_heads, ffn_dim)

x = tf.random.normal([2, 16, d_model])
c = tf.random.normal([2, d_model])

out = block(x, c)

init_diff = tf.reduce_mean(tf.abs(out - x)).numpy()

print(f"\nd_model={d_model}, n_heads={n_heads}, ffn_dim={ffn_dim}")
print(f"\n입력 x shape: {x.shape}")
print(f"조건 c shape: {c.shape}")
print(f"출력 shape: {out.shape}")
print(f"\n초기 상태 |out - x| 평균: {init_diff:.8f}")
print(f"  → Zero-init 확인: {'통과' if init_diff < 1e-4 else '실패'}")

n_params = sum(p.numpy().size for p in block.trainable_variables)
print(f"\n블록 파라미터 수: {n_params:,}")
print(f"  - adaLN MLP: ~{d_model * d_model * 4 + d_model * 4 * 6:,}")
print(f"  - Multi-Head Attention: ~{4 * d_model * d_model:,}")
print(f"  - FFN: ~{d_model * ffn_dim + ffn_dim * d_model:,}")

print("\n[해설]")
print("  DiT 블록 = adaLN-Zero (조건 주입) + Self-Attention + FFN")
print("  adaLN MLP가 6개 파라미터 벡터를 생성합니다: γ1, β1, α1, γ2, β2, α2")
print("  모두 0으로 초기화되어 학습 시작 시 항등 함수로 동작합니다.")"""))

# ── Cell 10: Bonus Problem ──
cells.append(md(r"""## 종합 도전: 소형 DiT 학습 <a name='bonus'></a>

### 문제

여러 개의 DiT 블록을 쌓고, Flow Matching Loss로 합성 데이터(Moving MNIST 스타일)에 대해 학습하세요:

1. 간단한 2D 데이터 생성 (원형 패턴)
2. DiT 블록 3개 스택
3. Flow Matching 학습 루프
4. Euler 샘플링으로 결과 확인"""))

# ── Cell 11: Bonus Solution ──
cells.append(code(r"""# ── 종합 도전 풀이: 소형 DiT 학습 ──────────────────────────────────
print("=" * 45)
print("종합 도전 풀이: 소형 DiT 학습")
print("=" * 45)

# 합성 데이터: 8x8 "이미지" 위에 원형 패턴
def make_circle_data(n=1000, img_size=8):
    images = np.zeros((n, img_size, img_size), dtype=np.float32)
    for i in range(n):
        cx = np.random.randint(2, img_size - 2)
        cy = np.random.randint(2, img_size - 2)
        r = np.random.uniform(1.0, 2.5)
        for x in range(img_size):
            for y in range(img_size):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                images[i, x, y] = max(0, 1.0 - dist / r)
    return images

data_img = make_circle_data(500, 8)
data_flat = data_img.reshape(500, -1)
print(f"학습 데이터: {data_img.shape} → 평탄화: {data_flat.shape}")
print(f"값 범위: [{data_flat.min():.3f}, {data_flat.max():.3f}]")

# Mini DiT
class MiniDiT(tf.keras.Model):
    def __init__(self, data_dim=64, d_model=64, n_blocks=3, n_heads=4, ffn_dim=128):
        super().__init__()
        self.input_proj = tf.keras.layers.Dense(d_model)
        self.time_embed = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='silu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.blocks = [DiTBlock(d_model, n_heads, ffn_dim) for _ in range(n_blocks)]
        self.output_proj = tf.keras.layers.Dense(data_dim)

    def call(self, x_flat, t):
        t_emb = self.time_embed(tf.concat([tf.sin(t * np.pi * 4), tf.cos(t * np.pi * 4)], axis=-1))
        h = self.input_proj(x_flat)
        h = h[:, tf.newaxis, :]
        for blk in self.blocks:
            h = blk(h, t_emb)
        h = h[:, 0, :]
        return self.output_proj(h)

dit = MiniDiT(data_dim=64, d_model=64, n_blocks=3, n_heads=4, ffn_dim=128)
optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def dit_train_step(x1_batch):
    bs = tf.shape(x1_batch)[0]
    t = tf.random.uniform([bs, 1], 0.0, 1.0)
    x0 = tf.random.normal(tf.shape(x1_batch))
    x_t = (1.0 - t) * x0 + t * x1_batch
    v_target = x1_batch - x0

    with tf.GradientTape() as tape:
        v_pred = dit(x_t, t)
        loss = tf.reduce_mean(tf.square(v_pred - v_target))

    grads = tape.gradient(loss, dit.trainable_variables)
    optimizer.apply_gradients(zip(grads, dit.trainable_variables))
    return loss

dataset = tf.data.Dataset.from_tensor_slices(data_flat).shuffle(500).batch(64).repeat()
it = iter(dataset)

losses = []
for step in range(300):
    batch = next(it)
    loss = dit_train_step(batch)
    losses.append(float(loss))
    if (step + 1) % 100 == 0:
        print(f"스텝 {step+1:4d} | FM Loss: {loss:.6f}")

# Euler 샘플링
def dit_euler_sample(model, n_samples=16, n_steps=30, data_dim=64):
    x = tf.random.normal([n_samples, data_dim])
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt
        t = tf.fill([n_samples, 1], t_val)
        v = model(x, t)
        x = x + dt * v
    return x.numpy()

samples = dit_euler_sample(dit, n_samples=8)
samples_img = samples.reshape(-1, 8, 8)
samples_img = np.clip(samples_img, 0, 1)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))

for i in range(8):
    axes[0, i].imshow(data_img[i], cmap='hot', vmin=0, vmax=1)
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('원본', fontweight='bold', fontsize=10)

for i in range(8):
    axes[1, i].imshow(samples_img[i], cmap='hot', vmin=0, vmax=1)
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('생성', fontweight='bold', fontsize=10)

plt.suptitle('Mini DiT: Flow Matching 학습 결과', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('chapter17_diffusion_transformers/practice/mini_dit_results.png', dpi=100, bbox_inches='tight')
plt.close()

n_total_params = sum(p.numpy().size for p in dit.trainable_variables)
print(f"\nMini DiT 총 파라미터: {n_total_params:,}")
print(f"최종 Loss: {losses[-1]:.6f}")
print(f"생성 이미지 값 범위: [{samples_img.min():.3f}, {samples_img.max():.3f}]")
print(f"\n그래프 저장됨: chapter17_diffusion_transformers/practice/mini_dit_results.png")

print("\n[해설]")
print("  3개의 DiT 블록 + Flow Matching으로 간단한 패턴을 학습했습니다.")
print("  adaLN-Zero의 Zero-init 덕분에 학습이 안정적으로 시작됩니다.")
print("  실제 DiT는 수백 블록, 수십억 파라미터로 고해상도 이미지/비디오를 생성합니다.")"""))

path = '/workspace/chapter17_diffusion_transformers/practice/ex02_dit_block_with_adaln.ipynb'
create_notebook(cells, path)
