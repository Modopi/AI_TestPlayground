#!/usr/bin/env python3
"""Generate chapter13_genai_diffusion/practice/ex02_implement_cfg_generation.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ─────────────────────────────────────────────
cells.append(md(r"""# 실습 퀴즈: 클래스 조건부 MNIST CFG 생성

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다
- Classifier-Free Guidance(CFG)를 사용한 조건부 생성과 Guidance Scale에 따른 품질-다양성 Trade-off를 실험합니다

## 목차
- [Q1: 조건부/비조건부 Noise Prediction](#q1)
- [Q2: CFG 공식 적용](#q2)
- [Q3: Guidance Scale 실험](#q3)
- [Q4: FID/IS 개념 이해](#q4)
- [종합 도전: Mini CFG Generation Pipeline](#bonus)"""))

# ── Cell 2: Imports ──────────────────────────────────────────────
cells.append(code(r"""# ── 라이브러리 임포트 ──────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
print(f"NumPy 버전: {np.__version__}")

# DDPM 관련 상수
T = 200
beta_min, beta_max = 1e-4, 0.02
betas = np.linspace(beta_min, beta_max, T).astype(np.float32)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas).astype(np.float32)

print(f"타임스텝 수: {T}")
print(f"alpha_bar[0] = {alpha_bars[0]:.6f}, alpha_bar[-1] = {alpha_bars[-1]:.6f}")"""))

# ── Cell 3: Q1 Problem ──────────────────────────────────────────
cells.append(md(r"""---

## Q1: 조건부/비조건부 Noise Prediction <a name='q1'></a>

### 문제

CFG를 사용하려면 **하나의 모델**이 다음 두 가지를 예측할 수 있어야 합니다:

1. **비조건부** $\epsilon_\theta(x_t, t, \varnothing)$: 조건 없이 노이즈 예측
2. **조건부** $\epsilon_\theta(x_t, t, c)$: 클래스 $c$를 받아 노이즈 예측

학습 시 일정 확률 $p_{uncond}$로 조건을 드롭(null로 대체)합니다.

간단한 MLP 기반 조건부 노이즈 예측 모델을 구현하세요.

**여러분의 예측:**
- $p_{uncond} = 0.1$이면 학습 데이터의 `?`%는 비조건부로 학습됩니다
- 비조건부 학습이 없으면 ($p_{uncond} = 0$) CFG가 작동할까요? `?`"""))

# ── Cell 4: Q1 Solution ──────────────────────────────────────────
cells.append(code(r"""# ── Q1 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q1 풀이: 조건부/비조건부 Noise Prediction")
print("=" * 45)

# 간단한 CFG 모델 (2D 데이터용)
class CFGNoiseModel(tf.keras.Model):
    # 조건부/비조건부 통합 노이즈 예측 모델
    def __init__(self, data_dim=2, n_classes=4, time_dim=16, hidden_dim=128):
        super().__init__()
        self.n_classes = n_classes

        # 시간 임베딩
        self.time_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(time_dim, activation='swish'),
            tf.keras.layers.Dense(time_dim)
        ])

        # 클래스 임베딩 (null class = n_classes)
        self.class_embed = tf.keras.layers.Embedding(n_classes + 1, time_dim)

        # 메인 네트워크
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='swish'),
            tf.keras.layers.Dense(hidden_dim, activation='swish'),
            tf.keras.layers.Dense(hidden_dim, activation='swish'),
            tf.keras.layers.Dense(data_dim)
        ])

    def call(self, x_t, t, class_label=None, p_uncond=0.0, training=False):
        # 시간 임베딩
        t_emb = self.time_mlp(tf.cast(tf.reshape(t, [-1, 1]), tf.float32))

        # 조건 드롭 (학습 시)
        if class_label is not None:
            if training and p_uncond > 0:
                mask = tf.random.uniform([tf.shape(class_label)[0]]) < p_uncond
                null_label = tf.fill(tf.shape(class_label), self.n_classes)
                class_label = tf.where(mask, null_label, class_label)
            c_emb = self.class_embed(class_label)
        else:
            batch = tf.shape(x_t)[0]
            null_label = tf.fill([batch], self.n_classes)
            c_emb = self.class_embed(null_label)

        combined = tf.concat([x_t, t_emb + c_emb], axis=-1)
        return self.net(combined)

# 모델 생성 및 테스트
model = CFGNoiseModel(data_dim=2, n_classes=4)

# 더미 데이터
batch = 8
x_dummy = tf.random.normal((batch, 2))
t_dummy = tf.constant([50] * batch, dtype=tf.int32)
c_dummy = tf.constant([0, 1, 2, 3, 0, 1, 2, 3], dtype=tf.int32)

# 비조건부 예측
eps_uncond = model(x_dummy, t_dummy, class_label=None)
print(f"비조건부 예측 shape: {eps_uncond.shape}")

# 조건부 예측
eps_cond = model(x_dummy, t_dummy, class_label=c_dummy)
print(f"조건부 예측 shape: {eps_cond.shape}")

# 학습 시 드롭 시뮬레이션
eps_train = model(x_dummy, t_dummy, class_label=c_dummy, p_uncond=0.1, training=True)
print(f"학습 시 예측 shape: {eps_train.shape}")

total_params = sum(np.prod(w.shape) for w in model.trainable_weights)
print(f"\n총 파라미터 수: {total_params:,}")

print("\n[해설]")
print("  p_uncond = 0.1 → 학습 데이터의 10%는 비조건부로 학습됩니다.")
print("  비조건부 학습 없으면 (p_uncond=0), 모델이 epsilon(x_t, ∅)을 학습하지 않아")
print("  CFG 공식의 비조건부 항을 계산할 수 없습니다 → CFG 불가!")"""))

# ── Cell 5: Q2 Problem ──────────────────────────────────────────
cells.append(md(r"""---

## Q2: CFG 공식 적용 <a name='q2'></a>

### 문제

다음 CFG 공식을 구현하세요:

$$\tilde\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + w \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)]$$

주어진 모델에서 조건부와 비조건부 예측을 뽑고, guidance scale $w$로 결합하는 함수를 작성하세요.

**여러분의 예측:**
- $w = 1$일 때 결과는 `?`과 같다
- $w = 0$일 때 결과는 `?`과 같다"""))

# ── Cell 6: Q2 Solution ──────────────────────────────────────────
cells.append(code(r"""# ── Q2 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q2 풀이: CFG 공식 적용")
print("=" * 45)

def cfg_predict(model, x_t, t, class_label, guidance_scale):
    # CFG 노이즈 예측
    # 1. 비조건부 예측
    eps_uncond = model(x_t, t, class_label=None)

    # 2. 조건부 예측
    eps_cond = model(x_t, t, class_label=class_label)

    # 3. CFG 결합
    eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    return eps_cfg, eps_uncond, eps_cond

# 테스트
x_test = tf.random.normal((4, 2))
t_test = tf.constant([100, 100, 100, 100], dtype=tf.int32)
c_test = tf.constant([0, 1, 2, 3], dtype=tf.int32)

print(f"{'w':<5} | {'eps_cfg norm':>12} | {'eps_uncond와 같나':>18} | {'eps_cond와 같나':>16}")
print("-" * 60)

for w in [0.0, 0.5, 1.0, 3.0, 7.5]:
    eps_cfg, eps_u, eps_c = cfg_predict(model, x_test, t_test, c_test, w)
    norm = tf.reduce_mean(tf.norm(eps_cfg, axis=-1)).numpy()
    same_as_uncond = tf.reduce_all(tf.abs(eps_cfg - eps_u) < 1e-5).numpy()
    same_as_cond = tf.reduce_all(tf.abs(eps_cfg - eps_c) < 1e-5).numpy()
    print(f"w={w:<3.1f} | {norm:>12.4f} | {str(same_as_uncond):>18} | {str(same_as_cond):>16}")

print("\n[해설]")
print("  w=0: CFG = eps_uncond (비조건부와 동일)")
print("  w=1: CFG = eps_cond (조건부와 동일)")
print("  w>1: 조건 방향을 과강조 (외삽)")
print("  CFG = (1-w)*uncond + w*cond 로 재작성 가능")"""))

# ── Cell 7: Q3 Problem ──────────────────────────────────────────
cells.append(md(r"""---

## Q3: Guidance Scale 실험 <a name='q3'></a>

### 문제

2D 가우시안 혼합 데이터를 사용하여, 다양한 guidance scale에서 CFG 생성 결과를 비교하세요.

4개의 클래스 중심:
- 클래스 0: $(-2, -2)$
- 클래스 1: $(2, -2)$
- 클래스 2: $(-2, 2)$
- 클래스 3: $(2, 2)$

$w = [0, 1, 3, 7, 15]$에서 생성 결과를 시각화하고, 다양성과 정확도를 측정하세요.

**여러분의 예측:** $w$가 커지면 생성된 점들이 클래스 중심에 `?` 해질 것이다."""))

# ── Cell 8: Q3 Solution ──────────────────────────────────────────
cells.append(code(r"""# ── Q3 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q3 풀이: Guidance Scale 실험")
print("=" * 45)

# 4클래스 GMM 데이터
class_centers = np.array([[-2, -2], [2, -2], [-2, 2], [2, 2]], dtype=np.float32)
class_std = 0.5
n_per_class = 200

# 학습 데이터 생성
np.random.seed(42)
data_all = []
labels_all = []
for c in range(4):
    pts = class_centers[c] + np.random.randn(n_per_class, 2).astype(np.float32) * class_std
    data_all.append(pts)
    labels_all.append(np.full(n_per_class, c))
data_all = np.concatenate(data_all, axis=0)
labels_all = np.concatenate(labels_all, axis=0)

# Oracle CFG 시뮬레이션
def oracle_cfg_sample(n_samples, target_class, guidance_scale, alpha_bars, class_centers, n_steps=100):
    T_steps = len(alpha_bars)
    step_indices = np.linspace(0, T_steps - 1, n_steps + 1, dtype=int)[::-1]

    x = np.random.randn(n_samples, 2).astype(np.float32)

    for i in range(len(step_indices) - 1):
        t_c = step_indices[i]
        t_p = step_indices[i + 1]
        ab_c = alpha_bars[t_c]
        ab_p = alpha_bars[t_p]

        # 비조건부: 모든 클래스의 가중 평균 스코어
        eps_uncond = np.zeros_like(x)
        for ctr in class_centers:
            eff_mu = np.sqrt(ab_c) * ctr
            eff_var = ab_c * class_std**2 + (1 - ab_c)
            eps_uncond += -(np.sqrt(1 - ab_c)) * (-(x - eff_mu) / eff_var) / len(class_centers)

        # 조건부: 타겟 클래스의 스코어
        ctr = class_centers[target_class]
        eff_mu = np.sqrt(ab_c) * ctr
        eff_var = ab_c * class_std**2 + (1 - ab_c)
        eps_cond = -(np.sqrt(1 - ab_c)) * (-(x - eff_mu) / eff_var)

        # CFG
        eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # DDIM step
        x0_hat = (x - np.sqrt(1 - ab_c) * eps_cfg) / (np.sqrt(ab_c) + 1e-8)
        x0_hat = np.clip(x0_hat, -8, 8)
        x = np.sqrt(ab_p) * x0_hat + np.sqrt(1 - ab_p) * eps_cfg

    return x

# 다양한 w에서 실험
w_values = [0.0, 1.0, 3.0, 7.0, 15.0]
target_class = 0

fig, axes = plt.subplots(1, len(w_values), figsize=(20, 4))

diversity_scores = []
accuracy_scores = []

for idx, w in enumerate(w_values):
    np.random.seed(42)
    samples = oracle_cfg_sample(200, target_class, w, alpha_bars, class_centers, n_steps=50)

    ax = axes[idx]
    # 모든 클래스 중심 표시
    for c, ctr in enumerate(class_centers):
        color = 'red' if c == target_class else 'gray'
        marker = '*' if c == target_class else 'o'
        ax.scatter(*ctr, c=color, s=150, marker=marker, zorder=5, edgecolors='black')
    ax.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.4, c='blue')

    # 다양성: 생성 샘플의 표준편차
    diversity = np.mean(np.std(samples, axis=0))
    diversity_scores.append(diversity)

    # 정확도: 타겟 클래스 중심과의 평균 거리
    dist_to_target = np.mean(np.linalg.norm(samples - class_centers[target_class], axis=1))
    accuracy_scores.append(dist_to_target)

    ax.set_title(f'w = {w}', fontweight='bold', fontsize=12)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle(f'CFG 생성 결과 (클래스 {target_class} 타겟)', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/practice/q3_cfg_guidance_experiment.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/practice/q3_cfg_guidance_experiment.png")

print(f"\n{'w':>5} | {'다양성(std)':>12} | {'타겟 거리':>12} | {'판정':>12}")
print("-" * 50)
for w, div, acc in zip(w_values, diversity_scores, accuracy_scores):
    verdict = "비조건부" if w == 0 else ("균형" if 1 <= w <= 5 else ("고품질" if acc < 1.5 else "과강조"))
    print(f"w={w:>3.0f} | {div:>12.3f} | {acc:>12.3f} | {verdict:>12}")

print("\n[해설]")
print("  w 증가 → 타겟 클래스 중심으로 집중 (다양성↓, 정확도↑)")
print("  w가 너무 크면 생성 분포가 축소되어 모드 붕괴 위험")"""))

# ── Cell 9: Q4 Problem ──────────────────────────────────────────
cells.append(md(r"""---

## Q4: FID/IS 개념 이해 <a name='q4'></a>

### 문제

생성 모델의 품질 평가에 사용되는 두 가지 주요 지표를 이해하세요:

**Inception Score (IS):**

$$IS = \exp\!\left(\mathbb{E}_x\left[D_{KL}(p(y|x) \| p(y))\right]\right)$$

- $p(y|x)$: 생성 이미지의 클래스 확신도 (sharp할수록 좋음)
- $p(y)$: 전체 생성 이미지의 클래스 분포 (uniform할수록 좋음)

**Fréchet Inception Distance (FID):**

$$FID = \|\mu_r - \mu_g\|^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

- $(\mu_r, \Sigma_r)$: 실제 데이터의 Inception 특성 통계
- $(\mu_g, \Sigma_g)$: 생성 데이터의 Inception 특성 통계

**여러분의 예측:** FID는 낮을수록 좋나요, 높을수록 좋나요? `?`"""))

# ── Cell 10: Q4 Solution ─────────────────────────────────────────
cells.append(code(r"""# ── Q4 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q4 풀이: FID/IS 개념 이해")
print("=" * 45)

# 간소화된 FID 계산 (2D 가우시안으로 시뮬레이션)
def compute_simple_fid(real_data, gen_data):
    # 2D 데이터에서의 FID 근사
    mu_r = np.mean(real_data, axis=0)
    mu_g = np.mean(gen_data, axis=0)
    sigma_r = np.cov(real_data.T)
    sigma_g = np.cov(gen_data.T)

    diff = mu_r - mu_g
    mean_term = np.sum(diff ** 2)

    # sqrt(Sigma_r @ Sigma_g) 계산
    product = sigma_r @ sigma_g
    eigenvalues = np.linalg.eigvals(product)
    eigenvalues = np.maximum(eigenvalues.real, 0)
    sqrt_product_trace = np.sum(np.sqrt(eigenvalues))

    cov_term = np.trace(sigma_r) + np.trace(sigma_g) - 2 * sqrt_product_trace

    return mean_term + cov_term

def compute_simple_is(class_probs):
    # 간소화된 IS
    p_y_given_x = class_probs
    p_y = np.mean(p_y_given_x, axis=0, keepdims=True)
    kl_div = np.sum(p_y_given_x * (np.log(p_y_given_x + 1e-10) - np.log(p_y + 1e-10)), axis=1)
    return np.exp(np.mean(kl_div))

# 실제 데이터 (4클래스)
real_data = data_all.copy()

# 다양한 w에서 FID/IS 계산
print(f"\n{'w':>5} | {'FID (↓좋음)':>12} | {'IS (↑좋음)':>10} | {'평가':>10}")
print("-" * 48)

fid_values = []
is_values = []

for w in [0.0, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0]:
    np.random.seed(42)
    # 각 클래스에서 균등하게 생성
    gen_samples = []
    class_probs_list = []
    for c in range(4):
        samples = oracle_cfg_sample(n_per_class, c, w, alpha_bars, class_centers, n_steps=50)
        gen_samples.append(samples)

        # 간소화된 클래스 확률 (거리 기반)
        for s in samples:
            dists = np.array([np.linalg.norm(s - ctr) for ctr in class_centers])
            probs = np.exp(-dists) / np.sum(np.exp(-dists))
            class_probs_list.append(probs)

    gen_all = np.concatenate(gen_samples, axis=0)
    class_probs_arr = np.array(class_probs_list)

    fid = compute_simple_fid(real_data, gen_all)
    is_score = compute_simple_is(class_probs_arr)

    fid_values.append(fid)
    is_values.append(is_score)

    quality = "낮음" if fid > 5 else ("보통" if fid > 1 else "높음")
    print(f"w={w:>3.0f} | {fid:>12.3f} | {is_score:>10.3f} | {quality:>10}")

# 시각화
w_plot = [0.0, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
ax1.plot(w_plot, fid_values, 'bo-', lw=2.5, ms=8, label='FID (↓ 좋음)')
ax1.set_xlabel('Guidance Scale (w)', fontsize=11)
ax1.set_ylabel('FID', fontsize=11)
ax1.set_title('FID vs Guidance Scale', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(w_plot, is_values, 'ro-', lw=2.5, ms=8, label='IS (↑ 좋음)')
ax2.set_xlabel('Guidance Scale (w)', fontsize=11)
ax2.set_ylabel('Inception Score', fontsize=11)
ax2.set_title('IS vs Guidance Scale', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/practice/q4_fid_is_metrics.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n그래프 저장됨: chapter13_genai_diffusion/practice/q4_fid_is_metrics.png")

print("\n[해설]")
print("  FID는 낮을수록 좋습니다 (생성≈실제)")
print("  IS는 높을수록 좋습니다 (다양하면서도 뚜렷한 클래스)")
print("  w↑ → IS↑ (클래스 확신도 증가) but 과도하면 다양성↓")
print("  실무에서는 FID를 더 신뢰합니다 (IS는 모드 붕괴 감지 못함)")"""))

# ── Cell 11: Bonus Problem ───────────────────────────────────────
cells.append(md(r"""---

## 종합 도전: Mini CFG Generation Pipeline <a name='bonus'></a>

### 미니 구현

완전한 CFG 파이프라인을 구축하세요:
1. 조건부/비조건부 통합 모델 학습 (2D GMM 데이터)
2. CFG를 적용한 DDIM 샘플러로 생성
3. 각 클래스별 생성 결과 + 다양한 $w$의 효과를 시각화"""))

# ── Cell 12: Bonus Solution ──────────────────────────────────────
cells.append(code(r"""# ── 종합 도전 풀이 ──────────────────────────────────────────────
print("=" * 45)
print("종합 도전: Mini CFG Generation Pipeline")
print("=" * 45)

# 간단한 학습 루프
cfg_model = CFGNoiseModel(data_dim=2, n_classes=4, time_dim=32, hidden_dim=128)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 학습 데이터 텐서
x0_tf = tf.constant(data_all, dtype=tf.float32)
labels_tf = tf.constant(labels_all, dtype=tf.int32)
alpha_bars_tf = tf.constant(alpha_bars, dtype=tf.float32)

n_epochs = 100
batch_size = 256
p_uncond = 0.1
losses = []

print("모델 학습 시작...")
for epoch in range(n_epochs):
    indices = np.random.permutation(len(data_all))[:batch_size]
    x0_batch = tf.gather(x0_tf, indices)
    c_batch = tf.gather(labels_tf, indices)

    t_batch = tf.random.uniform([batch_size], 0, T, dtype=tf.int32)
    ab_batch = tf.gather(alpha_bars_tf, t_batch)

    eps = tf.random.normal(tf.shape(x0_batch))
    x_t = tf.sqrt(tf.reshape(ab_batch, [-1, 1])) * x0_batch + \
          tf.sqrt(1 - tf.reshape(ab_batch, [-1, 1])) * eps

    with tf.GradientTape() as tape:
        eps_pred = cfg_model(x_t, t_batch, class_label=c_batch, p_uncond=p_uncond, training=True)
        loss = tf.reduce_mean((eps - eps_pred) ** 2)

    grads = tape.gradient(loss, cfg_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, cfg_model.trainable_variables))
    losses.append(loss.numpy())

    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1:>3}/{n_epochs}, Loss: {loss.numpy():.4f}")

print(f"학습 완료! 최종 Loss: {losses[-1]:.4f}")

# CFG + DDIM 샘플링
def cfg_ddim_sample(model, n_samples, target_class, guidance_scale, alpha_bars_np, n_steps=50):
    step_indices = np.linspace(0, len(alpha_bars_np) - 1, n_steps + 1, dtype=int)[::-1]
    x = tf.random.normal((n_samples, 2))
    c_label = tf.fill([n_samples], target_class)

    for i in range(len(step_indices) - 1):
        tc, tp = step_indices[i], step_indices[i + 1]
        abc = alpha_bars_np[tc]
        abp = alpha_bars_np[tp]

        t_tensor = tf.fill([n_samples], tc)

        eps_uncond = model(x, t_tensor, class_label=None)
        eps_cond = model(x, t_tensor, class_label=c_label)
        eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        x0_hat = (x - tf.sqrt(1 - abc) * eps_cfg) / (tf.sqrt(abc) + 1e-8)
        x0_hat = tf.clip_by_value(x0_hat, -8, 8)
        x = tf.sqrt(abp) * x0_hat + tf.sqrt(1 - abp) * eps_cfg

    return x.numpy()

# 전체 결과 시각화
fig, axes = plt.subplots(2, 5, figsize=(22, 8))

w_test_values = [0.0, 1.0, 3.0, 7.0, 15.0]

for row, target_cls in enumerate([0, 3]):
    for col, w_val in enumerate(w_test_values):
        tf.random.set_seed(42)
        samples = cfg_ddim_sample(cfg_model, 200, target_cls, w_val, alpha_bars, n_steps=50)

        ax = axes[row, col]
        for c, ctr in enumerate(class_centers):
            color = 'red' if c == target_cls else 'lightgray'
            ax.scatter(*ctr, c=color, s=120, marker='*', zorder=5, edgecolors='black')
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.4, c='blue')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title(f'w = {w_val}', fontweight='bold', fontsize=12)
        if col == 0:
            ax.set_ylabel(f'클래스 {target_cls}', fontsize=12, fontweight='bold')

plt.suptitle('Mini CFG Pipeline: 클래스별 × Guidance Scale별 생성 결과',
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/practice/bonus_cfg_pipeline.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/practice/bonus_cfg_pipeline.png")

# 학습 곡선
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(losses, 'b-', alpha=0.3, lw=0.5)
window = 10
smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
ax.plot(range(window-1, len(losses)), smoothed, 'r-', lw=2, label=f'{window}-epoch 이동평균')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('MSE Loss', fontsize=11)
ax.set_title('CFG 모델 학습 곡선', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('chapter13_genai_diffusion/practice/bonus_training_curve.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter13_genai_diffusion/practice/bonus_training_curve.png")

print("\n[결론]")
print("  1. 조건 드롭(p_uncond=0.1)이 CFG의 핵심 학습 전략입니다")
print("  2. w=0: 비조건부 → w=1: 표준 조건부 → w>1: 조건 과강조")
print("  3. 실제 Stable Diffusion은 w=7.5, DALL-E 3는 동적 w를 사용합니다")
print("  4. CFG는 추가 파라미터 없이 단순한 추론 기법으로 생성 품질을 크게 향상시킵니다!")"""))

# ── Create notebook ──────────────────────────────────────────────
path = '/workspace/chapter13_genai_diffusion/practice/ex02_implement_cfg_generation.ipynb'
create_notebook(cells, path)
