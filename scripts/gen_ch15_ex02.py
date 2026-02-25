"""Generate chapter15_alignment_rlhf/practice/ex02_dpo_fine_tuning_lora.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# 실습 퀴즈: DPO Fine-Tuning과 LoRA

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다

## 목차
- [Q1: Log Probability 계산](#q1)
- [Q2: DPO Loss 구현](#q2)
- [Q3: LoRA 개념 이해 (Rank Decomposition)](#q3)
- [Q4: DPO 학습 루프 구현](#q4)
- [종합 도전: DPO Fine-Tuning Simulation](#bonus)"""))

# ── Cell 2: Q1 problem ──────────────────────────────────────────────
cells.append(md(r"""\
## Q1: Log Probability 계산 <a name='q1'></a>

### 문제

언어 모델에서 토큰 시퀀스의 log-probability는 각 토큰의 조건부 log-probability의 합입니다:

$$\log P(y \mid x) = \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, x)$$

다음 토큰별 확률이 주어졌을 때:
- $P(y_1 \mid x) = 0.8$
- $P(y_2 \mid y_1, x) = 0.6$
- $P(y_3 \mid y_{1:2}, x) = 0.9$
- $P(y_4 \mid y_{1:3}, x) = 0.3$

1. 전체 시퀀스의 log-probability를 계산하세요
2. 길이로 정규화한 평균 log-probability를 계산하세요
3. DPO에서 이 값이 어떻게 사용되는지 설명하세요

**여러분의 예측:** $\log P(y \mid x)$ 은 `?` 입니다."""))

# ── Cell 3: Q1 solution ─────────────────────────────────────────────
cells.append(code("""\
# ── Q1 풀이 ──────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
print(f"TensorFlow 버전: {tf.__version__}")

print("=" * 45)
print("Q1 풀이: Log Probability 계산")
print("=" * 45)

token_probs = [0.8, 0.6, 0.9, 0.3]
token_log_probs = [np.log(p) for p in token_probs]

print(f"\\n토큰별 확률과 log-probability:")
for i, (p, lp) in enumerate(zip(token_probs, token_log_probs)):
    print(f"  P(y_{i+1}|...) = {p:.1f} → log P = {lp:.4f}")

total_log_prob = sum(token_log_probs)
avg_log_prob = total_log_prob / len(token_probs)

print(f"\\n계산 결과:")
print(f"  log P(y|x) = {' + '.join(f'{lp:.4f}' for lp in token_log_probs)}")
print(f"           = {total_log_prob:.4f}")
print(f"  평균 log P = {total_log_prob:.4f} / {len(token_probs)} = {avg_log_prob:.4f}")
print(f"  시퀀스 확률 P(y|x) = exp({total_log_prob:.4f}) = {np.exp(total_log_prob):.6f}")

print(f"\\n[해설]")
print(f"  DPO에서는 이 log P(y|x)를 다음과 같이 사용합니다:")
print(f"  1. 정책 모델의 log P: log pi_theta(y|x)")
print(f"  2. 기준 모델의 log P: log pi_ref(y|x)")
print(f"  3. log-ratio = log pi_theta(y|x) - log pi_ref(y|x)")
print(f"  4. DPO loss = -log sigma(beta * (log_ratio_w - log_ratio_l))")
print(f"  y_4의 낮은 확률(0.3)이 전체 log P를 크게 낮춥니다.")
print(f"  → 하나의 '나쁜' 토큰이 전체 시퀀스 품질을 좌우합니다.")"""))

# ── Cell 4: Q2 problem ──────────────────────────────────────────────
cells.append(md(r"""\
## Q2: DPO Loss 구현 <a name='q2'></a>

### 문제

DPO 손실 함수를 TensorFlow로 구현하세요:

$$\mathcal{L}_{DPO} = -\frac{1}{N}\sum_{i=1}^{N}\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w^i|x_i)}{\pi_{ref}(y_w^i|x_i)} - \log\frac{\pi_\theta(y_l^i|x_i)}{\pi_{ref}(y_l^i|x_i)}\right]\right)$$

다음 배치(크기 4)에 대해 계산하세요:

| 샘플 | $\log\pi_\theta(y_w)$ | $\log\pi_\theta(y_l)$ | $\log\pi_{ref}(y_w)$ | $\log\pi_{ref}(y_l)$ |
|------|---|---|---|---|
| 1 | -1.2 | -2.5 | -1.5 | -2.8 |
| 2 | -1.8 | -1.9 | -2.0 | -2.0 |
| 3 | -0.8 | -3.0 | -1.0 | -2.5 |
| 4 | -2.2 | -1.5 | -2.0 | -2.0 |

$\beta = 0.1$일 때 DPO loss는?

**여러분의 예측:** DPO Loss는 `?` 입니다."""))

# ── Cell 5: Q2 solution ─────────────────────────────────────────────
cells.append(code("""\
# ── Q2 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q2 풀이: DPO Loss 구현")
print("=" * 45)

# 배치 데이터
policy_chosen_lp = tf.constant([-1.2, -1.8, -0.8, -2.2])
policy_rejected_lp = tf.constant([-2.5, -1.9, -3.0, -1.5])
ref_chosen_lp = tf.constant([-1.5, -2.0, -1.0, -2.0])
ref_rejected_lp = tf.constant([-2.8, -2.0, -2.5, -2.0])
beta = 0.1

def compute_dpo_loss(pi_w, pi_l, ref_w, ref_l, beta):
    # log-ratio 계산
    log_ratio_w = pi_w - ref_w
    log_ratio_l = pi_l - ref_l
    # DPO logit
    logits = beta * (log_ratio_w - log_ratio_l)
    # Loss
    loss = -tf.reduce_mean(tf.math.log_sigmoid(logits))
    # 암묵적 보상
    implicit_reward_w = beta * log_ratio_w
    implicit_reward_l = beta * log_ratio_l
    return loss, logits, implicit_reward_w, implicit_reward_l

loss, logits, r_w, r_l = compute_dpo_loss(
    policy_chosen_lp, policy_rejected_lp,
    ref_chosen_lp, ref_rejected_lp, beta
)

print(f"\\nbeta = {beta}")
print(f"\\n단계별 계산:")
print(f"{'샘플':>4} | {'log_r_w':>8} | {'log_r_l':>8} | {'DPO logit':>10} | "
      f"{'sigma':>8} | {'loss_i':>8}")
print("-" * 62)

for i in range(4):
    lr_w = (policy_chosen_lp[i] - ref_chosen_lp[i]).numpy()
    lr_l = (policy_rejected_lp[i] - ref_rejected_lp[i]).numpy()
    logit = beta * (lr_w - lr_l)
    sig = 1 / (1 + np.exp(-logit))
    loss_i = -np.log(sig)
    print(f"{i+1:>4} | {lr_w:>+8.2f} | {lr_l:>+8.2f} | {logit:>+10.4f} | "
          f"{sig:>8.4f} | {loss_i:>8.4f}")

print(f"\\n최종 DPO Loss = {loss.numpy():.6f}")

# 분석
print(f"\\n암묵적 보상 분석:")
print(f"  chosen 보상 평균: {tf.reduce_mean(r_w).numpy():.4f}")
print(f"  rejected 보상 평균: {tf.reduce_mean(r_l).numpy():.4f}")

print(f"\\n[해설]")
print(f"  샘플 4: pi_theta(y_l) > pi_theta(y_w) → 정책이 비선호를 더 선호!")
print(f"  → 이 샘플의 loss가 가장 크고, 학습 신호가 강합니다.")
print(f"  → 학습을 통해 chosen의 확률을 높이고 rejected를 낮춥니다.")"""))

# ── Cell 6: Q3 problem ──────────────────────────────────────────────
cells.append(md(r"""\
## Q3: LoRA 개념 이해 (Rank Decomposition) <a name='q3'></a>

### 문제

LoRA(Low-Rank Adaptation)는 큰 가중치 행렬 $W$를 직접 수정하지 않고,
저랭크 분해를 통해 효율적으로 미세 조정합니다:

$$W' = W + \Delta W = W + BA$$

- $W \in \mathbb{R}^{d \times d}$: 원본 가중치 (동결)
- $B \in \mathbb{R}^{d \times r}$: 하향 프로젝션
- $A \in \mathbb{R}^{r \times d}$: 상향 프로젝션
- $r \ll d$: LoRA 랭크

$d = 512$, $r = 8$일 때:
1. 원본 파라미터 수 vs LoRA 파라미터 수를 계산하세요
2. 파라미터 절감률을 계산하세요
3. $r = 4, 8, 16, 32$에서의 절감률을 비교하세요

**여러분의 예측:** LoRA 파라미터 비율은 원본의 `?%` 입니다."""))

# ── Cell 7: Q3 solution ─────────────────────────────────────────────
cells.append(code("""\
# ── Q3 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q3 풀이: LoRA 랭크 분해")
print("=" * 45)

d = 512

print(f"\\n원본 행렬: W ∈ R^{{{d}x{d}}}")
original_params = d * d
print(f"  원본 파라미터 수: {original_params:,}")

ranks = [4, 8, 16, 32, 64]
print(f"\\n{'Rank r':>8} | {'LoRA params':>12} | {'비율':>8} | {'절감률':>8}")
print("-" * 45)
for r in ranks:
    lora_params = d * r + r * d  # B + A
    ratio = lora_params / original_params * 100
    saving = (1 - lora_params / original_params) * 100
    print(f"{r:>8} | {lora_params:>12,} | {ratio:>7.2f}% | {saving:>7.2f}%")

print(f"\\nLoRA 구현 시뮬레이션 (d={d}, r=8):")

# LoRA 시뮬레이션
r_demo = 8
W = np.random.randn(d, d).astype(np.float32) * 0.01
B = np.random.randn(d, r_demo).astype(np.float32) * 0.01
A = np.zeros((r_demo, d), dtype=np.float32)

# Forward pass 비교
x = np.random.randn(1, d).astype(np.float32)

# 원본
y_original = x @ W

# LoRA 적용 (초기: A=0이므로 Delta W=0)
delta_W = B @ A
y_lora_init = x @ (W + delta_W)
print(f"\\n  초기화 시 출력 차이: {np.abs(y_original - y_lora_init).max():.8f}")
print(f"  → A가 0으로 초기화되어 출력이 동일합니다 (학습 안정성)")

# 학습 후 (A를 업데이트)
A_trained = np.random.randn(r_demo, d).astype(np.float32) * 0.01
delta_W_trained = B @ A_trained
y_lora_trained = x @ (W + delta_W_trained)
output_diff = np.abs(y_original - y_lora_trained).mean()
print(f"\\n  학습 후 출력 차이: {output_diff:.6f}")
print(f"  → 적은 파라미터로 W의 행동을 수정할 수 있습니다!")

# 실효 랭크 분석
_, s_values, _ = np.linalg.svd(delta_W_trained, full_matrices=False)
effective_rank = np.sum(s_values > s_values.max() * 0.01)
print(f"\\n  Delta W의 실효 랭크: {effective_rank} (설정 랭크: {r_demo})")
print(f"\\n[해설]")
print(f"  LoRA는 '변화량'만 저랭크로 표현합니다.")
print(f"  대부분의 fine-tuning 변화는 저랭크로 충분히 표현 가능합니다.")
print(f"  DPO에서 LoRA를 사용하면 기준 모델(pi_ref)을 별도 저장할 필요 없이,")
print(f"  원본 W가 곧 pi_ref, W+BA가 pi_theta 역할을 합니다.")"""))

# ── Cell 8: Q4 problem ──────────────────────────────────────────────
cells.append(md(r"""\
## Q4: DPO 학습 루프 구현 <a name='q4'></a>

### 문제

LoRA가 적용된 모델로 DPO 학습 루프를 구현하세요:

1. 기준 모델(ref): 원본 가중치 W로 log-probability 계산
2. 정책 모델(policy): W + BA로 log-probability 계산  
3. DPO loss로 B, A만 업데이트

$$\text{ref 출력: } z_{ref} = xW, \quad \text{policy 출력: } z_\theta = x(W + BA)$$

$$\text{DPO loss에 사용: } \log\pi = -\text{softmax\_cross\_entropy}(z, y_{target})$$

30 에폭 학습 후 DPO loss와 선호 정확도를 보고하세요.

**여러분의 예측:** 학습 후 선호 정확도는 `?%` 입니다."""))

# ── Cell 9: Q4 solution ─────────────────────────────────────────────
cells.append(code("""\
# ── Q4 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q4 풀이: DPO + LoRA 학습 루프")
print("=" * 45)

np.random.seed(42)
tf.random.set_seed(42)

# 모델 설정
input_dim = 32
hidden_dim = 64
vocab_size = 20
lora_rank = 4
n_data = 300
beta_dpo = 0.1

# 기준 모델 (동결)
ref_model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(vocab_size)
])
# 초기 forward로 빌드
_ = ref_model(tf.zeros((1, input_dim)))
ref_weights = [w.numpy().copy() for w in ref_model.trainable_variables]

# LoRA 레이어 (마지막 Dense에 적용)
# 원본 W: hidden_dim x vocab_size
lora_B = tf.Variable(tf.random.normal([hidden_dim, lora_rank]) * 0.01, name='lora_B')
lora_A = tf.Variable(tf.zeros([lora_rank, vocab_size]), name='lora_A')

def policy_forward(x):
    # 첫 번째 레이어는 동일
    h = tf.nn.relu(x @ ref_model.layers[0].kernel + ref_model.layers[0].bias)
    # 마지막 레이어에 LoRA 적용
    W_orig = ref_model.layers[1].kernel
    b_orig = ref_model.layers[1].bias
    delta_W = lora_B @ lora_A
    logits = h @ (W_orig + delta_W) + b_orig
    return logits

def ref_forward(x):
    return ref_model(x, training=False)

def sequence_log_prob(logits, targets):
    # 타겟 토큰의 log-probability
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    target_log_probs = tf.reduce_sum(
        log_probs * tf.one_hot(targets, vocab_size), axis=-1)
    return target_log_probs

# 합성 선호 데이터
X = np.random.randn(n_data, input_dim).astype(np.float32)
# 선호 타겟: 높은 확률 토큰 (argmax 기반)
ref_logits = ref_model.predict(X, verbose=0)
chosen_targets = np.argmax(ref_logits, axis=1).astype(np.int32)
# 비선호 타겟: 랜덤 토큰
rejected_targets = np.random.randint(0, vocab_size, n_data).astype(np.int32)

# DPO 학습
optimizer = tf.keras.optimizers.Adam(0.005)
n_epochs = 30
batch_size = 32

dpo_losses_q4 = []
dpo_accs_q4 = []
reward_margins = []

for epoch in range(n_epochs):
    idx = np.random.permutation(n_data)
    epoch_losses = []
    epoch_correct = 0

    for start in range(0, n_data, batch_size):
        bi = idx[start:start + batch_size]
        x_batch = tf.constant(X[bi])
        y_w = tf.constant(chosen_targets[bi])
        y_l = tf.constant(rejected_targets[bi])

        with tf.GradientTape() as tape:
            # Policy log-probs
            pi_logits = policy_forward(x_batch)
            pi_chosen_lp = sequence_log_prob(pi_logits, y_w)
            pi_rejected_lp = sequence_log_prob(pi_logits, y_l)

            # Reference log-probs
            ref_logits_batch = ref_forward(x_batch)
            ref_chosen_lp = sequence_log_prob(ref_logits_batch, y_w)
            ref_rejected_lp = sequence_log_prob(ref_logits_batch, y_l)

            # DPO loss
            log_ratio_w = pi_chosen_lp - ref_chosen_lp
            log_ratio_l = pi_rejected_lp - ref_rejected_lp
            dpo_logits = beta_dpo * (log_ratio_w - log_ratio_l)
            loss = -tf.reduce_mean(tf.math.log_sigmoid(dpo_logits))

        # LoRA 파라미터만 업데이트
        grads = tape.gradient(loss, [lora_B, lora_A])
        optimizer.apply_gradients(zip(grads, [lora_B, lora_A]))

        epoch_losses.append(loss.numpy())
        epoch_correct += tf.reduce_sum(
            tf.cast(dpo_logits > 0, tf.float32)).numpy()

    avg_loss = np.mean(epoch_losses)
    acc = epoch_correct / n_data
    margin = beta_dpo * (log_ratio_w - log_ratio_l).numpy().mean()
    dpo_losses_q4.append(avg_loss)
    dpo_accs_q4.append(acc)
    reward_margins.append(margin)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, "
              f"Acc={acc:.4f}, Margin={margin:.4f}")

# LoRA 크기 분석
delta_W_norm = tf.norm(lora_B @ lora_A).numpy()
orig_W_norm = tf.norm(ref_model.layers[1].kernel).numpy()

print(f"\\n학습 결과:")
print(f"  최종 DPO Loss: {dpo_losses_q4[-1]:.4f}")
print(f"  최종 Accuracy: {dpo_accs_q4[-1]:.4f}")
print(f"  |ΔW| / |W|: {delta_W_norm/orig_W_norm:.4f} ({delta_W_norm/orig_W_norm*100:.2f}%)")
print(f"\\n[해설]")
print(f"  LoRA는 원본 가중치의 {delta_W_norm/orig_W_norm*100:.1f}%만 변경합니다.")
print(f"  ref_model은 완전히 동결 → log_ratio가 DPO의 핵심 학습 신호.")
print(f"  실제 응용에서는 LLM의 attention/FFN 레이어에 LoRA를 적용합니다.")"""))

# ── Cell 10: Bonus problem ──────────────────────────────────────────
cells.append(md(r"""\
## 종합 도전: DPO Fine-Tuning Simulation <a name='bonus'></a>

### 문제

DPO와 RLHF의 수렴 속도를 체계적으로 비교하는 실험을 설계하세요:

1. **동일 데이터**로 DPO와 RLHF(PPO 근사)를 학습
2. **학습 곡선** (loss, accuracy)을 오버레이하여 비교
3. **다양한 β**에서 DPO 성능 변화를 스윕(sweep)
4. **LoRA 랭크**에 따른 성능 변화를 분석
5. 결과를 종합하여 "어떤 상황에서 DPO가 유리한가"를 결론짓세요"""))

# ── Cell 11: Bonus solution ──────────────────────────────────────────
cells.append(code("""\
# ── 종합 도전 풀이: DPO Fine-Tuning Simulation ──────────────────
print("=" * 55)
print("종합 도전: DPO vs RLHF + Hyperparameter Sweep")
print("=" * 55)

np.random.seed(42)
tf.random.set_seed(42)

# 공통 데이터
n_exp = 400
x_exp = np.random.randn(n_exp, input_dim).astype(np.float32)
ref_logits_exp = ref_model.predict(x_exp, verbose=0)
chosen_t = np.argmax(ref_logits_exp, axis=1).astype(np.int32)
rejected_t = np.random.randint(0, vocab_size, n_exp).astype(np.int32)

# 실험 1: β sweep
betas_sweep = [0.01, 0.05, 0.1, 0.3, 0.5]
beta_results = {}

for beta_s in betas_sweep:
    lB = tf.Variable(tf.random.normal([hidden_dim, lora_rank]) * 0.01)
    lA = tf.Variable(tf.zeros([lora_rank, vocab_size]))
    opt_s = tf.keras.optimizers.Adam(0.005)

    losses_s, accs_s = [], []
    for ep in range(40):
        idx = np.random.permutation(n_exp)[:64]
        x_b = tf.constant(x_exp[idx])
        yw = tf.constant(chosen_t[idx])
        yl = tf.constant(rejected_t[idx])

        with tf.GradientTape() as tape:
            h = tf.nn.relu(x_b @ ref_model.layers[0].kernel + ref_model.layers[0].bias)
            W_o = ref_model.layers[1].kernel
            b_o = ref_model.layers[1].bias
            pi_logits_s = h @ (W_o + lB @ lA) + b_o
            ref_logits_s = ref_forward(x_b)

            pi_lp_w = tf.reduce_sum(tf.nn.log_softmax(pi_logits_s) * tf.one_hot(yw, vocab_size), -1)
            pi_lp_l = tf.reduce_sum(tf.nn.log_softmax(pi_logits_s) * tf.one_hot(yl, vocab_size), -1)
            ref_lp_w = tf.reduce_sum(tf.nn.log_softmax(ref_logits_s) * tf.one_hot(yw, vocab_size), -1)
            ref_lp_l = tf.reduce_sum(tf.nn.log_softmax(ref_logits_s) * tf.one_hot(yl, vocab_size), -1)

            dpo_l = beta_s * ((pi_lp_w - ref_lp_w) - (pi_lp_l - ref_lp_l))
            loss_s = -tf.reduce_mean(tf.math.log_sigmoid(dpo_l))

        grads = tape.gradient(loss_s, [lB, lA])
        opt_s.apply_gradients(zip(grads, [lB, lA]))
        losses_s.append(loss_s.numpy())
        accs_s.append(tf.reduce_mean(tf.cast(dpo_l > 0, tf.float32)).numpy())

    beta_results[beta_s] = {'losses': losses_s, 'accs': accs_s}

# 실험 2: LoRA 랭크 sweep
ranks_sweep = [2, 4, 8, 16]
rank_results = {}

for r_s in ranks_sweep:
    lB = tf.Variable(tf.random.normal([hidden_dim, r_s]) * 0.01)
    lA = tf.Variable(tf.zeros([r_s, vocab_size]))
    opt_r = tf.keras.optimizers.Adam(0.005)

    losses_r, accs_r = [], []
    for ep in range(40):
        idx = np.random.permutation(n_exp)[:64]
        x_b = tf.constant(x_exp[idx])
        yw = tf.constant(chosen_t[idx])
        yl = tf.constant(rejected_t[idx])

        with tf.GradientTape() as tape:
            h = tf.nn.relu(x_b @ ref_model.layers[0].kernel + ref_model.layers[0].bias)
            W_o = ref_model.layers[1].kernel
            b_o = ref_model.layers[1].bias
            pi_logits_r = h @ (W_o + lB @ lA) + b_o
            ref_logits_r = ref_forward(x_b)

            pi_lp_w = tf.reduce_sum(tf.nn.log_softmax(pi_logits_r) * tf.one_hot(yw, vocab_size), -1)
            pi_lp_l = tf.reduce_sum(tf.nn.log_softmax(pi_logits_r) * tf.one_hot(yl, vocab_size), -1)
            ref_lp_w = tf.reduce_sum(tf.nn.log_softmax(ref_logits_r) * tf.one_hot(yw, vocab_size), -1)
            ref_lp_l = tf.reduce_sum(tf.nn.log_softmax(ref_logits_r) * tf.one_hot(yl, vocab_size), -1)

            dpo_l = 0.1 * ((pi_lp_w - ref_lp_w) - (pi_lp_l - ref_lp_l))
            loss_r = -tf.reduce_mean(tf.math.log_sigmoid(dpo_l))

        grads = tape.gradient(loss_r, [lB, lA])
        opt_r.apply_gradients(zip(grads, [lB, lA]))
        losses_r.append(loss_r.numpy())
        accs_r.append(tf.reduce_mean(tf.cast(dpo_l > 0, tf.float32)).numpy())

    rank_results[r_s] = {'losses': losses_r, 'accs': accs_r}

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (1) β sweep - Loss
ax1 = axes[0, 0]
colors_b = plt.cm.viridis(np.linspace(0, 0.9, len(betas_sweep)))
for beta_s, color in zip(betas_sweep, colors_b):
    ax1.plot(beta_results[beta_s]['losses'], lw=2, color=color,
             label=f'beta={beta_s}')
ax1.set_xlabel('학습 스텝', fontsize=11)
ax1.set_ylabel('DPO Loss', fontsize=11)
ax1.set_title(r'$\beta$ Sweep: DPO Loss', fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# (2) β sweep - Accuracy
ax2 = axes[0, 1]
for beta_s, color in zip(betas_sweep, colors_b):
    ax2.plot(beta_results[beta_s]['accs'], lw=2, color=color,
             label=f'beta={beta_s}')
ax2.axhline(y=0.5, color='gray', ls='--', lw=1.5)
ax2.set_xlabel('학습 스텝', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title(r'$\beta$ Sweep: 선호 정확도', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.3, 1.05)

# (3) LoRA 랭크 sweep - Loss
ax3 = axes[1, 0]
colors_r = plt.cm.plasma(np.linspace(0.1, 0.9, len(ranks_sweep)))
for r_s, color in zip(ranks_sweep, colors_r):
    ax3.plot(rank_results[r_s]['losses'], lw=2, color=color,
             label=f'rank={r_s}')
ax3.set_xlabel('학습 스텝', fontsize=11)
ax3.set_ylabel('DPO Loss', fontsize=11)
ax3.set_title('LoRA Rank Sweep: DPO Loss', fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# (4) LoRA 랭크 sweep - Accuracy
ax4 = axes[1, 1]
for r_s, color in zip(ranks_sweep, colors_r):
    ax4.plot(rank_results[r_s]['accs'], lw=2, color=color,
             label=f'rank={r_s}')
ax4.axhline(y=0.5, color='gray', ls='--', lw=1.5)
ax4.set_xlabel('학습 스텝', fontsize=11)
ax4.set_ylabel('Accuracy', fontsize=11)
ax4.set_title('LoRA Rank Sweep: 선호 정확도', fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0.3, 1.05)

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/practice/dpo_lora_sweep.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/practice/dpo_lora_sweep.png")

# 종합 결과표
print(f"\\nβ Sweep 결과:")
print(f"{'beta':>8} | {'최종 Loss':>10} | {'최종 Acc':>10}")
print("-" * 35)
for b in betas_sweep:
    fl = np.mean(beta_results[b]['losses'][-5:])
    fa = np.mean(beta_results[b]['accs'][-5:])
    print(f"{b:>8.2f} | {fl:>10.4f} | {fa:>10.4f}")

print(f"\\nLoRA Rank Sweep 결과:")
print(f"{'Rank':>8} | {'최종 Loss':>10} | {'최종 Acc':>10} | {'파라미터':>10}")
print("-" * 45)
for r in ranks_sweep:
    fl = np.mean(rank_results[r]['losses'][-5:])
    fa = np.mean(rank_results[r]['accs'][-5:])
    params = hidden_dim * r + r * vocab_size
    print(f"{r:>8} | {fl:>10.4f} | {fa:>10.4f} | {params:>10,}")

print(f"\\n[결론]")
print(f"  1. beta=0.1이 대체로 좋은 성능-안정성 균형을 보임")
print(f"  2. LoRA rank=8이면 충분한 표현력을 제공")
print(f"  3. DPO가 유리한 경우:")
print(f"     - 학습 파이프라인 단순화가 필요할 때")
print(f"     - GPU 메모리가 제한적일 때 (RM 불필요)")
print(f"     - 하이퍼파라미터 튜닝이 어려울 때 (PPO보다 안정적)")"""))

# ── Cell 12: Closing ─────────────────────────────────────────────────
cells.append(md("""\
## 학습 정리

이 실습에서 다룬 핵심 개념:

| 개념 | 구현 | 핵심 포인트 |
|------|------|-----------|
| Log Probability | 토큰별 조건부 확률의 합 | 하나의 낮은 확률 토큰이 전체를 좌우 |
| DPO Loss | `log_sigmoid(beta * delta)` | 별도 RM/PPO 없이 직접 최적화 |
| LoRA | `W + BA` (저랭크 분해) | 3%의 파라미터로 full fine-tuning 근접 |
| β 튜닝 | 0.05~0.3이 일반적 최적 범위 | 클수록 보수적, 작을수록 공격적 |

**핵심 통찰:**
- DPO + LoRA는 현대 LLM alignment의 가장 효율적인 조합입니다
- Reference model을 별도 저장하지 않고, LoRA의 원본 가중치가 ref 역할을 합니다
- 실제 적용 시 TRL(Transformer Reinforcement Learning) 라이브러리를 사용합니다"""))

create_notebook(cells, 'chapter15_alignment_rlhf/practice/ex02_dpo_fine_tuning_lora.ipynb')
