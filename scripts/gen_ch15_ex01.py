"""Generate chapter15_alignment_rlhf/practice/ex01_train_reward_model.ipynb"""
import sys, os
sys.path.insert(0, '/workspace/scripts')
from nb_helper import md, code, create_notebook

cells = []

# ── Cell 1: Header ──────────────────────────────────────────────────
cells.append(md("""\
# 실습 퀴즈: Reward Model 훈련

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다

## 목차
- [Q1: Bradley-Terry 확률 계산](#q1)
- [Q2: Reward Model 손실 함수 구현](#q2)
- [Q3: Preference Pair 데이터 생성](#q3)
- [Q4: Reward Model 학습 및 평가](#q4)
- [종합 도전: Full RM Training Pipeline](#bonus)"""))

# ── Cell 2: Q1 problem ──────────────────────────────────────────────
cells.append(md(r"""\
## Q1: Bradley-Terry 확률 계산 <a name='q1'></a>

### 문제

Bradley-Terry 선호 모델에서, Reward Model이 두 응답에 대해 다음 점수를 출력했습니다:

$$r_\theta(x, y_w) = 3.2, \quad r_\theta(x, y_l) = 1.8$$

1. 선호 확률 $P(y_w \succ y_l)$을 계산하세요
2. 보상 차이가 2배($r(y_w)=6.4, r(y_l)=3.6$)가 되면 확률은?
3. 두 결과의 차이가 의미하는 바를 설명하세요

**Bradley-Terry 수식:**
$$P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l)) = \frac{1}{1 + e^{-(r(y_w) - r(y_l))}}$$

**여러분의 예측:** $P(y_w \succ y_l)$ 은 `?` 입니다."""))

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
print("Q1 풀이: Bradley-Terry 확률 계산")
print("=" * 45)

def bt_probability(r_w, r_l):
    diff = r_w - r_l
    return 1.0 / (1.0 + np.exp(-diff))

# Case 1: r(y_w) = 3.2, r(y_l) = 1.8
r_w1, r_l1 = 3.2, 1.8
p1 = bt_probability(r_w1, r_l1)
print(f"\\n[Case 1] r(y_w)={r_w1}, r(y_l)={r_l1}")
print(f"  보상 차이: {r_w1 - r_l1:.1f}")
print(f"  P(y_w > y_l) = sigma({r_w1 - r_l1:.1f}) = {p1:.6f}")
print(f"  = 1 / (1 + exp(-{r_w1 - r_l1:.1f})) = 1 / (1 + {np.exp(-(r_w1-r_l1)):.4f}) = {p1:.6f}")

# Case 2: 2배 보상
r_w2, r_l2 = 6.4, 3.6
p2 = bt_probability(r_w2, r_l2)
print(f"\\n[Case 2] r(y_w)={r_w2}, r(y_l)={r_l2}")
print(f"  보상 차이: {r_w2 - r_l2:.1f}")
print(f"  P(y_w > y_l) = sigma({r_w2 - r_l2:.1f}) = {p2:.6f}")

# 해설
print(f"\\n[해설]")
print(f"  Case 1 확률: {p1:.4f} (약 {p1*100:.1f}%)")
print(f"  Case 2 확률: {p2:.4f} (약 {p2*100:.1f}%)")
print(f"  차이: {(p2-p1)*100:.2f}%p 증가")
print(f"  → 보상 값 자체가 아닌 '차이'가 확률을 결정합니다.")
print(f"  → 차이가 같으면(1.4 vs 2.8) 확률이 달라집니다.")
print(f"  → 시그모이드 함수의 비선형성 때문입니다.")"""))

# ── Cell 4: Q2 problem ──────────────────────────────────────────────
cells.append(md(r"""\
## Q2: Reward Model 손실 함수 구현 <a name='q2'></a>

### 문제

Bradley-Terry 기반 Reward Model의 손실 함수를 TensorFlow로 구현하세요:

$$\mathcal{L}_{RM}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log \sigma\left(r_\theta(x_i, y_w^i) - r_\theta(x_i, y_l^i)\right)$$

다음 배치에 대해 손실을 계산하세요:
- 배치 크기: 4
- 선호 보상: `[2.1, 1.5, 3.0, 0.8]`
- 비선호 보상: `[1.0, 1.2, 0.5, 0.9]`

**여러분의 예측:** 손실값은 `?`, 정확도는 `?/4` 입니다."""))

# ── Cell 5: Q2 solution ─────────────────────────────────────────────
cells.append(code("""\
# ── Q2 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q2 풀이: Reward Model 손실 함수")
print("=" * 45)

def reward_model_loss(r_chosen, r_rejected):
    # Bradley-Terry 기반 RM 손실
    diff = r_chosen - r_rejected
    loss = -tf.reduce_mean(tf.math.log_sigmoid(diff))
    accuracy = tf.reduce_mean(tf.cast(diff > 0, tf.float32))
    return loss, accuracy, diff

r_chosen = tf.constant([2.1, 1.5, 3.0, 0.8])
r_rejected = tf.constant([1.0, 1.2, 0.5, 0.9])

loss, acc, diffs = reward_model_loss(r_chosen, r_rejected)

print(f"\\n입력 데이터:")
print(f"  r_chosen:   {r_chosen.numpy()}")
print(f"  r_rejected: {r_rejected.numpy()}")
print(f"  차이 (Δr):  {diffs.numpy()}")

print(f"\\n단계별 계산:")
for i in range(4):
    d = diffs.numpy()[i]
    sig = 1 / (1 + np.exp(-d))
    log_sig = np.log(sig)
    print(f"  [{i}] Δr={d:+.1f} → σ(Δr)={sig:.4f} → log σ={log_sig:.4f} "
          f"{'✅' if d > 0 else '❌'}")

print(f"\\n결과:")
print(f"  RM Loss = -mean(log_sigmoid) = {loss.numpy():.4f}")
print(f"  정확도 = {acc.numpy():.2f} ({int(acc.numpy()*4)}/4)")
print(f"\\n[해설]")
print(f"  샘플 4번째: r_chosen(0.8) < r_rejected(0.9) → 순서 역전!")
print(f"  이 역전된 샘플이 손실을 크게 만듭니다.")
print(f"  학습을 통해 chosen의 보상을 높이고 rejected를 낮춰야 합니다.")"""))

# ── Cell 6: Q3 problem ──────────────────────────────────────────────
cells.append(md("""\
## Q3: Preference Pair 데이터 생성 <a name='q3'></a>

### 문제

실제 RLHF/DPO 학습에서는 선호 쌍(Preference Pair) 데이터가 필요합니다.
다음 조건으로 합성 데이터를 생성하세요:

1. 프롬프트: 10차원 랜덤 벡터 (300개)
2. 선호 응답: 프롬프트의 선형 변환 + 작은 노이즈
3. 비선호 응답: 프롬프트의 선형 변환 + 큰 노이즈 + 약간의 편향
4. 품질 점수: 프롬프트와 응답의 코사인 유사도로 계산

**여러분의 예측:** 선호 응답의 평균 품질 점수는 비선호보다 `?`만큼 높습니다."""))

# ── Cell 7: Q3 solution ─────────────────────────────────────────────
cells.append(code("""\
# ── Q3 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q3 풀이: Preference Pair 데이터 생성")
print("=" * 45)

np.random.seed(42)
n_pairs = 300
prompt_dim = 10
response_dim = 10

# 프롬프트 생성
prompts = np.random.randn(n_pairs, prompt_dim).astype(np.float32)

# 이상적 변환 행렬
W_ideal = np.random.randn(prompt_dim, response_dim).astype(np.float32) * 0.3

# 선호 응답: 이상적 변환 + 작은 노이즈
chosen_responses = prompts @ W_ideal + np.random.randn(n_pairs, response_dim).astype(np.float32) * 0.1
# 비선호 응답: 이상적 변환 + 큰 노이즈 + 편향
rejected_responses = prompts @ W_ideal + np.random.randn(n_pairs, response_dim).astype(np.float32) * 0.8 + 0.5

# 품질 점수: 이상적 응답과의 코사인 유사도
ideal_responses = prompts @ W_ideal
def cosine_similarity(a, b):
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b + 1e-8)

chosen_quality = cosine_similarity(chosen_responses, ideal_responses)
rejected_quality = cosine_similarity(rejected_responses, ideal_responses)

print(f"\\n데이터셋 통계:")
print(f"  프롬프트 수: {n_pairs}")
print(f"  프롬프트 차원: {prompt_dim}")
print(f"  응답 차원: {response_dim}")

print(f"\\n품질 점수 (코사인 유사도):")
print(f"  선호 응답:   평균={chosen_quality.mean():.4f}, 표준편차={chosen_quality.std():.4f}")
print(f"  비선호 응답: 평균={rejected_quality.mean():.4f}, 표준편차={rejected_quality.std():.4f}")
print(f"  평균 차이:   {(chosen_quality - rejected_quality).mean():.4f}")

# 올바른 순서 비율 (chosen > rejected)
correct_order = (chosen_quality > rejected_quality).mean()
print(f"\\n순서 정확도: {correct_order:.1%} (chosen > rejected)")
print(f"  → {correct_order:.1%}의 쌍에서 선호 응답이 실제로 더 높은 품질")

print(f"\\n[해설]")
print(f"  노이즈가 작은 chosen이 이상적 응답에 더 가깝습니다.")
print(f"  하지만 {(1-correct_order)*100:.1f}%의 쌍은 역전되어 있어 노이지한 라벨입니다.")
print(f"  실제 RLHF에서도 인간 평가자 간 불일치율이 20-30%에 달합니다.")"""))

# ── Cell 8: Q4 problem ──────────────────────────────────────────────
cells.append(md(r"""\
## Q4: Reward Model 학습 및 평가 <a name='q4'></a>

### 문제

Q3에서 생성한 데이터로 Reward Model을 학습하고 평가하세요:

1. MLP 기반 Reward Model 구축 (입력: prompt+response → 스칼라 보상)
2. Bradley-Terry 손실로 30 에폭 학습
3. 선호 정확도(Pairwise Accuracy)와 Calibration을 측정

$$\text{Pairwise Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[r_\theta(x_i, y_w^i) > r_\theta(x_i, y_l^i)\right]$$

**여러분의 예측:** 학습 후 선호 정확도는 `?%` 입니다."""))

# ── Cell 9: Q4 solution ─────────────────────────────────────────────
cells.append(code("""\
# ── Q4 풀이 ──────────────────────────────────────────────────────
print("=" * 45)
print("Q4 풀이: Reward Model 학습 및 평가")
print("=" * 45)

# 데이터 준비: (prompt, response) → concat
X_chosen = np.concatenate([prompts, chosen_responses], axis=1)
X_rejected = np.concatenate([prompts, rejected_responses], axis=1)

# 학습/검증 분리
n_train = int(n_pairs * 0.8)
train_chosen, val_chosen = X_chosen[:n_train], X_chosen[n_train:]
train_rejected, val_rejected = X_rejected[:n_train], X_rejected[n_train:]

# Reward Model 구축
reward_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu',
                          input_shape=(prompt_dim + response_dim,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 학습 루프
n_epochs = 30
batch_size = 32
train_losses = []
train_accs = []
val_accs = []

for epoch in range(n_epochs):
    epoch_losses = []
    epoch_correct = 0

    indices = np.random.permutation(n_train)
    for start in range(0, n_train, batch_size):
        batch_idx = indices[start:start + batch_size]
        with tf.GradientTape() as tape:
            r_w = reward_model(train_chosen[batch_idx], training=True)[:, 0]
            r_l = reward_model(train_rejected[batch_idx], training=True)[:, 0]
            diff = r_w - r_l
            loss = -tf.reduce_mean(tf.math.log_sigmoid(diff))

        grads = tape.gradient(loss, reward_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, reward_model.trainable_variables))
        epoch_losses.append(loss.numpy())
        epoch_correct += tf.reduce_sum(tf.cast(diff > 0, tf.float32)).numpy()

    avg_loss = np.mean(epoch_losses)
    train_acc = epoch_correct / n_train
    train_losses.append(avg_loss)
    train_accs.append(train_acc)

    # 검증 정확도
    r_w_val = reward_model(val_chosen, training=False)[:, 0]
    r_l_val = reward_model(val_rejected, training=False)[:, 0]
    val_acc = tf.reduce_mean(tf.cast((r_w_val - r_l_val) > 0, tf.float32)).numpy()
    val_accs.append(val_acc)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

# Calibration 평가
r_w_all = reward_model(X_chosen).numpy().flatten()
r_l_all = reward_model(X_rejected).numpy().flatten()
pred_probs = 1 / (1 + np.exp(-(r_w_all - r_l_all)))

# 확률 구간별 실제 정확도
n_bins = 5
bin_edges = np.linspace(0, 1, n_bins + 1)
calibration_data = []
for i in range(n_bins):
    mask = (pred_probs >= bin_edges[i]) & (pred_probs < bin_edges[i + 1])
    if mask.sum() > 0:
        actual_acc = (r_w_all[mask] > r_l_all[mask]).mean()
        predicted_prob = pred_probs[mask].mean()
        calibration_data.append((predicted_prob, actual_acc, mask.sum()))

print(f"\\n최종 결과:")
print(f"  학습 정확도: {train_accs[-1]:.4f}")
print(f"  검증 정확도: {val_accs[-1]:.4f}")

print(f"\\nCalibration 평가:")
print(f"{'예측 확률':>12} | {'실제 정확도':>12} | {'샘플 수':>8}")
print("-" * 38)
for pred, actual, n in calibration_data:
    print(f"{pred:>12.3f} | {actual:>12.3f} | {n:>8d}")

print(f"\\n[해설]")
print(f"  Reward Model이 선호 응답에 더 높은 보상을 줄수록 정확도가 높습니다.")
print(f"  Calibration은 '예측 확률과 실제 정확도가 얼마나 일치하는가'를 측정합니다.")
print(f"  이상적으로는 예측 80% → 실제 80%가 되어야 합니다.")"""))

# ── Cell 10: Bonus problem ──────────────────────────────────────────
cells.append(md(r"""\
## 종합 도전: Full RM Training Pipeline <a name='bonus'></a>

### 문제

Q1-Q4의 모든 구성 요소를 통합하여 완전한 Reward Model 학습 파이프라인을 구현하세요:

1. **데이터 생성**: 500개의 선호 쌍 + 학습/검증/테스트 분리 (70/15/15)
2. **모델 학습**: Bradley-Terry 손실 + 학습률 스케줄링
3. **평가**: Pairwise Accuracy + Calibration 플롯
4. **분석**: 보상 분포 히스토그램 + 학습 곡선

이 파이프라인이 실제 RLHF에서 어떻게 사용되는지 설명하세요."""))

# ── Cell 11: Bonus solution ──────────────────────────────────────────
cells.append(code("""\
# ── 종합 도전 풀이: Full RM Training Pipeline ────────────────────
print("=" * 55)
print("종합 도전: Full Reward Model Training Pipeline")
print("=" * 55)

np.random.seed(42)

# 1. 데이터 생성
n_total = 500
prompt_d = 10
response_d = 10
prompts_full = np.random.randn(n_total, prompt_d).astype(np.float32)
W_transform = np.random.randn(prompt_d, response_d).astype(np.float32) * 0.3

chosen_full = prompts_full @ W_transform + np.random.randn(n_total, response_d).astype(np.float32) * 0.1
rejected_full = prompts_full @ W_transform + np.random.randn(n_total, response_d).astype(np.float32) * 0.8 + 0.3

X_c = np.concatenate([prompts_full, chosen_full], axis=1)
X_r = np.concatenate([prompts_full, rejected_full], axis=1)

# 학습/검증/테스트 분리 (70/15/15)
n1 = int(n_total * 0.7)
n2 = int(n_total * 0.85)
tr_c, va_c, te_c = X_c[:n1], X_c[n1:n2], X_c[n2:]
tr_r, va_r, te_r = X_r[:n1], X_r[n1:n2], X_r[n2:]
print(f"\\n데이터 분할: 학습={n1}, 검증={n2-n1}, 테스트={n_total-n2}")

# 2. 모델 + 학습률 스케줄링
rm_full = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(prompt_d + response_d,)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.002, decay_steps=50 * (n1 // 32 + 1))
opt_full = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 학습
epochs_full = 50
batch_sz = 32
full_train_losses, full_val_losses = [], []
full_train_accs, full_val_accs = [], []

for epoch in range(epochs_full):
    idx = np.random.permutation(n1)
    e_losses = []
    e_correct = 0
    for s in range(0, n1, batch_sz):
        bi = idx[s:s+batch_sz]
        with tf.GradientTape() as tape:
            rw = rm_full(tr_c[bi], training=True)[:, 0]
            rl = rm_full(tr_r[bi], training=True)[:, 0]
            loss = -tf.reduce_mean(tf.math.log_sigmoid(rw - rl))
        grads = tape.gradient(loss, rm_full.trainable_variables)
        opt_full.apply_gradients(zip(grads, rm_full.trainable_variables))
        e_losses.append(loss.numpy())
        e_correct += tf.reduce_sum(tf.cast((rw-rl) > 0, tf.float32)).numpy()

    # 검증
    vw = rm_full(va_c, training=False)[:, 0]
    vl = rm_full(va_r, training=False)[:, 0]
    v_loss = -tf.reduce_mean(tf.math.log_sigmoid(vw - vl)).numpy()
    v_acc = tf.reduce_mean(tf.cast((vw-vl) > 0, tf.float32)).numpy()

    full_train_losses.append(np.mean(e_losses))
    full_val_losses.append(v_loss)
    full_train_accs.append(e_correct / n1)
    full_val_accs.append(v_acc)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}: TrLoss={np.mean(e_losses):.4f}, "
              f"VaLoss={v_loss:.4f}, TrAcc={e_correct/n1:.4f}, VaAcc={v_acc:.4f}")

# 3. 테스트 평가
tw = rm_full(te_c, training=False).numpy().flatten()
tl = rm_full(te_r, training=False).numpy().flatten()
test_acc = (tw > tl).mean()
print(f"\\n테스트 셋 Pairwise Accuracy: {test_acc:.4f}")

# 4. 시각화
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (1) 학습 곡선
ax1 = axes[0, 0]
ax1.plot(full_train_losses, 'b-', lw=2, label='Train Loss')
ax1.plot(full_val_losses, 'r--', lw=2, label='Val Loss')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Reward Model 학습 곡선', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# (2) 정확도 곡선
ax2 = axes[0, 1]
ax2.plot(full_train_accs, 'b-', lw=2, label='Train Acc')
ax2.plot(full_val_accs, 'r--', lw=2, label='Val Acc')
ax2.axhline(y=0.5, color='gray', ls=':', lw=1.5, label='Random')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Pairwise Accuracy', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.4, 1.05)

# (3) 보상 분포
ax3 = axes[1, 0]
ax3.hist(tw, bins=25, alpha=0.6, color='green', label='Chosen', density=True)
ax3.hist(tl, bins=25, alpha=0.6, color='red', label='Rejected', density=True)
ax3.axvline(x=tw.mean(), color='darkgreen', ls='--', lw=2)
ax3.axvline(x=tl.mean(), color='darkred', ls='--', lw=2)
ax3.set_xlabel('보상 점수', fontsize=11)
ax3.set_ylabel('밀도', fontsize=11)
ax3.set_title('테스트 셋 보상 분포', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# (4) Calibration 플롯
ax4 = axes[1, 1]
pred_p = 1 / (1 + np.exp(-(tw - tl)))
n_cal_bins = 10
cal_predicted, cal_actual = [], []
for i in range(n_cal_bins):
    lo = i / n_cal_bins
    hi = (i + 1) / n_cal_bins
    mask = (pred_p >= lo) & (pred_p < hi)
    if mask.sum() > 2:
        cal_predicted.append(pred_p[mask].mean())
        cal_actual.append((tw[mask] > tl[mask]).mean())

ax4.plot([0, 1], [0, 1], 'k--', lw=1.5, label='완벽한 Calibration')
ax4.plot(cal_predicted, cal_actual, 'bo-', lw=2, ms=8, label='RM Calibration')
ax4.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax4.set_xlabel('예측 확률', fontsize=11)
ax4.set_ylabel('실제 정확도', fontsize=11)
ax4.set_title('Calibration Plot', fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('chapter15_alignment_rlhf/practice/rm_training_pipeline.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: chapter15_alignment_rlhf/practice/rm_training_pipeline.png")

print(f"\\n파이프라인 최종 요약:")
print(f"{'메트릭':<20} | {'값':>10}")
print("-" * 35)
print(f"{'학습 최종 Loss':<20} | {full_train_losses[-1]:>10.4f}")
print(f"{'검증 최종 Loss':<20} | {full_val_losses[-1]:>10.4f}")
print(f"{'학습 Accuracy':<20} | {full_train_accs[-1]:>10.4f}")
print(f"{'검증 Accuracy':<20} | {full_val_accs[-1]:>10.4f}")
print(f"{'테스트 Accuracy':<20} | {test_acc:>10.4f}")
print(f"{'보상 마진':<20} | {(tw-tl).mean():>10.4f}")

print(f"\\n[해설]")
print(f"  이 Reward Model은 RLHF의 2단계에서 사용됩니다:")
print(f"  1. 인간 평가자가 만든 선호 쌍으로 RM을 학습")
print(f"  2. 학습된 RM이 새로운 응답에 점수를 매김")
print(f"  3. PPO가 RM 점수를 최대화하도록 정책을 학습")
print(f"  Calibration이 좋을수록 PPO의 학습 신호가 신뢰할 수 있습니다.")"""))

# ── Cell 12: Closing ─────────────────────────────────────────────────
cells.append(md("""\
## 학습 정리

이 실습에서 다룬 핵심 개념:

| 개념 | 구현 | 핵심 포인트 |
|------|------|-----------|
| Bradley-Terry | `sigma(r_w - r_l)` | 보상 '차이'가 확률 결정 |
| RM Loss | `-log_sigmoid(Δr)` | 선호 보상 > 비선호 보상으로 학습 |
| 선호 데이터 | 노이즈 차이로 품질 구분 | 실제로는 20-30% 라벨 노이즈 존재 |
| Calibration | 예측 확률 vs 실제 정확도 | PPO 학습 신호의 신뢰성 결정 |

**다음 실습 →** `ex02_dpo_fine_tuning_lora.ipynb`에서 DPO + LoRA를 사용한 직접 선호 최적화를 구현합니다."""))

create_notebook(cells, 'chapter15_alignment_rlhf/practice/ex01_train_reward_model.ipynb')
