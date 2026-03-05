# CLAUDE.md — TensorFlow 강의 자료 생성 에이전트

## 역할 및 목표

너는 **최정상 ML 엔지니어이자 최고의 교육자**다.  
`SYLLABUS.md`의 **Chapter 12 ~ Chapter 17** 강의 자료를 `.ipynb` 파일로 직접 생성하는 것이 임무다.

---

## 프로젝트 구조

```
{프로젝트 루트}/
├── SYLLABUS.md                  ← 반드시 먼저 전체 정독
├── CLAUDE.md                    ← 현재 파일
├── chapter01_basics/            ← 참고용 기존 강의 (읽기 전용)
│   ├── 01_tensorflow_intro.ipynb
│   ├── 02_tensors_operations.ipynb
│   ├── 03_automatic_differentiation.ipynb
│   └── practice/
│       └── ex01_tensor_quiz.ipynb
│   ...                          ← Ch02~Ch11 동일 구조 (참고용)
├── chapter12_modern_llms/       ← 생성 대상
├── chapter13_genai_diffusion/   ← 생성 대상
├── chapter14_extreme_inference/ ← 생성 대상
├── chapter15_alignment_rlhf/    ← 생성 대상
├── chapter16_sparse_attention/  ← 생성 대상
└── chapter17_diffusion_transformers/ ← 생성 대상
```

---

## 시작 전 필수 절차 (반드시 순서대로)

### Step 1. SYLLABUS 정독
```
Read: SYLLABUS.md (전체)
```
챕터별 학습 목표, 수학적 기초, 강의 파일 목록, 키워드를 정확히 파악한다.

### Step 2. 기존 강의 파일 스타일 파악
아래 파일들의 스타일은 이미 분석되어 이 CLAUDE.md에 내장되어 있다.  
단, 특정 패턴이 불명확할 때만 직접 열어서 재확인한다.

```
(선택적 재확인용)
Read: chapter01_basics/01_tensorflow_intro.ipynb
Read: chapter01_basics/03_automatic_differentiation.ipynb
Read: chapter11_custom_kernels/01_gpu_architecture_basics.ipynb
Read: chapter11_custom_kernels/03_triton_kernel_programming.ipynb
```

### Step 3. 작업 계획 수립 및 보고
작업 시작 전 다음을 출력한다:
```
생성 예정 파일 목록:
  chapter12_modern_llms/
    01_xxx.ipynb   (예상 셀 수: ~N)
    02_xxx.ipynb   ...
  ...
총 파일 수: X개
```

---

## 노트북 스타일 명세 (실제 파일에서 추출)

아래는 Ch01~Ch11 파일에서 직접 확인한 패턴이다. **모든 파일은 이 명세를 정확히 따라야 한다.**

---

### 1. 파일 헤더 (첫 번째 마크다운 셀 필수 구조)

```markdown
# Chapter XX: [제목]

## 학습 목표
- 목표 1
- 목표 2 (보통 4~6개, `~이해한다` / `~구현한다` 형식)

## 목차
1. [수학적 기초: 제목](#1.-수학적-기초)
2. [섹션명](#2.-섹션명)
...
N. [정리](#N.-정리)
```

---

### 2. 수학 섹션 (마크다운 셀 — Ch11 고급 스타일 기준)

```markdown
## 1. 수학적 기초 <a name='1.-수학적-기초'></a>

### [개념명]

$$수식$$

- $변수$: 의미 설명

**요약 표:**

| 구분 | 수식 | 설명 |
|------|------|------|
| 이름 | $수식$ | 의미 |

---

### 🐣 초등학생을 위한 [주제] 친절 설명!

#### 🔢 [개념]이 뭔가요?

> 💡 **비유**: [구체적 비유]

[3~4줄 쉬운 설명]

---

### 📝 연습 문제

#### 문제 1: [제목]

[문제 설명, 수식 포함]

<details>
<summary>💡 풀이 확인</summary>

[수식 + 수치 풀이 + 결론]
</details>

#### 문제 2: ...
```

**규칙:**
- 섹션 번호 시작: `## 1.`, `## 2.` ...
- 앵커 태그 필수: `<a name='1.-수학적-기초'></a>`
- 수식 → 변수 설명 → 직관적 해석 → 비유 순서
- 🐣 친절 설명과 📝 연습 문제는 **수학 섹션 직후** 배치 (생략 금지)
- 구분선(`---`)으로 각 소섹션 분리

---

### 3. 코드 셀 스타일

**기본 패턴:**
```python
# ── 섹션 제목 ──────────────────────────────────────────────────
# 주석: 이 셀이 하는 일 한 줄 설명

import_or_setup = ...

# 핵심 계산
result = ...

# 항상 print로 결과 출력 (silent 셀 금지)
print(f"결과: {result}")
print(f"shape: {tensor.shape}")
print(f"값: {value.numpy():.4f}")
```

**구분선 패턴 (Ch11 스타일):**
```python
# ---------------------------------------------------
# 섹션 제목
# ---------------------------------------------------
```

**규칙:**
- 셀 첫 줄에 `# ──` 또는 `# ---` 구분선
- 각 의미 단위마다 빈 줄 + 한국어 주석
- **모든 중간 결과를 print로 출력 필수**
- 한 셀 = 하나의 개념 (너무 길면 분할)
- 변수명: 영어, 주석/출력: 한국어

---

### 4. 셀 흐름 패턴 (Ch11 기준)

```
[MD] ## 1. 수학적 기초  ← 수식 + 표
[MD] 🐣 친절 설명      ← 비유 + 표
[MD] 📝 연습 문제      ← 문제 + <details> 풀이
[CODE] import numpy, matplotlib, tensorflow
[MD] ## 2. 섹션명
[CODE] 개념 구현 + 시각화 (핵심 코드)
[CODE] 확장 실험 (이전 결과 기반)
[MD] ## 3. 다음 섹션
[CODE] ...
[MD] ## N. 정리        ← 핵심 수식 표 + 다음 챕터 예고
```

**규칙:**
- 수학 섹션: 마크다운 3연속 허용 (수식/친절설명/연습문제)
- 임포트 셀: 항상 수학 섹션 이후, 본문 코드 이전
- 시각화 코드: 별도 셀로 분리

---

### 5. 임포트 셀 표준

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 헤드리스 환경 필수
import matplotlib.pyplot as plt
import tensorflow as tf
import time  # 필요 시

np.random.seed(42)
print(f"TensorFlow 버전: {tf.__version__}")
```

**규칙:**
- `matplotlib.use('Agg')` 반드시 `import matplotlib.pyplot` 전에 위치
- `np.random.seed(42)` 재현성 확보
- TF 버전 print 필수

---

### 6. 시각화 셀 표준

```python
# ---------------------------------------------------
# [시각화 제목]
# ---------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
ax1.plot(x, y, 'b-o', lw=2.5, ms=8, label='레이블')
ax1.axhline(y=기준값, color='gray', ls='--', lw=1.5)
ax1.fill_between(x, y1, y2, alpha=0.1, color='red')
ax1.set_xlabel('x축', fontsize=11)
ax1.set_ylabel('y축', fontsize=11)
ax1.set_title('제목', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('저장경로/파일명.png', dpi=100, bbox_inches='tight')
plt.close()
print("그래프 저장됨: 경로/파일명.png")
```

**규칙:**
- `plt.show()` 절대 금지 → `plt.savefig()` + `plt.close()` 사용
- `tight_layout()` 필수
- fontsize 명시: 축 라벨 11, legend 9
- `fontweight='bold'` 타이틀 적용
- `grid(True, alpha=0.3)` 항상 포함
- figsize: 단일 그래프 `(10, 5)`, 2열 `(13, 5)`, 3열 `(15, 4~5)`

---

### 7. 비교 표/분석 코드 (Ch11 스타일)

```python
# 수치 비교 표 출력 패턴
print(f"{'항목':<20} | {'값1':>12} | {'값2':>12} | {'비율':>8}")
print("-" * 58)
for item, v1, v2 in data:
    ratio = v1 / v2
    print(f"{item:<20} | {v1:>12.2f} | {v2:>12.2f} | {ratio:>7.1f}x")
```

---

### 8. 정리 섹션 (마지막 마크다운 셀 필수)

```markdown
## N. 정리 <a name='N.-정리'></a>

### 핵심 개념 요약

| 개념 | 설명 | 중요도 |
|------|------|--------|
| ... | ... | ⭐⭐⭐ |

### 핵심 수식

$$수식1$$

$$수식2$$

### 다음 챕터 예고
**Chapter XX: [제목]** — [한두 줄 요약]
```

---

### 9. practice/ 파일 구조

```markdown
# 실습 퀴즈: [주제명]

## 사용 방법
- 각 문제 셀을 읽고, **직접 답을 예측한 후** 풀이 셀을 실행하세요
- 코드 실행 전에 종이에 계산해보는 것을 권장합니다

## 목차
- [Q1: 제목](#q1)
- [Q2: 제목](#q2)
...
- [종합 도전: 미니 구현](#bonus)
```

각 문제 셀:
```markdown
## Q1: [제목] <a name='q1'></a>

### 문제

[수식 + 코드 힌트]

**여러분의 예측:** [예측 항목]은 `?` 입니다.
```

각 풀이 코드 셀:
```python
# ── Q1 풀이 ──────────────────────────────────────────────────
# 코드 + 상세 해설 print
print("=" * 45)
print("Q1 풀이: [제목]")
print("=" * 45)
# ... 풀이 코드 ...
print("[해설]")
print("  결론 설명")
```

**규칙:**
- Q1~Q4 + 종합 도전 구성 (Ch01 practice 파일 패턴 참고)
- 각 문제: 문제 마크다운 셀 + 풀이 코드 셀 쌍
- 종합 도전: 해당 챕터 핵심 개념을 미니 구현으로 검증

---

## 품질 기준

### 1. LaTeX 수식 품질 (필수)

**올바른 예:**
```
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

변수 설명:
- $Q \in \mathbb{R}^{S \times d_k}$: Query 행렬
- $d_k$: Key 차원 (스케일링 인자)
```

**금지:**
- 수식만 있고 변수 설명 없음
- 수식 없이 텍스트로만 이론 설명
- 잘못된 LaTeX (`$\frac a b$` → `$\frac{a}{b}$`)

### 2. 최신 정보 웹 검색 규칙

**반드시 검색 후 작성:**
- 모델 파라미터 수 (Llama 3 8B KV head 수 등 세부 수치)
- arxiv 논문 발표 날짜 및 저자
- 벤치마크 수치 (MMLU, HumanEval 리더보드)
- 2024년 이후 발표 모델/기법의 세부 사양

**검색 불필요:**
- 기본 수학 수식 (Attention, Transformer 구조)
- 잘 알려진 개념 (ReLU, LayerNorm, Dropout)
- TensorFlow/Python 코드 패턴

**수치 표기 시:**
- 검증된 수치: 그대로 사용
- 불확실한 수치: `[주의: YYYY-MM 기준, 확인 필요]` 명시

### 3. 코드 실행 가능성

- `# TODO` / `pass` placeholder 절대 금지
- 모든 코드 셀은 실제 실행 가능
- print 출력이 없는 코드 셀 금지
- 실행 오류 가능성이 있는 코드는 try-except 처리

---

## 금지 사항

| 금지 항목 | 이유 |
|-----------|------|
| PyTorch 코드 | TF 전용 강의 |
| `plt.show()` | Apple Silicon 헤드리스 환경 오류 |
| `# TODO` / `pass` | 미완성 파일 배포 금지 |
| 수식 없는 이론 마크다운 셀 | 수학적 엄밀성 필수 |
| 배치 생성 (여러 챕터 동시) | 챕터별 순차 작업 필수 |
| 미검증 최신 수치 그대로 사용 | 오정보 방지 |
| silent 코드 셀 (출력 없음) | 학습자 결과 확인 불가 |
| `WidthType.PERCENTAGE` 등 환경 의존 코드 | 호환성 문제 |

---

## 챕터별 핵심 요구사항

### Ch12: 최신 LLM 아키텍처 (chapter12_modern_llms/)

**필수 포함 수식/시각화:**
- GQA 발전 과정: MHA → MQA → GQA 비교 시각화
  $$\text{KV head 수} \ll \text{Q head 수} \Rightarrow \text{메모리} \frac{n_{kv}}{n_q}\text{배 절약}$$
- KV Cache 메모리 계산 (Llama 3 8B 실제 파라미터 사용):
  $$M_{KV} = 2 \times n_{kv} \times d_{head} \times S \times B \times \text{bytes}$$
- MoE Router: top-k 게이팅 단계별 출력

**참고 자료 (검색 확인):**
- DeepSeek-V3: arxiv 2412.19437
- Llama 3: Meta AI 공식 블로그

---

### Ch13: 생성 AI — 확산 모델 (chapter13_genai_diffusion/)

**필수 포함 수식/시각화:**
- Forward Process:
  $$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I\right)$$
- Reparameterization:
  $$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar\alpha_t}\,x_0,\, (1-\bar\alpha_t)I\right), \quad \bar\alpha_t = \prod_{s=1}^t (1-\beta_s)$$
- DDIM vs DDPM 샘플링 단계별 비교 (같은 노이즈 → 다른 경로)

**참고 자료:**
- DDPM: arxiv 2006.11239
- DDIM: arxiv 2010.02502
- Score SDE: arxiv 2011.13456

---

### Ch14: 극한 추론 최적화 (chapter14_extreme_inference/)

**필수 포함 수식/시각화:**
- FlashAttention IO complexity:
  $$\text{HBM reads/writes} = O\!\left(\frac{N^2 d}{M}\right), \quad M = \text{SRAM 크기}$$
- PagedAttention: ASCII 다이어그램으로 블록 할당 시각화
- Quantization 오차:
  $$\Delta = \frac{x_{max} - x_{min}}{2^b - 1}, \quad \text{SNR} \propto 2^{2b}$$

**참고 자료:**
- FlashAttention-3 논문
- vLLM 공식 문서
- AWQ: arxiv 2306.00978

---

### Ch15: 정렬 기법 RLHF/DPO (chapter15_alignment_rlhf/)

**필수 포함 수식/시각화:**
- PPO Clip 목적함수 3구간 시각화:
  $$L^{CLIP}(\theta) = \mathbb{E}\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$
- DPO vs RLHF 수학적 등가성:
  $$\mathcal{L}_{DPO}(\pi_\theta) = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$
- Constitutional AI: 한국어 헌법 프롬프트 예시 코드 포함

**참고 자료:**
- InstructGPT: OpenAI 2022
- DPO: arxiv 2305.18290
- Constitutional AI: Anthropic 2022

---

### Ch16: 희소 Attention 및 최신 기법 (chapter16_sparse_attention/)

**필수 포함 수식/시각화:**
- MLA (Multi-head Latent Attention) KV 압축 흐름도:
  $$c_{KV} = W^{DKV} h_t, \quad [k_t^C; v_t^C] = W^{UKV} c_{KV}$$
- GLA vs MHA vs MLA 메모리 비교 그래프 (시퀀스 길이 축)
- FP8/FP16/BF16 수치 범위 비교 표

**참고 자료:**
- DeepSeek-V3: arxiv 2412.19437
- GLA: arxiv 2312.06635

---

### Ch17: Diffusion Transformers (chapter17_diffusion_transformers/)

**필수 포함 수식/시각화:**
- DiT 패치 임베딩: 2D (이미지) vs 3D (비디오) 비교
- adaLN-Zero:
  $$y = x + \alpha_1 \cdot \text{Attn}(\text{LN}(x)), \quad \text{LN: } (\gamma, \beta) \leftarrow \text{MLP}(c)$$
- Flow Matching vs DDPM 경로 시각화 (직선 경로 vs 곡선 경로):
  $$\frac{dx}{dt} = v_\theta(x, t), \quad v^{OT} = x_1 - x_0 \text{ (직선)}$$

**참고 자료:**
- DiT: arxiv 2212.09748
- HunyuanVideo: arxiv 2412.17601
- Flow Matching: arxiv 2210.02747

---

## 작업 완료 보고 형식

각 챕터 완성 후:
```
✅ chapter12_modern_llms/ 완료
  파일 목록:
    01_gqa_rope_swiglu.ipynb     (셀 수: 32)
    02_moe_architecture.ipynb    (셀 수: 28)
    practice/ex01_attention_quiz.ipynb (셀 수: 20)
  처리된 수식: 15개
  웹 검색 사용: 3회
    - Llama 3 8B KV head 수 (Meta 블로그)
    - DeepSeek-V3 MoE 구성 (arxiv)
    - 최신 MMLU 벤치마크 수치
  
  → 다음: chapter13_genai_diffusion/ 시작합니다
```

전체 완료 후:
```
🎉 모든 챕터 생성 완료 (Ch12~Ch17)
총 파일 수: XX개
총 셀 수: 약 XXX셀
수식 처리: XX개
웹 검색 사용: XX회
```
