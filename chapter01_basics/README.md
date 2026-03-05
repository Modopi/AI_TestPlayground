# Chapter 01: TensorFlow 기초 및 자동 미분 (Automatic Differentiation)

이 디렉토리는 TensorFlow 2.x의 핵심 자료구조인 Tensor의 기본 사용법과 딥러닝 학습의 핵심인 **자동 미분**의 원리를 익히는 공간입니다.

## 📂 파일 구성

- `01_tensorflow_intro.ipynb`: TensorFlow 2.x의 철학과 Eager Execution 동작, 디바이스(GPU 등) 확인 방법.
- `02_tensors_operations.ipynb`: Tensor의 속성(rank, shape, dtype), 인덱싱 및 수학적 연산 방식, NumPy와의 연동 방법.
- `03_automatic_differentiation.ipynb`: 딥러닝 모델이 스스로 학습 방향을 찾는 원리인 **자동 미분(Automatic Differentiation)**과 **GradientTape**의 사용법 및 선형 회귀의 기본 구현.
- `practice/ex01_tensor_quiz.ipynb`: 위 개념들을 점검하는 실습 문제들.

---

## 🧠 핵심 개념: 딥러닝에서의 🌟자동 미분(Automatic Differentiation)🌟

이 챕터의 핵심 목표는 딥러닝 모델이 "어떻게 스스로 답을 찾아가며 학습하는지" 그 원리를 이해하는 것입니다. 이 과정에서 가장 중요한 역할을 하는 나침반이 바로 **미분**입니다.

### 1. 왜 미분(기울기)이 필요할까요?

딥러닝 모델의 목표는 '현재 모델이 예측한 값과 실제 정답 사이의 차이(오차, Loss)'를 최대한 줄이는 것입니다. 오차라는 거대한 산에서 눈을 가린 채 가장 낮은 골짜기를 찾아 내려가야 한다고 상상해 보세요. 이때 우리가 현재 서 있는 지점에서 "어느 방향으로, 얼마나 가파르게 내려가야 하는지" 알려주는 나침반이 바로 **미분(Gradient)**입니다.

### 2. 경사하강법 (Gradient Descent)

나침반(미분)이 알려주는 가장 가파른 내리막길 방향으로 가중치(파라미터)를 조금씩, 반복적으로 이동시켜 오차가 가장 작은 최적의 상태(골짜기의 바닥)를 시뮬레이션하며 찾아가는 알고리즘입니다.

### 3. TensorFlow의 핵심 도구: `tf.GradientTape()`

수많은 수학 공식이 들어있는 복잡한 딥러닝 연산을 사람이 일일이 손으로 미분할 수는 없습니다. 그래서 텐서플로우는 **`tf.GradientTape`**이라는 마법의 테이프(도구)를 제공합니다.

```python
with tf.GradientTape() as tape:  # 🎬 여기서부터 녹화 시작!
    # 이 안에서 일어나는 모든 계산 과정을 테이프에 꼼꼼히 기록합니다.
    y = x ** 2

# ⏪ 녹화된 테이프를 역재생하듯(역전파) 되감으면서
# 결과값 y가 변동할 때 원인값 x가 미치는 영향(미분값=기울기)을 자동으로 순식간에 계산해 냅니다!
dy_dx = tape.gradient(y, x)
```

`tf.GradientTape`을 사용하면 모델과 가중치가 복잡해져도 텐서플로우 엔진이 기울기(미분값)를 빠르고 정확하게 대신 구해줍니다! 이것이 최신 AI들이 효율적으로 학습할 수 있는 근본적인 베이스입니다.
