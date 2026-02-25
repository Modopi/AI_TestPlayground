# TensorFlow 학습 커리큘럼

> **환경**: Python 3.11 + TensorFlow 2.x (Apple Silicon: tensorflow-macos + tensorflow-metal)
> **형식**: Jupyter Notebook (.ipynb) — 이론 + 수학 + 코드 + 출력 통합
> **수학 원칙**: 각 챕터 해당 수준의 연산 단위 수식 포함. 증명은 생략하고 수식의 의미·흐름에 집중.

---

## 환경 설정

### Conda 환경 생성 (Apple Silicon Mac)

```bash
# 환경 생성
conda env create -f environment.yml

# 환경 활성화
conda activate tf_study

# VSCode Jupyter 커널 등록
python -m ipykernel install --user --name tf_study --display-name "Python (tf_study)"
```

### 설치 확인

```bash
python main.py
# 출력 예시: tensorflow_version : 2.x.x
```

VSCode에서 노트북 열기 → 우측 상단 커널 선택 → **Python (tf_study)** 선택

---

## 전체 커리큘럼 지도

```
chapter01_basics/              ← 시작점: Tensor와 자동미분
chapter02_keras_api/           ← 모델 구성 3가지 방식
chapter03_training_mechanics/  ← 학습 원리와 제어
chapter04_data_pipeline/       ← 데이터 효율적 처리
chapter05_computer_vision/     ← CNN과 이미지 학습
chapter06_nlp/                 ← 텍스트와 시계열 학습
chapter07_advanced_architectures/ ← Transformer, VAE, GAN
chapter08_production/          ← 모델 배포와 최적화
projects/                      ← 종합 실전 프로젝트
```

---

## Chapter 01 — TensorFlow 기초

**디렉토리**: `chapter01_basics/`

**학습 목표**
- TensorFlow 2.x의 핵심 자료구조 Tensor를 이해한다
- 자동 미분(Automatic Differentiation)의 원리를 이해하고 활용한다

**수학적 기초**
| 개념 | 수식 |
|------|------|
| 행렬 곱 | $C_{ij} = \sum_k A_{ik} B_{kj}$ |
| 편미분 | $\frac{\partial f}{\partial x}$ |
| 연쇄 법칙 | $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$ |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_tensorflow_intro.ipynb` | TF 2.x 철학, Eager Execution, 디바이스 확인 |
| `02_tensors_operations.ipynb` | Tensor rank/shape/dtype, 인덱싱, 수학 연산, NumPy 연동 |
| `03_automatic_differentiation.ipynb` | GradientTape, 선형회귀 수동 구현 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_tensor_quiz.ipynb` | Tensor 조작 퀴즈 5문제, GradientTape 기울기 계산 |

**주요 키워드**: `tf.Tensor`, `tf.Variable`, `tf.constant`, `tf.GradientTape`, `@tf.function`

---

## Chapter 02 — Keras API 3종

**디렉토리**: `chapter02_keras_api/`

**학습 목표**
- Sequential, Functional, Subclassing 세 가지 모델 구성 방식과 각각의 사용 시점을 이해한다

**수학적 기초**
| 개념 | 수식 |
|------|------|
| 완전 연결 레이어 | $\mathbf{y} = f(\mathbf{W}\mathbf{x} + \mathbf{b})$ |
| 파라미터 수 (Dense) | $n_{in} \times n_{out} + n_{out}$ |
| ReLU | $f(x) = \max(0, x)$ |
| Sigmoid | $\sigma(x) = \frac{1}{1 + e^{-x}}$ |
| Softmax | $\sigma(\mathbf{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$ |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_sequential_api.ipynb` | Sequential 모델, MNIST 손글씨 분류 |
| `02_functional_api.ipynb` | 다중 입출력, 잔차 연결 (Residual Connection) |
| `03_subclassing_api.ipynb` | `tf.keras.Model` 상속, 커스텀 레이어 |
| `04_layers_and_activations.ipynb` | Dense/Dropout/BN/LN, 활성화 함수 비교 시각화 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_build_your_first_model.ipynb` | Fashion MNIST를 3가지 API로 구현, summary 비교 |

**주요 키워드**: `keras.Sequential`, `keras.Model`, `keras.Input`, `model.summary()`

---

## Chapter 03 — 학습 메카니즘

**디렉토리**: `chapter03_training_mechanics/`

**학습 목표**
- 손실 함수, 옵티마이저, 콜백을 이해하고 커스텀 학습 루프를 직접 구현한다

**수학적 기초**
| 개념 | 수식 |
|------|------|
| MSE | $L = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2$ |
| Binary Cross-Entropy | $L = -[y \log p + (1-y)\log(1-p)]$ |
| Categorical Cross-Entropy | $L = -\sum_i y_i \log p_i$ |
| SGD 업데이트 | $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$ |
| Adam (1차 모멘트) | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ |
| Adam (2차 모멘트) | $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ |
| Adam 업데이트 | $\theta \leftarrow \theta - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ |
| L2 정규화 | $L_{total} = L_{task} + \lambda \sum_j w_j^2$ |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_loss_functions.ipynb` | MSE/MAE/Huber/Cross-Entropy, 커스텀 손실 |
| `02_optimizers.ipynb` | SGD/Adam/AdamW, 학습률 스케줄러, 수렴 비교 시각화 |
| `03_metrics_and_evaluation.ipynb` | Accuracy/Precision/Recall/F1, ROC, Confusion Matrix |
| `04_callbacks.ipynb` | ModelCheckpoint, EarlyStopping, TensorBoard, 커스텀 콜백 |
| `05_custom_training_loop.ipynb` | GradientTape 수동 루프, `@tf.function` 최적화 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_optimizer_comparison.ipynb` | 동일 모델에 SGD/Adam/RMSprop 적용 후 수렴 비교 |
| `practice/ex02_custom_callback.ipynb` | 학습률 시각화 커스텀 콜백 구현 |

**주요 키워드**: `compile`, `fit`, `GradientTape`, `EarlyStopping`, `@tf.function`

---

## Chapter 04 — 데이터 파이프라인

**디렉토리**: `chapter04_data_pipeline/`

**학습 목표**
- `tf.data` API로 대용량 데이터를 효율적으로 처리하는 파이프라인을 구축한다

**수학적 기초**
| 개념 | 수식 |
|------|------|
| Min-Max 정규화 | $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ |
| Z-score 표준화 | $x' = \frac{x - \mu}{\sigma}$ |
| 2D 회전 행렬 | $R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$ |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_tf_data_api.ipynb` | Dataset 생성, map/filter/batch/shuffle/prefetch, AUTOTUNE |
| `02_data_augmentation.ipynb` | RandomFlip/Rotation/Zoom, 증강 결과 시각화 |
| `03_tfrecord_format.ipynb` | TFRecord 생성·읽기, `tf.train.Example` |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_build_data_pipeline.ipynb` | 이미지 디렉토리 → 전처리 → 배치 파이프라인 완성 |

**주요 키워드**: `tf.data.Dataset`, `.map()`, `.batch()`, `.prefetch()`, `tf.data.AUTOTUNE`

---

## Chapter 05 — 컴퓨터 비전

**디렉토리**: `chapter05_computer_vision/`

**학습 목표**
- CNN의 합성곱 연산 원리를 이해하고, 전이학습으로 실제 이미지 분류 문제를 해결한다

**수학적 기초**
| 개념 | 수식 |
|------|------|
| 2D 합성곱 | $(I * K)[i,j] = \sum_m \sum_n I[i+m,\, j+n] \cdot K[m,n]$ |
| 출력 크기 | $O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$ |
| 파라미터 수 (Conv2D) | $K_h \times K_w \times C_{in} \times C_{out} + C_{out}$ |
| FLOPs (Conv2D) | $2 \times K_h K_w \times C_{in} \times C_{out} \times H_{out} \times W_{out}$ |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_cnn_basics.ipynb` | 합성곱 원리·시각화, Padding/Stride, Pooling, 특징 맵 시각화 |
| `02_cnn_architectures.ipynb` | LeNet → VGG → ResNet (Skip Connection), `tf.keras.applications` |
| `03_transfer_learning.ipynb` | Feature Extraction vs Fine-Tuning, 단계별 동결 해제 |
| `04_object_detection_intro.ipynb` | 분류/탐지/분할 개념, IoU/mAP, TF Hub 추론 예시 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_cifar10_classifier.ipynb` | CIFAR-10 CNN 구현, 학습 곡선 시각화 |
| `practice/ex02_transfer_learning_flowers.ipynb` | Flowers 데이터셋, EfficientNetB0 전이학습 |

**주요 키워드**: `Conv2D`, `MaxPooling2D`, `GlobalAveragePooling2D`, `tf.keras.applications`, `layer.trainable`

---

## Chapter 06 — 자연어 처리 (NLP)

**디렉토리**: `chapter06_nlp/`

**학습 목표**
- 텍스트 전처리, 임베딩, RNN 계열 모델을 이해하고 한국어 텍스트 분류를 구현한다

**수학적 기초**
| 개념 | 수식 |
|------|------|
| 코사인 유사도 | $\cos(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$ |
| RNN 상태 업데이트 | $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$ |
| LSTM forget gate | $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ |
| LSTM input gate | $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ |
| LSTM cell update | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ |
| LSTM output | $h_t = o_t \odot \tanh(C_t)$ |
| GRU update gate | $z_t = \sigma(W_z [h_{t-1}, x_t])$ |
| GRU 최종 출력 | $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_text_preprocessing.ipynb` | 토큰화, TextVectorization, 패딩·마스킹, 한국어 형태소 개요 |
| `02_word_embeddings.ipynb` | One-Hot 한계, Embedding 레이어, GloVe/FastText 로드, t-SNE 시각화 |
| `03_rnn_lstm_gru.ipynb` | RNN 기울기 소실, LSTM 게이트, GRU, 양방향 RNN |
| `04_text_classification.ipynb` | 감성 분석: CNN/LSTM/GRU/BiLSTM 비교 |
| `05_sequence_to_sequence.ipynb` | Encoder-Decoder 구조, Attention 소개 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_sentiment_analysis.ipynb` | IMDB 데이터, Embedding+LSTM 감성 분류 |
| `practice/ex02_korean_text_classification.ipynb` | NSMC(네이버 영화 리뷰), KoNLPy 형태소 분석 + 분류 |

**주요 키워드**: `TextVectorization`, `Embedding`, `LSTM`, `GRU`, `Bidirectional`, `return_sequences`

---

## Chapter 07 — 고급 아키텍처

**디렉토리**: `chapter07_advanced_architectures/`

**학습 목표**
- Transformer의 Attention 메카니즘을 이해하고, 생성 모델(VAE, GAN)의 원리를 구현한다

**수학적 기초**
| 개념 | 수식 |
|------|------|
| Scaled Dot-Product Attention | $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ |
| Multi-Head Attention | $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^O$ |
| Positional Encoding (sin) | $PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$ |
| Positional Encoding (cos) | $PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$ |
| VAE ELBO | $\mathcal{L} = \mathbb{E}[\log p(x\|z)] - D_{KL}(q(z\|x) \| p(z))$ |
| GAN 목적 함수 | $\min_G \max_D\; \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$ |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_attention_mechanism.ipynb` | Bahdanau Attention, Self-Attention, Attention Weight 시각화 |
| `02_transformer_basics.ipynb` | Positional Encoding, Encoder/Decoder Block 구현 |
| `03_bert_fine_tuning.ipynb` | BERT 사전학습 목표, HuggingFace + TF 백엔드, 한국어 BERT Fine-Tuning |
| `04_generative_models_vae.ipynb` | Autoencoder → VAE, 재파라미터화 트릭, MNIST 생성 |
| `05_generative_models_gan.ipynb` | DCGAN 구현, Wasserstein Loss, 생성 이미지 시각화 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_transformer_text_classification.ipynb` | Transformer Encoder로 텍스트 분류 |
| `practice/ex02_simple_gan.ipynb` | 간단한 GAN으로 손글씨 이미지 생성 |

**주요 키워드**: `MultiHeadAttention`, `LayerNormalization`, `transformers`, `VAE`, `GAN`

---

## Chapter 08 — 프로덕션 배포

**디렉토리**: `chapter08_production/`

**학습 목표**
- 학습된 모델을 실제 서비스에 배포하기 위한 저장·변환·최적화 워크플로우를 익힌다

**수학적 기초**
| 개념 | 수식 |
|------|------|
| Int8 양자화 | $x_{int} = \text{round}\!\left(\frac{x_{float}}{s}\right) + z$ |
| 역양자화 | $x_{float} \approx s \cdot (x_{int} - z)$ |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_model_saving_formats.ipynb` | `.keras` / SavedModel / HDF5 형식 비교, 가중치 저장 |
| `02_tensorboard.ipynb` | 스칼라/이미지/히스토그램 로깅, 프로파일러 |
| `03_tensorflow_lite.ipynb` | TFLite 변환, Post-Training Quantization, 추론 |
| `04_performance_optimization.ipynb` | Mixed Precision, `@tf.function`, 분산학습 개요 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_export_and_serve.ipynb` | 모델 학습 → 저장 → TFLite 변환 → 추론 전체 파이프라인 |

**주요 키워드**: `tf.saved_model`, `TFLiteConverter`, `tf.summary`, `MirroredStrategy`

---

## 최종 프로젝트

| 프로젝트 | 설명 | 주요 기술 |
|----------|------|-----------|
| `project01_image_classifier/` | 커스텀 이미지 분류기 (EfficientNetB0 전이학습 + TFLite 변환) | CNN, 전이학습, tf.data |
| `project02_sentiment_analysis/` | 한국어 감성 분석 API (KoBERT Fine-Tuning + FastAPI 서빙) | BERT, HuggingFace, SavedModel |
| `project03_korean_nlp/` | 한국어 NLP 파이프라인 (형태소 분석 + 주제 분류) | KoNLPy, Bi-LSTM, TFRecord |

---

## 참고 자료

- [TensorFlow 공식 문서](https://www.tensorflow.org/guide)
- [Keras API 레퍼런스](https://keras.io/api/)
- [TensorFlow 튜토리얼](https://www.tensorflow.org/tutorials)
- [HuggingFace Transformers + TF](https://huggingface.co/docs/transformers/keras_callbacks)
- Goodfellow et al., *Deep Learning* (2016) — 수학적 기초
- Vaswani et al., *Attention Is All You Need* (2017) — Transformer
- Kingma & Welling, *Auto-Encoding Variational Bayes* (2013) — VAE
- Goodfellow et al., *Generative Adversarial Networks* (2014) — GAN
