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
chapter09_distributed_systems/ ← [Advanced] 분산 시스템과 병렬 처리
chapter10_memory_optimization/ ← [Advanced] 대규모 모델 메모리 최적화 (ZeRO/Offload)
chapter11_custom_kernels/      ← [Advanced] 커스텀 GPU 커널 프로그래밍 (CUDA/Triton)
chapter12_modern_llms/         ← [Advanced] 최신 대형 언어 모델 아키텍처 (Llama/MoE)
chapter13_genai_diffusion/     ← [Advanced] 생성 AI 심화 (Diffusion/SDE)
chapter14_extreme_inference/   ← [Advanced] 극단적 추론 최적화 (PagedAttention/AWQ)
chapter15_alignment_rlhf/      ← [Advanced] AI 얼라인먼트와 강화학습 (RLHF/DPO)
chapter16_sparse_attention/    ← [State-of-the-Art] 최신 거대 모델의 효율성 (DeepSeek/Qwen)
chapter17_diffusion_transformers/ ← [State-of-the-Art] 비디오 생성 모델과 DiT (Sora/Hunyuan)
projects/                      ← 종합 실전 프로젝트 (Basic ~ SOTA)
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

## Chapter 09 — [Advanced] 분산 시스템과 병렬 처리 (Distributed Systems)

**디렉토리**: `chapter09_distributed_systems/`

**학습 목표**

- 복수의 GPU 및 멀티 노드 환경에서 모델을 학습시키기 위한 데이터, 텐서, 파이프라인 병렬화의 핵심 동작 원리를 이해하고 구현한다.

**수학적 기초**
| 개념 | 수식 |
|------|------|
| 통신 비용 (All-Reduce) | $T_{comm} = 2(N-1)\frac{K}{N} \cdot t_{byte}$ (Ring All-Reduce) |
| 텐서 병렬화 연산 | $Y = \text{GeLU}(X A_1) A_2$ (열방향 분할 및 행방향 분할 행렬 곱) |
| 파이프라인 버블 (Bubble) | $B = \frac{p-1}{m}$ (p: 스테이지 수, m: 마이크로배치 수) |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_distributed_training_basics.ipynb` | Data Parallelism(DP), DistributedDataParallel(DDP), 통신 토폴로지(All-Reduce/All-Gather) |
| `02_tensor_parallelism.ipynb` | Megatron-LM 방식의 1D 텐서 병렬화, Column/Row 병렬 Linear 레이어 구현 |
| `03_pipeline_parallelism.ipynb` | 마이크로배치 분할, Gpipe 및 1F1B 스케줄링 기법을 통한 파이프라인 병렬화 |
| `04_3d_parallelism.ipynb` | DP + TP + PP를 결합한 3D 병렬 처리 (Megatron-Turing NLG 아키텍처 리뷰) |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_implement_ddp_scratch.ipynb` | NCCL 백엔드를 모사하여 단순 파이썬으로 Ring All-Reduce 로직 구현 |
| `practice/ex02_1d_tensor_parallel_llm.ipynb` | 소형 Transformer 모델의 가중치를 2개의 GPU로 분리하여 Forward Pass 계산 후 검증 |

**주요 키워드**: `All-Reduce`, `Megatron-LM`, `Tensor Parallelism`, `Pipeline Parallelism`, `Microbatch`

---

## Chapter 10 — [Advanced] 대규모 모델 메모리 최적화 (Memory Optimization & ZeRO)

**디렉토리**: `chapter10_memory_optimization/`

**학습 목표**

- 모델의 파라미터, 옵티마이저 상태, 그래디언트가 차지하는 메모리를 수학적으로 예측하고 ZeRO 및 CPU Offloading 기법을 적용한다.

**수학적 기초**
| 개념 | 수식 |
|------|------|
| Adam 최적화기 상태 메모리 | $M_{adam} = 2 \times 4\text{B(fp32)} \times P_{count} = 8 \cdot P_{count}$ 바이트 |
| 총 학습 메모리 복잡도 | $M_{total} = P_{fp16} + G_{fp16} + P_{fp32} + M_{fp32} + V_{fp32}$ |
| ZeRO 파티셔닝 효율 | Stage 3 메모리 사용량 $\approx \frac{M_{total}}{N}$ (N: GPU 개수) |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_memory_profiling.ipynb` | 모델 파라미터, 활성화 연산(Activation), 옵티마이저가 차지하는 VRAM 정확히 계산하기 |
| `02_gradient_checkpointing.ipynb` | Activation Recomputation 원리 (메모리와 연산량의 Trade-off) |
| `03_zero_redundancy_optimizer.ipynb` | ZeRO Stage 1(Optimizer 상태), Stage 2(Gradient), Stage 3(Parameter) 분할 원리 |
| `04_cpu_and_nvme_offloading.ipynb` | DeepSpeed의 Zero-Offload 메커니즘, PCIe 대역폭 한계 극복 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_memory_calculator.ipynb` | 모델 아키텍처 및 배치 사이즈를 입력하면 필요한 VRAM을 출력하는 계산기 만들기 |
| `practice/ex02_deepspeed_zero3_training.ipynb` | 단일 GPU에서 OOM이 발생하는 14B 이상 모델을 ZeRO-3+Offload로 훈련 성공시키기 |

**주요 키워드**: `ZeRO Stage 3`, `Gradient Checkpointing`, `CPU Offload`, `DeepSpeed`, `OOM`

---

## Chapter 11 — [Advanced] 커스텀 GPU 커널 프로그래밍 (CUDA & Triton)

**디렉토리**: `chapter11_custom_kernels/`

**학습 목표**

- 파이썬 프레임워크의 오버헤드를 줄이고 텐서 코어를 최대로 활용하기 위해 사용자 정의 커널을 직접 개발하고 적용한다.

**수학적 기초**
| 개념 | 수식 |
|------|------|
| 메모리 대역폭 한계 | $\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Memory Access (Bytes)}}$ |
| 타일링 (Tiling) 연산 | 블록 행렬 곱: $C_{i,j} = \sum_k A_{i,k} \cdot B_{k,j}$ 를 Shared Memory 위에서 처리 |
| Softmax 안정화 퓨전 | $m = \max(x)$, $y = \frac{\exp(x-m)}{\sum \exp(x-m)}$ 연산을 메모리 왕복 한 번에 처리(Fusion) |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_gpu_architecture_basics.ipynb` | Streaming Multiprocessor(SM), Warp, Global/Shared/레지스터 메모리 계층 이해 |
| `02_cuda_cpp_extensions.ipynb` | C++과 pybind11을 활용하여 PyTorch/TensorFlow에 커스텀 C++(CUDA) 연산 연동 |
| `03_triton_kernel_programming.ipynb` | 파이썬 문법으로 GPU 커널 작성하기: Triton의 블록 포인터와 타일 매핑 |
| `04_kernel_profiling_and_fusion.ipynb` | Nsight Systems 및 프로파일러 활용, 여러 연산을 묶는 Kernel Fusion 전략 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_vector_add_cuda.ipynb` | 벡터 덧셈 연산을 순수 CUDA C++ 커널로 작성하고 파이썬에서 호출 |
| `practice/ex02_triton_fused_softmax_dropout.ipynb` | Triton을 이용하여 Softmax와 Dropout 연산을 하나의 커널로 퓨전하여 속도 측정 |

**주요 키워드**: `CUDA`, `Triton`, `Kernel Fusion`, `Shared Memory`, `Arithmetic Intensity`

---

## Chapter 12 — [Advanced] 최신 대형 언어 모델 아키텍처 (Modern LLMs & MoE)

**디렉토리**: `chapter12_modern_llms/`

**학습 목표**

- Llama-3, GPT-4 등 최신 SOTA 모델에 적용된 Attention 변형과 정규화 기법의 수학적 배경을 이해하고 거대 모델(MoE 포함)을 밑바닥부터 작성한다.

**수학적 기초**
| 개념 | 수식 |
|------|------|
| RMSNorm | $\bar{a}_i = \frac{a_i}{\sqrt{\frac{1}{n}\sum_{j=1}^n a_j^2 + \epsilon}} g_i$ |
| RoPE (Rotary PE) | $f(q, m) = (q_1\cos(m\theta) - q_2\sin(m\theta), q_2\cos(m\theta) + q_1\sin(m\theta), \dots)$ |
| SwiGLU | $\text{SwiGLU}(x) = \text{Swish}(xW) \otimes (xV)$ |
| MoE Router Loss | $L_{aux} = \alpha \cdot N \sum_{i=1}^N f_i \cdot P_i$ (Load Balancing을 위한 보조 손실) |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_llama_architecture_deepdive.ipynb` | 기존 Transformer와의 차이: RMSNorm, SwiGLU, KV-Cache 최적화 |
| `02_rotary_position_embedding.ipynb` | 절대좌표와 상대좌표의 결합: RoPE의 복소수 평면 공간 증명 및 구현 |
| `03_gqa_and_multi_query_attention.ipynb` | MHA -> MQA -> GQA 진화 과정 및 Q, K, V 헤드 비율에 따른 효율 비교 |
| `04_mixture_of_experts_moe.ipynb` | Expert 라우팅 메커니즘, Top-k 선택, 활성화 희소성(Sparsity) 제어 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_implement_llama_scratch.ipynb` | RoPE와 SwiGLU를 적용한 소형 Llama 모델을 모듈별로 직접 작성하기 |
| `practice/ex02_custom_moe_layer.ipynb` | 4개의 Expert 중 상위 2개(Top-2)를 선택하고 보조 손실을 계산하는 커스텀 레이어 작성 |

**주요 키워드**: `RoPE`, `SwiGLU`, `RMSNorm`, `GQA`, `MoE`, `Load Balancing Loss`

---

## Chapter 13 — [Advanced] 생성 AI 심화 (Diffusion Models & SDE)

**디렉토리**: `chapter13_genai_diffusion/`

**학습 목표**

- 확산 모델이 어떻게 순수한 수학적 노이즈 확률 변환에서 시작되는지 이해하고 (DDPM 모델), 시간 연속형 확률 미분 방정식(SDE)에 기반한 최신 모델들을 다룬다.

**수학적 기초**
| 개념 | 수식 |
|------|------|
| DDPM Forward Process | $q(x_t \| x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ |
| DDPM Reverse Process | $p_\theta(x_{t-1} \| x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$ |
| DDPM Evidence Lower Bound (ELBO) | $\mathcal{L}_{vlb} = L_0 + L_1 + \dots + L_T$ ➔ $\mathcal{L}_{simple}(\theta) = \mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2]$|
| SDE (확률미분방정식) | $dx = f(x,t)dt + g(t)dw$ |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_ddpm_theory_and_math.ipynb` | 마르코프 체인 기반의 확산. Forward와 Reverse process 수식 도출. |
| `02_unet_for_diffusion.ipynb` | 노이즈 예측기(Noise Predictor) 역할을 하는 DDPM 전용 UNet 아키텍처 해부 |
| `03_conditional_diffusion_cfg.ipynb` | Classifier-Free Guidance (CFG): 텍스트 등 조건을 부여하여 생성 방향성 조절 |
| `04_score_based_models_and_sde.ipynb` | Score-matching과 Langevin 역학. 이산(Discrete) 시간에서 연속(Continuous) 시간 SDE로 확장에 대한 개념 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_ddpm_forward_reverse.ipynb` | 특정 이미지에 스케줄러(Beta)를 적용해 노이즈를 입히고(Forward), 수식에 따라 제거하는(Reverse) 시뮬레이션 |
| `practice/ex02_implement_cfg_generation.ipynb` | 클래스 조건부 기반의 손글씨 또는 특정 도메인 이미지 생성 및 조건 조작 훈련 |

**주요 키워드**: `DDPM`, `ELBO`, `UNet`, `CFG (Classifier-Free Guidance)`, `SDE`, `Langevin Dynamics`

---

## Chapter 14 — [Advanced] 극단적 추론 최적화 (Extreme Inference Optimization)

**디렉토리**: `chapter14_extreme_inference/`

**학습 목표**

- 대규모 모델의 서빙에서 병목(Latency, Throughput)이 일어나는 물리적 원인을 파악하고, PagedAttention과 양자화를 이용한 해결책을 구축한다.

**수학적 기초**
| 개념 | 수식 |
|------|------|
| FlashAttention Tiling | $O\left(\frac{N^2 d}{M}\right)$ (Tiling & Recomputation을 통한 SRAM 중심 IO 최적화) |
| KV Cache 크기 | $2 \times L \times H \times D \times \text{batch} \times \text{seq\_len} \times \text{bytes}$ |
| AWQ (Activation-aware Weight) | $\min_Q \| WX - Q(W \cdot S)S^{-1}X \|^2$ (아웃라이어 인지 양자화) |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_inference_bottlenecks.ipynb` | Prefill 단계 연산량(Compute-bound) vs Decode 단계 대역폭(Memory-bound)의 개념 |
| `02_flash_attention_deepdive.ipynb` | Tiling 기법과 하드웨어 HBM/SRAM 구조. FlashAttention 버전별 성능 발전사 |
| `03_vllm_and_paged_attention.ipynb` | 메모리 파편화를 막는 vLLM PagedAttention 메커니즘, Continuous Batching 및 스케줄링 |
| `04_advanced_quantization_awq_gptq.ipynb` | 단순 PTQ를 넘어서는 최신 LLM 양자화(AWQ, GPTQ) 전략, Outlier 채널 보호 수학 적용 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_paged_attention_sim.ipynb` | OS의 Page Table(가상 메모리) 개념을 차용한 PagedAttention의 메모리 블록 할당 시뮬레이션 코드 |
| `practice/ex02_awq_quantization_eval.ipynb` | Llama 모델의 가중치를 AWQ 방식으로 W4A16 양자화한 후 Perplexity 손실 유무를 평가 |

**주요 키워드**: `FlashAttention`, `PagedAttention`, `Continuous Batching`, `AWQ`, `KV Cache`

---

## Chapter 15 — [Advanced] AI 얼라인먼트와 강화학습 (Alignment & RLHF)

**디렉토리**: `chapter15_alignment_rlhf/`

**학습 목표**

- 모델의 출력을 인간의 윤리적, 논리적 의도에 맞게 정렬(Align)하기 위해 보상 모델을 이용한 RLHF 파이프라인과 이를 수학적으로 단순화한 DPO 기법을 마스터한다.

**수학적 기초**
| 개념 | 수식 |
|------|------|
| RLHF Reward Model Loss | $\mathcal{L}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]$ |
| PPO 목적 함수 | $L^{CLIP}(\theta) = \mathbb{E}\left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$ |
| DPO (Direct Preference Obj)| $\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w\|x)}{\pi_{ref}(y_w\|x)} - \beta \log \frac{\pi_\theta(y_l\|x)}{\pi_{ref}(y_l\|x)} \right) \right]$ |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_rl_for_nlp_basics.ipynb` | 강화학습 기초: 마르코프 체인(MDP), Policy Gradient 시퀀스 생성 수식, Actor-Critic |
| `02_rlhf_pipeline_overview.ipynb` | InstructGPT 논문 기반 SFT(Supervised Tuning) → RM(Reward Model) → PPO 3단계 아키텍처 구성 요소 |
| `03_direct_preference_optimization.ipynb` | 보상 모델을 생략하는 DPO의 베이즈 기반 수학적 도출 및 최신 파생 기법 (ORPO, KTO) |
| `04_red_teaming_and_constitutional_ai.ipynb` | 모델의 안전성과 탈옥(Jailbreak) 방어. Anthropic의 RLAIF(AI를 이용한 피드백) 프레임워크 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_train_reward_model.ipynb` | 선호도(Preference) Pair 쌍 데이터를 활용하여 Bradley-Terry 모델 기반 Reward Model 훈련 로직 구현 |
| `practice/ex02_dpo_fine_tuning_lora.ipynb` | 허깅페이스 TRL 라이브러리와 LoRA(PEFT)를 적용해 베이스 모델을 DPO를 통해 지시학습 봇으로 바꾸기 |

**주요 키워드**: `RLHF`, `PPO`, `DPO`, `Bradley-Terry Model`, `Alignment`, `TRL`

---

## Chapter 16 — [State-of-the-Art] 최신 거대 모델의 효율성 (Sparse Attention & DeepSeek/Qwen)

**디렉토리**: `chapter16_sparse_attention/`

**학습 목표**

- DeepSeek-V3와 Qwen3.5 등 최신 중국발 오픈소스 LLM의 핵심 연산 효율화 기법인 Multi-head Latent Attention (MLA) 및 Sparse MoE의 원리를 분석하고 코드 레벨에서 이해한다.

**수학적 기초**
| 개념 | 수식 |
|------|------|
| MLA (Latent Compression) | $\mathbf{c}_t^{KV} = W_d^{KV} \mathbf{h}_t$ (고차원 활성화 값을 저차원 Latent 공간으로 투영하여 KV Cache 대폭 감소) |
| DeepSeekMoE 라우팅 | $\text{Top-}k$ 전문가 라우팅에서 Shared Expert와 Routed Expert 분리 (지식의 파편화 방지) |
| MTP (Multi-Token Prediction) | $L_{total} = L_{ce}(y_{t}) + \lambda \sum_{i=1}^{k} L_{ce}(y_{t+i})$ (복수의 미래 토큰을 동시 예측) |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_deepseek_v3_architecture.ipynb` | DeepSeek-V3 혁신 리뷰: Auxiliary-Loss-Free Load Balancing, FP8 혼합 정밀도 훈련 |
| `02_multi_head_latent_attention.ipynb` | GQA(Grouped-Query Attention)의 한계와 MLA를 통한 차원 축소 및 메모리 IO(Off-chip traffic) 최소화 |
| `03_qwen_hybrid_attention.ipynb` | Qwen3.5의 Gated Delta Networks (Linear Attention)와 Sparse MoE의 결합 |
| `04_long_context_optimization.ipynb` | DSA(DeepSeek Sparse Attention)와 Long-context 처리 비용 절감(>50%) 기법 |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_mla_from_scratch.ipynb` | 순수 PyTorch/TensorFlow로 KV 차원 압축 및 복원 과정을 거치는 MLA 커스텀 레이어 구현 |
| `practice/ex02_shared_expert_moe.ipynb` | 모든 토큰이 참조하는 Shared Expert와 특정 지식만 담당하는 Routed Expert 레이어 분리 코딩 |

**주요 키워드**: `DeepSeek-V3`, `Qwen3.5`, `Multi-head Latent Attention (MLA)`, `DeepSeekMoE`, `MTP`, `FP8 Training`

---

## Chapter 17 — [State-of-the-Art] 비디오 생성 모델과 DiT (Diffusion Transformers & Sora/Hunyuan)

**디렉토리**: `chapter17_diffusion_transformers/`

**학습 목표**

- OpenAI의 Sora와 Tencent의 HunyuanVideo를 구동하는 핵심 근간인 Diffusion Transformer (DiT) 구조를 해부하고, 시공간(Spatiotemporal) 비디오 압축 및 토큰화 원리를 구현한다.

**수학적 기초**
| 개념 | 수식 |
|------|------|
| DiT 조건부 삽입 (adaLN) | $\mathbf{h} = \text{LayerNorm}(\mathbf{h}) \odot (1 + \gamma_c) + \beta_c$ (시간 $t$와 텍스트 컨디션을 scale/shift로 주입) |
| 3D Causal VAE | 잠재 공간 축소: $z \in \mathbb{R}^{C \times (T/M_t) \times (H/M_h) \times (W/M_w)}$ (M은 압축 비율) |
| Patch-based Flattening | 3D 비디오 텐서 $(T, H, W)$를 분할 후 비겹침 패치 시퀀스로 1D (Sequence Length) 변환 |

**강의 파일**
| 파일 | 내용 |
|------|------|
| `01_from_unet_to_dit.ipynb` | 기존 U-Net 기반 Diffusion의 한계와 Scalability 법칙을 증명한 DiT(Diffusion Transformer) 트랜지션 |
| `02_spatiotemporal_vae.ipynb` | Sora 및 HunyuanVideo의 핵심: 3D Causal VAE를 활용한 시간적 깜박임(Flickering) 억제 및 시간/공간 압축 |
| `03_sora_and_patch_tokenization.ipynb` | 다양한 해상도/비율을 지원하게 만드는 Spatiotemporal Patches 구조와 1D 시퀀스 매핑 |
| `04_hunyuanvideo_dual_stream.ipynb` | HunyuanVideo의 Dual-stream to Single-stream 멀티모달 퓨전 (텍스트+비디오 토큰의 MLLM 기반 퓨전 구조) |

**실습 파일**
| 파일 | 내용 |
|------|------|
| `practice/ex01_spatiotemporal_patcher.ipynb` | 미니 3D 비디오 텐서를 입력받아 시공간 포지셔널 인코딩(Positional Encoding)이 추가된 DiT 패치 시퀀스로 변환하는 함수 구현 |
| `practice/ex02_dit_block_with_adaln.ipynb` | 트랜스포머 블록에 시간 단계 $t$와 클래스 조건을 주입하는 Adaptive Layer Norm (adaLN-Zero) 모듈 구현 및 노이즈 예측기 조립 |

**주요 키워드**: `DiT`, `Sora`, `HunyuanVideo`, `Spatiotemporal VAE`, `adaLN`, `3D Causal VAE`

---

## 최종 실전 프로젝트

| 프로젝트                             | 설명                                                                                   | 주요 기술                                     |
| ------------------------------------ | -------------------------------------------------------------------------------------- | --------------------------------------------- |
| `project01_image_classifier/`        | 커스텀 이미지 분류기 (EfficientNetB0 전이학습 + TFLite 변환)                           | CNN, 전이학습, tf.data                        |
| `project02_sentiment_analysis/`      | 한국어 감성 분석 API (KoBERT Fine-Tuning + FastAPI 서빙)                               | BERT, HuggingFace, SavedModel                 |
| `project03_korean_nlp/`              | 한국어 NLP 파이프라인 (형태소 분석 + 주제 분류)                                        | KoNLPy, Bi-LSTM, TFRecord                     |
| **`project04_custom_llm_server/`**   | **[Advanced] vLLM 기반 커스텀 LLM 추론 서버 (AWQ 양자화 + PagedAttention)**            | **AWQ, vLLM, FastAPI, XLA**                   |
| **`project05_diffusion_image_gen/`** | **[Advanced] 조건부(Conditional) Diffusion 스크래치 구현**                             | **DDPM, UNet, Classifier-Free Guidance**      |
| **`project06_dpo_aligned_bot/`**     | **[Advanced] DPO/LoRA 기반 나만의 안전한 지시학습(Instruct) AI 보조봇**                | **DPO, PEFT(LoRA), TRL**                      |
| **`project07_mini_sora_dit/`**       | **[SOTA] 움직이는 MNIST(Moving MNIST) 데이터를 활용한 소형 DiT 비디오 생성 모델 훈련** | **DiT, 3D Conv, adaLN, Spatiotemporal Patch** |

---

## 참고 문헌 및 논문 (References)

**기초 및 고전 논문/자료**

- [TensorFlow 공식 문서](https://www.tensorflow.org/guide)
- [Keras API 레퍼런스](https://keras.io/api/)
- [TensorFlow 튜토리얼](https://www.tensorflow.org/tutorials)
- [HuggingFace Transformers + TF](https://huggingface.co/docs/transformers/keras_callbacks)
- Goodfellow et al., _Deep Learning_ (2016) — 수학적 기초
- Vaswani et al., _Attention Is All You Need_ (2017) — Transformer
- Kingma & Welling, _Auto-Encoding Variational Bayes_ (2013) — VAE
- Goodfellow et al., _Generative Adversarial Networks_ (2014) — GAN

**시스템 및 Advanced 논문**

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (DeepSpeed)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [PagedAttention: Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180) (vLLM)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (DPO)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)

**SOTA 아키텍처 (DeepSeek, DiT, Sora)**

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (MLA, DeepSeekMoE, FP8 Training)
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- [Video generation models as world simulators](https://openai.com/research/video-generation-models-as-world-simulators) (OpenAI Sora Technical Report)
- [HunyuanVideo: A Systematic Framework For Large Video Generation Models](https://arxiv.org/abs/2412.17601) (Tencent)
