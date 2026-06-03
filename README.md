
# TinyGPT: Character-Level Decoder-Only Transformer

본 프로젝트는 셰익스피어 희곡 데이터(`tinyshakespeare`)를 학습하여 텍스트를 생성하는 문자 단위(Character-Level) 대형 언어 모델(LLM)의 구현체입니다. 

OpenAI의 GPT 시리즈와 동일한 Decoder-Only Transformer 아키텍처를 기반으로 하며, 단순히 프레임워크의 API를 호출하는 것을 넘어 텐서의 차원(Shape) 변화와 데이터 흐름의 물리적/수학적 의미를 추적하고 최적화하는 데 목적을 두었습니다.

---

## Model Architecture (모델 구조)

본 모델은 다음과 같은 하이퍼파라미터를 기반으로 설계되었습니다.

| 파라미터 | 설정값 | 설계 의도 |
| :--- | :--- | :--- |
| `vocab_size` | 65 | 데이터셋에 등장하는 고유 문자 및 기호의 총개수 |
| `block_size` | 64 | 모델이 한 번에 처리하는 문맥의 최대 길이 (Sequence Length, $T$) |
| `emb_dim` | 128 | 단일 토큰이 가지는 고유 특징 벡터의 차원 크기 (Channel, $C$) |
| `num_heads` | 4 | Multi-Head Attention 내부의 병렬 처리 부서(Head) 개수 |
| `num_layers` | 4 | Transformer Block (MHA + FFWD)의 중첩 횟수 |

---

## Data Flow Pipeline (텐서 형태 변환 추적)

모델의 순전파(Forward Pass) 과정에서 데이터가 어떻게 팽창하고 압축되는지, 그 형태(Shape)의 변화를 추적한 파이프라인입니다. 입력 배치 크기 `B=64`, 시간 축 `T=64`를 기준으로 작성되었습니다.

**1. Input Layer (`[64, 64]`)**
입력 데이터는 각 위치에 정수형 인덱스를 담은 2차원 텐서로 주어집니다.

**2. Embedding Layer (`[64, 64, 128]`)**
`Token Embedding`과 `Positional Embedding`이 합산됩니다. 단순한 정수 인덱스가 128차원의 의미를 지닌 실수 벡터 공간으로 팽창하며, 동시에 시퀀스 내의 '절대적 위치 좌표' 정보가 부여됩니다.

**3. Transformer Blocks (`[64, 64, 128]`)**
4개의 연속된 블록을 통과하는 동안 텐서의 차원 형태는 `[64, 64, 128]`로 일정하게 유지됩니다. 하지만 내부의 실수 값들은 어텐션 연산을 통해 주변 문맥 정보를 흡수하고, 피드포워드 연산을 통해 비선형적으로 고도화됩니다.

**4. LM Head / Output Layer (`[64, 64, 65]`)**
최종적으로 128차원의 은닉 상태(Hidden State)를 우리가 예측해야 할 65개의 타겟 어휘 확률(Logits) 공간으로 선형 투영(Linear Projection)합니다.

---

## Core Component Analysis (핵심 모듈 분석)

모델의 성능과 학습 안정성을 결정짓는 주요 컴포넌트들의 설계 의도와 내부 메커니즘입니다.

### Multi-Head Masked Attention


**병렬적 문맥 분석:** 128차원의 공간을 통째로 연산하지 않고, 4개의 Head(각 32차원)로 분할합니다. 이를 통해 모델은 문맥을 문법적, 감정적, 인물 관계적 등 다각도의 관점에서 동시에 분석할 수 있습니다.

**Causal Masking:** Autoregressive 모델의 특성상 미래의 데이터를 참조하는 것을 방지하기 위해, 상삼각행렬(Upper Triangular Matrix) 영역의 Attention Score를 `-inf`로 마스킹 처리하여 인과성(Causality)을 보장합니다.

**Projection (`Linear(128, 128)`):** 각 Head에서 독립적으로 처리된 결과를 단순히 이어 붙이는 것(`concat`)에 그치지 않고, 가중치 행렬을 통해 다시 한번 유기적으로 융합하는 화학적 결합 단계를 거칩니다.

### Position-wise FeedForward Network
**독립적 정보 압축:** Attention이 토큰 간의 '소통(Interaction)'을 담당한다면, FFWD는 각 위치(Position)에서의 '독립적 사고'를 담당합니다. 시간 축($T$)을 가로지르는 연산 없이, 각 토큰이 가진 128차원의 정보를 512차원으로 팽창시켜 비선형 특징(`ReLU`)을 추출한 뒤 다시 128차원으로 압축해 냅니다.

### Optimization & Stability
**Residual Connection (`x + layer(x)`):** 4개의 층을 거치면서 발생할 수 있는 기울기 소실(Gradient Vanishing) 현상을 방지하고, 초기 임베딩의 정체성을 보존하기 위해 입력값을 출력값에 바로 더해주는 우회로를 설계했습니다.

**Layer Normalization:** 깊은 신경망 연산 과정에서 활성화 값(Activation)이 폭주하는 것을 막기 위해, 임베딩 차원을 기준으로 평균과 분산을 정규화하여 학습의 안정성을 크게 높였습니다.

---

## System Workflow: Training & Inference

### 1. Training (학습 메커니즘)
**병렬 손실 계산 (Dimensional Alignment):** 시퀀스 모델은 한 번의 순전파로 $T$개의 시점에 대한 다음 토큰을 병렬로 예측합니다. 이때 PyTorch의 `F.cross_entropy` 함수가 요구하는 텐서 규격(`[Batch, Class, Time]`)을 맞추기 위해, 모델의 출력인 `[B, T, C]` 텐서를 `logits.transpose(1, 2)`로 축 변환하여 효율적인 병렬 채점을 수행합니다.

### 2. Autoregressive Sampling (추론 메커니즘)
**Slicing (`logits[:, -1, :]`):** 생성 단계에서는 이미 주어진 과거의 문맥을 다시 예측할 필요가 없습니다. 따라서 전체 시퀀스 $T$에 대한 예측 중 가장 마지막 위치(`-1`)의 확률 분포만 슬라이싱하여 다음 토큰 예측에 활용합니다.

**Stochastic Generation:** 단순히 가장 확률이 높은 토큰을 취하는 Greedy Search 대신, `F.softmax`로 도출된 확률 분포에 기반하여 `torch.multinomial` 샘플링을 수행합니다. 이를 통해 앵무새 같은 반복을 피하고 다채로운 텍스트 생성을 유도합니다.

**Sliding Window:** 새로 생성된 토큰을 기존 문맥의 맨 뒤에 덧붙이고(`torch.cat`), 최대 길이(`block_size`)를 초과할 경우 맨 앞 토큰을 버리는 슬라이딩 윈도우 방식을 적용해 무한한 길이의 텍스트를 생성해 냅니다.

---

## Quick Start

### Requirements
```bash
pip install torch
