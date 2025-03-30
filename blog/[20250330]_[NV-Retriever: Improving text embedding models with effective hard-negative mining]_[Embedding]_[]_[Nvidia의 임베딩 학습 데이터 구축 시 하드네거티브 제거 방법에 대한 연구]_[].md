
# NV-Retriever: Improving text embedding models with effective hard-negative mining

https://arxiv.org/html/2407.15831v2


## 1. 개요

**들어가기 전 용어 간단 정리**

| 용어 | 설명 |
| --- | --- |
| Contrastive Learning | Query와 Positive를 가깝게, Negative는 멀게 학습하는 방식 |
| Hard-Negative | Query와 비슷하지만 실제 정답은 아닌 문서. 학습 난이도 ↑ |
| False Negative | 실제로는 정답인데 Negative로 잘못 분류된 경우 |
| Positive-aware Mining | Positive 점수를 기준으로 잘못된 Negative를 제거하는 방법 |

#### 1.1 이 논문의 주된 연구 내용

- 임베딩 모델은 보통 **Contrastive Learning** 방식으로 학습되며, 이때 **Hard-Negative 샘플 선택**이 성능 향상에 매우 중요
- 본 논문은 효과적인 **Hard-Negative 샘플 선택 방법에 대한 연구 논문**

---

#### 1.2 기존 문제점

- 기존 연구에서는 **Hard-Negative Mining 기법 자체에 대한 분석이나 설명은 부족**했습니다.
- **Hard-Negative 샘플링 시** 잘못된 Negative (사실은 Positive인 샘플)가 들어갈 경우 학습을 방해할 수 있음.

---

#### 1.3 주요 기여 (논문의 핵심 Contribution)

### (1) **Positive-aware Hard-negative Mining 기법 제안**

→ Positive의 유사도 점수를 기준(anchor)으로 삼아 **False Negative를 제거**하고, contrastive learning 효율을 높이는 방법 제안

### (2) **Hard-negative Mining 기법에 대한 구성별 비교 연구**

→ Teacher 모델, base 모델 조합을 다양하게 바꾸며 어떤 방식이 효과적인지 **ablation study** 수행

### (3) **대규모 성능 검증: NV-Retriever-v1**

→ 제안 기법을 실제로 적용해 **MTEB Retrieval 벤치마크 1위 모델(NV-Retriever-v1)** 개발

→ 해당 모델은 **BEIR 기준 점수 60.9**로 최고 성능 기록 (2024년 7월)

---

## 2. Background


#### 2.1 Text Embedding Models

**텍스트 임베딩 모델이란?**

→ 다양한 길이의 텍스트를 고정된 차원의 벡터로 바꿔주는 모델. 검색, 추천, 의미 유사도 등 다양한 작업에 사용됨.

####🔹 주요 발전 흐름

- **Sentence-BERT (2019)**
    
    → BERT 기반 쌍(pair) 입력 구조 (Siamese/Triplet Network)로 문장을 벡터 공간에 투영
    
    → 다양한 pooling, loss 함수 실험
    
- **DPR (2020)**
    
    → Query와 Passage를 각각 별도 BERT로 처리하는 Bi-Encoder 구조 제안
    
    → CL 기반으로 학습
    
- **E5 시리즈 (2022)**
    
    → Pre-training(비지도) → Fine-tuning(지도) 두 단계 구성
    
    → 여기서 Pre-training은 Masked Language Modeling이 아닌 Contrastive Pre-training으로 제목, 문서 쌍 간의 유사도 학습을 의미 
    
    → 질문, 문서 쌍의 Retrieval Task 성능을 높이는데 제목, 문서 쌍의 유사도를 학습하는 이유 : 질문, 문서 쌍의 데이터는 라벨링된 데이터로 데이터의 개수가 많지 않기 때문에 비교적 개수가 많고 광범위의 데이터인 제목, 문서 쌍의 데이터를  먼저 학습
    
    → Contrastive Pre-training 의 단점 : 광범위한 데이터를 수집해야 하기 때문에 데이터 수집의 어려움과 학습할 데이터의 양이 많기 때문에 소요되는 컴퓨팅 리소스도 크다는 단점 존재
    
- **E5-Mistral (2023)**
    
    → Encoder(BERT) 대신 **Decoder 기반 LLM(Mistral-7B)** 사용
    
    → 다양한 언어/태스크의 synthetic data로 1회만 fine-tuning
    
    → 1회 fine-tuning이 갖는 의미 : Encoder 모델의 경우 보통 (Contrastive) Pre-training(비지도) → Fine-tuning(지도) 두 단계 구성되는데 되고 Pre-training의 경우 많은 컴퓨팅 리소스가 필요한데 이 부분을 생략할 수 있음
    
    → (Contrastive) Pre-training을 생략해도 되는 이유 : LLM 모델의 경우 Encoder 모델 보다 더 광범위하고 많은 데이터로 사전 학습 되었기 때문에 해당 단계를 생략하고 Fine-tuning 만 해도 성능이 높게 나올 수 있음
    
- **Gemini Embedding (2025)**
    
    → 이 논문 이후에 나온 모델이지만 **E5-Mistral** 방식과 다른 부분 소개 : LLM 모델을 base 모델 사용 시 생략된 Pre-training 단계를 사용
    
    → Pre-training 단계를 사용하는 이유 : Pre-training 단계에서 autoregressive generation을 encoding으로 매개변수 조정의 목적이 있다.
    

#### 🔹 평가 벤치마크

- **BEIR (2021)**
    
    → 18개 retrieval task로 구성, zero-shot 평가 목적
    
- **MTEB (2022)**
    
    → BEIR + 분류, 클러스터링 등 총 56개 task로 확장
    
    → Retrieval task는 BEIR 중 15개를 포함
    

---

#### 2.2 Hard-Negative Mining for Fine-tuning

Contrastive Learning에서는 (query, positive, negative)의 **triplet**이 필요

→ 이때 “negative”가 **얼마나 어려운(hard)** 샘플이냐가 학습 품질에 큰 영향을 줌

### **🔹 Negative 샘플링 방식**

- **In-Batch Negative**
    
    → 같은 배치 내 다른 쿼리의 Positive를 Negative로 활용
    
    → 효율적이지만 대부분 무관한 텍스트라서 **너무 쉬운 negative**가 됨 → 학습 효과 낮음
    
- **Memory Bank / Cross-Batch (Xiong et al., 2020)**
    
    → 과거 배치의 embedding을 모아두거나 여러 GPU 배치 공유
    
- **Hard Negative Mining**
    
    → 실제 query와 비슷하지만 정답이 아닌 문서를 찾아 negative로 활용
    
    → BM25나 embedding 기반 retrieval 모델로 뽑음
    
    → DPR도 BM25로 hard-negative를 미리 뽑아서 사용
    

---

#### 🔍 False Negatives 문제

> “Query와 매우 유사한 문서 중 일부는 실제로 Positive일 수도 있다.”
> 
- MS MARCO 실험에 따르면, 가장 유사한 100개 중 **70%가 실제 Positive**일 가능성이 있음 (Qu et al., 2020)
    
    → 이를 모르고 negative로 넣으면 모델 혼란
    
    → 해결: 유사도 점수가 높은 문서는 제거하거나 denoise
    

#### 기존의 해결법

- **Cross-Encoder로 필터링** (RocketQA 등)
- **LLM으로 relevance score 판단**
- 문제: 대규모 데이터에 쓰기엔 비용 큼 (모든 쿼리-문서 쌍에 대해 추론 필요)
- **Snowflake-arctic-embed-l**: threshold 조절 (e.g. 점수 0.4, 0.5, 0.8) 실험

#### 이 논문을 읽기 전 나의 해결법

- **Snowflake 논문(**https://arxiv.org/html/2405.05374v1)**을 참고(부록 부분)**하여 Retriever 모델의 threshold 조절 (e.g. 점수 0.4, 0.5, 0.8)을 통한 필터링
- 이 논문에서 말하는 기존의 방법들을 사용

---

## 3. Methodology


### **3.1 Positive-aware Hard-negative Mining Methods**

#### 🔸 배경 요약

- Contrastive Learning에서는 query와 positive를 가깝게, negative는 멀게 만들도록 학습함.
- 일반적으로 **Top-K 가장 유사한 문서들**을 negative로 선택하는데, 이 방식은 **False Negative(실제로는 정답인데 label이 없음)**를 포함할 위험이 큼.

#### 🔸 기존 방법들 요약

| 기법 | 설명 | 한계 |
| --- | --- | --- |
| Top-K shifted by N | 상위 N개는 무시하고 그 이후 K개를 사용 | relevance 점수 무시, 좋은 negative를 버릴 수도 있음 |
| TopK-Abs | 절대 점수 기준(threshold 이상은 제거) | positive의 점수를 고려하지 않음 |

---

#### 🌟 **저자들의 제안: Positive-aware Mining**

→ **Positive passage의 유사도 점수(`sim(q, d⁺)`)를 기준(anchor)**으로 활용해

False Negative 가능성이 높은 negative를 제거하는 방식

#### 핵심 아이디어:

> “Positive 점수보다 너무 유사한 Negative는 False Negative일 수 있으니 제외하자.”
> 

---

#### 🔧 제안된 두 가지 필터 방식:

| 방법 | 설명 | 유사도 임계값 기준 |
| --- | --- | --- |
| **TopK-MarginPos** | `Negative < Positive - margin` | 절대값 margin (예: 0.2 이상 차이) |
| **TopK-PercPos** | `Negative < Positive * percentage` | 비율 margin (예: 90% 이하면 통과) |

#### 📌 알고리즘 구조 (의사 코드 형태)

```python
# q : query, p : positive, n : negative
for n in top_k_negatives:
    if sim(n, q) < sim(p, q) - margin:  # TopK-MarginPos
        keep n
```

→ 또는 `sim(n, q) < sim(p, q) * 0.9` (TopK-PercPos)

#### 장점

- Positive score를 기준으로 false negative를 걸러낼 수 있음

---

#### **3.2 Research Questions**

이 논문은 아래 세 가지 연구 질문(RQ)을 실험을 통해 검증합니다:

| RQ | 내용 |
| --- | --- |
| RQ1 | 서로 다른 **teacher 모델**로 hard-negative를 mining할 때 downstream 성능 차이가 얼마나 나는가? |
| RQ2 | 서로 다른 teacher에서 mining한 hard-negatives를 **ensemble**하면 더 좋은가? |
| RQ3 | 서로 다른 **hard-negative mining 방법**(Naive Top-K vs Positive-aware 등)의 비교 |

---

#### 3.3 **Experiments Setup**

#### 3.3.1 Training

- Base model: `e5-large-unsupervised` 또는 `Mistral-7B-v0.1`
- 학습 데이터셋 (총 287k 쿼리):
    - Natural Questions (NQ)
    - StackExchange (2023 덤프)
    - SQuAD

#### 3.3.2 Evaluation

- 평가 데이터셋 (BEIR / MTEB 중 일부):
    - NQ
    - HotpotQA
    - FiQA-2018
        
        → 이 3가지는 **Q&A + RAG 환경**에 가장 밀접한 태스크임
        
- RQ3 실험에서는 **NV-Retriever-v1**의 전체 학습 데이터로 스케일 확장 실험도 진행

---

## 4. 실험 결과 및 해석 요약


#### **4.1 RQ1: 서로 다른 teacher model로 mining할 경우 결과는?**

#### 🔍 실험 개요

- 동일한 base 모델: `e5-large-unsupervised`
- 동일한 학습 데이터셋에 대해 **서로 다른 teacher 모델**을 사용해 hard-negative 4개씩 mining
- 그 결과를 가지고 base 모델을 fine-tune → 성능 비교 (NDCG@10 기준)

#### 🧠 사용된 teacher 모델들:

| 모델명 | 특징 |
| --- | --- |
| **BM25** | Sparse baseline (가장 기본적인 검색 방식) |
| **e5-large-unsupervised** | 비지도 학습만 된 Dense 모델 |
| **e5-large-v2** | 지도 학습 포함된 E5 개선 버전 |
| **snowflake-arctic-embed-l** | E5와 유사하나 더 개선된 학습 방식 적용 |
| **e5-mistral-7b-instruct** | Mistral 기반 디코더 모델 (7B) |
| **NV-Embed-v1** | Mistral + Bi-directional attention 구조 (7.8B) |

#### 📊 실험 결과 요약

- **BM25로 뽑은 하드네거티브가 가장 성능이 낮음**
    
    → 기존 연구(Karpukhin et al., 2020)와 반대 결과. 최신 dense 모델에서는 BM25 negative가 더 이상 효과적이지 않음.
    
- **모델 규모 & fine-tuning 데이터 품질이 좋을수록 더 좋은 hard-negative 생성**
    
    → NV-Embed-v1, e5-mistral-7b-instruct ≫ 나머지 작은 dense 모델들
    
- **결론:**
    
    > 더 큰 teacher + 더 좋은 학습 데이터를 가진 모델일수록 hard-negative mining의 질이 높아진다.
    > 
    
    → 같은 맥락으로 gemini embedding 모델도 LLM 을 사용해서 hard-negative mining을 진행
    

---

#### 4.2 RQ2: 여러 teacher model의 hard-negative를 ensemble하면 더 좋을까?

#### 실험 동기

- 서로 다른 teacher 모델들이 뽑아내는 hard-negative가 **겹치는 게 거의 없음 (Jaccard 유사도 < 30%)**
- → 서로 다른 관점에서 추출된 hard-negative를 섞어보면 성능 향상 가능성 있음

#### 🔍 Ensemble 방법 두 가지

| 방법 | 설명 |
| --- | --- |
| **Cross-sample** | 각 데이터 샘플마다 하나의 teacher를 랜덤으로 선택해서 4개 negative를 받음 |
| **Intra-sample** | 각 샘플에 대해 4개 teacher 모델에서 각각 top-1 negative를 추출해 구성 |
- Intra-sample은 중복 가능 → 두 버전 실험:
    - **dedup:** 중복 제거 + 다음 순위 negative로 대체
    - **no-dedup:** 중복 그대로 사용

#### 📊 결과 요약 (Table 2)

- **Cross-sample**: baseline보다 못함 → 성능 X
- **Intra-sample (dedup)**: baseline보다 약간 향상
- **Intra-sample (no-dedup)**: **가장 성능 우수**

#### 🔎 왜 no-dedup이 더 좋았을까?

> “만약 여러 teacher가 같은 문서를 hard-negative로 선택했다면,
> 
> 
> 그건 정말 중요한 hard-negative일 수 있음.
> 
> 중복으로 등장시키면 contrastive loss에서 더 큰 weight를 받으므로 학습에 긍정적.”
> 

→ 질 좋은 하드 네거티브의 경우 중복하여 네거티브를 구성하는 것이 긍정적인 효과를 준다.

## 🧠 이 섹션에서 얻을 수 있는 인사이트

| 핵심 교훈 | 설명 |
| --- | --- |
| 좋은 hard-negative는 **좋은 teacher 모델이 있어야** 가능 | dense, 대형 모델이 더 좋은 negative를 찾아냄 |
| **Ensemble 전략은 신중히 설계되어야** 함 | 단순 조합보다, 중복을 전략적으로 활용하는 방식이 더 나음 |
| **중복 hard-negative가 반드시 나쁜 건 아님** | 오히려 모델 간 합의의 표시로, 중요한 학습 신호가 될 수 있음 |

---

#### 4.3 RQ3: Mining 방법별 성능 비교

#### 🔬 4.3.1. Ablation 실험 (마진/임계값 설정 실험)

| 방법 | 실험 설정 | 최적 성능 조건 | 성능 요약 |
| --- | --- | --- | --- |
| **Top-K Shifted by N** | 상위 N개를 제외 | N=10일 때 가장 좋음 | 상위 문서가 FN일 가능성이 높기 때문 |
| **TopK-Abs** | 점수 임계값으로 필터링 | 최대 0.7 이하가 가장 좋음 | 0.7보다 높으면 FN 포함 위험, 낮으면 약한 negative만 남음 |
| **TopK-MarginPos** | Positive 점수 - margin | margin=0.05일 때 best | margin이 너무 크면 좋은 negative도 제거함 |
| **TopK-PercPos** | Positive 점수 × 퍼센트 | 95%일 때 best | 그 이상은 FN 포함 가능성 ↑ |

---

#### 🎯 4.3.2. Sampling 전략

- **Sampled Top-K**: top-k 중 무작위 샘플링
- **Top-1 + Sampled**: 가장 상위 1개는 고정하고, 나머지는 무작위 샘플링

#### 실험 결과:

- `Top-1+Sampled` 방식이 더 안정적
- `k=10`까지만 성능 향상, 그 이상은 성능 하락
    - `top-k` 중에서 **네거티브 샘플을 무작위로 뽑는 방식**에서는,
    
    → `k=10` 정도까지는 성능이 올라감 (왜냐하면 유의미한 hard-negative들이 top 10 안에 들어있기 때문)
    → 그런데 `k` 값을 **더 크게 (예: 30, 50, 100)** 설정하면,
    
    - 덜 관련된, 학습에 별로 도움 안 되는 쉬운 negative들이 섞이게 되고
    - 모델이 **혼란스럽거나 덜 효과적인 학습**을 하게 되어 **성능이 오히려 떨어짐**

---

#### 🥇 4.3.3. Mining 방법 최종 비교 (Base: e5-large-unsupervised, 334M)

| Mining Method | 요약 | 결과 |
| --- | --- | --- |
| **Naive Top-K** | FN 제거 안 함 | 가장 낮은 성능 |
| **Top-K Shifted** | 상위 FN 제거 | 성능 증가 |
| **TopK-Abs** | 절대 점수 기준 필터링 | 무난한 성능 |
| **TopK-MarginPos** | Positive 점수 기반 margin | 🔼 상위권 |
| **TopK-PercPos** | Positive × 퍼센트 (95%) | ✅ **최고 성능** |

→ **결론:** Positive-aware 방식이 **FN(False Negative)을 효율적으로 제거해 가장 효과적인 contrastive 학습을 유도**

<img src="https://github.com/user-attachments/assets/81e59e33-789a-4fd1-9e33-9ba691f38c9e" width="600"/>


---

#### 💪 4.3.4. 대형 모델(Mistral-7B-v0.1, 7.1B)로 일반화 실험

- 각 쿼리당 **hard-negative 1개**만 사용 (메모리 제한)
- 앞서 찾은 best config들 그대로 사용
- **TopK-PercPos**가 Mistral에서도 **최고 성능**

<img src="https://github.com/user-attachments/assets/6e203c1b-ce72-451f-ac6a-abea0e743c1d" width="800"/>

---

#### 4.4 RQ3b: NV-Retriever-v1 규모로 확장 실험

#### 실험 조건

- Base 모델: **Mistral-7B-v0.1**
- Teacher 모델: **E5-Mistral-7B**
- Train set: 15개 BEIR Retrieval 데이터셋, 총 **728,160 쿼리**
- 목적: NV-Retriever-v1과 **동일한 세팅으로 mining 방법 비교**

---

| Mining Method | Avg. NDCG@10 (15개 데이터셋) |
| --- | --- |
| **TopK-PercPos (95%)** | **60.55** → 거의 NV-Retriever-v1 수준 |
| **TopK-MarginPos** | 2개 데이터셋에서 best |
| **기타 방법들** | 모두 성능 열세 (FN 제거 부족) |
- NV-Retriever-v1 (60.9점)는 이후 classification/clustering 태스크까지 추가 학습한 결과
- 즉, **TopK-PercPos로만 학습한 모델이 거의 동등한 성능**을 보임

---

## 5. 결론

| 항목 | 요약 |
| --- | --- |
| 🎁 기여 | Positive-aware hard-negative mining 방법 제안 (TopK-MarginPos, TopK-PercPos) |
| 🧪 실험 | Ablation + Teacher 모델 다양화 + Ensemble + 대형 모델 적용 + MTEB 전체 평가 |
| 🧠 발견 | 모델 성능은 mining 방식에 매우 민감하며, false negative를 제거하면 성능이 확연히 향상됨 |
| 🔬 확장성 | 이 방법은 **multi-modal 모델 (예: 이미지+텍스트)**에도 효과적일 가능성 있음 |

---

## 🔚 이 논문이 말하는 궁극적인 메시지

> 🔎 “좋은 임베딩 모델은 단순히 좋은 모델 구조만으로 만들어지지 않는다.
> 
> 
> 학습에 쓰이는 **Negative 샘플의 ‘질’**이 성능을 결정짓는 핵심이며,
> 
> 이를 개선하는 **Positive-aware mining**은 SOTA를 달성하는 데 필요한 ‘숨은 비결’이다.”
>
