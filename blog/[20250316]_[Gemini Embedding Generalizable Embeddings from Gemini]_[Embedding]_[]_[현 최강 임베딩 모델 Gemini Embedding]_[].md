# 📌 Gemini Embedding

https://arxiv.org/html/2503.07891v1

✅ **Gemini LLM 기반 강력한 임베딩 모델 구축**

- 기존 대형 언어 모델(LLM)의 방대한 지식을 활용하여 성능을 극대화
- Mean Pooling + 선형 변환을 통해 최종 임베딩 벡터 생성

✅ **효율적인 학습 기법 적용**

- **Pre-finetuning + Fine-tuning + Model Soup(모델 앙상블) 기법 활용**
- 다양한 데이터 혼합 전략 및 하드 네거티브 샘플링 기법 도입

✅ **Gemini LLM을 활용한 데이터 품질 향상**

- **합성 데이터 생성(Synthetic Data Generation)** → 검색/분류 데이터 품질 향상
- **데이터 필터링(Data Filtering)** → 잘못된 Positive/Negative 예제 제거
- **하드 네거티브 샘플링(Hard Negative Mining)** → 검색 성능 개선

---

## **주요 기여점:**

1. Gemini LLM 기반 임베딩 모델
    - Google의 최신 대형 언어 모델(LLM)인 Gemini를 활용하여 강력한 텍스트 임베딩 모델을 개발함.
    - 다국어 및 코드 이해 능력을 활용해 다양한 언어 및 텍스트 모달리티에서 우수한 일반화 성능을 보임.
    
2. 다양한 다운스트림 작업 적용 가능
    - 사전 계산된 임베딩을 사용하여 분류(Classification), 유사성 측정(Similarity), 군집화(Clustering), 랭킹(Ranking), 정보 검색(Retrieval) 등 다양한 작업 수행 가능.

3. 대규모 벤치마크에서 최고 성능 기록
    - Massive Multilingual Text Embedding Benchmark (MMTEB)에서 기존 SOTA(State-of-the-Art) 모델을 뛰어넘는 성능을 달성.
    - **250개 이상의 언어, 100개 이상의 평가 작업**에서 실험을 진행하며 강력한 임베딩 품질을 입증함.

4. LLM을 활용한 데이터 정제 및 학습 최적화
    - Gemini를 활용하여 저품질 데이터를 필터링하고, 검색을 위한 긍정적(Positive) 및 부정적(Negative) 예제 샘플을 선정하며, 풍부한 합성 데이터를 생성하여 학습 데이터의 질을 높임.
    - Contrastive Learning(대조 학습) 기법을 사용하여 더 나은 의미적 표현을 학습.

5. 모델 성능 향상을 위한 추가 기법 적용
    - **Task Prompting 및 Pre-Finetuning** 단계 추가: 기존 Gecko 모델의 성공을 기반으로 더욱 성능을 높임.
    - **Model Soup 기법 활용**: 여러 개의 Fine-tuned 체크포인트를 평균화(ensemble)하여 최종 성능을 향상.

6. 광범위한 평가 및 최고 성능 달성
    - **MMTEB 벤치마크에서 1위** (Borda rank 기준).
    - 평균 점수 68.32로 기존 최고 모델(multilingual-e5-large-instruct)보다 **+5.09 개선**.
    - 다양한 작업에서 최고 점수 기록(XOR-Retrieve, MTEB 등).
    - **고자원(High-resource) 언어뿐만 아니라 저자원(Low-resource) 언어에서도 강력한 성능**을 보임(예: 마케도니아어).

---

### **논문의 주요 기술적 요소**

1. **LLM 기반 데이터 정제**
    - Gemini LLM을 이용해 데이터 품질을 높이고, 노이즈를 제거함.
    - 하드 네거티브 샘플링 및 합성 데이터 생성 기법 활용.
2. **Contrastive Learning (대조 학습) 기반 학습**
    - 긍정적/부정적 예제를 구분하면서 더 정교한 의미적 임베딩을 학습함.
3. **Model Soup (모델 체크포인트 앙상블 기법) 적용**
    - 여러 Fine-tuned 모델을 평균화하여 최적의 임베딩 성능을 도출.

---

## **📌 3.1 모델 아키텍처**

### **✅ 모델의 주요 구조**

1. **Gemini LLM에서 초기화**
    - Gemini의 **방대한 지식과 언어 이해 능력**을 그대로 가져와 초기 임베딩 모델을 구성함.
    - 즉, 기존 LLM을 기반으로 하는 **사전 학습(pre-training)** 단계로 볼 수 있음.
2. **임베딩 풀링 (Pooling)**
    - 전체 시퀀스를 하나의 벡터로 압축하기 위해 **Mean Pooling** 사용.
    → 즉, 토큰 임베딩들의 **평균값을 계산**하여 최종 문장 임베딩 생성
    - 참고: Mean Pooling은 단순하지만 **일반화 성능이 뛰어나다**는 연구 결과(Suganthan et al., 2025)에 근거함.
3. **최종 선형 변환 (Linear Projection)**

✅ **요약:**

Gemini LLM을 기반으로 Transformer가 토큰을 임베딩 벡터로 변환 → Mean Pooling으로 문장 벡터를 생성 → 최종적으로 선형 변환을 거쳐 원하는 차원의 벡터로 출력.

---

## **📌 3.2 학습 목표 (Training Objective)**

Gemini Embedding 모델은 **Noise-Contrastive Estimation (NCE) Loss**를 사용하여 학습됩니다.

이 방식은 **"좋은 임베딩은 비슷한 문장을 가깝게, 다른 문장을 멀리 배치해야 한다"**는 개념을 학습하는 방법입니다.

### **✅ 학습 데이터 구성**

- 하나의 학습 샘플은 다음과 같이 구성됨:
    - **Query**
    - **Positive Target**
    - **Hard Negative Target**
    - **Task String** (작업 유형을 나타내는 문자열, 예: "질문 답변" 또는 "사실 검증")
- 해당 데이터를 임베딩 벡터로 변환하는 과정
    - **Mean Pooling 적용** 후 **선형 변환을 거쳐 최종 임베딩 벡터 생성.**

### **✅ NCE 손실 함수**

손실 함수는 **코사인 유사도(Cosine Similarity)** 를 사용하여 정의됨.

✅ **요약:**

**NCE Loss를 사용해 "올바른 문장을 더 가깝게, 잘못된 문장을 멀리 배치"하는 방식으로 학습하며, 다양한 차원에서도 성능을 유지하도록 MRL 기법을 적용.**

---

## **📌 3.3 학습 과정 (Recipe)**

**Gemini Embedding 모델은 3단계로 학습**됩니다.

### **✅ 1. Pre-Finetuning (사전 미세 조정)**

- 초기에는 **노이즈가 포함된 대량의 (query, target) 쌍 데이터**를 사용하여 학습.
- **Hard Negative를 포함하지 않음.**
- **대규모 배치(batch size 증가)** 를 사용하여 학습 안정성을 높임.
- 목표: **Autoregressive LLM(Gemini) → Encoding 중심 모델로 변환.**

### **✅ 2. Fine-tuning (본격적인 미세 조정)**

- **작업(task)-특화 데이터셋을 사용**하여 훈련.
- **(query, target, hard negative) 삼중 샘플을 포함**.
- **작은 배치 크기 사용 (1024 이하)**
→ 같은 배치 내에서 같은 작업(Task)만 포함하도록 제한하여 의미적 구별을 쉽게 함.
- **최적의 하이퍼파라미터 탐색** 수행.

### **✅ 3. Model Soup (모델 앙상블)**

- **Fine-tuned 모델들의 체크포인트를 평균화하여 최적 성능을 달성.**
- 체크포인트들을 단순 평균하거나 가중 평균하여 더 일반화된 성능을 얻음.

✅ **요약:**

1. **Pre-Finetuning:** LLM을 임베딩 모델로 전환 (하드 네거티브 미포함, 대규모 배치).
2. **Fine-Tuning:** Hard Negative 포함, Task-Specific 데이터셋 사용, 최적의 하이퍼파라미터 탐색.
3. **Model Soup:** 여러 Fine-tuned 모델을 평균화하여 최종 모델 생성.

---

## **🔹 4. 데이터셋 및 품질 향상 기법**

Gemini Embedding 모델은 **다양한 다국어 임베딩 작업 및 코드 검색(Code Retrieval) 작업**을 학습하기 위해 여러 데이터셋을 혼합하여 사용합니다.

또한, Gemini LLM을 활용하여 **① 합성 데이터 생성, ② 데이터 필터링, ③ 하드 네거티브 샘플링**을 수행하여 데이터 품질을 높였습니다.

---

## **📌 4.1 학습 데이터 구성 (Training Data Mixture)**

### **✅ Pre-Finetuning 데이터**

- 목표: **다양한 데이터셋을 모델이 최대한 많이 접하도록 함** (일반화 성능 극대화).
- **대규모 웹 코퍼스(웹 문서 데이터)** 를 사용.
    - **제목(Title)과 본문(Passage) 쌍**을 **(Query, Positive Target) 데이터**로 활용.
    - 단순한 기법이지만 **임베딩 모델 성능 향상에 효과적**임 (Neelakantan et al., 2022; Lee et al., 2024 참고).

### **✅ Fine-Tuning 데이터**

- **3가지 목적에 맞춘 데이터 구성:**
    1. **Task 다양성(Task Diversity):**
        - 기존 Gecko 모델(Lee et al., 2024)에서 사용된 학술 데이터셋 일부 포함.
        - 합성 데이터(Synthetic Data) 추가 활용.
    2. **언어 다양성(Language Diversity):**
        - MTEB(Massive Text Embedding Benchmark)와 달리 **특정 도메인 데이터는 제외**.
        - 이유: 기존 MTEB 데이터셋은 특정 테스트 셋에서만 성능이 높아지는 **데이터 편향(Data Bias) 문제**가 있음.
    3. **코딩 데이터(Coding Capability):**
        - 코드 검색(Code Retrieval) 관련 작업을 포함하여 코드 기반 임베딩 모델 성능도 높임.
- **데이터 혼합 비율(Tuning Mixture Rate) 결정 방법:**
    - **Grid Search**(격자 탐색) 기법을 사용하여 **각 데이터셋별 최적의 훈련 스텝 수**를 찾음.

✅ **요약:**

**Pre-Finetuning에서는 대규모 웹 데이터를 사용하고, Fine-Tuning에서는 작업별 최적화된 데이터셋을 구성하여 학습.**

특히, 기존 MTEB 모델들이 가진 **데이터 편향 문제를 해결**하려고 함.

---

## **📌 4.2 Gemini를 활용한 데이터 품질 향상 (Improving Data Quality with Gemini)**

### **✅ 1. 합성 데이터 생성 (Synthetic Data Generation)**

- 최근 벤치마크(MMTEB, Enevoldsen et al., 2025)는 검색(Retrieval) 외에도 다양한 작업을 포함하고 있음.
- 이를 반영하기 위해 **검색(Retrieval) 및 분류(Classification) 작업에 대한 합성 데이터(Synthetic Data)를 생성**.

🔹 **검색(Retrieval) 작업을 위한 합성 데이터**

- 기존 연구(FRet, SWIM-IR)에서 사용된 **합성 쿼리 생성 방법**을 개선.
- **Gemini LLM을 활용한 Few-Shot Prompting**으로 웹 문서에 대한 합성 쿼리를 생성.
- 생성된 데이터를 **Gemini Auto-Rater**가 평가하여 **비현실적인 검색 쿼리(품질이 낮은 데이터)를 필터링**.

🔹 **분류(Classification) 작업을 위한 합성 데이터**

- **Counterfactual (반사실적)**, **Sentiment (감성 분석)**, **Review Classification (리뷰 분류)** 등의 **영어 데이터셋 생성**.
- 품질 향상을 위해 **"다단계 프롬프팅(Multi-Stage Prompting)"** 기법을 활용:

✅ **요약:**

**Gemini LLM을 활용해 검색 및 분류 작업을 위한 데이터를 직접 생성하며, 품질 평가 모델(Auto-Rater)로 정제.**

---

### **✅ 2. 데이터 필터링 (Data Filtering)**

- 학습 데이터셋에는 **사람이 주석(Annotation)한 데이터도 포함됨**.
- 문제점: 검색(Retrieval) 데이터셋에는 **잘못된 Positive(긍정) 또는 Negative(부정) 타겟이 포함될 가능성**이 있음.
- **Gemini LLM을 이용한 Few-Shot Prompting 기법**을 사용하여 **품질이 낮은 데이터를 제거**.

✅ **요약:**

**Gemini LLM으로 잘못된 라벨링 데이터를 자동 필터링하여 학습 데이터 품질을 높임.**

---

### **✅ 3. 하드 네거티브 샘플링 (Hard Negative Mining)**

- **하드 네거티브 샘플링 방법**
    1. **Gemini 초기화 임베딩 모델을 먼저 학습.**
    2. 이 모델을 이용해 **각 쿼리(Query)에 대한 가장 가까운 Top-k 후보 문서(Nearest Neighbors) 검색.**
    3. **Gemini LLM을 이용하여 후보 문서의 점수 계산.**
    4. 두 가지 프롬프팅 기법 사용:
        - **Graded Classification:** 문서가 얼마나 관련 있는지 등급을 매김.
        - **Query Likelihood:** 해당 문장이 실제 검색 질의(Query)로 얼마나 가능성이 높은지 평가.
    5. 두 점수를 **Reciprocal Rank Fusion (RRF)** 기법으로 결합하여 최적의 Hard Negative 선택.
    6. **가장 낮은 점수를 받은 후보 문서(k번째 문서)가 최적의 하드 네거티브로 선택됨.**

✅ **요약:**

**Gemini LLM을 활용해 최적의 Hard Negative를 선택하여 검색 모델의 일반화 성능을 향상.**

---

## **🔹 6. Ablation Study (성능 분석 연구)**

이 섹션에서는 **Gemini Embedding 모델이 어떻게 강력한 성능을 달성했는지**를 심층 분석합니다.

특히, **다국어 일반화 능력**, **데이터 품질 향상 기법의 효과**, 그리고 **하드 네거티브 샘플링의 영향**을 살펴봅니다.

---

## **📌 6.1 다국어 일반화 성능 (Does Gemini Embedding Generalize to Multilingual Tasks?)**

🔹 **실험 목적:**

- Gemini Embedding이 **여러 언어와 다양한 작업에서 일반화(Generalization)할 수 있는지** 확인.
- **Fine-tuning 전후의 성능 차이**를 분석하여 **Pre-finetuning과 Fine-tuning의 영향력**을 평가.

🔹 **실험 결과:**

1. **Pre-finetuning 만으로도 성능 크게 향상**
    - Fine-tuning 없이 Pre-finetuning만 수행해도 **다국어(Multilingual) 벤치마크에서 성능이 크게 향상**됨.
    - 이는 **Gemini LLM의 기존 지식이 강력한 역할을 한다는 증거**.
2. **영어(English-only) 데이터로 학습해도 다국어 성능이 강함**
    - Fine-tuning을 **영어 데이터만으로 진행**해도 **MTEB(Multilingual) 벤치마크에서 높은 성능**을 달성.
    - **XTREME-UP 벤치마크에서 기존 임베딩 모델을 초월하는 성능**을 보임.
    - **➡ Gemini Embedding은 단일 언어(영어) 데이터만 사용해도 다국어 일반화 능력이 뛰어남**.
3. **언어 다양성보다 작업(Task) 다양성이 Fine-tuning 성능에 더 중요함**
    - **다국어(Multilingual-only) 데이터셋으로만 학습**한 모델은 성능이 떨어짐.
    - 이유: **검색(Retrieval) 데이터만 포함**하고, **분류(Classification) 같은 다양한 작업을 포함하지 않았기 때문**.
    - **➡ Fine-tuning 시 언어보다 작업(Task)의 다양성이 성능 향상에 더 중요**.

✅ **요약:**

**Gemini Embedding은 영어 데이터로만 학습해도 다국어에서 강력한 성능을 보이며, Fine-tuning에서는 언어 다양성보다 작업(Task) 다양성이 더 중요하다.**

---

## **📌 6.2 Gemini가 데이터 품질을 어떻게 개선하는가? (How Does Gemini Improve Data Quality?)**

이 실험에서는 **Gemini LLM을 활용한 데이터 품질 개선 기법(합성 데이터 생성, 데이터 필터링, 하드 네거티브 샘플링)의 효과**를 분석합니다.

---

### **✅ 1. 합성 데이터 생성 (Synthetic Data Generation)**

🔹 **실험 목적:**

- **Gemini LLM을 활용한 합성 데이터(Synthetic Data)가 실제 데이터와 비교해 성능을 개선하는지 확인.**
- 특히, **분류(Classification) 작업에서 효과적인지 평가**.

🔹 **실험 결과:**

1. **Gemini LLM이 생성한 합성 데이터만으로도 강력한 성능을 달성**
    - **실제 데이터 없이(zero-shot) 합성 데이터만 사용**해도 기존 데이터셋을 학습한 모델과 비슷한 성능을 달성.
    - 예: Gecko 모델(AmazonPolarity 데이터셋 사용)과 **동등한 성능**.
2. **Multi-stage prompting 전략이 데이터 품질을 향상시킴**
    - **"다단계 프롬프팅(Multi-stage prompting)"** 전략 덕분에 더 **현실적이고 다채로운 데이터 생성 가능**.
    - 예: 사용자, 제품, 영화 등의 정보를 계층적으로 생성하여 보다 **다양한 리뷰 데이터셋 구축 가능**.
3. **합성 데이터는 기존 데이터보다 편향(Bias)이 적을 가능성이 있음**
    - 합성 데이터는 **실제 데이터셋보다 특정 편향(Bias)이 줄어들 가능성**이 있음.
    - → **데이터 편향을 줄이면서도 성능을 유지하는 새로운 가능성 제시**.

✅ **요약:**

**Gemini LLM이 생성한 합성 데이터는 실제 데이터와 유사한 성능을 보이며, Multi-stage prompting 기법을 활용해 품질을 개선하고 데이터 편향을 줄이는 데 기여할 수 있다.**

---

### **✅ 2. 데이터 필터링 (Data Filtering)**

🔹 **실험 목적:**

- Gemini LLM을 이용하여 **검색(Retrieval) 데이터셋의 오류를 제거했을 때 성능이 향상되는지 평가**.

🔹 **실험 방법:**

- *MIRACL 데이터셋(18개 언어의 검색 데이터)**을 **Gemini LLM으로 필터링 후 학습**.
- 필터링 전후 모델의 성능 비교.

🔹 **실험 결과:**

1. **데이터 필터링 후 성능이 전반적으로 향상됨.**
    - 대부분의 언어에서 **성능이 향상되었으며, 일부 언어에서는 소폭 감소**.
    - 이는 잘못된 Positive/Negative 데이터를 제거하여 **학습 데이터 품질이 향상된 결과**.
2. **영어 데이터 혼합이 다국어 성능을 더 높이는 데 기여**
    - 다국어 데이터만 사용하는 것보다 **영어 데이터를 혼합하는 것이 더 나은 성능을 보임**.
    - → **Gemini Embedding 모델은 영어 기반 학습이 다국어 확장에도 긍정적인 영향을 미침**.

✅ **요약:**

**Gemini LLM을 활용한 데이터 필터링은 검색 데이터셋의 오류를 제거하여 성능을 향상시키며, 영어 데이터의 혼합이 다국어 모델 성능에도 긍정적인 영향을 미친다.**

---

### **✅ 3. 하드 네거티브 샘플링 (Hard Negative Mining)**

🔹 **실험 목적:**

- Gemini LLM이 선정한 하드 네거티브 샘플이 **검색(Retrieval) 성능 향상에 얼마나 기여하는지 분석**.

🔹 **실험 결과:**

1. **하드 네거티브 샘플링이 검색 성능을 향상시킴**
    - **4개 데이터셋에서 일관되게 검색 성능이 향상됨**.
    - 하드 네거티브 샘플을 포함할수록 모델이 **더 정밀한 의미 구분 학습 가능**.
2. **그러나 하드 네거티브가 너무 많으면 과적합(Overfitting) 발생**
    - 하드 네거티브 샘플이 **너무 많으면 오히려 성능이 감소**.
    - 이유: 모델이 **너무 어려운 예제만 학습하면서 일반화 성능이 저하됨**.
    - **➡ 적절한 하드 네거티브 샘플 수 조절이 필요함.**
3. **향후 연구 방향: Regularization 기법 탐색**
    - **과적합 방지를 위한 정규화(Regularization) 기법 연구 필요**.

✅ **요약:**

**하드 네거티브 샘플링은 검색 성능을 향상시키지만, 과도한 샘플링은 오히려 성능 저하를 유발할 수 있어 적절한 균형이 필요하다.**

---

# 질문사항

## 질문 1

**하드네거티브 선택 시 : 가장 높은 점수가 아닌 낮은 점수를 받은 후보 문서(k번째 문서)가 최적의 하드 네거티브로 선택한 이유**

## **🔹 Hard Negative 샘플링의 균형 문제**

### **1. 너무 쉬운 네거티브 (Easy Negative)**

- 완전히 무관한 문장을 Negative로 사용하면 모델이 쉽게 구별할 수 있음.
- 학습 효과가 적고, 실제 검색 태스크에서 유용한 임베딩을 학습하기 어려움.
- **→ 효과적인 학습을 위해서는 더 "어려운" 네거티브가 필요함.**

### **2. 너무 어려운 네거티브 (Hardest Negative)**

- Hard Negative가 지나치게 정답(Positive)과 유사하면, 모델이 혼란스러워질 수 있음.
- 예를 들어, 질문 "Who discovered America?"에 대한 정답이 *"Christopher Columbus"* 라면,
    - **적절한 Hard Negative**: *"Vikings were the first Europeans to reach America."*
    - **너무 어려운 Hard Negative**: *"Christopher Columbus landed in the Caribbean in 1492."*
    - → 후자는 사실상 정답과 너무 비슷하여 모델이 잘못된 신호를 학습할 가능성이 있음.
- **→ 과적합(Overfitting) 문제 발생 가능**.
    - 모델이 **"진짜 정답과 구별할 수 없는 문장"을 무리하게 학습**하려고 하면, 일반화 성능이 저하됨.

---

## **🔹 논문의 선택: 가장 낮은 점수(=적당히 어려운) 후보를 하드 네거티브로 활용**

논문에서는 **Gemini LLM을 활용하여 Nearest Neighbors를 찾고, 그중 가장 낮은 점수를 받은 후보를 Hard Negative로 선택**하는 방식을 채택했습니다.

즉, **"너무 쉬운(Easy) 네거티브는 배제하고, 너무 어려운(Hardest) 네거티브도 피하는 전략"**입니다.

### **🔑 왜 가장 낮은 점수를 받은 문서를 선택했을까?**

1. **"적당히 어려운(Moderate Hard) 네거티브"가 학습에 가장 효과적**
    - 모델이 충분히 고민하면서도 학습 가능한 수준의 네거티브를 선택.
    - 지나치게 어려운 Hard Negative는 잘못된 신호를 줄 위험이 있음.
2. **일반화 성능(Generalization)을 높이기 위해**
    - Retrieval 모델의 목표는 **훈련 데이터가 아닌 새로운 데이터에서도 성능을 발휘하는 것**.
    - 너무 어려운 Hard Negative를 학습하면 훈련 데이터에는 최적화되지만, 새로운 데이터에서는 성능이 낮아질 수 있음.
3. **과적합(Overfitting) 방지**
    - Hard Negative가 지나치게 어렵다면 모델이 특정한 데이터를 외우려는 경향을 보일 수 있음.
    - 논문에서도 **하드 네거티브가 너무 많으면 오히려 성능이 떨어진다**고 언급함.

---

## **🔹 결론**

하드 네거티브는 **적절한 난이도의 샘플을 선택하는 것이 중요**합니다.

논문에서 **가장 낮은 점수를 받은 문서를 하드 네거티브로 선택한 이유는 "적당히 어려운 네거티브"를 찾아서 학습 효과를 극대화하면서도 일반화 성능을 유지하기 위해서입니다.**

즉, **너무 쉬운 것도 안 되고, 너무 어려운 것도 안 되는 균형점을 찾은 전략**이라고 볼 수 있습니다.
