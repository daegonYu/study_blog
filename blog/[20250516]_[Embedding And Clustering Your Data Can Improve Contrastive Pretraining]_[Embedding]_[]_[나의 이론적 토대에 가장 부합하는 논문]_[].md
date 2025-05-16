# Embedding And Clustering Your Data Can Improve Contrastive Pretraining
https://arxiv.org/html/2407.18887v1

# Abstract

> “데이터를 의미적으로 클러스터링한 뒤 contrastive 학습을 하면 성능이 좋아진다”
> 

**🔍 기존 배경**

- 대규모 텍스트 임베딩 학습에서는 보통 **여러 소스(source)**에서 가져온 데이터를 섞어 학습 → in-batch negatives 가 random하게 구성
- 하지만 최근 연구들 (예: TAS-B)에서 보니 Topic, source **단위로 묶어서 학습**하면 더 좋은 성능

**🆕 이 논문의 제안**

- **사전 학습된 임베딩 모델로 임베딩 → k-means 클러스터링 → 각 클러스터를 하나의 배치 단위로 사용**

---

**📈 실험: MSMARCO 기반 성능 향상**

- BERT 기반 임베딩 모델을 MSMARCO query-passage pair로 contrastive 학습

---

**🔗 다른 방법과의 연결**

- **TAS-B**: topic-aware sampling → 이 논문도 토픽 기반 샘플링의 일종 (cluster ≈ topic)
- **ANCE**: nearest neighbor 기반 하드네거티브 채굴 → 비슷한 개념의 ‘의미 기반 군집’
- 따라서 이 논문은 **TAS-B와 ANCE를 하나의 틀에서 바라봄**

---

**🚀 이 논문의 실질적 의의**

- 단순한 방법 (사전 임베딩 + KMeans)으로 **contrastive 학습 효율을 높일 수 있음**
- 학습 데이터를 어떻게 조직화하느냐도 중요한 연구 영역이라는 점을 **강조**

# 1. Introduction

![image](https://github.com/user-attachments/assets/7c997135-a466-4a78-803f-9695397ab9ed)


보라색 : 랜덤으로 미니 배치 구성

**🔍 배경: 왜 source 단위로 묶는 게 좋은가?**

- **Arctic Embed** 보고서 (Merrick et al., 2024)는 contrastive 학습 시,
    - 데이터를 **여러 출처(source)**에서 무작위로 섞는 것보다
    - **한 source로만 구성된 minibatch**로 학습하는 방식이 **retrieval 성능을 향상**시킨다는 것을 실험으로 보여줌
- **Nomic-Embed** 보고서도 같은 전략을 사용하며, 이 방식이 모델이 “source-specific shortcut”을 배우는 것을 막아준다고 주장함
    
    > 예: 특정 source의 스타일이나 패턴만 보고 맞추는 단순 학습을 막음
    > 
    
    source 단위로 minibatch 구성 시  source의 스타일이나 패턴으로 Anchor와 Positive를 구분하지 못하며 따라서 의미 기반의 유사도 학습이 유도됨
    
---

**🆕 이 논문의 제안: "semantic sub-source"**

- 단순히 source 단위로만 데이터를 나누는 게 아니라,
- 각 source 내부의 데이터를 **pretrained embedding으로 임베딩 → k-means로 클러스터링**
    
    → 즉, **“의미 기반 하위 source(semantic sub-sources)”**를 만들어서 더 미세한 stratification 적용
    
- **결과**: MSMARCO 실험에서 NDCG@10이 향상됨 → 효과 입증

---

**🔗 기존 연구와의 연결**

- 이 방법은 **TAS (Topic Aware Sampling)**, **ANCE (Hard Negative Mining)**의 핵심 아이디어와 이론적으로 연결됨
- 완전한 이론 정립은 아니지만, **contrastive pretraining을 개선하는 통합적 관점**을 제시하려고 함

---

**💡 Figure 1 핵심 해석**

- SciDocs 실험에서, 단순히 batch size만 키운 것(purple)은 성능이 부족
- 반면, **stratified minibatch (source별로 나눈 것)**는 성능이 더 좋았고
- *batch size 증가 + stratification 병행(light blue)**이 가장 좋은 성능
- 즉, **좋은 데이터 구성 = 학습 품질 향상** → 일종의 *커리큘럼 러닝* 효과도 있음 → 이 논문에서 처음부터 Clustered dataset을 minibatch로 구성하면 처음에 수렴이 잘 안된다고 함. 따라서 처음에는 난이도가 쉬운 랜덤 혹은 낮은 cluster K 로 구성 → 학습이 진행될수록 cluster K를 키워서 cluster 세분화 및 학습 난이도를 높여서 학습하는 방식을 제안

# 2. Methodology

**💡 핵심 아이디어**

> "기존에는 source 단위로만 나눴다면, 이제는 의미(semantic) 단위로 쪼개서 contrastive 학습에 사용하자."
> 

---

**⚙️ 전체 프로세스: 4단계로 요약**

1. **입력 준비**
    - **N개의 쿼리-아이템 쌍**이 있는 대규모 contrastive 학습 데이터셋
    - 그리고 하나의 **사전 학습된 텍스트 임베딩 모델**
2. **임베딩 생성**
3. **k-means 클러스터링**
    - 위의 임베딩 벡터들을 대상으로 **k-means** 알고리즘을 적용
    - 각 임베딩이 **k개의 클러스터 중 하나에 할당**됨
4. **데이터 분할 및 stratified 학습**
    - 원래 query-passage 쌍들을 각 passage가 속한 **클러스터 기준으로 나눔**
    - contrastive 학습 시 minibatch는 **각 클러스터 안에서 샘플링**
        
        → 이전의 "source 단위" stratification을 **"cluster 단위"**로 대체
        

본 논문은 **pretrained 임베딩 + clustering**을 이용해서 **암묵적 topic (= semantic cluster)**를 만들어냄

# 3. Experiments — 핵심 실험 및 결과 요약

**🧪 실험 목적**

> 실제로 “의미 기반 클러스터링”을 적용하면 contrastive 학습 성능이 향상되는지 검증한다.
> 

---

📌 **3.1 Clustering Details — 실험 설정 및 클러스터 통계**

**📂 사용 데이터**

- **MSMARCO passage dataset**
    - 원래는 쿼리 + 정답 passage + hard negative가 있음
    - 실험에서는 **정답 쿼리-패시지 쌍만 사용** (hard negative 제거)
    - → 약 50만 쌍만으로 학습 (엄밀히 말하면 “대규모”는 아님)

**🧠 클러스터링 방식**

- 임베딩 모델: **Arctic Embed M** (Merrick et al., 2024)
- 클러스터링 알고리즘: **Spherical k-means** (from FAISS)
- 클러스터 수 k=10

**🧪 임베딩 대상**

- 실험 1: **passage 임베딩 → 클러스터링**
- 실험 2: **query 임베딩 → 클러스터링**

**📈 클러스터 통계**

- Passage 클러스터의 평균 유사도 (코사인): **0.26 ~ 0.49**
- Query 클러스터의 평균 유사도: **0.63 ~ 0.76**
- → Query는 더 강하게 응집된 클러스터, Passage는 더 흩어짐

---

**3.2 Training — 학습 설정**

- 사용 모델: **BERT base**, [CLS] 토큰 기반 임베딩 사용
- Stratification 방식:
    - **원본(Mixed)**: 전체 데이터를 섞어서 학습
    - **Stratified by cluster**: 같은 클러스터 내 샘플끼리만 배치 구성

![image](https://github.com/user-attachments/assets/1d0c7a38-7433-46b6-a2d5-7ce22333310b)


**📉 Figure 2: 학습 손실 곡선**

- Cluster-stratified 방식이 **더 높은 평균 손실(loss)**
    
    → 즉, **더 어려운 contrastive 학습이 이루어짐** (Hard negative 많음)
    
- 손실의 **변동성도 큼**
    
    → 클러스터별 난이도 차이로 minibatch 난이도가 들쭉날쭉
    
---

**3.3 Results — 성능 결과**

**✅ 평가 지표**

- **MTEB Retrieval benchmark** 전체 + **MSMARCO dev split (NDCG@10)**

**🎯 MSMARCO 평가 결과**

- passage 또는 query 클러스터 기반 모두:
    - **NDCG@10이 약 2% 향상** → 클러스터링의 효과 확인

**🌐 전체 MTEB 성능**

- **15개 Retrieval 태스크 중 11개에서 성능 향상**
- 하지만 일부 도메인(ClimateFEVER, SciDocs 등)에서는 **성능 하락**
- 반면, **FiQA**와 같은 도메인에서는 **큰 향상** (→ 클러스터와 도메인 유사성이 있다고 유추)

**💬 저자 해석**

- 일부 클러스터가 특정 도메인(FiQA 등)과 **유사한 분포**를 가지기 때문에 성능이 향상되었을 것이라고 **추정**

**💡 핵심 통찰**

- 단순한 stratification이 아니라, **의미 기반 클러스터로 구성한 배치가 contrastive 학습에 더 적합**
- 특정 도메인에 가까운 클러스터가 존재할 경우 **도메인 적응력도 자연스럽게 향상**

# 4. In Search Of Deeper Understanding

> 실험 결과의 이론적 배경을 설명하며, TAS-B와 ANCE로 대표되는 기존 기법과의 연관성에서 이 논문의 클러스터링 기법을 해석
> 

---

**🔹 4.1 Concept One: Clustering Hypothesis & Topic-Aware Sampling (TAS)**

**📌 핵심 메시지**

> TAS-B의 아이디어를 확장하여, labeled-negative 없이도 large-scale pretraining에 적용 가능한 방식으로 바꾼 것이 이 논문의 클러스터링 기법이다. → 적용하기 쉽다!
> 

**📘 TAS-B (Hofstätter et al., 2021)**

- TAS는 **query 임베딩 기반 clustering** 후, 한 클러스터에서만 배치 샘플링 → **topical coherence** 유지
- 단점: **query당 여러 개의 labeled negatives**가 필요한 세팅에만 적용 가능 → pretraining용 대규모 쌍 데이터셋에는 부적합

**🧠 이 논문의 클러스터링은 TAS의 일반화**

- TAS처럼 **의미 기반 클러스터 구성** 후, 같은 클러스터에서만 샘플링
- labeled negative 없이도 사용 가능 → **pretraining에 적합한 TAS 스타일**

**🧪 Cluster Hypothesis (Jardine & van Rijsbergen, 1971)**

- 완전히 다른 topic끼리 묶은 negative는 너무 쉬움 (easy negatives)
- → 그래서 **같은 topic 내에서의 negative (harder)**가 더 유익함

---

**🔹 4.2 Concept Two: ANCE Perspective on Hard Negative Mining**

**📘 ANCE (Xiong et al., 2020)의 방식**

1. 모델로 전체 데이터를 임베딩
2. ANN(Approximate Nearest Neighbor) 인덱스로 query에 가장 유사한 negative를 뽑음
3. 이 **hard negative**를 포함해 배치를 구성 → 학습 효과 극대화

**Arctic Embed 보고서와 연결**

- **Arctic Embed**도 ANCE와 유사하게, 사전 임베딩으로 hard negative를 뽑아 fine-tuning
- 단, 이 논문에서는 **고정된 임베딩으로 1회만 mining**
- 또, **"너무 어려운 negative는 제외"** → 적절한 난이도 유지 (hard but not too hard) → 이것은 여러 논문에서 공통된 특징으로 나오는 부분이다.

**💡 쉽게 말해 요약**

> 무작위로 배치를 뽑으면 "다 다른 주제"라서 학습이 너무 쉬움.
> 
> 
> → 그러면 contrastive learning이 "구분"이 아니라 "구별만" 배우게 됨
> 
> 대신, **같은 주제(topic/클러스터) 안에서 학습**하면
> 
> 더 **미묘한 차이**를 구분하는 법을 배움 = **좋은 임베딩 모델로 발전**
> 

---

**🔹 4.3 A Possible Synthesis**

> “왜 클러스터 기반 배치 구성이 contrastive 학습에 효과적인가?”
> 
> 
> 이 질문에 대해 **TAS-B + ANCE + 기하학적 직관**을 결합하여 이론적으로 설명합니다.
> 

**🔷 두 가지 전제 (Assumptions)**

1. **모델이 잘 학습되어 easy negative는 대부분 걸러낸 상태이다.**
    - 즉, randomly sampled negative는 학습에 거의 기여하지 않음 → **학습 정체 발생**
2. **토픽이 잘 반영된 embedding 공간에서 같은 topic의 아이템들은 서로 가깝게 위치해 있다.**
    - 즉, **같은 topic 아이템들은 클러스터처럼 모여 있다**

**🔺 삼각부등식(triangle inequality)을 통한 직관**

Figure 3(a): 같은 토픽의 벡터들

- query와 positive가 서로 유사하고, 같은 topic 내 다른 item들도 가까이 있음
- 이 topic 안의 다른 문서들도 일정 수준 이상 query와 유사할 수밖에 없음 → **유효한 hard negative가 될 가능성↑**

Figure 3(b): 다른 토픽의 벡터들

- 어떤 negative가 query와 **멀리 떨어져 있다면**, 같은 클러스터 내 다른 문서들도 대부분 멀 수밖에 없음
- → **off-topic negative는 거의 항상 easy negative** → 배치에서 제외하는 게 좋음

**🧠 이론적 결론**

- **같은 topic 내에서 sampling 하면**, query와 의미는 다르지만 **위치가 가까운 적당한 negative (hard negative)**를 배치에 포함시킬 수 있음
- → 학습에 계속 기여할 수 있는 contrastive signal 제공 → 나의 이론과 동일하다.

---

**🔍 4.4 Reality Check**

**🔎 더 미세한 클러스터링? (예: TAS-B의 2,000개 클러스터)**

- 클러스터 밀도를 높일 수는 있지만, 이 논문은 **k=10처럼 적은 클러스터 수**에서도 효과를 보였음
- 즉, 이론과 실험의 괴리는 “모든 negative가 hard해야 한다”는 전제는 틀렸지만, **일부만 hard여도 학습은 충분히 이루어짐 → 모든 negative가 hard하면 내 경험상 오히려 성능이 낮아짐**

# 5. Limitations and Alternatives

> “클러스터링 기반 contrastive pretraining”이 항상 좋은 건 아니며,
> 
> 
> **언제, 어떻게** 쓰는지가 중요하다는 점을 지적
> 

---

**🔹 5.1 A Need For Curriculum Learning (점진적 학습 전략 필요)**

💡 문제점: 클러스터링이 **초기 학습 단계에서는 별 효과 없음**

- Figure 1에서도 보였듯이: 학습 초기 몇 백 step 동안은 **source-stratification의 효과가 거의 없음**
- 즉, 모델이 **easy negative를 구분할 줄도 모르는 상태**에서는,
    - clustering이 주는 “의미 기반 hard negative”조차도 너무 어려움 → 학습 비효율

**⚠️ 후반부에도 잠재적 문제가 있음**

- 모델이 잘 학습되어도, **클러스터가 너무 크면**:
    - batch 내에 **의미 있는 hard negative**가 실제로 포함되지 않을 위험이 있음
- 예: 클러스터 내 10,000개 중 hard negative는 20개뿐인데, batch로 32개만 뽑으면 못 뽑힐 수도 있음

✅ 제안: **Curriculum Learning**

> “학습 진행에 따라 데이터 구성 방식도 바뀌어야 한다”
> 
- 초기에는 랜덤 배치 or coarse source 기반 stratification
- 학습이 충분히 되면 → **클러스터 기반 구성**
- 마지막 단계에서는 → **정밀한 hard negative mining 방식으로 전환**

---

**🔹 5.2 Why Not Just Hard Negative Mine For Pretraining?**

> “차라리 처음부터 hard negative mining 하면 안 되나?”
> 
> 
> 이에 대한 저자의 답변은 **효율성과 비용의 문제**입니다.
> 

**🔧 Hard Negative Mining의 한계 (Pretraining 기준)**

- 좋은 contrastive 학습은 query당 **여러 개의 hard negative**가 필요함
- 즉, **query 하나를 학습시키려면:**
    - 1 positive + n negatives → 총 **n+1 개 문장을 임베딩**
    - → embedding 연산 비용이 **표준 contrastive 학습보다 훨씬 큼**
    - → 수억 쌍의 pretraining 데이터에는 **비용적으로 부담**


**🧠 클러스터링 vs. Hard Negative Mining**

| 항목 | 클러스터링 | Hard Negative Mining |
| --- | --- | --- |
| 장점 | 비용 적음, 대규모 데이터에 적합 | 정확한 hard negative 확보 가능 |
| 단점 | hard negative 보장 불가 | 비용 큼, 효율 낮음 |


**✅ 이 논문이 주장하는 메시지**

> “클러스터링은 완벽하지 않지만, 효율성과 효과의 균형 면에서 매우 실용적인 전략이다.”
> 
> 
> 특히 **pretraining처럼 대규모 데이터 처리 상황**에서는
> 
> 복잡한 hard negative mining 대신, **semantic clustering 기반 배치 구성**이 좋은 대안이 될 수 있다.
> 

# 7. Paths for Future Work

> “클러스터링 기반 contrastive pretraining은 강력하지만, 여전히 할 수 있는 것이 많다.”
> 

---

🔹 7.1 Tiny Dense Clusters — **미니배치보다 더 작은 클러스터**

- 현재 논문은 **1개 클러스터 → 여러 배치**라는 전제를 가짐
- 그러나 **TAS-B**는 오히려 **1 클러스터 = 1배치**처럼 매우 작게 나눔 (평균 200개 쿼리)
- ▶ 아이디어:
    - **작고 조밀한 클러스터**가 더 좋은 hard negative를 제공할 수 있음
    - 또는 여러 클러스터를 섞은 배치가 오히려 **초기 학습엔 더 유리**할 수도 있음

---

🔹 7.2 Smarter Clustering — **더 똑똑한 클러스터링 방식**

- 현재 논문은 query 또는 passage만을 기반으로 클러스터링
- ▶ 하지만 **query와 item 정보 모두를 결합**하면 더 정밀한 클러스터 가능
- 예시:
    - **Spectral Clustering** (Cho et al., 2024): query와 item 임베딩을 모두 반영해 효과 증명
    - 또는 단순히 **임베딩을 concat해서** 클러스터링하는 것도 가능성 있음

---

🔹 7.3 Data Efficiency and Filtering — **쓸모없는 쌍은 걸러내자**

- 실제 데이터에는:
    - 너무 쉬운 negative (정보 부족)
    - 의미 중복 negative (비용은 드는데 의미는 하나)
- ▶ **효율적인 contrastive 학습**을 위해 **데이터 필터링**이 필요
- 클러스터링을 활용해:
    - **밀도 낮은 outlier**는 따로 묶고
    - 의미 있는 negative들이 **배치 안에서 유효하게 학습** 할 수 있음

---

🔹 7.4 Clustering Beyond Text — **텍스트 말고 비전, 멀티모달로 확장**

- 이 논문은 텍스트 임베딩에 초점을 맞췄지만,
- ▶ 유사한 방법론을 **이미지 임베딩**이나 **멀티모달 모델 학습**에 적용 가능
    - 예: 이미지 → vision transformer로 임베딩 → 클러스터링
    - 예: 이미지-텍스트 쌍을 임베딩 후 contrastive 학습

---

🔹 7.5 An Evolving Curriculum — **클러스터링도 점진적으로 조정하자**

- ANCE처럼 **학습 도중 re-embedding & re-mining**을 반복하듯,
- ▶ 이 논문도 **re-clustering → re-sampling**을 반복하면 더 나은 커리큘럼 구성 가능
    - 학습 초기엔 coarse한 클러스터
    - 후반부엔 finer 클러스터로 점점 세분화
- ✔️ 즉, **클러스터 granularity도 학습 단계에 따라 바뀌어야 할 수 있음**

---

# 고찰

1. 이 논문의 이론이 나의 생각과 매우 유사하다. 배울 점이 있다면 clustering 유사도를 수치화하여 비교한 점이다. 난 정성 평가했는데 이 논문은 수치적으로 정량 평가한 것이 배울 점이다.
