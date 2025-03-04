https://arxiv.org/pdf/2410.02525

![image](https://github.com/user-attachments/assets/730f07e9-a2af-45ad-8096-e41673c0b916)

## 🔹 핵심 요약
### 🚀 기존 방식의 문제점
Biencoder 모델(예: DPR, GTR, Contriever 등)은 문서와 쿼리를 독립적으로 임베딩하므로 맥락 정보(context)가 반영되지 않음.
통계적 검색 모델(BM25 등)은 IDF(Inverse Document Frequency) 등 전역적인 문서 통계를 활용하지만, 딥러닝 기반 dense embedding 모델들은 이런 정보를 학습하지 못함.
결과적으로, 도메인 간 성능 편차(out-of-domain generalization)가 큼 → 한 도메인에서 훈련된 모델이 새로운 도메인에서 성능이 저하됨.
🔥 논문의 주요 기여
✅ 문서 임베딩을 "맥락화(Contextualization)"하는 두 가지 방법 제안

### 📌 새로운 contrastive learning 목표
문서 간 유사도를 학습할 때, 단순한 batch 내 contrastive loss 대신 유사한 문서를 클러스터링하여 훈련
즉, 훈련 과정에서 비슷한 주제의 문서들을 함께 학습하여 문맥 정보를 반영하는 embedding을 형성

### 📌 새로운 contextual architecture
기존 BERT-style 인코더를 확장하여 주변 문서 정보를 함께 인코딩하는 구조 도입
주변 문서 정보를 추가로 활용하여 보다 도메인에 적응적인(document-aware) 임베딩을 생성
biencoder 구조를 유지하면서도 문맥 정보를 추가로 반영할 수 있도록 설계

---

## 🔹 논문 상세 분석

### 1️⃣ 기본 개념: 기존 Biencoder vs CDE
Biencoder 방식:
문서(Document)와 쿼리(Query)를 독립적으로 dense vector로 변환 후 similarity score를 계산

f(d,q)=ϕ(d)⋅ψ(q)
CDE 방식:
문서 임베딩을 만들 때 해당 문서뿐만 아니라, 연관된 문서들의 정보도 반영
![image](https://github.com/user-attachments/assets/730f07e9-a2af-45ad-8096-e41673c0b916)

즉, M1에서 주변 문서들을 임베딩한 후, M2에서 최종 임베딩을 생성하는 2단계 프로세스
### 2️⃣ Contextual Training (문맥을 반영한 contrastive learning)
기존 contrastive learning은 random negative sampling을 사용하여 문서를 구분
그러나, 이 방식은 특정 도메인에서는 중요한 단어(예: "NFL")가 너무 자주 등장해서 구분이 어려움
해결책:
문서를 의미적으로 유사한 클러스터로 그룹화하여 contrastive learning을 수행
단순한 랜덤 negative sampling이 아니라 "문맥적으로 구별하기 어려운 문서들"을 함께 학습하는 방식
### 3️⃣ Contextual Document Encoder (CDE)
2-Stage Encoding 구조
M1: 주변 문서들을 먼저 encoding
M2: 문서 d를 M1의 출력과 함께 입력하여 최종 embedding 생성
맥락 정보를 활용하는 기법
Sequence Dropout: 특정 확률로 문맥 정보를 제거하여 일반화 성능 향상
Position-agnostic Encoding: 문서 순서 정보를 제거하여 도메인 적응성을 높임
Gradient Caching: 훈련 속도를 높이기 위한 최적화 기법 사용

---

## 🔹 실험 결과
✅ 다양한 retrieval 벤치마크에서 기존 Biencoder 대비 성능 향상

특히, Out-of-domain retrieval 성능이 크게 향상됨
MTEB 벤치마크에서 최고 성능을 기록 (250M 이하 모델 기준)
추가적인 hard negative mining, score distillation 없이도 성능 향상
→ 즉, 기존 방법들이 필요했던 복잡한 학습 기법 없이도 효과적임

---

## 🔹 논문의 핵심 기여와 의의
Dense Document Embedding을 Context-aware하게 확장
기존 biencoder 방식 대비 문맥을 반영하여 도메인 적응성이 향상됨
Contrastive Learning 개선
문서 클러스터링 기반 contrastive loss를 적용하여 학습 효과를 극대화
효율적인 Contextual Encoding 아키텍처 제안
두 단계로 문서 임베딩을 생성하여 retrieval 성능을 향상시킴
Retrieval 모델을 도메인 특화 없이도 효과적으로 학습 가능
기존에 필요했던 복잡한 hard negative sampling 없이도 성능 향상


논문에서는 retrieval을 중심으로 설명했지만, 문서 분류, 요약, 질의응답(QA) 등에도 적용 가능
특히, 법률, 의료, 금융 도메인처럼 도메인 특성이 중요한 분야에서 효과적일 가능성이 높음
💡 LLM과 결합하면 더 강력한 검색 시스템 가능

LLM 기반 RAG(Retrieval-Augmented Generation) 모델에서 더 정확한 문서 검색이 가능해질 것
LLM이 참조할 문서를 더 정확하게 선택할 수 있도록 도와줌

---

## 🔹 결론 및 요약
✅ 기존 biencoder 방식의 한계를 해결하는 새로운 Context-aware document embedding (CDE) 제안
✅ Contrastive learning을 개선하여 문서 간 유사성을 효과적으로 학습
✅ 검색 모델의 out-of-domain 성능을 크게 향상시키며, 추가적인 학습 비용 없이도 적용 가능
✅ 실제 검색 시스템 및 LLM 기반 RAG 모델에도 활용 가능

---

## Detail

### 1️⃣ False Negative Filtering (거짓 부정 샘플 제거)
Retrieval 시스템에서 negative sample(비슷하지 않은 문서)을 선정할 때, 실제로는 관련이 있는 문서가 잘못된 negative로 포함될 가능성이 있음.
논문에서는 false negative를 줄이기 위해 새로운 filtering 기법을 제안:
특정 문서 d에 대해, surrogate scoring function **f(q, d')**을 사용해 비슷한 문서를 걸러냄.
이렇게 해서 거짓 negative가 학습에 영향을 주는 걸 방지.
논문에서는 이를 수식으로 표현:

​![image](https://github.com/user-attachments/assets/8175f930-3a41-4f63-acf7-bc1e5e874a8e)

 
여기서 **S(q, d)**는 쿼리 q와 문서 d 사이의 점수가 특정 threshold를 넘는 문서들의 집합 → 즉, 확실한 negative만 유지하는 방식.
이 과정에서 GTR 등의 사전 훈련된 모델을 활용하여 filtering 수행.
🛠 빠뜨린 내용 요약
✅ 기존 contrastive learning에서 negative sample이 잘못 포함되는 문제를 해결하기 위해 false negative filtering 기법을 적용함.
✅ 사전 학습된 임베딩 모델을 사용해 특정 문서가 false negative가 되는지 판단 후 제거.

### 2️⃣ Batch Packing & Clustering 최적화
논문에서 배치(batch)를 구성하는 방법도 상당히 신경 써서 설계되었음.

기존 contrastive learning에서는 batch에 임의로 negative sample을 포함하지만, 이 논문에서는 더 도전적인(difficult) negative를 포함하는 방식을 사용.
클러스터링을 통해 batch를 구성하는 방식:
먼저 K-Means clustering을 활용해 문서를 유사한 그룹으로 나눔.
이후, 같은 클러스터 내에서 training batch를 구성.
이 과정에서 batch 내 문서들이 서로 더 유사하도록 만들고, contrastive learning의 효과를 극대화.
⚡ Packing 전략

batch 크기가 고정되어야 하는데, clustering 결과로 배치 크기가 다를 수 있음.
이를 해결하기 위해:
랜덤 배치 분할(Random Partitioning)
Greedy Traveling Salesman 기반 배치 구성
각 배치 내에서 최대한 비슷한 문서들끼리 포함하도록 구성.
배치 내 문서 간 차이를 최소화하여 contrastive learning 효과를 증가시킴.
🛠 빠뜨린 내용 요약
✅ 학습 배치를 단순 랜덤 샘플링이 아닌 클러스터 기반으로 구성하여, contrastive learning이 더 효과적으로 작동하도록 설계함.
✅ Greedy TSP 기반 batch packing 기법을 사용해 batch 크기를 조정하고, 훈련 샘플 간 유사성을 유지함.

### 3️⃣ Two-Stage Gradient Caching (메모리 효율 최적화)
논문에서는 훈련 시 메모리 사용량을 최적화하는 기법도 포함하고 있음.

새로운 모델 구조는 기존 biencoder보다 연산량이 증가하는데, 이를 해결하기 위해 Gradient Caching 기법을 적용.
이 기법은 크게 두 단계로 나뉨:
M1 단계(주변 문서의 embedding을 계산하는 단계)
이 단계에서 gradient를 계산하지 않고, embedding만 생성하여 캐싱.
M2 단계(최종 embedding을 생성하는 단계)
여기서 backpropagation을 수행하여 메모리 사용량을 줄임.
💡 이 기법의 장점

기존 contrastive learning보다 메모리를 절약하면서도, 큰 배치를 활용한 학습이 가능해짐.
특히, contextual document embedding을 훈련할 때 추가적인 연산량을 최소화할 수 있음.
🛠 빠뜨린 내용 요약
✅ 모델 학습 시 메모리 사용량을 줄이기 위해 Two-Stage Gradient Caching 기법을 도입.
✅ M1과 M2 단계를 분리하여 일부 연산을 캐싱함으로써, 메모리를 아끼고 학습 속도를 최적화함.

### 4️⃣ Implementation Details (구현 관련 추가 내용)
논문에서는 실험 설정을 매우 구체적으로 설명함.
🔹 사전 훈련된 모델: NomicBERT (137M parameters)
🔹 FAISS 기반 K-Means Clustering 사용 (배치 구성 시 활용)
🔹 Optimizer: Adam, warmup step 1000, learning rate 2e-5 → 선형 감소
🔹 Supervised + Unsupervised 학습 병행
🔹 BEIR, MTEB 벤치마크에서 평가
🛠 빠뜨린 내용 요약
✅ FAISS 기반 클러스터링을 활용하여 batch 구성 최적화.
✅ NomicBERT 모델을 기반으로 학습을 수행하고, 다양한 하이퍼파라미터 튜닝을 진행.

## 🔥 Detail 정리
📌 False Negative Filtering
✅ 기존 contrastive learning의 문제점(잘못된 negative sample 포함)을 해결하기 위해 filtering 기법을 추가

📌 Batch Packing & Clustering 최적화
✅ 배치를 단순 랜덤 샘플링하는 것이 아니라 문서 유사도 기반 클러스터링 후 구성
✅ Greedy Traveling Salesman 기반 batch packing 기법을 활용

📌 Two-Stage Gradient Caching (메모리 최적화 기법)
✅ 훈련 과정에서 gradient caching을 적용해 메모리 사용량을 절감

📌 Implementation 세부 정보
✅ NomicBERT (137M parameters) 기반으로 실험 진행
✅ FAISS 기반 K-Means Clustering 활용
✅ BEIR, MTEB 벤치마크에서 성능 평가


