# Jasper and Stella: distillation of SOTA embedding models

https://arxiv.org/html/2412.19048v2

# Abstract

### 💡 논문의 핵심 아이디어

> 여러 개의 성능 좋은 대형 모델(teacher)로부터
> 
> 
> **작은 하나의 student 모델**이 지식을 효과적으로 학습하도록
> 
> → **3단계 distillation** 기법을 고안
> 

또한,

- **Matryoshka Representation Learning (MRL)** 기법으로
    
    → 벡터 차원도 줄이면서 성능 유지
    

---

## 🏗️ 기술적 구조 요약

| 구성 요소 | 설명 |
| --- | --- |
| **Multi-stage Distillation** | 성능 좋은 여러 teacher 모델 → 1개의 student 모델에 distill |
| **3가지 Loss 함수** | teacher의 지식을 다각도로 전달하기 위한 loss 설계 |
| **MRL** | student 모델의 벡터 차원을 줄이는 기법(**Matryoshka Representation Learning)** (예: 1024 → 384 등) |
| **Student 모델 이름** | **Jasper (2B 파라미터)**, Stella 기반으로 설계됨 |

---

# 1. Introduction

### 1️⃣ **배경 및 문제점**

- **텍스트 임베딩 모델**은 자연어 처리의 핵심 (검색, 생성 등에서 사용)
- 문장이나 문서를 벡터로 바꿔, 의미적으로 가까운 문장끼리 **벡터 공간상에서 가까워지도록** 학습함

✅ 하지만 현재 MTEB 벤치마크에서 상위권인 모델들은:

- **매우 많은 파라미터 수 (~7B)**
- **벡터 차원이 매우 큼 (4096-dim)**

📌 **문제점**: 실무에 적용하기엔 *느리고 비효율적이며 비용이 큼* (서빙 부담, 저장 문제 등)

---

### 2️⃣ **제안 방식: 작고 효율적인 임베딩 모델 만들기**

> 큰 모델 여러 개의 "지식"을 하나의 작은 모델(Student)에 압축해서 담아내는 방식 제안
> 

### ⛓️ **Multi-stage Distillation 구조**

- **여러 Teacher 모델** → **하나의 Student 모델**
- 단순 KD가 아닌, **3가지 Loss를 이용한 다단계 학습**

| Loss 종류 | 역할 및 설명 |
| --- | --- |
| 🔹 Cosine Loss | 개별 문장의 벡터 차이를 최소화함 (절대적 유사도 정렬) |
| 🔸 Similarity Loss | 문장 쌍의 의미적 거리 유지 (semantic pairwise 차이) |
| 🔺 Relative Similarity Loss | 배치 내 ranking 정보 유지 (좋은 것, 나쁜 것 구별) |

---

### 3️⃣ **벡터 차원 줄이기: Matryoshka Representation Learning (MRL)**

- 여러 Teacher의 벡터를 concat 하면서 dimension이 커짐
    
    → **Matryoshka Representation Learning** 기법을 적용해 student 벡터 차원을 효과적으로 축소 (정보 손실 최소화)
    

🔍 참고: Matryoshka Representation Learning 방식의 학습은 여러 차원으로 나눠, 다양한 차원에서도 의미 보존되도록 학습

---

### 4️⃣ **추가 기법들**

- 🔁 **4단계 distillation**: 각 단계마다 loss와 fine-tune 대상이 다름
- 🖼️ **멀티모달 확장**: Vision encoder 도입 + **Self-distillation**으로 텍스트-비전 정렬

---

### 5️⃣ **성과**

- **Jasper 모델 (2B)** 성능:
    - MTEB 평균 71.54점 (56개 데이터셋 기준)
    - **7B 모델과 유사한 성능**, **2B 이하 모델 대비 우월**
    - 2024년 12월 기준 MTEB 리더보드 3위

---

## 🔍 핵심 기여 요약

|  | 기여 내용 |
| --- | --- |
| ✅ 1 | 여러 teacher로부터 **3가지 loss 기반 multi-stage distillation** 설계 |
| ✅ 2 | 2B 크기의 Jasper 모델로 **7B 모델에 근접한 성능 달성** |
| ✅ 3 | **Matryoshka Representation Learning 적용으로 차원 축소 + 성능 유지** |
| ✅ 4 | 멀티모달 대응, self-distill까지 포함한 확장성 확보 |

---

# **2. Methods**

### 🗂️ 개념 정의 (2.1 Definitions)

![image.png](attachment:8dc20957-6f29-45c9-a8e0-aac048c50b84:image.png)

---

## 🧱 2.2 모델 구조 요약

Jasper는 **텍스트 + 이미지 임베딩이 가능한 멀티모달 모델**입니다:

![image.png](attachment:752ddb54-f3f4-41a3-b4e1-d44eed61c081:image.png)

1. **Language Encoder**: mean pooling 기반 문장 임베딩
2. **Vision Encoder**: 이미지 → 토큰 벡터
3. **Pooler**: 이미지 토큰 → 텍스트 차원에 맞게 매핑
4. **FC Layers**: 다양한 차원의 최종 벡터 출력

---

## 🔄 2.3 Stage 1 & 2: 다중 Teacher로부터 텍스트 distillation

### 📌 핵심 개념

- NV-Embed-v2 (4096-d) + Stella (8192-d) → **concat해서 12,288차원**
- Student 벡터도 FC1을 거쳐 12,288차원으로 만들고 align 학습

### 💥 사용된 Loss 3종

![image.png](attachment:3e87574f-349b-4bf4-8696-090324245e14:image.png)

> 최종 손실:
> 

![image.png](attachment:ba631879-227e-47d6-8c88-361d4e0ecbc0:image.png)

🧩 차이점:

- Stage 1: FC1만 학습
- Stage 2: FC1 + encoder 마지막 3층 학습

---

## 🔻 2.4 Stage 3: 벡터 차원 축소

### ⚠️ 문제

- concat된 벡터가 12,288차원 → 실무 적용 어려움

### ✅ 해결책

- FC2, FC3, FC4로 각각 차원 축소 (예: FC3 → 512차원)
- **Matryoshka Representation Learning** 기법 참고 → 여러 크기의 벡터가 유효하도록 학습

![image.png](attachment:1ce27beb-fca7-4c31-a4de-bd40c0610a60:image.png)

---

### 🌱 Self-distillation 기법 제안

- FC1 (12288-d)의 벡터를 "teacher"로 보고,
- FC2/3/4 벡터를 "student"로 삼아 자체 학습 가능
- 이건 논문에서는 **제안만 하고 실험은 안함**

---

## 🖼️ 2.5 Stage 4: 멀티모달 학습

- **Image-caption pairs**로 학습
- Caption 벡터 → Teacher
    
    Image 벡터 → Student
    
    → Self-distillation 방식으로 이미지와 텍스트 벡터 정렬
    
    → 3개의 loss(cosine + sim + resim) 를 평균내어 최종 loss 산출
    

📌 현재는 *기초적인 정렬 수준*만 달성, 향후 개선 여지 많음

---

## ✅ Jasper 학습 구조 요약

| Stage | 주요 목표 | 학습 대상 | 손실 함수 |
| --- | --- | --- | --- |
| 1 | teacher 정렬 | FC1 | cosine + sim + resim |
| 2 | encoder 학습 | FC1 + encoder 마지막 3층 | 동일 |
| 3 | 벡터 축소 | FC2~FC4 + 전체 모델 | sim + resim |
| 4 | 비전 정렬 | Vision encoder | cosine + sim + resim |

---

## 3. Experiments 요약 + 해설

### 📊 Table 1: MTEB 성능 비교

| 모델 | 파라미터 | 평균 점수 | Retrieval | STS | 분류 | 기타 |
| --- | --- | --- | --- | --- | --- | --- |
| **NV-Embed-v2** | 7.85B | **72.31** | 60.65 | 62.65 | 90.37 | Top-1 |
| **bge-en-icl** | 7.1B | 71.67 | 59.86 | 62.16 | 88.95 |  |
| **Jasper (ours)** | 1.9B | **71.54** | 60.91 | 61.33 | 88.49 | 🏅 Top-3 |
| **Stella (student init)** | 1.5B | 71.19 | 61.21 | 61.01 | 87.63 |  |
| **gte-Qwen2** | 1.7B | 67.16 | 59.98 | 58.29 | 82.47 |  |
| **voyage-lite-02** | 1.2B | 67.13 | 58.24 | 56.60 | 79.25 |  |

✅ **결론**:

- Jasper는 **2B 미만 모델 중 최고 성능**
- 파라미터 수가 3~4배 많은 상위 모델과도 **성능 격차 거의 없음**
- 특히 **retrieval / STS** 부문에서 **NV-Embed와 유사하거나 더 나은 성과**

---

## ⚙️ 3.1 Implementation Details

| 항목 | 설정 |
| --- | --- |
| 초기화 | `stella_en_1.5B_v5` + `google/siglip 400M` |
| 총 파라미터 | 1.9B |
| 손실 가중치 | λ₁=10, λ₂=200, λ₃=20, margin=0.015 |
| GPU | RTX A6000 × 8 |
| 학습 precision | bf16 + DeepSpeed ZeRO-2 |
| 최대 입력 길이 | 512 tokens |

### 📌 Stage별 학습 전략

| Stage | 목표 | 러닝레이트 | 최종 스텝 |
| --- | --- | --- | --- |
| 1 | Teacher KD | 1e-4 | 4000 |
| 2 | KD + Encoder 튜닝 | 8e-5 | 7000 |
| 3 | 벡터 차원 축소 | 7e-5 | 2200 |
| 4 | 멀티모달 정렬 | 1e-4 | 3500 |

---

## 📚 3.2 Dataset 구성 전략

### 🔤 텍스트 학습 데이터 (Stage 1~3)

- **Fineweb-Edu** (80%) + **Sentence-Transformers 데이터** (20%)
- 총 텍스트 수: **800만 문서**

📌 전처리 방식:

- 30%는 1~10문장 단위로 분할 → **문장 다양성 증가**
- 0.08%는 단어 순서를 섞음 → **순서에 강인한 학습 유도**

### 🖼️ 이미지-캡션 데이터 (Stage 4)

- 사용 데이터: **BAAI/Infinity-MM**
- caption → teacher, image → student로 self-distillation

---

## ✅ 3.3 성능 요약 (핵심 포인트)

| 측면 | 해설 |
| --- | --- |
| 🔹 파라미터 수 대비 성능 | 7B 모델과 거의 동등한 성능 (→ distillation 효과 뚜렷) |
| 🔸 Retrieval, STS 강세 | RAG, FAQ 같은 응용에서 특히 중요 |
| 🔹 벡터 차원 축소에도 성능 유지 | MRL 기반 차원 축소 전략 유효 |
| 🔸 멀티모달 예비 성능 확보 | 추후 Cross-modal alignment 확장 가능성 보여줌 |

---

## 4. Discussion

### 🧭 4.1 Instruction Robustness

> "Instruction 기반 임베딩 모델의 프롬프트 민감도" 실험
> 

📌 배경

- 최근의 embedding 모델들(BGE 등)은 task-specific instruction을 붙여서 query를 인코딩함
    
    → `Query: {text}`, `Represent this sentence for retrieval: {text}` 등
    
- 하지만 실사용에서는 instruction이 **다양하게 변형**될 수 있음 → 이에 대한 **강건성(robustness)** 중요

📌 실험

- GPT-4o로 다양한 instruction 변형 생성
- 동일한 task에 대해 Jasper 모델이 프롬프트에 얼마나 영향을 받는지 테스트

📌 결과

- Jasper는 instruction이 바뀌어도 성능 저하 없이 **일관된 벡터 표현** 생성
- ✅ **Instruction Robustness 확보**

💡 **의의**

- 실사용에서 사용자가 다양한 형태의 질의를 던져도 안정적인 검색 성능 가능
- FAQ/RAG에서 **자연어 질의 다양성**에 유연하게 대응 가능

---

### 🖼️ 4.2 Vision Encoding의 개선 여지

📌 현재 설정:

- Stage 4에서는 이미지-문장 정렬을 위한 **기초 self-distillation만 수행**
- Vision encoder는 `google/siglip` 사용했으나, 훈련은 매우 제한적

📌 한계 및 향후 계획:

| 이슈 | 개선 방향 |
| --- | --- |
| Loss 진동 발생 (unstable training) | 학습 스케줄링/데이터 증강/contrastive loss 도입 필요 |
| 이미지-문장 정렬 약함 | Stage 5: **VQA 데이터 기반 contrastive 학습** 예정 |
| 멀티모달 학습 깊이 부족 | modality fusion 구조 확장 or dual encoder → cross encoder 전환 고려 가능 |

---

# 질문 사항

## 1. Jasper가 Stella를 teacher로 삼는 구조 요약

| 역할 | 모델 | 설명 |
| --- | --- | --- |
| Init 모델 | **Stella (stella_en_1.5B_v5)** | Jasper가 이 모델의 파라미터로부터 시작 |
| Teacher 1 | **Stella (동일)** | Jasper가 따라야 할 target vector 중 하나 |
| Teacher 2 | **NV-Embed-v2** | 추가로 distill할 SOTA 임베딩 모델 (7.8B) |

즉,

- Jasper는 **Stella의 구조와 파라미터를 그대로 이어받되**
- 동시에 Stella의 벡터 표현을 **teacher의 기준값 중 하나로 사용**해서 그에 맞춰 출력 벡터를 정렬함

---

## 🔍 왜 자기 자신을 teacher로 삼을까?

이건 **knowledge distillation의 실용적 응용**입니다:

- 일반적으로 distillation은 큰 모델 → 작은 모델로 하는데,
- Jasper는 같은 크기의 모델(Stella 1.5B)로 초기화하고도,
- **다중 teacher(= Stella + NV-Embed)의 표현을 모두 반영하여 재학습**합니다.

즉,

> "Stella에서 시작하지만 Stella를 초월한 Jasper를 만들기 위한 전략"
> 
> 
> (Stella가 가진 지식 + NV-Embed가 가진 표현력 ⇒ Jasper가 모두 흡수)
> 

---

## 📌 정리

- Jasper는 **Stella로 초기화되었지만**,
    
    동시에 **Stella를 teacher로 삼아 output 정렬**을 수행함
    
- 이는 **"자기 자신 + 외부 teacher"로부터 지식을 증류**하는 셀프 디스틸 전략이며,

# 2. 이미지 파트 distillation 구조 요약 (Stage 4 기준)

| 역할 | 설명 |
| --- | --- |
| 📜 텍스트 (caption) | **teacher vector**로 사용 |
| 🖼️ 이미지 | vision encoder의 출력 → **student vector**로 간주 |
| 📐 손실 계산 | 텍스트 벡터와 이미지 벡터 간 cosine/similarity/resim loss 계산 (총 3개 loss 평균) |
| 🔁 방식 | **self-distillation** (텍스트=스스로 만든 기준 → 이미지가 그걸 따라가도록 학습) |
| ❌ 별도 이미지 teacher | 없음 (CLIP이나 siglip과 같은 사전학습된 teacher를 사용하지 않음) |

## 🤔 왜 이미지 teacher가 없을까?

1. **리소스/시간 부족** (논문에서도 언급)
2. Stage 4는 **초기 시도 수준**이며, 향후 Stage 5에서
    - **VQA 데이터**
    - **contrastive learning**
    - 또는 **CLIP/siglip을 teacher로 삼는 확장 구조**
    
    를 고려하고 있다고 암시함
    

---

# 3. 사용한 데이터 셋

**사용한 데이터셋은 모두 비지도 학습(unlabeled, unsupervised)용 데이터셋입니다.**

즉, **정답(label)이 있는 supervised dataset은 사용하지 않았습니다.**

## 📚 사용된 데이터셋 정리 (논문 기준)

### 🔹 Stage 1~3 (텍스트 임베딩 학습)

| 데이터셋 | 비율 | 설명 |
| --- | --- | --- |
| `fineweb-edu` | 80% | **문단 중심의 비지도 텍스트 데이터** |
| `sentence-transformers/embedding-training-data` | 20% | 비지도 문장/질문 데이터 (다양성 확보 목적) |

📌 주의: `sentence-transformers`에서 사용하는 데이터셋 중 일부는 NLI 등 supervised한 것도 있지만,

**논문에서 사용한 버전은 labeled task용이 아닌 비지도용 pair data**입니다.

### 🔹 Stage 4 (멀티모달 정렬 학습)

| 데이터셋 | 설명 |
| --- | --- |
| `BAAI/Infinity-MM` | **이미지-캡션 쌍** (caption을 텍스트 teacher로 사용) |

---

## 🧠 논문이 명시한 핵심 문장

> “We do not need any supervised data. Without considering resource constraints, we can use trillions of unsupervised texts for distillation training…”
> 

→ 즉, 논문 자체가 **supervised data 없이도 SOTA 성능을 낼 수 있는 embedding 모델을 만든다**는 것이 핵심 취지입니다.

---

# 4. 2.4 Stage 3: 벡터 차원 축소 부분에서 왜 cosine loss는 제외했을까?

> Stage 3에서는 teacher 벡터와 student 벡터의 차원이 다르기 때문에 cosine loss를 쓸 수 없습니다.
> 
> 
> → 대신 **similarity loss**와 **relative similarity loss**만 사용합니다.
> 

---

## 🧠 배경 이해

### 🔹 Stage 1 & 2:

- teacher 벡터 = concat(4096 + 8192) = **12,288차원**
- student 벡터 = FC1을 통해 동일하게 **12,288차원**
- ✅ 따라서 cosine 유사도 직접 계산 가능 → `ℒ_cosine` 사용

---

### 🔻 Stage 3: 차원 축소

- student 벡터는 FC2/FC3/FC4를 통해 **512, 768, 1024차원 등 다양한 저차원**으로 압축됨
- 하지만 teacher 벡터는 여전히 **12,288차원**

⚠️ 문제:

> Cosine 유사도 = 두 벡터의 내적 / (norm 곱)
> 
> 
> → **차원이 다르면 정의 자체가 불가능**
> 

---

## 🔍 왜 다른 loss는 가능한가?

| Loss | 차원 맞춰야 하나? | 설명 |
| --- | --- | --- |
| `cosine loss` | ✅ 필요함 | 벡터 간 직접 비교 |
| `similarity loss` | ❌ 불필요 | 유사도 행렬끼리 비교 → 내적 기반 similarity 비교 |
| `relative similarity loss` | ❌ 불필요 | positive vs. negative 간 **상대 유사도**만 고려하면 됨 |

→ 즉, **cosine loss는 절대 표현 정렬이 필요한 반면**,

다른 두 loss는 **상대 관계만 맞추면 되기 때문에 차원이 달라도 학습이 가능**합니다.

---

# 5. **Relative Similarity Loss 부연설명**

좋습니다. 이 논문에서 사용한 는 일반적인 contrastive loss의 변형으로,

**CoSENT (Cosine Sentence Embedding Training)**에서 영감을 받아 설계된 **ranking 기반의 margin loss**입니다.

---

## 🎯 목적

> 문장 간 상대적 유사도(순서)를 보존하기 위해,
> 
> 
> **positive 쌍은 negative 쌍보다 더 높은 유사도를 가지도록** 학습시킴.
> 

---

## 📐 수식 (논문 정의 기준)

![image.png](attachment:6fc5c989-461f-41b6-a965-d0a82945db7c:image.png)

---

## 🧠 작동 방식 (예시)

1. Teacher 모델 기준으로 **문장 A > 문장 B** (유사도가 더 높음)라고 판단되면
2. Student는 반드시 **A 쌍의 유사도 > B 쌍의 유사도 + margin**이 되도록 학습
3. 이를 위반하는 경우에만 loss 발생 → hinge 기반 margin loss

---

## 🔍 왜 필요한가?

- **Cosine loss**는 절대 벡터 정렬 → 의미는 유지해도 방향이 조금 다르면 패널티
- **Similarity loss**는 전체 유사도 행렬 정렬 → 노이즈에 민감할 수 있음
- 👉 **Relative similarity loss**는 순서 정보만 보존 → **일반화 성능에 매우 유리**

특히 배치 내에서 다양한 문장쌍을 비교할 수 있기 때문에

→ **긍정/부정 쌍의 구조적인 관계**를 학생 모델이 잘 배울 수 있음

---

# 6. vision 부분 세부설명

### 🔹 Stage 4: 멀티모달 학습

- image-caption 쌍을 이용해 **기초적인 정렬(aligning)**만 수행
- vision encoder(siglip 기반)는 오직 **self-distillation** 방식으로 학습됨
    - caption의 벡터 → teacher
    - 이미지 벡터 → student
    - 세 가지 loss(cosine, sim, resim)를 평균

### 🔹 평가 관련 직접 언급

논문 4.2절(Discussion)에 다음과 같은 문장이 나옵니다:

> “We were only able to equip the Jasper model with a basic image encoding capability. ... Overall, there is considerable room for enhancement in the multimodal training.”
> 

즉:

- 실험은 했으나 정량적 지표 없음
- 성능을 검증할 **멀티모달 벤치마크(VQA, MSCOCO, Flicker30k 등)**는 사용되지 않음
- 향후 Stage 5에서 vision-language contrastive learning 도입을 예고만 함

---

## 🧠 요약하면

| 항목 | 상태 |
| --- | --- |
| 이미지 입력 허용 | ✅ yes (`siglip` encoder 사용) |
| 이미지-텍스트 정렬 학습 | ✅ yes (Stage 4에서 self-distillation 수행) |
| 멀티모달 성능 평가 | ❌ no (논문에 평가 지표 없음) |
| 향후 확장 가능성 | ✅ Stage 5에서 VQA 기반 contrastive 학습 언급 |

---

## 📌 전문가 관점 정리

> Jasper는 현재 텍스트 중심 모델이며, vision encoder는 "기초적인 정렬 기능만 탑재된 상태"입니다.
> 
> 
> → **멀티모달 확장 가능성을 열어둔 프로토타입 수준**으로 보는 것이 적절합니다.
> 

Text 부분 teacher 모델 크기 : NV-Embed 7.85B  + stella 1.5B → 약 9B

이미지 부분 teacher 모델 크기 : 400M?

# 7. Teacher 모델에 자기자신(stella)이 존재하는 이유

왜 **NV-Embed 하나만으로 distillation을 하지 않고**,

**초기화(init) 모델인 Stella를 teacher로도 추가했는가?**

→ 여기에 담긴 이유는 **기술적, 실용적, 학습 안정성 측면 모두를 고려한 전략**입니다.

---

## 🔍 요약: Stella도 teacher로 쓴 이유

| 이유 | 설명 |
| --- | --- |
| 🧠 **초기 파라미터와의 정렬** | 초기화된 Stella의 표현을 유지하면서 학습 방향성을 잡아줌 → **학습 안정성 향상** |
| 🎯 **다양한 표현 학습** | NV-Embed와 Stella는 **표현 방식이 다름** → 두 모델의 지식을 **보완적으로 distill** |
| ⚖️ **Teacher blending** | NV-Embed만 쓰면 특정 표현에 **과도하게 쏠림 현상**이 생길 수 있음 |
| 🚫 **NV-Embed는 7.8B** | 너무 크고 표현이 복잡해서 작은 모델(Jasper)에게 그대로 넘기면 **표현 왜곡** 위험 → Stella가 중간 수준 teacher 역할 |

---

## 🔬 조금 더 자세히 풀어보면…

### 1️⃣ 학습 안정성과 수렴 유도

- Jasper는 **Stella로 초기화됨**
- 학습 초기에 **Stella와 완전히 다른 표현**을 따라가려 하면 → **loss landscape가 불안정**
- Stella를 teacher로 두면 초기 표현과 teacher 표현이 **자연스럽게 연결**되어 → **수렴 속도 및 안정성 개선**

→ ⚠️ 학습 초기에 “모델 자신이 원래 내던 벡터”와 정렬되면 **gradient 방향이 너무 튀지 않음**

---

### 2️⃣ 표현 다양성 확보

- NV-Embed는 7.8B로 더 크고, 다양한 정보를 담고 있음
    
    → 그러나 Jasper가 그 표현만 따라가면 **모델이 과적합되거나 벡터 품질이 왜곡**될 수 있음
    
- Stella는 **자연스럽고 깔끔한 표현**을 가진 1.5B 모델
    
    → 이를 함께 teacher로 사용하면 Jasper는 **두 표현의 중간 지점**에서 **더 일반화된 임베딩**을 학습하게 됨
    

---

### 3️⃣ Teacher 간 상보성 (complementarity)

- NV-Embed → SOTA 모델이지만, 학습적으로 부담이 큼
- Stella → 구조적으로 유사하고, 표현이 Jasper와 호환됨
- → 이 둘을 **concat하여 teacher 벡터로 사용**하면:
    - NV-Embed의 정보량
    - Stella의 안정성과 표현력
        
        을 **모두 distill 가능**
        

---

## ✅ 결론: Stella를 teacher로 쓰는 건 "합리적이고 전략적인 선택"

| 항목 | Stella teacher로 쓰는 효과 |
| --- | --- |
| 초기 안정성 | init 모델과의 표현 정렬로 학습 수렴 빨라짐 |
| 표현 다양성 | NV-Embed와 다른 표현을 추가하여 일반화 |
| distill 품질 | teacher 표현 분포를 soft하게 만들어줌 |
| 학습 난이도 조절 | 너무 큰 teacher 하나만 쫓는 risk 분산 |

추가적으로 Jasper 논문에서 **Stella를 teacher로 함께 사용하는 이유**는,

LLM에서 DPO(Direct Preference Optimization)가 KL-divergence를 손실에 포함시키는 것과 **정확히 유사한 목적**을 가지고 있습니다.

---

## 🔁 유사한 구조: DPO의 KL-divergence vs Jasper의 Stella teacher

| 개념 | DPO | Jasper |
| --- | --- | --- |
| 목적 | 기존 reference 모델과 **큰 차이를 방지** | init 모델(Stella)과 **급격한 표현 변화 방지** |
| 방식 | KL-divergence를 통해 π와 π₀ 간 거리 제약 | Stella 벡터를 teacher로 삼아 정렬 loss 적용 |
| 효과 | 학습 안정성 유지, catastrophic drift 방지 | 모델 초기 표현 보존 + 안정적인 학습 방향성 |

---

## 📌 Jasper에서 Stella 사용도 같은 맥락

- Stella = init 모델이자 teacher → **초기 표현 분포의 암묵적인 기준점**
- Jasper가 NV-Embed만 따라가면 표현이 지나치게 바뀌는 위험
- 따라서 Stella를 함께 teacher로 삼으면:
    
    > "출발점(Stella)의 분포를 일정 부분 유지하며 NV-Embed로 수렴"
    > 
    > 
    > → 결과적으로 DPO의 KL 역할과 동일한 **drift-regularization 효과**
    > 

---

## ✅ 결론

> DPO의 KL-divergence와 Jasper의 Stella-distillation은 본질적으로 같은 의도:
> 
> 
> **"기준 모델(reference)의 표현을 보존하면서, 더 나은 목표 모델로 부드럽게 이동"**
>
