# EXAONE Deep: Reasoning Enhanced Language Models

https://arxiv.org/pdf/2503.12524v2

### ✅ 모델 개요

- **EXAONE Deep 시리즈**는 LG AI Research에서 공개한 **추론 특화 LLM**입니다.
- 세 가지 사이즈로 제공:
    - **2.4B**, **7.8B**, **32B**
- 기존의 **EXAONE 3.5** 시리즈를 기반으로, **추론(task reasoning)**에 강하도록 fine-tuning됨.

### ✅ 학습 방식

- **세 가지 학습 기법**을 활용:
    1. **SFT (Supervised Fine-Tuning)** – 답변 스타일이나 정답을 학습하는 기본 방식
    2. **DPO (Direct Preference Optimization)** – 선호를 반영하는 학습법
    3. **Online RL (Reinforcement Learning) -** GRPO 사용 ****

---

### ✅ 2.1 Data

### 📌 사용된 데이터 양

- **SFT (지도 학습):**
    - 약 **1.6M 인스턴스**, 총 **120억 토큰**
    - → 체계적인 추론 학습을 위한 데이터
    - (DeepSeek R1의 cold start 데이터라고 생각하면 될 것 같다.)
    
    SFT 데이터 예시
    
    <thought> 태그 안에 논리 전개 → 마지막에 정답 생성
    
   ![image](https://github.com/user-attachments/assets/a594bf37-7aaa-4ecd-9187-e56398b40be5)

    
- **DPO (선호 기반 학습):**
    - **2만 개**의 선호 데이터 (A보다 B가 더 좋다 식의 비교 데이터)
- **Online RL:**
    - **1만 개**의 추가 인스턴스 사용
    
    ---
    

### ✅ 2.2 Training (학습 방식)

### 🔧 베이스 모델

- **EXAONE 3.5 Instruct** 모델에서 출발
    - 이미 instruction-following 성능이 있는 모델
    - 이를 기반으로 reasoning 능력 특화 학습 진행

### 🧠 학습 방식 요약

- **SFT**:
    - `<thought>...</thought>` 구조로 논리적 사고를 유도하는 데이터로 학습
- **DPO**:
    - SimPER 알고리즘 사용(https://github.com/tengxiao1/SimPER)
- **Online RL**:
    - **GRPO**

### 🖥️ 학습 환경

- 학습 프레임워크: **NVIDIA NeMo** (https://github.com/NVIDIA/NeMo)

---

## 🚧 한계점

- **EXAONE Deep**은 **"추론(task reasoning)"에 특화된 파인튜닝 모델**입니다.
- 즉, **논리적 사고**, **수학 문제 해결**, **코딩**, **과학 질문** 같은 **Reasoning 중심 태스크에는 매우 강력**하지만...
- **실제 서비스나 일반적인 사용 시나리오** (예: 일반 Q&A, 요약, 챗봇 대화 등)에서는 제한이 있을 수 있음.

---

## ✨ 핵심 요점 정리

| 항목 | 특징 |
| --- | --- |
| 데이터 | SFT 120억 토큰, DPO 2만개, RL 1만개 |
| 학습 방식 | SFT + DPO(SimPER) + Online RL(GRPO) 조합 |
| CoT 구조 | `<thought>` 태그로 추론 흐름 유도 |
