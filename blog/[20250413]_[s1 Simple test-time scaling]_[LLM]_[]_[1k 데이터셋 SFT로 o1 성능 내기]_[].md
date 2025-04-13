**s1: Simple test-time scaling**

https://arxiv.org/html/2501.19393v3


### 전하는 말

이 논문은 데이터 관점에서 보면 도움이 됩니다. 1K개의 데이터를 샘플링하기 위해 데이터 수집과 정제를 어떻게 진행 했는지가 핵심 포인트라고 생각합니다.

# 1. Abstract

- **Test-time scaling**: 모델을 더 "오래" 생각하게 하면 성능이 좋아진다.
- 기존에는 이 개념을 RL이나 MCTS 같은 복잡한 방법으로 구현했지만,
- 이 논문은 **SFT (1,000개 데이터) + Budget Forcing**이라는 간단한 방법만으로 OpenAI o1 수준 성능 달성!

---

**1) s1K 데이터셋**

- Google Gemini에서 추출한 고품질 reasoning 문제 1,000개
- 3가지 기준으로 선별:
    
    👉 난이도 (challenging)
    
    👉 다양성 (topic diversity)
    
    👉 품질 (reasoning trace의 신뢰도)
    

> ❗ Ablation으로 증명: 무작위 샘플이나 reasoning trace 길이 기준만 쓰면 성능 뚝 떨어짐 (~30%)
> 

**2) Budget Forcing (생각 시간 조절 기법)**

- 모델이 “너무 빨리 답하려고 할 때” → **“Wait”를 넣어 더 생각하게 함**
- 모델이 “생각을 너무 오래할 때” → **“종료 토큰”을 강제로 삽입해 생각 마침**
- 이렇게 해서 **test-time compute 조절이 가능**해짐

**3) 모델: s1-32B**

- Qwen2.5-32B-Instruct를 SFT만으로 fine-tune (26분, 16xH100)
- test-time에서 Budget Forcing을 써서 reasoning 향상
- OpenAI o1-preview보다 AIME24, MATH에서 최대 27% 더 잘함

# 2. Reasoning data curation to create s1K

> s1K란?
> 
> 
> 59,000개의 문제 중 **품질, 난이도, 다양성** 기준으로 1,000개만 선별한 **샘플 효율이 매우 높은 reasoning 학습 데이터셋**입니다.
> 

**✅ 주요 흐름**

```
59K raw → (품질 필터) → 51K → (난이도 필터) → 24K → (다양성 샘플링) → s1K (1K)
```

---

**2.1 Initial Collection (59K 수집 과정)**

**🔍 수집 기준:**

- **Quality**: 형식 문제 없는 고품질 문제만 사용
- **Difficulty**: 단순 문답이 아닌, **추론 과정이 필요한 고난도 문제**
- **Diversity**: 수학뿐 아니라 과학, 컴퓨터 등 **도메인 다양성 확보**

**📚 데이터 출처:**

- **기존 데이터셋 활용**
    - `NuminaMATH`: 3만 개 이상 수학 문제 (가장 많은 비중)
    - `OlympicArena`: 물리, 생물, 지리 등 다양한 올림피아드 문제
    - `AGIEval`: SAT, LSAT, GMAT류 시험 문제 (언어/논리 포함)
    - `OmniMath`: 경쟁용 수학 문제
- **자체 제작 데이터셋**
    - `s1-prob`: 스탠포드 확률 박사시험 문제 + 손글씨 해설
    - `s1-teasers`: 퀀트 인터뷰용 퍼즐 문제 (난이도 상급만 채택)

**🧠 Reasoning Trace 생성 방법**

- **Google Gemini API**를 이용해:
    - 문제에 대한 reasoning step과 최종 답변을 자동 생성
    - 데이터 구성 `(질문, 추론과정, 정답)`

**🧹 후처리**

- 평가셋(MATH500, AIME24 등)과 **8-gram 중복 검사로 decontamination**
- 중복 제거 및 포맷 오류 샘플 제거

---

**2.2 Final Selection (1K 선별 방식)**

**핵심 목표**

> 수천 개 학습 데이터 없이도 OpenAI o1 수준에 도달하기 위해,
> 
> 
> → 가장 "작고 효율적인" 1,000개를 고르기
> 

---

**🔍 Step 1: Quality 필터링**

- API 오류 있는 문제 제외 → 54K
- 포맷 문제 있는 문제 제외 (그림 없음, ASCII art 등) → 51K

---

**🔍 Step 2: Difficulty 필터링**

**✅ 난이도 측정 방법:**

1. **Qwen2.5-7B**와 **Qwen2.5-32B** 두 모델이 정답을 맞혔는지 확인
2. **둘 중 하나라도 정답이면 → 너무 쉬움 → 제거**
3. **추론 trace 길이**가 길수록 난이도 높은 문제로 간주

> ❗ 모델이 정답을 맞히는 문제는 학습 데이터로 부적절 → "오히려 어려운 문제만 사용"
> 

**이 과정 결과:**

→ 24,496개로 압축

---

**🔍 Step 3: Diversity 필터링**

**분포 고려 방법:**

- Claude 3.5 Sonnet을 사용해 **Mathematics Subject Classification (MSC)** 기준으로 분류
    
    예: 대수학, 해석학, 물리학, 생물학, 논리학 등 50개 도메인
    

**샘플링 방식:**

- 무작위로 도메인 하나 선택
- 그 안에서 추론 trace가 긴 문제일수록 확률 높게 뽑음
- 이를 반복해 **1,000개 도메인 다양성 보장된 최종 셋 완성**

---

**🔍 부가 정보**

- **정답률**: s1K 전체에서 Claude 채점 기준 정답률은 53.6%
- 정답이 틀린 문제도 일부 포함 → 이유: **모델이 답을 틀리더라도 추론 과정이 중요하다는 철학**

---

[작성중...]
