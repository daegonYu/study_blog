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

# 3. Test-time scaling


**🔧 3.1 Method — Test-time Scaling 기법 분류와 Budget Forcing**


**✅ [분류] Test-time Scaling 유형**

| 유형 | 설명 | 예시 |
| --- | --- | --- |
| **Sequential** | 앞의 추론 결과가 뒤에 영향을 미치는 방식 | “Wait”를 반복하며 추론 trace 연장 |
| **Parallel** | 여러 개 답을 독립적으로 만들고 결합 | 예: 다중 샘플 후 majority voting |
- 저자들은 **Sequential 방식이 더 강력하다**고 주장합니다.
    
    → 이유: 이전 결과를 바탕으로 **점진적 개선과 더 깊은 reasoning** 가능
    
---

**🧠 [제안 방법] Budget Forcing**

**“모델이 너무 빨리 끝내려 할 때 붙잡고 더 생각하게 만들기”**

→ 핵심 아이디어: 생각 시간을 “강제로 늘리거나 줄이는” 디코딩 간 개입

▪ 최대 토큰 강제 종료 (Early Exit)

- 추론 단계에서 일정 token 수 초과 시 →
    
    → `"end-of-thinking"` 토큰이나 `"Final Answer:"`를 **강제로 삽입**
    
- 목적: **더 이상 생각하지 말고, 지금까지 생각한 것 기반으로 답을 내라**

**▪ 최소 토큰 강제 연장 (Wait Loop)**

- 모델이 생각을 멈추려 할 때 →
    
    → `"Wait"`라는 토큰을 reasoning trace 끝에 **강제로 추가**
    
- 결과: 모델이 중간 결과를 **다시 점검하거나 다른 방향을 탐색**

---

**🔬 [비교 대상] Baselines**

| 방법 | 설명 |
| --- | --- |
| **Conditional length-control** | prompt에 “생각 5단계만 해” 식으로 유도 |
| 1) Token conditional | 토큰 수로 제어 (예: 500자 이내) |
| 2) Step conditional | “생각 단계” 수로 제어 (예: 3단계만 생각) |
| 3) Class conditional | “짧게 생각” / “길게 생각” 같은 일반 지시어 |
| **Rejection sampling** | 여러 샘플 중 compute budget에 맞는 것만 사용 (오라클(하단 부연설명 참고) 방식) |


**📏 3.2 Metrics — **좋은 Test-time Scaling 방법의 3대 조건**


**✅ ① Control (제어력)**

**정의**: 모델이 원하는 **compute 범위 안에서 생각을 끝내는 비율**

- 보통은 최대 토큰 수(`a_max`)만 제약함
- **값이 100%에 가까울수록**, 정확히 원하는 생각 시간 안에서 작동함

---

**✅ ② Scaling (스케일링 기울기)**

정의: **더 많은 생각(token)**을 할수록 **정확도가 얼마나 올라가는가?**

→ 실제로는 정확도 곡선의 평균 기울기 계산

- 양의 기울기여야 의미 있음
- **기울기 크기**가 클수록 → scaling 효과 큼 (적은 비용으로 성능 큰 향상)

---

**✅ ③ Performance (최고 성능)**

정의: 가능한 토큰 범위 내에서 달성한 **최고 정확도**

- 최종 목표: 높은 성능에 도달하는 것
- 하지만 어떤 방법은 token을 아무리 늘려도 flat하거나 context 한계로 fail함

---

**📊 Figure 4 요약**

| 그림 | 설명 |
| --- | --- |
| **(a)** Sequential Scaling – Budget Forcing | “Wait”를 2/4/6번 넣을수록 → 정확도 증가→ **선형적으로 성능이 향상**됨 (좋은 scaling 곡선) |
| **(b)** Parallel Scaling – Majority Voting | 64번 샘플링 후 다수결→ scaling 효과는 낮음 |

# 4. Result

**🧪 4.1 Setup — 실험 세팅**

**🛠 모델 훈련**

- **Base model**: `Qwen2.5-32B-Instruct`
- **방법**: SFT (Supervised Fine-Tuning)만 진행
- **데이터**: `s1K` (앞에서 만든 1,000개 고품질 reasoning 샘플)
- **훈련 시간**: 단 26분 (PyTorch FSDP, 16 × H100 GPU 사용)
- **결과 모델**: `s1-32B`

---

**🧪 평가 데이터셋**

| 데이터셋 | 설명 |
| --- | --- |
| **AIME24** | 2024 미국 고등학생 수학 경시 문제 30개 |
| **MATH500** | OpenAI가 사용한 수학 난이도 벤치마크 500개 |
| **GPQA-Diamond** | 박사 수준 과학 문제 198개 (생물, 화학, 물리) |

> AIME24는 벡터 그래픽(Asymptote) 기반 수식 포함, 모델은 이미지 입력 불가 → 코드 입력으로 대체
> 

---

**🧪 비교 대상 모델들**

1) 💬 OpenAI 시리즈 (비공개 모델)

- o1-preview, o1-mini, o1 (모두 Test-time Scaling의 대표 모델)

2) 📖 공개 모델 (Open Weights)

- Qwen2.5-32B
- QwQ-32B
- DeepSeek r1 (SFT만 사용, 80만 샘플)
- Sky-T1, Bespoke-32B (17K 샘플 사용)

3) 🧠 Gemini 2.0 Flash Thinking (Google)

- s1K의 reasoning trace 생성에 사용된 API
- 평가 오류(“recitation error”)로 일부 테스트셋은 제외

---

**📊 Table 1 분석 — 모델별 성능 비교**

| 모델 | SFT 데이터 수 | AIME24 | MATH500 | GPQA |
| --- | --- | --- | --- | --- |
| **o1 (OpenAI)** | N.A. | 74.4 | 94.8 | 77.3 |
| **Gemini 2.0** | N.A. | 60.0 | N.A. | N.A. |
| **Qwen2.5-32B** | N.A. | 26.7 | 84.0 | 49.0 |
| **QwQ-32B** | N.A. | 50.0 | 90.6 | 54.5 |
| **r1-32B** | ~800K | 79.8 | 97.3 | 71.5 |
| **Sky-T1** | 17K | 43.3 | 82.4 | 56.8 |
| **Bespoke-32B** | 17K | 63.3 | 93.0 | 58.1 |
| **s1 w/o BF** | 1K | 50.0 | 92.6 | 56.6 |
| **s1-32B (★)** | 1K | 56.7 | 93.0 | 59.6 |

---

**🔍 4.2 Performance 분석**

**📈 Test-time Scaling 효과**

- Figure 1 & Figure 4: Budget Forcing을 통해 **생각을 길게 하면 성능이 증가**
- “Wait”를 2, 4, 6번 넣어볼수록 → AIME24 성능이 계속 증가하다가 **6회 이상 반복 시** 지나친 반복은 루프만 만들고 실질적 reasoning 증진 없음

**📌 핵심 결과**

- **Parallel (majority voting)** 기반 scaling은 `s1-32B` 성능 못 따라감
- → **Sequential 방식이 superior**함을 실험적으로 입증

---

**🥇 Sample Efficiency**

> 같은 모델(Qwen2.5-32B) 기반인데 단 1K 샘플로 성능이 이만큼 올라감
> 
- `Qwen2.5-32B`: 26.7 (AIME24)
- `s1-32B`: 56.7 (**2배 이상 향상**)
- r1은 성능은 더 좋지만 **800K 샘플 사용** → s1-32B보다 **800배 덜 효율적**

**💡 요약**

- **단 1K SFT 데이터 + Budget Forcing으로 59.6% GPQA까지 도달**
- Gemini 2.0에서 reasoning trace를 distill했는데, AIME24 기준 거의 근접함
    
    → **Distillation 품질도 우수했음을 반증**
    
---




# 부가 설명

1. 🔍 오라클 방식 (Rejection Sampling as Oracle)

💡 기본 개념

오라클 방식은 **"이론적으로 가장 잘 작동할 수 있는 최적의 상황을 가정"**하는 평가 기준입니다.

이 논문에서는 다음과 같은 방식으로 오라클을 정의합니다:

> “여러 번 샘플링해서, 그 중에서 우리가 원하는 조건(= 정해진 compute budget)에 딱 맞는 것만 골라낸다.”
> 

즉,

1. 모델이 여러 번 답을 생성하게 합니다 (예: temperature 1.0, 다양성 확보)
2. 각 생성 결과를 검사해서 **thinking token 수가 우리가 원하는 범위 안에 있는 것만 채택**
3. 이 과정을 반복해 조건에 맞는 답이 나올 때까지 샘플링

---

📌 왜 오라클인가?

- 이런 방식은 현실적으론 **비효율적**이지만,
    
    "조건을 완벽히 만족하는 결과만 쓴다"는 점에서
    
    **성능 측면에서는 가장 이상적인 upper bound**가 됩니다.
    

---

💣 한계점

| 문제 | 설명 |
| --- | --- |
| ⏱️ **비용 큼** | 조건 만족할 때까지 샘플링 → 매우 많은 시도 필요 |
| ❌ **불확실함** | 조건을 만족하는 답이 아예 없을 수도 있음 |
| 🧠 **이해도 낮음** | 모델이 "왜" 그 답을 골랐는지 reasoning 과정을 알기 어려움 |

---

✅ 비교 요약: 오라클 vs Budget Forcing

| 항목 | Oracle (Rejection Sampling) | Budget Forcing |
| --- | --- | --- |
| 접근법 | "만족할 때까지 뽑기" | "처음부터 길이 강제" |
| 제어력 | 낮음 (샘플링 실패 가능) | 높음 (100% 제어 가능) |
| 효율성 | 매우 낮음 (계산 비용 큼) | 매우 높음 |
| 실제 적용 | 실용적 아님 (이론적 upper bound) | 실용적, 단순한 구현 가능 |
| 목적 | 성능 최대치 파악 | 실제 scaling 구현 및 테스트 |
