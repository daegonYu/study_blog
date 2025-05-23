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


**📏 3.2 Metrics — 좋은 Test-time Scaling 방법의 3대 조건**


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
    
# 5. Ablations

**🧪 5.1 Data Ablation — 왜 s1K가 특별한가?**

**🎯 목적**

> s1K(1,000개 고품질 데이터)가 단순히 운이 좋은 게 아니라,
> 
> 
> **정말로 ‘좋은 구성 방식’으로 선택되었는가?** → 실험으로 검증
> 

---

**🧠 실험 대상: s1K vs 대체 방식**

| 이름 | 구성 방식 | 특징 |
| --- | --- | --- |
| **1K-random** | s1K 풀에서 무작위 1K | Quality만 반영 (Difficulty, Diversity 無) |
| **1K-diverse** | 도메인 다양성 최우선 | 난이도 고려 안 함 |
| **1K-longest** | 추론 token 가장 긴 1K | 오직 Difficulty 기반 |
| **59K-full** | 전체 59K 학습 | 최대한의 양 (GPU 394시간 필요) |
| **s1K** | Quality + Difficulty + Diversity 기반 | 본 논문 핵심 방식 (GPU 7시간만 필요) |

---

**📊 Table 2 해설**

| Model | AIME24 | MATH500 | GPQA | 의미 |
| --- | --- | --- | --- | --- |
| **1K-random** | 36.7 | 90.6 | 52.0 | 무작위는 확실히 낮음 (성능 저하 확인) |
| **1K-diverse** | 26.7 | 91.2 | 54.6 | 도메인 다양성만 있어도 성능 저조 |
| **1K-longest** | 33.3 | 90.4 | **59.6** | GPQA는 높지만 전체적으로 s1K보다 약함 |
| **59K-full** | 53.3 | 92.8 | 58.1 | 양을 늘리면 좋아지긴 하나, 비용↑ |
| **s1K** | 50.0 | **93.0** | 57.6 | 압도적인 효율성 (성능과 비용의 균형) |

✅ 결론: Quality + Difficulty + Diversity **세 가지를 모두 고려해야 최고의 sample 효율성 확보 가능**

---

**🔧 5.2 Test-time Scaling Method Ablation**

**🎯 목적**

> 다양한 방법 중 Budget Forcing(BF)이 왜 가장 효과적인가?
> 

---

**📊 Table 3 해설**

| Method | Control | Scaling (↑좋음) | Performance (↑좋음) | 설명 |
| --- | --- | --- | --- | --- |
| **BF (★)** | 100% | **15** | **56.7** | 완벽한 제어 + 우수한 scaling |
| TCC | 40% | -24 | 40.0 | 모델이 token 수 정확히 못 셈 |
| TCC + BF | 100% | 13 | 40.0 | BF는 제어 되지만 성능은 낮음 |
| SCC | 60% | 3 | 36.7 | 스텝 단위 제어, 효과 미미 |
| SCC + BF | 100% | 6 | 36.7 | 제어는 OK, scaling은 낮음 |
| CCC | 50% | 25 | 36.7 | “길게 생각해라” 식의 프롬프트, scaling만 좋음 |
| RS (Rejection Sampling) | 100% | **-35** | 40.0 | 성능은 낮고 scaling은 반대로 작동함 (!) |

→ **BF만이 세 가지 지표(제어력 + scaling + 성능) 모두 우수**

---

**🧠 Rejection Sampling 왜 안 될까?**

| Compute 제약 | 결과 경향 |
| --- | --- |
| 짧은 generation (≤4000 tokens) | 바로 정답 접근, 성능 ↑ |
| 긴 generation (≥8000 tokens) | 시행착오 반복 → 정답에서 멀어짐 |

> 🎯 Hypothesis: 모델이 초기에 "감이 올 때"는 빨리 맞추고,
> 
> 
> **헷갈릴수록 토큰 수만 늘고 오히려 실패함**
> 

---

**📏 Table 4 – “Wait” vs 다른 문자열 비교**

| 방식 | AIME24 | GPQA | 의미 |
| --- | --- | --- | --- |
| No extrapolation | 50.0 | 57.6 | 기본값 |
| 2x without string | 50.0 | 55.1 | “Wait” 없이 그냥 더 기다리게 |
| “Alternatively” | 50.0 | 59.6 | 대안 탐색 유도 효과 O |
| “Hmm” | 50.0 | 59.6 | 내적 회의 유도 |
| **“Wait” (★)** | **53.3** | **59.6** | 가장 단순하면서도 효과 가장 좋음 |

✅ 결론: **“Wait”이라는 단어가 가장 범용적이며 반복적인 reasoning을 잘 유도**

# 부록(상세 설명)

(My view: 사실 부록이 핵심이라고 생각한다.)

## C. s1K details

**✅ C.3 Grading Prompt – 정답 여부는 어떻게 판단했는가?**

**🧠 목적**

s1K를 만들 때 각 문제에 대해 LLM이 낸 답이 **맞았는지 틀렸는지 판단**이 필요합니다.

이를 위해 **Claude 3.5/3.7** 모델을 채점기로 사용하며, 다음과 같은 **프롬프트를 통일해 사용**했습니다.

**📋 프롬프트 구조 (핵심 요약)**

```
# Problem
{문제}

## Attempt
{학생의 답변}

## Correct answer
{정답}

[판단 기준: 숫자면 모호성 없이 같아야 함, 추론 과정이면 reasoning이 타당한지 평가]

Explain your reasoning.
End with: "Yes" or "No"

```

→ Claude가 “답이 맞는지”를 최종적으로 “Yes” / “No”로 명확히 판단하게 만듭니다.

→ Claude 3.7은 **최종 1,000개에 대한 채점**에만 사용 (더 정밀한 버전)

---

**✅ C.4 s1K Diversity Selection – 도메인 다양성은 어떻게 확보했는가?**

**📌 핵심 알고리즘 (Algorithm 1)**

1. 먼저 `24,496`개의 후보 중 **AIME, GPQA**, **긴 reasoning trace가 있는 MATH 문제**를 먼저 확보
2. 그 외에는 도메인 기반으로 아래 방식으로 추출:
    - 랜덤하게 하나의 도메인 선택 (예: 기하학, 생물학 등)
    - 그 도메인 내 문제들을 **thinking token 길이로 정렬**
    - 길이가 길수록 sampling 확률 증가 (power-law 분포)
    - 하나 뽑아서 전체에서 제거
    - 50개 도메인 돌아가며 반복 → 총 1,000개 확보

👉 **긴 reasoning을 더 선호하면서**, 도메인 다양성도 보장되도록 설계

---

**✅ C.5 Decontamination – 평가셋 중복 제거는 어떻게?**

- 평가셋(MATH500, GPQA Diamond, AIME24)과 **8-gram 중복 검사**를 시행

---

# ✅ D. Training Details – 훈련 세팅은 어떻게 했는가?

| 항목 | 내용 |
| --- | --- |
| 🔧 모델 | Qwen2.5-32B-Instruct (이미 SFT된 상태에서 추가 학습) |
| 🧠 학습 대상 | 질문 자체는 학습 X, 오직 reasoning trace + 답변에만 loss 부여 |
| ⚙️ 하이퍼파라미터 | batch 16, epoch 5, 총 315 step, lr=1e-5, AdamW |
| ⏱️ 훈련 시간 | 단 26분 (16×H100 GPU) |
| 💾 정밀도 | bfloat16, warmup 5%, cosine decay |

![image](https://github.com/user-attachments/assets/3fe10b29-426c-4a87-984b-442c4e3e20b5)


(My view: 학습 로그 중 Gradient Norm 부분에 150step 쯤 확 튀는 것을 볼 수 있다. 나의 학습의 경우도 저렇게 튀는 모습이 있었는데 그것이 크게 이상한 점이 아니라는 것을 깨달았다.)

**D.1. Training Ablations: Sequence length**

> “훈련 시 사용한 sequence length가 실제 reasoning 성능과 추론 시 비용에 어떤 영향을 미치는가?”
> 

---

**📊 훈련 sequence length 와 성능의 관계**

| 모델 | 훈련 sequence length 길이 | 학습 데이터 잘림 비율 | AIME24 | MATH500 | GPQA |
| --- | --- | --- | --- | --- | --- |
| **Model A** | 4,096 | 74% 잘림 | 30.0% / 20,721 tokens | 90.0% / 5,324 | 52.5% / 6,841 |
| **Model B** | 32,768 | 0% 잘림 | **50.0% / 6,984 tokens** | **91.0% / 3,268** | **53.0% / 3,568** |

✔️ 분모는 inference 시 사용된 평균 reasoning token 수

→ 즉, **정확도는 높고 reasoning 길이는 짧을수록** 좋음 (속도/비용/정확도 모두 잡는 구조)

---

**📌 원인과 해석**

**🔴 문제점: 짧은 시퀀스 길이 (4096)**

- 훈련 시 전체 샘플의 **74%가 중간에 잘림(cutoff)** → 특히 **정답(answer)** 부분이 잘림
- 이로 인해:
    - 모델은 **답을 생성하는 경험이 부족**
    - 추론 시 **답을 내기 전에 reasoning을 질질 끄는** 경향 (→ 길어짐)

---

**🟢 해결: 긴 시퀀스 길이 (32768)**

- 거의 모든 샘플이 **완전하게 포함됨**
- 모델이 다음과 같은 습관을 학습:
    - `chain of thought` → 곧바로 `final answer`로 이어짐
    - → 추론 과정에서도 **짧고 효율적인 reasoning** 유도
- 결과적으로:
    - **성능 향상 + 추론 토큰 수 감소** = 비용 절감 효과도 큼

---

**✅  Training Ablations: Sequence length 요약**

| 항목 | 요약 |
| --- | --- |
| 📉 짧은 시퀀스 | 정답 학습 부족 → reasoning 길어짐, 성능 저하 |
| 📈 긴 시퀀스 | 완전 학습 → reasoning과 정답 간 연결 강함 → 추론 효율 ↑ |
| 🔁 실전 Tip | **sequence length를 무작정 줄이면 추론 비용이 높아지고 성능도 떨어질 수 있음** |
| 🧠 모델 행태 | 시퀀스 길이는 “답까지 자연스럽게 이어지는 경험”을 줄 수 있는 핵심 하이퍼파라미터 |

(My view: 당연하다고 생각 들 수 있지만 나의 경우 GRPO 학습 시(이 논문은 SFT로 학습하였습니다.) 충분한 sequence length로 학습했을 때 그렇지 않을 때보다 loss는 더 잘 수렴하고 Reward std는 줄고 총 Reward 값은 상승하는 효과를 보았다. 적절한 sequence length로 학습하는 것은 중요하다. 물론 GPU VRAM 소모량이 크겠지만 모델 성능을 높이기 위해서는 양보하면 안될 것 같다.)


# 부가 설명

**1. 🔍 오라클 방식 (Rejection Sampling as Oracle)**

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

**✅ 비교 요약: 오라클 vs Budget Forcing**

| 항목 | Oracle (Rejection Sampling) | Budget Forcing |
| --- | --- | --- |
| 접근법 | "만족할 때까지 뽑기" | "처음부터 길이 강제" |
| 제어력 | 낮음 (샘플링 실패 가능) | 높음 (100% 제어 가능) |
| 효율성 | 매우 낮음 (계산 비용 큼) | 매우 높음 |
| 실제 적용 | 실용적 아님 (이론적 upper bound) | 실용적, 단순한 구현 가능 |
| 목적 | 성능 최대치 파악 | 실제 scaling 구현 및 테스트 |


### 2. TCC, SCC

**🔢 TCC (Token-Conditional Control)**

**🧠 개념**

- 모델에게 직접 “생각은 300토큰까지만 해”와 같은 **정확한 토큰 수 조건**을 주는 방식입니다.
- 예시 프롬프트:
    
    > “Please think carefully and stop after 300 tokens of reasoning before giving your final answer.”
    > 

---

**🚫 문제점**

1. **LLM은 자신이 몇 token을 생성했는지 모르기 때문에**,
    
    실제로 300 token 넘겨도 계속 reasoning하거나 더 짧게 끝내버리는 경우가 많습니다.
    
2. 이 방식은 프롬프트 기반 유도이기 때문에 **명시적인 강제성이 없습니다.**

---

**💡 결론**

> TCC는 이론적으로 정밀해 보이지만,
> 
> 
> LLM의 내부 구조상 **자기 토큰 수를 정확히 셀 수 없기 때문에** 실패합니다.
> 

---

### 🔁 SCC (Step-Conditional Control)

**🧠 개념**

- reasoning을 여러 “스텝”으로 나누고, “3스텝까지만 생각해”처럼 **step 수를 제한**
- 각 스텝은 보통 약 100 token 내외로 간주
- 예시 프롬프트:
    
    > “Think in exactly 3 reasoning steps before giving your answer.”
    > 

---

**⚠️ 문제점**

1. 모델이 **스텝 개수는 맞추지만**,
    
    실제로는 **한 스텝에 토큰을 몰아서 쓰는 방식**으로 **제약을 우회**합니다.
    
    - 예: “1단계: 매우 길게 설명… 2단계: 짧게… 3단계: 끝”
2. 스텝 수를 바꿔도 **총 토큰 수는 거의 일정하거나 증가하지 않음**

---

**✅ 두 방식의 한계 정리**

| 항목 | TCC | SCC |
| --- | --- | --- |
| 방식 | “토큰 수 제한” | “생각 단계 수 제한” |
| 문제 | LLM이 token 수 못 셈 | 단계 수는 맞춰도 token 분배를 조작 |
| 결과 | 제어력, 성능, scaling 모두 낮음 | scaling 거의 없음 |
| 결론 | **직접적인 강제력이 없어 실패** | **형식만 제어, 실질적 compute 제어 실패** |

---

**왜 Budget Forcing이 우월한가?**

- **프롬프트가 아닌 디코딩 중 직접 개입 (e.g. “Wait” 추가, end-of-thinking 억제)**
- → 제어력이 **100%**이며, scaling도 긍정적

