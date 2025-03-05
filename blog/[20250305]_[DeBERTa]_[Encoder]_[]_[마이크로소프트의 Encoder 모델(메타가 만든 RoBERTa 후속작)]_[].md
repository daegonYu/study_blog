# DeBERTa: Decoding-Enhanced BERT with Disentangled Attention

저자: Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen (Microsoft)

DeBERTa(Decoding-Enhanced BERT with Disentangled Attention)는 BERT와 RoBERTa를 개선한 모델로, 두 가지 핵심 기술을 도입하여 성능을 향상시켰습니다.

## 1. DeBERTa의 핵심 개선점
### ✅ 1) Disentangled Attention (분리된 어텐션 메커니즘)
기존 BERT는 각 단어를 ‘내용(Content) + 위치(Position)’을 합쳐 하나의 벡터로 표현합니다.

DeBERTa는 내용(Content)과 위치(Position)를 각각 별도의 벡터로 표현하여 어텐션을 계산합니다.

이렇게 하면 단어의 내용과 상대적인 위치를 독립적으로 고려할 수 있어, 문맥을 더 정교하게 이해할 수 있습니다.

#### 📌 왜 상대적 위치 정보가 중요한가?
예를 들어, 문장에서 **"딥 러닝"**이 나란히 있을 때와 서로 떨어져 있을 때는 의미가 다릅니다.

기존 방식은 단어의 절대적 위치를 포함하지만, 상대적 위치를 제대로 반영하지 못했습니다.

DeBERTa는 각 단어 쌍의 상대적 위치도 어텐션 계산에 반영하여 더 효과적으로 문맥을 이해합니다.

### ✅ 2) Enhanced Mask Decoder (강화된 마스크 디코더)
기존 BERT의 MLM(Masked Language Model) 기법을 개선한 방식입니다.

일반적인 MLM에서는 마스크된 단어를 주변 문맥을 이용해 예측하는데, BERT는 절대 위치 정보를 입력 단계에서 추가합니다.

DeBERTa는 절대 위치 정보를 마지막 소프트맥스 이전 단계에서 추가하여, 상대적 위치 정보 학습을 방해하지 않도록 했습니다.

이를 통해 상대적 위치 정보의 학습을 유지하면서도 절대 위치 정보의 이점을 활용할 수 있습니다.

#### 📌 절대적 위치 정보가 중요한 이유
문장에서 “새로운 상점이 새 쇼핑몰 옆에 생겼다” 라는 문장이 있을 때, ‘상점’과 ‘쇼핑몰’은 비슷한 문맥에서 등장합니다.

상대적 위치 정보만 사용하면 둘 다 “새로운”이라는 단어와 같은 거리에 있기 때문에 혼동할 수 있습니다.

하지만 절대 위치 정보를 활용하면 “상점”이 주어(Subject)라는 점을 구분할 수 있습니다.

## 2. DeBERTa의 성능 향상

### 📈 기존 모델과 비교한 성능

DeBERTa는 RoBERTa-Large보다 적은 데이터로도 더 좋은 성능을 보였습니다.

SuperGLUE 벤치마크에서 인간 성능(89.8%)을 최초로 초과하며, SOTA 모델이 되었습니다.

1.5B 파라미터 DeBERTa 모델이 11B 파라미터 T5보다 성능이 높음 → 더 효율적인 구조!

## 3. Scale-invariant Fine-Tuning (SiFT)

DeBERTa는 Fine-tuning을 안정적으로 수행하기 위한 새로운 정규화 기법(SiFT, Scale-invariant Fine-Tuning)을 적용했습니다.

NLP에서는 단어 임베딩의 크기(벡터의 노름)가 단어마다 다를 수 있습니다.

SiFT는 임베딩을 정규화(normalization)한 후 작은 변화를 가하는 방법을 사용하여 학습을 안정적으로 만듭니다.

특히 대형 모델(1.5B 파라미터 이상)에서 성능 향상 효과가 크다고 보고되었습니다.

## 4. 요약 및 핵심 포인트
🔹 DeBERTa = RoBERTa + 2가지 핵심 기술 (분리된 어텐션 + 강화된 마스크 디코더)

🔹 상대적 위치 정보와 절대적 위치 정보를 분리하여 더 효과적으로 학습

🔹 Fine-tuning 시 SiFT 기법을 도입하여 학습 안정성 향상

🔹 SuperGLUE 벤치마크에서 최초로 인간 성능을 초과한 모델


## 💡 쉽게 이해하는 비유
👉 기존 BERT가 **"책을 읽을 때 단어 하나하나를 조합해서 문맥을 이해하는 방식"**이라면,

👉 DeBERTa는 **"단어 간의 거리(위치)를 독립적으로 고려하여 문장을 이해하는 방식"**입니다.

즉, 단어의 의미뿐만 아니라 상대적 위치 정보도 제대로 반영하는 더 똑똑한 모델이라고 보면 됩니다! 🚀

향후 NLP 연구에서 상대적 위치 정보와 절대적 위치 정보를 어떻게 조합할 것인가가 중요한 문제로 떠오를 것

# 활용 사례
https://ai.kakaobank.com/f982c5b8-9cbd-4ef1-8ebd-d7359a70284b
SuperGLUE에서 인간을 초월하는 성능을 기록하며 최고 성능을 달성



