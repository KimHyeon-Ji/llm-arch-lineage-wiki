# 08. Decoding — 한 스텝에 토큰 하나여야 하나

`00-foundations` `0.6`에서 본 사실을 다시 꺼내보자.

일반적인 dense decode에서 토큰 하나를 만들려면 **모델 가중치 대부분을 읽어야 한다.**
BF16 70B 모델의 이론적 weight 크기는 약 140 GB다. 계산은 바이트당 1 FLOP 남짓이고,
H100 SXM의 BF16 dense 기준 균형점은 약 295 FLOP/byte다.

즉 이렇게 된다.

> **GPU 연산 능력의 1% 미만만 쓰고 있다.**
> 나머지 99%는 데이터가 오기를 기다린다.

이 파일의 발상은 그 남는 능력을 쓰자는 것이다.

> **어차피 가중치를 다 읽었으니, 그 김에 토큰 여러 개를 처리하면 거의 공짜 아닌가?**

`(1, d) × (d, n)` GEMV나 `(4, d) × (d, n)` GEMM이나 **읽어야 할 가중치는 똑같다.**
계산량만 4배가 되는데, 그 계산 능력은 어차피 놀고 있었다.

문제는 하나다. **다음 토큰을 알아야 그 다음 토큰을 만들 수 있다.**
자기회귀 생성의 본질적인 직렬성이다. 이 파일은 그 벽을 우회하는 방법들이다.

## 이 장의 발전 계보와 시스템 영향

| 단계 | 대표 기법 | 해결하려는 병목 | 핵심 아이디어 | 시스템 영향 |
|---|---|---|---|---|
| 기준점 | autoregressive decode | 한 step에 한 token, 매번 weight 전체 read | next-token을 순차 생성 | 낮은 arithmetic intensity, TPOT가 memory bandwidth에 묶임 |
| 외부 초안 | **Speculative Decoding** | target model 호출 횟수 | 작은 draft가 여러 token을 제안하고 target이 한 번에 검증 | target weight read amortization, draft memory·rollback 추가 |
| 내부 초안 | **Medusa** | 별도 draft model 관리 | 여러 prediction head가 미래 token 후보 생성 | 배포 단순화, head·tree verification compute 증가 |
| feature 예측 | **EAGLE** | token-level draft의 불확실성 | hidden feature를 예측해 후보 생성 | acceptance 향상 가능, 별도 feature predictor 필요 |
| 학습 목표 통합 | **MTP** | 미래 token 표현과 초안 품질 | 학습 때 여러 미래 token을 동시에 예측 | 학습 compute·모듈 증가, 추론 가속과 품질 개선을 함께 노림 |

> **이 장의 병목 이동:** 한 token마다 반복되는 **weight memory traffic**을 여러 후보에
> 나눠 쓰는 대신, 성능이 **acceptance rate·검증 compute·batch 크기**에 의존하게 된다.
> 작은 batch에서는 latency를 줄이지만 큰 batch에서는 남는 compute가 적어 이득이 줄어든다.

---

## 계보 지도

```
        next-token prediction
        한 스텝에 토큰 하나
                │
        [추측하고 검증한다]
                │
      Speculative Decoding (2022)
      작은 draft 모델 + 큰 target 검증
                │
        ┌───────┴────────┐
        │                │
  [draft 모델이 부담]  [학습 목표를 바꾼다]
        │                │
   Medusa (2024)        MTP (2024)
   여러 head            다음 k개를 예측
   EAGLE (2024)         (DeepSeek-V3)
   feature 수준 예측         │
        │                    │
        └────────┬───────────┘
                 │
        MTP 모듈을 draft로 재활용
        (별도 모델 불필요)
```
> **그림 8.0** — 축8의 계보

**목차**

| | 절 | 한 줄 |
|---|---|---|
| [8.1](#81-speculative-decoding--추측하고-검증하기) | **Speculative Decoding** | 추측하고 검증하기 |
| [8.2](#82-medusa와-eagle--draft-모델-없이) | **Medusa · EAGLE** | draft 모델 없이 |
| [8.3](#83-mtp--학습-목표를-바꾸다) | **MTP** | 학습 목표를 바꾸다 |
| [8.4](#84-시스템-관점--acceptance-rate가-모든-것을-정한다) | **시스템 관점** | acceptance rate가 전부 ★ |

---

## 8.1 Speculative Decoding — 추측하고 검증하기

### 아이디어

작은 모델은 빠르고 큰 모델은 정확하다. 둘을 조합한다.

```
 ① draft:  작은 모델이 γ개 토큰을 빠르게 추측
            "the cat sat on the"

 ② verify: 큰 모델이 그 γ+1개를 한 번에 검증 (병렬 forward)
            the ✓  cat ✓  sat ✓  on ✗ ...

 ③ accept: 맞는 데까지 채택, 틀린 지점은 큰 모델의 답으로 교체
            "the cat sat" + (큰 모델이 고른 다음 토큰)
```
> **그림 8.1** — 3개를 채택하면 큰 모델의 forward 한 번으로 4개 토큰이 나온다.

②가 핵심이다. **큰 모델이 γ+1개를 한 번의 forward로 검증한다.**
가중치는 어차피 한 번만 읽으므로, 검증 비용이 토큰 하나 만들 때와 거의 같다.

### 두 모델은 실제로 어떻게 같이 도나

여기서 막히기 쉽다. "작은 모델"과 "큰 모델"이 정확히 뭐고, 둘이 어떻게
**협업**하는지가 안 보이기 때문이다. 하나씩 짚어보자.

**둘은 완전히 다른 체크포인트다.** 예를 들어 target이 Llama-2-70B라면, draft는
Llama-2-7B처럼 **같은 계열의 훨씬 작은 모델**을 쓰는 게 실전에서 가장 흔하다.
아키텍처가 아예 달라도(레이어 수·attention 종류·MoE 여부 등) 원칙적으로는
상관없다 — 딱 하나만 맞으면 된다.

> **draft와 target은 반드시 같은 tokenizer(vocab)를 써야 한다.**
> 검증이 "토큰 ID 하나하나의 확률"을 비교하는 것이기 때문이다. 서로 다른
> 방식으로 텍스트를 쪼개면 애초에 비교할 대상이 없다. 반대로 말하면
> **vocab만 맞으면 draft의 내부 구조는 target과 전혀 무관해도 된다** —
> 심지어 별도로 distill해서 학습한, 태생부터 다른 작은 모델도 쓸 수 있다.

**둘 다 같은 서버, 같은 GPU 메모리 위에 동시에 올라간다.** 채팅으로 물어보고
답을 받는 것처럼 "두 모델이 대화"하는 게 아니다. **하나의 추론 프로세스 안에서
두 forward 함수를 순서대로 호출하는 것**에 가깝다.

```
                     한 대의 서버 / 같은 GPU
 ┌──────────────────────────────────────────────────────┐
 │  draft model (7B, 자기만의 작은 KV cache)              │
 │    "the" → "cat" → "sat" → "on"   (γ=4번, 순차 decode) │
 │                     │                                  │
 │                     ▼  토큰 ID γ개를 그대로 넘김          │
 │  target model (70B, 자기만의 큰 KV cache)               │
 │    [the, cat, sat, on] 이 이어진다고 가정하고            │
 │    각 위치의 확률분포를 **한 번의 forward**로 동시에 계산   │
 │                     │                                  │
 │                     ▼                                  │
 │  accept/reject 비교 (프로세스 안의 그냥 숫자 비교)         │
 └──────────────────────────────────────────────────────┘
```
> **그림 8.1b** — "API로 주고받기"가 아니라, 한 프로세스가 작은 모델을 γ번
> 돌려 토큰을 뽑고, 그 토큰들을 큰 모델에 **입력으로** 넣어 한 번에 확률을
> 매기게 하는 것이다.

정리하면 이렇다.

| 질문 | 답 |
|---|---|
| 큰 모델 = Llama/DeepSeek 같은 실제 서비스 모델? | 맞다. target이 바로 그 모델이다 |
| 작은 모델도 완전히 다른 구조? | 아키텍처는 달라도 되지만, **보통은 같은 계열의 축소판**을 쓴다 (관리·성능 이유) |
| 둘이 "대화"하듯 답을 주고받나? | 아니다. draft가 뽑은 **토큰 ID 나열**을 target의 **입력**으로 그대로 넣는 것뿐이다 |
| 통신이 필요한가(네트워크 호출 등)? | 아니다. 같은 프로세스, 같은 GPU 메모리 안에서 함수 호출하듯 순서대로 실행된다 |

**"검증"이 왜 대화가 아니라 그냥 forward인지**도 짚을 필요가 있다. target
모델은 "이 토큰이 맞아?"라는 질문에 답하는 게 아니라, 원래 하던 일 그대로
**"이 앞 문맥이 주어지면 다음 토큰의 확률분포는 무엇인가"** 를 계산할 뿐이다.
다만 이번엔 그 문맥이 이미 draft가 채워둔 `γ`개짜리 시퀀스라서, 한 번의
forward로 `γ+1`개 위치의 확률분포가 **한꺼번에** 나온다. 그 확률과 draft가
실제로 뽑았던 토큰을 위치별로 비교해서 accept/reject를 정하는 것이 ③번이다.

### 근사가 아니다

여기가 중요하다. speculative decoding은 **출력 분포가 원래 모델과 정확히 동일하다.**
품질을 조금 내주고 속도를 사는 게 아니다.

거부 샘플링(rejection sampling)을 쓰기 때문이다. draft가 제안한 토큰을 target의
확률에 따라 확률적으로 받아들이고, 거부하면 보정된 분포에서 다시 뽑는다.
그 결과 전체 분포가 target 모델 그대로 유지된다.

> 💡 **이 위키에서 "품질 손실이 전혀 없는" 몇 안 되는 기법이다.**
> `01`의 KV 압축도, `05`의 MoE도, `07`의 parallel block도 전부 무언가를 내줬다.
> 여기는 순수하게 낭비되던 연산 능력을 회수하는 것이다.

### 비용

| 무엇 | 왜 |
|---|---|
| draft 모델 실행 | 매 스텝 γ번. 작지만 공짜는 아니다 |
| draft 모델 메모리 | 별도 모델을 상주시켜야 한다 |
| 거부된 토큰의 KV | 일단 계산했다가 버린다 |
| 두 모델 관리 | 어휘가 같아야 하고, 버전 동기화 필요 |

**acceptance rate가 낮으면 draft 비용만 낭비된다.** 이 계열의 성패는 전부 여기 달렸다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | 작은 모델이 여러 토큰을 추측하고, 큰 모델이 한 번의 forward로 검증한다 |
| **장점** | · **출력 분포가 완전히 동일** — 품질 손실이 없다<br>· decode의 남는 연산 능력을 회수한다<br>· 모델 구조를 바꾸지 않아도 된다 |
| **한계** | · **acceptance rate에 성패가 달렸다** — 낮으면 오히려 손해<br>· draft 모델을 따로 두고 관리해야 한다<br>· 거부된 토큰의 KV 계산이 낭비<br>· **배치가 크면 이득이 줄어든다** (`8.4`) |
| **대표 구현** | vLLM, SGLang, TensorRT-LLM 등 주요 프레임워크 전반 |
| **다음으로** | draft 모델을 따로 두는 부담을 없애려면 → **Medusa · EAGLE** |

---

## 8.2 Medusa와 EAGLE — draft 모델 없이

*EAGLE = Extrapolation Algorithm for Greater Language-model Efficiency*

### 문제

draft 모델은 관리 부담이 크다. 메모리를 차지하고, target과 어휘가 맞아야 하고,
target을 바꾸면 draft도 다시 골라야 한다.

**target 모델 자체가 draft 역할을 하게 할 수 없을까?**

### Medusa — head를 여러 개 붙인다

target 모델의 마지막 hidden state에 **추가 head를 여러 개** 붙인다.
head 1은 t+1을, head 2는 t+2를, head 3은 t+3을 예측한다.

> 💡 여기서 "head"는 `01-attention`의 **attention head와 다른 개념**이다.
> Q/K/V를 쪼개는 그 head가 아니라, hidden state 위에 얹은 **작은 출력층 하나**
> (원래 LM head와 똑같이 생긴 `hidden → vocab` linear layer)를 뜻한다.
> 이름만 같을 뿐 서로 아무 관계가 없다.

```
              ┌─► head 0 ─► t+1 (원래 LM head)
 hidden ──────┼─► head 1 ─► t+2
              ├─► head 2 ─► t+3
              └─► head 3 ─► t+4
```

**"각 head가 여러 후보를 낸다"는 게 정확히 무슨 뜻인지 짚어보자.** head 하나는
보통의 LM head와 똑같이 **vocab 전체에 대한 확률분포**를 내놓는다. Medusa는
거기서 1등만 쓰지 않고 **상위 `s`개**를 후보로 남긴다. head 0에서 2개, head 1에서
2개, head 2에서 2개를 남기면 — 이어붙일 수 있는 조합이 `2×2×2 = 8`가지가 되고,
이게 **트리**다.

```
                    (t+1 후보)          (t+2 후보)         (t+3 후보)
              ┌─► "cat" ──┬─► "sat" ──┬─► "on"
 hidden ──────┤           │           └─► "by"
              │           └─► "ran" ──┬─► "to"
              │                       └─► "in"
              └─► "dog" ──┬─► ...
                          └─► ...
```
> **그림 8.2** — head마다 상위 `s`개를 남기면 경로 조합이 트리를 이룬다.
> "cat→sat→on"과 "cat→ran→to"는 **서로 다른 후보 시퀀스**다.

**그런데 트리를 "한 번에" 검증한다는 게 어떻게 가능한가?** `8.1`의 검증은 후보가
**일렬로 늘어선 시퀀스 하나**였다 — "the cat sat on the"처럼 앞이 정해지면 뒤가
자동으로 이어졌다. 트리는 다르다. 같은 t+1 자리에 "cat"과 "dog"가 동시에 있고,
서로는 무관해야 한다 ("cat" 다음 "sat"을 계산할 때 "dog" 쪽 경로가 섞여 들어가면
안 된다).

target 모델은 이걸 **tree attention mask**로 푼다. 트리의 모든 노드를 한 시퀀스에
욱여넣고, 각 노드가 attention에서 **자기 조상 노드만 보게** 마스크를 씌운다.
"sat"은 "cat"만 보고 "dog"는 못 보며, "ran"도 "cat"만 보고 "sat"·"on"은 못 본다.
같은 forward 한 번으로 **트리 전체의 모든 경로**를 동시에 검증할 수 있는 이유가
이 마스크다 — `8.1`에서 `γ+1`개 위치를 한 번에 검증하던 것의 트리 버전이다.

메모리 부담은 작다 — 별도 모델이 없고, head는 hidden state 위의 작은 linear
layer 몇 개뿐이다.

한계는 head들이 서로를 못 본다는 것이다. head 2는 head 1이 실제로 뭘 골랐는지
모르고 (같은 원래 hidden state만 보고) t+3을 예측한다. 트리를 넓게 펼쳐도
**각 head 예측 자체의 정확도**는 원래 hidden state 하나에 갇혀 있는 셈이라
acceptance rate가 제한적이다.

### EAGLE — feature 수준에서 예측한다

EAGLE의 개선은 Medusa의 바로 그 한계를 겨냥한다. **토큰이 아니라 feature(hidden
state)를 예측**하고, **이전 단계에서 예측한 feature를 다음 단계의 입력에 그대로
반영**한다. Medusa의 head들이 서로 독립적으로 원래 hidden state만 봤다면, EAGLE은
자기가 만든 결과를 다음 예측에 다시 넣는 **진짜 자기회귀** 구조다.

```
 Medusa:  hidden ──► head1 ──► t+1 후보
          hidden ──► head2 ──► t+2 후보     (둘 다 같은 hidden만 봄, 서로 무관)

 EAGLE:   hidden_t ──► 예측기 ──► hidden'_{t+1} ──► LM head ──► t+1 후보
                                     │
                                     └──► 예측기 ──► hidden'_{t+2} ──► t+2 후보
                                          (직전 예측을 이어받아 계속 진행)
```

토큰 수준의 불확실성("다음 단어가 정확히 뭘까")보다 feature 수준의 불확실성
("다음 hidden state가 대략 어느 방향일까")이 다루기 쉽다는 관찰에 기반하며,
**직전 단계의 정보를 이어받는다**는 점에서 Medusa보다 acceptance rate가
눈에 띄게 높다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | 별도 draft 모델 대신 target 모델에 붙인 head(Medusa) 또는 feature 예측기(EAGLE)로 초안을 만든다 |
| **장점** | · **별도 모델이 필요 없다** — 메모리와 관리 부담 감소<br>· target과 어휘·표현이 자동으로 일치<br>· EAGLE은 acceptance rate가 높다 |
| **한계** | · head를 학습시키는 추가 단계가 필요<br>· Medusa는 head 간 의존성이 없어 acceptance가 제한적<br>· 트리 검증은 구현이 복잡하고 커널 지원이 필요 |
| **대표 구현** | 주요 서빙 프레임워크에서 옵션으로 제공 |
| **다음으로** | head를 나중에 붙이지 말고 **처음부터 그렇게 학습**하면 → **MTP** |

---

## 8.3 MTP — 학습 목표를 바꾸다

*Multi-Token Prediction*

### 왜 나왔나

Medusa의 head는 사후에 붙인 것이다. 본체는 여전히 "다음 토큰 하나"만 예측하도록
학습되었다.

**처음부터 여러 토큰을 예측하도록 학습하면** 어떨까.

### 아이디어

학습 목표를 next-token 하나에서 **다음 `k`개**로 확장한다.

```
 기존:  t 까지 보고  →  t+1 을 맞춰라
 MTP:   t 까지 보고  →  t+1, t+2, ..., t+k 를 맞춰라
```

DeepSeek-V3는 다음 토큰 외에 한 토큰을 추가로 예측하는 MTP 모듈을 사용했다.

### MTP는 어떻게 생겼나 — Medusa와 뭐가 다른가

`8.2`의 Medusa head들은 서로 독립적으로 **원래 hidden state 하나만** 보고
각자 t+1, t+2, t+3을 예측했다. MTP는 다르게 짠다 — **직전 모듈의 출력을 다음
모듈의 입력으로 이어받는 사슬(chain)**이다.

```
 Medusa:  hidden ──► head1 ──► t+1       hidden ──► head2 ──► t+2
                     (서로 독립, 둘 다 원래 hidden만 봄 — 8.2 참고)

 MTP:     본체 ──► hidden_t ──► t+1 예측
                       │
                       ▼
                  MTP 모듈 1  (입력: hidden_t + t+1의 embedding)
                       │
                       ▼
                  hidden'_t ──► t+2 예측
```

MTP 모듈은 "이전 표현 + 방금 나온 토큰의 embedding"을 받아 다음 표현을 만드는
작은 블록이다. **인과적 사슬을 유지한다**는 점에서 Medusa보다 `8.2`의 EAGLE에
훨씬 가깝다 — 이 구조 덕분에 (같은 위치를 예측하더라도) Medusa보다 acceptance
rate가 좋다.

### 학습 때와 추론 때, 같은 모듈이 다르게 쓰인다

**학습 때는 정답을 이미 안다.** t+1, t+2, ..., t+k가 전부 데이터에 있으니
MTP 모듈 전부를 **한 번에 병렬로** 학습시킬 수 있다 (teacher forcing) — 모듈
1은 실제 정답 t+1을 입력으로 받아 t+2를 예측하도록 학습된다.

**추론(생성) 때는 정답이 없다.** 그래서 학습 때처럼 병렬로 돌릴 수 없고, 학습된
모듈들을 **순서대로 실행하는 작은 자기회귀 루프**로 다시 쓴다.

```
 ① 본체가 t+1을 정상적으로 만든다
 ② MTP 모듈 1이 (hidden_t, t+1) → t+2 후보를 speculate
 ③ (모듈이 더 있다면) 모듈 2가 (모듈1의 출력, t+2 후보) → t+3 후보를 speculate
 ④ 이렇게 만든 [t+1, t+2, ...]를 본체에 다시 넣어 8.1과 같은 방식으로 검증
```

즉 **"학습된 MTP 모듈이 그대로 draft 역할을 한다"** 는 말은, 학습 때 정답을 넣어
병렬로 가르친 모듈들을 추론 때는 **자기 예측을 서로 이어붙이며 순차 실행**한다는
뜻이다 — 학습 때의 연산 그래프를 그대로 재생하는 게 아니라, 학습된 **가중치**를
재사용해 추론 전용 자기회귀 루프를 새로 도는 것이다.

### 두 가지 효과

**① 학습 품질** — 이게 원래 목적이었다.
한 스텝 앞만 보면 모델이 근시안적으로 학습된다. 여러 스텝을 예측하게 하면
**더 멀리 내다보는 표현**을 배우고, 이는 일반적인 성능 향상으로 이어진다.

**② 추론 가속** — 부수 효과인데 실전에서 매우 유용하다.
위에서 본 자기회귀 루프가 **그대로 draft 역할을 한다.** 별도 모델도, 사후 학습도
필요 없다.

> 💡 **MTP는 `8.1`~`8.2`와 성격이 다르다.**
> speculative decoding과 Medusa는 순수하게 추론 최적화다.
> MTP는 **학습 기법인데 추론 가속이 딸려온 것**이다.
> 그래서 채택 문턱이 낮았다 — 어차피 품질 때문에 넣을 것이었으니까.

### 추론에서 달라진 것

| | 일반 | MTP |
|---|---|---|
| 추가 파라미터 | — | MTP 모듈 (작다) |
| 학습 비용 | 기준 | 약간 증가 |
| 품질 | 기준 | **향상** |
| draft 모델 | 필요 (spec decode 시) | **불필요** |
| 스텝당 토큰 | 1 | acceptance에 따라 1~`k+1` |

### 코드와 텐서

검증 단계에서 무엇이 달라지는지가 핵심이다.

| | 일반 decode | speculative 검증 |
|---|---|---|
| 입력 | `(B, 1, d)` | `(B, **γ+1**, d)` |
| 투영 연산 | `(B,1,d) × (d,·)` → **GEMV** | `(B,γ+1,d) × (d,·)` → **작은 GEMM** |
| 읽는 가중치 | 전부 | **전부 — 똑같다** |
| attention | `(B,n_h,1,S)` | `(B,n_h,γ+1,S)` |

> 💡 **두 번째 줄과 세 번째 줄이 이 파일의 전부다.**
> GEMV가 GEMM이 되면서 연산량은 늘지만, **읽는 가중치는 그대로다.**
> `00-foundations` `0.3`에서 정리한 GEMM/GEMV의 차이가 여기서 이득으로 바뀐다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | 학습 목표를 다음 `k`개 토큰 예측으로 확장한다. 학습된 MTP 모듈이 추론 시 draft가 된다 |
| **장점** | · **품질이 향상된다** — 원래 목적이 이것<br>· **별도 draft 모델 없이** speculative decoding 가능<br>· 모듈이 본체와 함께, 인과적 사슬로 학습되어 acceptance rate가 좋다<br>· 추가 파라미터가 작다 |
| **한계** | · **처음부터 그렇게 학습해야 한다** — 기존 모델에 적용 불가<br>· 학습 비용이 약간 늘어난다<br>· `k`를 키울수록 먼 토큰의 예측 정확도가 급격히 떨어짐 |
| **대표 모델** | **DeepSeek-V3 / V3.2 / V4** |
| **다음으로** | 이 모든 것의 효과는 하나의 숫자에 달려 있다 → **acceptance rate** |

---

## 8.4 시스템 관점 — acceptance rate가 모든 것을 정한다

### 이득의 구조

스텝당 평균 채택 토큰 수를 `τ`라 하자. 처리량이 대략 `τ`배가 된다.
`τ`는 acceptance rate와 draft 길이 `γ`로 정해진다.

| acceptance | 결과 |
|---|---|
| 높음 (0.8+) | 큰 이득. `γ`를 늘릴수록 좋다 |
| 중간 (0.5) | 적당한 이득. `γ`를 너무 키우면 낭비 |
| 낮음 (0.3-) | **draft 비용만 손해** |

acceptance는 **작업 종류에 따라 크게 달라진다.** 코드나 정형화된 텍스트는 예측이
쉬워서 높고, 창의적 생성은 낮다. 같은 시스템이 요청에 따라 다르게 동작한다는 뜻이다.

### 배치 크기와 상충한다

여기가 이 절의 핵심이다.

`00-foundations` `0.6`에서 봤듯 **배치를 키우면 가중치 읽기가 amortize되어
FFN이 compute-bound로 넘어간다.** 그런데 speculative decoding도 같은 원리로 이득을 낸다.

```
 배치가 작다  →  GPU가 놀고 있다  →  speculative가 그 여유를 쓴다  →  큰 이득
 배치가 크다  →  GPU가 이미 바쁘다 →  여유가 없다               →  이득 감소
```

**둘은 같은 자원을 노린다.** 그래서 상충한다.

| 상황 | 유리한 전략 |
|---|---|
| 저지연 요구, 소규모 배치 | **speculative decoding** |
| 고처리량 요구, 대규모 배치 | 배치 키우기 |
| 중간 | 둘의 균형점을 찾아야 함 |

> 💡 **아키텍처 결정이 서빙 시나리오에 의존한다는 패턴이 또 나온다.**
> `05-moe` `5.4`의 granularity도, `07-shape` `7.1`의 depth도 같은 구조였다.
> **"무엇이 좋은가"에 답하려면 "어떤 배치로 서빙할 것인가"를 먼저 정해야 한다.**

### 그 외 시스템 부담

| 무엇 | 왜 |
|---|---|
| **KV cache 증가** | 거부될 토큰의 KV도 일단 계산해서 넣었다가 롤백해야 한다 |
| **지연 변동** | acceptance가 요청마다 달라 스텝 시간이 들쭉날쭉 |
| **스케줄링 복잡도** | 배치 안 요청들이 서로 다른 수의 토큰을 뱉는다 |
| **KV 롤백** | 거부된 부분을 캐시에서 되돌리는 로직 필요 (PagedAttention과 얽힘) |

두 번째가 운영에서 성가시다. 평균 지연은 좋아지는데 **분산이 커진다.**
SLO를 p99로 관리하면 개선이 기대보다 작아 보일 수 있다.

### 정리

이 파일 전체를 한 표로 정리하면 이렇다.

| 기법 | draft 출처 | 학습 필요 | 품질 영향 | 채택 문턱 |
|---|---|---|---|---|
| **Speculative** | 별도 작은 모델 | 없음 | **없음** | 낮음 (모델만 있으면) |
| **Medusa** | 추가 head | head 학습 | 없음 | 중간 |
| **EAGLE** | feature 예측기 | 예측기 학습 | 없음 | 중간 |
| **MTP** | 학습된 MTP 모듈(chain) | **본 학습에 포함** | **향상** | 높음 (재학습) |

MTP가 최근 넓게 퍼진 이유가 마지막 두 열에 있다. **품질 때문에 어차피 넣을 것이었고,
추론 가속은 딸려온 것**이라 채택 결정이 쉬웠다.

---

## 이 다음

여기까지가 "무엇을 계산할 것인가"의 이야기였다.
`09-numerics.md`는 다른 질문을 던진다. **값 하나에 몇 비트를 쓸 것인가.**

그 축은 특이하다. 지금까지의 모든 압력에 **동시에** 작용한다.

---

## Sources

**T1 — 논문**
- Leviathan et al. (2022), *Fast Inference from Transformers via Speculative Decoding*, arXiv:2211.17192
- Chen et al. (2023), *Accelerating Large Language Model Decoding with Speculative Sampling* — 분포 동일성 증명
- Cai et al. (2024), *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*
- Li et al. (2024), *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*
- DeepSeek-AI (2024), *DeepSeek-V3*, arXiv:2412.19437 — MTP 설계와 학습 효과
- Gloeckle et al. (2024), *Better & Faster Large Language Models via Multi-token Prediction* — MTP의 품질 근거

**T2 — 구현**
- vLLM speculative decoding 문서 — acceptance rate 측정, 배치와의 상호작용
- SGLang / TensorRT-LLM의 EAGLE·Medusa 구현
- KV cache 롤백 처리 (PagedAttention과의 연동)

**범위와 주의**
- speculative decoding의 이득은 draft 길이만이 아니라 acceptance rate, 검증 batch,
  sampling 설정과 서빙 부하에 따라 달라진다.
- MTP의 학습 품질 효과와 추론 가속 효과는 구분해서 평가해야 한다.
