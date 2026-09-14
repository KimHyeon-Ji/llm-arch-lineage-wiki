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

네 가지 서로 다른 방식으로 우회한다 — **① 누군가 미리 추측하고 검증한다
(별도 모델, 또는 같은 모델의 일부)**, **② 추측을 아예 학습 목표에 넣어버린다**,
**③ 추측이라는 개념 자체를 버리고 병렬 방정식으로 재해석한다**,
**④ 추측을 만드는 일마저 병렬화한다 — 초안 블록 하나를 한 번의 forward로 뽑는다.**

## 이 장의 발전 계보와 시스템 영향

| 단계 | 대표 기법 | 해결하려는 병목 | 핵심 아이디어 | 시스템 영향 |
|---|---|---|---|---|
| 기준점 | autoregressive decode | 한 step에 한 token, 매번 weight 전체 read | next-token을 순차 생성 | 낮은 arithmetic intensity, TPOT가 memory bandwidth에 묶임 |
| 외부 초안 | **Speculative Decoding** | target model 호출 횟수 | 작은 draft가 여러 token을 제안하고 target이 한 번에 검증 | target weight read amortization, draft memory·rollback 추가 |
| 자기 초안 | **Self-Speculative (LayerSkip)** | 별도 draft 모델·파라미터 관리 | 같은 모델의 앞쪽 레이어로 초안, 전체로 검증 | 추가 메모리 0, 초안 품질이 target 앞부분에 의존 |
| 추측 자체를 없앰 | **Lookahead Decoding** | draft 모델·추가 학습 없이 병렬성 확보 | Jacobi iteration 궤적에서 n-gram을 뽑아 그 자리에서 검증 | 학습·모델 불필요, 실이득은 draft 기반보다 작은 편 |
| 내부 초안 | **Medusa** | 별도 draft model 관리 | 여러 prediction head가 미래 token 후보 생성 | 배포 단순화, head·tree verification compute 증가 |
| feature 예측 | **EAGLE(-2/3)** | token-level draft의 불확실성, 정적 트리 | hidden feature 예측 + 문맥에 따라 동적으로 자라는 트리 | acceptance 대폭 향상, 별도 feature predictor 필요 |
| 학습 목표 통합 | **MTP** | 미래 token 표현과 초안 품질 | 학습 때 여러 미래 token을 동시에 예측 | 학습 compute·모듈 증가, 추론 가속과 품질 개선을 함께 노림 |
| 병렬 초안 | **DFlash** (block diffusion) | draft 자체가 순차적이라 `γ`에 비례해 늘어나는 초안 지연 | 마스크 블록을 한 번의 forward로 denoise, target feature를 draft 전 레이어 K/V에 주입 | 초안 지연이 `γ`와 무관해짐, 대신 뒤쪽 위치 acceptance 급락(suffix decay) |
| 부하 인지 검증 | **DSpark** | 통과 못 할 draft가 batch 용량을 먹는 검증 낭비 | 병렬 backbone + 1차 마르코프 head, survival 확률과 엔진 처리량 프로파일로 요청별 검증 길이 결정 | 고부하에서도 이득 유지, confidence head·cost table·가변 길이 검증 batch |

> **이 장의 병목 이동:** 한 token마다 반복되는 **weight memory traffic**을 여러 후보에
> 나눠 쓰는 대신, 성능이 **acceptance rate·검증 compute·batch 크기**에 의존하게 된다.
> 작은 batch에서는 latency를 줄이지만 큰 batch에서는 남는 compute가 적어 이득이 줄어든다.
> 마지막 두 줄은 그 다음 단계다 — **초안 생성의 직렬성**까지 없애고 나면, 남는 문제는
> "한정된 검증 예산을 어느 요청의 어느 위치에 줄 것인가"라는 **스케줄링**이 된다.

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
      ┌─────────┼──────────┬───────────────┬──────────────────┐
      │         │          │               │                  │
 [draft 모델   [모델 하나로   [학습 목표를    [추측 자체를       [초안 만드는 일도
  관리 부담]    스스로 추측]  바꾼다]        없앤다]            병렬로]
      │         │          │               │                  │
  Medusa        LayerSkip    MTP (2024)   Lookahead        block diffusion
  (2024)        (2024)       다음 k개      Decoding (2023)   drafting
  여러 head     early-exit   예측          Jacobi iteration  DFlash (2026)
  EAGLE (2024)  self-spec    (DeepSeek-V3)  + n-gram pool    forward 1회 = γ개
  feature 예측       │            │               │                  │
      │              │            │               │            DSpark (2026)
 EAGLE-2/3            │            │               │            semi-AR head
 동적 트리 (2024-25)   │            │               │            + 부하 인지 검증
      │              │            │               │                  │
      └──────┬───────┴─────┬──────┘               │                  │
             │              │                      │                  │
             └──────────────┴──────────────────────┴──────────────────┘
                             │
                 [시스템 쪽 귀결] acceptance rate가
                 모든 이득을 좌우하고, 배치 크기와 상충한다
```
> **그림 8.0** — 축8의 계보. 같은 뿌리(순차 생성의 직렬성)에서 다섯 갈래로 갈라진다.
> 맨 오른쪽 갈래만 **검증이 아니라 초안 쪽**의 직렬성을 겨냥한다.

**목차**

| | 절 | 한 줄 |
|---|---|---|
| [8.1](#81-speculative-decoding--추측하고-검증하기) | **Speculative Decoding** | 추측하고 검증하기 |
| [8.2](#82-self-speculative-decoding--모델-하나로-초안과-검증을-동시에) | **Self-Speculative** | 모델 하나로 초안과 검증을 |
| [8.3](#83-lookahead-decoding--추측하지-않고-병렬로-푼다) | **Lookahead Decoding** | 추측하지 않고 병렬로 |
| [8.4](#84-medusa와-eagle--draft-모델-없이) | **Medusa · EAGLE** | draft 모델 없이 |
| [8.5](#85-mtp--학습-목표를-바꾸다) | **MTP** | 학습 목표를 바꾸다 |
| [8.6](#86-diffusion-초안--초안-자체를-한-번에-만든다) | **Diffusion 초안** | 초안을 블록째 한 번에 |
| [8.7](#87-시스템-관점--acceptance-rate가-모든-것을-정한다) | **시스템 관점** | acceptance rate가 전부 ★ |

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
| **한계** | · **acceptance rate에 성패가 달렸다** — 낮으면 오히려 손해<br>· draft 모델을 따로 두고 관리해야 한다<br>· 거부된 토큰의 KV 계산이 낭비<br>· **배치가 크면 이득이 줄어든다** (`8.7`) |
| **대표 구현** | vLLM, SGLang, TensorRT-LLM 등 주요 프레임워크 전반 |
| **다음으로** | draft 모델을 따로 두는 부담을 없애려면 → **Self-Speculative Decoding** |

---

## 8.2 Self-Speculative Decoding — 모델 하나로 초안과 검증을 동시에

*LayerSkip*

### 왜 나왔나

`8.1`이 안고 있는 가장 큰 관리 부담은 **별도 draft 모델**이었다. `8.4`에서
볼 Medusa·EAGLE은 "target에 작은 head를 붙이자"는 답을 택하는데, 그것도
결국 **새 파라미터를 학습**해야 한다.

더 급진적인 질문을 던질 수 있다. **새 파라미터를 아예 안 만들면 안 되나?**
target 모델 안에는 이미 레이어가 수십 개 있다. 그중 **앞쪽 일부만 쓰면
그 자체로 "작고 빠른 모델"이 되지 않을까.

### 아이디어와 구조

📌 [T1] LayerSkip은 학습 단계에서 두 가지를 함께 한다.

```
 ① layer dropout — 학습 중 뒤쪽 레이어일수록 더 자주 건너뛴다
    (앞쪽 레이어가 "혼자서도 어느 정도 쓸모 있는 표현"을 만들도록 강제)

 ② early-exit loss — 중간 레이어의 hidden state에도 LM head를 씌워
    바로 손실을 계산한다 (본체의 최종 손실과 함께 학습)
```

이렇게 학습하면 **모델의 앞쪽 `E`개 레이어만 통과한 hidden state로도** 그럭저럭
다음 토큰을 맞힐 수 있게 된다 — 원래 모델의 최종 출력에 쓰이는 그 LM head를
그대로, 중간에서도 씌울 수 있는 것이다.

```
 보통 모델:          x → L1 → L2 → ... → L_N → LM head → 토큰
                                           (여기서만 예측 가능)

 LayerSkip 학습 후:  x → L1 → ... → L_E  → LM head → "초안" 토큰
                               │
                               └──► L_{E+1} → ... → L_N → LM head → "진짜" 토큰
```

**draft를 만드는 것도, 검증하는 것도 같은 모델, 같은 가중치다.** 그래서
"self-speculative"다 — `8.1`의 draft/target 두 체크포인트 구도와 근본적으로 다르다.

### 추론에서 달라진 것

| | `8.1` (별도 draft 모델) | Self-Speculative (LayerSkip) |
|---|---|---|
| draft 전용 파라미터 | 별도 체크포인트 전체 | **0 — 같은 모델 재사용** |
| draft 전용 메모리 | 필요 | **불필요** |
| draft·target vocab 문제 | 있음 (버전 동기화 등) | **애초에 없음 (같은 모델)** |
| draft 품질의 근거 | 별도로 학습·distill된 작은 모델 | target **앞쪽 레이어만으로 학습된** 표현력 |
| 적용 대상 | 아무 target에나 (draft만 구하면) | **이 방식으로 학습된 모델만** |

### 토큰 하나가 지나가는 길

```
 ① draft:   x를 L1~L_E까지만 통과 → LM head → 토큰 하나 예측
            (이 앞쪽 E개 레이어를 γ번 반복해서 γ개 토큰을 만든다)
 ② verify:  γ개 토큰을 L1~L_N 전체로 검증
            — ①에서 이미 계산해둔 L1~L_E의 활성값(KV 포함)을 재사용하고,
              나머지 L_{E+1}~L_N만 새로 계산한다
 ③ accept:  `8.1`과 동일한 rejection sampling
```

**②의 재사용이 이 기법의 진짜 최적화 지점이다.** draft와 verify가 완전히
다른 두 forward가 아니라, **앞부분을 공유하는 하나의 forward를 어디서
끊어 쓰느냐**의 문제로 바뀐다.

### 코드와 텐서

| # | 연산 | 비고 |
|---|---|---|
| 1 | draft: `L1~L_E` forward, `γ`회 반복 | 원래 모델의 일부만 씀 — 새 가중치 없음 |
| 2 | verify: `L_{E+1}~L_N` forward, `γ+1`개 위치를 한 번에 | `8.1`처럼 GEMV가 GEMM으로 |
| 3 | verify가 재사용하는 것 | draft 단계의 `L1~L_E` KV·activation |

> 💡 **`8.1`은 "두 모델의 가중치를 각각 읽는" 비용이 있었다.** 여기는 **한
> 모델의 가중치를 부분적으로만 읽는다** — draft 단계는 `E/N`만큼만 읽는다.
> 별도 draft 모델을 서빙하는 부담이 "이 모델을 어디서 끊어 쓸까"라는
> 훨씬 가벼운 문제로 바뀐 셈이다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | 학습 시 앞쪽 레이어도 단독으로 예측 가능하도록 만들고, 추론 때 앞쪽 레이어로 초안을, 전체 레이어로 검증을 한다 |
| **장점** | · **추가 파라미터·메모리가 0** — 별도 모델 관리 불필요<br>· draft-target vocab 불일치 문제가 원천적으로 없음<br>· 검증 단계가 draft 단계의 활성값을 재사용해 낭비가 적음 |
| **한계** | · **이 방식으로 직접 학습(또는 fine-tune)된 모델에만 적용 가능** — 기존 체크포인트에 바로 쓰기는 어려움<br>· 초안 품질이 target 자신의 앞부분에 의존해 EAGLE만큼의 acceptance는 어려움<br>· 논문이 보고하는 speedup은 과제별 1.8~2.2배로 EAGLE 계열(3배 이상)보다 낮은 편 |
| **대표 구현** | Meta LayerSkip · Draft & Verify · SWIFT 등 후속 연구 |
| **다음으로** | 초안을 아예 만들지 않고 병렬로 풀 수는 없을까 → **Lookahead Decoding** |

---

## 8.3 Lookahead Decoding — 추측하지 않고 병렬로 푼다

### 왜 나왔나

지금까지 나온 기법 — `8.1`의 별도 모델, `8.2`의 레이어 재사용, `8.4`의 head,
`8.5`의 MTP 모듈 — 은 전부 **같은 틀 안에 있다.** "누군가(다른 모델이든,
앞쪽 레이어든, head든)가 먼저 추측하고, target이 검증한다."

Lookahead Decoding은 그 틀 자체를 벗어난다. **추측하는 별도의 무언가가
없다.** target 모델 혼자서, 자기회귀 생성을 아예 다른 문제로 바꿔 푼다.

### 아이디어와 구조

📌 [T1] 출발점은 수학적 재해석이다. 순서대로 하나씩 정하는 자기회귀 생성을,
**여러 개의 미지수를 가진 비선형 연립방정식**으로 볼 수 있다. 이런 방정식은
**Jacobi iteration**으로 병렬로 풀 수 있다 — 모든 미지수를 아무 값으로나
초기화한 뒤, **한 스텝에 전부를 동시에** 갱신하는 걸 반복해서 정답(고정점)에
수렴시킨다.

```
 순차 생성(보통 decode):  x1 → x2 → x3 → x4    (한 번에 하나씩, 정확히 하나만)

 Jacobi 생성:    [x1  x2  x3  x4] 를 아무 값으로나 초기화
                      │  한 스텝: 전부 동시에 "지금 값 기준 다음 추정값" 계산
                      ▼
                 [x1' x2' x3' x4']
                      │  또 한 스텝 ...
                      ▼
                 수렴하면 순차 생성과 **정확히 같은** 결과
```

📌 [T1] 문제는 Jacobi 자체는 수렴이 느려서 그것만으로는 이득이 크지 않다는
것이다. Lookahead Decoding의 기여는, **이 반복 궤적(trajectory)에서 스쳐
지나가는 여러 n-gram 후보를 버리지 않고 모아둔다**는 데 있다. 고정 크기의
2D 윈도우(시퀀스 축 × 반복 축)로 궤적을 관리하면서, 서로 다른 위치에서
만들어지는 n-gram들을 **pool**에 쌓는다.

```
 ┌─────────────────────────────────────┐
 │  lookahead branch                    │
 │  Jacobi 반복으로 여러 위치를 동시에 갱신 │
 │  → 지나가는 n-gram들을 pool에 저장     │──┐
 └─────────────────────────────────────┘  │
                                           ▼
                              지금 시퀀스 접두어와 이어지는
                              유망한 n-gram을 pool에서 골라
 ┌─────────────────────────────────────┐  │
 │  verification branch                 │◄─┘
 │  그 n-gram들을 `8.1`과 같은 방식으로   │
 │  target 한 번의 forward로 검증        │
 └─────────────────────────────────────┘
```
> **그림 8.3** — 두 갈래가 **같은 forward 스텝 안에서 동시에** 돈다. lookahead가
> 미래 후보를 계속 만들어내고, verification이 그걸 그때그때 확인한다.

📌 [T1] draft 모델도, 추가 학습도 필요 없다 — **순수하게 추론 알고리즘**이다.
논문은 GPU 한 장 기준 데이터셋에 따라 **1.5~2.3배**의 지연 감소를 보고한다.

### 추론에서 달라진 것

| | `8.1` (별도 draft) | Lookahead Decoding |
|---|---|---|
| draft 모델 | 필요 | **불필요** |
| 추가 학습 | 불필요 (모델만 있으면) | **불필요** |
| 스텝당 계산 위치 수 | `γ+1` (draft가 만든 만큼) | **윈도우 크기 + 후보 n-gram 수** — 조절 가능 |
| 유지할 상태 | draft의 KV cache | **n-gram pool** (텍스트 패턴 캐시) |
| 이득의 원천 | draft 모델의 예측력 | **Jacobi 궤적이 우연히 맞히는 패턴들** |

### 토큰 하나가 지나가는 길

```
 ① 고정 크기 2D 윈도우 안에서 여러 위치를 Jacobi 방식으로 한 스텝 갱신
    (지금 값을 기준으로 "다음 추정값"을 동시에 계산)
 ② 이 갱신 과정에서 스쳐 지나간 n-gram들을 pool에 기록
 ③ 지금 확정된 접두어와 이어지는, pool 안의 유망한 n-gram들을 후보로 뽑음
 ④ ①의 윈도우가 만든 후보 + ③에서 뽑은 후보를 한데 모아 target이 한 번의
    forward로 검증 (`8.1`과 같은 원리)
 ⑤ 맞는 데까지 채택. pool은 계속 갱신되며 다음 스텝에도 재사용됨
```

### 코드와 텐서

| # | 연산 | shape | 비고 |
|---|---|---|---|
| 1 | 윈도우 갱신 | `(B, W×N, d)` 정도로 확장된 입력 | `W`=윈도우 폭, `N`=반복 깊이 |
| 2 | pool에서 후보 조회 | 텐서 연산 아님 | n-gram 매칭 — 해시/캐시 조회 |
| 3 | 검증 | `(B, 후보 개수, d)` → GEMM | `8.1`과 원리 동일 |

> 💡 **draft 모델이 하던 일을 "과거에 이미 스쳐간 패턴의 기억"이 대신한다.**
> `02-linear-attention`에서 상태를 "사전"에 비유했던 것과 닮은 발상이다 —
> 다만 여기서 사전은 **신경망이 아니라 최근 n-gram들의 캐시**다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | 자기회귀 생성을 Jacobi iteration으로 재해석해 여러 위치를 병렬로 갱신하고, 그 궤적에서 나온 n-gram을 그 자리에서 검증한다 |
| **장점** | · **draft 모델도 추가 학습도 필요 없다** — 순수 추론 알고리즘<br>· 출력 분포가 원래 모델과 동일<br>· 어떤 기존 모델에도 바로 적용 가능 |
| **한계** | · 실측 이득(1.5~2.3배)이 EAGLE 계열(3배 이상)보다 작은 편<br>· n-gram pool 관리·매칭에 별도 구현 복잡도가 있음<br>· 반복(iteration) 기반이라 튜닝할 하이퍼파라미터(윈도우 크기 등)가 있음 |
| **대표 구현** | hao-ai-lab LookaheadDecoding (ICML 2024) |
| **다음으로** | 다시 "누군가 추측하는" 쪽으로 — target에 직접 head를 붙이면 → **Medusa · EAGLE** |

---

## 8.4 Medusa와 EAGLE — draft 모델 없이

*EAGLE = Extrapolation Algorithm for Greater Language-model Efficiency*

### 문제

draft 모델은 관리 부담이 크다. 메모리를 차지하고, target과 어휘가 맞아야 하고,
target을 바꾸면 draft도 다시 골라야 한다. `8.2`의 self-speculative는 이 문제를
풀었지만 대신 **모델을 처음부터 그 방식으로 학습해야 한다**는 제약이 남았다.

**target 모델은 그대로 두고, 위에 작은 것만 얹어서 draft 역할을 하게 할 수 없을까?**

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
> **그림 8.4** — head마다 상위 `s`개를 남기면 경로 조합이 트리를 이룬다.
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

### EAGLE-2 · EAGLE-3 — 정적 트리에서 동적 트리로

📌 [T1] 위에서 본 Medusa식 트리는 **고정된 모양**이었다 — 어느 자리든 상위
`s`개를 남기는 규칙이 항상 같았다. EAGLE-2는 이게 최선이 아니라고 짚는다.
**acceptance rate는 위치뿐 아니라 문맥에도 달려 있다** — 어떤 문맥에서는
2등 후보도 자주 맞고, 어떤 문맥에서는 1등도 잘 안 맞는다.

📌 [T1] EAGLE-2는 feature 예측기 자신의 **확신도(confidence)**를 트리를
키우는 기준으로 쓴다. 확신도가 높은 가지는 더 깊게, 낮은 가지는 일찍 쳐낸다
— **트리 모양이 매 스텝, 매 문맥마다 달라진다.**

```
 Medusa/EAGLE-1:  자리마다 항상 상위 s개  → 트리 모양 고정
 EAGLE-2:         확신도 높은 가지만 더 키움 → 트리 모양이 문맥마다 다름
```

📌 [T1] 결과로 EAGLE-1 대비 20~40% 더 빠르다고 보고되며 (전체로는 3~4배대
가속), Medusa의 정적 트리보다 **같은 검증 예산으로 더 많이 맞힌다.**

📌 [T1] EAGLE-3는 다른 방향에서 개선한다. EAGLE-1/2는 "다음 하나의 hidden
feature"만 예측하도록 제약돼 있었는데, EAGLE-3는 **여러 층의 semantic
feature를 함께 융합**해서 예측하도록 이 제약을 푼다 — feature 하나에 갇히지
않고 여러 수준의 표현을 동시에 쓴다.

> 💡 **`5.4`의 granularity 논쟁과 결이 비슷하다.** 고정된 트리(Medusa)는
> 단순하지만 낭비가 있고, 동적 트리(EAGLE-2)는 문맥에 맞춰 예산을 쓰지만
> 구현이 복잡하다 — "얼마나 정적으로 둘 것인가"가 이 계열에서도 반복되는
> 트레이드오프다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | 별도 draft 모델 대신 target 모델에 붙인 head(Medusa) 또는 feature 예측기(EAGLE)로 초안을 만든다. EAGLE-2/3는 트리를 문맥에 맞춰 동적으로 키운다 |
| **장점** | · **별도 모델이 필요 없다** — 메모리와 관리 부담 감소<br>· target과 어휘·표현이 자동으로 일치<br>· EAGLE-2/3는 acceptance rate가 매우 높다 (3~4배대 가속) |
| **한계** | · head·예측기를 학습시키는 추가 단계가 필요<br>· Medusa는 head 간 의존성이 없어 acceptance가 제한적<br>· 트리 검증(특히 동적 트리)은 구현이 복잡하고 커널 지원이 필요 |
| **대표 구현** | 주요 서빙 프레임워크에서 옵션으로 제공 (vLLM·SGLang의 EAGLE·EAGLE-2 지원) |
| **다음으로** | head를 나중에 붙이지 말고 **처음부터 그렇게 학습**하면 → **MTP** |

---

## 8.5 MTP — 학습 목표를 바꾸다

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

`8.4`의 Medusa head들은 서로 독립적으로 **원래 hidden state 하나만** 보고
각자 t+1, t+2, t+3을 예측했다. MTP는 다르게 짠다 — **직전 모듈의 출력을 다음
모듈의 입력으로 이어받는 사슬(chain)**이다.

```
 Medusa:  hidden ──► head1 ──► t+1       hidden ──► head2 ──► t+2
                     (서로 독립, 둘 다 원래 hidden만 봄 — 8.4 참고)

 MTP:     본체 ──► hidden_t ──► t+1 예측
                       │
                       ▼
                  MTP 모듈 1  (입력: hidden_t + t+1의 embedding)
                       │
                       ▼
                  hidden'_t ──► t+2 예측
```

MTP 모듈은 "이전 표현 + 방금 나온 토큰의 embedding"을 받아 다음 표현을 만드는
작은 블록이다. **인과적 사슬을 유지한다**는 점에서 Medusa보다 `8.4`의 EAGLE에
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

> 💡 **MTP는 `8.1`~`8.4`와 성격이 다르다.**
> speculative decoding과 Medusa·EAGLE은 순수하게 추론 최적화다.
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
| **다음으로** | 여기까지 초안은 전부 **순차적으로** 만들어졌다. 초안 블록을 한 번에 뽑을 수는 없나 → **Diffusion 초안** |

---

## 8.6 Diffusion 초안 — 초안 자체를 한 번에 만든다

*block diffusion drafting · DFlash · DSpark*

### 왜 나왔나

`8.1`~`8.5`가 병렬화한 것은 전부 **검증**이었다. target forward 한 번으로 `γ+1`개
위치를 한꺼번에 확인한다. 그런데 그 앞에 아직 손대지 않은 직렬성이 하나 남아 있다.

**초안을 만드는 일 자체는 여전히 순차적이다.**

EAGLE의 feature 예측기도, MTP 모듈 사슬도, LayerSkip의 앞쪽 레이어도 결국 작은
자기회귀 루프다. `γ`개를 뽑으려면 `γ`번 돈다.

```
 자기회귀 draft:  T_draft = γ · t_step      ← γ에 비례해서 늘어난다
```

`t_step`이 작을 뿐 **구조는 본체와 똑같다.** 그래서 `γ`를 키우면 어느 지점부터
검증에서 버는 것보다 초안에서 잃는 게 커진다 — `8.4`에서 트리를 아무리 영리하게
키워도 draft 깊이를 마음껏 늘리지 못했던 이유가 여기에 있다.

> **초안 블록 하나를, 한 번의 forward로 통째로 만들 수는 없나?**

마침 그 모양으로 생성하는 모델 계열이 옆에서 따로 자라고 있었다.
**diffusion language model**이다.

### 잠깐 — diffusion language model이란

자기회귀는 왼쪽에서 오른쪽으로 한 칸씩 채운다. masked diffusion LM은 다르게 간다.
**출력 자리를 전부 `[MASK]`로 깔아놓고, 반복하면서 일부씩 확정(unmask)한다.**

```
 step 0:  [M] [M] [M] [M] [M] [M] [M] [M]
 step 1:  [M] cat [M] [M] on  [M] [M] mat     ← 확신 있는 자리부터 확정
 step 2:  The cat [M] [M] on  the [M] mat
 step 3:  The cat sat down on the big mat     ← 완성
```
> **그림 8.6** — 매 스텝 모든 위치를 동시에 예측하고, 확신도가 높은 자리만
> 확정한다. 스텝 수가 토큰 수보다 훨씬 적을 수 있다는 게 이 방식의 매력이다.

LLaDA(8B), Dream(7B) 같은 공개 모델이 이 방식으로 자기회귀 모델과 견줄 만한
품질에 도달했다. 그런데 **실제 속도는 기대만큼 나오지 않았다.** 이유가 두 가지다.

📌 [T1] **① KV cache를 못 쓴다.** 모든 위치가 서로를 보는 bidirectional attention
이라 `01-attention`에서 본 causal 구조의 전제가 깨진다. 이미 확정된 토큰의 K/V도
다음 스텝에서 문맥이 바뀌면 달라지므로, 원칙적으로는 **매 스텝 시퀀스 전체를
다시 계산해야 한다.** 병렬성을 얻는 대신 캐시를 잃는 거래다.

📌 [T1] **② 한 스텝에 여러 자리를 동시에 확정하면 의존성이 깨진다.** 각 위치는
자기 자리의 **주변분포**만 보고 고른다. "a"와 "the"가 각각 0.5인 자리가 둘 있을 때
독립적으로 뽑으면, 결합분포에서는 거의 일어나지 않는 조합이 나온다.
그래서 한 번에 많이 확정할수록 품질이 떨어진다.

두 문제에 대한 대응이 이 절의 재료가 된다.

| 문제 | 대응 | 방식 |
|---|---|---|
| KV cache 부재 | **Fast-dLLM** | 블록 단위 **근사** KV cache — 블록 안 여러 스텝 동안 K/V를 재사용. 학습 불필요, LLaDA·Dream에서 최대 27.6배 처리량 보고 |
| 동시 확정의 품질 저하 | **confidence 기반 병렬 디코딩** | 확신도가 임계값을 넘는 자리만 확정하고 나머지는 다음 스텝으로 미룬다 |
| 둘 다 구조적으로 | **Block Diffusion (BD3-LM)** | **블록 사이는 자기회귀, 블록 안은 diffusion.** KV cache와 가변 길이 생성이 되살아난다 |

> 💡 **마지막 줄이 이 장과 만나는 지점이다.** "블록 하나를 한 번에, 블록끼리는
> 순서대로"는 speculative decoding의 draft가 정확히 원하는 모양이다 —
> 검증으로 확정된 앞부분을 조건으로 삼아, 다음 `γ`개를 한 번에 제안하면 된다.

### DFlash — block diffusion을 draft로 쓴다

📌 [T1] DFlash(2026, ICML 2026)는 draft 모델을 작은 자기회귀 모델에서
**가벼운 block diffusion 모델**로 바꾼다.

```
 EAGLE:   [확정된 문맥] ─► 예측기 ─► t+1 ─► 예측기 ─► t+2 ─► ... ─► t+γ
                            (γ번 순차 실행)

 DFlash:  [확정된 문맥] + [M][M][M]...[M]  ─── forward 1회 ───►  t+1 ... t+γ
                          γ개 마스크                              (동시에)
```
> **그림 8.6b** — draft의 forward 횟수가 `γ`에서 **1**로 떨어진다.
> 이미지 diffusion과 달리 denoising을 여러 번 돌지 않는다 — 어차피 target이
> 검증하므로, **한 스텝짜리 거친 초안이면 충분하다.**

설계에서 중요한 건 세 가지다.

**① 초안 지연이 `γ`와 무관해진다.**

```
 자기회귀 draft:   T_draft = γ · t_step   (γ에 비례)
 block diffusion:  T_draft = t_parallel   (γ와 무관, O(1))
```

📌 [T1] 그래서 길이를 공격적으로 잡을 수 있다 — 5-layer DFlash가 **16 토큰**을
내는 쪽이 EAGLE-3가 **8 토큰**을 내는 쪽보다 지연도 낮고 acceptance length도 높다.

**② target의 feature를 draft의 모든 레이어에 주입한다 (KV injection).**

📌 [T1] target의 2번째 층부터 뒤에서 3번째 층 사이에서 **균등하게 고른 5개 층**의
hidden state를 이어붙여 draft hidden 공간으로 projection한 뒤, draft의 **모든 레이어
K/V projection에 직접 주입**한다. 입력에만 조건을 넣는 EAGLE식 조건화와 다른 점이고,
덕분에 **draft 레이어를 깊게 할수록 acceptance length가 같이 는다.** projection
파라미터는 Qwen3.5-35B 기준 약 42MB로, 수십 GB짜리 본체 옆에서는 무시할 수준이다.

> 💡 `8.4`의 EAGLE-3가 "여러 층의 semantic feature를 융합"한 것과 같은 방향이다.
> **target이 이미 계산해둔 표현을 draft에게 얼마나 넘겨주느냐**가 이 계열의
> 공통 조절 손잡이다.

**③ 학습을 추론 상황과 똑같이 만든다.**

📌 [T1] 응답에서 anchor 토큰을 무작위로 뽑아 그 자리를 블록의 첫 위치로 삼고
나머지를 마스킹한다. 추론 때 "직전 검증에서 확정된 토큰이 다음 블록을 조건화하는"
상황을 그대로 재현하는 것이다. 손실은 위치 가중 cross-entropy로,
`w_k = exp(-(k-1)/γ)` — **앞쪽 위치를 더 무겁게 본다.** 앞이 틀리면 뒤는 어차피
전부 버려지기 때문이다.

**검증은 하나도 바뀌지 않는다.** 표준 speculative sampling + tree attention이고,
따라서 `8.1`의 **"근사가 아니다"가 그대로 적용된다** — 출력 분포는 target 단독
샘플링과 같다.

📌 [T1] 보고된 수치: Qwen3-8B 평균 4.9배, Qwen3-4B 수학 과제 최대 5.15배,
EAGLE-3(tree size 16) 대비 2.4배, SGLang 서빙 concurrency 1에서 5.1배.

### 그런데 뒤로 갈수록 틀린다 — suffix decay

병렬 초안에는 구조적인 약점이 있다. 한 번의 forward로 `γ`개를 동시에 내면
**k번째 위치는 k-1번째가 실제로 무엇으로 뽑혔는지 모른다.** 앞 토큰을 조건으로
받는 게 아니라, 가능한 모든 앞 토큰에 대해 **주변화(marginalize)한** 예측이다.
DSpark 논문은 이걸 *multi-modal collision*이라 부른다 — 여러 갈래의 가능성이
한 분포에 뭉개져서, 어느 갈래에도 정확히 맞지 않는 토큰이 나온다.

📌 [T1] 위치별 조건부 acceptance를 재보면 방향이 정반대다.

| 위치 | DFlash (병렬 초안) | EAGLE-3 (자기회귀 초안) |
|---|---|---|
| 1번째 | **~0.88** (math) — 더 깊은 backbone의 capacity 이득 | ~0.81 |
| 2~7번째 | code 0.87 → **0.78**, chat 0.72 → **0.63**으로 급락 | 평평하거나 오히려 상승 |

```
 acceptance
   ▲
   │ ●─── 병렬 초안: 첫 자리는 이기는데
   │  ╲
   │   ╲──●
   │ ○────○────○   자기회귀 초안: 낮게 시작해서 평평하게 간다
   └────────────────────►  블록 안 위치 k
```
> **그림 8.6c** — 첫 자리는 병렬 초안이, 뒤 자리는 자기회귀 초안이 이긴다.
> 그러면 **둘을 겹치면 되지 않나** — 그게 다음 항목이다.

### DSpark — 병렬 backbone 위에 아주 얇은 순차 head

📌 [T1] DSpark(2026, DeepSeek-AI · Peking University)는 DFlash를 **병렬 backbone**
으로 그대로 쓰고, 그 위에 **1차 마르코프 의존성만** 얹는다. 완전한 자기회귀도,
완전한 병렬도 아닌 **semi-autoregressive** 구조다.

위치 `k`의 로짓에, 직전에 **실제로 샘플된** 토큰 `x_{k-1}`에만 의존하는
전이 편향(transition bias)을 더한다.

```
 B = W1 W2,   W1 ∈ R^(V×r),  W2 ∈ R^(r×V),   r = 256 (기본값)

 B(x_{k-1}, ·) = W1[x_{k-1}] W2  ∈ R^V        ← 위치 k의 로짓에 더한다
```

`V×V` 크기의 전이 행렬을 **저랭크로 압축해 들고 있는 것**이다. 순차 실행이
필요하긴 하다 — `x_{k-1}`이 정해져야 `x_k`의 편향을 계산할 수 있으니. 다만 그
순차 부분이 **레이어 forward가 아니라 embedding lookup 한 번 + `(r, V)` 곱
한 번**이다. 사실상 공짜다.

> 💡 **저랭크 압축이라는 도구가 또 나온다.** `01-attention` `2.3`의 MLA는 KV를,
> `02-linear-attention`은 attention 행렬을 저랭크로 눌렀다. 여기서는
> **토큰 전이 행렬**을 누른다 — "전부 들고 있지 말고 중요한 랭크만"이라는
> 같은 손잡이가 전혀 다른 자리에서 쓰이고 있다.

📌 [T1] 효과는 그림 8.6c의 곡선을 평평하게 펴는 것이다 — math에서 0.93으로
시작해 블록 끝까지 높은 조건부 acceptance를 유지한다. 결과적으로 평균 채택 길이
`τ`가 EAGLE-3 대비 26.7~30.9%, DFlash 대비 16.3~18.4% 높아진다 (Qwen3-4B/8B/14B).

### 검증 예산을 요청마다 다르게 — confidence-scheduled verification

여기서부터가 DSpark의 두 번째 절반이고, 이 장에서 처음 나오는 발상이다.

지금까지 `γ`(또는 트리 크기)는 **고정**이었다. 문제는 이것이다 —
**검증에 넣은 토큰은 맞든 틀리든 batch 용량을 먹는다.** 서버가 한가할 때는
상관없지만, 부하가 높을 때 통과 확률 0.3짜리 draft를 검증에 밀어넣는 것은
**다른 요청의 자리를 빼앗는 짓**이다. `8.7`에서 볼 "배치 크기와 상충한다"가
여기서 구체적인 스케줄링 문제로 나타난다.

DSpark는 두 조각으로 푼다.

**① 살아남을 확률을 예측한다 — confidence head**

📌 [T1] 각 draft 위치 `k`마다 스칼라 하나를 낸다.

```
 c_k = σ( w^T [ h_k ; W1[x_{k-1}] ] ) ∈ (0,1)

 의미: "앞의 토큰들이 전부 채택됐다는 조건에서,
        위치 k의 draft 토큰이 target 검증을 통과할 조건부 확률"
```

그대로 쓰면 **과신(overconfident)한다.** 그래서 사후 보정을 한 단계 넣는다 —
*Sequential Temperature Scaling*: 왼쪽에서 오른쪽으로 결합확률을 차례로
보정하며, 각 위치에서 1차원 grid search로 Expected Calibration Error를 최소화한다.

**② 그 확률로 검증 길이를 고른다 — hardware-aware prefix scheduler**

목표 함수는 개별 요청의 지연이 아니라 **시스템 전체 처리량**이다.

```
 Θ = τ · SPS(B)

  τ       기대 채택 토큰 수 (confidence로 계산)
  B       target에 보내는 총 batch 크기 (모든 요청의 검증 토큰 합)
  SPS(B)  그 batch 크기에서 엔진이 내는 초당 step 수
          ← 엔진 초기화 때 한 번 프로파일해 cost table로 들고 있는다
```

📌 [T1] 스케줄러는 greedy로 푼다 — **모든 요청의 가능한 prefix 확장을 survival
확률 내림차순으로 한 줄로 세우고**, 높은 것부터 하나씩 검증 예산에 넣으면서
cost table로 `Θ`를 갱신한다. `Θ`가 더 이상 오르지 않으면 멈춘다.

```
 요청 A: [0.95] [0.91] [0.72] [0.40] ...
 요청 B: [0.93] [0.60] ...              ─► 전부 한 줄로 정렬
 요청 C: [0.97] [0.94] [0.88] [0.81] ...

 정렬:  0.97 0.95 0.94 0.93 0.91 0.88 0.81 0.72 0.60 0.40 ...
                                       ↑
                           Θ = τ·SPS(B)가 꺾이는 지점에서 컷

 결과: C는 4개, A는 2개, B는 1개 — 요청마다 검증 길이가 다르다
```
> **그림 8.6d** — 검증 예산이 "요청당 몇 개"가 아니라 **"시스템 전체에서
> 확률 높은 순서대로"** 배분된다. 부하가 오르면 `SPS(B)` 곡선이 컷을 앞으로
> 당기고, 한가하면 뒤로 민다.

📌 [T1] 실제 동작: MTP-1이 요청당 정적으로 2 토큰을 검증하던 자리에서, 중간
부하일 때 요청당 대략 4~6 토큰으로 늘어난다.

📌 [T1] DeepSeek-V4 서빙 시스템에 실제 트래픽으로 배포한 결과, 같은 처리량
수준에서 사용자당 생성 속도가 **V4-Flash 60~85%**, **V4-Pro 57~78%** 빨라졌다
(MTP-1 기준). 특히 사용자당 120 tok/s 같은 **엄격한 interactivity 목표**에서는
baseline의 처리량이 급락하는데 DSpark는 그렇지 않아, 이전에는 닿지 못하던
운영 구간이 열렸다고 보고한다. 코드·체크포인트는 MIT로 공개됐다.

> 💡 **도메인 편차는 그대로 남는다.** Qwen3-4B 기준 채택 길이가 math ~5.57,
> code ~5.12인데 chat은 ~3.49다. `8.7`에서 볼 "작업 종류가 acceptance를 정한다"는
> 관찰이, 가장 정교한 스케줄러를 붙인 뒤에도 사라지지 않는다.

### 추론에서 달라진 것

| | 자기회귀 draft (`8.4` EAGLE) | 병렬 draft (DFlash) | semi-AR + 스케줄 (DSpark) |
|---|---|---|---|
| draft forward 횟수 | `γ`회 | **1회** | **1회** + 저랭크 곱 |
| draft 지연 | `γ · t_step` | `t_parallel` (`γ`와 무관) | `t_parallel` + α (무시 가능) |
| 위치별 acceptance | 평평, 낮게 시작 | **첫 자리 높고 뒤로 급락** | **높고 평평** |
| 검증 길이 | 고정 `γ` / 고정 트리 | 고정 블록 | **요청·부하마다 동적** |
| 추가 상태 | draft KV | draft KV + target feature 주입 | + confidence head · 엔진 cost table |
| 출력 분포 | 보존 | 보존 | 보존 |

### 토큰 하나가 지나가는 길

```
 ① 조건화:   직전 검증에서 확정된 토큰 + target 5개 층의 hidden을
             draft 모든 레이어의 K/V로 주입
 ② draft:    [M] × γ 를 넣고 backbone forward 1회 → γ개 위치의 로짓
 ③ 순차 보정: k=1..γ 순서로 x_{k-1}을 보고 B(x_{k-1},·)를 더해 샘플
             (레이어를 도는 게 아니라 저랭크 곱 한 번씩)
 ④ 확률 예측: confidence head가 위치마다 c_k → STS로 보정
 ⑤ 스케줄:   전체 요청의 c를 정렬해 Θ = τ·SPS(B)가 최대가 되는 지점까지만 채택
 ⑥ verify:   잘린 길이만큼 target이 한 번에 검증 (tree attention)
 ⑦ accept:   `8.1`과 동일한 rejection sampling — 분포는 그대로
```

**⑤가 이 절의 새로움이다.** `8.1`~`8.5`에서 "몇 개를 검증할까"는 하이퍼파라미터였다.
여기서는 **매 스텝, 요청마다, 현재 부하를 보고 정해지는 값**이 된다.

### 코드와 텐서

| # | 연산 | 텐서 | 비고 |
|---|---|---|---|
| 1 | target feature 추출 | `(B, 1, d)` × 5개 층 → concat `(B, 1, 5d)` | projection으로 draft 공간에 매핑 |
| 2 | draft backbone forward | `(B, γ, d_draft)` → 로짓 `(B, γ, V)` | **forward 1회** — `γ`와 무관 |
| 3 | 전이 편향 | `W1[x_{k-1}]`: `(B, r)` → `× W2`: `(B, V)` | `k`마다 한 번, `r` = 256 |
| 4 | confidence | `(B, γ)` | head 하나, 위치당 스칼라 |
| 5 | target 검증 | `(B, L_k+1, d)` — `L_k`가 요청마다 다름 | **ragged batch** |

> 💡 **5번이 서빙 쪽에 새 부담을 만든다.** `8.1`에서 검증 입력은 `(B, γ+1, d)`로
> 모든 요청이 같은 길이였다. 여기서는 요청마다 길이가 달라서, `10-serving`의
> continuous batching·PagedAttention과 얽히는 지점이 하나 더 늘어난다.
> **이득을 얻은 자리와 복잡도를 치른 자리가 정확히 같다.**

### 정리

| | |
|---|---|
| **핵심 아이디어** | 초안을 **블록째 한 번의 forward로** 만든다 (block diffusion). 병렬 초안의 suffix decay는 저랭크 1차 마르코프 head로 메우고, 검증 길이는 survival 확률과 엔진 처리량 프로파일로 요청마다 다르게 정한다 |
| **장점** | · **초안 지연이 `γ`에 비례하지 않는다** — 긴 블록을 공격적으로 제안 가능<br>· target feature를 모든 draft 레이어에 주입해 acceptance가 높다<br>· **검증 예산을 부하에 맞춰 배분** — 고부하에서 이득이 덜 깎인다<br>· 검증은 표준 speculative sampling이라 **출력 분포가 보존된다** |
| **한계** | · draft를 **target마다 새로 학습**해야 한다 (backbone + head + confidence)<br>· 병렬 backbone은 EAGLE류 예측기보다 무겁다 — 깊이로 acceptance를 사는 구조<br>· suffix decay는 **완화지 제거가 아니다** — 블록을 키울수록 다시 나타난다<br>· confidence 보정·cost table·요청별 가변 길이 검증 등 **서빙 쪽 상태가 늘어난다**<br>· 보고된 수치는 각 논문의 모델·엔진 조건에 한정된다 |
| **대표 구현** | DFlash (z-lab, ICML 2026) · **DSpark** (DeepSeek-V4 서빙 시스템, MIT 공개, SGLang·llama.cpp 지원) |
| **다음으로** | 이 모든 기법의 이득은 결국 하나의 숫자로 수렴한다 → **acceptance rate** |

> 💡 **본체까지 diffusion으로 갈 것인가는 아직 열린 질문이다.**
> LLaDA·Dream, 그리고 상용 쪽의 Mercury·Gemini Diffusion처럼 **본체 자체를**
> diffusion으로 돌리는 시도가 따로 진행 중이다. 다만 지금 프로덕션에 먼저 안착한
> 것은 **"본체는 자기회귀, 초안만 diffusion"** 이라는 절충이다 —
> 검증이 분포를 지켜주니 초안은 거칠어도 되고, 거칠어도 되는 자리에서
> diffusion의 병렬성은 순수한 이득이기 때문이다.

---

## 8.7 시스템 관점 — acceptance rate가 모든 것을 정한다

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
FFN이 compute-bound로 넘어간다.** 그런데 이 장의 모든 기법(`8.1`~`8.5`)이
같은 원리로 이득을 낸다 — 놀고 있는 연산 능력을 회수하는 것이었으니까.

```
 배치가 작다  →  GPU가 놀고 있다  →  이 장의 기법들이 그 여유를 쓴다  →  큰 이득
 배치가 크다  →  GPU가 이미 바쁘다 →  여유가 없다                    →  이득 감소
```

**둘은 같은 자원을 노린다.** 그래서 상충한다.

| 상황 | 유리한 전략 |
|---|---|
| 저지연 요구, 소규모 배치 | **이 장의 기법들** (`8.1`~`8.5`) |
| 고처리량 요구, 대규모 배치 | 배치 키우기 |
| 중간 | 둘의 균형점을 찾아야 함 |

> 💡 **`8.6`의 DSpark는 이 표를 "둘 중 하나 고르기"에서 "매 스텝 계산하기"로
> 바꾼 것이다.** `Θ = τ · SPS(B)`를 최대화하는 검증 길이를 부하마다 다시 푸니까,
> 위 표의 세 행이 **하나의 스케줄러 안에서 연속적으로 이어진다.**
> 다만 트레이드오프 자체가 사라지는 건 아니다 — 한가할 때 길게, 바쁠 때 짧게라는
> 결론은 그대로고, **그 결정을 사람이 아니라 엔진이 한다**는 점이 달라진다.

> 💡 **아키텍처 결정이 서빙 시나리오에 의존한다는 패턴이 또 나온다.**
> `05-moe` `5.4`의 granularity도, `07-shape` `7.1`의 depth도 같은 구조였다.
> **"무엇이 좋은가"에 답하려면 "어떤 배치로 서빙할 것인가"를 먼저 정해야 한다.**

### 그 외 시스템 부담

| 무엇 | 왜 |
|---|---|
| **KV cache 증가** | 거부될 토큰의 KV도 일단 계산해서 넣었다가 롤백해야 한다 (`8.2`의 재사용 활성값도 마찬가지) |
| **지연 변동** | acceptance가 요청마다 달라 스텝 시간이 들쭉날쭉 |
| **스케줄링 복잡도** | 배치 안 요청들이 서로 다른 수의 토큰을 뱉는다 |
| **KV 롤백** | 거부된 부분을 캐시에서 되돌리는 로직 필요 (PagedAttention과 얽힘) |
| **pool·트리 상태 관리** | Lookahead의 n-gram pool, Medusa/EAGLE의 트리 후보처럼 draft/verify 방식마다 별도 상태가 필요 |

두 번째가 운영에서 성가시다. 평균 지연은 좋아지는데 **분산이 커진다.**
SLO를 p99로 관리하면 개선이 기대보다 작아 보일 수 있다.

### 정리

이 파일 전체를 한 표로 정리하면 이렇다.

| 기법 | draft 출처 | 학습 필요 | 품질 영향 | 채택 문턱 |
|---|---|---|---|---|
| **Speculative** (`8.1`) | 별도 작은 모델 | 없음 | **없음** | 낮음 (모델만 있으면) |
| **Self-Speculative** (`8.2`) | 같은 모델의 앞쪽 레이어 | 초기 학습/fine-tune 필요 | 없음 | 중간 (재학습 필요) |
| **Lookahead** (`8.3`) | n-gram pool (Jacobi 궤적) | **없음** | 없음 | **가장 낮음** — 알고리즘만 |
| **Medusa** (`8.4`) | 추가 head | head 학습 | 없음 | 중간 |
| **EAGLE(-2/3)** (`8.4`) | feature 예측기 | 예측기 학습 | 없음 | 중간 |
| **MTP** (`8.5`) | 학습된 MTP 모듈(chain) | **본 학습에 포함** | **향상** | 높음 (재학습) |
| **DFlash** (`8.6`) | block diffusion 초안 (forward 1회) | draft 학습 | 없음 | 중간~높음 (target마다 학습) |
| **DSpark** (`8.6`) | 병렬 backbone + 마르코프 head | draft·head·confidence 학습 | 없음 | 높음 (학습 + 엔진 통합) |

MTP가 최근 넓게 퍼진 이유가 마지막 두 열에 있다. **품질 때문에 어차피 넣을 것이었고,
추론 가속은 딸려온 것**이라 채택 결정이 쉬웠다. 반대로 Lookahead Decoding은
**아무 모델에나 즉시 적용**할 수 있다는 점에서 채택 문턱이 가장 낮다 — 대신
얻는 이득도 가장 작다. **문턱과 이득이 반비례**하는 것이 이 장 전체의 패턴이다.

`8.6`의 DFlash·DSpark는 그 반비례선의 반대쪽 끝에 있다. **target마다 draft를
학습시키고 엔진에 스케줄러까지 넣어야** 하지만, 초안의 직렬성을 없애고 검증
예산을 부하에 맞춰 배분하는 만큼 이득도 가장 크다. 자기 모델과 서빙 스택을
전부 소유한 쪽에서 먼저 나온 것이 우연은 아니다.

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
- Fu et al. (2024), *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*,
  arXiv:2402.02057 (ICML 2024) — Jacobi iteration, n-gram pool, 1.5~2.3배 가속
- Elhoushi et al. (2024), *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding*,
  arXiv:2404.16710 (ACL 2024) — layer dropout, early-exit loss, KV 재사용
- Cai et al. (2024), *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*
- Li et al. (2024), *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*
- Li et al. (2024), *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*,
  arXiv:2406.16858 — 문맥 기반 동적 트리, EAGLE-1 대비 20~40% 추가 가속
- Li et al. (2024/2025), *EAGLE-3* — 다층 semantic feature 융합
- DeepSeek-AI (2024), *DeepSeek-V3*, arXiv:2412.19437 — MTP 설계와 학습 효과
- Gloeckle et al. (2024), *Better & Faster Large Language Models via Multi-token Prediction* — MTP의 품질 근거
- Nie et al. (2025), *Large Language Diffusion Models* (LLaDA), arXiv:2502.09992 — 8B 규모 masked diffusion LM
- Arriola et al. (2025), *Block Diffusion: Interpolating Between Autoregressive and Diffusion
  Language Models*, arXiv:2503.09573 (ICLR 2025 Oral) — 블록 간 AR + 블록 내 diffusion, KV cache·가변 길이
- Wu et al. (2025), *Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache
  and Parallel Decoding*, arXiv:2505.22618 — 블록 단위 근사 KV cache, confidence 임계 병렬 디코딩
- Chen, Liang, Liu (2026), *DFlash: Block Diffusion for Flash Speculative Decoding*,
  arXiv:2602.06036 (ICML 2026) — block diffusion drafter, target feature KV injection
- DeepSeek-AI · Peking University (2026), *DSpark: Confidence-Scheduled Speculative Decoding with
  Semi-Autoregressive Generation*, arXiv:2607.05147 — 저랭크 전이 편향, confidence head·STS,
  hardware-aware prefix scheduler, DeepSeek-V4 배포 결과

**T2 — 구현**
- hao-ai-lab **LookaheadDecoding** — Jacobi 기반 병렬 디코딩 참조 구현
- Meta **LayerSkip** — self-speculative decoding 참조 구현
- vLLM speculative decoding 문서 — acceptance rate 측정, 배치와의 상호작용
- SGLang / TensorRT-LLM의 EAGLE·EAGLE-2·Medusa 구현
- KV cache 롤백 처리 (PagedAttention과의 연동)
- NVlabs **Fast-dLLM** — diffusion LLM용 근사 KV cache·병렬 디코딩 참조 구현
- z-lab **DFlash** — block diffusion drafter 참조 구현
- **DSpark / DeepSpec** (MIT) — draft 학습·평가 파이프라인과 체크포인트 공개,
  SGLang(GPU 서빙)·llama.cpp(엣지) 쪽 지원

**범위와 주의**
- speculative decoding의 이득은 draft 길이만이 아니라 acceptance rate, 검증 batch,
  sampling 설정과 서빙 부하에 따라 달라진다.
- MTP의 학습 품질 효과와 추론 가속 효과는 구분해서 평가해야 한다.
- Self-Speculative(LayerSkip)와 Lookahead Decoding의 speedup 수치는 해당 논문의
  실험 조건(모델 크기, 과제, 하드웨어)에 한정된다. `8.4`의 EAGLE-2/3 가속 배수와
  단순 비교하지 않는다 — 벤치마크 설정이 서로 다르다.
- `8.6`의 DFlash 배수(Qwen3 계열, greedy, SGLang)와 DSpark의 프로덕션 수치
  (DeepSeek-V4 자체 서빙 시스템, MTP-1 기준 상대값)는 **측정 조건이 서로 다르고
  다른 절의 수치와도 다르다.** 특히 DSpark의 이득은 confidence 보정과 엔진
  프로파일이 그 시스템에 맞춰져 있다는 전제 위에 있다.
- 본체를 diffusion으로 돌리는 dLLM(LLaDA·Dream 등)의 처리량 수치와, 본체는
  자기회귀로 두고 **초안에만** block diffusion을 쓰는 `8.6`의 수치는 서로 다른
  이야기다. 섞어 읽지 않는다.
