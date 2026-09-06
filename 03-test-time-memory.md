# 03. Test-Time Memory — Titans · ATLAS · HOPE

`02-linear-attention`을 다시 펼쳐보자. 그 파일은 결국 **상태 갱신 규칙의 역사**였다.

```
 S = S + k vᵀ             안 잊는다              (Linear Attention)
 S = α S + k vᵀ           시간으로 잊는다         (Mamba)
 S = (I − β k kᵀ) S + β k vᵀ   내용으로 고친다   (DeltaNet)
```

`2.3`에서 delta rule을 소개하며 **"온라인 학습에서 오래 쓰인 그것을 그대로 가져온다"** 고 했다.
그 말을 끝까지 밀면 어떻게 될까.

delta rule은 사실 **선형 연상 메모리에 대한 경사하강 한 스텝**이다. 그렇다면 —

- 메모리를 행렬이 아니라 **신경망(MLP)** 으로 두면?
- 갱신을 흉내가 아니라 **진짜 경사하강**으로 하면? momentum과 weight decay까지 붙여서?

Google Research의 **Titans**가 정확히 그 질문에서 출발한다.
그리고 이 축은 하나 더 밀고 나간다 — **갱신 규칙 자체를 학습 대상으로 만드는 것**(HOPE).

> ⚠️ **이 파일은 연구 단계 내용이다.**
> 대규모 프로덕션 서빙의 비용·안정성 근거는 아직 확립되지 않았다. 다른 파일들과 달리
> **"이렇게 서빙된다"가 아니라 "이런 방향이 있다"** 로 읽어야 한다.
>
> 이 문서는 사실(📌 [T1] 논문/공식 자료로 확인된 내용, 📌 [T2] 구현·공식 문서로 확인된
> 내용)과 해석(💡 이 위키가 챕터를 가로질러 종합한 관점 — 논문에 직접 나오는 문장이
> 아니다)을 구분해 표기한다. 표시가 없는 서술은 바로 위/아래에 있는 📌 근거의 직접적인
> 수식·정의를 그대로 풀어 쓴 것이다. 특히 **"정리" 표의 장점·한계**는 대부분 📌 근거를
> 요약한 것이지만, 일부(예: 디버깅 난이도 같은 엔지니어링 추정)는 이 문서의 판단이며
> 해당 항목에 별도로 표시했다.

---

## 계보 지도

```
        [02-linear-attention 의 끝]
                  │
            DeltaNet (2024)
        S ← (I − βkkᵀ)S + βkvᵀ
        = 선형 메모리의 경사하강 1스텝
                  │
        [메모리를 신경망으로, 갱신을 진짜 경사하강으로]
                  │
             Titans (2025)
        MLP 메모리 + surprise(gradient)
        + momentum + weight decay
                  │
        ┌─────────┴──────────┐
        │                    │
  [윈도우와 2차 정보]   [갱신 규칙을 학습]
        │                    │
    ATLAS (2025)         HOPE (2025)
    Omega rule           Nested Learning
    Muon 내부 최적화      self-modifying + CMS
```
> **그림 3.0** — 축3의 계보. `02`의 마지막 지점에서 갈라져 나온다.

**목차**

| | 절 | 한 줄 |
|---|---|---|
| [3.1](#31-두-계보가-만나는-지점--rnn에서-test-time-memory까지) | **RNN 계보** | 상태 갱신이 온라인 학습으로 바뀌기까지 |
| [3.2](#32-titans--놀라움으로-기억한다) | **Titans** | 놀라움으로 기억한다 |
| [3.3](#33-atlas--창을-넓히고-2차-정보를-쓴다) | **ATLAS** | 창을 넓히고 2차 정보를 |
| [3.4](#34-hope와-nested-learning--갱신-규칙을-학습한다) | **HOPE** | 갱신 규칙을 학습한다 |
| [3.5](#35-시스템-관점--추론-중에-학습이-일어난다) | **시스템 관점** | read-mostly에서 multi-rate read+write로 ★ |

---

## 3.1 두 계보가 만나는 지점 — RNN에서 test-time memory까지

`02`는 이름은 Linear Attention이지만, 계산 모양은 점점 RNN에 가까워졌다.
과거 토큰을 전부 다시 읽는 대신 **고정 크기 상태를 읽고 쓴다.** Titans는 여기서 상태를
신경망으로 바꾸고, 쓰기 규칙을 실제 최적화 과정으로 해석한다.

| 모델 | 해결하려는 문제 | 핵심 아이디어 | Memory update 방식 | Titans·HOPE와의 연관성 |
|---|---|---|---|---|
| **Vanilla RNN** | 가변 길이 시퀀스를 고정 상태로 처리 | 이전 hidden state를 다음 스텝으로 전달 | `h_t = f(h_{t-1}, x_t)` — 매 토큰 전체 상태 덮어쓰기 | “토큰별 상태 갱신”이라는 출발점 |
| **LSTM / GRU** | 장기 의존성과 gradient 소실 | 입력·망각·출력 gate로 보존량 제어 | additive cell update + learned gate | 무엇을 잊을지 입력에 따라 정한다 |
| **SSM / S4** | 긴 시퀀스와 병렬 학습 | 구조화된 선형 동역학을 convolution/scan으로 계산 | 선형 recurrence, 시간축 병렬화 가능 | recurrent state와 병렬 학습을 양립시킨다 |
| **Mamba** | 고정 SSM이 내용에 따라 기억하지 못함 | selective SSM | 입력 의존 감쇠·입력·출력 계수 | 갱신 속도를 내용에 따라 바꾼다 |
| **DeltaNet / KDA** | 이미 저장된 연상을 내용 기반으로 고치기 | delta rule + channel-wise decay | 예측 오차를 지우고 새 key-value를 기록 | 선형 메모리의 GD 한 스텝으로 해석된다 |
| **Titans** | 선형 상태의 표현력과 고정 갱신 규칙 | MLP 메모리 + surprise | 실제 gradient + momentum + weight decay | 상태 갱신을 **test-time learning**으로 올린다 |
| **HOPE** | 모든 메모리가 같은 규칙·주기로 갱신됨 | Nested Learning + CMS | 모듈별 갱신 주기 + self-modifying rule | 갱신 규칙과 주기 자체를 학습 대상으로 만든다 |

### Transformer · Mamba · Titans/HOPE의 layer 구성 차이

| | Transformer | Mamba | Titans · HOPE |
|---|---|---|---|
| **주요 layer 경로** | norm → self-attention → residual → norm → FFN → residual | projection → short conv → selective SSM → gate/output projection | local attention과 **neural memory**를 MAC·MAG·MAL로 결합; HOPE는 여러 주기의 CMS block |
| **시퀀스 상태** | 토큰마다 쌓이는 KV cache | layer마다 고정 크기 SSM state | bounded local-attention KV + 요청별 neural-memory parameter/state |
| **긴 과거를 읽는 법** | KV를 직접 읽어 정확히 검색 | 상태에 압축된 요약을 읽음 | neural memory를 질의하고 필요하면 local attention과 결합 |
| **추론 중 쓰기** | 새 토큰의 KV append — 모델 가중치는 고정 | recurrence state 갱신 | **메모리 손실의 gradient로 상태를 학습**; HOPE는 주기별 갱신 |
| **주요 시스템 부담** | KV 용량·대역폭 | scan/recurrence kernel, 상태 관리 | forward + update/backward, 요청별 가변 상태, write traffic |

여기서 “쓰기”를 구분해야 한다. Transformer도 KV를 append하고 Mamba도 상태를 덮어쓴다.
Titans가 새로 들고 온 것은 **학습 가능한 메모리 가중치에 손실과 optimizer를 적용하는 쓰기**다.
그래서 이 장은 RNN의 반복 상태와 Transformer의 attention이 만나는 지점이면서,
동시에 추론과 학습의 경계가 흐려지는 지점이다.

---

## 3.2 Titans — 놀라움으로 기억한다

### 왜 나왔나

`02-linear-attention`의 계보는 **무엇을 어떻게 잊을 것인가**를 정교하게 다듬어 왔다.
KDA에 이르러 채널마다 다른 속도로 잊는 데까지 갔다. 그런데 두 가지가 그대로 남았다.

**① 메모리가 행렬 하나다.**
`S`는 `d × d` 행렬이고, `o = q S`는 **선형 변환**이다. 아무리 갱신을 잘해도
키와 값 사이의 **비선형 관계**는 담을 수 없다.

**② 갱신 규칙이 고정이다.**
`(I − βkkᵀ)S + βkvᵀ`는 설계자가 정한 형태다. `α`와 `β`가 입력에 따라 변할 뿐,
**"어떻게 기억할지"의 골격은 학습되지 않는다.**

📌 [T1] Titans 논문은 이 자리를 **단기 기억(attention)과 장기 기억(신경 메모리)**의
구도로 다시 짠다: "attention ... performs as a short-term memory, while neural
memory ... acts as a long-term, more persistent, memory." (Behrouz et al. 2024,
논문 원문 인용)

| | 무엇에 해당하나 | 성격 |
|---|---|---|
| **단기 기억** | attention | 정확하지만 `O(S)`로 비싸다 |
| **장기 기억** | 신경 메모리 모듈 | 압축되지만 **추론 중에 학습된다** |

`02`의 고정 상태가 "싸지만 뭉개지는 장기 기억"이었다면, Titans는 **그 장기 기억을
학습 가능한 신경망으로 바꾼다** — ①과 ②를 한 번에 푸는 길이다.

### 아이디어와 구조

📌 [T1] 메모리 모듈 `M`은 `L_M ≥ 1` 층짜리 MLP다.
논문은 **깊은 메모리(`L_M ≥ 2`)가 선형(행렬) 메모리보다 낫다**고 보고한다
(ablation: linear memory 28.49 vs 전체 모델 27.01 perplexity) — 비선형 의존
관계를 담을 수 있기 때문이고, 위 ① 한계를 정면으로 푼다.

📌 [T1] 학습 목표는 **연상 메모리 손실**이다.

```
 ℓ(M ; x_t) = ‖ M(k_t) − v_t ‖²
```

키를 넣으면 값이 나오도록 메모리를 학습시킨다.

> 💡 `02-linear-attention` `2.1`에서 상태를 **"키를 주면 값을 돌려주는 사전"** 이라 불렀다.
> Titans는 그 비유를 문자 그대로 구현한다. **사전이 신경망이고, 진짜로 학습된다.**

**놀라움(surprise) = 그 손실의 gradient**

```
 surprise = ∇ℓ(M_{t−1} ; x_t)
```

| 상황 | gradient | 해석 |
|---|---|---|
| 메모리가 이미 잘 예측함 | 작다 | **안 놀랍다** → 조금만 갱신 |
| 메모리가 틀림 | 크다 | **놀랍다** → 많이 갱신 |

예측 오차가 곧 기억할 이유가 된다.

**momentum — 놀라움의 여운**

놀라운 일이 벌어진 **직후의 토큰들도** 중요할 수 있다.
그래서 surprise를 한 스텝에서 끝내지 않고 누적한다.

```
 S_t = η_t · S_{t−1}  −  θ_t · ∇ℓ(M_{t−1} ; x_t)
       └ 과거 놀라움 ┘     └ 순간 놀라움 ┘
```

**forgetting — weight decay**

```
 M_t = (1 − α_t) · M_{t−1} + S_t
```

| `α_t` | 효과 |
|---|---|
| → 1 | 메모리를 **싹 비운다** |
| → 0 | 과거를 건드리지 않고 **새 정보만 얹는다** |

📌 [T1] 논문은 이 전체가 **"momentum과 weight decay를 쓴 mini-batch 경사하강으로
메타 신경망을 최적화하는 것과 동치"** 라고 정리하고, Mamba·LRU의 게이팅과 연결짓는다.

#### 메모리를 어디에 붙이나 — MAC, MAG, MAL

*Memory as Context · Memory as Gate · Memory as Layer*

메모리 모듈을 만들었으면 attention과 어떻게 조합할지가 남는다.
`02` `2.6`의 하이브리드 설계와 같은 구조의 질문이고, Titans는 세 가지를 제시한다.

```
 MAC — Memory as Context
   메모리에서 꺼낸 것을 컨텍스트에 "붙여서" attention에 넣는다
   [지속 메모리] ‖ [메모리가 꺼낸 것] ‖ [현재 청크]  →  attention

 MAG — Memory as Gate
   두 갈래를 게이트로 섞는다
   ┌ 슬라이딩 윈도우 attention ┐
   ┤                          ├─ ⊗ ─►
   └ 신경 메모리              ┘

 MAL — Memory as Layer
   직렬로 쌓는다
   메모리 층 ─► 슬라이딩 윈도우 attention 층
```
> **그림 3.2a** — 세 가지 배치

| 변형 | 방식 | 장점 | 한계 |
|---|---|---|---|
| **MAC** | 청크로 나누고, 메모리 출력을 컨텍스트에 이어붙임 | **attention이 장기 기억을 쓸지 스스로 결정**. 표현력 최고 | 청크 분할이 필요, 무겁다 |
| **MAG** | SWA 갈래 + 메모리 갈래를 게이트로 결합 | 분할이 없어 **학습 효율 좋음** | 아주 긴 의존 관계에 덜 강함 |
| **MAL** | 메모리 층 → SWA 층 직렬 | 가장 단순 | **각 층이 병목**이 되어 제약이 큼 |

> 💡 **`01-attention` `1.7`의 NSA와 구도가 닮았다.**
> NSA도 압축·선택·지역 세 갈래를 학습된 게이트로 섞었다.
> **"싼 요약 + 정확한 지역 정보"를 어떻게 결합할 것인가**는 축을 넘나들며 반복되는 질문이다.
>
> 그리고 MAG는 `02` `2.6`의 3:1 하이브리드와 사실상 같은 발상이다 —
> 값싼 장기 갈래와 정확한 지역 갈래를 함께 두는 것.

#### 계보 한눈에 보기

`02-linear-attention` `2.4`의 표를 한 줄 늘리면 이렇게 된다.

| 모듈 | 메모리의 형태 | 갱신 방식 |
|---|---|---|
| Linear Attention | 행렬 | 누적 |
| Mamba | 행렬 | 감쇠 + 누적 |
| DeltaNet | 행렬 | delta rule (= 선형 메모리의 GD 1스텝) |
| Gated DeltaNet · KDA | 행렬 | delta rule + 감쇠 (채널별) |
| **Titans** | **MLP (깊이 `L_M`)** | **경사하강 + momentum + weight decay** |

> 💡 (이 위키의 요약) **"delta rule을 끝까지 밀면 Titans"** 라는 표현으로 이해하면
> 도움이 된다. DeltaNet은 선형 메모리에 대해 경사하강을 한 스텝 흉내 낸 것이고,
> Titans는 **비선형 메모리에 대해 옵티마이저를 통째로 얹은 것**이다 — 다만 이건
> 위 표의 사실들을 이 위키가 하나의 문장으로 압축한 것이지, 논문이 쓴 표현은 아니다.

### 추론에서 달라진 것

`02`까지의 모든 모듈은 추론 시 가중치가 고정이었다 — 상태(`S`)는 갱신되지만
그건 activation이지 **weight**가 아니었다. Titans는 이 경계를 처음 넘는다.

| | DeltaNet · KDA (고정 갱신 규칙) | Titans |
|---|---|---|
| 갱신되는 것 | 상태 행렬 `S` (activation) | **메모리 MLP의 가중치 자체** |
| 갱신 방식 | 설계자가 정한 delta rule 한 스텝 | **실제 손실의 gradient + momentum + weight decay** |
| 추론 중 연산 | forward만 | **forward + 메모리에 대한 backward** |
| 요청별로 달라지는 것 | 상태 값 | **메모리의 가중치 값** — 사실상 요청마다 다른 모델 |

📌 [T1] `3.2` "아이디어와 구조"에서 본 "Mamba·LRU 게이팅과 동치"라는 논문의 정리도
이 표와 같은 지점을 가리킨다 — 다만 Titans는 그 게이팅을 **명시적인 손실의
gradient**로 유도한다는 차이가 있다.

### 토큰 하나가 지나가는 길

`00-foundations` `0.2` 표에 세 단계가 새로 끼어든다.

| # | 하는 일 | `02`까지와 다른 점 |
|---|---|---|
| 1 | `q, k, v` 만들기 | 동일 |
| 2 | 메모리로 `k`를 조회 — `v_pred = M_{t-1}(k_t)` | MLP forward (`L_M`개 층) |
| 3 | 손실과 gradient 계산 — `∇ℓ(M_{t-1}; x_t)` | **새로 생김** — MLP backward |
| 4 | momentum 누적 — `S_t` | **새로 생김** |
| 5 | weight decay로 메모리 갱신 — `M_t` | **새로 생김. 여기서 가중치가 실제로 바뀐다** |
| 6 | `q`로 갱신된 메모리를 읽어 출력 — `M_t(q_t)`, MAC/MAG/MAL로 attention과 결합 | MLP forward + 결합 |

**3~5번이 이 장 전체의 핵심이다.** `01`, `02`에서는 한 토큰이 지나가며 forward만
했다. 여기서는 **같은 토큰이 지나가면서 작은 신경망 하나를 학습시킨다.**

### 코드와 텐서

메모리 `M`을 `L_M`개 층, 각 층의 가중치를 `W_1 … W_{L_M}`이라 하자
(`W_i`의 shape는 `(d_{i-1}, d_i)`, `d_0 = d_k`, `d_{L_M} = d_v`).

| # | 연산 | shape 변화 | 종류 |
|---|---|---|---|
| 1 | `k_t` 투영 | `(B,1,d) → (B,1,d_k)` | GEMV |
| 2 | **메모리 forward** `M(k_t)` | `(B,1,d_k) → ⋯ → (B,1,d_v)`, `L_M`단계 | GEMV × `L_M` |
| 3 | 손실 `‖M(k_t)−v_t‖²` | `(B,1,d_v)` → 스칼라 | elementwise + reduction |
| 4 | **메모리 backward** — 각 `W_i`의 gradient | 결과 shape = **각 `W_i`와 동일**한 `(d_{i-1}, d_i)` | GEMV × `L_M` (역방향) |
| 5 | momentum 누적 `S_t` | `W_i`와 동일 shape, 층마다 | elementwise |
| 6 | weight decay 블렌드 `M_t` | `W_i`와 동일 shape, **가중치 자체를 덮어씀** | elementwise |
| 7 | `q_t`로 읽기 `M_t(q_t)` | `(B,1,d_k) → (B,1,d_v)`, `L_M`단계 | GEMV × `L_M` |

> 💡 **`01`의 표, `02`의 표와 근본적으로 다른 줄이 4~6번이다.**
> `01`·`02`에서 "상태"는 언제나 `(n_h, d_h, d_h)`처럼 **토큰/헤드 모양의 텐서**였다.
> 여기서 갱신되는 `S_t`, `M_t`는 **가중치 모양의 텐서**다 — `01`·`02`가 "무엇을
> 캐시할까"를 물었다면, 여기서는 "**얼마나 큰 모델을 초당 몇 번 미분할까**"를 묻는다.

### 학습은 어떻게 병렬화하나

`02` `2.1`에서 본 그 문제가 여기서도 나온다. 재귀는 순차적이라 GPU를 못 채운다.

📌 [T1] 해법도 같은 계열이다.

- 청크 안의 누적 갱신을 **행렬곱으로 변환**한다 (학습률과 감쇠를 대각 행렬로)
- momentum은 **parallel associative scan**으로 청크 단위 동시 계산

결과적으로 `O(N)` 복잡도를 유지하면서 GPU/TPU의 행렬곱을 쓴다.

### 실험에서 확인된 범위

📌 [T1] **"2M+ 컨텍스트"는 논문 abstract의 주장**이다 ("effectively scale to
larger than 2M context window size"). 실제로 표로 제시되는 결과는 RULER
16K 벤치마크까지다 — 2M 토큰에서의 정량적 표는 논문에 없다.

📌 [T1] 16K RULER S-NIAH(single needle-in-a-haystack) 결과는 이렇다
(Table 2, 170M~760M 규모 모델 기준).

| 모델 | S-NIAH 정확도 |
|---|---|
| **Titans (MAC)** | **97.4%** |
| Titans (LMM, 순수 신경 메모리) | 80.2% |
| TTT | 88.4% |
| DeltaNet | 5.4% |
| Mamba2 | 0.0% |

고정 상태 계열(DeltaNet·Mamba2)이 16K에서 사실상 recall에 실패하는 지점에서
neural memory가 크게 앞선다는 뜻이다. 다만 이 결과는 해당 모델 크기(≤760M)와
합성 retrieval benchmark(RULER)의 실험 조건에 한정되고, **"2M 토큰까지 이
정확도가 유지된다"는 별도로 검증된 사실이 아니다** — abstract의 주장과 표로
확인된 실험 범위를 구분해서 읽어야 한다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | 장기 기억을 MLP로 두고, 예측 오차(surprise)의 gradient로 **추론 중에 학습**시킨다. momentum과 weight decay 포함 |
| **장점** | · **비선형 메모리**로 선형 상태보다 큰 표현력 제공 (ablation: linear 28.49 vs 전체 27.01 perplexity)<br>· 16K RULER S-NIAH에서 DeltaNet(5.4%)·Mamba2(0.0%) 대비 크게 높은 recall(MAC 97.4%)<br>· abstract는 **2M+ 컨텍스트**까지 확장 가능하다고 주장하지만, 그 규모의 정량적 실험표는 논문에 없음<br>· 갱신 규칙이 GD + momentum + weight decay 형태로 정리됨 |
| **한계** | · **추론 중 backward가 필요** — 서빙 스택의 전제가 깨진다<br>· 깊은 메모리(`L_M`↑)는 **선형으로 느려진다**<br>· 갱신이 **현재 토큰 하나**의 gradient에 기반<br>· ⚠️ 프로덕션 채택 사례가 확인되지 않음 |
| **대표 모델** | 연구 단계 (Google Research) |
| **다음으로** | 갱신을 현재 토큰이 아니라 **윈도우 전체**에 대해 하면 → **ATLAS** |

---

## 3.3 ATLAS — 창을 넓히고 2차 정보를 쓴다

### 왜 나왔나

Titans의 갱신은 **현재 토큰 하나에 대한 gradient**다 (`3.2`의 `∇ℓ(M_{t-1};x_t)` 참고).
과거 토큰은 momentum과 감쇠를 통해 **암묵적으로만** 반영된다.

📌 [T1] ATLAS 논문은 이를 일반화된 문제로 짚는다: "one of the critical drawback
of most existing recurrent models is their online nature, in which they optimize
the inner objective ... with respect to only the current input" (Behrouz et al.
2025). "방금 것 하나만 보고 메모리를 고치는 게 최선인가?"는 이 비판을 Titans에
적용해본 질문이다.

### 아이디어와 구조

📌 [T1] ATLAS는 **과거 토큰의 윈도우 전체에 대해** 메모리를 최적화한다 (Omega rule).
마지막 토큰이 아니라 창 안의 토큰들을 함께 놓고 메모리를 맞춘다.

| | 무엇에 대해 최적화하나 |
|---|---|
| Titans | 현재 토큰 (과거는 암묵적 감쇠) |
| **ATLAS** | **과거 토큰의 슬라이딩 윈도우** |

📌 [T1] 더 눈에 띄는 건 이쪽이다. ATLAS는 **Muon 옵티마이저를 메모리의 내부 최적화에 쓴다.**
**2차 정보의 근사를 사용하는 첫 병렬화 가능 recurrent 아키텍처**라고 소개한다
(논문 표현: "to the best of our knowledge").

📌 [T1] 정확한 형태(논문 Eq. 32-33)는 이렇다.

```
 S_t = θ_t · S_{t-1} + ∇[Σ γ_i(t) ‖M(φ*(k_i)) − v_i‖²]     (윈도우 전체에 대한 momentum)
 M_t = α_t · M_{t-1} − η_t · NewtonSchulz_k(S_t)             (직교화한 뒤 갱신)
```

`NewtonSchulz_k`는 행렬을 **준정직교(semi-orthogonal) 행렬에 가깝게 만드는 반복
연산**이다 (`k → ∞`이면 `S_t`에 가장 가까운 준정직교 행렬로 수렴). `k`는 그 자체로
**test-time compute 파라미터**다 — 반복을 더 돌릴수록 메모리화 품질이 올라간다고
논문은 설명한다.

Muon이 선택된 이유가 시스템적이다 — **대부분의 연산이 행렬곱이라 시퀀스 방향으로
병렬화된다.** 2차 정보를 쓰면서도 `02` `2.1`의 chunked 병렬화를 유지할 수 있다.

> 💡 **최적화 이론이 아키텍처 안으로 들어오고 있다.**
> 축2가 "1차 경사하강을 재귀로 푼 것"이었다면 ATLAS는 2차로 올라간다.
> 이 축에서는 **옵티마이저 선택이 곧 아키텍처 설계**다.

### 추론에서 달라진 것

| | Titans | ATLAS |
|---|---|---|
| 손실을 계산하는 범위 | 현재 토큰 하나 | **과거 토큰의 슬라이딩 윈도우**(길이 `c`) |
| 내부 최적화 | 1차 (gradient + momentum) | **2차 근사** (Muon, Newton-Schulz 반복) |
| 스텝당 연산 | forward + backward 1회 | forward + backward + **윈도우 집계 + `k`번의 추가 행렬곱(직교화)** |
| 병렬화 단위 | 토큰 | 윈도우(청크) — 청크 경계에서 이전 청크의 마지막 상태에 대해 gradient 계산 |

📌 [T1] 정확한 스텝당 연산량(FLOPs)이나 추론 시 메모리 사용량은 논문에 나오지
않는다 — 위 표는 논문이 밝힌 메커니즘을 정성적으로 비교한 것이다.

### 토큰 하나가 지나가는 길

Titans의 여섯 단계에서 2~5번이 바뀐다.

| # | 하는 일 | Titans와 다른 점 |
|---|---|---|
| 1 | `q, k, v` 만들기 | 동일 |
| 2 | 최근 `c`개 토큰의 윈도우를 구성 | **새로 생김** — 현재 토큰 하나가 아니라 창 전체 |
| 3 | 윈도우 전체에 대한 손실과 gradient(momentum `S_t`) | Titans는 토큰 하나, ATLAS는 `Σ γ_i(t)` 가중합 |
| 4 | Newton-Schulz로 `S_t`를 직교화 (`k`회 반복) | **완전히 새로 생김** |
| 5 | 직교화된 값으로 메모리 갱신 | Titans는 momentum을 그대로 뺐지만, 여기는 직교화를 거친 것을 뺀다 |
| 6 | `q`로 읽기 | 동일 |

### 코드와 텐서

💡 논문은 pseudocode 수준의 연산 횟수를 밝히지 않는다. 아래는 위 수식(Eq. 32-33,
📌 [T1])에서 이 위키가 구성한 것이지, 논문이 직접 제시한 표가 아니다.

| # | 연산 | shape | 종류 |
|---|---|---|---|
| 1 | 윈도우 손실 gradient 누적 | 각 `W_i`와 동일 shape, `c`개 토큰에 대해 반복·가중합 | GEMV × `L_M` × `c` |
| 2 | Newton-Schulz 반복 (`k`회) | `W_i`와 동일한 행렬에 대해 반복 | **GEMM × `k`** (행렬-행렬 곱) |
| 3 | 메모리 갱신 | `W_i`와 동일 | elementwise |

> 💡 Titans의 "코드와 텐서"에서는 메모리 갱신 관련 연산이 전부 GEMV·elementwise였는데,
> 여기서는 **Newton-Schulz 반복이 GEMM을 요구한다.** Muon을 고른 이유가 "행렬곱
> 위주라 병렬화하기 좋아서"였다는 위 "아이디어와 구조"의 설명과 맞아떨어진다.

### 실험에서 확인된 범위

📌 [T1] 논문은 10M 토큰 BABILong에서 Titans의 성능이 떨어지는 반면 ATLAS는
**80% 이상의 정확도**를 유지했다고 보고한다. 1M 토큰까지는 Titans와 ATLAS가
비슷한 성능이었고, 10M에서 격차가 벌어진다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | Omega rule로 **과거 토큰 윈도우 전체**에 대해 메모리를 최적화하고, 내부 옵티마이저로 **Muon**을 써서 2차 정보를 근사한다 |
| **장점** | · Titans의 online one-step update 한계를 보완<br>· **2차 정보를 근사하면서 병렬화 가능** — Muon 연산을 행렬곱으로 구성<br>· 논문은 10M BABILong에서 **80% 이상 정확도**를 보고 |
| **한계** | · 윈도우와 2차 근사로 스텝당 연산이 더 늘어난다<br>· Titans의 시스템 문제(추론 중 backward)를 그대로 물려받음<br>· ⚠️ 연구 단계 |
| **대표 모델** | 연구 단계 (Google Research) |
| **다음으로** | 갱신 규칙 자체를 학습 대상으로 만들면 → **HOPE** |

---

## 3.4 HOPE와 Nested Learning — 갱신 규칙을 학습한다

> ⚠️ **이름 주의.** 여기의 **HOPE**는 `04-position` `4.6`의 **HoPE 위치 인코딩**과
> 전혀 다른 연구다.

### 왜 나왔나

Titans와 ATLAS는 **메모리를 무엇으로, 어떻게 최적화할지**를 발전시켰지만, 둘 다
전제 하나는 공유한다 — attention은 매 토큰, 신경 메모리는 (토큰 단위든 윈도우
단위든) 매번 같은 규칙으로 갱신된다. **모든 메모리가 같은 주기, 같은 규칙으로
갱신되어야 하나?**

📌 [T1] 모델을 **하나의 연속된 학습 과정이 아니라, 서로 다른 갱신 주기를 가진
다층 최적화 문제의 시스템**으로 보는 것이 Nested Learning의 출발점이다.

핵심 주장은 **아키텍처와 옵티마이저를 분리하지 말자**는 것이다. 📌 [T1] 논문
원문: "the model's architecture and the rules used to train it ... are
fundamentally the same concepts; they are just different 'levels' of
optimization." 둘은 별개가 아니라 **같은 것의 서로 다른 층위**라는 시각이다.

> 💡 이 위키의 구성을 되돌아보면 흥미롭다. `06-norm-residual`은 "학습 안정성"이었고
> `02`는 "상태 갱신"이었다. Nested Learning은 **그 둘이 같은 문제의 다른 층위**라고 본다.

### 아이디어와 구조

**CMS — 메모리를 스펙트럼으로**

📌 [T1] **Continuum Memory System.** 메모리를 단기/장기 둘로 나누지 않고
**갱신 주기가 서로 다른 모듈들의 스펙트럼**으로 둔다 — 논문은 이를 서로 다른
빈도 `f_1 > f_2 > ... > f_k`로 동작하는 MLP 블록들의 체인으로 설명한다.

```
 표준 Transformer
   attention (매 토큰 갱신)  ────────  FFN (학습 때만 갱신, 추론 중 고정)
        빠름                                    느림 (사실상 정지)

 CMS
   ●───────●───────●───────●───────●
   빠름                              느림
   즉각 정보                    추상 지식 축적
```
> **그림 3.4** — 두 극단 사이를 여러 단계로 채운다.

> 💡 `01-attention` `1.3`에서 말한 **"두 극단 사이에 눈금 긋기"** 가 또 나온다.
> GQA가 MHA와 MQA 사이에 눈금을 그었듯, CMS는 **attention과 FFN 사이**에 눈금을 긋는다.

**self-modifying — 갱신 방식을 갱신한다**

📌 [T1] HOPE는 **자기 자신을 갱신하는 방식을 갱신할 수 있다.**
자기 참조적(self-referential) 과정으로 메모리를 최적화한다는 것이다.

📌 [T1] 구체적으로는 Q, K, V 투영과 **학습률까지도** 고정 파라미터가 아니라
컨텍스트에서 나오는 함수로 바뀐다: "Qₜ = f_Q(context_t), Kₜ = f_K(context_t),
Vₜ = f_V(context_t)" — 이 `f`들 자체가 학습되는 associative memory다.

| | in-context 학습의 층위 |
|---|---|
| Titans | **2단계** — 파라미터 갱신 + in-context |
| **HOPE** | **무한 층위** + CMS 블록 |

📌 [T1] 결과로는 Titans, Samba, 표준 Transformer 대비 언어모델링 perplexity와
상식 추론 정확도, needle-in-haystack 성능이 모두 개선되었다고 보고한다.

⚠️ 다만 **HOPE는 개념 증명(proof-of-concept)** 으로 소개된다.
공식 구현이 공개되지 않았고 커뮤니티 재현만 있는 상태다.

### 추론에서 달라진 것

| | Titans | HOPE |
|---|---|---|
| 메모리 모듈 수 | 1개 (MLP) | **CMS — 서로 다른 빈도(`f_1 > f_2 > ... > f_k`)의 MLP 체인** |
| 갱신 빈도 | 매 토큰 동일 | **모듈마다 다르다** — 빠른 모듈은 자주, 느린 모듈은 드물게 |
| Q/K/V 생성 | 고정된 선형 투영 | **컨텍스트에서 나오는 함수**(`f_Q, f_K, f_V`) |
| 학습률 | 하이퍼파라미터 | **컨텍스트 의존 함수로 학습됨** |

### 토큰 하나가 지나가는 길

💡 (이 위키의 요약) CMS 블록마다 갱신 빈도가 다르므로, **모든 블록이 매 토큰
forward는 하되, backward/갱신은 자기 주기에 해당할 때만** 일어난다고 이해하면
된다 — 정확한 갱신 스케줄(예: 몇 토큰마다인지)까지는 이 위키가 확인한 범위에서
논문이 pseudocode로 공개하지 않았다.

| # | 하는 일 |
|---|---|
| 1 | 컨텍스트에서 `Q_t, K_t, V_t`와 학습률을 함수로 생성 |
| 2 | 빠른 주기 CMS 블록 — 거의 매 토큰 forward + 갱신 |
| 3 | 느린 주기 CMS 블록 — forward는 하되, 자기 주기가 아니면 갱신은 건너뜀 |
| 4 | 결과를 결합해 출력 |

### 정리

| | |
|---|---|
| **핵심 아이디어** | 모델을 **갱신 주기가 다른 다층 최적화 문제의 시스템**으로 보고, 갱신 방식 자체를 자기 참조적으로 학습한다. 메모리는 주기 스펙트럼(CMS)으로 구성 |
| **장점** | · **아키텍처와 옵티마이저를 하나의 틀로 통합**<br>· CMS가 서로 다른 갱신 주기의 메모리를 하나의 관점으로 설명<br>· continual learning과 long-context memory를 함께 다루는 설계 공간 제시 |
| **한계** | · ⚠️ **연구 단계** — 대규모 프로덕션의 비용·안정성 근거가 아직 부족<br>· 💡(이 위키의 추정) 자기수정 구조라 학습·디버깅과 상태 관리가 더 복잡할 것으로 보임 — 논문이 직접 측정한 항목은 아님<br>· ⚠️ **이름이 `04-position`의 HoPE와 충돌** |
| **대표 모델** | 연구 단계 (Google Research) |

---

## 3.5 시스템 관점 — 추론 중에 학습이 일어난다

💡 **이 절부터 이 파일 끝까지는 대부분 이 위키의 시스템 분석이다.** Titans·ATLAS·HOPE
논문은 서빙 시스템에 대한 함의를 직접 다루지 않는다 (실제로 세 논문 모두 프로덕션
서빙 벤치마크를 보고하지 않는다). 아래 내용은 `3.1`~`3.4`에서 확인한 📌 사실(메모리
갱신에 gradient·optimizer가 쓰인다는 점 등)에서 **이 위키가 논리적으로 추론한 결과**이지,
논문이 검증했거나 실측한 내용이 아니다. 표시가 없어도 이 절의 서술은 기본적으로 추론이라고
읽으면 된다.

이 축이 시스템에 던지는 문제는 앞의 어떤 파일과도 다르다 — 왜냐하면 이 위키가 지금까지
다룬 다른 모든 아키텍처(01~02, 04~10)는 **추론 시 가중치가 고정**이라는 전제를 공유하기
때문이다. 그 전제가 여기서 처음 깨진다.

> **이 위키의 다른 모든 모듈은 추론 시 가중치가 고정이다. 여기는 아니다.**

| | 기존 정적 가중치 계열 | test-time memory |
|---|---|---|
| 추론 시 파라미터 | **고정** | **메모리 모듈이 계속 갱신된다** |
| 요청별 상태 | KV cache 또는 상태 행렬 | **MLP 가중치** |
| 스텝당 연산 | forward만 | **forward + 메모리에 대한 backward** |
| 상태 크기 | 컨텍스트 `S`에 비례하거나 고정 행렬 | 고정 MLP state + bounded local-attention KV |

### read-mostly → read + write → multi-rate read + write

💡 아래 표의 "무엇을 읽나/쓰나" 열은 각 논문의 메커니즘(📌, `3.1`·`3.2`·`3.4` 참고)을
그대로 옮긴 것이지만, **마지막 열("시스템의 중심 질문")은 논문에 나오는 문구가 아니라
이 위키가 각 단계에서 시스템적으로 가장 중요하다고 판단한 질문을 요약한 것**이다.

| 단계 | 추론 workload | 무엇을 읽나 | 무엇을 쓰나 | 시스템의 중심 질문 (💡 이 위키의 요약) |
|---|---|---|---|---|
| **Transformer** | **read-mostly** | 고정 가중치 + 누적 KV | 새 토큰의 KV append | 저장량과 memory traffic을 얼마나 줄일까 |
| **Mamba·KDA** | fixed-rule state read/write | 고정 가중치 + 압축 상태 | 설계된 recurrence로 상태 갱신 | 상태를 정확성과 처리량 사이 어디에 둘까 |
| **Titans** | **learned read + write inference** | 고정 backbone + neural memory | gradient와 optimizer로 memory update | update 비용과 요청별 상태를 어떻게 격리할까 |
| **HOPE** | **multi-rate read + write inference** | 주기가 다른 memory spectrum | 빠른 기억은 자주, 느린 기억은 드물게 갱신 | 어떤 지식을 얼마나 자주 업데이트할까 |

💡 (이 위키의 종합) Transformer 최적화가 주로 "얼마나 덜 읽을 것인가"를 물었다면,
Titans 이후에는 **"무엇을 읽고 무엇을 쓸 것인가"** 가 함께 중요해진다고 볼 수 있다.
HOPE의 CMS는 여기에 시간축을 하나 더 붙여 **"각 기억을 어느 주기로 쓸 것인가"** 를
묻는다 — 이 세 문장의 프레이밍 자체는 논문의 것이 아니라 이 위키가 압축한 것이다.

### 새로 생기는 것 다섯 가지

아래 다섯 가지 중 **①②③은 이 위키의 추정**이다 — Titans·ATLAS·HOPE 논문 어디에도
실제 서빙 배포 시도나 그때 발생한 문제는 보고되지 않는다. `10-serving`에서 확인한
기존 서빙 스택의 전제(📌)에, 이 축의 메커니즘(📌)을 대입했을 때 논리적으로 예상되는
마찰 지점을 이 위키가 도출한 것이다. ④는 논문이 직접 언급한 한계이고, ⑤는 논문이
제시한 해법이라 성격이 다르다.

**① 💡(추정) 추론 경로에 backward가 들어온다.**
서빙 프레임워크는 forward-only를 전제로 만들어져 있다. 커널 선택, 메모리 풀,
스케줄링이 전부 그 가정 위에 있다. `10-serving`에서 본 스택이 이 전제와 충돌할
것으로 예상되지만, 실제로 이런 backward-inference 서빙을 시도하고 무엇이
깨졌는지 보고한 사례는 아직 없다.

**② 💡(추정) 요청별 상태가 "가중치"다.**
`10-serving` `10.3`에서 하이브리드 모델의 prefix caching이 어려웠던 이유가
"상태가 시퀀스 전체에 뭉쳐 있어서"였다. 여기는 그 상태가 **MLP 가중치**다.
SGLang이 Mamba 상태용으로 별도 풀을 만든 선례(`10.3`, 📌 [T2])에 비추면 비슷한
관리 계층이 필요하리라 예상되지만, test-time-memory 전용으로 실제 구현된 사례는
아직 없다.

**③ 💡(추정) prefix caching이 더 어려울 것으로 보인다.**
메모리 갱신이 **비선형이고 경로 의존적**이라 중간 지점의 상태를 재구성하기 어렵다.
`10.3`의 Marconi식 체크포인트(📌 [T1], 하이브리드 SSM 모델 대상)와 같은 접근이
필요해 보이지만, 스냅샷 하나의 크기가 MLP 전체라는 점에서 그대로 적용 가능한지는
검증되지 않았다.

**④ 깊은 메모리는 느리다.**
📌 [T1] 논문도 인정한다 — `L_M`이 커질수록 **선형으로 느려진다.**
`3.2`의 "깊을수록 좋다"와 정면으로 상충하는 제약이다.

**⑤ 병렬화는 풀렸다.**
📌 [T1] 청크 단위 행렬곱 + parallel associative scan으로 학습 병렬화는 해결됐다
(`3.2` "학습은 어떻게 병렬화하나" 참고). `02` `2.1`에서 본 chunked parallel과
같은 계열의 해법이다. 이게 없었으면 대규모 실험 자체가 어려웠을 것이다.

### 압력의 관점에서

💡 (이 위키의 분석) `00-foundations` `0.4`의 분류를 이 축에 적용하면 이런
거래로 정리된다 — 이 분류 자체가 논문에는 없는, 이 위키의 프레임워크다.

| | |
|---|---|
| **더는 것** | 길이에 비례하는 full KV cache가 사라지거나 local window로 제한된다. 장기 상태는 **컨텍스트 길이와 무관** |
| **지는 것** | 스텝당 연산이 늘어난다 (backward), 서빙 스택의 전제가 깨진다 |

**메모리 용량 압력을 연산과 시스템 복잡도로 바꾼다.**
`99-landscape` `99.7`에서 본 대로 용량이 최대 병목이라면 방향은 맞다. 다만
**그 대가가 서빙 인프라 전체의 재작성일 수 있다**는 건 위 ①②③(추정)에 근거한
이 위키의 평가이지, 실측된 마이그레이션 비용이 아니다.

### 미래 accelerator가 새로 가져야 할 것 — 연구 방향 (💡 이 위키의 추정)

아직 Titans·HOPE 전용 가속기가 발표된 것은 아니다. 아래는 논문에 나온 구조적
사실(📌)에서 이 위키가 도출한 요구사항 목록이며, 하드웨어 업계나 논문 저자가
실제로 이렇게 제안했다는 뜻은 아니다.

| 필요한 기능 | 왜 필요한가 | 가능한 하드웨어·런타임 방향 |
|---|---|---|
| **Update Engine** | 작은 메모리 모듈의 forward·gradient·optimizer가 매 토큰/청크 반복 | update fusion, optimizer-state 전용 datapath |
| **Frequency-aware memory hierarchy** | CMS의 빠른 기억과 느린 기억은 접근·갱신 빈도가 다름 | 자주 쓰는 state는 SRAM/HBM 가까이, 느린 state는 HBM/host tier에 배치 |
| **write-aware scheduler** | read-only decode와 update가 대역폭·동기화 지점을 두고 경쟁 | read/write phase overlap, update batching, 주기별 scheduling |
| **요청별 격리와 snapshot** | 사용자마다 memory parameter가 달라짐 | copy-on-write state, checkpoint/delta log, 빠른 rollback |
| **분산 update 일관성** | memory를 여러 칩에 sharding하면 gradient와 state가 함께 이동 | update locality, 계층적 collective, stale update 허용 범위 제어 |

> **미래 accelerator의 질문은 “몇 FLOPS인가”만이 아니다.**
> **어떤 지식을 어느 계층에 놓고, 얼마나 자주 업데이트할 것인가**가 설계 변수가 된다.

---

## 이 다음

여기까지가 `01-attention`에서 갈라진 메모리 계보의 끝이다.
KV를 덜 읽는 attention에서 고정 recurrent state로, 다시 추론 중 학습되는 neural memory로 왔다.

`04-position.md`에서는 attention 안쪽의 다른 축으로 돌아간다.
**모델은 토큰 순서를 어떻게 알고, 학습 때보다 긴 입력을 어떻게 다루는가.**
MLA가 RoPE와 충돌한 이유와 이름이 같은 HoPE/HOPE의 차이도 거기서 이어진다.

---

## Sources

**T1 — 논문**
- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory* — gate를 통한 장기 상태 갱신
- Gu et al. (2021), *Efficiently Modeling Long Sequences with Structured State Spaces* — S4
- Gu & Dao (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
  — 입력 의존 selective state update
- Yang et al. (2024), *Parallelizing Linear Transformers with the Delta Rule* —
  선형 연상 메모리의 오차 수정과 Titans로 이어지는 delta-rule 해석
- Behrouz et al. (2024), *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663
  — 신경 메모리 MLP, surprise gradient, momentum·weight decay, MAC/MAG/MAL, short/long-term
  memory 프레이밍, 16K RULER S-NIAH 결과(Table 2), abstract의 2M+ context 주장
- Behrouz et al. (2025), *ATLAS: Learning to Optimally Memorize the Context at Test Time*,
  arXiv:2505.23735 — Omega rule, Muon 내부 최적화(Eq. 32-33, Newton-Schulz), 10M BABILong
- Behrouz et al. (2025), *Nested Learning: The Illusion of Deep Learning Architectures*,
  arXiv:2512.24695 (NeurIPS 2025) — HOPE, CMS, self-modifying Q/K/V·학습률
- Google Research 블로그, *Introducing Nested Learning* — 중첩 최적화 관점, CMS 설명

**연결되는 파일**
- `02-linear-attention` 2.3~2.4 — delta rule과 이 축의 출발점
- `02-linear-attention` 2.6 — 하이브리드 설계, 고정 상태의 검색 한계
- `01-attention` 1.7 — NSA의 세 갈래 구조 (MAC/MAG/MAL과 대조)
- `10-serving` 10.3 — 하이브리드 상태의 prefix caching 문제

**범위와 주의**
- Titans·ATLAS·HOPE는 연구 제안이다. 대규모 프로덕션 서빙에서의 비용과 안정성은
  아직 확립된 사실로 다루지 않는다.
- `3.5`의 update engine과 다중 주기 메모리 계층은 논문의 구조에서 도출한
  **시스템 연구 방향**이지, 실측된 가속기 설계가 아니다.
- Titans의 "2M+ context" 확장성은 논문 abstract의 서술이고, 표로 제시된 실험은
  16K RULER까지다. 이 문서는 abstract 주장과 표 결과를 구분해 표기한다 (`3.2`).
- `3.5`(시스템 관점) 이후는 이 파일에서 **가장 추정이 많은 구간**이다. Titans·ATLAS·
  HOPE 세 논문 모두 서빙 인프라와의 통합이나 가속기 설계를 다루지 않는다 — `3.5`의
  다섯 가지 중 ①②③, "압력의 관점에서", "미래 accelerator" 표는 논문의 구조적 사실에서
  이 위키가 논리적으로 도출한 **추정**이고, 실제로 이렇게 배포되거나 측정된 적은 없다.
- 💡로 표시된 문단은 이 위키가 여러 논문의 사실을 엮어 내린 해석이며, 논문 저자의
  결론이 아니다.
- ATLAS의 "코드와 텐서"(`3.3`)는 논문의 수식(Eq. 32-33)에서 이 위키가 구성한 것이고,
  HOPE의 "토큰 하나가 지나가는 길"(`3.4`)은 CMS의 개념 설명에서 이 위키가 추정한
  절차다 — 둘 다 논문이 직접 제시한 pseudocode가 아니다.
