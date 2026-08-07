# 05. Norm & Residual — 깊은 모델을 버티게 하는 것

지금까지의 파일들은 전부 **비용**에 관한 것이었다. 메모리를 줄이고, 연산을 줄이고,
통신을 관리하는 이야기였다.

이 파일은 다르다. `00-foundations` `0.4`의 분류로 보면 **안정성 압력**이다.
여기 나오는 기법들은 빨라지려고 나온 게 아니다. **애초에 학습이 되게 하려고** 나왔다.

문제는 이렇게 시작한다.

> 레이어를 60층, 80층 쌓으면 gradient가 아래까지 도달하지 못하거나,
> 중간에 값이 폭주해서 학습이 터진다. **깊게 쌓는 것 자체가 어렵다.**

residual connection과 normalization은 그 문제에 대한 답이고, 이 파일은
그 답이 20년 가까이 다듬어져 온 과정이다. 마지막에는 **residual 자체를 다시 설계한**
DeepSeek-V4의 mHC까지 간다.

---

## 계보 지도

```
              Post-LN (2017)
          LN(x + Sublayer(x))
          gradient가 막힌다
                  │
              Pre-LN (2019~)
          x + Sublayer(LN(x))
          gradient는 뚫렸는데
          표현이 무뎌진다
                  │
        ┌─────────┴─────────┐
        │                   │
  [norm을 싸게]       [norm을 더 놓기]
        │                   │
   RMSNorm (2019)      QK-Norm
   평균을 버린다        sandwich / peri-LN
        │             zero-centered
        │             depth-scaled gain
        └─────────┬─────────┘
                  │
        [residual 자체를 다시]
                  │
        Hyper-Connections (2024)
        residual stream을 4개로
                  │
              mHC (2025)
        항등 사상을 되찾다
        (DeepSeek-V4)
```
> **그림 5.0** — 축5의 계보

**목차**

| | 절 | 한 줄 |
|---|---|---|
| [5.1](#51-post-ln--residual-위에-norm이-있으면) | **Post-LN** | residual 위에 norm이 있으면 |
| [5.2](#52-pre-ln--위치를-바꾸다) | **Pre-LN** | 위치를 바꾸다 |
| [5.3](#53-rmsnorm--평균을-버리다) | **RMSNorm** | 평균을 버리다 |
| [5.4](#54-qk-norm과-변형들--폭주를-막는-여러-방법) | **QK-Norm 외** | 폭주를 막는 여러 방법 |
| [5.5](#55-hyper-connections--residual을-넓히다) | **Hyper-Connections** | residual을 넓히다 |
| [5.6](#56-mhc--항등-사상을-되찾다) | **mHC** | 항등 사상을 되찾다 |

---

## 5.1 Post-LN — residual 위에 norm이 있으면

### 구조

Transformer 원 논문의 배치는 이랬다.

```
 x ──┬──► Sublayer ──► (+) ──► LayerNorm ──► 출력
     └────────────────┘
```

sublayer 결과를 더한 **다음에** 정규화한다.

### 문제

residual connection의 존재 이유는 **gradient가 아래층까지 곧장 흐르는 통로**를 만드는 것이다.
그런데 Post-LN은 그 통로 위에 정규화를 올려놓았다. gradient가 층을 지날 때마다
LayerNorm을 통과해야 하고, 깊어질수록 왜곡이 누적된다.

실무적으로는 이렇게 나타난다.

- **learning rate warmup 없이는 학습이 터진다**
- 층이 깊어질수록 하이퍼파라미터에 민감해진다
- 12층은 괜찮은데 48층은 발산한다

### 정리

| | |
|---|---|
| **핵심 아이디어** | sublayer 출력을 residual에 더한 뒤 정규화 |
| **장점** | · 각 층의 출력 분포가 잘 정돈된다<br>· 충분히 안정화만 되면 Pre-LN보다 최종 성능이 좋다는 보고가 있다 |
| **한계** | · **residual 경로 위에 norm이 있어 gradient 흐름이 막힌다**<br>· warmup 등 학습 기교 없이는 깊은 모델을 학습할 수 없다<br>· 층이 깊어질수록 불안정 |
| **대표 모델** | 원조 Transformer, BERT |
| **다음으로** | norm을 residual 경로 **밖으로** 빼면 → **Pre-LN** |

---

## 5.2 Pre-LN — 위치를 바꾸다

### 아이디어

정규화를 sublayer **앞**으로 옮긴다. 그러면 residual 경로가 깨끗해진다.

```
 Post-LN:  x ──┬──► Sublayer ──► (+) ──► LN ──►
               └───────────────────┘
                                  ▲
                          여기 norm이 끼어 있다

 Pre-LN:   x ──┬──► LN ──► Sublayer ──► (+) ──►
               └──────────────────────────┘
                          ▲
                  residual 경로에 아무것도 없다
```
> **그림 5.2** — 화살표 하나 옮긴 것이 전부인데 학습 안정성이 크게 달라진다.

이제 gradient는 residual을 타고 **아무 방해 없이** 첫 층까지 흐른다.
warmup 없이도 학습되고, 100층이 넘어도 터지지 않는다.

### 그런데 새 문제가 생긴다

Pre-LN에서는 residual stream이 **층을 지날수록 계속 커진다.** 매 층이 뭔가를 더하는데
그것을 정규화하는 지점이 없기 때문이다.

결과적으로 깊은 층의 기여가 상대적으로 작아진다. 100층짜리 모델에서 90번째 층이
더하는 값은 이미 커질 대로 커진 residual에 비하면 미미하다.
**표현이 무뎌진다(representation collapse).**

> 💡 **여기서 시소가 만들어진다.**
>
> | | gradient 흐름 | 표현력 |
> |---|---|---|
> | Post-LN | ❌ 막힘 | ✅ 좋음 |
> | Pre-LN | ✅ 뚫림 | ❌ 무뎌짐 |
>
> 한쪽을 얻으면 한쪽을 잃는다. `5.5`의 Hyper-Connections가 이 시소를 정면으로 겨냥한다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | 정규화를 sublayer 앞으로 옮겨 residual 경로를 비운다 |
| **장점** | · **gradient가 아래층까지 곧장 흐른다**<br>· warmup 없이도 학습되고 깊이를 크게 늘릴 수 있다<br>· 하이퍼파라미터에 덜 민감 |
| **한계** | · **residual stream이 층마다 커져 깊은 층의 기여가 묻힌다**<br>· 완전히 안정화된 Post-LN보다 최종 성능이 떨어진다는 보고 |
| **대표 모델** | GPT-2 이후 사실상 전부 — Llama, Qwen, Gemma, DeepSeek |
| **다음으로** | 위치는 정해졌다. 이제 정규화 자체를 싸게 만든다 → **RMSNorm** |

---

## 5.3 RMSNorm — 평균을 버리다

### 아이디어

LayerNorm은 두 가지를 한다. **중심을 맞추고(평균 빼기), 크기를 맞춘다(표준편차로 나누기).**

```
 LayerNorm:  (x − μ) / σ · γ + β
 RMSNorm:     x / RMS(x) · γ            RMS(x) = √(mean(x²))
```

RMSNorm은 **평균 빼기와 bias를 버린다.** 크기만 맞춘다.

관찰은 단순했다. 평균 중심화가 성능에 크게 기여하지 않더라는 것이다.
그렇다면 뺄 이유가 없다.

### 왜 이득인가 — 연산 관점

| | LayerNorm | RMSNorm |
|---|---|---|
| reduction 횟수 | **2회** (평균, 분산) | **1회** (제곱합) |
| 파라미터 | `γ`, `β` | `γ`만 |
| 연산 | 빼기 + 나누기 | 나누기 |

reduction이 하나 줄어든 것이 핵심이다. `00-foundations` `0.3`에서 봤듯
**reduction은 축을 따라 값을 모아야 해서 동기화가 필요하고 병렬화가 덜 된다.**

레이어마다 정규화가 두 번(attention 앞, FFN 앞) 있고 레이어가 80개면
reduction이 160회에서 80회로 줄어든다. 개별로는 작지만 memory-bound인 decode에서
무시할 수 없는 차이다.

### 코드와 텐서

| # | 연산 | shape 변화 | 종류 |
|---|---|---|---|
| 1 | 제곱합 | `(B,1,d)` → `(B,1,1)` | **reduction** |
| 2 | 역제곱근 | `(B,1,1)` | elementwise |
| 3 | 스케일 + gain | `(B,1,d)` 유지 | elementwise |

연산량은 미미하지만 **`x`를 두 번 읽어야 한다** (제곱합 한 번, 스케일 한 번).
그래서 실제 커널은 정규화와 다음 GEMV를 융합(fusion)해서 읽기를 줄인다.
논문에는 없고 구현에만 있는 최적화다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | LayerNorm에서 평균 중심화와 bias를 빼고 크기 정규화만 남긴다 |
| **장점** | · **reduction이 2회에서 1회로** — 동기화 비용 감소<br>· 파라미터와 연산이 줄어든다<br>· 성능 손실이 사실상 없다 |
| **한계** | · 이득이 크지 않아 단독으로는 지엽적인 개선<br>· 평균 중심화가 필요한 상황이 있는지는 완전히 정리되지 않음 |
| **대표 모델** | **Llama 전 라인, Qwen, Gemma, DeepSeek, Mistral** — 2023년 이후 사실상 전부 |
| **다음으로** | 정규화를 더 놓아야 할 곳이 있었다 → **QK-Norm** |

---

## 5.4 QK-Norm과 변형들 — 폭주를 막는 여러 방법

### QK-Norm

학습이 진행되면 attention의 logit(`q·k`)이 점점 커지는 현상이 있다.
logit이 커지면 softmax가 포화되어 거의 one-hot이 되고, gradient가 사라지면서
학습이 불안정해진다. 심하면 손실이 발산한다.

해법은 직접적이다. **`q`와 `k`를 각각 정규화한 뒤 내적한다.**

```
 기존:     softmax( q · kᵀ / √d_h )
 QK-Norm:  softmax( norm(q) · norm(k)ᵀ / √d_h )
```

크기가 통제되니 logit이 폭주할 수 없다. 단순한데 효과가 확실해서
2024년 이후 많은 모델이 채택했다.

### 변형들

**zero-centered RMSNorm** — gain 파라미터를 `γ` 대신 `1 + w`로 두고 `w`를 학습한다.
초기값이 자연스럽게 항등에 가까워져 학습 초반이 안정된다.
Qwen3-Next에서 gated attention(`01-attention` `1.5`)과 함께 쓰인다.

**depth-scaled gain** — gain을 `1/√L`로 초기화한다 (`L`은 총 층수).
`5.2`에서 본 "residual stream이 층마다 커지는" 문제를 초기화 단계에서 억제한다.
학습이 진행되며 필요한 만큼 커지도록 둔다. Arcee Trinity Large가 쓴다.

**norm 배치 변형** — sublayer 앞뒤 모두에 norm을 두는 sandwich norm,
그 변형인 peri-LN 등이 있다. Post-LN과 Pre-LN 사이 어딘가를 찾으려는 시도들이다.

> 💡 `01-attention` `1.3`에서 말한 **"두 극단 사이에 눈금 긋기"** 가 여기서도 반복된다.
> Post-LN도 Pre-LN도 아닌 중간 지점을 찾는 것이다.

### 반례 하나 — QK-Norm을 뺀 모델

⚠️ Tiny Aya는 **QK-Norm을 의도적으로 제거**했다. 긴 컨텍스트 성능과 상호작용한다는
이유였다고 보고된다.

이런 반례를 기록해두는 게 중요하다. QK-Norm이 무조건 좋은 게 아니라,
**어떤 조건에서 무엇과 충돌하는지**가 아직 완전히 정리되지 않았다는 뜻이기 때문이다.

### 코드와 텐서

`01-attention` `1.1` 표의 4번(RoPE) 근처에 두 줄이 추가된다.

| # | 연산 | shape 변화 | 종류 |
|---|---|---|---|
| 3.5 | `q` 정규화 | `(B,1,n_h,d_h)` 유지 | **reduction + elementwise** |
| 3.6 | `k` 정규화 | `(B,1,n_kv,d_h)` 유지 | **reduction + elementwise** |

헤드 차원(`d_h`=128) 안에서의 reduction이라 크기가 작다. 비용은 거의 없다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | `q`와 `k`를 정규화한 뒤 내적해 attention logit의 폭주를 막는다 |
| **장점** | · **학습 안정성이 크게 개선** — 발산을 직접적으로 방지<br>· 비용이 거의 없다 (작은 reduction 두 번)<br>· 큰 모델·긴 학습에서 특히 효과적 |
| **한계** | · ⚠️ **긴 컨텍스트 성능과 상호작용한다는 보고**가 있어 무조건적이지 않다<br>· 정규화 위치·형태(zero-centered 등)에 따라 결과가 달라짐<br>· 근본 원인(왜 logit이 커지는가)을 해결하는 건 아니다 |
| **대표 모델** | OLMo 2 · Gemma 3 · Qwen3 시리즈 · GLM-4.5 · MiniMax M2 · Arcee Trinity<br>제외: **Tiny Aya**(의도적 제거) |
| **다음으로** | 여기까지는 전부 norm을 어디에 어떻게 놓을지의 문제였다. 이제 **residual 자체를 바꾼다** → **Hyper-Connections** |

---

`5.2`에서 만든 시소를 다시 보자. Post-LN은 gradient가 막히고, Pre-LN은 표현이 무뎌진다.
`5.4`의 변형들은 그 사이 어딘가를 찾으려는 시도였지만, 결국 **같은 축 위에서
위치를 조정하는 것**이었다.

2024년의 질문은 달랐다. **residual 경로가 하나뿐이어야 할 이유가 있나?**

---

## 5.5 Hyper-Connections — residual을 넓히다

### 아이디어

residual stream을 **여러 개로 늘린다.** 확장률 `n`을 4로 두면 스트림이 4줄이 된다.

```
 기존 residual                Hyper-Connections (n=4)

   x                          x₁  x₂  x₃  x₄
   │                           │   │   │   │
   ├─► Sublayer ─┐             └─┬─┴───┴───┘
   │             │            [어디서 읽을지 학습]
   └────────────(+)                │
   │                            Sublayer
   ↓                               │
                              [어디에 쓸지 학습]
                                   │
                          x₁' x₂' x₃' x₄'
```
> **그림 5.5** — 각 레이어가 어느 스트림에서 읽고 어느 스트림에 쓸지를 학습으로 결정한다.

두 종류의 연결이 있다.

| 연결 | 무엇 |
|---|---|
| **depth-connection** | 레이어를 지나며 스트림이 어떻게 이어지는지 |
| **width-connection** | 스트림들끼리 어떻게 섞이는지 |

둘 다 학습된 가중치로 결정된다.

### 왜 시소를 벗어나나

레이어마다 **자기에게 맞는 residual 경로를 고를 수 있기 때문**이다.

어떤 레이어는 gradient가 잘 흐르는 깨끗한 경로가 필요하고, 어떤 레이어는
자기 출력이 잘 반영되는 경로가 필요하다. 스트림이 하나면 모든 레이어가
같은 타협을 강요받지만, 여러 개면 나눠 쓸 수 있다.

파라미터는 거의 늘지 않는다. 추가되는 건 혼합 계수뿐이고, 이건 `n × n` 정도의
작은 행렬이다.

### 대가

**residual stream이 `n`배가 된다.** 확장률 4면 활성 메모리가 4배다.
파라미터가 아니라 활성값이므로 학습에서 특히 부담이 되고,
추론에서도 배치가 클수록 영향이 있다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | residual stream을 `n`개로 확장하고, 각 레이어가 어느 스트림에서 읽고 쓸지를 학습한다 |
| **장점** | · **Pre-LN / Post-LN 시소를 구조적으로 벗어난다**<br>· 레이어마다 다른 residual 전략을 쓸 수 있다<br>· 추가 파라미터가 거의 없다 (혼합 계수만) |
| **한계** | · **활성 메모리가 `n`배** (확장률 4면 4배)<br>· **residual의 항등 사상 성질이 깨진다** → 학습 불안정, 확장성 제약<br>· 구현이 복잡해지고 커널 융합이 어려워짐 |
| **대표 모델** | 연구 단계 → mHC로 이어짐 |
| **다음으로** | 항등 사상이 깨진 것이 결정적 문제였다. 그것만 되찾으면 → **mHC** |

---

## 5.6 mHC — 항등 사상을 되찾다

### 무엇이 문제였나

residual connection이 깊은 네트워크를 학습 가능하게 만든 핵심 이유는
**항등 사상(identity mapping)** 이다.

```
 x + f(x)     f가 아무것도 안 하면(f=0)  →  x 가 그대로 통과
```

레이어가 "아무것도 안 하기"를 선택할 수 있다는 것. 이 성질 덕분에
층을 아무리 쌓아도 최소한 나빠지지는 않는다.

Hyper-Connections는 스트림들을 학습된 가중치로 섞으면서 **이 성질을 잃는다.**
혼합 행렬이 임의의 값을 가질 수 있으니 신호가 증폭되거나 소멸할 수 있다.
그래서 학습이 불안정해지고 규모를 키우기 어려웠다.

### 아이디어 — 혼합 행렬을 제약한다

mHC의 답은 **혼합 행렬이 아무 값이나 갖지 못하게 하는 것**이다.
구체적으로는 **이중 확률 행렬(doubly stochastic matrix)** 로 만든다.

```
 이중 확률 행렬의 조건
   · 모든 원소가 0 이상
   · 각 행의 합이 1
   · 각 열의 합이 1
```

이런 행렬의 집합을 **Birkhoff polytope**이라 부른다.
"manifold-constrained"의 manifold가 이것이다.

> 💡 **왜 이 제약이 항등 사상을 되찾아주나.**
> 행 합과 열 합이 모두 1이라는 건 **섞되 총량은 보존한다**는 뜻이다.
> 신호를 스트림 사이에 재분배할 수는 있지만 증폭하거나 소멸시킬 수는 없다.
> 그리고 항등 행렬 자체가 이 집합에 속하므로, **"아무것도 안 하기"가 여전히 선택지에 있다.**

### 어떻게 사영하나 — Sinkhorn-Knopp

임의의 행렬을 이중 확률 행렬로 만드는 고전적 방법이 있다.
**행을 정규화하고, 열을 정규화하고, 이를 반복한다.** Sinkhorn-Knopp 반복이다.

📌 [T1] mHC는 **20회 반복**으로 근사 사영한다.
행렬 크기가 `n × n` = 4×4 정도로 아주 작아서, 반복을 20번 해도 비용이 미미하다.

📌 [T1] 4배 넓은 residual stream을 쓰면서 **학습 시간 오버헤드는 약 6.7%** 로 보고된다.

이 설정은 DeepSeek-V4 `config.json`에 그대로 들어 있다.

| 필드 | 값 | 뜻 |
|---|---|---|
| `hc_mult` | **4** | residual stream 확장률 `n` |
| `hc_sinkhorn_iters` | **20** | Sinkhorn-Knopp 반복 횟수 |
| `hc_eps` | 1e-06 | 수치 안정용 epsilon |

논문의 설정이 실제 배포 모델에 그대로 쓰였음을 확인할 수 있다.
`hc_` 접두사가 **Hyper-Connections**를 가리킨다는 점도 여기서 분명해진다 —
`01-attention` 1.9의 HCA(Heavily Compressed Attention)와는 **별개**다.

### 추론에서 달라진 것

| | 일반 residual | mHC |
|---|---|---|
| residual stream | 1개 | **4개** |
| 활성 메모리 | 기준 | **약 4배** |
| 레이어 간 연산 | 덧셈 | 덧셈 + **작은 혼합 행렬 곱** |
| Sinkhorn 반복 | — | 학습 시. 추론에서는 계수가 고정 ⚠️ |
| KV cache | — | **영향 없음** |

마지막 줄이 중요하다. **mHC는 KV cache와 무관하다.**
DeepSeek-V4에서 CSA/HCA(`01-attention` `1.9`)와 함께 발표되어 묶여 보이기 쉽지만,
전혀 다른 문제를 푸는 별개의 기여다.

> ⚠️ 추론 시 Sinkhorn 사영을 미리 계산해둘 수 있는지는 원문 확인 전이다.
> 혼합 계수가 학습 후 고정이라면 사영도 한 번만 하면 되는 것이 자연스럽지만,
> 확인 전까지는 단정하지 않는다.

### 코드와 텐서

| # | 연산 | shape 변화 | 종류 |
|---|---|---|---|
| 1 | 스트림에서 읽기 | `(B,1,n,d)` → `(B,1,d)` | 가중합 (elementwise) |
| 2 | sublayer | — | (attention 또는 FFN) |
| 3 | 혼합 행렬 사영 | `(n,n)` → `(n,n)` | **Sinkhorn 반복** (학습 시) |
| 4 | 스트림에 쓰기 | `(B,1,d)` → `(B,1,n,d)` | 작은 행렬 곱 + elementwise |

`n`=4이므로 3·4번의 행렬은 4×4다. **연산량은 무시할 수준이고, 비용은 거의 전부
1·4번에서 `n`배가 된 활성 메모리 트래픽**이다.

### 정리

| | |
|---|---|
| **핵심 아이디어** | Hyper-Connections의 혼합 행렬을 이중 확률 행렬로 제약해 residual의 항등 사상 성질을 되찾는다 |
| **장점** | · **HC의 표현력을 유지하면서 학습 안정성 회복**<br>· 항등 행렬이 제약 집합 안에 있어 "아무것도 안 하기"가 가능<br>· 📌 4배 residual stream에 학습 시간 오버헤드 약 6.7%<br>· KV cache와 무관해 attention 쪽 기법과 자유롭게 조합 |
| **한계** | · **활성 메모리가 약 4배** — 배치 크기 상한에 영향<br>· Sinkhorn 반복이라는 생소한 연산이 추가됨<br>· ⚠️ 추론 시 사영을 미리 계산 가능한지 원문 미확인<br>· 채택 사례가 아직 DeepSeek-V4 계열뿐 |
| **대표 모델** | **DeepSeek-V4-Pro / V4-Flash** |
| **다음으로** | residual의 구조를 봤으니, 이제 모델 전체의 **형상**을 본다 → **06-shape** |

---

## 이 파일의 정리

| 모듈 | 무엇을 바꿨나 | 비용 |
|---|---|---|
| **Post-LN** | 원형 | — |
| **Pre-LN** | norm을 residual 밖으로 | 없음 |
| **RMSNorm** | 평균 중심화 제거 | **감소** |
| **QK-Norm** | q·k를 정규화 | 거의 없음 |
| **zero-centered / depth-scaled** | gain의 초기화·파라미터화 | 없음 |
| **Hyper-Connections** | residual stream을 4개로 | **활성 메모리 4배** |
| **mHC** | 혼합 행렬을 이중 확률로 제약 | 활성 4배 + 6.7% 학습 시간 |

읽어둘 흐름은 이렇다.

1. **`5.1`→`5.2`는 화살표 하나 옮긴 이야기**인데, 그것으로 깊은 모델의 학습 가능성이 갈렸다.
2. **`5.3`→`5.4`는 그 위에서의 미세 조정**이다. 개별 효과는 작지만 큰 모델에서는 결정적이다.
3. **`5.5`→`5.6`은 residual 자체를 재설계한다.** 7년 동안 건드리지 않던 부분이라
   최근 아키텍처 변화 중 가장 근본적인 축에 속한다.

그리고 이 파일 전체가 **비용이 아닌 압력도 아키텍처를 바꾼다**는 사실을 보여준다.
RMSNorm과 mHC를 제외하면 여기 나온 것들은 속도와 거의 무관하다.
그래도 없으면 모델이 학습되지 않는다.

---

## Sources

**T1 — 논문**
- Vaswani et al. (2017), *Attention Is All You Need*, arXiv:1706.03762 — Post-LN
- Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture*, arXiv:2002.04745 — Pre-LN 분석
- Zhang & Sennrich (2019), *Root Mean Square Layer Normalization*, arXiv:1910.07467 — RMSNorm
- Dehghani et al. (2023), *Scaling Vision Transformers to 22B* — QK-Norm 대중화
- OLMo 2 기술 리포트 — QK-Norm 채택 근거
- Zhu et al. (2024), *Hyper-Connections*, arXiv:2409.19606 — 확장률, depth/width connection
- DeepSeek-AI (2025), *mHC: Manifold-Constrained Hyper-Connections*, arXiv:2512.24880
  — Birkhoff polytope 사영, Sinkhorn-Knopp 20회, 확장률 4, 학습 시간 +6.7%
- DeepSeek-AI (2026), *DeepSeek-V4*, arXiv:2606.19348 — mHC 실전 적용 ⚠️ 원문 미대조

**T2 — 구현**
- HuggingFace `transformers` RMSNorm 구현
- vLLM / TensorRT-LLM의 norm + GEMV 커널 융합
- mHC 공개 구현 (커뮤니티) — 참고용

**T3 — 참고**
- Sebastian Raschka, 분기별 아키텍처 리뷰 — QK-Norm·zero-centered·depth-scaled gain 채택 현황,
  **Tiny Aya의 QK-Norm 제거** 사례

**미검증 항목**
- mHC의 추론 시 Sinkhorn 사영 사전 계산 가능 여부 — 구현 코드 대조 필요
- Tiny Aya의 QK-Norm 제거 근거 — 2차 자료 기반
