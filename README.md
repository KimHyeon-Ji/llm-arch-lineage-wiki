# LLM Architecture Lineage Wiki

LLM 아키텍처를 **아이디어의 계보**로 정리한 자료다.
각 모듈이 왜 나왔고, 어떻게 생겼고, **토큰 하나가 지나갈 때 어떤 연산이 일어나며**,
그것이 시스템에 무엇을 요구하는지를 함께 다룬다.

MHA에서 시작해 DeepSeek-V4의 CSA/HCA, Kimi K3의 KDA까지 이어진다.

---

## 어떻게 읽나

### 처음이라면 — 이 순서

```
 00-foundations  →  01-attention  →  02-linear-attention  →  09-serving  →  99-landscape
   기준점 세우기     KV 압축 본류      고정 상태라는 대안       실전 제약        전체 종합
```

`00`은 반드시 먼저 읽어야 한다. 여기서 정한 기호와 직관을 나머지 전부가 쓴다.

### "이 모듈이 뭔지"만 알고 싶다면

각 절 끝의 **정리 카드**만 봐도 된다. 핵심 아이디어 / 장점 / 한계 / 대표 모델 /
다음으로가 표 하나에 들어 있다.

| 찾는 것 | 어디로 |
|---|---|
| MQA, GQA, MLA | [01-attention](01-attention.md) 1.2~1.4 |
| **CSA, HCA** (DeepSeek-V4) | [01-attention](01-attention.md) 1.9 |
| NSA, MoBA, DSA | [01-attention](01-attention.md) 1.7~1.8 |
| **KDA** (Kimi Linear/K3) | [02-linear-attention](02-linear-attention.md) 2.4 |
| Gated DeltaNet (Qwen3-Next) | [02-linear-attention](02-linear-attention.md) 2.4 |
| RoPE, YaRN, **HoPE** | [03-position](03-position.md) |
| MoE, expert granularity | [04-moe](04-moe.md) |
| **mHC** (DeepSeek-V4) | [05-norm-residual](05-norm-residual.md) 5.6 |
| MTP, speculative decoding | [07-decoding](07-decoding.md) |
| MXFP4 vs NVFP4 | [08-numerics](08-numerics.md) 8.3 |
| PagedAttention, prefix caching | [09-serving](09-serving.md) |
| **모델별 조합 비교** | [99-landscape](99-landscape.md) 99.1 |

### 시스템 관점에서 보고 싶다면

```
 00-foundations 0.3~0.6  →  09-serving 9.7  →  99-landscape 99.6
   자원과 병목의 기초         무엇이 먼저 터지나    시스템이 요구받는 것
```

---

## 파일 지도

| 파일 | 다루는 압력 | 핵심 질문 |
|---|---|---|
| **[00-foundations](00-foundations.md)** | — | 기준점. 토큰 하나가 어떻게 흐르고 무엇이 비싼가 |
| **[01-attention](01-attention.md)** ★ | 메모리 | KV를 어떻게 줄일 것인가 (10개 모듈) |
| **[02-linear-attention](02-linear-attention.md)** | 메모리 | KV를 아예 안 만들면? |
| **[03-position](03-position.md)** | 길이 일반화 | 순서를 어떻게 알려주고, 학습보다 긴 입력을 어떻게 다루나 |
| **[04-moe](04-moe.md)** | 연산량 · 통신 | 파라미터는 늘리고 연산은 그대로 |
| **[05-norm-residual](05-norm-residual.md)** | 안정성 | 깊은 모델을 어떻게 버티게 하나 |
| **[06-shape](06-shape.md)** | 직렬 지연 | 몇 층을 얼마나 넓게 |
| **[07-decoding](07-decoding.md)** | 직렬 지연 | 한 스텝에 토큰 하나여야 하나 |
| **[08-numerics](08-numerics.md)** | 전부 | 값 하나에 몇 비트를 쓸 것인가 |
| **[09-serving](09-serving.md)** ★ | 전부 | 실제로 돌릴 때 무엇이 먼저 터지나 |
| **[99-landscape](99-landscape.md)** ★ | — | 누가 무엇을 골랐고 시스템에 무엇을 요구하나 |

부속 문서

| | |
|---|---|
| [PLAN.md](PLAN.md) | 이 위키의 구성 원칙과 작성 규칙 |
| [CONTESTED.md](CONTESTED.md) | 자료마다 다르게 말하는 것들 (C1~C5) |
| [snippets/](snippets/) | 실행 가능한 최소 구현과 등가성 검증 |

---

## 전체를 관통하는 것

`00-foundations` `0.4`에서 세우는 관점이다.

Transformer의 뼈대는 7년째 거의 그대로인데 세부는 계속 바뀌어 왔다.
그 변화를 밀어낸 압력이 여섯 가지다.

| 압력 | 문제 | 대응 |
|---|---|---|
| **메모리** | 컨텍스트가 길어지면 KV cache를 감당할 수 없다 | 01, 02, 08 |
| **연산량** | 모델을 키우면 토큰당 연산이 그대로 늘어난다 | 04 |
| **직렬 지연** | 레이어는 순서대로 통과할 수밖에 없다 | 06, 07 |
| **통신** | 모델이 여러 GPU에 흩어져 있다 | 04, 09 |
| **안정성** | 깊고 큰 모델은 학습이 잘 터진다 | 05 |
| **길이 일반화** | 학습 때보다 긴 입력을 다뤄야 한다 | 03 |

그리고 매번 확인하게 되는 것 하나.

> **한 압력을 풀면 대개 다른 압력으로 옮겨간다.**
> MoE는 연산량을 풀지만 통신 부담을 만든다. 희소 attention은 읽는 양을 줄이지만
> 메모리 접근을 흩어놓는다.
>
> **공짜인 구조는 없다. 무엇을 무엇과 바꿨는가 — 이게 각 모듈에서 확인할 것이다.**

---

## 문서를 읽는 규칙

### 출처 표시

| 표시 | 뜻 |
|---|---|
| 📌 **[T1]** | 논문·공식 리포트·공식 구현에서 확인한 것 |
| 📌 **[T2]** | vLLM·SGLang·커널 구현 등에서 확인한 것 |
| 📎 **[T3, 미검증]** | 해설 자료 기반. **원문 대조 전** |
| ⚠️ | 자료 간 충돌 또는 미확인 — [CONTESTED.md](CONTESTED.md) 참조 |
| ✅ | `snippets/`에서 코드로 검증한 것 |
| 💡 | 직관·해석 (사실 주장이 아님) |

**⚠️가 붙은 것은 그대로 인용하지 말 것.** 특히 DeepSeek-V4(CSA/HCA/mHC)와
Kimi K3 관련 내용은 원문 대조가 덜 되어 있다.

### 각 절의 구조

모듈마다 여섯 부분이다.

```
 왜 나왔나            앞 모듈이 어디서 막혔는지
 아이디어와 구조       말 → 그림 → 수식 순서. 가장 긴 부분
 추론에서 달라진 것    before/after 표
 토큰 하나가 지나가는 길  decode 한 스텝의 순서
 코드와 텐서          어떤 연산이 어떤 shape 변화로 (GEMM/GEMV/gather/…)
 정리                핵심 / 장점 / 한계 / 대표 모델 / 다음으로
```

모듈과 모듈 사이에는 **〈이어지는 흐름〉**이 있다. 앞 모듈이 남긴 문제가
다음 모듈을 불러오는 부분이고, 계보의 본체가 거기 있다.

---

## 현재 상태

### 검증 수준

| 범위 | 상태 |
|---|---|
| MHA ~ MLA, RoPE, MoE 기본, 정규화 | **T1 확인** |
| NSA, MoBA, DSA, KDA, mHC | **T1 확인** (파라미터·수식 포함) |
| **CSA, HCA (DeepSeek-V4)** | ⚠️ **원문 미대조** — 골격만 |
| **Kimi K3 세부** | ⚠️ 2차 자료 기반 |
| GLM-5, Gemma 4 세부 | ⚠️ 2차 자료 기반 |
| 하드웨어 스펙 (B200, Rubin) | ⚠️ 공식 스펙 미확인 |
| `99-landscape` 99.6 결론 | ⚠️ **구조에서 추론한 가설** |

### 미해결 (CONTESTED)

| # | 쟁점 |
|---|---|
| C1 | HCA = "Heavily Compressed" vs "Hyper-Connected" |
| C2 | HoPE 동명이인 3종 |
| C3 | CSA의 압축률 `m` |
| C4 | B200·Rubin 하드웨어 스펙 |
| C5 | NSA → DSA → CSA 계승 관계 |

자세한 내용과 해소 방법은 [CONTESTED.md](CONTESTED.md).

### 다음에 할 일

1. **DeepSeek-V4 원문 대조** — C1, C3 해소. `01-attention` 1.9와 `05` 5.6이 바뀔 수 있다
2. Kimi K3 / GLM-5 기술 리포트 대조
3. `snippets/` 확장 — MLA absorption, DeltaNet chunked 등가성
4. NVIDIA 공식 스펙시트로 `00-foundations` 0.8 갱신
