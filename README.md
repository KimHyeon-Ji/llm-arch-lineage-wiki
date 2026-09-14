# LLM Architecture Lineage Wiki

LLM 아키텍처를 **아이디어의 계보**로 정리한 자료다.
각 모듈이 왜 나왔고, 어떻게 생겼고, **토큰 하나가 지나갈 때 어떤 연산이 일어나며**,
그것이 시스템에 무엇을 요구하는지를 함께 다룬다.

MHA에서 시작해 sparse·linear attention을 거쳐, 추론 중 메모리를 갱신하는
Titans·HOPE까지 이어진다. 최신 모델은 사례로 쓰되, 계보를 이해하는 데 필요한
구조와 시스템 영향에 초점을 둔다.

---

## 어떻게 읽나

### 처음이라면 — 이 순서

```
 00-foundations  →  01-attention  →  02-linear-attention  →  03-test-time-memory  →  10-serving  →  99-landscape
   기준점 세우기     KV를 덜 읽기       고정 상태로 바꾸기          읽으면서 쓰기          실전 제약        전체 종합
```

`00`은 반드시 먼저 읽어야 한다. 여기서 정한 기호와 직관을 나머지 전부가 쓴다.

### "이 모듈이 뭔지"만 알고 싶다면

각 절 끝의 **정리 카드**만 봐도 된다. 핵심 아이디어 / 장점 / 한계 / 대표 모델 /
다음으로가 표 하나에 들어 있다.

| 찾는 것 | 어디로 |
|---|---|
| MQA, GQA, MLA | [01-attention](01-attention.md) 2.1~2.3 |
| **CSA, HCA** (DeepSeek-V4) | [01-attention](01-attention.md) 3.4 |
| NSA, MoBA, DSA | [01-attention](01-attention.md) 3.2~3.3 |
| **KDA** (Kimi Linear/K3) | [02-linear-attention](02-linear-attention.md) 2.3 |
| Gated DeltaNet (Qwen3-Next) | [02-linear-attention](02-linear-attention.md) 2.3 |
| RoPE, YaRN, **HoPE** | [04-position](04-position.md) |
| MoE, expert granularity | [05-moe](05-moe.md) |
| **mHC** (DeepSeek-V4) | [06-norm-residual](06-norm-residual.md) 6.6 |
| MTP, speculative decoding | [08-decoding](08-decoding.md) |
| **DFlash, DSpark** (block diffusion 초안) | [08-decoding](08-decoding.md) 8.6 |
| MXFP4 vs NVFP4 | [09-numerics](09-numerics.md) 9.3 |
| PagedAttention, prefix caching | [10-serving](10-serving.md) |
| **Titans, ATLAS, HOPE** (Google) | [03-test-time-memory](03-test-time-memory.md) |
| **Sleep** (wake/sleep 분리, 오프라인 consolidation) | [03-test-time-memory](03-test-time-memory.md) 3.5 |
| **모델별 조합 비교** | [99-landscape](99-landscape.md) 99.1 |

### 시스템 관점에서 보고 싶다면

```
 00-foundations 0.3~0.8  →  03-test-time-memory 3.5  →  10-serving 10.7  →  99-landscape 99.6~99.7
   자원·HW 병목의 변화          read+write 추론          무엇이 먼저 터지나    전체 시스템 종합
```

### 계보와 시스템 인사이트를 표로 먼저 보고 싶다면

각 컴포넌트 장의 도입부에 같은 형식의 표가 있다.

| 컴포넌트 | 바로가기 | 이 장에서 보이는 병목 이동 |
|---|---|---|
| **Attention** | [01 문서 지도](01-attention.md#문서-지도) | KV capacity → read bandwidth → sparse gather |
| **Linear Attention** | [02 문서 지도](02-linear-attention.md#문서-지도) | KV traffic → state-update kernel·memory accuracy |
| **Test-Time Memory** | [03 RNN·update 계보](03-test-time-memory.md#31-두-계보가-만나는-지점--rnn에서-test-time-memory까지) | fixed state → gradient update → multi-rate update |
| **Position** | [04 시스템 영향표](04-position.md#이-장의-발전-계보와-시스템-영향) | 위치 표현 → context extrapolation → attention/KV layout 제약 |
| **MoE** | [05 시스템 영향표](05-moe.md#이-장의-발전-계보와-시스템-영향) | dense compute → weight capacity → all-to-all communication |
| **Norm & Residual** | [06 시스템 영향표](06-norm-residual.md#이-장의-발전-계보와-시스템-영향) | gradient stability → low precision·depth → activation traffic |
| **Shape** | [07 시스템 영향표](07-shape.md#이-장의-발전-계보와-시스템-영향) | parameter budget → serial depth·TP communication |
| **Decoding** | [08 시스템 영향표](08-decoding.md#이-장의-발전-계보와-시스템-영향) | serial weight read → speculative verification·acceptance rate → 검증 예산 스케줄링 |
| **Numerics** | [09 시스템 영향표](09-numerics.md#이-장의-발전-계보와-시스템-영향) | bit reduction → scale/outlier·kernel support |
| **Serving** | [10 시스템 영향표](10-serving.md#이-장의-발전-계보와-시스템-영향) | HBM capacity → fragmentation·scheduling → network transfer |

컴포넌트를 가로지르는 두 개의 종합표도 있다.

| 종합 관점 | 바로가기 | 읽을 인사이트 |
|---|---|---|
| **시대별 병목과 HW 변화** | [00-foundations 0.8](00-foundations.md#08-하드웨어-참고표) | memory bandwidth → communication → memory update·hierarchy로 병목의 단위가 칩→Pod→시간 계층으로 커지는 흐름 |
| **Efficient Transformer 전체 계보** | [99-landscape 99.0](99-landscape.md#990-efficient-transformer-발전-계보--시스템-영향) | throughput·memory traffic·KV cache·정확성 사이의 거래가 어떻게 이동했는지 |

이 표들을 순서대로 읽으면 모델 기법의 이름보다 먼저 **“무엇이 병목이었고, 무엇을
다른 비용과 교환했으며, 그 결과 시스템이 무엇을 새로 부담하게 되었는가”**가 보인다.

---

## 파일 지도

| 파일 | 다루는 압력 | 핵심 질문 |
|---|---|---|
| **[00-foundations](00-foundations.md)** | — | 기준점. 토큰 하나가 어떻게 흐르고 무엇이 비싼가 |
| **[01-attention](01-attention.md)** ★ | 메모리 | KV를 어떻게 줄이고, 필요한 토큰만 읽을 것인가 |
| **[02-linear-attention](02-linear-attention.md)** | 메모리 | KV를 아예 안 만들면? |
| **[03-test-time-memory](03-test-time-memory.md)** | 메모리 갱신 | ⚠️ **연구 단계** — 추론 중에 학습하는 메모리 (Titans·ATLAS·HOPE·Sleep) |
| **[04-position](04-position.md)** | 길이 일반화 | 순서를 어떻게 알려주고, 학습보다 긴 입력을 어떻게 다루나 |
| **[05-moe](05-moe.md)** | 연산량 · 통신 | 파라미터는 늘리고 연산은 그대로 |
| **[06-norm-residual](06-norm-residual.md)** | 안정성 | 깊은 모델을 어떻게 버티게 하나 |
| **[07-shape](07-shape.md)** | 직렬 지연 | 몇 층을 얼마나 넓게 |
| **[08-decoding](08-decoding.md)** | 직렬 지연 | 한 스텝에 토큰 하나여야 하나 |
| **[09-numerics](09-numerics.md)** | 전부 | 값 하나에 몇 비트를 쓸 것인가 |
| **[10-serving](10-serving.md)** ★ | 전부 | 실제로 돌릴 때 무엇이 먼저 터지나 |
| **[99-landscape](99-landscape.md)** ★ | — | 누가 무엇을 골랐고 시스템에 무엇을 요구하나 |

부속 자료

| | |
|---|---|
| [snippets/](snippets/README.md) | 실행 가능한 최소 구현과 등가성 검증 |

```
python snippets/test_equivalence.py
```

표준 라이브러리만 사용한다. GQA의 head 공유와 MLA의 projection 흡수처럼
수식만으로 놓치기 쉬운 등가성을 작은 텐서로 확인할 수 있다.

---

## 전체를 관통하는 것

`00-foundations` `0.4`에서 세우는 관점이다.

초기 Transformer의 큰 틀은 유지되지만, 내부 모듈과 실행 방식은 계속 바뀌어 왔다.
그 변화를 밀어낸 압력이 일곱 가지다.

| 압력 | 문제 | 대응 |
|---|---|---|
| **메모리** | 컨텍스트가 길어지면 KV cache를 감당할 수 없다 | 01, 02, 09 |
| **메모리 갱신** | 고정 상태는 무엇을 얼마나 오래 기억할지 스스로 바꾸기 어렵다 | 03 |
| **연산량** | 모델을 키우면 토큰당 연산이 그대로 늘어난다 | 05 |
| **직렬 지연** | 레이어는 순서대로 통과할 수밖에 없다 | 07, 08 |
| **통신** | 모델이 여러 GPU에 흩어져 있다 | 05, 10 |
| **안정성** | 깊고 큰 모델은 학습이 잘 터진다 | 06 |
| **길이 일반화** | 학습 때보다 긴 입력을 다뤄야 한다 | 04 |

그리고 매번 확인하게 되는 것 하나.

> **한 압력을 풀면 대개 다른 압력으로 옮겨간다.**
> MoE는 연산량을 풀지만 통신 부담을 만든다. 희소 attention은 읽는 양을 줄이지만
> 메모리 접근을 흩어놓는다.
>
> **공짜인 구조는 없다. 무엇을 무엇과 바꿨는가 — 이게 각 모듈에서 확인할 것이다.**

---

## 문서를 읽는 규칙

### 출처와 해석 표시

| 표시 | 뜻 |
|---|---|
| 📌 **[T1]** | 논문·공식 기술 리포트가 직접 뒷받침하는 내용 |
| 📌 **[T2]** | 공식 구현·프레임워크·하드웨어 문서에서 확인한 내용 |
| ✅ | 공식 설정·코드 또는 재현 가능한 예제로 확인한 내용 |
| ⚠️ | 구현·하드웨어·워크로드에 따라 달라지는 조건부 내용 |
| 💡 | 구조에서 도출한 직관이나 시스템 관점의 해석 |

각 파일 끝에는 핵심 논문, 공식 문서, 구현 자료를 구분해 적었다. 벤치마크 수치는
논문의 실험 조건 안에서만 읽고, 프레임워크 지원 상태는 사용 시점의 공식 문서를
다시 확인하는 것이 안전하다.

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
