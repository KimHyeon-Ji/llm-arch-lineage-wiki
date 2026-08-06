# 99. Landscape — 누가 무엇을 골랐고, 시스템에 무엇을 요구하나

앞의 아홉 파일은 축별로 "어떤 아이디어가 있는가"를 정리했다.
이 파일은 방향을 뒤집는다. **실제 모델들이 그 축들을 어떻게 조합했는가.**

`01-attention`에서 CSA를 읽었다면 "DeepSeek-V4가 쓴다"까지는 안다.
여기서는 반대로 묻는다. **DeepSeek-V4는 왜 그 조합을 골랐고, 그 조합은 시스템에
무엇을 요구하는가.**

> ⚠️ **이 파일은 미검증 항목이 가장 많다.** 최신 모델일수록 원문 대조가 덜 되어 있다.
> ⚠️ 표시된 항목은 2차 자료 기반이므로 그대로 인용하지 말 것.

**목차**

| | 절 |
|---|---|
| [99.1](#991-마스터-매트릭스) | 마스터 매트릭스 |
| [99.2](#992-계열별-진화-궤적) | 계열별 진화 궤적 |
| [99.3](#993-kv-축소-전략-일곱-가지와-조합-가능성) | KV 축소 전략과 조합 가능성 |
| [99.4](#994-moe-구성-비교) | MoE 구성 비교 |
| [99.5](#995-하이브리드-구성-비교) | 하이브리드 구성 비교 |
| [99.6](#996-그래서-시스템은-무엇을-요구받는가) | **시스템은 무엇을 요구받는가** ★ |

---

## 99.1 마스터 매트릭스

| 모델 | attention | 위치 | MoE | residual/norm | decoding | numerics |
|---|---|---|---|---|---|---|
| **Llama 3** | GQA (64/8) | RoPE | dense | Pre-LN, RMSNorm | — | BF16 |
| **Mistral Small** | GQA + SWA | RoPE | dense/MoE | RMSNorm | — | BF16 |
| **Gemma 3** | GQA + **SWA 5:1** | RoPE | dense | RMSNorm, QK-Norm | — | BF16 |
| **Gemma 4 (E2B/E4B)** | GQA + **KV sharing** | RoPE | dense | RMSNorm | — | 온디바이스 |
| **gpt-oss** | GQA + SWA, **학습된 sink** | RoPE | MoE | RMSNorm | — | **MXFP4** |
| **DeepSeek-V3** | **MLA** | **decoupled RoPE** + YaRN | 256/8 + shared, **aux-loss-free** | RMSNorm | **MTP-1** | **FP8 학습** |
| **DeepSeek-V3.2** | MLA + **DSA** (top-k 2048) | decoupled RoPE | 위와 동일 | RMSNorm | MTP | FP8 |
| **DeepSeek-V4** ⚠️ | **CSA + HCA 교차** | ⚠️ | MoE | **mHC** | MTP | ⚠️ |
| **Qwen3** | GQA | RoPE + YaRN | 235B-A22B | QK-Norm | — | BF16 |
| **Qwen3-Next** | **Gated DeltaNet 3:1 + Gated Attention** | **partial RoPE** | shared expert, expert 4배 | **zero-centered RMSNorm** | — | BF16 |
| **Qwen3.5** | Gated DeltaNet + Gated Attention | partial RoPE | fine-grained | QK-Norm | — | — |
| **Kimi K2 / K2.5** | **MLA** | RoPE | DeepSeek식 | RMSNorm | — | — |
| **Kimi Linear** | **KDA 3:1** + ShortConv | **NoPE** (full 층) | — | Attention Residuals ⚠️ | — | — |
| **Kimi K3** ⚠️ | **KDA 3:1** | NoPE ⚠️ | **Stable LatentMoE** (896/16) ⚠️ | ⚠️ | — | **MXFP4** ⚠️ |
| **GLM-4.5 / 4.7** | GQA | RoPE | 160 experts | QK-Norm | MTP | — |
| **GLM-5** ⚠️ | **MLA + DSA** | RoPE | **256 experts** | QK-Norm | MTP | — |
| **MiniMax M2 / M2.5** | GQA | RoPE | MoE | QK-Norm | MTP | — |
| **Ling 2.5** | **Lightning Attention** hybrid + MLA | RoPE | fine-grained MoE | — | — | — |
| **Nemotron 3** | **Mamba-2 + attention** hybrid | — | MoE / LatentMoE ⚠️ | — | — | — |
| **Arcee Trinity Large** | SWA + **Gated Attention**, global 층 **NoPE** | partial | **coarse MoE** (의도적) | **depth-scaled gain** | — | — |
| **Tiny Aya** | SWA, **QK-Norm 제거** | NoPE | dense | **parallel block** | — | — |

읽는 법 두 가지.

**세로로 읽으면** 각 축의 채택 현황이 보인다. GQA는 거의 전부, MLA는 긴 컨텍스트를
노리는 대형 모델, linear 하이브리드는 Qwen·Moonshot 계열에 몰려 있다.

**가로로 읽으면** 각 모델의 설계 철학이 보인다. 다음 절이 그 이야기다.

---

## 99.2 계열별 진화 궤적

### DeepSeek — KV 압축을 끝까지 밀어붙인다

```
 V2 (2024)   MLA 도입 — 저차원 압축 + 흡수
    ↓
 V3 (2024)   MLA 실전화, aux-loss-free, MTP, FP8 학습
    ↓
 V3.2 (2025) + DSA — 압축에 희소 선택을 결합
    ↓
 V4 (2026)   CSA/HCA — 압축과 선택을 직렬로 겹침, + mHC ⚠️
```

**일관된 방향이 있다. "KV를 줄인다"를 한 번도 놓지 않았다.**
NSA(연구) → MLA(압축) → DSA(선택) → CSA(둘의 결합)로 계속 쌓아올렸다.

동시에 **학습·시스템 쪽도 함께 밀었다.** FP8 학습, aux-loss-free 균형, DeepEP,
그리고 V4에서는 residual 자체(mHC)까지 건드렸다.
**아키텍처와 시스템을 한 팀이 함께 설계한다는 인상**을 주는 유일한 계열이다.

### Qwen — 하이브리드와 게이팅

```
 Qwen3 (2025)       GQA + QK-Norm, MoE 확대
    ↓
 Qwen3-Next (2025)  Gated DeltaNet 3:1 + Gated Attention
                    partial RoPE, zero-centered RMSNorm, expert 4배
    ↓
 Qwen3.5 (2026)     같은 방향 심화
```

**DeepSeek와 정반대 선택이다.** KV를 압축하는 대신 **KV를 안 만드는 층을 늘렸다.**
그리고 Gated Attention(NeurIPS 2025 Best Paper)을 자기들이 만들어 자기 모델에 넣었다.

정규화 쪽 미세 조정(zero-centered RMSNorm)도 이 계열의 특징이다.
**안정성에 신경을 많이 쓰는 팀**으로 보인다.

### Moonshot (Kimi) — 선형 attention에 올인

```
 K2 (2025)         MLA — DeepSeek식을 따라감
    ↓
 MoBA (2025)       블록 라우팅 연구
    ↓
 Kimi Linear (2025) KDA 3:1 + ShortConv + NoPE(full 층)
    ↓
 K3 (2026)         KDA를 플래그십에 적용, Stable LatentMoE, MXFP4 ⚠️
```

**계열이 바뀐 사례다.** K2까지는 MLA(축1)를 따라가다가, Kimi Linear에서
축2로 전환하고 K3에서 그것을 플래그십에 올렸다.

MoBA(축1의 희소)도 이들이 만들었다는 점이 흥미롭다. **두 갈래를 다 시도해보고
축2를 골랐다**고 볼 수 있다.

### Zhipu (GLM) — 실용적 통합

```
 GLM-4.5 (2025)  GQA + MoE 160 experts + QK-Norm
    ↓
 GLM-4.7         MTP 추가
    ↓
 GLM-5 (2026)    MLA + DSA 채택, expert 256으로, 층수 92 → 78 ⚠️
```

**독자 기법보다 검증된 것을 빠르게 통합하는 전략**으로 보인다.
MLA도 DSA도 DeepSeek가 만든 것이고, GLM-5는 그것을 가져다 썼다.

대신 **자기만의 판단을 시스템 쪽에서 했다.** 층수를 92에서 78로 줄인 것이
그 예다 (`06-shape` 6.1). 품질 대신 추론 지연을 택한 것이다.

### Google (Gemma) — 온디바이스와 SWA

```
 Gemma 2/3   SWA를 local:global 5:1로 — 하이브리드 배치의 선구
    ↓
 Gemma 4     E2B/E4B에서 PLE + cross-layer KV sharing
```

**축1 안에서 하이브리드**를 한 계열이다. linear attention 대신 SWA를 쓰고
일부 층만 full로 남겼다.

그리고 **온디바이스를 정면으로 겨냥한 유일한 계열**이다. PLE로 파라미터를
가속기 밖에 두고, 레이어 간 KV 공유로 캐시를 줄인다. 데이터센터 모델들과
전혀 다른 제약에서 출발한 설계다.

### 그 외

| 계열 | 특징 |
|---|---|
| **Meta (Llama)** | GQA를 대중화. 이후 아키텍처 혁신보다 규모·데이터 쪽 |
| **Mistral** | SWA 대중화. 이후 MLA 채택 ⚠️ |
| **NVIDIA (Nemotron)** | **Mamba-2 + attention 하이브리드** — 축2를 SSM 쪽으로 |
| **inclusionAI (Ling)** | Lightning Attention 하이브리드 + MLA |
| **Arcee (Trinity)** | **의도적 coarse MoE** — 추론 처리량 우선. 학계보다 서빙 관점 |
| **OpenAI (gpt-oss)** | SWA + 학습된 sink + MXFP4 — 배포 효율 중심 |

---

## 99.3 KV 축소 전략 일곱 가지와 조합 가능성

`01`, `02`, `08`에 흩어져 있던 것을 한자리에 모으면 이렇다.

| # | 전략 | 방법 | 줄이는 것 | 대표 |
|---|---|---|---|---|
| ① | **헤드 공유** | MQA, GQA | 저장량 `1/g` | Llama 3, Qwen3 |
| ② | **차원 압축** | MLA | 저장량 (GQA 2.25그룹 상당) | DeepSeek, Kimi K2, GLM-5 |
| ③ | **읽기 제한 (고정)** | SWA | 저장량 상한 `W` | Gemma 3, Mistral |
| ④ | **읽기 제한 (학습)** | NSA, MoBA, DSA | **읽는 양만** | DeepSeek-V3.2, GLM-5 |
| ⑤ | **토큰 압축** | CSA, HCA | 저장량 `1/m` | DeepSeek-V4 ⚠️ |
| ⑥ | **레이어 공유** | CLA, YOCO | 저장량 `1/그룹` | Gemma 4 E2B/E4B |
| ⑦ | **정밀도** | FP8/FP4 KV | 저장량 `1/2`, `1/4` | 광범위 |
| — | **대체** | linear attention | **KV 자체를 없앰** | Qwen3-Next, Kimi K3 |

### 조합 가능성

이게 이 절의 핵심이다. **어떤 것끼리 곱해지고 어떤 것끼리 배타적인가.**

|  | ① 헤드 | ② 차원 | ③ SWA | ④ 희소 | ⑤ 토큰압축 | ⑥ 레이어 | ⑦ 비트 |
|---|---|---|---|---|---|---|---|
| **① 헤드 공유** | — | ✗ 배타 | ✓ | ✓ | ✓ | ✓ | ✓ |
| **② 차원 압축** | ✗ | — | △ | **✓ (DSA)** | **✓ (CSA)** | ✓ | ✓ |
| **③ SWA** | ✓ | △ | — | △ | △ | ✓ | ✓ |
| **④ 희소 선택** | ✓ | **✓** | △ | — | **✓** | ✓ | ✓ |
| **⑤ 토큰 압축** | ✓ | ✓ | △ | ✓ | — | ✓ | ✓ |
| **⑥ 레이어 공유** | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| **⑦ 정밀도** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

- **①과 ②는 배타적이다.** 둘 다 "헤드/차원을 어떻게 저장할지"를 정하는 것이라
  하나만 고른다.
- **⑥ 레이어 공유와 ⑦ 정밀도는 모든 것과 곱해진다.** 방향이 직교하기 때문이다.
- **③ SWA는 층 단위로 섞는 방식**이라 다른 것과 "조합"이라기보다 "층을 나눠 갖는" 관계다.

### 그런데 실제로는 다 안 쓴다

이론상 `GQA × CLA × FP8 KV × SWA`가 가능하지만 그렇게 하는 모델은 없다.
이유가 있다.

| 왜 | 설명 |
|---|---|
| **품질이 곱으로 깎인다** | 각각 조금씩 잃는 것이 누적된다 |
| **구현 복잡도** | 커널 조합이 폭발한다 |
| **수확 체감** | KV가 이미 작으면 더 줄여도 병목이 다른 데로 옮겨간다 |

마지막이 중요하다. `09-serving` `9.7`에서 봤듯 **KV를 충분히 줄이면 병목이
가중치 읽기나 통신으로 옮겨간다.** 그 지점을 넘으면 KV를 더 줄여도 안 빨라진다.

**⑥ 레이어 공유가 대형 모델에서 안 쓰이는 이유**도 여기 있을 것으로 보인다.
MLA로 이미 충분히 줄였는데 품질 손실을 더 감수할 이유가 없다.
반대로 온디바이스(Gemma 4 E2B/E4B)는 용량이 절대적 제약이라 쓴다.

---

## 99.4 MoE 구성 비교

| 모델 | 총 / 활성 | `E` | `k` | shared | granularity | 균형 |
|---|---|---|---|---|---|---|
| Mixtral 8×7B | 47B / 13B | 8 | 2 | ✗ | 굵음 | aux loss |
| **DeepSeek-V3** | **671B / 37B** | **256** | **8** | ✓ | **잘게** | **aux-loss-free** |
| Qwen3-235B | 235B / 22B | 다수 | — | ✓ | 잘게 | — |
| Qwen3-Next | — | **4배 증가** | — | ✓ | 잘게 | — |
| GLM-4.7 → GLM-5 | — | **160 → 256** | — | ✓ | 잘게 | — |
| Kimi K3 ⚠️ | 2.8T ⚠️ | **896** ⚠️ | **16** ⚠️ | — | 매우 잘게 | LatentMoE ⚠️ |
| DeepSeek-V4-Pro ⚠️ | **1.6T / 49B** ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| DeepSeek-V4-Flash ⚠️ | 284B / 13B ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| **Arcee Trinity Large** | — | 적음 | — | — | **의도적으로 굵게** | — |

### 두 가지 추세

**① expert 개수가 계속 늘어난다.** 8 → 64 → 256 → 896.
`04-moe` `4.3`에서 본 대로 조합의 수가 표현력을 만들기 때문이다.

**② 총/활성 비율이 커진다.** Mixtral은 3.6배, DeepSeek-V3는 18배,
V4-Pro는 32배 ⚠️. **파라미터는 늘리되 연산은 안 늘린다**는 방향이 강화되고 있다.

### 그리고 반대 방향 하나

📌 **Arcee Trinity Large만 굵게 갔다.** 추론 처리량을 위해서다.

이 하나의 예외가 `04-moe` `4.4`의 논점을 증명한다.
**fine-grained가 항상 옳은 게 아니라, 대규모 배치 서빙을 전제할 때 옳다.**
그 전제가 다르면 결론도 달라진다.

---

## 99.5 하이브리드 구성 비교

"모든 층을 같게 만들지 않는다"는 것이 최근 공통 패턴이다.
**무엇과 무엇을 섞는지**가 갈린다.

| 모델 | 값싼 층 | 비싼 층 | 비율 | 비싼 층이 하는 일 |
|---|---|---|---|---|
| **Gemma 3** | SWA | full attention | **5:1** | 전역 문맥 |
| **Qwen3-Next / 3.5** | Gated DeltaNet | Gated Attention | **3:1** | 정확한 검색 |
| **Kimi Linear / K3** | KDA | full attention (**NoPE**) | **3:1** | 정확한 검색 |
| **Nemotron 3** | Mamba-2 | attention | ⚠️ | 전역 검색 |
| **Ling 2.5** | Lightning Attention | full | ⚠️ | 전역 검색 |
| **DeepSeek-V4** ⚠️ | CSA (정밀·좁게) | HCA (거칠게·넓게) | ⚠️ | **역할 분담이 다르다** |
| **Arcee Trinity** | SWA | global (**NoPE**) | ⚠️ | 전역 문맥 |

### 두 종류의 하이브리드

대부분은 **"값싸게 지역 + 비싸게 전역"** 구조다. 그런데 DeepSeek-V4는 다르다.

```
 일반적 하이브리드:  [좁게 보는 층] × N  +  [전부 보는 층] × 1
 DeepSeek-V4:        [정밀하게 일부] ↔ [거칠게 전부]  교대
```

V4에는 "전부를 원본 그대로 보는 층"이 없다 ⚠️. 정밀도와 범위를 서로 다르게
가진 두 종류가 교대한다. **더 나아간 형태**로 볼 수 있지만, 원문 확인 전이라
단정하지 않는다.

### 공통 설계 질문

| 질문 | 관찰된 답 |
|---|---|
| 비율은? | **3:1 또는 5:1**이 지배적 |
| 비싼 층 위치는? | 균등 배치가 보통 |
| 비싼 층의 위치 인코딩은? | **NoPE가 늘고 있다** (Kimi Linear, Trinity) |
| 비싼 층의 attention 종류는? | 자유 — GQA, MLA, Gated 무엇이든 |

세 번째가 흥미롭다. 값싼 층이 지역 위치를 처리하니 비싼 층은 RoPE의
장기 감쇠에 방해받지 않는 편이 낫다는 판단으로 보인다 (`03-position` 3.5).

---

## 99.6 그래서 시스템은 무엇을 요구받는가

이 위키 전체의 결론이다.

### 각 선택이 요구하는 것

| 아키텍처 선택 | **요구하는 것** | **덜어주는 것** |
|---|---|---|
| MLA (차원 압축) | 흡수 전용 커널, latent GEMM 효율, **연산량 증가** | HBM 용량, KV 읽기 대역폭 |
| 희소 선택 (NSA/DSA/CSA) | **비정형 gather**, top-k 정렬, 낮은 접근 지역성 | attention 연산, KV 읽기량 |
| 고정 상태 (KDA/Mamba) | 상태 갱신의 순차성, 전용 chunked 커널 | **KV 용량이 소멸** |
| 레이어 공유 (CLA/YOCO) | 레이어 간 의존 → **PP 제약** | HBM 용량 (세로 방향) |
| **MoE (fine-grained)** | **all-to-all 대역폭**, EP 스케일, 부하 균형 | 활성 연산량 |
| MoE (coarse) | 노드당 메모리 용량 | all-to-all 트래픽 |
| **하이브리드 (이질 층)** | **층별로 다른 자원 프로파일의 스케줄링** | 평균 비용 |
| 깊이 ↓ 너비 ↑ | TP 통신 대역폭 | **직렬 지연** |
| MTP / speculative | draft 검증용 여유 연산, 지연 변동 흡수 | 메모리 대역폭 (실효 배치↑) |
| FP8 / FP4 | 해당 포맷 tensor core, 양자화 오버헤드 | **용량·대역폭·연산 동시** |
| 1M 컨텍스트 | 위 전부 + PD 분리 + CP + KV 전송 | — |

### 결론 — 압력은 사라지지 않고 옮겨간다

`00-foundations` `0.4`에서 세운 관점으로 전체를 보면 이렇게 정리된다.

> **2026년의 프론티어 모델들은 메모리 용량 압력을 상당 부분 해소했다.**
> MLA·희소·고정 상태·양자화가 겹치면서, KV cache는 더 이상 유일한 벽이 아니다.
>
> **그런데 그 압력은 사라진 게 아니라 세 곳으로 옮겨갔다.**
>
> | 어디로 | 무엇 때문에 |
> |---|---|
> | **① 노드 간 통신** | MoE의 all-to-all. expert가 잘아질수록, 모델이 커질수록 심해진다 |
> | **② 비정형 메모리 접근** | 희소 선택의 gather. 읽는 양은 줄었는데 실효 대역폭이 안 나온다 |
> | **③ 이질적 스케줄링** | linear/full, CSA/HCA, dense/sparse가 층마다 섞여 균일성이 깨졌다 |
>
> **따라서 차세대 서빙 시스템의 병목은 HBM 용량이 아니라
> 노드 간 대역폭, gather 효율, 그리고 이질적 레이어의 스케줄링이다.**

### 이 결론의 근거와 한계

**근거로 삼은 것**

- `04-moe` 4.6 — MoE 모델에서 레이어당 all-to-all 두 번, 61층이면 120회 이상
- `01-attention` 1.7 — NSA가 블록 단위 선택을 택한 이유가 gather 효율
- `02-linear-attention` 2.5, `01` 1.9 — 층별 프로파일 불균형
- 📎 [T3] NVLink 6가 per-GPU 3.6 TB/s로 두 배가 되었고, 자료들이 이를 MoE
  all-to-all과 연결짓는다 (`CONTESTED.md` C4)

**한계**

- ⚠️ 마지막 항목은 공식 스펙 대조 전이다. **하드웨어 근거로 삼기에는 약하다.**
- ⚠️ DeepSeek-V4의 세부(CSA/HCA 비율, 압축률)를 모르므로 ③의 강도를 정확히 말할 수 없다.
- 실측 프로파일링 없이 구조에서 추론한 결론이다. **가설로 읽어야 한다.**

이 결론을 확정하려면 실제 워크로드에서 통신 시간·gather 효율·층별 실행 시간을
측정해야 한다. 그건 이 위키의 범위 밖이고, 다음 단계의 일이다.

---

## 남은 질문들

정리하면서 답을 못 낸 것들이다.

| 질문 | 왜 답을 못 냈나 |
|---|---|
| CSA/HCA의 정확한 구조와 비율 | V4 원문 미대조 (C1, C3) |
| 축1(압축)과 축2(고정 상태) 중 무엇이 이길까 | 둘 다 살아 있고 판단할 근거가 부족 |
| 레이어 공유(CLA/YOCO)가 대형 모델에 안 오는 이유 | 품질 문제인지 수확 체감인지 불명 |
| 하이브리드 모델의 prefix caching | 프레임워크 지원 상황 확인 필요 |
| LatentMoE의 실체 | 원문 미대조 |
| 3:1 비율의 근거 | 왜 하필 3:1인지 이론적 설명을 못 찾음 |

---

## Sources

이 파일은 앞의 아홉 파일에서 정리한 내용을 종합한 것이다.
개별 출처는 각 파일의 Sources를 참조.

**추가로 참조한 것**

**T1**
- 각 모델의 HuggingFace `config.json` — `99.1` 매트릭스의 수치
- DeepSeek-V4 (arXiv:2606.19348) ⚠️ **원문 미대조**
- Kimi K3 기술 리포트 ⚠️ **원문 미대조**
- GLM-5 기술 리포트 ⚠️ **원문 미대조**

**T3**
- Sebastian Raschka, *LLM Architecture Gallery* 및 분기별 아키텍처 리뷰
  — `99.1` 매트릭스의 채택 현황, Arcee Trinity의 coarse MoE, Tiny Aya의 선택들
- 각 모델 릴리스에 대한 2차 분석 자료

**미검증 항목 (이 파일 전반)**
- ⚠️ 표시된 모든 셀
- `99.6`의 결론 — 구조에서 추론한 **가설**이며 실측 근거가 없다
- 계열별 서술의 "의도" 해석 — 대부분 결과물에서 역추론한 것
