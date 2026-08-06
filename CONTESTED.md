# CONTESTED — 소스가 서로 다르게 말하는 것들

자료마다 내용이 다르거나, 아직 원문으로 확인하지 못한 항목을 모아둔다.

**여기 있는 항목을 본문에서 쓸 때는 반드시 `⚠️ 미검증` 표시를 함께 단다.**
확인되면 본문을 고치고 이 표의 상태를 `해소`로 바꾼다. 항목을 지우지는 않는다 —
"한때 헷갈렸던 지점"이라는 기록 자체가 나중에 쓸모가 있다.

| # | 상태 | 쟁점 |
|---|---|---|
| [C1](#c1) | 미해소 | HCA의 정식 명칭 |
| [C2](#c2) | 미해소 | HoPE 동명이인 3종 |
| [C3](#c3) | 미해소 | CSA의 압축률 `m` |
| [C4](#c4) | 미해소 | B200·Rubin 하드웨어 스펙 |
| [C5](#c5) | 미해소 | NSA → DSA → CSA 계승 관계 |

---

## C1

**HCA는 무엇의 약자인가**

| 주장 | 소스 |
|---|---|
| **H**eavily **C**ompressed **A**ttention | DeepSeek-V4 관련 기술 기사 다수 |
| **H**yper-**C**onnected **A**ttention | Sebastian Raschka, LLM Architecture Gallery (T3) |

DeepSeek-V4에는 attention 쪽의 CSA/HCA와, residual stream 쪽의 **mHC
(Manifold-Constrained Hyper-Connections)** 가 **둘 다** 들어간다.
후자의 "Hyper-Connections"가 HCA로 옮겨붙은 혼동으로 보인다.

기능 설명("훨씬 높은 압축률로 압축한 뒤 그 위에서 dense attention")을 보면
Heavily Compressed 쪽이 앞뒤가 맞는다. 다만 **원문 확인 전까지는 단정하지 않는다.**

- **해소 방법**: arXiv:2606.19348 원문에서 약어 정의 확인
- **영향 범위**: `01-attention.md` (CSA·HCA 절), `99-landscape.md`

> 이 항목이 T3 교차검증 규칙을 만든 계기다. T3는 계보 지도를 그리는 데는 탁월하지만
> 세부 명칭에서 이런 오염이 생긴다.

---

## C2

**HoPE라는 이름의 논문이 최소 3개다**

| arXiv | 제목 | 분야 |
|---|---|---|
| 2410.21216 | *HoPE: A Novel Positional Encoding Without Long-Term Decay…* | LLM 위치 인코딩 |
| 2505.20444 | *HoPE: Hybrid of Position Embedding for Long Context VLM* | 비전-언어 모델 |
| 2509.05218 | *HoPE: Hyperbolic Rotary Positional Encoding* | 쌍곡 공간 RoPE |

서로 다른 연구다. 이름만 같다.

- **처리 방침**: `03-position.md`에서 다루는 HoPE는 **2410.21216**으로 확정한다.
  해당 절 서두에 3종을 모두 병기해 혼동을 막는다.
- **상태**: 방침은 정해졌으나 각 논문 원문 대조는 미완

---

## C3

**CSA는 몇 개 토큰을 하나로 묶는가**

2차 자료들이 `m ≈ 4`라고 적고 있으나 원문 확인 전이다.
top-k 선택 개수 `t`, CSA와 HCA의 레이어 배치 비율도 마찬가지다.

- **해소 방법**: arXiv:2606.19348 원문 확인
- **처리 방침**: 확인 전까지 본문에서는 **기호로만 쓰고 구체값을 단정하지 않는다.**
  ("`m`개 토큰을 하나로 묶는다. 2차 자료는 `m≈4`로 전하나 미확인이다.")
- **영향 범위**: `01-attention.md`, `99-landscape.md`

---

## C4

**B200·Rubin의 대역폭과 연산 성능**

`00-foundations.md` 0.7 스펙표의 아래 두 줄이 미확인이다.

| 항목 | 본 값 | 문제 |
|---|---|---|
| B200 BF16 dense | ~2.2 PFLOPS | 벤더 표기가 sparsity 2배를 포함하는 경우가 많아 dense 환산이 불확실 |
| R100 HBM4 대역폭 | ~20 TB/s | 2차 기사들이 "22 TB/s per GPU"라고 하나, 스택당 대역폭으로 환산하면 과대해 보임 |
| R100 메모리 | 288 GB | 미확인 |

- **해소 방법**: NVIDIA 공식 데이터시트·아키텍처 백서
- **처리 방침**: 표에 ⚠️ 유지. 이 값들에 기대는 결론은 쓰지 않는다.

---

## C5

**NSA → DSA → CSA는 정말 한 줄기인가**

구조적으로는 명백해 보인다. 셋 다 "압축 + 선택"의 조합이고,
NSA(2502.11089)가 가장 먼저 계층적 희소 구조를 제시했다.

다만 **후속 논문들이 명시적으로 NSA를 계승했다고 선언했는지**는 확인하지 못했다.
계보 그림에서 화살표를 실선으로 그릴지 점선으로 그릴지가 여기에 달려 있다.

- **해소 방법**: DeepSeek-V3.2, V4 논문의 related work 대조
- **처리 방침**: 확인 전까지 계보 그림에서 **점선**으로 표시하고
  "구조적 유사성 기준"이라고 명시한다.
- **영향 범위**: `01-attention.md` 계보 지도

---

## 해소된 항목

아직 없다.
