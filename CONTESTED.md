# CONTESTED — 소스가 서로 다르게 말하는 것들

자료마다 내용이 다르거나, 아직 원문으로 확인하지 못한 항목을 모아둔다.

**여기 있는 항목을 본문에서 쓸 때는 반드시 `⚠️ 미검증` 표시를 함께 단다.**
확인되면 본문을 고치고 상태를 `해소`로 바꾼다. 항목을 지우지는 않는다 —
"한때 헷갈렸던 지점"이라는 기록 자체가 나중에 쓸모가 있다.

| # | 상태 | 쟁점 |
|---|---|---|
| [C1](#c1) | ✅ **해소** | HCA의 정식 명칭 |
| [C2](#c2) | 🟡 부분 해소 | HoPE 동명이인 3종 |
| [C3](#c3) | ✅ **해소** | CSA의 압축률 `m` |
| [C4](#c4) | ✅ **해소** | B200·Rubin 하드웨어 스펙 |
| [C5](#c5) | 🟡 부분 해소 | NSA → DSA → CSA 계승 관계 |
| [C6](#c6) | 🟡 출처 정정 | 인접 레이어 KV 유사도 0.72~0.87 |

---

## C1

**HCA는 무엇의 약자인가** — ✅ **해소 (2026-08-06)**

| 주장 | 소스 | 판정 |
|---|---|---|
| **H**eavily **C**ompressed **A**ttention | DeepSeek-V4 논문 본문 | ✅ **정답** |
| **H**yper-**C**onnected **A**ttention | Sebastian Raschka, LLM Architecture Gallery (T3) | ❌ 오류 |

arXiv:2606.19348 원문에서 **"Heavily Compressed Attention"** 으로 확인했다.

혼동의 원인도 확인됐다. DeepSeek-V4에는 attention 쪽의 CSA/HCA와,
residual stream 쪽의 **mHC(Manifold-Constrained Hyper-Connections)** 가 **둘 다** 들어간다.
후자의 "Hyper-Connections"가 HCA로 옮겨붙은 것이다.

`config.json`이 결정적이었다. `hc_mult`, `hc_sinkhorn_iters`, `hc_eps` 필드가
**mHC 전용**이고, CSA/HCA 쪽은 `compress_ratios`로 완전히 별개로 관리된다.
**같은 모델의 서로 다른 두 기여다.**

- **반영**: `01-attention.md` 1.9, `05-norm-residual.md` 5.6
- **교훈**: 이 항목이 T3 교차검증 규칙을 만든 계기였다. T3는 계보 지도를 그리는 데는
  탁월하지만 세부 명칭에서 이런 오염이 생긴다. **규칙이 실제로 값을 했다.**

---

## C2

**HoPE라는 이름의 논문이 최소 3개다** — 🟡 **부분 해소 (2026-08-07)**

✅ **2410.21216의 메커니즘은 원문에서 확인했다.** RoPE의 특정 성분을
위치 무관 성분으로 교체하고 **고주파 신호만 남긴다**고 명시하며,
**"장기 감쇠는 시대에 뒤떨어졌다"** 고 직접 주장한다.
`03-position.md` 3.6의 서술이 원문과 일치한다.

⚠️ 다만 **실험 수치는 초록에 없어 확인하지 못했고**, 나머지 두 논문(2505.20444,
2509.05218)은 제목 수준에서만 구분했다.

| arXiv | 제목 | 분야 |
|---|---|---|
| 2410.21216 | *HoPE: A Novel Positional Encoding Without Long-Term Decay…* | LLM 위치 인코딩 |
| 2505.20444 | *HoPE: Hybrid of Position Embedding for Long Context VLM* | 비전-언어 모델 |
| 2509.05218 | *HoPE: Hyperbolic Rotary Positional Encoding* | 쌍곡 공간 RoPE |

서로 다른 연구다. 이름만 같다.

- **처리 방침**: `03-position.md`에서 다루는 HoPE는 **2410.21216**으로 확정.
  해당 절 서두에 3종을 병기해 혼동을 막는다.
- **상태**: 방침은 정해졌으나 각 논문 원문 대조는 미완

---

## C3

**CSA는 몇 개 토큰을 하나로 묶는가** — ✅ **해소 (2026-08-06)**

| 항목 | 값 | 출처 |
|---|---|---|
| CSA 압축률 `m` | **4** | V4 논문 (V4-Flash 설정) |
| HCA 압축률 `m'` | **128** | V4 논문 |
| top-k (V4-Pro) | **1024** | `config.json` `index_topk` |
| top-k (V4-Flash) | **512** | V4 논문 |
| indexer 헤드 / 차원 | **64 / 128** | 논문 + `config.json` |
| 레이어 배치 | **CSA:HCA = 1:1 교대** | `config.json` `compress_ratios` |

레이어 배치는 추측할 필요가 없었다. `config.json`의 `compress_ratios`가
**레이어별 압축률을 그대로 나열**한다.

```
V4-Pro (61층): [128, 128, 4, 128, 4, 128, ..., 4, 128, 0]
                └─2개─┘  └───── 4 ↔ 128 교대 ─────┘  └끝┘
```

추가로 확인된 것

- 압축은 **학습된다** (`W^aKV`, `W^bKV`, `W^aZ`, `W^bZ` + 학습되는 위치 bias)
- **겹치는 두 계열**(`C^a`, `C^b`)을 써서 압축 출력 하나에 `2m`개 토큰이 반영된다
- 첫 두 층은 **순수 sliding window** (V4-Flash 기준, `sliding_window`=128)

- **반영**: `01-attention.md` 1.9, `99-landscape.md`
- **잔여**: 마지막 층의 `compress_ratio`=0 이 정확히 무엇을 뜻하는지 미확인

---

## C4

**B200·Rubin의 대역폭과 연산 성능** — ✅ **해소 (2026-08-07)**

| 항목 | 확정 값 |
|---|---|
| B200 메모리 / 대역폭 | **192 GB HBM3e / 8.0 TB/s** |
| B200 BF16 dense | **2.25 PFLOPS** (sparsity 포함 표기는 4.5 PF) |
| B200 FP8 / FP4 dense | 4.5 PF / **9 PF** |
| B200 균형점 | **~281 FLOP/byte** |
| R100 메모리 / 대역폭 | **288 GB HBM4 (8 스택) / 22 TB/s** |
| R100 NVFP4 | 추론 **50 PFLOPS**, 학습 35 PFLOPS |

**내가 의심했던 것이 틀렸다.** "22 TB/s per GPU"를 과대라고 봤는데,
**8개 HBM4 스택 합산**이라 스택당 ~2.75 TB/s다. 무리한 수치가 아니다.

- **잔여**: 여러 자료가 일치하지만 **공식 데이터시트 PDF를 직접 대조하지는 않았다.**
  R100의 BF16 dense 수치는 확인하지 못했다.
- **반영**: `00-foundations.md` 0.8

---

## C6

**인접 레이어 KV 코사인 유사도 0.72~0.87** — 🟡 **출처 정정 (2026-08-07)**

`01-attention.md` 1.10에서 CLA/YOCO의 근거로 인용한 수치다.
확인해보니 **출처가 CLA 원 논문이 아니었다.**

| | |
|---|---|
| 실제 출처 | xKV(2503.18893), CommonKV(2508.16134), cross-layer KV 공유 체계연구(2410.14442) 등 **여러 후속 연구의 측정값을 묶은 범위** |
| 문제 | 각 논문의 측정 모델·레이어·조건이 다르다. **단일 수치처럼 인용하면 안 된다** |

- **처리 방침**: 수치는 남기되 **"여러 연구에서 관측된 범위"** 로 명시하고,
  방향성(인접 레이어가 상당히 비슷하다)만 근거로 쓴다.
- **반영**: `01-attention.md` 1.10
- **교훈**: 널리 인용되는 숫자일수록 **원 출처를 거슬러 올라가야 한다.**
  이 값은 여러 2차 자료에 그대로 복사되어 있었다.

---

## C5

**NSA → DSA → CSA는 정말 한 줄기인가** — 🟡 **부분 해소 (2026-08-06)**

| 연결 | 상태 | 근거 |
|---|---|---|
| **DSA → CSA** | ✅ **확정 (실선)** | V4 논문이 명시적으로 인용 — "CSA는 압축 엔트리 위에 DSA를 적용한다" |
| **NSA → DSA/CSA** | 🟡 **미확정 (점선 유지)** | 확인한 범위에서 V4 논문에 NSA 언급이 없다 |

DSA 쪽은 인용을 넘어 **구현 자체를 재사용**한다. CSA의 선택 단계가
DSA의 lightning indexer 그대로다.

NSA 쪽은 구조적 유사성이 명백하다 — 둘 다 "압축 + 선택"의 조합이고
NSA가 먼저 나왔다. 하지만 계승을 선언한 문장을 확인하지 못했으므로
**계보 지도에서 점선으로 남긴다.**

- **반영**: `01-attention.md` 계보 지도, 1.9 "계보상 무엇이 새로운가"
- **잔여**: V3.2 논문의 related work를 직접 확인하면 NSA 연결이 확정될 수 있다

---

## 해소 기록

| 날짜 | 항목 | 방법 |
|---|---|---|
| 2026-08-06 | C1, C3, C5(부분) | DeepSeek-V4 논문 HTML + HuggingFace `config.json` + vLLM/SGLang 배포 문서 대조 |
| 2026-08-07 | C2(부분), C4, C6(정정) | HoPE 원문 초록 · NVIDIA 스펙 자료 · CLA 인용 수치 역추적 |
| 2026-08-07 | — | Kimi K3 / GLM-5 `config.json`, Nemotron 3 LatentMoE 논문, Gemma 3n 공식 문서 대조 |
| 2026-08-07 | — | **Kimi K3 기술 리포트(arXiv:2607.24653)** ar5iv 대조 · Kimi Linear 비율 ablation ·<br>MoE all-to-all 실측 자료 · 하이브리드 prefix caching · GLM-5.2 IndexShare |

## 열린 질문 — 현황

| 질문 | 상태 |
|---|---|
| 왜 3:1인가 | ✅ **해소** — Kimi Linear ablation. 7:1은 validation 붕괴, 1:1은 추론 오버헤드 |
| K3의 마지막 93층이 MLA인 이유 | ✅ **해소** — 전역 attention 보장용으로 backbone 끝에 추가 |
| Attention Residuals의 실제 동작 | ✅ **해소** — 학습된 pseudo-query + 층 출력 softmax, Block AttnRes 8블록 |
| 하이브리드 prefix caching | ✅ **해소** — Marconi, SGLang MambaRadixCache |
| all-to-all 실측 비중 | ✅ **해소** — 노드 안 ~20%, 노드 간 40~60%, EP=6에서 77% |
| GLM-5.2 IndexShare | ✅ **해소** — 4층마다 indexer 공유, 연산 75% 제거, FLOPs 2.9배 감소 |
| V4에서 활성 expert 8 → 6 | 🟡 **논문에 설명이 없음을 확인** |
| V4의 CSA:HCA 1:1 배치 근거 | 🟡 **논문에 ablation이 없음을 확인** |
| V4-Pro 마지막 층 `compress_ratio`=0 | ⚠️ **미해결** |
| K3에서 decoupled RoPE와 NoPE의 결합 방식 | ⚠️ **미해결** |
| **희소 attention의 gather 실효 대역폭** | ⚠️ **미해결** — 측정 자료 없음. `99-landscape` 99.6 ②의 근거가 약한 이유 |
| CLA/YOCO가 대형 모델에 안 오는 이유 | ⚠️ **미해결** |

> 🟡 표시는 **"자료를 찾았는데 거기에 답이 없더라"** 는 뜻이다.
> ⚠️와 구분해서 적어둔다 — 더 찾아볼 여지가 있는 것과 이미 확인한 것은 다르다.
