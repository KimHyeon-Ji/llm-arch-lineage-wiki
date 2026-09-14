# 99. Landscape — 설계 선택이 시스템을 어떻게 바꾸나

앞의 장들은 컴포넌트별 계보를 설명했다. 이 장은 그 선택들을 한 화면에 놓고
**무엇을 줄였고, 어떤 비용이 새로 생겼는가**를 비교한다. 최신 모델은 대표 사례만
포함하며, 공개 기술 리포트와 공식 설정으로 확인 가능한 구조에 한정한다.

**목차**

| | 절 |
|---|---|
| [99.0](#990-efficient-transformer-발전-계보--시스템-영향) | Efficient Transformer 발전 계보 |
| [99.1](#991-대표-모델-매트릭스) | 대표 모델의 컴포넌트 조합 |
| [99.2](#992-세-가지-진화-궤적) | 압축·고정 상태·쓰기 가능한 메모리 |
| [99.3](#993-kv를-줄이는-방법과-조합) | KV 절감 방법과 조합 |
| [99.4](#994-moe--연산을-줄이고-통신을-늘리다) | MoE의 시스템 거래 |
| [99.5](#995-하이브리드--층마다-역할을-나누다) | 하이브리드 설계 |
| [99.6](#996-모델-규모를-시스템-크기로-번역하기) | 모델 규모와 시스템 크기 |
| [99.7](#997-시스템이-준비해야-할-것) | 전체 시스템 요구사항 |

---

## 99.0 Efficient Transformer 발전 계보 — 시스템 영향

여기서는 FlashAttention처럼 **같은 연산을 더 잘 실행하는 커널**이 아니라,
모델이 저장하고 읽고 쓰는 대상을 바꾼 아키텍처만 비교한다.

| 시기 | 대표 기법 | 해결하려는 병목 | 핵심 아이디어 | 시스템 영향 |
|---|---|---|---|---|
| 2017~ | **MHA** | 장거리 관계의 병렬 모델링 | 모든 head의 토큰별 KV를 유지 | prefill은 compute, decode는 weight·KV traffic의 영향이 큼 |
| 2019~ | **MQA·GQA** | KV 용량·대역폭 | query head가 K/V head를 공유 | KV 감소, 더 큰 batch 가능 |
| 2020~ | **SWA·sparse attention** | 긴 문맥의 전체 attention | window 또는 선택된 block만 읽음 | 연산·읽기량 감소, 선택·gather 비용 증가 |
| 2024~ | **MLA·CSA·HCA** | KV 차원과 후보 수 | 저차원 또는 압축 토큰으로 저장·검색 | KV 감소, projection·indexer·불규칙 접근 필요 |
| 2023~ | **Mamba·DeltaNet·KDA** | 길이에 비례하는 KV | 과거를 고정 크기 recurrent state로 요약 | 해당 층의 KV 소멸, scan·state-update kernel 필요 |
| 2025~ | **Titans·ATLAS** | 고정 update의 표현력 | 추론 중 neural memory를 최적화 | 읽기뿐 아니라 update compute·write traffic·요청별 상태 관리 발생 |
| 2026~ | **Sleep** | 온라인 갱신만으로는 지식이 휘발됨 | 입력 없는 시간대에 빠른 기억을 파라미터로 굳힘 | 서빙 계획에 오프라인 consolidation 슬롯이 들어오고, 사용자별로 가중치가 갈라짐 |
| 2025~ | **HOPE·CMS** | 단일 갱신 속도 | 메모리마다 다른 갱신 주기 | multi-rate read+write 실행과 계층적 상태 관리가 연구 과제 |

```
토큰별 KV를 모두 읽기
  → head를 공유해 덜 저장하기
  → window·sparse 선택으로 덜 읽기
  → latent·압축 토큰으로 더 작게 저장하기
  → 고정 상태로 요약하며 읽고 쓰기
  → 메모리 내용과 갱신 규칙까지 추론 중 바꾸기
```

초기의 중심 질문은 **“KV를 얼마나 덜 읽을까”**였다. linear attention 이후에는
**“상태를 어떻게 갱신할까”**가 추가됐다. Titans·HOPE 계열이 실용화된다면 accelerator도
연산량뿐 아니라 어떤 상태를 어느 주기로 갱신할지 지원해야 한다.

---

## 99.1 대표 모델 매트릭스

이 표의 목적은 순위를 매기는 것이 아니라, 한 모델 안에서 컴포넌트가 어떻게 겹치는지
보여주는 것이다. 숫자는 해당 리포트의 대표 설정이며 다른 크기 변형에는 그대로
적용되지 않을 수 있다.

| 모델 | token mixing | 위치 | channel mixing | residual·norm | decoding·numerics |
|---|---|---|---|---|---|
| **Llama 3 70B** | GQA (64 query / 8 KV heads) | RoPE | dense SwiGLU | Pre-Norm, RMSNorm | BF16 기준 |
| **Gemma 3** | local SWA와 global attention을 5:1로 배치 | RoPE | dense | RMSNorm, QK-Norm | — |
| **DeepSeek-V3** | MLA | decoupled RoPE | 256 routed, top-8 + shared expert | RMSNorm | MTP, FP8 mixed-precision training |
| **DeepSeek-V3.2** | MLA + DSA | decoupled RoPE | V3 계열 MoE | RMSNorm | MTP |
| **DeepSeek-V4-Pro** | CSA와 HCA 교대 | RoPE 계열 | 384 routed, top-6 + shared expert | mHC | MTP → **DSpark**(block diffusion 초안 + 부하 인지 검증), 부분별 FP4/FP8 배포 |
| **Qwen3-Next** | Gated DeltaNet과 Gated Attention의 hybrid | partial RoPE | MoE | zero-centered RMSNorm | — |
| **Kimi Linear** | KDA와 MLA를 3:1로 배치 | MLA 층은 NoPE | MoE | — | 장문 decode용 recurrent kernel |
| **Kimi K3** | 3 KDA + 1 Gated MLA 반복, 마지막에 MLA 추가 | **모든 MLA 층도 NoPE** | Stable LatentMoE, 896 중 16 routed expert 활성 | Block AttnRes | MXFP4-aware post-training |

세로로 읽으면 채택 흐름이 보인다. GQA·MLA는 **KV를 저장하는 방식**을 바꾸고,
sparse attention은 **어떤 KV를 읽을지** 바꾼다. KDA는 일부 층의 토큰별 KV를
**recurrent state로 대체**한다. MoE와 양자화는 이 선택들과 독립적으로 겹칠 수 있다.

Kimi K3가 좋은 종합 예다.

```
sequence 방향 : KDA 3층 + Gated MLA 1층
depth 방향    : Block Attention Residuals
channel 방향  : Stable LatentMoE
numerics      : MXFP4-aware post-training
```

실제 모델은 한 가지 계보를 선택하지 않는다. 서로 다른 병목을 겨냥한 컴포넌트를
겹치되, 학습 안정성과 서빙 구현이 감당할 수 있는 조합만 남긴다.

---

## 99.2 세 가지 진화 궤적

### A. 토큰별 메모리를 압축한다

```
MHA → MQA/GQA → MLA → MLA + DSA → CSA/HCA
```

이 계열은 토큰별 상태를 유지한다. 대신 head를 공유하고, 차원이나 토큰 수를
압축하고, 필요한 항목만 고른다. **정확한 token-to-token 검색을 보존하기 쉽지만**
컨텍스트가 길어지면 저장량 또는 indexer 비용이 남는다.

대표 사례는 DeepSeek 계열이다.

| 단계 | 바뀐 것 | 남은 비용 |
|---|---|---|
| MLA | head별 KV를 latent로 압축 | 모든 토큰의 latent 유지 |
| DSA | 본 attention은 top-k 토큰만 읽음 | indexer는 전체 latent를 scan |
| CSA | 압축 엔트리 위에서 sparse 선택 | indexer·top-k·gather 필요 |
| HCA | 더 강하게 압축한 엔트리를 dense하게 읽음 | 세부 정보 손실 가능성 |

### B. 토큰별 메모리를 고정 상태로 바꾼다

```
linear attention → selective SSM → delta rule → gated delta rule → KDA
```

해당 층의 메모리가 sequence length에 비례하지 않는다. 대신 매 토큰 state를
갱신해야 하며, 임의 위치의 원문을 정확히 꺼내는 능력은 full attention보다 불리할 수
있다. 그래서 Kimi Linear·Kimi K3처럼 일부 global attention 층을 남기는 설계가 쓰인다.

### C. 상태의 갱신 자체를 학습한다

```
고정 recurrent update → Titans의 neural-memory update
                    → ATLAS의 더 표현력 있는 update
                    → HOPE의 multi-rate memory
```

이 계열은 아직 연구 단계다. 핵심 변화는 inference를 read-mostly workload로 보던
전제에서 벗어난다는 점이다. 모델별 weight 외에 **요청별로 변하는 상태**, update의
일관성, checkpoint·복구와 multi-tenant 격리가 새로운 시스템 문제가 된다.

---

## 99.3 KV를 줄이는 방법과 조합

| 축 | 방법 | 주로 줄이는 것 | 남는 비용 |
|---|---|---|---|
| head | MQA·GQA | KV head 수 | query 표현력과 head grouping 선택 |
| feature | MLA | 토큰당 KV 차원 | latent projection과 전용 kernel |
| sequence-local | SWA | 층당 읽는 범위; 구현에 따라 저장 상한 | window 밖 직접 검색 불가 |
| sequence-selective | NSA·DSA | attention이 읽는 후보 | indexer·top-k·gather |
| token compression | CSA·HCA | 저장 토큰 수와 후보 수 | 압축 손실과 이질적 layer 실행 |
| layer | CLA·YOCO | 레이어별 중복 KV | layer coupling과 pipeline 제약 |
| precision | FP8·FP4 KV | 항목당 byte | scale metadata, dequantization, 품질 검증 |
| replacement | Mamba·KDA | 해당 층의 토큰별 KV | recurrent state와 update kernel |

조합할 때는 절감률을 단순히 곱하기 전에 세 가지를 확인해야 한다.

1. **같은 축을 두 번 줄이는가.** GQA와 MLA는 보통 대안 관계다.
2. **품질 손실이 누적되는가.** window·압축·저정밀을 함께 쓰면 개별 오차가 겹친다.
3. **병목이 이동했는가.** KV를 충분히 줄이면 weight read, compute, communication 또는
   불규칙 gather가 다음 병목이 된다.

> 💡 절감률보다 중요한 질문은 **“이 절감 뒤에 어떤 kernel과 통신이 critical path에
> 남는가”**다.

---

## 99.4 MoE — 연산을 줄이고 통신을 늘리다

MoE는 총 파라미터를 늘리면서 토큰당 활성 파라미터를 제한한다. 총 파라미터는
**weight residency**, 활성 파라미터는 주로 **토큰당 compute·weight read**를 결정한다.

| 모델 | 총 / 활성 파라미터 | routed expert 선택 | 핵심 시스템 포인트 |
|---|---|---|---|
| Mixtral 8×7B | 약 47B / 13B | 8개 중 2개 | 비교적 굵은 expert |
| DeepSeek-V3 | 671B / 37B | 256개 중 8개 + shared | fine-grained expert, aux-loss-free balancing |
| DeepSeek-V4-Pro | 1.6T / 49B | 384개 중 6개 + shared | 더 높은 sparsity, expert parallel 필요 |
| Kimi K3 | 2.8T / 104B | 896개 중 16개 + shared | LatentMoE와 load balancing, 대규모 EP |

```
토큰 hidden state
  → router
  → dispatch all-to-all
  → expert grouped GEMM
  → combine all-to-all
```

expert를 잘게 나누면 선택 조합은 늘지만, expert당 token 수가 줄어 GEMM이 작아지고
통신·동기화 비중이 커질 수 있다. 따라서 expert 수가 많다는 사실만으로 효율을
판단할 수 없다. topology-aware placement, token balancing, grouped GEMM, 통신·계산
overlap을 함께 봐야 한다.

LatentMoE는 이 문제에 대한 한 방향이다. expert 사이에 오가는 hidden vector를 더 작은
latent dimension으로 바꾸어 communication payload와 expert weight를 줄이는 대신,
down/up projection을 추가한다. 즉 **통신을 projection compute와 교환**한다.

---

## 99.5 하이브리드 — 층마다 역할을 나누다

긴 문맥 모델은 모든 층을 같은 방식으로 만들 필요가 없다.

| 모델 | 값싼 층 | 전역·정밀 층 | 배치 의도 |
|---|---|---|---|
| Gemma 3 | local SWA | global attention | 지역 mixing을 반복하고 주기적으로 전역 연결 |
| Qwen3-Next | Gated DeltaNet | Gated Attention | 고정 상태와 token-level 검색 결합 |
| Kimi Linear / K3 | KDA | Gated MLA | recurrent 효율과 global retrieval 결합 |
| DeepSeek-V4 | CSA | HCA | 정밀한 sparse 검색과 거친 global summary를 교대 |

하이브리드의 평균 FLOPs만 보면 놓치는 것이 있다.

- 층마다 KV 크기와 kernel이 달라 **메모리 풀을 따로 계산**해야 한다.
- recurrent 층과 attention 층은 prefix-cache hit 조건이 다르다.
- pipeline stage별 일이 달라져 균형이 깨질 수 있다.
- CUDA graph, kernel fusion과 quantization 경로가 layer type마다 달라진다.

따라서 하이브리드는 평균 비용을 줄이는 대신 **스케줄링과 상태 관리의 이질성**을
늘린다. `10-serving`의 hybrid KV cache와 prefix caching이 이 문제를 다룬다.

---

## 99.6 모델 규모를 시스템 크기로 번역하기

### 가중치

가중치의 이론적 최소 크기는 다음과 같다.

$$
\text{weight bytes} \approx N_{\text{total}} \times \frac{b_w}{8}
$$

1T 파라미터는 BF16이면 약 2 TB, 8비트면 약 1 TB, 4비트면 약 0.5 TB다.
실제 배포에는 scale·metadata, padding, 복제, runtime workspace가 추가된다.

MoE에서는 두 숫자를 구분해야 한다.

| 숫자 | 주로 결정하는 것 |
|---|---|
| **총 파라미터** | 전체 weight를 어디에 배치할지, 최소 aggregate memory |
| **활성 파라미터** | 토큰당 expert compute와 weight traffic |

활성 파라미터가 작아도 모든 expert weight는 어딘가에 상주하거나 필요할 때 가져와야
한다. 그래서 MoE는 compute를 줄여도 expert parallel과 weight placement 문제를 남긴다.

### KV와 recurrent state

일반 attention의 KV는 대략 다음과 같다.

$$
\text{KV bytes}
= B \times S \times L \times 2 \times n_{kv} \times d_h \times b_{kv}
$$

여기서 `b_kv`는 항목당 byte다. KV는 batch `B`, context `S`, layer `L`에 모두
비례한다. 반면 KDA·Mamba 층의 recurrent state는 sequence length와 무관하지만,
요청마다 별도 상태가 필요하다.

### 하드웨어를 읽는 법

| 자원 | 모델 설계와의 연결 |
|---|---|
| HBM 용량 | weight·KV·activation을 동시에 수용 가능한가 |
| HBM 대역폭 | decode에서 weight와 KV를 얼마나 빨리 읽는가 |
| scale-up link | TP all-reduce와 노드 내 EP가 감당 가능한가 |
| scale-out fabric | 노드 간 EP·PP·KV transfer가 critical path가 되는가 |
| on-chip SRAM | tile·scale·state update를 재사용할 수 있는가 |

B200의 192 GB HBM3e·8 TB/s와 Rubin의 최대 288 GB HBM4·22 TB/s처럼 용량과
대역폭은 함께 늘고 있다. 다만 peak 사양은 실제 workload의 achieved bandwidth가
아니며, kernel locality와 topology가 이용률을 결정한다.

---

## 99.7 시스템이 준비해야 할 것

### 설계 선택 → 시스템 요구

| 아키텍처 선택 | 새로 요구하는 것 | 덜어주는 것 |
|---|---|---|
| MQA·GQA | head grouping에 맞는 kernel·TP 배치 | KV 용량·읽기량 |
| MLA | latent projection 흡수, MLA 전용 kernel | KV 차원 |
| sparse attention | indexer·top-k·지역성 있는 gather | 읽는 토큰과 attention FLOPs |
| KDA·Mamba | recurrent·chunked kernel, 요청별 state 관리 | 해당 층의 토큰별 KV |
| CLA·YOCO | layer coupling을 반영한 pipeline | 레이어별 KV 중복 |
| fine-grained MoE | EP all-to-all, balancing, grouped GEMM | 활성 compute |
| LatentMoE | down/up projection | expert payload와 weight 크기 |
| 하이브리드 | layer-type-aware allocation·prefix cache·scheduling | 평균 메모리·연산량 |
| speculative·MTP | draft·verify scheduling, rollback | 토큰당 weight-read 비용 |
| FP8·FP4 | scale 관리, 지원 kernel, 품질 검증 | 용량·대역폭·compute |
| test-time memory | update engine, 상태 격리·복구·수명 관리 | 고정 update의 표현력 한계 |

### 시대별 병목과 하드웨어 관점

| 시대 | 두드러진 병목 | 시스템·하드웨어의 대응 |
|---|---|---|
| 초기 Transformer | matrix compute와 memory bandwidth | systolic/tensor core, HBM, on-chip buffer |
| 초거대 dense·MoE | aggregate memory와 device communication | TPU Pod, NVLink/NVSwitch, ICI, collective 최적화 |
| 긴 문맥 serving | KV 용량·대역폭과 prefill/decode 간섭 | PagedAttention, prefix cache, PD 분리, KV transfer |
| hybrid·state model | layer별 kernel·상태·cache 규칙 차이 | type-aware memory manager와 scheduler |
| Titans·HOPE 이후의 연구 방향 | memory update와 여러 갱신 주기 | update engine, frequency-aware hierarchy, state isolation |

### 최종 인사이트

1. **병목은 사라지기보다 이동한다.** KV 압축은 projection·indexer·gather를,
   MoE는 all-to-all을, 저정밀은 scale 관리와 검증을 만든다.
2. **평균 FLOPs만으로는 serving 비용을 예측할 수 없다.** weight·KV byte, achieved
   bandwidth, collective와 layer별 critical path를 함께 봐야 한다.
3. **하이브리드가 늘수록 시스템도 모델 구조를 알아야 한다.** 모든 층에 같은 KV page,
   prefix rule과 quantization kernel을 적용하기 어렵다.
4. **read-mostly 최적화만으로는 test-time memory를 설명할 수 없다.** 추론 중 상태를
   갱신한다면 update 주기, write traffic, 요청 간 격리와 복구가 일급 설계 대상이 된다.

> Transformer 최적화의 질문은 “연산을 얼마나 줄였나”에서
> **“어떤 정보를 어디에 두고, 얼마나 자주 읽고 쓰며, 어느 링크를 건너는가”**로
> 확장되고 있다.

---

## Sources

개별 수식과 구현 출처는 각 컴포넌트 장의 Sources를 참조한다. 이 장의 비교에 직접
사용한 핵심 자료는 다음과 같다.

**논문·기술 리포트**

- Vaswani et al. (2017), *Attention Is All You Need*, arXiv:1706.03762
- Ainslie et al. (2023), *GQA*, arXiv:2305.13245
- Gu & Dao (2023), *Mamba*, arXiv:2312.00752
- DeepSeek-AI (2024), *DeepSeek-V3*, arXiv:2412.19437
- DeepSeek-AI (2025), *DeepSeek-V3.2* — DeepSeek Sparse Attention
- DeepSeek-AI (2026), *DeepSeek-V4*, arXiv:2606.19348
- Kimi Team (2025), *Kimi Linear*, arXiv:2510.26692
- Kimi Team (2026), *Kimi K3*, arXiv:2607.24653
- Behrouz et al. (2025), *Titans*, arXiv:2501.00663
- Behrouz et al. (2025), *ATLAS*, arXiv:2505.23735
- Behrouz et al. (2025), *Nested Learning*, arXiv:2512.24695

**공식 설정·시스템 문서**

- 각 모델의 공식 `config.json`과 model card
- vLLM, *Hybrid KV Cache Manager*
- Kwon et al. (2023), *PagedAttention*, arXiv:2309.06180
- Pan et al. (2024), *Marconi: Prefix Caching for the Era of Hybrid LLMs*, arXiv:2411.19379
- NVIDIA B200·HGX Rubin 공식 사양

**범위와 주의**

- 모델 수치와 하드웨어 사양은 특정 변형의 값이다. 다른 크기·정밀도·제품에
  일반화하지 않는다.
- 벤치마크 배수와 사용자 수 추산은 workload·runtime 의존성이 커서 이 종합표에서
  제외했다.
