# LLM inference의 메모리 이동 — 학습 자료 작성 계획

작성일: **2026-11-04** · 상태: 학습 자료의 구성·예제·검증 계획

## 1. 목표

**요청 하나가 들어와 답변을 생성하고 끝날 때까지, 각 레이어에서 어떤 데이터가 어디에 있고 언제 읽히거나 복사되는지 설명할 수 있게 만든다.**

Transformer의 기본 연산은 조금 알고 있지만 GPU 메모리 구조는 처음 공부하는 독자를 기준으로 쓴다. 설명은 한국어로 하고, weight·activation·KV cache·prefill·decode 같은 용어는 처음에 뜻을 붙인다.

자료를 읽고 다음 질문에 그림과 간단한 계산으로 답할 수 있어야 한다.

1. Weight, KV cache, activation, 임시 버퍼는 각각 누가 만들고 얼마나 오래 보관하는가?
2. CPU DRAM, GPU HBM, L2/L1 cache, shared memory, register는 어떤 관계인가?
3. Prefill과 decode에서 같은 레이어가 읽고 쓰는 데이터는 어떻게 달라지는가?
4. 레이어 0의 처리가 끝나면 weight·KV·hidden state는 각각 어디에 남는가?
5. HBM에 이미 올라간 weight도 왜 매 생성 스텝의 메모리 대역폭을 사용하는가?
6. CPU/SSD offload나 여러 GPU를 쓰면 어떤 전송이 추가되고, 무엇이 그 전송을 결정하는가?

## 2. 기존 자료와 연결

| 기존 자료 | 새 자료에 활용할 부분 |
|---|---|
| [00-foundations](../00-foundations.md) | Decoder 구조, 텐서 shape, prefill/decode의 기본 기호 |
| [01-attention](../01-attention.md) | GQA와 attention 수식 |
| [메모리 요구량 보고서](../reports/llm-memory-requirements.md) | Weight·KV 용량 계산과 실제 모델 비교 |
| [10-serving](../10-serving.md) | Paging, prefix caching, batching, GPU 간 통신 |
| [MiniMax M2.5·M3](minimax-m2.5-m3.md) | 기본 흐름을 이해한 뒤 실제 모델의 attention·MoE에 적용하는 후속 사례 |

새 자료의 중심은 **실행 순서와 데이터의 위치 변화**다. 기존 자료의 용량 수치를 이용할 때도, 저장 용량과 실제 HBM 트래픽을 따로 표시한다.

## 3. 설명의 기준을 먼저 고정한다

### 기본 실행 환경

- HBM을 사용하는 NVIDIA GPU 한 장에서 실행하는 decoder-only Transformer.
- Pre-norm, RoPE, GQA, SwiGLU를 사용하는 dense 모델.
- Weight와 KV가 GPU 메모리에 들어가는 상황에서 시작한다.
- Weight·KV·activation 저장 dtype은 BF16으로 두고, 누산 dtype은 별도로 표시한다.
- 요청 1개, prefix cache miss, 일반적인 autoregressive generation을 먼저 따라간다.
- GPU 내부의 구체적인 커널 예시는 Hopper 계열을 기준으로 설명하고, 구현·버전에 따른 경로 차이를 표시한다.

Offload, batching, quantization, multi-GPU는 기본 실행을 설명한 뒤 조건을 하나씩 바꾸어 비교한다. 실제 엔진의 설정과 커널 이름은 본문 작성 시 버전을 고정한다.

### 모든 장에서 함께 사용할 예제

**손으로 따라가는 예제:** 레이어 3개, 프롬프트 `x1 x2 x3 x4`, 출력 `y1 y2 y3`. 모델의 작은 수치보다 데이터가 생기는 순서에 집중한다.

| Forward | 입력 | 각 레이어의 처리 후 유효 KV | 이번에 예측하는 토큰 |
|---|---|---|---|
| Prefill | `x1 x2 x3 x4` | `x1 … x4`의 KV, 4칸 | `y1` |
| Decode 1 | `y1` | `x1 … x4, y1`의 KV, 5칸 | `y2` |
| Decode 2 | `y2` | `x1 … x4, y1, y2`의 KV, 6칸 | `y3` |

이 표로 **첫 출력 토큰은 prefill의 마지막 위치 logits에서 나오며, 그 출력 토큰의 KV는 다음 forward에서 만들어진다**는 순서를 명확히 한다. 한 forward 도중에는 레이어마다 KV 갱신이 순차적으로 진행되는 모습도 보여준다.

**Byte 계산 예제:** 아래의 교육용 설정을 끝까지 재사용한다. 특정 공개 모델의 정확한 복제라고 가정하지 않는다.

| 기호 | 의미 | 기준값 |
|---|---|---|
| `L` | 레이어 수 | 32 |
| `d` | Hidden size | 4,096 |
| `d_ff` | SwiGLU 중간 폭 | 14,336 |
| `n_h`, `n_kv`, `d_h` | Query head, KV head, head 차원 | 32, 8, 128 |
| `V` | Vocabulary 크기 | 32,000, embedding과 LM head는 별도 weight |
| `B` | 이번 forward에 참여하는 요청 수 | 1부터 시작 |
| `S` | Decode 시작 시 이미 KV에 들어 있는 토큰 수 | 2,048부터 시작 |
| `b_w`, `b_kv`, `b_x` | Weight·KV·activation 원소당 byte | 각각 2 |

Prefill의 프롬프트 길이는 `S_prompt`로 쓴다. Decode attention의 유효 길이는 현재 입력 토큰까지 포함해 `S+1`이다. Byte·KiB·MiB·GiB를 기본으로 쓰고, 제품의 GB 및 GB/s 표기는 변환 기준을 붙인다.

## 4. 목차와 장별 산출물

향후 본문은 `study/inference-memory/` 아래에 다음 순서로 작성한다.

| 파일 | 중심 질문 | 반드시 포함할 설명·그림 | 읽은 뒤 할 수 있어야 하는 것 |
|---|---|---|---|
| `README.md` | 어디부터 읽으면 되나? | 전체 지도, 용어표, 짧은 학습 경로 | 기본 경로와 심화 경로 선택 |
| `00-one-request.md` | 요청 하나가 끝날 때까지 무슨 일이 생기나? | 로딩→tokenization→prefill→sampling→decode→종료 타임라인 | 첫 출력과 이후 출력을 구분 |
| `01-memory-map.md` | 데이터 종류와 물리 메모리는 어떻게 대응하나? | 메모리 계층도, 데이터별 위치·수명 표 | KV cache와 하드웨어 cache의 차이 설명 |
| `02-prefill.md` | 프롬프트 전체가 각 층을 지나갈 때 무엇이 생기나? | 레이어×토큰 KV 격자, activation shape, 첫 토큰 생성 | 레이어별 KV가 채워지는 순서 설명 |
| `03-decode-layer.md` | 한 레이어의 연산마다 무엇을 읽고 쓰나? | 아래의 연산별 데이터 이동표, 레이어 0→1→2 추적 | Weight·KV·hidden의 서로 다른 수명 설명 |
| `04-inside-kernels.md` | HBM의 텐서는 어떻게 연산에 사용되나? | GEMM tile, register/shared memory 재사용, FlashAttention | 모델 레이어와 CUDA kernel을 구분 |
| `05-bytes-and-latency.md` | 얼마나 저장하고 얼마나 옮기며 무엇이 느린가? | 용량 계산, 고유 데이터 양과 실측 트래픽 비교, 간단한 roofline | Batch·길이 변화에 따른 병목을 조건부로 예측 |
| `06-offload-and-serving.md` | 느린 계층이나 다른 GPU가 필요하면 무엇이 달라지나? | Weight/KV offload 타임라인, paging·공유·통신 비교 | 전송 대상·시점·정책·비용 설명 |
| `07-exercises-and-profiling.md` | 내 이해를 어떻게 확인하나? | 손 계산, CPU 추적 예제, 선택적 GPU 관찰, 해설 | 예측과 관찰 결과의 차이 설명 |

각 장은 **질문 → 구체적인 예제 → 그림 → 읽기/쓰기 표 → 필요한 계산 → 이해 확인** 순서로 쓴다. 수식의 기호는 해당 데이터가 그림에 등장한 다음에 소개한다.

## 5. 가장 중요한 본문: 한 레이어를 끝까지 추적한다

`03-decode-layer.md`는 아래 순서를 중심으로 작성한다. 먼저 별도 연산으로 나눈 기준 구현을 설명하고, `04`에서 fusion에 따라 어떤 중간 저장이 줄어드는지 겹쳐 보여준다.

| 단계 | 추적할 데이터 | 반드시 답할 질문 |
|---|---|---|
| Embedding | Token ID, 선택된 embedding 행, hidden | 테이블 전체를 연산에 읽는가, 필요한 행을 조회하는가? |
| RMSNorm | Hidden, norm weight, 통계·출력 | 어떤 값이 임시로 만들어지고, 다음 연산까지 보관되는가? |
| Q/K/V projection | Hidden, `Wq/Wk/Wv`, Q/K/V | Weight의 backing storage와 연산에 사용하는 tile은 어디에 있는가? |
| RoPE + cache append | Q, 새 K/V, cache 위치 정보 | 무엇을 위치 변환하고, 어느 레이어의 KV 슬롯에 쓰는가? |
| Attention | Q, 해당 레이어의 과거 K/V, 현재 K/V | 과거 데이터는 왜 다시 필요하고, 어떤 단위로 가져오는가? |
| Output projection + residual | Attention 출력, `Wo`, residual | Residual은 어디까지 살아 있어야 하는가? |
| 두 번째 norm + SwiGLU | Norm weight, `W_gate/W_up/W_down`, 중간 activation | FFN 내부에서 생성·소비되는 버퍼와 weight tile은 무엇인가? |
| 다음 레이어 | 갱신된 hidden, 다음 레이어의 weight·KV | 다음 층에 전달되는 데이터와 현재 층에 남는 데이터는 무엇인가? |
| Final norm + LM head | 마지막 hidden, vocabulary weight, logits | Embedding 조회와 LM head의 접근 범위는 어떻게 다른가? |
| Sampling + 응답 | Logits 또는 분포, 선택된 token ID | GPU에서 처리할 수 있는 부분과 CPU에 전달할 결과는 무엇인가? |

각 단계의 상세 표에는 다음 열을 공통으로 둔다.

```text
시점 | 레이어/연산 | 텐서·shape·dtype | backing storage
     | 읽는 위치 → 사용하는 위치 | 쓰는 위치 | 재사용 범위 | 수명 | 이동을 결정하는 주체
```

같은 표를 레이어 0, 1, 2에 적용해 다음 관계를 반복해서 확인한다.

- **Hidden state는 다음 레이어로 전달된다.** 별도 kernel 사이에서는 보통 device memory 버퍼를 거치며, fusion 여부에 따라 중간 경로가 달라진다.
- **KV는 레이어별로 남는다.** 다음 생성 스텝에서 해당 레이어가 자기 과거 KV를 다시 참조한다.
- **Weight는 레이어마다 다르다.** 기본 예제에서는 GPU 메모리에 상주하며, 실행 중 필요한 부분을 읽어 사용한다.

레이어 번호와 실행 kernel을 함께 그려, 한 모델 레이어가 여러 kernel로 실행될 수 있음을 보여준다. 이 대응은 선택한 프레임워크·커널 버전에서 확인한다.

## 6. 메모리 계층을 설명할 때 지킬 원칙

계층도는 **장치 간 전송**과 **GPU 내부의 접근·재사용**을 나누어 그린다.

| 구분 | 다룰 저장소·연결 | 설명할 초점 |
|---|---|---|
| 시스템의 저장·전송 | NVMe/SSD, CPU DRAM, PCIe, HBM | 초기 로딩, host/device 복사, offload, staging buffer |
| GPU 내부 | HBM, L2, SM의 L1/shared memory, register | Cache hit/miss, 명시적 tile 적재, 연산 중 재사용 |
| 여러 GPU·노드 | GPU별 HBM, NVLink/PCIe/network | Activation 통신, KV 전달, 분할·복제 |

CUDA memory space의 정의는 [CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#memory-hierarchy)를 기준으로 정리하고, GEMM의 구체적인 tiling·재사용은 [CUTLASS 설명](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)으로 연결한다.

오해하기 쉬운 지점은 별도 설명 상자로 둔다.

1. **SRAM은 물리 메모리 기술이다.** L2/L1 cache와 shared memory의 역할·공유 범위·관리 방식은 따로 설명한다. Register file도 별도 저장소로 표시한다.
2. **하드웨어 cache와 shared memory의 관리 방식은 다르다.** Cache 동작과 kernel이 명시적으로 수행하는 데이터 적재를 구분한다.
3. **읽기와 이주는 다르다.** Tile을 사용한다고 HBM의 원본 weight가 사라지는 것은 아니다. 그림에 원본과 임시 사본을 함께 표시한다.
4. **모든 접근이 같은 직렬 경로를 거치지는 않는다.** 커널·명령·GPU 세대에 따라 cache 우회나 비동기 복사 등이 달라짐을 표시한다.
5. **Tensor Core는 연산 장치다.** 저장소와 연결해 그리되 메모리 계층의 한 칸으로 세지 않는다.

세부 항목으로 register spill과 CUDA의 `local memory`도 다룬다. 이름만 보고 빠른 on-chip 저장소라고 판단하지 않도록 실제 backing 위치를 확인하게 한다.

`04`의 attention 설명은 [FlashAttention 논문](https://arxiv.org/abs/2205.14135)을 기준으로, 전체 attention score 행렬을 HBM에 저장하는 기준 구현과 tile·online softmax를 사용하는 방식을 비교한다. **영구 KV 저장과 attention 중간 결과 저장의 차이**를 같은 그림에 표시한다.

## 7. 숫자는 용량·접근량·실측을 나누어 가르친다

`05`에서는 우선 다음 세 값을 다른 열에 쓴다.

| 값 | 의미 | 확인 방법 |
|---|---|---|
| 저장 용량 | Weight·유효 KV 등 데이터 자체의 byte 수 | Shape와 dtype으로 계산 |
| 연산이 참조하는 고유 데이터 양 | 중복을 제거했을 때 필요한 데이터의 byte 수 | 알고리즘과 접근 범위에서 계산 |
| 실제 계층별 트래픽 | HBM↔GPU 내부, CPU↔GPU 등 경계를 실제로 넘은 byte 수 | 선택한 구현과 profiler에서 확인 |

### 반드시 넣을 계산

- Weight 저장 용량: `P × b_w`. 행렬별로 더하고 embedding·LM head의 tying 여부도 표시한다.
- 전체 KV payload: `2 × B × L × S × n_kv × d_h × b_kv`.
- 한 decode forward의 새 KV payload: `2 × B × L × n_kv × d_h × b_kv`.
- Hidden·Q/K/V·FFN activation: shape와 dtype으로 계산하고, 동시에 살아 있는 버퍼를 따져 peak를 구한다.
- 실제 GPU 메모리 예산: weight + KV 예약량 + activation/workspace + graph·통신 버퍼 + allocator 여유분.

교육용 설정에서 검산할 기준값은 다음과 같다.

| 항목 | 값 |
|---|---|
| 레이어 하나·토큰 하나의 K+V | `2 × 8 × 128 × 2 = 4,096 B = 4 KiB` |
| 모든 레이어·토큰 하나의 K+V | `32 × 4 KiB = 128 KiB` |
| 요청 하나·2,048토큰의 전체 KV payload | `2,048 × 128 KiB = 256 MiB` |
| 다음 입력 토큰 처리로 늘어나는 KV payload | `128 KiB` |

이 값은 KV의 유효 데이터 양이다. Paged allocation의 빈 슬롯, metadata, 정렬, scale, 복제는 별도 계산한다. 고유 과거 KV payload를 모두 참조하는 경우에도 실제 HBM read량은 cache hit, head 간 재사용, tile 재적재 등에 따라 달라진다.

Weight 접근량도 행렬별로 구한다. 특히 embedding은 선택된 행 조회이고, 일반적인 LM head는 vocabulary 전체의 logits를 계산하므로 접근 범위가 다름을 확인한다.

### 병목을 설명하는 순서

1. 같은 weight tile을 여러 토큰에 재사용할 수 있는 prefill/batch와 토큰 한 개의 decode를 비교한다.
2. `B`와 `S`를 각각 바꾸어 weight·KV·activation의 용량과 접근량 변화를 계산한다.
3. `T_memory,ideal = D_HBM / BW_HBM`, `T_compute,ideal = FLOPs / peak_compute`로 이상적인 시간 눈금을 만든다.
4. 대역폭·연산의 peak 값과 실효 값을 구분하고, kernel launch·동기화·통신·전송을 추가로 설명한다.

Prefill과 decode의 병목은 길이·배치·커널 조건과 함께 설명한다. 이 계산으로 나온 시간은 가정에 따른 추정으로 표시하고, GPU 실측과 구분한다.

## 8. Tiered memory와 serving은 조건을 바꾸어 비교한다

`06`에서는 같은 요청의 타임라인에 아래 변화만 하나씩 추가한다.

| 상황 | 이동·관리가 달라지는 대상 | 반드시 보여줄 시점·구분 |
|---|---|---|
| Weight CPU/SSD offload | 실행에 필요한 레이어 weight | 사용 전 적재, 사용 후 GPU 사본 정리, 다음 토큰에서 재적재 가능성 |
| KV CPU offload | 레이어별 KV | 현재 레이어의 KV 준비, 다음 레이어 prefetch, 사용 후 반환·유지 정책 |
| Paged KV cache | KV block과 block table | 슬롯 할당·주소 매핑과 실제 host/device 복사를 구분 |
| Prefix caching | 기존 요청에서 만든 prefix KV | Hit 시 prefill 계산 재사용, decode에서의 KV 참조, 보관·eviction 정책 |
| Continuous batching | 여러 요청의 activation·KV와 공유 weight | 스텝마다 활성 요청이 달라지는 모습과 요청별 KV의 분리 |
| Tensor / pipeline parallelism | 분할 weight·KV, activation, 통신 버퍼 | 어떤 layer/연산 경계에 collective 또는 hidden 전달이 생기는가 |
| Prefill/decode 분리 | Prefill이 만든 KV | Decode 시작 전 KV를 다른 worker로 전달하는 시점과 비용 |

Weight offload는 [Accelerate 공식 문서](https://huggingface.co/docs/accelerate/main/concept_guides/big_model_inference)의 구체적인 정책을 사례로 사용한다. 읽기 전용 weight의 CPU 원본이 남는 경우에는 사용 후 GPU 사본 정리와 GPU→CPU 재복사를 구분한다.

KV offload는 [Transformers 공식 문서](https://huggingface.co/docs/transformers/kv_cache#cache-offloading)의 레이어별 적재·prefetch 사례를 사용한다. 다른 엔진도 같은 정책이라고 일반화하지 않는다. Prefetch가 진행 중이면 현재 계산용 KV와 전송 중인 다음 KV를 위한 공간이 함께 필요할 수 있으므로, 버퍼와 stream도 타임라인에 표시한다.

Paging·공유는 [PagedAttention 논문](https://arxiv.org/abs/2309.06180)을 기준으로 설명한다. 소프트웨어 KV block을 GPU thread block, 하드웨어 cache line, OS page와 구분한다. 커널 읽기는 [vLLM v0.8.1의 설명](https://docs.vllm.ai/en/v0.8.1/design/kernel/paged_attention.html)을 고정된 교육용 사례로 사용한다.

CPU↔GPU 전송에는 `전송 byte / 실효 링크 대역폭`의 시간 눈금을 붙인다. 다음 레이어 적재와 현재 레이어 연산이 겹치는 경우를 그려, 의존성·추가 버퍼·전송 시간이 overlap의 효과를 결정함을 보여준다. 초기 checkpoint 로딩과 매 생성 스텝의 offload 전송도 다른 선으로 표시한다.

## 9. 도표와 실습 계획

### 핵심 도표 6개

1. **저장소 지도:** CPU와 GPU의 경계, HBM backing storage와 on-chip 사본, 전송을 결정하는 주체.
2. **요청 타임라인:** 로딩·prefill·sampling·decode·종료와 첫 토큰/후속 토큰의 구분.
3. **레이어×토큰 KV 격자:** 각 레이어가 KV를 쌓고 자기 행을 다시 읽는 모습.
4. **한 레이어 확대도:** 연산·shape·weight 읽기·KV 읽기/쓰기·activation 수명을 연결.
5. **커널 tile 비교:** GEMM 재사용과 일반 attention/FlashAttention의 중간 저장 비교.
6. **Offload 타임라인:** CPU DRAM, HBM 버퍼, GPU 연산, prefetch stream의 관계.

시간 흐름은 왼쪽→오른쪽, 레이어는 위→아래로 통일한다. 도표에서 weight·KV·activation을 같은 색으로 유지하고, 읽기·쓰기·할당/해제를 선 모양과 글자로 구분한다. 그림은 Mermaid 또는 재생성 가능한 도식 코드로 관리한다.

### 실습은 두 단계로 작성한다

**GPU 없이 하는 필수 실습**

- 작은 모델의 실행 순서를 출력하는 Python 예제를 만든다. 레이어별 입력 위치, KV 유효 길이, 생성/소비한 텐서를 기록한다.
- 프롬프트 4개·출력 3개의 시간표를 손으로 먼저 채운 뒤 예제 출력과 비교한다.
- 교육용 설정으로 KV 용량·새 KV 쓰기량·주요 weight 행렬 크기를 계산한다.
- `B`와 `S`를 바꾸어 용량과 고유 접근량을 표로 비교한다.
- CPU 예제의 출력은 논리적 실행 추적이라고 표시한다. HBM·SRAM의 실제 접근을 재현했다고 해석하지 않는다.

**GPU를 사용할 수 있을 때 하는 선택 실습**

- 작은 실제 모델에서 prefill과 decode를 분리하고, 선택한 레이어·연산에 profiler 구간을 붙인다.
- PyTorch profiler / Nsight Systems로 kernel 실행과 host/device 복사 시점을 관찰한다.
- Nsight Compute로 HBM·L2 트래픽, register/shared memory 사용량을 확인한다.
- Tensor/device 위치와 allocator 수치는 저장 위치·예약량에 대한 증거로 읽는다. `nvidia-smi`만으로 SRAM 사용량이나 레이어별 트래픽을 추론하지 않는다.
- 반복 실행의 warmup, 동기화, backend, dtype, GPU·소프트웨어 버전, profiler replay/cache 설정을 결과와 함께 기록한다.

Profiler의 cache flush·replay가 관찰 결과에 영향을 줄 수 있으므로 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)를 근거로 측정 조건을 설명한다. CPU/GPU 실습 모두 질문·예상 결과·관찰 결과·해설을 함께 제공한다.

## 10. 실제 작성 순서와 완료 기준

| 단계 | 작성할 자료 | 완료 기준 |
|---|---|---|
| 1. 전체 흐름 | `README`, `00`, `01`, 저장소 지도·요청 타임라인 | 첫 토큰 생성 시점과 각 데이터의 위치·수명을 설명할 수 있음 |
| 2. 레이어 추적 | `02`, `03`, KV 격자·한 레이어 확대도 | Prefill과 decode 각각에서 레이어 0→1→2의 읽기/쓰기를 따라갈 수 있음 |
| 3. 커널·비용 | `04`, `05`, tile 비교·계산 예제 | HBM 상주와 on-chip 읽기를 구분하고, 저장 byte와 트래픽 byte의 차이를 설명할 수 있음 |
| 4. 계층 확장·확인 | `06`, `07`, offload 그림·실습·해설 | Offload/공유/통신으로 추가되는 이동을 식별하고 예측과 관찰을 비교할 수 있음 |

첫 학습은 `00 → 01 → 02 → 03`으로 요청과 레이어를 따라간다. 이후 `04 → 05 → 06 → 07`에서 내부 동작과 시스템 조건을 확장한다.

후속 부록에서는 quantization의 scale·변환 버퍼, MoE의 resident expert weight와 실제 접근 expert, MLA·sliding window·recurrent state, speculative decoding의 추가 KV를 다룬다. 기본 그림의 어느 부분이 바뀌는지 기존 위키·MiniMax 자료에 연결한다.

### 최종 점검

- [ ] 모든 이동 화살표에 데이터 이름·출발지·도착지·실행 시점을 적었는가?
- [ ] 런타임의 명시적 복사, 하드웨어 cache 동작, kernel의 tile 적재를 구분했는가?
- [ ] 레이어를 넘는 hidden과 레이어별로 남는 KV를 구분했는가?
- [ ] Weight 재사용·KV 증가·activation 수명을 prefill/decode 양쪽에서 설명했는가?
- [ ] 유효 데이터, 예약된 공간, peak memory, 실제 트래픽을 구분했는가?
- [ ] 요청 종료 후 KV 재사용·prefix 보관·allocator 예약 유지의 정책 차이를 설명했는가?
- [ ] 수식과 계산 예제가 일치하고, 가정·단위·구현 버전이 표시되어 있는가?
- [ ] GPU를 사용할 수 없어도 필수 학습 경로와 이해 확인 문제를 끝낼 수 있는가?
- [ ] 논문·공식 구현이 보장하는 사실, 설명용 가정, 실측 결과를 구분했는가?
