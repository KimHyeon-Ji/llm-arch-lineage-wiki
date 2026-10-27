# LLM 메모리 분석 — 먼저 찾아본 기존 자료

조사일: **2026-10-27**. 논문·보고서·분석 글의 원문과 저자·발행기관 페이지를 우선 확인했다. 아래는 자료의 적합성을 확인한 1차 조사이며, 논문마다 확인한 범위를 따로 적었다.

**관련 자료는 이미 있다.** 모델별 용량 비교는 Raschka의 갤러리·계산기가, 가중치와 KV의 데이터 이동을 함께 이해하는 데에는 2026년 I/O 종설이 특히 가깝다. 이 자료들을 출발점으로 쓰고, 빠진 최신 상태·정밀도·접근 특성만 보완하는 편이 좋다.

## 먼저 볼 자료 5개

| 자료 | 이 보고서에 유용한 점 | 사용할 때 구분할 점 |
|---|---|---|
| **1. [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) + [Memory Calculator](https://sebastianraschka.com/llm-architecture-gallery/memory-calculator/)** — Sebastian Raschka | 여러 모델의 전체·활성 파라미터, 구조, 토큰당 KV와 가중치+KV 추정치를 살펴볼 수 있다. 원하는 “모델별 한눈 비교”와 가장 가까움 | 연구자 본인의 분석·계산 자료다. 실제 실행 메모리와 구분해야 함 |
| **2. [I/o for LLM inference: a survey of storage and memory bottlenecks](https://link.springer.com/article/10.1007/s10462-026-11651-1)** — Artificial Intelligence Review, 2026-07-24 | 가중치·KV·활성값의 데이터 이동을 함께 정리한다. **용량과 읽기량을 연결하는 전체 틀**로 적합 | 종설에 실린 예시 조건과 모델 범위를 확인. 최신 모델 전부를 비교한 표로 보지는 않음 |
| **3. [Memory-Efficient LLM Inference Algorithms](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2026/EECS-2026-175.html)** — Coleman Hooper, UC Berkeley 박사학위논문·기술 보고서, 2026-05-15 | 공식 요약에서 2장의 성능 분석, 3장의 가중치 양자화, 4장의 KV 양자화를 확인. **길어지는 문맥에서 읽기 부담이 어떻게 달라지는지** 조사할 때 유용 | 이번에는 공식 요약·장별 안내를 확인. 본문 PDF는 도구의 크기 제한으로 열지 못했으므로 그림·실험 수치는 아직 미검토 |
| **4. [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/)** — How To Scale Your Model | 가중치·KV·batch·추론 속도의 관계를 설명하는 공개 기술 문서. **우리 보고서의 설명 순서**를 잡는 데 유용 | 계산 예제와 하드웨어 가정을 사용하는 해설 자료. 결과를 곧바로 특정 최신 모델의 실측값으로 옮기지 않음 |
| **5. [Efficient Attention Methods: Hardware-efficient, Sparse, Compact, and Linear Attention](https://attention-survey.github.io/)** — 저자 공개 종설·프로젝트 | 저장을 줄이는 방법, 계산을 건너뛰는 방법, 상태로 요약하는 방법을 나누어 이해하기 좋음 | 방법을 분류하는 자료다. 최신 모델의 실제 배포 용량은 모델별 공식 자료로 보완 |

이 가운데 **1번은 숫자·비교 방식, 2번은 분석의 뼈대, 4번은 설명 방식**에 우선 활용한다. 3번은 실험 근거를 더 확인할 때, 5번은 구조 변화의 의미를 정리할 때 참고한다.

## 가장 가까운 자료를 조금 더 살펴보면

### 1. Raschka의 갤러리와 계산기: 처음부터 표를 새로 만들 필요를 줄여준다

갤러리에는 기준일에 DeepSeek-V4.1-Flash, Qwen3.8-Flash-Next, Kimi K3 등의 항목도 있었다. 모델별 설정·기술 보고서로 연결되어 있어 후보와 기준값을 찾기 좋다. [갤러리](https://sebastianraschka.com/llm-architecture-gallery/)

갤러리에 모델 항목이 있다는 사실과, 그 모델의 전체 상태 계산이 제공된다는 사실은 구분한다. 최신 hybrid의 고정 상태와 local·indexer 등은 별도로 확인한다.

계산기는 컨텍스트 길이, 동시 요청 수, 가중치·KV 정밀도를 바꿔 보는 도구다. 다만 다음 항목을 제외하거나 단순화한다고 저자가 명시한다.

- recurrent state와 sparse 보조 버퍼 등은 계산 범위에서 빠진다.
- sliding window는 실제 보존 상한을 반영하지 않은 전체 보존 가정이 사용될 수 있다.
- 양자화 부가 정보, 임시 버퍼, 실행 엔진의 예약 메모리는 포함하지 않는다.
- 복잡한 구조의 전체 캐시가 설명되지 않으면 결과를 비워 두기도 한다.

따라서 **기존 숫자를 모으는 출발점**으로 쓰되, 전체 실행 용량이라고 그대로 옮기지 않는다. 저자의 [계산 가정](https://sebastianraschka.com/llm-architecture-gallery/memory-calculator/)과 [KV 계산 노트](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)를 함께 확인한다.

### 2. 2026년 I/O 종설: 질문과 가장 많이 겹치는 논문

가중치·KV·활성값의 세 흐름을 구분하고, 양자화·캐시 압축·offload·batching 등을 연결한다. 특히 **§3 가중치 I/O, §4 KV I/O, §8 기법의 조합**이 이번 주제에 가깝다.

확인한 그림은 **Figure 3**이다. 문맥 길이에 따른 모델별 KV 용량과, 가중치 대비 KV 비율을 두 패널로 보여준다. 우리가 원하는 그래프와 매우 비슷하다. **Figure 1**은 각 최적화가 어떤 데이터 흐름과 계층을 겨냥하는지 정리한다. [논문 원문](https://link.springer.com/article/10.1007/s10462-026-11651-1)

그림의 축과 비교 방식을 참고하고, 최신 모델이 빠진 부분을 보완한다. 원그림의 복제·변형에는 원문의 CC BY-NC-ND 4.0 조건이 있으므로, 원문 링크와 자체 데이터 도표를 우선 검토한다.

### 3. Efficient Attention 종설: “무엇을 줄였는가”를 설명하는 데 유용

저자 페이지의 **Figure 1**은 방법을 네 갈래로 나누고, **Table 2**는 compact attention의 토큰별 KV 크기 등을 정리한다. 이 분류를 바탕으로 본문에서는 다음처럼 풀어 쓴다.

| 독자가 이해할 표현 | 조사할 기술적 차이 |
|---|---|
| 같은 내용을 더 작게 저장한다 | head 공유·낮은 차원의 표현·압축 |
| 필요한 부분을 골라 읽는다 | sparse 선택·색인·gather |
| 과거를 요약한 상태를 계속 고친다 | linear attention·recurrent state |
| 남길 대화 구간을 제한한다 | sliding window·회수 정책 |

이는 설명을 위한 재구성이다. 한 모델이 여러 방법을 함께 쓸 수 있고, “덜 읽음”이 곧 “덜 저장함”을 뜻하지는 않는다. [저자 공개 자료](https://attention-survey.github.io/)

## 보완 자료

| 자료 | 확인한 내용과 활용 | 한계 |
|---|---|---|
| [Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization](https://aclanthology.org/2026.findings-acl.1916/) — Findings of ACL 2026, 7월 | 초록·발행 정보를 확인. 실행·스케줄링, 배치·이동, 표현·보존 관점으로 KV 시스템을 정리 | 용량 비교표보다 서빙·캐시 관리의 분류에 유용 |
| [The KV Cache Is the New Memory Wall](https://arxiv.org/abs/2609.30854) — 2026-09-25 프리프린트 | 본문에서 길이·batch에 따른 weight/KV 읽기 모델을 확인. Figure 1은 Llama 계열의 문맥별 KV 용량 | 본문은 계산값과 실측을 구분한다. 해당 그림도 계산 결과이며, 프리프린트라는 상태를 표시 |
| [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) — NVIDIA, 2023-11-17 | 가중치와 KV, prefill·decode, batching을 설명하는 공식 기술 글 | 기초 설명에 활용. 최신 구조의 수치 근거는 별도 필요 |
| [Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) — NVIDIA, 2025-12-08 | Figure 4에서 KV 메모리 예산에 따른 첫 응답 지연·캐시 적중률 비교를 확인 | 특정 모델·포맷·플랫폼의 제작사 실험. 용량 절감과 성능 개선을 일반 비율로 확대하지 않음 |
| [Efficient Attention Mechanisms for Large Language Models: A Survey](https://arxiv.org/abs/2507.19595) — 2025년 초고, 2026-02-07 v3 | 초록·버전을 확인. linear·sparse·hybrid를 설명하는 추가 종설 | 분류가 겹치므로 우선 읽을 자료를 늘리기보다 필요한 내용만 보충 |
| [SGLang 팀의 대규모 DeepSeek 배포 사례](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/) — 2025-05-05 | Attention Layers 절에서 DP Attention을 통한 장치 간 KV 중복 제거 설명을 확인 | 특정 배포의 사례. 모든 TP 구성의 KV 복제 배수를 고정하는 근거로 쓰지 않음 |
| [Engram 공식 저장소](https://github.com/deepseek-ai/Engram) | 결정적 주소 계산을 이용한 대형 embedding table의 host offload 가능성을 공식 설명에서 확인 | 기능의 가능성과 특정 모델의 기본 상주 위치를 구분 |
| [Memory Solutions for Gen AI Part 31: KV Cache Meets Memory Wall](https://counterpointresearch.com/en/reports/Memory-Solutions-for-Gen-AI-Part-31-KV-Cache-Meets-Memory-Wall) — Counterpoint Research | 발행기관 페이지에서 관련 산업 보고서의 존재 확인 | 공개 페이지에서 본문·수치·그림은 확인하지 못함. 이번 수치 근거로 사용하지 않음 |

## 이미 있는 자료와 추가할 부분

| 궁금한 점 | 기존 자료로 얻을 수 있는 것 | 추가 확인할 부분 |
|---|---|---|
| 모델 가중치는 얼마나 큰가? | 갤러리의 전체 파라미터와 계산기의 정밀도별 추정 | 공식 배포 포맷, 추가 모듈, 집계 범위 |
| 문맥이 길면 KV는 얼마나 늘어나는가? | 기존 계산식·토큰별 비교값·길이별 그래프 | 최신 압축·공유 구조와 window 보존량 |
| sparse·linear가 무엇을 줄이는가? | 종설의 분류와 원 논문의 비교 | 모델별 실제 KV, 고정 state, 보조 캐시 |
| 언제 가중치보다 KV가 중요해지는가? | I/O 종설·Scaling Book의 길이·batch 분석 | 보고서 대표 모델의 조건에 맞는지 |
| 읽기·쓰기는 어떻게 다른가? | 기존 I/O 분석과 공식 서빙 문서 | 가중치 읽기, KV 추가·재조회, state 갱신을 쉬운 표로 통합 |
| 최신 모델들을 한 표에서 비교할 수 있는가? | 갤러리가 상당 부분 제공 | 같은 조건·포함 범위로 정리하고 누락된 상태를 보충 |

**이번 검색에서 확인한 범위에서는, 최신 모델의 전체 상태와 읽기·쓰기까지 조건을 통일해 완성한 한 자료를 확인하지는 못했다.** 하지만 필요한 기초 분석과 계산의 상당 부분은 이미 있다. 기존 자료를 묶고 검증·보완하면 된다.

## 보고서에 반영할 결정

1. **기존 분석을 먼저 읽고 활용한다.** 특히 갤러리·계산 노트와 I/O 종설을 기준으로 수집표를 만든다.
2. **직접 계산은 조건을 맞추거나 빠진 부분을 채울 때 한다.** 새 계산기·범용 엔진 구현을 먼저 만들지 않는다.
3. 조사는 **8~10개**, 본문은 **대표 모델 6~8개와 핵심 도표 4개**로 제한한다. 전체 모델·공식·검증 기록과 GPU 배포 사례는 부록으로 보낸다.
4. 첫 문장부터 “얼마나 필요한가 → 길어지면 어떻게 되나 → 무엇이 달라졌나 → 어떻게 읽고 쓰나” 순서로 설명한다.
5. 외부 그림은 원문·조건을 확인해 필요한 것만 쓰고, 복잡한 그림은 내용을 이해한 뒤 단순한 자체 도표로 설명한다.

## 외부 검토 뒤 재확인한 사항

- **Springer 종설:** 검토자는 접근하지 못했으나, 이 환경에서는 같은 기준일에 원문 HTML·Figure 3 설명·라이선스를 재확인했다. 기존 기록을 유지한다. [원문](https://link.springer.com/article/10.1007/s10462-026-11651-1)
- **Qwen3.8-27B:** 공식 모델 카드의 DeltaNet·full attention 혼합을 확인했다. FFN은 dense, attention은 hybrid로 따로 표기한다. 검토에 적힌 고정 상태의 수치는 구현 dtype·보조 상태를 확인한 후 사용한다. [공식 카드](https://huggingface.co/Qwen/Qwen3.8-27B)
- **고정 상태의 실행 예외:** vLLM 문서에서 Mamba의 block 경계 prefix 상태 보관과 page padding을 확인했다. 요청별 최소 상태와 엔진이 추가로 보관하는 상태를 나누어 설명한다. [공식 문서](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
- **DeepSeek의 replay:** 공식 카드에서 SWA Bounded Replay와 persistent/global cache의 구분을 확인했다. 보관량 감소를 실행 중 전체 상태의 동일 비율 감소로 확대하지 않는다. [공식 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)

수정한 전체 계획: [llm-memory-requirements-plan.md](llm-memory-requirements-plan.md).
