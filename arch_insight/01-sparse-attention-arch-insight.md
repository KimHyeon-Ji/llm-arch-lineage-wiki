# Sparse Attention → Architecture Insight

> **핵심 질문: 적게 읽는 attention을, 실제로도 빠르게 읽게 하려면 어떤 하드웨어가 필요한가?**

첫 아이디어 논의용 · **decode attention 중심** · 배경: [세미나 요약 §4–5](../seminar/01-attention-seminar_sum_ver.md). 아래 아키텍처는 연산 특성에서 도출한 **설계 가설**이다.
`S` = 누적 토큰 수, `W` = 최근 윈도우, `b` = 블록 크기, `k` = 선택 개수. GQA·MLA는 **저장 표현**, 아래 계열은 **읽는 범위·단위**를 바꾸는 축이다.

## 1. 계열 지도 — 무엇을 고르고, 무엇을 읽나?

| 계열 | 주요 연산 | 메모리 접근·KV 저장 | 아키텍처 키워드 |
|---|---|---|---|
| **고정 지역형 — SWA** | 최근 `W`개 attention, 오래된 KV 교체 | 최근 구간 순차 읽기; 해당 층 KV는 `W`로 제한 | **ring buffer · 작은 working set · 주소 예측** |
| **토큰 선택형 — [DSA](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)** | 작은 색인 `S`개 점수 → top-k → 선택 KV attention | 색인은 순차, KV는 흩어진 토큰 읽기; **전체 KV + 색인 보존** | **scan · selection · gather · 동시 메모리 요청** |
| **블록 선택형 — [NSA](https://arxiv.org/html/2502.11089v1)·[MoBA](https://arxiv.org/html/2502.13189v1)** | 압축/요약 점수 → 블록 선택 → 원본 KV attention¹ | 블록 사이는 불연속, 블록 안은 연속; 원본 KV 보존 | **block gather · GQA 그룹 재사용 · tile** |
| **색인 압축 + 블록 선택형 — [QSA](https://arxiv.org/html/2608.30320v1)** | 블록 색인 `S/b`개 점수 → top-k → 원본 KV attention | **색인만 축약**, KV는 토큰별 보존; 선택 블록 안을 이어 읽기 | **큰 burst · 검색량 감소 · 읽기 낭비 절충** |
| **KV 압축 + 선택형 — [CSA](https://arxiv.org/html/2606.19348v1)** | KV를 묶어 저장 → 압축 색인 점수 → top-k → 압축 KV attention | 저장·검색 대상 모두 축소; **gather는 남음** | **압축 엔진 · 압축 대기 state · 혼합 캐시** |
| **KV 고압축 + 전부 읽기형 — [HCA](https://arxiv.org/html/2606.19348v1)** | KV를 크게 묶어 저장 → 압축 엔트리 전체 attention | 압축 이력 순차 읽기; **선택용 top-k·gather 없음** | **streaming · dense tile · 압축 처리량** |

¹ NSA는 **압축·선택·지역** 세 갈래를 결합한다. CSA·HCA도 압축 이력 외에 **최근 원본 KV**를 함께 읽는다. 세미나 설정 예: DSA 2,048토큰 / QSA 4토큰×512블록 / CSA 4:1 / HCA 128:1.

**구분 포인트:** QSA는 *고르는 단위*를 묶고, CSA는 *저장하는 내용*을 묶는다. **미선택 KV ≠ 삭제 가능한 KV**다. HCA는 sparse 계열의 비교 대상이지만, 압축 이력 위 연산은 dense다.

## 2. 연산 특성 → 아키텍처 아이디어

```text
선택형의 경로: 색인 순차 스캔 → 점수 집계·top-k → 주소 생성·변환 → KV gather → QKᵀ·softmax·PV
필요한 자원:   저정밀 내적       비교·reduction      주소 처리       메모리 병렬성     tile 연산·재사용
```

| 관찰 → 인사이트 | 제안할 아키텍처 아이디어 | 기대 효과 / 검증할 대가 |
|---|---|---|
| **① 고르기도 비싸다.** 본 attention을 `k`개로 줄여도 색인은 `S` 또는 `S/b`개를 훑는다 | **메모리 근처 selector**: 저정밀 내적·점수 집계·타일별 top-k·전역 병합. 큰 점수 배열을 외부 메모리에 쓰지 않고 선택 주소를 전달 | 색인 데이터 이동·중간 쓰기 감소 / **selector 연산량, 병합·장치 간 전달 비용** |
| **② 적은 바이트 ≠ 짧은 지연.** 흩어진 요청을 충분히 겹쳐야 한다 | **sparse load 전용 전단**: 페이지·채널별 요청 묶기 + 다수 outstanding read + SRAM으로 직접 gather. 공유 KV는 헤드들이 재사용 | 실효 대역폭·연산기 공급률 향상 / **주소 재배열 비용, bank 쏠림, SRAM 용량** |
| **③ 동적 주소는 top-k 뒤에 안다.** 다음 스텝 주소 예측에는 근거가 필요하다 | **선택 결과 기반 prefetch**: 타일 `n` 연산 중 `n+1` KV를 DMA로 가져오는 이중 버퍼. 선택 공유 모델은 후속 층의 **별도 KV**도 미리 요청 | fetch와 attention 겹치기 / **숨길 연산 시간·버퍼 크기**. 이전 스텝 선택으로 예측하려면 중복률·낭비 측정 |
| **④ 캐시마다 수명이 다르다.** 최근 KV·압축 대기는 자주 갱신, 과거 KV는 반복 읽기 | **수명별 KV 관리**: 최근 window·압축 대기는 SRAM 우선, 색인은 순차 스캔용 영역, 과거 KV·압축 엔트리는 별도 HBM page pool | 지역성·할당 효율 개선 / **요청 수에 따른 SRAM 압박**. DDR/CXL 확장은 선택 miss 전송 지연이 허용될 때 |
| **⑤ 블록 크기가 HW 효율을 바꾼다.** 큰 블록은 연속 읽기에 유리하지만 불필요 토큰도 가져온다 | **선택 블록·KV page·연산 tile 공동 설계**: burst 정렬, 블록 안 연속 배치, GQA/MLA 공유 헤드에 KV broadcast. HCA/SWA에는 순차 경로 제공 | 전송 활용률·KV 재사용 향상 / **블록 과다 읽기·패딩**, 선택 단위를 바꾸면 품질도 검증 |

**접근 패턴을 볼 때:** 논리적으로 이웃한 토큰도 paged KV에서는 물리적으로 떨어질 수 있고, 흩어진 토큰도 내부 벡터는 연속일 수 있다. **선택 토큰 수 ≠ 물리 메모리 거래 수.** page 배치·burst·동시 요청 수가 관건이다. 읽어 온 뒤에는 공유 헤드의 dense tile 연산이 가능하므로 연산기 규모도 함께 산정한다.

## 3. 첫 논의에서 잡을 후보와 판별 기준

**우선 후보: `selector → 비동기 gather/DMA → SRAM → attention 연산기`를 연결한 sparse decode 경로.**
연산기 TFLOPS와 HBM 최대 BW 외에 **주소 처리량·동시 read 수·선택/이동/연산의 겹침**을 설계 변수로 둔다. 첫 비교는 **같은 선택 결과**에서 일반 gather 대비 이 경로가 빨라지는지 확인한다.

**최소 측정:** 단계별 지연(scan / top-k / gather / attention), 유효 KV 바이트 ÷ 실제 전송 바이트, 메모리 동시 요청 수, prefetch 적중·낭비율, 요청당 KV 용량. `S·batch·k·b`를 바꾸며 병목 이동을 본다. 짧은 컨텍스트는 선택 오버헤드가 클 수 있어 dense 경로와도 비교한다. 이는 **attention 경로**의 가설이며, 전체 decode 효과는 가중치 읽기·FFN·통신을 포함해 판단한다.

구현 참고: 선택 주소로 KV를 읽는 [FlashMLA sparse decode](https://github.com/deepseek-ai/FlashMLA#deepseek-sparse-attention-dsa-decoding), 최근 KV·압축 대기를 별도 state로 관리하는 [DeepSeek-V4 §3.5](https://arxiv.org/html/2606.19348v1). 설계 후보의 효과·신규성은 후속 검증 대상이다.
