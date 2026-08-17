# 10. Serving — 실제로 돌릴 때 벌어지는 일

앞의 아홉 파일은 **모델이 어떻게 생겼는가**를 다뤘다.
이 파일은 그 모델을 GPU에 올려 실제 사용자를 받을 때 벌어지는 일을 다룬다.

여기서 앞의 결정들이 전부 청구서로 돌아온다.

- MLA를 쓰면 KV 페이지 구조가 달라진다 (`10.2`)
- linear 하이브리드를 쓰면 층마다 KV 할당이 달라진다 (`10.2`)
- MoE를 쓰면 all-to-all이 병목이 된다 (`10.6`)
- GQA의 `n_kv`가 TP 분할 가능한 GPU 수를 제한한다 (`10.6`)
- 층수가 많으면 지연이 늘고, 이건 GPU를 붙여도 안 줄어든다 (`10.6`)

그리고 이 파일의 마지막 절에서 **"어떤 조건에서 무엇이 먼저 터지는가"** 를 종합한다.

## 이 장의 발전 계보와 시스템 영향

| 단계 | 대표 기법 | 해결하려는 병목 | 핵심 아이디어 | 시스템 영향 |
|---|---|---|---|---|
| 기준점 | static batching · contiguous KV | 요청 길이 차이와 KV 단편화 | 요청별 고정 buffer·batch | 구현은 단순하지만 memory 낭비와 head-of-line blocking |
| KV 가상화 | **PagedAttention** | 연속 KV 할당과 단편화 | KV를 고정 크기 page/block으로 관리 | serving capacity·batch 크기 증가, page table·kernel integration 필요 |
| 계산 재사용 | **prefix caching** | 반복되는 system prompt·agent history prefill | 같은 prefix의 KV/state를 공유 | TTFT·prefill FLOPs 감소, cache key·eviction·state 일관성 필요 |
| 동적 스케줄링 | **continuous batching · chunked prefill** | 요청 도착·길이 차이와 prefill 간섭 | step마다 batch 재구성, 긴 prefill을 chunk로 분할 | throughput·tail latency 향상, scheduler 복잡도 증가 |
| 단계 분리 | **PD disaggregation** | compute-bound prefill과 memory-bound decode 충돌 | 서로 다른 worker/hardware pool에서 실행 | 자원 특화 가능, KV transfer와 load balancing이 새 병목 |
| 모델 분산 | **TP · EP · PP · CP** | 한 device의 memory·compute 한계 | tensor·expert·layer·context 축으로 sharding | capacity 확장, collective·all-to-all·pipeline bubble 증가 |

> **이 장의 병목 이동:** 단일 GPU의 **HBM capacity** → 요청 간 **fragmentation·scheduling** →
> worker 간 **KV transfer와 collective communication**. 모델 아키텍처가 줄인 비용은 여기서
> 더 큰 batch나 더 긴 context로 소비되므로, 최종 병목은 workload에 따라 다시 달라진다.

---

## 지도

```
        [두 단계의 성격이 다르다]
                 │
        prefill (compute-bound)  ↔  decode (memory-bound)
                 │
        ┌────────┼────────┬──────────┐
        │        │        │          │
   [메모리 관리] [계산 재사용] [스케줄링] [분리]
        │        │        │          │
  PagedAttention prefix  continuous   PD
        │       caching  batching  disaggregation
        │        │      chunked prefill  │
        │        │        │          │
        └────────┴────────┴──────────┘
                 │
            [여러 GPU로]
                 │
        TP · PP · EP · CP · DP
```

**목차**

| | 절 | 한 줄 |
|---|---|---|
| [10.1](#101-두-단계는-정말-다른-작업이다) | **prefill vs decode** | 정말 다른 작업이다 |
| [10.2](#102-pagedattention--kv를-페이지로) | **PagedAttention** | KV를 페이지로 |
| [10.3](#103-prefix-caching--아예-계산하지-않기) | **prefix caching** | 아예 계산하지 않기 ★ |
| [10.4](#104-continuous-batching과-chunked-prefill) | **배칭과 chunked prefill** | 언제 무엇을 실행할까 |
| [10.5](#105-pd-disaggregation--아예-분리하기) | **PD disaggregation** | 아예 분리하기 |
| [10.6](#106-병렬화--무엇을-쪼개고-무엇을-주고받나) | **병렬화** | 무엇을 쪼개고 주고받나 ★ |
| [10.7](#107-종합--무엇이-먼저-터지는가) | **종합** | 무엇이 먼저 터지는가 ★ |

---

## 10.1 두 단계는 정말 다른 작업이다

`00-foundations` `0.7`의 표를 다시 가져온다.

| | **prefill** | **decode** |
|---|---|---|
| 한 번에 처리 | `S`개 토큰 | 1개 |
| 연산 형태 | **GEMM** | **GEMV** |
| 병목 | **연산 성능** | **메모리 대역폭** |
| GPU 활용률 | 높음 | 낮음 |
| 사용자 지표 | **TTFT** (첫 글자까지) | **TPOT** (글자 간 간격) |
| 배치를 키우면 | 이미 충분 — 별 이득 없음 | **큰 이득** |

같은 가중치, 같은 코드다. 그런데 요구하는 자원이 정반대다.

### 섞으면 서로를 방해한다

한 GPU에서 둘을 함께 돌리면 문제가 생긴다.

```
 시간 ──────────────────────────────────────►

 decode  ▓ ▓ ▓ ▓ ▓                     ▓ ▓ ▓
                    ████████████████
                    ↑ 긴 prefill이 들어옴
                      그동안 decode가 멈춘다
```

사용자 입장에서는 **글자가 흘러나오다가 뚝 끊긴다.**
평균 지연은 괜찮은데 체감이 나쁘다.

`10.4`와 `10.5`가 이 문제에 대한 두 가지 답이다.

---

## 10.2 PagedAttention — KV를 페이지로

### 문제

KV cache를 어떻게 할당할까. 가장 단순한 방법은 **요청마다 최대 길이만큼 연속 공간을
미리 잡는 것**이다.

이러면 두 가지가 낭비된다.

| 낭비 | 왜 |
|---|---|
| **내부 단편화** | 최대 4096 토큰으로 잡았는데 실제로 200개만 쓰면 나머지는 놀고 있다 |
| **외부 단편화** | 요청이 들고 나면서 메모리에 구멍이 생겨 큰 연속 공간을 못 잡는다 |

실제로 이 낭비가 상당해서, 쓸 수 있는 메모리의 상당 부분을 못 쓰는 상황이 생겼다.

### 아이디어

**운영체제의 가상 메모리를 그대로 가져온다.**

KV cache를 고정 크기 **페이지(블록)** 로 쪼개고, 논리적 순서와 물리적 위치를
**페이지 테이블**로 연결한다.

```
 논리 (요청이 보는 것)      페이지 테이블      물리 (실제 메모리)

 토큰 0~15   ──────────►  블록 7    ──────►  [블록 7]
 토큰 16~31  ──────────►  블록 3    ──────►  [블록 3]
 토큰 32~47  ──────────►  블록 12   ──────►  [블록 12]
                                             (연속일 필요가 없다)
```
> **그림 10.2** — 논리적으로 연속인 KV가 물리적으로는 흩어져 있어도 된다.

얻는 것이 셋이다.

| 효과 | 왜 |
|---|---|
| **필요할 때만 할당** | 실제 길이만큼만 페이지를 잡는다 |
| **단편화 제거** | 고정 크기라 구멍이 안 생긴다 |
| **페이지 공유** | 같은 내용의 페이지를 여러 요청이 공유할 수 있다 → `10.3` |

메모리 활용률이 오르면 **배치를 더 키울 수 있고**, 배치가 커지면
`00-foundations` `0.6`에 따라 가중치 읽기가 amortize되어 처리량이 오른다.

### 아키텍처가 페이지 구조를 바꾼다

여기서 앞 파일들과 연결된다. **모듈마다 페이지에 담기는 것이 다르다.**

| 아키텍처 | 페이지에 담기는 것 |
|---|---|
| MHA / GQA | K와 V, 각각 `n_kv × d_h` |
| **MLA** (`01` 1.4) | latent `c` 하나 + 공유 `k_R` — **구조가 다르다** |
| **SWA** (`01` 1.6) | 원형 버퍼. 오래된 페이지를 회수 |
| **희소 계열** (`01` 1.7~1.9) | 저장은 그대로지만 **읽기가 흩어진다** |
| **하이브리드** (`02` 2.5) | **층마다 다르다** — linear 층은 KV가 아예 없다 |
| **CLA / YOCO** (`01` 1.10) | 레이어 그룹이 페이지를 공유 |

마지막 두 줄이 까다롭다. 모든 층이 같은 모양이라는 전제로 짜인 메모리 관리자는
하이브리드 모델을 그대로 다룰 수 없다. **아키텍처 변화가 서빙 시스템의 재작성을
요구하는 지점**이다.

---

## 10.3 prefix caching — 아예 계산하지 않기

### 관찰

실제 요청들을 보면 **앞부분이 겹치는 경우가 아주 많다.**

| 상황 | 겹치는 부분 |
|---|---|
| 같은 서비스의 요청들 | 시스템 프롬프트 |
| few-shot 프롬프팅 | 예시 전체 |
| **멀티턴 대화** | **이전 대화 전부** |
| 문서 QA | 문서 본문 |
| 에이전트 루프 | 지금까지의 도구 호출 기록 |

그리고 **겹치는 부분의 KV는 매번 똑같다.** 같은 토큰을 같은 순서로 넣으면
같은 K와 V가 나온다. 다시 계산할 이유가 없다.

### 아이디어

**계산한 prefix의 KV를 저장해두고 재사용한다.**

`10.2`의 페이지 공유가 그대로 쓰인다. 같은 prefix를 가진 요청들이 **같은 물리 페이지를
가리키게** 하면 된다.

SGLang의 **RadixAttention**은 여기에 자료구조를 붙였다. prefix들을 radix tree로 관리해
새 요청이 들어오면 **가장 긴 공통 prefix를 자동으로 찾아** 그만큼 건너뛴다.

```
              [시스템 프롬프트]
                     │
        ┌────────────┼────────────┐
    [대화 A]      [대화 B]     [대화 C]
        │            │
    [턴 2]        [턴 2]
        │
    [턴 3]  ← 새 요청은 여기까지 캐시 적중, 이후만 계산
```
> **그림 10.3** — radix tree로 관리되는 prefix 캐시

### 왜 "최고 레버리지"인가

이 위키의 다른 기법들은 전부 **계산이나 메모리 접근을 줄였다.**
prefix caching은 **아예 하지 않는다.**

멀티턴 대화를 생각해보면 극적이다. 10턴째 요청에서 앞 9턴은 이미 계산되어 있다.
prefill 비용이 거의 0이 된다. TTFT가 몇 배로 좋아진다.

> 💡 **에이전트 워크로드에서 특히 그렇다.** 도구를 호출하고 결과를 붙여
> 다시 모델에 넣는 루프는 **매번 앞부분이 전부 같다.**
> 캐시 없이 돌리면 같은 계산을 수십 번 반복하게 된다.

### 주의할 점

| 무엇 | 왜 |
|---|---|
| 캐시 메모리 | KV를 들고 있어야 하니 그만큼 배치 여유가 준다 |
| 축출 정책 | 무엇을 버릴지 (LRU 등) 정책이 성능을 좌우 |
| **위치 인코딩** | prefix가 같은 위치에서 시작해야 KV가 재사용 가능하다 |
| **하이브리드 모델** | ⚠️ linear 층은 KV가 아니라 상태를 들고 있어서 재사용 방식이 다르다 |

### 하이브리드 모델의 prefix caching

마지막 줄이 실제로 문제였다. Moonshot도 K3 공식 블로그에서 KDA가
**"기존 prefix caching에 새로운 난제를 던진다"** 고 직접 인정한다.

왜 어려운지는 `02-linear-attention` `2.1`에서 본 구조 때문이다.
KV cache는 **토큰마다 독립된 항목**이라 앞부분만 잘라내 재사용할 수 있다.
고정 상태는 **시퀀스 전체가 하나로 뭉쳐 있어서** 잘라낼 수가 없다.

해법은 **상태를 스냅샷으로 찍어두는 것**이다.

| 접근 | 무엇 |
|---|---|
| **Marconi** (arXiv:2411.19379) | recurrent state는 부분 prefix로 되돌릴 수 없어 exact-match entry가 필요하다는 제약을 반영하고, 재사용 가능성과 비용을 함께 본 admission·eviction 정책을 사용 |
| **하이브리드 cache manager** | attention KV와 recurrent state의 크기·hit 조건이 달라 layer type별 allocation·prefix 규칙이 필요 |

핵심은 **할당과 재사용 규칙이 다르다**는 점이다. attention KV는 토큰 block으로
나눌 수 있지만 recurrent state는 앞의 모든 토큰을 누적한 결과다. 따라서 full attention,
SWA와 recurrent layer의 cache hit를 교차해 판단하고, 서로 다른 state 크기를 함께
배치해야 한다. 구체적 지원 범위는 프레임워크 버전과 모델 조합에 따라 달라진다.

---

## 10.4 continuous batching과 chunked prefill

### continuous batching

전통적인 배칭은 **배치 전체가 끝날 때까지 기다린다.** 짧은 요청이 먼저 끝나도
자리를 비우지 않고 대기한다. 낭비가 크다.

continuous batching은 **매 iteration마다 스케줄링한다.** 끝난 요청은 즉시 빼고
대기 중인 요청을 그 자리에 넣는다.

```
 static batching              continuous batching

 A ████████░░░░░░              A ████████
 B ████░░░░░░░░░░              B ████ C ██████
 C 대기...........              D ██████████
                               (빈자리가 생기는 즉시 투입)
```

GPU가 노는 시간이 크게 줄어 처리량이 오른다. 지금은 사실상 모든 서빙
프레임워크의 기본 동작이다.

### chunked prefill

`10.1`에서 본 간섭 문제에 대한 답이다. **긴 prefill을 조각내서 decode 사이에 끼워넣는다.**

```
 섞지 않을 때:  ▓▓▓ ████████████████ ▓▓▓        decode가 오래 멈춘다
 chunked:       ▓▓▓ ██ ▓ ██ ▓ ██ ▓ ██ ▓▓▓        조금씩 나눠 처리
```

TTFT는 조금 늘지만 **TPOT가 안정된다.** 사용자 체감으로는 대개 이쪽이 낫다.

부수 효과가 하나 있다. prefill 조각과 decode 토큰이 한 배치에 섞이면서
**GEMV가 GEMM에 가까워진다.** `08-decoding`의 speculative decoding이 얻는 이득과
비슷한 성격이다.

---

## 10.5 PD disaggregation — 아예 분리하기

*Prefill-Decode Disaggregation*

### 아이디어

chunked prefill은 두 작업을 잘 섞는 방법이었다. 다른 답도 있다. **아예 분리한다.**

```
 ┌─────────────────┐         ┌─────────────────┐
 │  prefill 풀      │   KV    │  decode 풀       │
 │  연산 성능 중심   │ ──────► │  대역폭·용량 중심 │
 │  GPU × N         │  전송   │  GPU × M         │
 └─────────────────┘         └─────────────────┘
```
> **그림 10.5** — 두 단계를 별도 GPU 풀에서 돌린다.

얻는 것이 셋이다.

| 효과 | 설명 |
|---|---|
| **간섭 제거** | prefill이 decode를 막지 않는다 |
| **자원 비율 독립 조정** | 워크로드에 따라 prefill:decode 노드 비율을 바꿀 수 있다 |
| **하드웨어 이질화** | prefill은 연산이 센 GPU, decode는 대역폭·용량이 큰 GPU |

세 번째가 흥미롭다. `00-foundations` `0.8`에서 봤듯 세대가 지날수록
연산 성능이 대역폭보다 빨리 는다. **prefill과 decode가 원하는 하드웨어가 갈라지고 있다는 뜻**이고,
분리해두면 각각에 맞는 것을 붙일 수 있다.

### 대가 — KV를 옮겨야 한다

prefill 노드가 만든 KV를 decode 노드로 보내야 한다.
컨텍스트가 길면 이게 수 GB다. 전송 계층(NIXL 등)이 필요하고,
전송 시간이 TTFT에 더해진다.

그래서 **KV가 작을수록 PD 분리가 쉬워진다.** `01-attention`의 압축 계보와
`09-numerics`의 KV 양자화가 여기서 다시 값을 한다.

### prefix caching과의 결합

둘을 함께 쓰면 자연스러운 라우팅이 나온다.

```
 새 프롬프트        → prefill 노드로 (KV를 계산해 캐시에 씀)
 이어지는 요청      → decode 노드로 직행 (캐시 적중, prefill 생략)
```

에이전트나 멀티턴 워크로드에서 특히 잘 맞는다.

---

## 10.6 병렬화 — 무엇을 쪼개고 무엇을 주고받나

큰 모델은 한 GPU에 안 들어간다. 쪼개는 방식이 여럿이고, 각각 **무엇을 쪼개고
무엇을 통신하는지**가 다르다.

| 방식 | 쪼개는 것 | 통신 | decode 적합성 |
|---|---|---|---|
| **TP** (텐서) | **너비** — 헤드, FFN 차원 | 레이어마다 all-reduce | **좋음.** 지연을 실제로 줄인다 |
| **PP** (파이프라인) | **깊이** — 레이어 | 스테이지 간 P2P | 버블. **작은 배치에서 비효율** |
| **EP** (expert) | **expert** | **all-to-all** | MoE에 필수. 통신이 무겁다 |
| **CP/SP** (컨텍스트) | **시퀀스** | ring 형태 교환 | 긴 컨텍스트에 필요 |
| **DP** (데이터) | **요청** | 없음 (모델 복제) | 처리량. 메모리를 그만큼 씀 |

### 아키텍처가 거는 제약

앞 파일들에서 하나씩 나왔던 제약이 여기서 모인다.

| 제약 | 출처 |
|---|---|
| **TP degree ≤ `n_kv`** — 아니면 KV를 복제해야 한다 | `01` 1.2~1.3 (MQA·GQA) |
| MLA는 latent에 헤드 차원이 없어 **TP 분할 방식이 다르다** ⚠️ | `01` 1.4 |
| **레이어 간 KV 공유는 PP를 제약한다** | `01` 1.10 (CLA·YOCO) |
| **하이브리드는 층마다 자원 프로파일이 달라 균형이 안 맞는다** | `02` 2.5 |
| MoE는 **EP가 사실상 필수**이고 all-to-all이 따라온다 | `05` 5.6 |
| **깊이는 PP로 쪼개도 지연이 안 줄어든다** | `07` 7.1 |

첫 줄이 실무에서 자주 걸린다. `n_kv`=8인 모델은 TP를 8보다 크게 못 쓴다.
더 쪼개려면 KV를 복제해야 하고, 그러면 GQA로 아낀 메모리가 도로 늘어난다.
**아키텍처 하이퍼파라미터 하나가 배포 가능한 GPU 구성을 제한한다.**

### 조합해서 쓴다

실제로는 여러 방식을 겹친다. 예를 들어

```
 노드 안:  TP 8   (NVLink로 빠르게 all-reduce)
 노드 간:  EP     (expert를 노드에 분산)
 그 위에:  DP     (전체를 복제해 처리량 확보)
```

`05-moe` `5.6`에서 본 것처럼 **노드를 넘는 통신이 노드 안보다 훨씬 느리므로**,
통신이 잦은 TP는 노드 안에, EP는 노드 간에 두는 식으로 배치한다.

이는 고정 규칙이 아니다. 실제 배치는 NVLink·fabric 토폴로지, expert 크기와
collective 구현을 측정해 정한다. 핵심은 통신 빈도가 높은 병렬 축을 더 빠른 링크의
범위 안에 두는 것이다.

---

## 10.7 종합 — 무엇이 먼저 터지는가

이 위키 전체를 실무 관점에서 요약하면 이 표가 된다.

| 조건 | 먼저 터지는 것 | 대응 (파일) |
|---|---|---|
| 짧은 컨텍스트, **작은 배치** | 가중치 읽기 (대역폭) | 배치 키우기 · 양자화(`09`) · speculative(`08`) |
| 짧은 컨텍스트, **큰 배치** | 연산 (compute) | MoE(`05`) · 저정밀 연산(`09`) |
| 긴 컨텍스트, 작은 배치 | **KV 읽기** | KV 압축·희소(`01`) · linear(`02`) |
| **긴 컨텍스트, 큰 배치** | **KV 용량 — 아예 못 돌린다** | KV 압축(`01`) · KV 양자화(`09`) · PagedAttention(`10.2`) |
| MoE + 여러 노드 | **all-to-all 통신** | EP 배치 최적화 · 통신 중첩(`05` 5.6) |
| 층이 깊은 모델, 저지연 요구 | **직렬 지연** | 층수 줄이기(`07`) · speculative(`08`) |
| 멀티턴 · 에이전트 | prefill 반복 | **prefix caching(`10.3`)** |
| prefill과 decode 혼재 | 상호 간섭 | chunked prefill(`10.4`) · PD 분리(`10.5`) |
| 하이브리드 모델 | 층별 불균형 | 스케줄링 — **아직 정리 중인 영역** |

### 읽는 법

이 표가 말하는 것은 **"어느 기법이 좋은가"에 보편적인 답이 없다**는 것이다.

- 배치가 작으면 speculative decoding이 크게 이득이지만, 배치가 크면 이득이 줄어든다 (`08` 8.4)
- fine-grained MoE는 큰 배치에서 유리하고 작은 배치에서 불리하다 (`05` 5.4)
- 깊은 모델은 품질에 유리하고 지연에 불리하다 (`07` 7.1)

> **아키텍처를 평가하려면 서빙 시나리오를 먼저 정해야 한다.**
> 배치 크기, 컨텍스트 길이, 지연 요구, GPU 구성 — 이것이 정해지지 않으면
> "이 구조가 좋다"는 말은 성립하지 않는다.

`99-landscape.md`는 이 관점으로 실제 모델들의 선택을 정리한다.

---

## 이 다음

`99-landscape.md`는 축별 설명을 다시 모델별·시스템별로 가로질러 읽는다.
Efficient Transformer의 전체 계보와 각 선택이 랙·메모리·네트워크에 남기는 요구사항을
한자리에서 비교한다.

---

## Sources

**T1 — 논문**
- Kwon et al. (2023), *Efficient Memory Management for LLM Serving with PagedAttention*, arXiv:2309.06180
- Zheng et al. (2023), *SGLang / RadixAttention*, arXiv:2312.07104
- Yu et al. (2022), *Orca* — continuous batching(iteration-level scheduling)
- Agrawal et al. (2023), *Sarathi* — chunked prefill
- Zhong et al. (2024), *DistServe* — prefill/decode 분리
- Patel et al. (2023), *Splitwise* — phase 분리와 하드웨어 이질화
- Shoeybi et al. (2019), *Megatron-LM* — 텐서 병렬
- Liu et al. (2023), *Ring Attention* — 컨텍스트 병렬
- Pan et al. (2024), *Marconi: Prefix Caching for the Era of Hybrid LLMs*, arXiv:2411.19379

**T2 — 구현**
- vLLM 문서 — PagedAttention, continuous batching, chunked prefill, prefix caching
- SGLang 문서 — RadixAttention, PD 분리, DeepSeek 계열 최적화
- **llm-d / NIXL** — KV 전송 계층, 분산 서빙 스택
- DeepEP — MoE all-to-all
- NVIDIA NVLink / NVSwitch 문서 — 통신 대역폭

**범위와 주의**
- `10.7`의 병목 표는 정성적 진단 순서다. 전환점은 모델, batch, context와 하드웨어를
  profile해서 정해야 한다.
- sparse attention은 읽는 KV 수를 줄여도 gather 지역성이 나쁘면 대역폭을 충분히 쓰지
  못할 수 있다. 이 효과는 커널과 선택 패턴에 따라 측정해야 한다.
- 프레임워크 기능 지원은 빠르게 바뀌므로 배포 시점의 공식 문서를 확인한다.
