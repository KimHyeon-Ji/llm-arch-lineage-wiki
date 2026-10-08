# 01-attention 세미나 — Codex 제안

> 원본: `01-attention.md`  
> 결과물: Confluence 한 페이지  
> 형식: 페이지를 아래로 스크롤하며 발표  
> 목표: 본편 **약 34분** + 질의응답  
> 상태: 세미나 페이지·발표 노트·그림 초안 작성, 리허설 전 검수 중

## 1. 발표의 중심 질문

발표를 끝까지 끌고 갈 질문은 하나다.

> **한 토큰을 만들 때, KV cache에서 무엇을 얼마나 저장하고 무엇을 얼마나 읽는가?**

각 기법을 새로운 구조로 따로 외우게 하지 않고, 아래 세 축 중 무엇을 바꾸는지로 설명한다.

```
KV cache = [layer, position, stored width]
              L       S         C

저장량       ~ L × S × C
비싼 읽기량  ~ L × R × C

C : 토큰 하나를 얼마나 얇게 저장하는가
R : 비싼 attention이 실제로 몇 position을 읽는가
L : 몇 layer가 별도 cache를 갖는가
```

- **Compressed Attention**: `C`를 줄인다.
- **Sparse Attention**: `R`을 줄인다.
- **Layer Sharing**: `L`을 줄인다. 이번 발표에서는 큰 그림에서만 언급한다.

각 기법에서 반복해서 답할 질문도 세 개뿐이다.

1. `C`가 줄었는가?
2. `R`이 줄었는가?
3. 그 대가로 어떤 계산·정렬·메모리 접근이 생겼는가?

## 2. 전체 스토리

```text
Attention 한 스텝
  → 입력과 출력은 한 토큰인데 왜 과거 S개를 읽는가

KV cache
  → context 길이에 비례한다
  → 요청마다 별도로 생긴다
  → 현재 KV 한 줄을 쓰고, 과거 KV 전체를 반복해서 읽는다

저장 폭 C를 줄이기
  → MHA → MQA → GQA → MLA

읽는 위치 R을 줄이기
  → SWA → DSA → CSA/HCA → QSA

마무리
  → 얼마나 얇게 저장할까
  → 몇 개를 읽을까
  → 고르는 비용을 어떻게 낼까
```

## 3. 34분 러닝타임

| 구간 | 분 | 청중이 가져갈 문장 | 보여줄 것 |
|---|---:|---|---|
| 0. 훅 | 1 | 가중치는 함께 쓰지만 KV cache는 요청마다 따로 생긴다 | 긴 요청 여러 개가 동시에 살아 있는 그림 |
| 1. Attention 한 스텝 | 4 | 입력과 출력은 한 토큰인데 중간에서 길이 `S`인 K와 V를 읽는다 | 단일 head tensor shape |
| 2. KV cache가 왜 문제인가 | 4 | `S`에 비례하고 요청별로 복제된다. 쓰기는 한 줄, 읽기는 매번 전체다 | cache shape와 저장량 식 |
| 3. 큰 그림 | 1.5 | 줄일 축은 `C`, `R`, `L`이다 | 전체 지도 |
| 4. MHA → MQA → GQA | 3.5 | query head는 유지하고 KV head의 공유 범위를 넓힌다 | `n_kv`: 128 → 1 → 8 예시 |
| 5. **MLA** | **5.5** | latent를 저장하고 cache를 펴는 대신 query 하나를 옮긴다 | naive 복원 vs absorption |
| 6. SWA | 1.5 | 최근 `W`개만 연속해서 읽는 가장 단순한 sparse 방식이다 | full mask → window mask |
| 7. **DSA** | **4.5** | 싼 `O(S)`로 찾고 비싼 attention은 top-k에만 한다 | 넓고 얕게 → 좁고 깊게 |
| 8. **CSA + HCA** | **4** | DSA의 검색 비용을 줄이기 위해 먼저 position 축을 묶는다 | 두 방식을 2열로 대비 |
| 9. **QSA** | **2.5** | Qwen은 고정 micro-block과 block 단위 접근으로 답했다 | CSA와 QSA 비교 |
| 10. 회수 | 2 | 모든 기법은 `C`, `R`, `L` 중 하나를 줄이고 다른 비용을 낸다 | 최종 비교표 |
| **합계** | **34** | 질의응답 20분 이상 확보 | |

중간에 별도의 토론 시간을 배치하지 않는다. 짧은 질문만 그 자리에서 받고 본 토론은
마지막에 진행한다.

## 4. Confluence 페이지의 14개 정지 화면

여기서 화면은 슬라이드가 아니라, 스크롤을 멈추고 한 메시지를 설명할 수 있는 한 viewport다.

1. **왜 KV cache인가** — 긴 context × 동시 요청
2. **decode 한 스텝** — `q @ Kᵀ`, `p @ V`
3. **cache shape** — `[L, S, n_kv, d_h, K/V]`
4. **쓰기와 읽기** — 현재 `k,v` 한 줄 append / 과거 `K,V`의 `S`줄 scan
5. **세 개의 손잡이** — `C`, `R`, `L`
6. **MHA·MQA·GQA 한 화면** — 세 개를 독립 섹션으로 만들지 않는다
7. **MLA ① 저장** — `K,V` 대신 `c_KV + k_R`
8. **MLA ② 흡수** — cache `S`개를 펴지 않고 query 하나를 옮긴다
9. **Full → SWA → learned sparse** — 고정 범위에서 내용 기반 선택으로
10. **DSA** — indexer → top-k → gather → MLA attention
11. **DSA가 남긴 청구서** — 전체 scan, top-k, 불규칙 gather
12. **CSA/HCA** — `S/4`에서 고르기 vs `S/128`을 전부 읽기
13. **QSA** — 학습 압축 대신 고정 micro-block
14. **최종 지도** — 저장 폭 / 읽는 위치 / 새로 생긴 비용

원고를 쓴 뒤 정지 화면이 14개보다 많아지면 설명을 추가한 것이 아니라 사실상
슬라이드를 추가한 것으로 보고 합치거나 뺀다.

## 5. 도입부 — Attention과 KV cache

### 5.1 Attention은 부드러운 검색이다

shape를 보여주기 전에 다음 네 줄로 의미를 잡는다.

- `q`: 지금 찾고 싶은 정보
- `K`: 각 토큰이 내건 색인
- `V`: 선택됐을 때 가져올 실제 내용
- `q @ Kᵀ`로 고르고 `p @ V`로 내용을 섞는다

이후 head 하나의 decode 연산만 보여준다.

```text
현재 토큰
q_t                 (1 × d_h)
k_t, v_t            (1 × d_h)  ── append ──┐
                                            ▼
과거 cache
K, V                 (S × d_h)

score = q_t @ Kᵀ     (1 × S)    ← K의 S줄을 읽음
out   = softmax(score) @ V
                      (1 × d_h)  ← V의 S줄을 읽음
```

### 5.2 실제 cache shape

직관을 만든 뒤 batch와 layer 축을 한 번만 복원한다.

```text
KV cache per request = 2 × L × S × n_kv × d_h × bytes
                       └K,V┘
```

강조할 내용:

- context가 길어지면 `S`에 비례해 커진다.
- 가중치는 요청들이 공유하지만 cache는 각 요청의 대화 기록이라 공유할 수 없다.
- 실제 serving에서는 단순 batch 숫자보다 **동시에 살아 있는 요청의 전체 token 수**가 중요하다.
- decode에서는 새 KV 한 줄을 쓰고, 다음 토큰마다 과거 KV를 다시 읽는다.

### 5.3 쓰기와 읽기

| | 쓰기 | 읽기 |
|---|---|---|
| 하는 일 | 현재 토큰의 `k,v`를 만들어 append | `q @ Kᵀ`, `p @ V` |
| 범위 | 현재 토큰 한 줄 | 과거 cache의 `S`줄 |
| 주로 줄이는 계열 | Compressed | Sparse |

Compressed 방식은 저장량뿐 아니라 cache에서 읽어 오는 byte 수도 함께 줄인다.
Sparse 방식은 그중 실제로 읽을 position의 수를 줄인다.

## 6. 저장 폭을 줄이는 계보

### 6.1 MHA → MQA → GQA는 한 화면

MHA를 설명하기 전 다음 한 문장으로 여러 head가 필요한 이유를 짚는다.

> 한 head는 하나의 softmax 분포를 만든다. 서로 다른 관계를 동시에 찾으려면 여러 관점이 필요하다.

그 뒤 질문을 바꾼다.

> query의 관점은 여러 개 필요해도 K와 V까지 꼭 head마다 달라야 할까?

| | query head | KV head | 바뀐 것 | 한계 |
|---|---:|---:|---|---|
| MHA | 128 | 128 | head마다 별도 K,V | cache가 가장 큼 |
| MQA | 128 | 1 | 모든 query head가 K,V 하나를 공유 | 공유가 너무 강함 |
| GQA | 128 | 8 예시 | 그룹마다 K,V 하나 | 공유와 표현력의 절충 |

“계산이 그대로다”라고 단정하지 않는다. 다음처럼 말한다.

> query head 수와 attention 출력 수는 유지되지만 KV projection·저장량·메모리 읽기는 줄어든다.

전환 문장:

> GQA는 KV head의 **개수**를 줄였다. 개수를 1보다 작게 만들 수는 없으니,
> 이제 저장하는 **값 자체**를 줄여보자.

### 6.2 MLA는 두 장면만 깊게

남길 핵심은 두 개다.

1. **저장**: K와 V 대신 `c_KV(512) + k_R(64)`를 저장한다.
2. **사용**: `S`개의 latent를 K,V로 복원하지 않고 query 하나를 latent 공간으로 옮긴다.

```text
naive       : c_KV(S개) ── 복원 ──► K,V(S개) ── attention
absorption  : q(1개)    ── 변환 ──► q̃         ── c_KV와 attention
```

흡수는 아래 한 줄로 설명한다.

```text
q · (c Wᵀ) = (q W) · c
```

말로는 이렇게 읽는다.

> 왼쪽은 cache `S`개를 옮기고, 오른쪽은 현재 query `1`개를 옮긴다.

RoPE 설명은 다음 수준에서 멈춘다.

> 위치마다 다른 회전이 projection 사이에 끼면 흡수할 수 없어서, 위치 정보용 작은 키
> `k_R`을 별도 저장한다. `64`차원의 RoPE 세금이다.

Q 압축, TP, 실제 kernel 구현, projection 결합 논쟁은 질의응답용으로 내린다.

MLA의 마지막 문장:

> **저장 폭 `C`는 크게 줄었지만 읽는 위치 `R`은 여전히 `S`다.**

## 7. 읽는 위치를 줄이는 계보

Sparse 파트 전체의 딜레마를 먼저 제시한다.

> 적게 읽으려면 중요한 위치를 골라야 한다. 그런데 고르려고 전체를 비싸게 훑으면
> 절약이 사라지고, 고른 위치가 흩어지면 GPU가 느려진다.

### 7.1 SWA — 고르는 비용은 없지만 멀리 못 본다

- 최근 `W`개라는 고정 규칙이라 search·sort·gather가 없다.
- 연속 구간이라 구현이 단순하고 cache도 `W`에서 상한이 생긴다.
- 대신 한 층에서는 window 밖 정보에 직접 접근할 수 없다.
- attention sink는 “앞 토큰 몇 개를 예외로 남긴다” 한 문장만 둔다.

전환 문장:

> 최근 것이 항상 중요한 것은 아니다. 위치가 아니라 내용을 보고 고르게 하면 어떨까?

### 7.2 DSA — 싸게 찾고 비싸게 본다

```text
선택 경로: 별도 Indexer K cache S개 → 값싼 점수화 → top-k indices
본 경로  : MLA latent cache S개 → indices로 gather → k개만 Sparse MLA
```

핵심:

- 비싼 attention의 읽기 범위가 `S → k`로 줄어든다.
- indexer는 전체를 보지만 head 수와 정밀도를 낮춰 싸게 만든다.
- indexer는 **MLA latent를 그대로 재사용하지 않는다.** 자체 projection으로 만든
  128차원 key를 별도 cache에 저장한다.
- DSA는 MLA cache 자체를 줄이는 기법이 아니라 **MLA 위에 선택기를 붙인 것**이다.

반드시 함께 말할 대가:

1. 싼 `O(S)` scan은 남는다.
2. top-k 정렬 비용이 생긴다.
3. 선택된 token이 흩어져 있어 gather가 비연속적이다.

이 세 가지가 CSA·HCA·QSA의 출발점이다.

### 7.3 CSA와 HCA — 한 쌍으로 설명

둘을 별도 장으로 늘리지 않고 DeepSeek-V4의 상보적인 두 layer로 비교한다.

| | CSA | HCA |
|---|---|---|
| 먼저 | 4개씩 학습 압축: `S → S/4` | 128개씩 고압축: `S → S/128` |
| 다음 | indexer로 top-k 선택 | 압축 결과를 전부 dense attention |
| 얻는 것 | 일부를 더 정밀하게 봄 | 전체 범위를 빠짐없이 봄 |
| 치르는 값 | sort·gather가 남음 | 압축이 거칠어 세부 정보 손실 |
| 실제 배치 | **CSA/HCA layer를 1:1로 교대** | |

```text
CSA : 1,000,000 → 250,000 → top-1,024     좁고 정밀
HCA : 1,000,000 → 약 7.8천 압축 엔트리 → 전부 dense   넓고 거침
```

- CSA는 DSA를 압축 엔트리 위에 적용한다.
- HCA는 강하게 압축한 대신 선택·정렬·gather 자체를 없앤다.
- 두 방식 모두 최근 raw token 128개를 보는 local sliding-window branch를 함께 둔다.
- V4의 core attention은 1개 KV head(`head_dim=512`)다. **MLA 위에 CSA/HCA를 얹었다고
  설명하지 않는다.** 계보상 DSA의 indexer를 잇지만, 저장 구조는 V3.2 MLA와 다르다.
- 학습 압축의 세부 행렬과 config의 layer 목록은 본문에서 뺀다.

### 7.4 QSA — 같은 질문의 다른 답

QSA는 CSA와 바로 비교한다.

| 질문 | CSA | QSA |
|---|---|---|
| 무엇을 묶나 | KV 자체를 학습 압축 | 검색용 index key를 4-token 평균 |
| 무엇을 고르나 | 압축 KV 엔트리 | micro-block 512개 |
| 본 attention의 값 | **압축 KV** | 선택된 block의 **원본 GQA KV** |
| 메모리 접근 | 선택된 엔트리 gather | 연속된 4-token block 단위 gather |
| 무엇과 짝짓나 | HCA layer | Gated DeltaNet layer |

QSA에서 Gated DeltaNet 내부로 들어가지 않는다.

> 48층 중 36층은 고정 크기 상태를 갱신하고, 12층만 QSA로 정밀 검색한다.

Qwen3.8-Flash-Next의 공개 config에서는 `indexer_compress_ratio=4`,
`indexer_budget=2048`이다. 즉 압축 index key 중 상위 **512 block**을 고르고,
이를 원래 token 위치로 펼쳐 최대 2,048개 token의 GQA KV를 읽는다.

여기까지만 말하고, QSA의 본론을 다음으로 고정한다.

> token 하나씩 고르는 대신 연속된 block을 고르면 실제 하드웨어에서 읽기 좋다.

## 8. 마지막 비교표

| 기법 | 저장 폭 `C` | 비싼 읽기 위치 `R` | 새로 치르는 값 |
|---|---|---|---|
| MHA | 큼 | `S` | 기준점 |
| MQA/GQA | 작아짐 | `S` | KV 공유에 따른 표현력 절충 |
| MLA | 매우 작음 | `S` | query/output 쪽 projection, 전용 kernel |
| SWA | `W`에서 상한 | `W` | window 밖 직접 접근 불가 |
| DSA | MLA cache + 별도 index K | `k` | 싼 `S` scan + sort + gather |
| CSA | KV를 position 방향 `S/4`로 압축 | 압축 KV top-k | 학습 압축 + sort + gather |
| HCA | position을 `S/128`로 압축 | 압축 결과 전부 | 세부 정보 손실 |
| QSA | 12개 sparse layer의 GQA KV + 압축 index | 선택된 raw KV block | block 선택 + GDN과 혼합 |

마지막 멘트:

> MQA부터 QSA까지 이름은 많지만 질문은 세 개뿐입니다.  
> **한 토큰을 얼마나 얇게 저장할까, 그중 몇 개를 읽을까, 고르는 비용은 어떻게 낼까.**

## 9. 발표 본문에서 뺄 것

- Gated Attention
- NSA, MoBA
- CLA, YOCO, IndexShare 등 Layer Sharing의 개별 설명
- MHA/MQA/GQA 각각의 역사와 모델 목록
- 모든 기법의 전체 연산 표
- TP·DP 병렬화와 kernel 이름 비교
- MLA의 Q 압축 상세와 모든 projection shape
- CSA/HCA의 config 원문과 학습 압축 행렬 이름
- QSA 구간의 Gated DeltaNet 내부 수식
- 발표 도중의 별도 토론 구간

필요하면 삭제하지 않고 Confluence 맨 아래 **「더 궁금하면」 접기 영역**이나 발표자 노트로
내린다. 세미나 본문은 원리를 이해시키고, `01-attention.md`가 세부 참고서 역할을 맡는다.

## 10. 표현 규칙

- 한 화면에는 질문 하나와 답 하나만 둔다.
- 표는 4열 이하로 유지한다.
- tensor shape는 도입부에서 완전한 형태를 한 번만 보여준다.
- 이후에는 바뀐 축만 강조한다: `C`, `S → k`, `S → S/4` 등.
- 식보다 값의 개수와 화살표를 우선한다.
- 장난감 숫자와 실제 모델 수치를 명확히 구분한다.
- 화면이 모든 말을 대신하지 않게 설명 문장은 발표자 노트로 내린다.
- ASCII 그림은 Confluence 코드 블록 안에 넣는다.

## 11. 이미지 제작안

이미지는 AI 이미지 생성이 아니라 **직접 작성한 SVG**로 만든다. 이유는 세 가지다.

- tensor shape와 연결 관계를 논문·config에 맞춰 정확히 통제할 수 있다.
- 같은 palette와 typography를 유지하면서 문구·수치를 쉽게 수정할 수 있다.
- Confluence에는 PNG를 올리고, 원본 SVG는 수정 가능한 source로 보관할 수 있다.

공통 palette:

- query·주요 흐름: 보라 `#6C52D9`
- key·선택 경로: 분홍 `#C64A9B`
- value·본 attention: 초록 `#4B9B63`
- trade-off·주의: 주황 `#F2A444`
- 비활성 영역: 회색 `#E5E8F0`

| 파일 | 메시지 | 검증 기준 |
|---|---|---|
| `01-decode-attention` | query 1개가 K·V의 S줄을 읽음 | scaled dot-product attention shape |
| `02-kv-cache-write-read` | 요청별 cache와 한 줄 write / S줄 read | cache 식 |
| `03-three-levers` | `C`, `R`, `L` 세 축 | 발표용 분류 |
| `04-mha-mqa-gqa` | Q head 유지, KV head만 8→2→1 | MQA·GQA 논문 |
| `05-mla-cache` | 32,768 값 vs 576 값 | DeepSeek-V3 config |
| `06-mla-absorption` | S개 복원 vs query 1개 이동 | DeepSeek-V2 §2.1 |
| `07-swa` | full `S` → window `W` | Mistral·StreamingLLM |
| `08-dsa` | 별도 index cache → top-2048 → Sparse MLA | V3.2 코드·config |
| `09-csa-hca` | `S/4` top-k vs `S/128` dense + local 128 | V4 논문·config |
| `10-qsa` | index key 4:1 압축 → 512 block → raw GQA KV | Qwen 논문·config |

원본은 `seminar/assets/codex/*.svg`, Confluence 업로드용은 같은 이름의 `.png`다.

이미지에 관한 선택지는 사용자와 함께 조정한다.

- 현재는 AIBook처럼 밝은 배경과 얇은 선을 사용한다.
- 더 기술적인 느낌이 필요하면 tensor 축 표기를 늘린다.
- 발표 가독성이 더 중요하면 세부 config 숫자를 캡션으로 내려 보낸다.
- 회사별 색을 쓰지 않고 **역할별 색**을 고정한다. 같은 K는 언제나 분홍, V는 초록이다.

## 12. 제작 순서

1. `C/R/L` 기준 그림과 마지막 비교표를 먼저 만든다.
2. MLA 두 화면과 DSA 두 화면을 만든다. 이 네 화면을 설명 밀도의 기준으로 삼는다.
3. CSA/HCA 한 화면과 QSA 한 화면을 붙인다.
4. MHA/MQA/GQA 한 화면과 SWA 한 화면을 채운다.
5. Attention 연산과 KV cache 도입부를 작성한다.
6. 처음부터 끝까지 소리 내어 읽으며 **30분을 목표로** 자른다.
7. 현장에서는 스크롤·짧은 질문·예시가 남은 4분을 사용하도록 한다.

## 13. 참고 자료

- `01-attention.md` — 발표 내용의 기준 원본
- `seminar/PLAN.md` — 기존 통합 계획
- [AIBook — 어텐션](https://aibook.euiyun.com/chapters/attention.html) — 설명과 시각화 참고용
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — MLA와 absorption
- [DeepSeek-V3.2 구현](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) — DSA
- [DeepSeek-V4](https://arxiv.org/abs/2606.19348) — CSA/HCA
- [Qwen3.8-Next](https://arxiv.org/abs/2608.30320) — QSA와 GDN hybrid

AIBook은 구성에 직접 편입하지 않고 다음 표현을 다듬을 때만 참고한다.

- attention을 “부드러운 검색”으로 소개하는 방식
- MHA/MQA/GQA의 query head와 KV head 비교 그림
- 문맥 길이에 따른 KV cache 메모리 시각화
- MLA absorption 전후의 대비

## 14. 그림 재설계 — `seminar/figures`에서 배운 점

기존 Codex 그림은 개념을 *명명*하는 데 치우쳐 발표자가 “그래서 어떤 텐서가 얼마나 생기고 읽히는가”를 다시 말로 채워야 했다. `seminar/figures/make_figures.py`의 장점은 색 자체보다 **셀의 행·열, 길이, 선택된 위치와 연결선으로 실제 데이터의 모양을 보여준다는 점**이다. 그래서 `seminar/assets/codex/make_figures.py`를 새로 만들고 기존 10장 SVG/PNG를 다시 생성했다. 클로드의 파일은 수정하지 않는다.

| 그림 | 이제 시각화할 실제 구조 | 발표할 때 짚을 곳 |
|---|---|---|
| 01 decode | `1×d_h · d_h×S → 1×S`, 이어서 `1×S · S×d_h → 1×d_h` | 두 번 모두 같은 `S`축을 읽는다 |
| 02 KV cache | 길이가 다른 요청 3개의 별도 K/V 행렬과 끝에 붙는 새 행 | 가중치는 공유해도 cache는 요청별이다 |
| 03 비용 축 | 저장 너비 `C`와 실제로 읽는 행 수 `R`를 같은 셀 눈금으로 비교 | `C`를 줄이는 것과 `R`을 줄이는 것은 다르다 |
| 04 MHA/GQA/MQA | Q 8개에서 KV 8/2/1개로 연결, 각 KV 밑에 `S`행을 쌓음 | head 공유가 `S`행 전체의 복제 수를 줄인다 |
| 05 MLA cache | `32,768` 대 `576`을 값 수에 비례하는 길이로 표현 | 앞의 MHA는 동일 head 수를 둔 *가정 비교*다 |
| 06 MLA absorption | `S×512` 전체를 펴는 경로와 `H×128` query만 옮기는 경로 | 결합법칙은 content 부분에 적용한다 |
| 07 SWA | 10×10 causal mask와 `W=4` band | 각 행에서 보이는 열이 최근 4개뿐이다 |
| 08 DSA | 별도 `S×128` index cache 전체 scan과 `S×576` MLA cache top-k gather | 같은 cache를 두 번 읽는 그림이 아니다 |
| 09 CSA/HCA | 4→1 압축+선택과 128→1 압축+dense 읽기, CSA의 직전 블록 겹침 | 압축 entry는 원본 token과 동일하지 않다 |
| 10 QSA | 4개의 index key 평균→block key, 선택 block의 원본 KV 4개 읽기 | 검색용 압축과 본 attention용 KV는 다르다 |

정확성 규칙:

1. 셀 개수는 이해를 위한 예시이며 실제 dimension을 뜻하지 않는다. 실제 수치가 필요한 곳에는 **shape 또는 config 값**을 적는다.
2. 비례 막대는 동일 단위(`values/token/layer`)끼리만 비교한다. dtype·quantization이 다르면 bytes 비교로 해석하지 않는다.
3. DSA는 indexer용 별도 key와 MLA latent를 구분한다. QSA는 평균하는 대상이 index key이며, 본 attention은 선택 block의 원본 GQA KV를 읽는다.
4. CSA의 학습된 4:1 압축을 단순 평균으로 그리지 않는다. HCA의 `S/128`은 길이에 따른 대략적인 크기로 말한다. 작은 문맥에서 정확한 cache entry 수를 나눗셈으로 단정하지 않는다.
5. PNG는 Confluence 첨부용, SVG와 생성 스크립트는 수정 가능한 원본이다. 실행: `python seminar/assets/codex/make_figures.py`.

리허설 때는 먼저 01·04·05·08·09·10 그림만으로 스토리가 이어지는지 확인한다. 02·03·06·07은 질문과 시간에 따라 펼쳐 보는 보조 화면으로 쓸 수 있다.

