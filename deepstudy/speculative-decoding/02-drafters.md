# 02. drafter — 초안을 누가, 어떻게 만드나

> 갈래 전체를 먼저 한눈에 보고, 무엇이 표준인지 본 다음, 갈래마다 **무엇을 바꿨는지** 짧게 따라간다.

좋은 drafter의 조건은 둘이다 — **잘 맞히고**(`α`가 높다), **싸다**(초안 시간이 짧다). 갈래마다 이 중 하나를 개선하고, 대개 다른 비용을 하나 새로 만든다.

| 순서 | 내용 | |
|---|---|---|
| 1 | 한눈에 보기 | 갈래 |
| 2 | 무엇이 표준인가 | **질문 1** |
| 3 | 별도 draft 모델 — 출발점 | 자세히 |
| 4 | EAGLE-3 — target이 읽은 것을 받는다 | 자세히 |
| 5 | MTP — 같은 모양을 본 학습 때 붙인다 | 자세히 |
| 6 | DFlash · DSpark — 블록을 한 번에 뽑는다 | 자세히 |
| 7 | Gemma 4 MTP · n-gram — 캐시나 모델을 없앤다 | 자세히 |
| 8 | 흐름 | **질문 2** |

---

## 1. 한눈에 보기

![갈래별 target과 draft 모듈](figures/22-drafter-anatomy.png)

- **target은 모두 같다** — 초안을 받아 한 번에 검증하는 forward 하나다
- **화살표가 draft가 받는 것이다** — 토큰만 / target이 읽은 결과인 **hidden** / target의 **KV**
- **파란 상자가 draft 모듈의 크기다** — 막대 하나가 층 하나다
- **아래 칩이 초안을 뽑는 방식이다** — 화살표로 이어지면 1개씩 `γ`번, 테두리로 묶이면 한 번에

### 워크로드로 옮기면

시뮬레이터가 갈래마다 셀 것은 **"draft가 한 번 도는 일 × 몇 번 + 캐시"** 다. target 쪽은 어느 갈래든 decode 연산 표를 `K`줄로 늘리면 된다 ([03](03-compute-and-kv.md#1-verify-한-번은-decode-한-번보다-얼마나-비쌀까)).

| 갈래 | draft가 한 번 도는 일 | 몇 번 | draft 캐시 | 얻는 것 | 치르는 것 |
|---|---|---|---|---|---|
| 별도 draft 모델 | 작은 LLM의 decode 한 스텝 | `γ` | 자기 KV, 전 층 | 아무 target에나 붙는다 | 자기 prefill · KV · 덜 맞는다 |
| EAGLE-3 | decoder 1층 + 작은 head | `γ` | 1층 | 문맥을 다시 읽지 않는다 | 사후 학습 |
| MTP | target 층 하나와 같은 블록 + 전체 head | `γ` | 1층 | 학습할 것이 없다 | 블록이 크다 (V3: 14B) |
| DFlash · DSpark | 층 `L`개 × 블록 `γ + 1`줄 + head `γ`줄 (DFlash · 논문 실험 5층, DSpark 서비스용 MoE 3층) | **1** | `L`층 (DSpark 서비스용은 최근 128줄) | 초안을 길게 뽑아도 시간이 같다 | 뒤 자리가 약하다 · 캐시가 크다 |
| Gemma 4 MTP | 4층 (target KV를 읽는다) + 작은 head | `γ` | **없음** | 캐시 · prefill이 없다 | 초안마다 target KV 읽기 |
| n-gram · suffix | 문자열 매칭 | — | 없음 | 비용이 거의 없다 | 베낄 수 있을 때만 맞는다 |

EAGLE-3와 MTP는 층 종류와 head 크기만 다른 **같은 코드**다 ([04](04-simulator.md#1-무엇을-기준으로-고르나--이름이-아니라-비용-구조로)).

### 지도로 보면

![drafter 지도](figures/05-drafter-map.png)

- **세로 — 어디에 사나.** 별도 모델 / target에 붙는 얇은 모듈 / 모델 없음
- **가로 — 초안을 어떻게 뽑나.** 토큰마다 한 번씩(순차) / 블록 전체를 한 번에(병렬)
- 가운데 보라 테두리 줄이 지금의 주류다. 2026년에 그 안에서 무게가 **왼쪽(순차)에서 오른쪽(병렬)으로** 옮겨 가고 있다

---

## 2. 무엇이 표준인가

**검증의 원리는 같고, 모양은 엔진 · 모델별로 다르다.** 원리(한 번에 검증하고 target의 분포를 지킨다)와 채택 규칙(토큰별 거부 샘플링)이 기본이다. 모양은 일렬과 트리가 공존한다 — vLLM의 기본 경로, SGLang의 새 엔진(Spec V2, `topk = 1`), DFlash · DSpark는 일렬이고, SGLang의 EAGLE 경로는 Llama · Grok 등에서 트리(기본 분기 4)를 쓴다. 트리는 남는 연산이 많은 작은 배치에서 유리하다 ([03](03-compute-and-kv.md#3-갈래가-여럿이면--트리-검증)).

**drafter는 셋이 공존한다.**

| | 위치 | 근거 |
|---|---|---|
| **내장 MTP** | 모델이 들고 나오면 일단 쓰는 기본값 | 학습할 것이 없다. 엔진 1급 지원. DeepSeek의 서비스 기준선(MTP-1) |
| **EAGLE-3** | MTP가 없는 모델의 표준 | 공개 체크포인트가 가장 넓다. SGLang 문서의 권장 |
| **DFlash 계열** | 공개 측정의 최전선 | z-lab 체크포인트 20개 이상, NVIDIA · LMSYS · vLLM 측정, DeepSeek가 MTP-1에서 DSpark로 |

- 공개된 실험들에서는 대체로 **EAGLE-3 < 내장 MTP ≲ DFlash < DSpark ≲ DFlash 2** 순이었다. 다만 다섯을 한 조건에서 잰 실험은 없고, GPU · 엔진이 바뀌면 순서도 바뀐다
- 근거는 엔진 지원 · 체크포인트 · 공개 측정이다. 실제 사용 비중을 보여 주는 자료는 아니다

<details>
<summary>자세히 — 엔진 지원 (2026년 10월, 공식 문서 기준)</summary>

| drafter | vLLM (`method`) | SGLang (`--speculative-algorithm`) |
|---|---|---|
| 별도 draft 모델 | `draft_model` | `STANDALONE` |
| EAGLE · EAGLE-3 | `eagle` · `eagle3` | `EAGLE` · `EAGLE3` (**문서의 권장**) |
| 내장 MTP | `mtp` | EAGLE 경로 (`NEXTN`은 `EAGLE`의 다른 이름) |
| Gemma 4 MTP | `mtp` + assistant 체크포인트 | 문서에는 없다. 코드에 `frozen_kv_mtp` worker가 있다 (이름으로 보아 target KV를 읽는 MTP용) |
| 병렬 draft 모델 (PARD) | `parallel_drafting` | — |
| DFlash | `dflash` | `DFLASH` |
| DSpark | 지원, 검증 길이 조절은 DSpark만 | 코드에 DSpark 구성 요소가 있다 |
| n-gram · suffix | `ngram` · `ngram_gpu` · `suffix` | `NGRAM` |

- vLLM의 drafter 기반 클래스는 일렬 초안(`[배치, γ]`)만 만든다. 트리 처리는 없고 FIXME 주석만 있다
- TensorRT-LLM은 draft 모델 · EAGLE-3 · MTP · n-gram을 지원하고, NVIDIA는 gpt-oss-120b에서 DFlash를 TensorRT-LLM으로 돌린 결과를 냈다. llama.cpp는 draft 모델과 MTP(`--spec-type mtp`)를 지원한다

</details>

<details>
<summary>자세히 — 공개 비교와 그 조건</summary>

| 비교 | 조건 | 결과 | 출처 |
|---|---|---|---|
| EAGLE-3 vs DFlash | Qwen3-8B, temp 0, 7개 벤치 평균 | 2.02× vs **4.86×** | DFlash 논문 |
| EAGLE-3 vs DFlash | gpt-oss-120b, TensorRT-LLM, B300 | 평균 1.7× vs **2.3×** | NVIDIA |
| EAGLE-3 vs DFlash | Llama-3.1-8B, 같은 조건 | 평균 2.2× vs **2.8×** | NVIDIA |
| 내장 MTP (7스텝) vs DFlash | Qwen3.5-397B-A17B, 8 × B200, SGLang | 동시 1 ~ 32 전 구간 DFlash 우세, 동시 1에서 MTP의 1.5배 | LMSYS |
| 내장 MTP vs DFlash | Qwen3.6-35B-A3B, AMD MI300X · MI355X, vLLM | 1.28 ~ 1.49× vs **1.77 ~ 2.06×** | vLLM 블로그 |
| Gemma 4 MTP vs DFlash vs EAGLE-3 | gemma-4-26B-A4B, AMD, vLLM | 최대 2.74× · **2.87×** · 2.11 ~ 2.27× | vLLM 블로그 |
| MTP-1 vs DSpark | DeepSeek-V4 서비스 | 사용자당 속도 **+57 ~ 85%** | DSpark 논문 |
| 반례 | Qwen3-8B, AMD, vLLM | EAGLE-3는 MATH-500에서 **기준보다 느림** (최대 0.88×), DSpark 1.15 ~ 1.63× | vLLM 블로그 |

속도 배수는 각 출처의 조건 안에서만 읽는다. 표의 줄끼리 직접 비교하지 않는다.

</details>

---

## 3. 별도 draft 모델 — 출발점

*2022 · Google (Leviathan et al.), DeepMind (Chen et al.)*

같은 토크나이저를 쓰는 작은 LLM을 옆에 두고, 그것이 `γ`번 decode해서 초안을 낸다.

- **문맥을 처음부터 다시 읽는다** — 자기 prefill, 자기 KV를 따로 둔다
- **target이 이해한 것을 쓰지 못한다** — 같은 크기라면 덜 맞는다
- 대형 서빙에서는 다음 절의 방식에 밀렸다. 로컬 추론(llama.cpp)과 엔진의 일반 경로에 남아 있다

<details>
<summary>자세히 — 변형</summary>

- vLLM은 토크나이저가 다른 draft도 받는다 (`use_heterogeneous_vocab`, greedy만)
- **PARD** (2025, AMD) — 작은 LLM에 마스크 토큰을 넣어 여러 개를 한 번에 뽑는다 (vLLM `parallel_drafting`). 초안은 6절의 DFlash처럼 한 번에 뽑지만, 문맥은 자기 KV를 자기 prefill로 만든다

</details>

---

## 4. EAGLE-3 — target이 읽은 것을 받는다

*2025 · Li et al. · EAGLE(2024) → EAGLE-2(2024) → EAGLE-3*

target은 `sat` 자리에서 `on`을 고를 때 이미 문맥 전체를 읽었다. 그 결과가 **hidden**(층을 지나며 만든 벡터 한 줄)이다. drafter가 이것을 받으면 문맥을 다시 읽지 않아도 되므로 **층 하나로 충분하다.**

그런데 hidden만으로는 모자란다.

![drafter가 hidden과 토큰을 둘 다 받는 이유](figures/18-hidden-and-token.png)

- `h_sat`은 `on`을 **고르기 전**의 상태다. 다음 칸에 `on`이 뽑혔는지 `down`이 뽑혔는지 모른다
- 그래서 **실제로 뽑힌 토큰의 임베딩**을 함께 받는다 — hidden은 "여기까지의 이해", 임베딩은 "그다음에 온 토큰"이다

01의 초안 4개는 이렇게 나온다.

| 스텝 | 입력 | 출력 |
|---|---|---|
| 1 | target의 `h_sat` + `Emb(on)` | `the` |
| 2 | **drafter 자기 hidden** + `Emb(the)` | `mat` |
| 3 | 자기 hidden + `Emb(mat)` | `and` |
| 4 | 자기 hidden + `Emb(and)` | `slept` |

💡 스텝 2부터는 target의 hidden이 없어 자기 근사를 넣는다. 그래서 뒤 스텝일수록 덜 맞는다.

**EAGLE-3가 더한 것**

![EAGLE-3와 MTP 모듈](figures/07-thin-ar-drafters.png)

- target의 **세 층**(낮은 · 중간 · 높은)에서 hidden을 받아 섞는다 — 마지막 층만 쓰는 것보다 정보가 많다
- 출력 head를 **어휘 3.2만 개짜리**로 작게 둔다 — drafter 한 스텝의 비용에서 head가 큰 몫이다
- 학습 중에 자기 출력을 다시 넣어 스텝 2 · 3 상황을 미리 연습한다

<details>
<summary>자세히 — 구조와 보고된 수치</summary>

- 섞은 hidden과 임베딩에 각각 RMSNorm을 걸고 이어 붙여(`2d`) decoder 층 하나의 Q · K · V가 바로 받는다. Llama-3.1-8B용 설정: 층 1개, `d` 4096, FFN 14336, `draft_vocab_size: 32000`
- target은 얼려 두고 drafter만 학습한다. EAGLE-1/2의 "hidden을 맞히는" 손실을 버리고 토큰만 맞힌다
- EAGLE-2는 확신도가 높은 갈래만 키우는 동적 트리를 도입했다 (깊이 6, 노드 60개 안팎). EAGLE-3는 깊이를 8로 늘렸다

📌 EAGLE-3 논문, temperature 0, 트리 초안

| target | 속도 | `τ` |
|---|---|---|
| Llama-3.1-8B | 4.44× (EAGLE-2 3.23×) | 6.23 |
| Llama-3.3-70B | 4.12× | 5.88 |

- 일렬 초안에서는 낮다 — 예: Kimi-K2.5, 초안 4개에서 MAL 3.96 (vLLM 블로그)
- 다른 팀이 학습한 EAGLE-3는 낮을 수 있다 — DFlash 논문의 Qwen3-8B용은 트리 60개에서 `τ` 3.40, 2.02×

</details>

---

## 5. MTP — 같은 모양을 본 학습 때 붙인다

*2024 · DeepSeek · Multi-Token Prediction*

DeepSeek-V3는 모델을 처음 학습할 때부터 "그다음 토큰"을 맞히는 작은 모듈을 붙였다. 목적은 학습 품질이지만, 추론 때 **그대로 drafter로** 쓸 수 있다. 위 그림의 오른쪽이 MTP 모듈이다 — EAGLE-3와 거의 같은 모양이고, 다른 점은 셋이다.

- hidden을 **마지막 층 하나**에서 받는다
- 층이 **본체 층과 같은 블록**이다 (V3: MLA + MoE). 그래서 모듈 하나가 14B, 초안 한 스텝이 본체 한 스텝의 약 4.4%다
- 출력 head를 **target과 함께 쓴다** (어휘 전체)

📌 DeepSeek-V3: 두 번째 토큰 채택률 85 ~ 90%, 생성 속도 1.8배. DeepSeek는 이것(초안 1개, 요청당 2토큰 검증)을 **MTP-1**이라 부르며 서비스 기준선으로 써 왔다.

**여러 개를 뽑으려면** 같은 블록을 다시 돈다. 학습 때 없던 쓰임이라 뒤 자리가 약하다 — vLLM-Ascend 문서는 DeepSeek MTP로 3개 이상 뽑으면 정확도를 보장할 수 없다고 적는다. FastMTP(2025)는 이 반복 사용에 맞춰 MTP 층을 따로 미세조정한다.

**MTP를 가진 공개 모델** — DeepSeek-V3 · V3.2 · V4, Qwen3-Next · Qwen3.5 · Qwen3.6, GLM-4.5 · GLM-5.x, ERNIE-4.5, Nemotron 3 Super · Ultra, MiMo.

<details>
<summary>자세히 — 식과 세부</summary>

```
h'_i  =  M · [ RMSNorm(h_i) ; RMSNorm(Emb(t_(i+1))) ]          M : 2d → d
h''_i =  TRM( h'_i )                                            본체와 같은 Transformer 블록 하나
P(t_(i+2)) = OutHead( h''_i )                                    본체의 출력 head를 함께 쓴다
```

- 깊이 `D` = 1. 손실 가중치는 앞 10T 토큰 동안 0.3, 나머지 4.8T 토큰 동안 0.1
- Qwen3.5 · 3.6은 MTP 층 하나(`mtp_num_hidden_layers: 1`)를 여러 스텝 쓴다. SGLang 블로그의 Qwen3.5-397B 비교는 7스텝으로 돌렸다
- 추론 비용은 든다 — 블록과 head의 연산, 가중치 메모리, drafter KV ([03](03-compute-and-kv.md#4-drafter도-캐시가-필요할까))

</details>

---

## 6. 초안을 하나씩 만들어야 할까? — 병렬 drafter

EAGLE-3도 MTP도 초안을 **하나씩** 만든다. 초안 시간이 초안 수에 비례하니 길게 뽑을 수 없고, 층도 하나로 묶어 둬야 한다.

![순차 초안과 병렬 초안](figures/19-serial-vs-parallel.png)

- 위(순차): 앞에서 뽑은 토큰이 다음 입력이라 네 번 돈다
- 아래(병렬): `on` 뒤에 빈칸 `[M]`(마스크 토큰)을 붙여 **한 번에** 채운다. 초안이 늘어도 시간이 거의 같다

### DFlash

*2026.02 · UC San Diego (z-lab)*

- 블록 16칸(맨 앞은 방금 확정된 토큰, 나머지 15칸은 마스크)을 **5층**에 한 번 통과시킨다. forward가 한 번뿐이라 층을 깊게 쌓을 수 있다
- target 정보를 **모든 층의 K · V로** 넣는다 (KV injection). 입력에만 넣는 EAGLE-3와 달리 어느 층에서든 target의 표현을 직접 읽는다

![KV injection과 세 세대](figures/08-parallel-drafters.png)

📌 Qwen3-8B, temp 0, 7개 벤치 평균: EAGLE-3(트리 60) **2.02× · `τ` 3.40** → DFlash(블록 16) **4.86× · `τ` 6.49**. `τ`가 높고(깊은 층 + 모든 층 주입) 사이클이 짧다(forward 1번).

**약점 — 뒤 자리일수록 덜 맞는다.** 빈칸을 동시에 채우니 앞 칸에 무엇이 뽑힐지 모르고 고른다. "New York"과 "San Francisco"가 반반이면 **"New Francisco"**가 나올 수 있다.

<details>
<summary>자세히 — DFlash의 식 · 학습 · 수치</summary>

```
H_ctx = RMSNorm( W_c · [ H^(l1) ; … ; H^(l5) ] )      target 5개 층의 hidden을 섞는다 (Qwen3-8B용: 1, 9, 17, 25, 33번 층)
```

- 임베딩 표와 head는 target 것을 함께 쓴다. 여러 번 다듬지 않고 한 번에 끝낸다
- 학습: 응답에서 anchor 자리를 뽑아(시퀀스당 512개) 뒤를 마스킹하고 맞힌다. 손실 가중치 `w_k = exp(−(k−1)/λ)`로 앞 자리를 무겁게 본다 (블록 16이면 `λ` = 7. 논문은 이 상수도 `γ`로 적는다). target이 생성한 응답 약 80만 개로 학습
- 과제마다 다르다 — Qwen3-8B에서 MATH-500 6.08×, MT-Bench 2.75×
- LMSYS 분해 실험: "병렬만"은 `τ`가 EAGLE-3보다 낮은데도 더 빨랐고, "주입만"은 `τ`가 높은데 덜 빨랐다. 둘을 합쳐야 가장 빠르다
- 상위 1개가 정답일 비율이 0번 자리 85.4%에서 6번 자리 72.9%로 떨어진다 (DFlash 2 블로그, Qwen3-4B)

</details>

### DSpark — 틀릴 것 같은 초안은 검증하지 않는다

*2026.07 · DeepSeek-AI*

DFlash 위에 두 가지를 더한다.

- **직전 토큰으로 다음 칸을 고친다** — "New"가 뽑혔으면 다음 칸에서 "York"를 밀어 준다. 층을 도는 게 아니라 표 조회 + 작은 곱이라 사이클이 1% 남짓 늘 뿐이다
- **칸마다 "통과할 확률"을 내고, 검증할 길이를 정한다** — 검증에 넣은 초안은 맞든 틀리든 GPU를 먹는다. 그래서 모든 요청의 초안을 확률 순으로 세워, **시스템 전체의 초당 토큰이 가장 커지는 지점까지만** 넣는다. 한가하면 길게, 바쁘면 짧게

📌 DeepSeek-V4 서비스, MTP-1 대비 같은 처리량에서 사용자당 속도 **V4-Flash +60 ~ 85%, V4-Pro +57 ~ 78%**.

<details>
<summary>자세히 — DSpark의 식과 수치</summary>

```
마르코프 head:   로짓_k += W1[x_(k−1)] · W2        W1 : V × r,  W2 : r × V,  r = 256
confidence:      c_k = σ( wᵀ [ h_k ; W1[x_(k−1)] ] )    학습 목표 c_k* = 1 − ½ ‖p_draft − p_target‖₁
스케줄러:        Θ = τ · SPS(B)                       B = Σ_r (1 + ℓ_r),  τ = Σ_r (1 + Σ_(j ≤ ℓ_r) a_(r,j)),  a_(r,j) = c_(r,1) ⋯ c_(r,j)
```

- `SPS(B)`는 배치 크기 `B`에서 엔진이 초당 도는 스텝 수다. 엔진을 켤 때 한 번 재 둔다
- `a_(r,j)`를 큰 것부터 넣으며 `Θ`를 갱신하고 더 오르지 않으면 멈춘다. 서비스에서는 CUDA graph 때문에 두 스텝 전의 confidence로 총 길이를 정한다
- confidence는 그대로 쓰면 과신해서, 자리마다 온도를 맞추는 보정으로 오차를 3 ~ 8%에서 약 1%로 줄였다
- 오프라인 (Qwen3-4B, 초안 7개) `τ` — GSM8K: EAGLE-3 5.14 · DFlash 5.40 · **DSpark 6.11** / MT-Bench: 2.39 · 3.07 · **3.64**
- 서비스용 drafter: MoE 3층, **최근 128토큰만** 보는 attention, 블록 최대 5. 중간 부하에서 요청당 4 ~ 6토큰을 검증한다 (MTP-1은 2토큰 고정)

</details>

### DFlash 2

*2026.08 · Inco AI* — 칸마다 상위 16개 후보를 남기고, 이웃 칸끼리 "이어져도 자연스러운가" 점수를 매겨 **이어지는 경로 하나**를 고른다. 📌 Qwen3.5-4B 평균 `τ`: 내장 MTP 4.54 · DFlash 4.92 · DSpark 5.49 · **DFlash 2 5.97** (DFlash 2 저자의 측정).

---

## 7. drafter는 컨텍스트를 어디서 받나?

1절 그림의 화살표가 이 질문의 답이다. 여기서는 나머지 두 갈래를 본다.

**Gemma 4 MTP — target의 KV를 직접 읽는다** (*2026 · Google*)

- 별도 체크포인트로 배포되는 4층짜리 drafter다 (76M ~ 500M). 입력은 EAGLE · MTP처럼 hidden + 토큰이다
- 문맥은 **target의 KV를 cross-attention으로 직접 읽는다.** 그래서 drafter 쪽에 캐시도 prefill도 없다
- 작은 모델은 head를 어휘 26.2만 개 대신 토큰 군집 위의 top-k로 줄인다. Google은 최대 약 3배를 보고했다

💡 MTP라는 이름이 두 가지를 가리킨다 — DeepSeek식은 "본 학습에 넣은 블록", Gemma 4식은 "함께 배포하는 작은 drafter"다. 비용을 정하는 것은 이름이 아니라 **문맥을 어디서 읽느냐**다.

**n-gram · suffix — 모델 없이 베낀다**

- 마지막 몇 토큰과 같은 구간을 프롬프트와 생성 결과에서 찾아, 그 뒤에 왔던 토큰을 초안으로 낸다. suffix decoding은 이전 요청들의 생성 결과까지 찾는다
- 코드 수정 · 인용처럼 입력을 다시 쓰는 작업에서는 길게 맞고, 새 글을 쓰면 거의 안 맞는다
- 모델 가중치가 없다. 매칭은 CPU(`ngram`, suffix)나 GPU(`ngram_gpu`)에서 돈다. 검증 비용은 다른 갈래와 같다

---

## 8. 흐름

![연표](figures/06-timeline.png)

| 시기 | 무엇을 없앴나 |
|---|---|
| 2022 ~ 23 — 별도 모델 | 원리를 증명했다 |
| 2024 ~ 25 — target에 붙는 얇은 모듈 | **문맥을 다시 읽는 일.** EAGLE이 hidden을 받기 시작했고, DeepSeek-V3가 MTP를 본 학습에 넣어 모델이 drafter를 들고 나오게 됐다. EAGLE-3가 사후 drafter의 표준이 됐다 |
| 2026 — 병렬 블록 + 부하에 맞춘 검증 | **초안의 순차 반복.** DFlash가 블록을 한 번에 뽑고, DSpark · DFlash 2가 뒤 자리를 보정했다. DSpark는 "몇 개를 검증할까"를 매 스텝의 결정으로 바꿨고, Gemma 4는 drafter의 캐시를 없앴다 |

**MTP의 자리** — 학습할 것이 없고 엔진 지원이 좋아 **모델이 주는 기본값**이다. 다만 층 하나 · 순차라 길게 뽑기 어렵다. 더 빠르게 하려면 병렬 drafter로 간다 — DeepSeek 자신이 MTP-1에서 DSpark로 갔다.

> 💡 **"draft 모델이냐 MTP냐"는 이제 맞는 질문이 아니다.** 지금의 drafter는 거의 전부 target의 hidden을 받는 얇은 모듈이고, 갈리는 것은 **언제 학습했나 · 순차냐 병렬이냐 · target 정보를 어디로 받나**다.
> 모델을 낼 때 별도 drafter 체크포인트를 함께 내는 사례도 늘고 있다 — DeepSeek-V4의 DSpark, Gemma 4의 assistant.

---

## 정리

1. **target은 모든 갈래가 같다.** 다른 것은 draft가 무엇을 받고, 얼마나 크고, 몇 번 도는가다
2. **EAGLE · MTP** — target의 hidden과 다음 토큰의 임베딩을 받아 층 하나로 맞힌다. MTP는 같은 모양을 본 학습 때 붙인 것이다
3. **DFlash** — 블록을 한 번에 뽑고 target 정보를 모든 층에 넣는다. 뒤 자리의 약점을 **DSpark · DFlash 2**가 메우고, DSpark는 검증 길이까지 부하에 맞춘다
4. **Gemma 4 MTP** 는 draft 캐시를, **n-gram** 은 draft 모델을 없앴다

**다음 장** — 초안이 틀렸을 때 이미 써 버린 KV는 어떻게 될까? 같은 사이클을 메모리에서 다시 본다 → [03 연산과 KV cache](03-compute-and-kv.md)

---

## 참고

**논문**
- [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE](https://arxiv.org/abs/2401.15077) · [EAGLE-2](https://arxiv.org/abs/2406.16858) · [EAGLE-3](https://arxiv.org/abs/2503.01840)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) 2.2절 · 5.4.3절 — MTP 모듈의 식, 손실 가중치, 채택률
- [FastMTP](https://arxiv.org/abs/2509.18362) · [PARD](https://arxiv.org/abs/2504.18583)
- [DFlash](https://arxiv.org/abs/2602.06036) · [DSpark](https://arxiv.org/abs/2607.05147) · [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) 2.6절 · [H-Spec](https://arxiv.org/abs/2609.24197)

**구현 · 문서 · 측정**
- [vLLM — Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/) · [MTP](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/) · [llm_base_proposer.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py) · [ngram_proposer_gpu.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/ngram_proposer_gpu.py)
- [vLLM Speculators](https://github.com/vllm-project/speculators) · [Speculators — MTP](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp/)
- [SGLang — Speculative Decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding) · [LMSYS — DFlash and Spec V2 (2026-06)](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- [z-lab/dflash](https://github.com/z-lab/dflash) · [Inco AI — DFlash 2](https://inco.ai/blog/dflash2/) · [NVIDIA — DFlash on Blackwell](https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/)
- [vLLM 블로그 — Speculative Decoding on AMD GPUs (2026-08)](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus) · [Hugging Face — Welcome Gemma 4](https://huggingface.co/blog/gemma4)
- 설정 파일: [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) · [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16)

속도 배수는 각 출처의 모델 · GPU · 엔진 · 배치 조건 안에서만 읽는다. DFlash 2의 수치는 저자 블로그다. 엔진 지원 목록은 빠르게 바뀌니 쓰기 전에 공식 문서를 다시 본다.
