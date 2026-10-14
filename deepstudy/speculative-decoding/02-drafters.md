# 02. drafter의 지형 — 누가 초안을 만드나

> 한 번에 검증하고 target의 분포를 지킨다는 원리는 공통이다. 크게 갈리는 것은 **초안을 만드는 쪽**이다.
> 이 페이지는 지금 실제로 쓰이는 drafter의 구조를 보고, 무엇이 표준이고 흐름이 어디로 가는지 정리한다.

| 순서 | 내용 | |
|---|---|---|
| 1 | 두 축으로 놓으면 | 지도 |
| 2 | 별도 draft 모델 | 출발점, 그리고 밀려난 이유 |
| 3 | 얇은 순차 drafter — EAGLE-3 · MTP · Gemma 4 MTP | 지금의 기본값 |
| 4 | 병렬 drafter — DFlash · DSpark · DFlash 2 | 2026년의 최전선 |
| 5 | 모델 없는 방법 — n-gram · suffix | 보조 |
| 6 | 무엇이 표준인가 | **질문 1** |
| 7 | draft/target 구도와 MTP — 흐름 | **질문 2** |

---

## 1. 두 축으로 놓으면

![drafter 지도](figures/05-drafter-map.png)

- **세로 — 어디에 사나.** 독립된 모델인가, target에 붙어 target의 hidden과 LM head를 쓰는가, 모델이 아예 없는가
- **가로 — 초안을 어떻게 뽑나.** 토큰마다 forward를 한 번씩 도는가, 블록 전체를 한 번에 뽑는가
- 가운데 줄(파란색)이 지금의 주류다. 그 안에서 2026년에 **왼쪽에서 오른쪽으로** 무게가 옮겨 가고 있다

| drafter | 어디에 | target에서 받는 것 | 초안 | 학습 | 크기 예 |
|---|---|---|---|---|---|
| 별도 draft 모델 | 독립 LLM | 토큰만 | 순차 | 따로 (또는 기존 소형 모델) | 70B target에 8B draft |
| Medusa | target 위의 head 여러 개 | 마지막 층 hidden | 자리마다 독립 | 나중에 | head 몇 개 |
| **EAGLE-3** | 붙인 decoder 1층 | 낮은 · 중간 · 높은 층의 hidden | 순차 | 나중에, target 동결 | 1층 + 어휘 3.2만짜리 head |
| **MTP 모듈** | 본체에 딸린 블록 1개 | 마지막 층 hidden | 순차 | **본 학습과 함께** | DeepSeek-V3: 14B (본체 671B) |
| Gemma 4 MTP | 별도 체크포인트 | 마지막 층 hidden + **target의 KV** | 순차 | Google이 함께 배포 | 76M (E2B) ~ 500M (31B) |
| **DFlash** | 붙인 5층 | 5개 층 hidden → **모든 층의 K · V** | **병렬** | 나중에 | Qwen3-8B용 약 1B |
| **DSpark** | DFlash + 얇은 순차 head | DFlash와 같음 | 병렬 + 보정 | 나중에 | V4 서비스용: MoE 3층 |
| n-gram · suffix | 없음 | — | 문자열 매칭 | 없음 | 0 |

---

## 2. 별도 draft 모델 — 출발점, 그리고 밀려난 이유

*2022 · Google (Leviathan et al.), DeepMind (Chen et al.)*

같은 토크나이저를 쓰는 작은 LLM을 옆에 둔다. 작은 모델이 `γ`번 decode해서 초안을 내고, target이 한 번에 검증한다.

- 작은 모델도 **자기 prefill, 자기 KV cache**를 전부 가진다. 요청마다 컨텍스트 전체를 한 번 더 처리하고 저장한다
- target이 이미 계산한 표현을 **전혀 쓰지 않는다.** 같은 크기라면 target의 hidden을 받는 drafter보다 덜 맞는다
- 어휘가 같아야 하고, target을 바꾸면 draft도 새로 골라야 한다

대형 서빙에서는 아래의 "붙이는 drafter"에 밀렸다. 남아 있는 곳은 이렇다.

- **로컬 추론** — llama.cpp에서 같은 계열의 작은 모델을 draft로 쓰는 것이 흔하다
- **엔진의 일반 경로** — vLLM `draft_model`, SGLang `STANDALONE`. vLLM은 어휘가 다른 draft도 받는다 (`use_heterogeneous_vocab`, greedy만)
- **PARD** (2025, AMD) — 작은 LLM에 마스크 토큰을 넣어 **여러 개를 한 번에** 뽑도록 고친 것. 별도 모델 줄의 병렬판이다 (vLLM `parallel_drafting`). 초안은 DFlash처럼 한 번에 뽑지만, 컨텍스트는 **자기 KV를 자기 prefill로** 만든다 — target hidden을 받지 않는다

---

## 3. 얇은 순차 drafter — EAGLE-3 · MTP · Gemma 4 MTP

target 위에 **층 하나 남짓**을 붙인다. 입력은 target이 방금 만든 hidden과 토큰 임베딩이다.

![EAGLE-3와 MTP 모듈](figures/07-thin-ar-drafters.png)

### 공통 뼈대 — 무엇을 받아 무엇을 맞히나

위치 `i`에서 drafter가 받는 것은 한 쌍이다.

| 입력 | 담고 있는 것 |
|---|---|
| target의 hidden `h_i` | 토큰 `1 ~ i`까지를 읽은 target의 이해. target은 이것으로 `i + 1`번째 토큰을 골랐다 |
| `i + 1`번째 토큰의 임베딩 | 그 자리에 **실제로 정해진 토큰** |

이 둘로 **`i + 2`번째 토큰**을 맞힌다. hidden만으로는 `i + 1`에 무엇이 뽑혔는지 모르기 때문에 토큰을 함께 넣는다.

두 번째 스텝부터는 target의 hidden이 없다 (아직 target이 그 자리를 계산하지 않았다). 그래서 **drafter가 직전 스텝에 낸 자기 hidden**을 그 자리에 넣는다. 그림의 점선이 이것이다.

> 💡 **뒤 스텝일수록 덜 맞는 이유가 여기 있다.** 첫 스텝은 target의 진짜 hidden을 받지만, 그 뒤는 drafter 자신의 근사를 받는다. 오차가 쌓인다.

### EAGLE-3

*2025 · Li et al. · EAGLE(2024) → EAGLE-2(2024) → EAGLE-3*

- **target의 세 층**(낮은 · 중간 · 높은)에서 hidden을 꺼내 이어 붙이고, 선형 `3d → d`로 섞는다. 마지막 층만 쓰던 EAGLE-1/2보다 정보가 많다
- 섞은 hidden과 토큰 임베딩에 각각 RMSNorm을 걸고 이어 붙인다 (`2d`). **decoder 층 하나**의 Q · K · V 투영이 이 `2d`를 바로 받는다
- 출력 head는 **자기 것**이다. 공개 체크포인트는 자주 나오는 **3.2만 토큰**만 다룬다 (Llama-3.1-8B용 설정의 `draft_vocab_size: 32000`). target의 12.8만 어휘 head보다 4배 작다 — drafter 한 스텝의 비용에서 head가 큰 몫이라 의미가 크다 ([04](04-simulator.md#6-개략-검산))
- 학습: target은 얼려 둔다. EAGLE-1/2의 "hidden을 맞히는" 손실을 버리고 **토큰만** 맞힌다. 학습 중에 자기 출력을 다시 넣어 여러 스텝을 흉내 낸다 (training-time test)

📌 보고된 수치 (EAGLE-3 논문, temperature 0, 트리 초안)

| target | 속도 | `τ` |
|---|---|---|
| Llama-3.1-8B | 4.44× (EAGLE-2 3.23×) | 6.23 |
| Llama-3.3-70B | 4.12× | 5.88 |
| SGLang · H100 · 배치 1 · Llama-3.1-8B | 158 → 373 토큰/초 | — |

- 위 `τ`는 노드 수십 개짜리 트리 기준이다. 서빙에서 흔한 **일렬 초안**에서는 낮다 — 예: Kimi-K2.5, `γ` = 4에서 MAL 3.96 (vLLM 블로그, AMD GPU)
- 다른 팀이 학습한 EAGLE-3는 훨씬 낮을 수 있다. DFlash 논문의 Qwen3-8B용 EAGLE-3는 트리 60개에서 `τ` 3.40, 2.02×였다

### MTP 모듈 — DeepSeek-V3

*2024 · DeepSeek · Multi-Token Prediction*

본 학습 때부터 붙어 있다. 목적은 원래 **학습 품질**이다. 다음 토큰만이 아니라 그다음 토큰까지 맞히게 하면 표현이 좋아진다. 추론 때는 버려도 되고, 초안으로 쓸 수도 있다.

```
h'_i  =  M · [ RMSNorm(h_i) ; RMSNorm(Emb(t_(i+1))) ]          M : 2d → d
h''_i =  TRM( h'_i )                                            본체와 같은 Transformer 블록 하나
P(t_(i+2)) = OutHead( h''_i )                                    본체의 출력 head를 함께 쓴다
```

- 블록은 **본체의 층과 같은 종류**다. DeepSeek-V3라면 MLA + MoE 블록 하나다. 그래서 MTP 모듈 하나가 14B이고, 초안 한 스텝이 본체 한 스텝의 약 4.4%다
- 깊이 `D` = 1 (모듈 하나). 손실 가중치는 앞 10T 토큰 동안 0.3, 나머지 4.8T 토큰 동안 0.1
- 📌 두 번째 토큰의 채택률 85 ~ 90%, 생성 속도 1.8배 (DeepSeek-V3 기술 보고서)
- DeepSeek는 이것을 **MTP-1**이라 부르며 서비스의 기준선으로 써 왔다. 요청마다 **2토큰을 고정으로** 검증한다 (DSpark 논문)

**모듈 하나로 여러 개를 뽑으려면** — 같은 모듈을 다시 돈다. 출력 hidden을 입력 hidden 자리에 넣는 것이다.

- 학습 때는 모듈이 한 번만 쓰이도록 학습됐다. 두 번째 반복부터는 학습 때 없던 입력을 받는다. vLLM-Ascend 문서는 DeepSeek의 MTP를 3개 이상 뽑으면 정확도와 속도를 보장할 수 없다고 적는다
- **FastMTP** (2025, Tencent)는 이 반복 사용을 위해 MTP 층을 따로 미세조정한다. 스텝별 손실에 지수적으로 줄어드는 가중치(β = 0.6)를 준다. vLLM의 Speculators가 이 방식의 MTP 미세조정을 지원한다
- Qwen3.5 · Qwen3.6은 MTP 층 하나(`mtp_num_hidden_layers: 1`)를 여러 스텝 쓴다. SGLang 블로그의 Qwen3.5-397B 비교는 MTP를 7스텝으로 돌렸다

**MTP를 가진 공개 모델** — DeepSeek-V3 · V3.2 · V4, Qwen3-Next · Qwen3.5 · Qwen3.6, GLM-4.5 · GLM-5.x, ERNIE-4.5, Nemotron 3 Super · Ultra, MiMo. (트레이서가 받은 config 기준은 [04](04-simulator.md#2-어떤-모델로) 참고)

### Gemma 4 MTP — target의 KV를 직접 읽는 drafter

*2026 · Google*

이름은 MTP지만 생김새는 다르다. **별도 체크포인트**("assistant")로 배포된다.

- 4층짜리 작은 Transformer (지역 attention 3 + 전역 attention 1). 크기는 76M (E2B) ~ 500M (31B)
- 입력은 직전 스텝의 target 마지막 층 hidden과 토큰 임베딩이다 (임베딩 표는 따로 둔다)
- **컨텍스트는 target의 KV를 직접 읽는다** (cross-attention). 그래서 drafter 쪽에 컨텍스트 캐시도, 별도 prefill도 없다
- 작은 모델(E2B · E4B)은 어휘 26.2만 전체 대신 **토큰 군집 위의 top-k**로 head를 줄인다 (`d × 262,000` → `d × 4,096`)
- Google은 최대 약 3배를 보고했다. vLLM에서는 `method: "mtp"`에 assistant 체크포인트를 `model`로 준다

> 💡 **MTP라는 이름이 두 가지를 가리킨다.** DeepSeek식은 "본 학습에 함께 넣은 블록", Gemma 4식은 "함께 배포하는 작은 drafter"다.
> 시뮬레이터 입장에서 갈리는 것은 이름이 아니라 **캐시를 어디에 두느냐**다 ([03](03-compute-and-kv.md#6-drafter의-캐시--컨텍스트를-어디서-읽나)).

### 셋을 나란히

| | EAGLE-3 | MTP (DeepSeek식) | Gemma 4 MTP |
|---|---|---|---|
| 학습 | 나중에, target 동결 | 본 학습과 함께 | 함께 배포 |
| target에서 받는 것 | 세 층의 hidden | 마지막 층 hidden | 마지막 층 hidden + KV |
| 층 | decoder 1층 | 본체 블록 1개 | 4층 |
| 출력 head | 자기 것 (작은 어휘) | target 것 | 자기 것 (작은 모델은 군집 top-k) |
| 컨텍스트 캐시 | 1층 분량, 자기 것 | 1층 분량, 자기 것 | 없음 — target 것을 읽는다 |
| 여러 개 뽑기 | 같은 층을 반복 (학습 때 흉내 냄) | 같은 블록을 반복 (학습 때 없던 일) | 반복 |

---

## 4. 병렬 drafter — DFlash · DSpark · DFlash 2

순차 drafter는 아무리 얇아도 `γ`번 돈다. 초안 시간이 `γ`에 비례하니 `γ`를 크게 잡을 수 없고, 층을 두껍게 쌓을 수도 없다.

**블록 전체를 forward 한 번에 뽑으면** 둘 다 풀린다.

![병렬 drafter](figures/08-parallel-drafters.png)

### DFlash

*2026.02 · UC San Diego (z-lab) · Block Diffusion for Flash Speculative Decoding*

**입력** — 블록 하나 (Qwen3는 16칸, Llama-3.1은 10칸)

```
[ 방금 확정된 토큰 ][ M ][ M ] … [ M ]        첫 칸 = anchor, 나머지 = 마스크 토큰
        ↓  draft 5층을 한 번 통과  (블록 안의 자리들은 서로를 다 본다)
        [ d1 ][ d2 ] … [ d15 ]                  마스크 자리마다 토큰 하나
```

- anchor는 직전 verify에서 target이 고친(또는 보너스로 준) 토큰이다. 01의 "아직 KV가 없는 마지막 확정 토큰"과 같은 것이다
- 이미지 diffusion처럼 여러 번 다듬지 않는다. **한 번에 끝낸다** — 어차피 target이 검증하므로 거친 초안이면 된다
- 임베딩 표와 LM head는 **target 것을 함께 쓴다** (동결). 학습하는 것은 draft 층뿐이다

**target 정보를 넣는 곳 — 모든 층의 K · V** (KV injection)

```
H_ctx = RMSNorm( W_c · [ H^(l1) ; H^(l2) ; … ; H^(l5) ] )      target 5개 층의 hidden을 이어 붙여 섞는다
```

- Qwen3-8B용 drafter는 target 36층 중 **1, 9, 17, 25, 33번 층**을 쓴다 (공개 설정)
- `H_ctx`를 draft의 **모든 층**에서 각자의 K · V 투영으로 바꿔 캐시에 둔다. 블록의 자리들이 이것을 attention으로 읽는다
- EAGLE-3는 같은 정보를 입력 한 곳에만 넣었다. DFlash는 **모든 층이 직접 참조**한다. 그래서 draft를 깊게 쌓을수록 잘 맞는다

**학습** — 응답에서 anchor 자리를 무작위로 뽑아(시퀀스당 512개) 그 뒤를 마스킹하고 맞히게 한다. 손실 가중치 `w_k = exp(−(k−1)/λ)`로 **앞 자리를 무겁게** 본다 (블록 16이면 `λ` = 7. 논문은 이 감쇠 상수도 `γ`로 적지만, 여기서는 초안 수 `γ`와 헷갈리지 않게 `λ`로 쓴다). 앞이 틀리면 뒤는 어차피 버려지기 때문이다. 데이터는 target이 직접 생성한 응답 약 80만 개다.

📌 보고된 수치 (DFlash 논문, Qwen3, temperature 0, 7개 벤치 평균)

| target | EAGLE-3 (트리 60) | DFlash (블록 16) |
|---|---|---|
| Qwen3-4B | 2.08× · `τ` 3.48 | **4.91× · `τ` 6.54** |
| Qwen3-8B | 2.02× · `τ` 3.40 | **4.86× · `τ` 6.49** |

- 과제마다 다르다. Qwen3-8B에서 MATH-500은 6.08×, 대화(MT-Bench)는 2.75×다
- **빠른 이유가 둘이다** — `τ`가 높고(깊은 drafter + 모든 층 주입), 사이클이 짧다(forward 1번). LMSYS의 분해 실험에서 "병렬만" 쓰면 `τ`는 EAGLE-3보다 낮은데도 더 빨랐고, "주입만" 쓰면 `τ`는 높은데 덜 빨랐다. 둘을 합쳐야 가장 빠르다

**약점 — 블록 뒤쪽일수록 덜 맞는다.** `k`번째 자리는 `k − 1`번째가 무엇으로 뽑혔는지 모른 채 고른다. 가능한 앞 토큰들에 대해 뭉뚱그린 예측이다.

- "New York"과 "San Francisco"가 반반이면, 두 자리를 따로 뽑아 "New Francisco"가 나올 수 있다
- 상위 1개가 정답일 비율이 0번 자리 85.4%에서 6번 자리 72.9%로 떨어진다 (DFlash 2 블로그, Qwen3-4B)

### DSpark — 병렬 위에 얇은 순차 head, 그리고 검증 길이 고르기

*2026.07 · DeepSeek-AI · Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*

DFlash를 그대로 backbone으로 쓰고 두 가지를 더한다.

**① 마르코프 head — 직전 토큰으로 다음 자리를 고친다**

```
로짓_k  +=  B(x_(k−1), ·)        B = W1 · W2,   W1 : V × r,  W2 : r × V,   r = 256
```

- 병렬 패스가 끝난 뒤 왼쪽부터 한 자리씩, **실제로 뽑힌 직전 토큰** `x_(k−1)`의 행을 꺼내(`W1`) `W2`를 곱해 로짓에 더한다
- 순차 부분이지만 층을 도는 것이 아니라 **표 조회 + 작은 곱 한 번**이다. 사이클 시간이 0.2 ~ 1.3% 늘 뿐이다
- `V × V` 전이 행렬을 rank 256으로 눌러 든 것이다

**② confidence head와 스케줄러 — 검증할 길이를 요청마다 정한다**

```
c_k = σ( wᵀ [ h_k ; W1[x_(k−1)] ] )          k번째 초안이 통과할 확률 (앞이 다 통과했다는 조건)
학습 목표:  c_k* = 1 − ½ ‖ p_draft − p_target ‖₁       ← 01의 채택 확률 식 그대로
```

- 그대로 쓰면 과신한다. 왼쪽부터 자리마다 온도를 하나씩 맞추는 보정(Sequential Temperature Scaling)으로 보정 오차를 3 ~ 8%에서 약 1%로 줄였다
- 요청 `r`의 `j`번째 초안이 살아남을 확률 `a_(r,j) = c_(r,1) × … × c_(r,j)`

스케줄러는 **시스템 전체의 초당 토큰 수**를 최대로 만든다.

```
Θ  =  τ · SPS(B)

  B       이번 스텝에 target이 검증할 총 토큰 수  = Σ_r (1 + ℓ_r)       ℓ_r = 요청 r에서 검증할 초안 수
  τ       기대 확정 토큰 수  = Σ_r (1 + Σ_(j ≤ ℓ_r) a_(r,j))
  SPS(B)  배치 크기 B에서 엔진이 초당 도는 스텝 수 — 엔진을 켤 때 한 번 재 둔 표
```

- 모든 요청의 `a_(r,j)`를 **한 줄로 세워 큰 것부터** 하나씩 넣으며 `Θ`를 갱신하고, 더 오르지 않으면 멈춘다
- 한가하면 길게, 바쁘면 짧게 — 01의 "`γ`를 하나 늘릴지" 기준을 **매 스텝 · 요청마다** 푸는 것이다
- 실제 서비스에서는 CUDA graph 때문에 다음 배치 크기를 미리 알아야 해서, **두 스텝 전의 confidence**로 총 길이를 정하고 지금의 confidence로 순서를 정한다

📌 보고된 수치

| | 내용 |
|---|---|
| 오프라인 (Qwen3-4B, `γ` = 7, 스케줄러 끔) | `τ` — GSM8K: EAGLE-3 5.14 · DFlash 5.40 · **DSpark 6.11** / MT-Bench: 2.39 · 3.07 · **3.64** / Arena-Hard: 2.55 · 2.83 · **3.29** |
| 평균 향상 | EAGLE-3 대비 +26.7 ~ 30.9%, DFlash 대비 +16.3 ~ 18.4% (Qwen3 4B · 8B · 14B) |
| DeepSeek-V4 서비스 (DSpark-5 vs MTP-1) | 같은 처리량에서 사용자당 생성 속도 V4-Flash **+60 ~ 85%**, V4-Pro **+57 ~ 78%** |
| 검증 길이 | 중간 부하에서 요청당 약 4 ~ 6토큰 (MTP-1은 2토큰 고정). 부하가 오르면 줄어든다 |
| 서비스용 drafter | MoE 3층 (mHC 포함), **최근 128토큰만 보는 attention**, 블록 최대 5 |

- 오프라인 실험은 5층 drafter였고, 서비스용은 3층에 창 128로 줄였다. 창이 있으니 **drafter 캐시가 요청당 128줄로 묶인다**
- 체크포인트: DeepSeek-V4-Pro-DSpark · V4-Flash-DSpark (Hugging Face), 코드: deepseek-ai/DeepSpec. vLLM의 "Adaptive Verification"은 현재 DSpark만 지원한다

### DFlash 2 — 후보 경로 고르기

*2026.08 · Inco AI*

DSpark처럼 자리 사이의 의존을 되살리되, 더 싸게 한다.

- 자리마다 **상위 16개 후보**를 남긴다. 이웃한 자리의 후보 쌍 `(a, b)`에 점수 `U_t(b) + ⟨A(a) ⊙ H(h_t), B(b)⟩`를 매기고, 이어지는 경로 하나를 고른다. 점수는 한 번에 다 계산하고 마지막 경로 따라가기만 순차다 (+2.0M 파라미터, 사이클 +0.6%)
- 층 안의 attention · MLP 앞뒤에 **앞 자리와 섞는 2-tap conv**를 넣는다 (+16.5M, +0.7%)
- 📌 Qwen3.5-4B 평균 `τ`: 내장 MTP 4.54 · DFlash 4.92 · DSpark 5.49 · **DFlash 2 5.97** — DFlash 2 저자의 측정이다. 여기의 DSpark는 커뮤니티가 학습한 것이다

---

## 5. 모델 없는 방법 — n-gram · suffix

| 방법 | 하는 일 | 어디서 도나 | 엔진 |
|---|---|---|---|
| n-gram (prompt lookup) | 마지막 몇 토큰과 같은 구간을 프롬프트와 생성 결과에서 찾아, 그 뒤에 왔던 토큰을 초안으로 낸다 | vLLM `ngram`은 CPU, `ngram_gpu`는 **GPU** | vLLM `ngram` · `ngram_gpu`, SGLang `NGRAM` |
| suffix decoding | 프롬프트와 **이전 요청들의 생성 결과**로 suffix tree를 만들어 찾는다 | 호스트(CPU)의 suffix tree | vLLM `suffix` (Arctic Inference) |

- **모델 가중치가 없다.** 초안 비용은 CPU의 매칭 시간이거나(`ngram`, suffix), GPU 위의 작은 매칭 연산과 토큰 버퍼다(`ngram_gpu`). "GPU를 전혀 안 쓴다"는 아니다
- 검증 비용은 다른 drafter와 같다. target이 `K`줄을 처리한다
- 입력을 다시 베끼는 작업(코드 수정, RAG, 문서 편집)에서는 길게 맞고, 새 글을 쓰는 작업에서는 거의 안 맞는다
- vLLM 문서는 이득이 작지만 **최대 부하에서 일을 늘리지 않는** 방법으로 소개한다. 큰 배치에서 모델 기반 drafter의 보조로 쓴다

---

## 6. 무엇이 표준인가

### 엔진이 지원하는 것 (2026년 10월, 공식 문서 기준)

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

- TensorRT-LLM도 draft 모델 · EAGLE-3 · MTP · n-gram을 지원하고, NVIDIA는 gpt-oss-120b에서 DFlash를 TensorRT-LLM으로 돌린 결과를 냈다
- llama.cpp는 draft 모델과 MTP(`--spec-type mtp`)를 지원한다

### 표준은 세 층으로 나눠 본다

**① 검증의 원리 — 공통이다.** 초안을 한 번에 검증하고 target의 분포를 기준으로 판정한다는 원리는 같다. 채택 알고리즘은 다르다 — 토큰별 거부 샘플링(기본형), block verification(SGLang 옵션), 트리의 다중 후보. Medusa의 typical acceptance처럼 분포 일치를 포기하고 채택을 늘리는 방식도 있다 ([01](01-principle.md#3-채택-규칙--왜-출력이-같은가)). 서빙 엔진의 기본 경로는 토큰별 거부 샘플링이다.

**② 검증 모양 — 일렬로 모이는 중이다.**

- vLLM의 drafter 기반 클래스는 **일렬 초안만** 만든다 (`[배치, γ]` 모양). 이 클래스에는 트리 처리가 없고, 트리용 FIXME 주석만 남아 있다
- SGLang의 새 기본 엔진(Spec V2, overlap scheduler)은 `topk = 1`, 즉 **일렬만** 받는다. 트리는 이전 경로(EAGLE-2 방식)에 남아 있다
- DFlash · DSpark는 블록 하나를 일렬로 검증한다

> 💡 서빙이 일렬로 모이는 이유 — 배치가 크면 넓은 트리를 검증할 남는 연산이 없고, CUDA graph와 overlap scheduling은 모양이 고정된 쪽을 좋아하며, 롤백이 "길이만 되돌리기"로 단순해진다 ([03](03-compute-and-kv.md#5-트리-검증)). 트리는 배치 1의 지연을 짜내는 용도로 남는다.

**③ drafter — 하나가 아니라 셋이 공존한다.**

| | 위치 | 근거 |
|---|---|---|
| **내장 MTP** | 모델이 들고 나오면 일단 쓰는 기본값 | 따로 학습할 것이 없다 (추론 비용은 든다). 엔진 1급 지원. DeepSeek의 서비스 기준선(MTP-1) |
| **EAGLE-3** | MTP가 없는 모델의 사후 drafter 표준 | 체크포인트가 가장 넓다 (Llama · Qwen3 · gpt-oss · Gemma 4 · Kimi …). SGLang 문서의 권장 |
| **DFlash 계열** | 공개된 측정에서의 최전선. 지원과 체크포인트가 가장 빠르게 늘었다 | z-lab 체크포인트 20개 이상, NVIDIA · LMSYS · vLLM의 지원과 측정, DeepSeek 서비스가 MTP-1에서 DSpark로 |

- 이 표의 근거는 엔진 지원 · 공개 체크포인트 · 공개 측정이다. 실제 서비스에서 무엇이 얼마나 쓰이는지를 보여 주는 자료는 아니다

### 공개 비교 — 조건과 함께

| 비교 | 조건 | 결과 | 출처 |
|---|---|---|---|
| EAGLE-3 vs DFlash | Qwen3-8B, temp 0, 7개 벤치 평균 | 2.02× vs **4.86×** | DFlash 논문 |
| EAGLE-3 vs DFlash | gpt-oss-120b, TensorRT-LLM, B300, 같은 동시 요청 수 | 평균 1.7× vs **2.3×** | NVIDIA |
| EAGLE-3 vs DFlash | Llama-3.1-8B, 같은 조건 | 평균 2.2× vs **2.8×** | NVIDIA |
| 내장 MTP (7스텝) vs DFlash | Qwen3.5-397B-A17B, 8 × B200, SGLang | 동시 요청 1 ~ 32 전 구간 DFlash 우세, 동시 1에서 MTP의 1.5배 | LMSYS |
| 내장 MTP vs DFlash | Qwen3.6-35B-A3B, AMD MI300X · MI355X, vLLM | 1.28 ~ 1.49× vs **1.77 ~ 2.06×** | vLLM 블로그 |
| Gemma 4 MTP vs DFlash vs EAGLE-3 | gemma-4-26B-A4B, AMD, vLLM | 최대 2.74× · **2.87×** · 2.11 ~ 2.27× | vLLM 블로그 |
| MTP-1 vs DSpark | DeepSeek-V4 서비스 | 사용자당 속도 **+57 ~ 85%** | DSpark 논문 |
| 반례 | Qwen3-8B, AMD, vLLM | EAGLE-3는 MATH-500에서 **기준보다 느림** (최대 0.88×), DSpark 1.15 ~ 1.63× | vLLM 블로그 |

- 위 실험들에서는 같은 target 안에서 대체로 **EAGLE-3 < 내장 MTP ≲ DFlash < DSpark ≲ DFlash 2** 순으로 관찰됐다. 다섯을 같은 조건에서 함께 잰 실험은 없다 — 쌍마다 따로 잰 결과를 이어 붙인 순서다
- GPU · 엔진 · 체크포인트가 바뀌면 크기도 순서도 바뀐다 (위의 반례). vLLM 블로그의 결론도 "하나를 고를 수 없다 — 대표 작업에서 처리량과 자리별 채택률로 고르라"다

---

## 7. draft/target 구도와 MTP — 흐름

![연표](figures/06-timeline.png)

**① 2022 ~ 23 — 별도 모델.** 작은 LLM이 초안을 낸다. 원리는 증명됐지만 관리가 무겁고 덜 맞는다.

**② 2024 ~ 25 — target에 붙는 얇은 모듈.** "target이 이미 계산한 표현을 쓰자"가 핵심이다.

- Medusa는 head를 여러 개 붙였지만 head끼리 서로를 몰랐다. EAGLE은 hidden을 자기회귀로 이어 가며 이것을 고쳤다
- 같은 해 DeepSeek-V3가 MTP를 **본 학습에 넣었다.** 이때부터 모델이 drafter를 들고 나온다
- EAGLE-3는 여러 층의 hidden을 쓰고 토큰만 맞히도록 바꿔 사후 drafter의 표준이 됐다. Qwen3-Next · Qwen3.5 · GLM은 MTP 1층을 기본으로 싣고 나왔다

**③ 2026 — 초안을 병렬로, 검증을 부하에 맞춰.**

- DFlash가 초안의 직렬성을 없앴다. DSpark · DFlash 2가 그 대가(뒤 자리의 품질)를 싸게 메웠다
- DSpark의 스케줄러는 "몇 개를 검증할까"를 하이퍼파라미터에서 **매 스텝의 결정**으로 바꿨다
- Gemma 4는 drafter가 **target의 KV를 직접 읽게** 해서 drafter의 캐시와 prefill을 없앴다. H-Spec(2026.09) 같은 연구도 같은 방향이다

### MTP는 어디에 서 있나

| | |
|---|---|
| 강점 | 따로 학습할 것이 없다. 본 학습 품질도 올린다. 모든 엔진이 1급으로 지원한다 |
| 한계 | 층 하나 · 순차 — `γ`를 늘리기 어렵다. 모듈을 반복해 쓰면 뒤 자리가 약하다. 추론 때는 블록 · head 연산, 가중치 메모리(DeepSeek-V3는 14B), drafter KV가 든다 |
| 자리 | **모델이 주는 기본값.** 더 빠르게 하려면 병렬 drafter로 간다 — DeepSeek 자신이 그렇게 했다 |

> 💡 **"draft 모델이냐 MTP냐"는 이제 맞는 질문이 아니다.** 지금 쓰이는 drafter는 거의 전부 target의 hidden을 받는 얇은 모듈이고, 갈리는 것은 **언제 학습했나 · 순차냐 병렬이냐 · target 정보를 어디에 넣나** 세 가지다.
> 모델을 낼 때 MTP와 함께, 또는 대신 **별도 drafter 체크포인트**를 같이 내는 사례도 늘고 있다 — DeepSeek-V4의 DSpark, Gemma 4의 assistant, Speculators의 Kimi K3 · GLM 5.2용 DSpark.

---

## 참고

**논문**
- [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE](https://arxiv.org/abs/2401.15077) · [EAGLE-2](https://arxiv.org/abs/2406.16858) · [EAGLE-3](https://arxiv.org/abs/2503.01840)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) 2.2절 · 5.4.3절 — MTP 모듈의 식, 손실 가중치, 채택률
- [FastMTP](https://arxiv.org/abs/2509.18362) — MTP 층을 여러 스텝용으로 미세조정
- [PARD](https://arxiv.org/abs/2504.18583) — 작은 draft 모델을 병렬 초안용으로 고친다
- [DFlash](https://arxiv.org/abs/2602.06036) — 블록 diffusion drafter, KV 주입, Qwen3 결과와 동시 요청 수별 표
- [DSpark](https://arxiv.org/abs/2607.05147) — 마르코프 head, confidence head, `Θ = τ · SPS(B)` 스케줄러, DeepSeek-V4 서비스 결과
- [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) 2.6절 — MTP drafter의 구조, target KV cross-attention
- [H-Spec](https://arxiv.org/abs/2609.24197) — drafter 쪽 KV cache 없이 병렬 초안

**구현 · 문서 · 측정**
- [vLLM — Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/) · [MTP](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/) · [llm_base_proposer.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py) · [ngram_proposer_gpu.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/ngram_proposer_gpu.py)
- [vLLM Speculators](https://github.com/vllm-project/speculators) · [Speculators — MTP](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp/)
- [SGLang — Speculative Decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding) · [LMSYS — DFlash and Spec V2 (2026-06)](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- [z-lab/dflash](https://github.com/z-lab/dflash) · [Inco AI — DFlash 2](https://inco.ai/blog/dflash2/)
- [NVIDIA — DFlash on Blackwell](https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/)
- [vLLM 블로그 — Speculative Decoding on AMD GPUs (2026-08)](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)
- [Hugging Face — Welcome Gemma 4](https://huggingface.co/blog/gemma4)
- 설정 파일: [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) (`draft_vocab_size: 32000`) · [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) (5층, 블록 16, target 층 1 · 9 · 17 · 25 · 33)

**범위와 주의**
- 속도 배수는 각 출처의 모델 · GPU · 엔진 · 배치 조건 안에서만 읽는다. 표의 줄끼리 직접 비교하지 않는다
- DFlash 2의 수치는 DFlash 2 저자의 블로그다. 제3자 검증은 아직 적다
- 엔진의 지원 목록은 빠르게 바뀐다. 쓰기 전에 공식 문서를 다시 본다
