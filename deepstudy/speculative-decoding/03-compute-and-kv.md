# 03. 연산과 KV cache — 같은 사이클을 메모리에서 다시 본다

> 01의 사이클을 **연산과 메모리**로 다시 따라간다. 예시도 같다 — 프롬프트 `The cat sat`, KV가 없는 `on`, 초안 `the mat and slept`, 채택 2개, target이 고른 `.`.

| 순서 | 질문 |
|---|---|
| 1 | verify 한 번은 decode 한 번보다 얼마나 비쌀까? |
| 2 | 초안이 틀렸다면, 이미 써 버린 KV는 어떻게 될까? |
| 3 | 갈래가 여럿이면? — 트리 검증 |
| 4 | drafter도 캐시가 필요할까? |
| 5 | 배치를 키우면 이득은 어떻게 될까? |
| 6 | 모델 구조에 따라 깨지는 가정 |
| 7 | 정리 |

---

## 1. verify 한 번은 decode 한 번보다 얼마나 비쌀까?

verify는 decode와 **같은 forward**다. 바뀌는 것은 새 토큰의 줄 수뿐이다 — decode는 `on` 1줄, verify는 `on the mat and slept` 5줄.

![verify의 모양](figures/09-verify-shapes.png)

- **위 — 가중치 곱.** 같은 가중치 상자를 한 번 읽는다. 곱셈만 5배가 되는데, 원래 짧았으니 시간은 거의 그대로다
- **아래 — attention.** 새 토큰 5개가 모두 확정된 캐시(보라)를 읽는다. 이번에 쓴 5줄(파랑)은 **계단 모양**으로 본다 — `mat` 자리는 `on the mat`까지만 본다
- **KV 쓰기** — 1줄이 아니라 5줄을 쓴다

| 연산 | decode | verify (`K`줄) |
|---|---|---|
| 가중치 곱 | 가중치 읽기 | 읽기 그대로, 곱셈 `K`배 |
| attention | 캐시 `S`줄 읽기 | 곱셈 `K`배. 캐시 읽기는 커널에 따라 1 ~ `K`번 |
| KV 쓰기 · 로짓 | 1줄 | `K`줄 |

### 조건이 둘 있다

- **캐시를 한 번만 읽는다는 것은 이상적인 경우다.** 커널이 5개 쿼리 사이에서 캐시를 재사용해야 그렇다. 실제로 몇 번 읽는지는 커널의 타일 크기와 헤드를 묶는 방식이 정한다
- **MLA는 attention 곱셈이 먼저 막힌다.** 캐시 1바이트로 하는 곱셈이 decode만으로 130 ~ 240번이라, 검증 토큰 2개만 돼도 H100의 균형점(약 295)을 넘는다. 그러면 verify의 attention 시간이 `K × S`에 비례해 는다. GQA(Llama-3.1-8B)는 검증 토큰 16개에서도 약 64번이라 여유가 있다

<details>
<summary>자세히 — 한 사이클의 연산 목록 (Llama-3.1-8B + EAGLE-3)</summary>

요청 하나, 초안 `γ`개, 컨텍스트 `S`, 검증 토큰 `K = γ + 1`. `d` 4096 · 쿼리 헤드 32 · KV 헤드 8 · 헤드 차원 128 · 층 32 · 어휘 128,256.

| # | 단계 | 연산 | shape | 읽는 것 | 쓰는 것 |
|---|---|---|---|---|---|
| 1 | verify | 임베딩 | 토큰 `K`개 → `(K × d)` | 표에서 `K`줄 | |
| 2 | verify | 층마다 Q · K · V 투영 | `(K × d) · (d × (n_h + 2n_kv)·d_h)` | 가중치 | |
| 3 | verify | KV 쓰기 | 층마다 `K`줄 | | target KV `K`줄 |
| 4 | verify | attention | `Q (n_h, K, d_h)` × 캐시 `(n_kv, S + K, d_h)` | 캐시 `S + K`줄 | |
| 5 | verify | O 투영 · FFN | `(K × ·)` | 가중치 | |
| 6 | verify | hidden 꺼내기 | 3개 층에서 `(K × d)`씩 | | drafter에 넘길 버퍼 |
| 7 | verify | 출력 head | `(K × d) · (d × V)` → `(K × V)` | 출력 head | 로짓 `K`줄 |
| 8 | accept | 비교 · 거부 샘플링 | `K`자리 | 로짓 | 채택 수 `a` |
| 9 | 롤백 | 버린 줄 정리 | — | | 길이 값 또는 자리 표 |
| 10 ~ 12 | draft 첫 패스 | 세 층 섞기 + 1층 forward `a + 1`줄 + head 1줄 | | drafter KV `S`줄 | drafter KV `a + 1`줄, 초안 `d1` |
| 13 | draft 나머지 | (`γ − 1`)번 × [1층 1줄 + head 1줄] | | drafter KV | drafter KV 1줄씩 |

- 1 ~ 7은 decode 한 스텝과 같은 연산이다. vLLM은 한 엔진 스텝 안에서 verify(1 ~ 9) → 다음 사이클의 draft(10 ~ 13) 순서로 돈다
- **로짓 버퍼** — verify는 `K`자리 모두 분포가 필요하다. Qwen3-8B · 블록 16이면 FP32로 요청당 약 9.7 MB, 요청 64개면 약 622 MB다
- **바이트당 곱셈** — GQA는 쿼리 헤드 `g`개 × 검증 토큰 `K`개가 KV 한 줄을 함께 쓰므로 약 `g·K`다. MLA의 decode 값은 [attention 세미나](../../seminar/01-attention-seminar_sum_ver.md)의 "읽은 바이트당 연산"이다

</details>

---

## 2. 초안이 틀렸다면, 이미 써 버린 KV는 어떻게 될까?

verify는 5자리를 처리하며 **5줄을 이미 썼다.** `and` `slept`는 틀린 초안이었다.

![KV cache의 한 사이클](figures/10-kv-lifecycle.png)

- **①** 캐시에는 프롬프트 3줄뿐이다. `on`은 점선 — KV가 없다
- **②** verify가 5줄을 쓴다. `on`도 이제 KV가 생겼다
- **③** 맞은 `the` `mat`은 남기고, 틀린 `and` `slept`는 **버린다**
- **④** 캐시는 6줄이 됐고, target이 고친 `.`이 다음 사이클의 첫 줄을 기다린다

**남는 줄 수 = 이번에 확정된 토큰 수**다. 다만 한 칸 밀려 있다 — 확정된 것은 `the mat .`, 남은 줄은 `on the mat`이다.

**버리는 방법은 둘이고, 어느 쪽도 지우지 않는다.**

- **길이만 되돌린다** (vLLM) — 캐시 길이 값을 6으로 둔다. 버린 줄은 길이 밖이라 아무도 읽지 않고, 다음 verify가 덮어쓴다
- **자리를 반납한다** (SGLang) — 토큰 단위로 자리를 나눠 주는 엔진은 버린 줄의 자리를 돌려주고 "토큰 → 자리" 표만 고친다. 트리 검증에는 이 방식이 필요하다

**버리는 일의 양** — 사이클마다 평균 `γ + 1 − τ`줄을 버린다.

| | 검증 `K` | `τ` | 버리는 비율 |
|---|---|---|---|
| DeepSeek-V3 MTP-1 | 2 | 1.87 | 7% |
| Qwen3-8B · DFlash 블록 16 | 16 | 6.49 | **59%** |
| Qwen3-8B · EAGLE-3 트리 60 | 60 | 3.40 | **94%** |

배치가 작으면 남는 연산을 쓴 것이라 괜찮다. 배치가 크면 **다른 요청이 쓸 연산을 쓴 것**이다 (5절).

<details>
<summary>자세히 — 엔진은 실제로 어떻게 하나</summary>

**vLLM (v1)**
- 롤백: 요청의 `seq_lens`를 거부 수만큼 줄인다. 블록 표는 그대로다
- drafter의 attention 층은 따로 등록되지만 **target과 같은 블록 번호**를 쓴다 (`llm_base_proposer.py`). 💡 블록 하나를 잡으면 target 층과 drafter 층의 자리가 함께 잡힌다
- 초안은 argmax로 뽑아 `[배치, γ]`로 쌓는다. 병렬 drafter(`parallel_drafting`, DFlash)는 마스크 토큰 자리를 덧붙여 한 번에 낸다

**SGLang**
- 토큰 단위 자리 할당 + 요청마다 "토큰 → 자리" 표. 트리의 채택 경로가 흩어져 있어도 표가 가리키면 된다
- Spec V2(현재 기본)는 호스트의 뒷정리와 다음 배치의 자리 할당을 GPU 작업과 겹친다. 대신 `topk = 1`만 받는다. Qwen3-8B · B200 · 동시 32에서 처리량 11.4k → 15.3k 토큰/초 (LMSYS)
- DFlash는 target hidden을 drafter의 K · V로 바로 투영해 저장한다. prefix가 같은 요청끼리 radix cache로 함께 쓴다

**미리 잡아 둘 자리** — verify를 시작하려면 요청마다 `K`줄의 빈자리가 있어야 한다. 블록 단위 할당이면 경계를 넘을 때 새 블록을 먼저 잡는다.

</details>

---

## 3. 갈래가 여럿이면? — 트리 검증

`on` 다음이 `the`일지 `a`일지 모르겠다면 둘 다 초안으로 낼 수 있다. 초안이 **나무 모양**이 되고, 한 자리에 하나만 맞아도 된다.

![트리와 마스크](figures/11-tree-mask.png)

- **한 줄로 편다** — 노드 7개를 0 ~ 6번으로 늘어놓아 target에 넣는다
- **자기 조상만 보게 가린다** — `floor`는 `on` `the` `floor`만 본다. 옆 갈래의 `mat`은 다른 문장이다
- **위치는 깊이로 준다** — `the`와 `a`는 같은 위치다
- **채택된 경로의 줄만 남긴다** — 흩어져 있으니 자리 표로 골라 남긴다

| | 일렬 | 트리 |
|---|---|---|
| verify가 쓰는 줄 | `γ + 1` | 노드 수 (수십 개) |
| 롤백 | 길이만 되돌린다 | 흩어진 줄을 골라 남긴다 |
| 배치 1 | 좋다 | **더 좋다** |
| 큰 배치 | 좋다 | 버리는 줄이 너무 많다 (94%) |

💡 서빙 엔진이 일렬로 모이는 이유가 표의 아래 두 줄이다.

---

## 4. drafter도 캐시가 필요할까?

drafter도 문맥을 봐야 다음 토큰을 맞힌다. 02에서 본 "무엇을 받나"가 여기서 **캐시의 크기**가 된다.

![drafter별 캐시](figures/12-drafter-caches.png)

- 보라 막대가 target의 KV, 파란 막대가 drafter의 캐시다. **파란 막대의 폭이 층 수**다
- 별도 모델은 자기 층 전부, EAGLE-3 · MTP는 1층, DFlash · DSpark는 층 수만큼 둔다
- Gemma 4 MTP는 파란 막대가 없다 — target의 KV를 읽기만 한다

**EAGLE-3 · MTP — 확정된 자리는 진짜 hidden으로 다시 쓴다.**

초안을 만들 때 쓴 줄 중 스텝 2부터는 drafter 자기 hidden으로 만든 **추측 줄**이다. verify가 끝나면 `on` `the` `mat` 자리의 **target 진짜 hidden**이 생기므로, 다음 초안을 내기 전에 이 세 자리를 진짜 hidden으로 다시 계산해 덮어쓴다. 그 마지막 줄의 출력이 다음 사이클의 첫 초안이 된다.

**DFlash · DSpark — attention 없이 만든다.** target hidden을 층마다 K · V로 **투영만** 한 줄이라 prefill이 싸다. 대신 층 수만큼 저장한다 (5층이면 EAGLE-3의 5배). DSpark 서비스용은 최근 128줄만 둔다.

**Gemma 4 MTP — 저장하지 않는 대신 읽는다.** 캐시와 prefill은 0이지만, 초안 스텝마다 target의 KV를 읽는다. 💡 몇 개 층을 읽는지는 공개되지 않아 시뮬레이터에서는 매개변수로 둔다.

<details>
<summary>자세히 — drafter 캐시의 크기 (BF16)</summary>

| drafter | 캐시에 든 것 | 토큰당 크기 | prefill | 사이클마다 쓰는 줄 |
|---|---|---|---|---|
| 별도 draft 모델 | 자기 층 전부의 K · V | 그 모델의 KV 크기 | 자기 forward | `γ`줄 남짓 |
| EAGLE-3 | 1층의 K · V | Llama-3.1-8B용 4 KiB (target의 +3.1%) | 1층 forward | 진짜 `a + 1` + 추측 `γ − 1` |
| MTP (DeepSeek-V3) | MTP 블록의 MLA latent | 1.125 KiB (+1.6%) | 위와 같음 | 위와 같음 |
| Gemma 4 MTP | 없음 | 0 | 없음 | 이번 초안만 잠깐 |
| DFlash | 층마다 투영한 K · V | Qwen3-8B용 5층 20 KiB (+13.9%) | 투영만 | 진짜 `a + 1` × 층 수 |
| n-gram · suffix | 모델 캐시 없음 (CPU 토큰 열 · suffix tree, 또는 GPU 토큰 버퍼) | — | 없음 | 없음 |

계산: target은 Llama-3.1-8B 128 KiB(2 × 8 × 128 × 2 B × 32층), Qwen3-8B 144 KiB(× 36층), DeepSeek-V3 68.625 KiB(576 × 2 B × 61층).

</details>

---

## 5. 배치를 키우면 이득은 어떻게 될까?

실제 서버는 요청 수십 개를 한 forward에 묶는다. 그러면 01의 "곱셈은 원래 짧았다"가 달라진다.

forward 한 번이 처리하는 줄은 **모든 요청의 검증 토큰의 합**이다 (요청 32개 × 5자리 = 160줄). 이 줄 수가 약 300(H100의 균형점)에 다가가면 곱셈이 시간을 정하기 시작한다. speculative decoding이 쓰던 **남는 곱셈 능력을 배치가 먼저 써 버리는 것**이다.

![배치와 이득](figures/14-batch-speedup.png)

- 왼쪽: EAGLE(1세대)은 배치 24부터 점선 **아래** — 오히려 느려진다. EAGLE-3는 `τ`가 높아 배치 64에서도 1.38배다
- 오른쪽: DFlash도 동시 요청 1 → 32에서 5.1배 → 2.8배로 준다. 블록 16을 고정으로 검증하기 때문이다 — 이것을 부하에 맞추는 것이 DSpark다

**단, 컨텍스트가 짧을 때의 이야기다.** 컨텍스트가 길면 요청마다의 **KV 읽기**가 병목이 되는데, KV는 배치로 나눠 쓸 수 없다. 이때는 한 요청의 검증 토큰 `K`개가 그 요청의 KV 읽기를 나눠 쓰므로 큰 배치에서도 이득이 남는다 (📌 MagicDec, 2024). MLA는 예외 — 곱셈에 먼저 막힌다 (1절).

<details>
<summary>자세히 — 균형점과 배치 안에서 생기는 일</summary>

```
가중치 곱의 연산 강도 ≈ 줄 수 (FLOP / byte)       H100: 989 TFLOPS ÷ 3.35 TB/s ≈ 295     B200: 2.25 PFLOPS ÷ 8 TB/s ≈ 280
```

실제 커널은 작은 줄 수에서 효율이 덜 나와 꺾이는 지점이 더 앞에 온다.

| | |
|---|---|
| 토큰 예산 | 검증 토큰도 한 스텝의 처리 토큰 수에 들어간다. 💡 초안을 길게 잡으면 다른 요청의 prefill이 밀릴 수 있다 |
| 요청마다 다른 결과 | 요청마다 `a`가 달라 토큰 간격의 분산이 커진다 — p99는 평균만큼 좋아지지 않는다 |
| 요청마다 다른 검증 길이 | DSpark처럼 `K`가 다르면 검증 배치가 들쭉날쭉해진다. DSpark는 토큰을 펼쳐 한 줄로 처리하고, DeepSeek-V4에서 색인 attention과 압축 커널을 고쳤다 |

</details>

---

## 6. 모델 구조에 따라 깨지는 가정

지금까지의 두 가정 — **"가중치는 한 번만 읽는다"**, **"롤백은 길이만 되돌리면 된다"** — 이 깨지는 구조가 있다.

**MoE — 검증 토큰마다 다른 expert를 부른다.** DeepSeek-V3는 층마다 expert 256개 중 8개를 쓴다. decode에서 `on` 하나는 8개를 읽지만, verify에서 `on` `the` 둘은 각자 8개를 골라 최대 16개를 읽는다.

| 한 forward의 토큰 수 | 1 | 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|---|
| 불리는 expert 수 (라우팅이 고르다고 가정) | 8 | 15.8 | 30.5 | 57.4 | 102 | 222 |

- 배치 1에서 MTP-1을 검증하면 가중치 읽기가 약 1.54배가 되어 속도가 **약 1.2배**에 그친다. 배치가 커서 이미 expert를 거의 다 읽고 있다면 약 1.8배다 ([04](04-simulator.md#6-개략-검산--이-모형으로-공개-수치가-나오나)). 둘 다 가정 아래의 예시다
- 💡 **MoE에서는 배치가 작을 때 오히려 이득이 작다** — dense와 반대 방향이다

**희소 attention (DSA · CSA · QSA) — 검증 토큰마다 고르는 위치가 다르다.** 색인 연산이 `K`배, 읽는 줄은 `K`개 선택의 합집합이 된다. 💡 압축 캐시(CSA · HCA)는 검증 토큰이 채운 블록의 엔트리가 거부되면 다시 만들어야 한다.

**선형 attention 혼합 (Qwen3.5 · Kimi K3) — 길이로 되돌릴 수 없다.** 이 층은 과거를 고정 크기 상태 하나에 담고 토큰마다 **고쳐 쓴다.** 거부되면 `mat`까지 반영된 상태로 돌아가야 하는데, 이미 덮어썼다. 그래서 검증 토큰마다 중간 상태를 남겨 둬야 한다. 📌 vLLM에서 Qwen3.5에 n-gram speculation을 켜면 출력이 깨지는 버그가 실제로 보고됐다 (issue #39273).

**SWA** — 💡 오래된 줄을 지우는 캐시라면 창을 `W + K`줄로 잡아 둔다.

| 구조 | 깨지는 가정 | 시뮬레이터에서 셀 것 |
|---|---|---|
| MoE | 가중치는 한 번만 읽는다 | 한 forward의 토큰 수로 불리는 expert 수 |
| DSA · CSA · QSA | 캐시 읽기는 `K`와 무관하다 | 색인 연산 `K`배, 골라 읽는 줄의 합집합 |
| CSA · HCA | 롤백은 길이만 | 압축 엔트리를 다시 만드는 비용 |
| 선형 attention | 롤백은 길이만 | 중간 상태 `K`개 저장 |
| SWA | 롤백은 길이만 | 창 `W + K` |

---

## 7. 정리

1. **verify = decode의 새 토큰 줄을 `K`로.** 가중치 읽기는 그대로, 곱셈 · KV 쓰기 · 로짓은 `K`배. MLA의 attention은 곱셈에 먼저 막힌다
2. **target KV** — `K`줄을 쓰고 `a + 1`줄을 남긴다. 지우지 않고 길이를 되돌리거나 자리를 반납한다
3. **drafter 캐시** — 받는 것에 따라 크기가 다르다. EAGLE-3 · MTP는 확정된 자리를 진짜 hidden으로 다시 쓴다
4. **배치** — 남는 곱셈을 배치가 먼저 쓴다. 단 긴 컨텍스트에서는 이득이 남는다
5. **구조별 예외** — MoE의 expert 합집합, 희소 attention의 색인, 선형 attention의 상태

> **speculative decoding의 비용은 "verify가 decode보다 얼마나 비싼가"로 거의 다 설명된다.** 그 답이 배치 · 컨텍스트 길이 · 모델 구조에 따라 달라진다.

**다음 장** — 이 동작을 시뮬레이터의 상태와 비용 항목으로 옮긴다 → [04 시뮬레이터에 넣기](04-simulator.md)

---

## 참고

- [vLLM — llm_base_proposer.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py) — 일렬 초안, 첫 패스, target 블록 표 공유, 길이 되돌리기
- [vLLM — Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/) · [vLLM issue #39273](https://github.com/vllm-project/vllm/issues/39273) — 혼합 GDN 모델의 상태 롤백 부재
- [SGLang — Speculative Decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding) · [LMSYS — DFlash and Spec V2 (2026-06)](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- [EAGLE-3](https://arxiv.org/abs/2503.01840) — 배치별 처리량 표 · [DFlash](https://arxiv.org/abs/2602.06036) — 동시 요청 수별 속도 · [DSpark](https://arxiv.org/abs/2607.05147) — ragged 검증, 창 128
- [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) — target KV를 읽는 drafter
- [DeepSeek-V3 config](https://huggingface.co/deepseek-ai/DeepSeek-V3) — routed expert 256 · 상위 8, MLA 캐시 576
- [FlashAttention](https://arxiv.org/abs/2205.14135) — 타일 단위로 KV를 읽는 커널 · [MagicDec](https://arxiv.org/abs/2408.11049) — 긴 컨텍스트에서는 큰 배치에도 이득
