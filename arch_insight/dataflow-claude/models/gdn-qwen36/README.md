# Gated DeltaNet — Qwen3.6-27B linear attention 층

Hybrid 모델의 linear attention 층. KV cache 대신 요청마다 고정 크기 state(head 48개 × 128 × 128)를 두고, 매 스텝 전부 읽고 다시 쓴다. 층 1개 · decode 1스텝.

- 뷰어: [../../viewer/gdn-qwen36.html](../../viewer/gdn-qwen36.html) · [GitHub Pages](https://kimhyeon-ji.github.io/llm-arch-lineage-wiki/arch_insight/dataflow-claude/viewer/gdn-qwen36.html) · [claude.ai](https://claude.ai/artifact/W2dGw8hpqSRW7Aj5JKPtz5)
- 수치: [numbers.md](numbers.md)

## 왜 Qwen3.6-27B 차원인가

- 기준 보고서의 Hybrid 수치는 Qwen3.8-Flash-Next(state FP32 48 × 128 × 128 ≈ 층당 3.2 MB) 기준이다.
- Qwen3.8 config는 로컬에 없어서, **value head 48 · 128 × 128 · FP32 state**가 같은 Qwen3.6-27B를 썼다. state 크기는 보고서와 같고(3.21 MB/층), hidden 폭 · 가중치 크기는 Qwen3.6-27B 값이다.

## 차원

| 기호 | 값 | 뜻 |
|---|---:|---|
| `d` | 5,120 | hidden |
| `n_k` · `n_v` | 16 · 48 | key head · value(state) head. key head 1개를 value head 3개가 공유 |
| `d_k` · `d_v` | 128 · 128 | head 폭 → state 1개 = 128 × 128 |
| `d_conv` · `C` | 4 · 10,240 | 짧은 합성곱 탭 · q·k·v 채널 |
| `state_b` | 4 B (FP32 배포) / 2 B (BF16 가정) | state 원소 정밀도 |
| `n_lin` | 48 | linear 층 수 (64층 중, 4층마다 full 1층) |

## 단계와 커널 (현재 GPU 배치)

| 커널 | 단계 | HBM에서 읽는 주 데이터 | 칩 안 메모리에 두는 것 |
|---|---|---|---|
| K1 입력 투영 GEMM | l1 | W_qkv · W_z · W_b · W_a | — |
| K2 짧은 합성곱 | l2 · l3 | conv state 61 KB (읽고 덮어씀) | conv 창 (레지스터) |
| K3 state 갱신 | l4 ~ l8 | **state 3.15 MB (읽고 덮어씀)** | state 블록 · 감쇠한 state · 기억 읽기 값 · 새 state (레지스터) |
| K4 gate 정규화 | l9 | o · z | — |
| K5 출력 투영 GEMM | l10 | W_out | — |

## 참조 구현과 HF의 차이

| | HF `Qwen3_5GatedDeltaNet` | `ref_decode.py` (배포 커널 기준) |
|---|---|---|
| conv state | 4칸 | 3칸(K − 1) — `causal_conv1d_update`와 같음 |
| 비교 방식 | 전체 시퀀스 chunk gated delta rule | 1스텝 recurrent를 토큰마다 반복 |

`test_ref.py` 결과 (작은 차원 · 랜덤 가중치 · T = 80):
- state FP32: 모든 위치 최대 상대 오차 4.4e-6
- state BF16: 출력 상대 오차 중앙값 9e-4 · 최대 7.5e-3 — BF16 state 버튼의 정확도 대가, 정보용

## 관찰 (요청 1개 · H200 이론 하한)

수치는 모두 [numbers.md](numbers.md)에서 왔다.

- **linear 층은 T와 무관하다.** 층 1개 HBM 이동 238 MB가 4K에서도 1M에서도 같다. 같은 자리가 full attention 층이었다면 128K에서 KV 537 MB를 매 스텝 읽는다. state 왕복 6.29 MB는 그 1.2%다.
- **요청 1개에서는 가중치가 97%다**(232 MB). 요청별 이동(6.59 MB)이 가중치를 넘는 동시 요청 수는 **36개부터**이고, T와 상관없다.
- **state 갱신 커널(K3)은 순수 대역폭 문제다.** state 3.15 MB를 읽고 같은 양을 다시 쓰며(1.32 µs), 연산은 state 1 B당 1 FLOP 미만이다. Tensor core는 쓰지 않는다(행렬 × 벡터 · rank-1 갱신).
- **동시 요청이 많아지면 state 왕복이 커진다.** 64개면 매 층 403 MB. 칩 안 SRAM(레지스터 합 34.6 MB)에 다 둘 수 없어, state를 메모리 안에서 갱신하는 PIM이나 큰 SRAM이 후보가 된다(보고서 2.4).
- **conv state도 매 스텝 왕복한다.** 61 KB로 작지만, 한 칸 밀기 위해 전부 읽고 다시 쓴다.

## 남은 확인

- Qwen3.8-Flash-Next 실제 config(hidden 폭 · key head 수)로 바꿔 가중치 크기 재계산
- 커널 묶음: SGLang · vLLM의 Qwen3-Next 경로에서 K3 · K4 융합 여부와 in_proj 융합 형태 확인
- state BF16 저장의 정확도 영향은 작은 차원 랜덤 가중치 기준 — 실제 가중치로 재확인 필요

## 출처

- HF transformers 5.14 `models/qwen3_5/modeling_qwen3_5.py` — `Qwen3_5GatedDeltaNet.forward`(L441–563), `torch_recurrent_gated_delta_rule`(L329–370), `torch_causal_conv1d_update`(L224–239)
- Qwen3.6-27B `config.json` (text_config)
- fla `fused_recurrent_gated_delta_rule` · `causal_conv1d_update` — 배포 decode 커널
- 기준 보고서 [§1.6 · §2.4 · 부록 B](../../../sparse-hybrid-hw-report-v2-refined.md)
