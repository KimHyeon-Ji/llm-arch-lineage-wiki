# DSA — DeepSeek-V3.2

lightning indexer(색인) + top-k 2,048줄 선택 + 흡수형 MLA. 층 1개 · decode 1스텝 · attention 블록.

- 뷰어(약식): [../../viewer/dsa-v32-lite.html](../../viewer/dsa-v32-lite.html) · [GitHub Pages](https://kimhyeon-ji.github.io/llm-arch-lineage-wiki/arch_insight/dataflow-claude/viewer/dsa-v32-lite.html) · [claude.ai](https://claude.ai/artifact/PbR2Jbzr7ePzxU1KviAFPH)
- 수치: [numbers.md](numbers.md)

## 차원

| 기호 | 값 | 뜻 |
|---|---:|---|
| `d` | 7,168 | hidden |
| `c_q` · `c_kv` | 1,536 · 512 | q · KV latent |
| `n_h` · `d_nope` · `d_rope` · `d_v` | 128 · 128 · 64 · 128 | attention head |
| `n_I` · `d_I` | 64 · 128 | 색인 head |
| `k` | 2,048 | top-k |
| `kv_row` | 656 B (FP8 배포) / 1,152 B (BF16 비교) | KV 1줄 |
| `idx_row` | 132 B | 색인 키 1줄 (128 FP8 + scale) |

## 단계와 커널 (현재 GPU 배치)

| 커널 | 단계 | HBM에서 읽는 주 데이터 | 칩 안 메모리에 두는 것 |
|---|---|---|---|
| K1 입력 투영 | p1 | W_qa · W_kva | — |
| K2 KV 정규화 · 쓰기 | p2 · p3 | kv_a | KV 새 줄(레지스터) → 1줄 쓰기 |
| K3 q 업투영 | p4 | W_qb | — |
| K4 q 흡수 | p5 | W_UK | — |
| K5 색인 투영 | i1 ~ i4 | W_Iq · W_Ik · W_Iw | 색인 k 새 줄(레지스터) → 1줄 쓰기 |
| K6 색인 점수 | i5 · i6 | **색인 K cache T줄 전부** | 색인 키 타일(SMEM) · head별 점수 [64 × T](레지스터) |
| K7 top-k | s1 | 색인 점수 [T] | — |
| K8 Sparse MLA | a1 ~ a4 | **KV 2,048줄 gather** | 선택 KV(SMEM) · 점수 · 확률(레지스터) → split 부분 출력은 HBM에 씀 |
| K8b split 합치기 | a5 | split 부분 출력 | — |
| K9 W_UV 풀기 | o1 | W_UV | — (o 32 KB는 HBM 경유) |
| K10 출력 투영 | o2 | W_o | — |

## 참조 구현과 HF의 차이

| | HF `modeling_deepseek_v32` | `ref_decode.py` (배포 커널 기준) |
|---|---|---|
| KV cache | head 128개로 풀어 K · V 저장 | latent 576값(c_kv 512 + k_rope 64)만 저장 |
| sparse 선택 | 전체 T에 -inf 마스크 | 선택 2,048줄을 gather |
| 색인 정밀도 | BF16 · FP32, Hadamard 생략 | 선택 사항: Hadamard + FP8(블록 128, ue8m0) |

`test_ref.py` 결과 (작은 차원 · 랜덤 가중치 · T = 128):
- fp8=False: 128개 위치 모두 top-k 집합이 HF와 같고, 출력 최대 상대 오차 1.2e-6
- fp8=True: top-k 겹침률 평균 97.7%(최소 87.5%), 출력 상대 오차 중앙값 3.7e-2 — 배포 정밀도 영향, 정보용

## 관찰 (128K · KV FP8 · H200 이론 하한)

수치는 모두 [numbers.md](numbers.md)에서 왔다 (build.py가 생성).

- **요청 1개에서는 가중치가 지배한다.** attention 블록 가중치 219 MB > 요청별 이동 37.3 MB.
  요청별 이동이 가중치를 넘는 동시 요청 수는 **10개부터**다(32K는 27개, 1M은 2개부터).
  split 부분 출력은 B에 비례하지 않아서, 비례식 대신 B = 1, 2, 3 …을 차례로 넣어 처음 넘는 값을 찾았다.
- **요청별 이동은 색인 키 스트리밍과 split 부분 출력이 반씩이다.** 색인 키 17.3 MB(K6, 46%), split 부분 출력 8.4 MB × 쓰기·읽기 = 16.8 MB(K8 → K8b, 45%).
  KV gather(K8)는 1.34 MB로 T와 무관하다.
- **K6는 메모리 한계**(3.72 µs vs 연산 1.34 µs).
- **K8도 요청이 적으면 메모리 한계다**(2.06 µs vs 연산 0.58 µs). FlashMLA가 요청 1개의 2,048줄을 SM 66개 파트에 나눠(split 32개) 계산하고,
  FP32 부분 출력 [128 × 512]를 split마다 HBM에 썼다가 combine 커널(K8b, 1.78 µs)이 다시 읽기 때문이다.
  H200 · 2,048줄 기준 요청이 64개 전후가 되면 split이 사라지고, head 128개가 같은 KV 줄을 공유해 연산 쪽으로 넘어간다.
- **직렬 동기화 지점은 K6 → K7 → K8 → K8b.** 점수 [T]를 HBM에 썼다 다시 읽고(524 KB × 2), top-k가 끝나야 gather 주소가 정해지며, split 부분 출력도 HBM을 한 번 왕복한다.
- **qr · x는 여러 커널이 다시 읽는다**(K3 → K5, K1 · K5). 수명선이 짧고 작아 융합하면 칩 안 공유 후보다.

## 남은 확인

- split 수: FlashMLA 공개 main의 head128 sparse decode(SM100) 규칙(SM ÷ 2 파트, 64줄 블록, 고정 비용 3)을 H200 132 SM에 적용했다. V3.2 배포 당시 SM90 경로의 실제 파라미터와 대조 필요
- head 배치: 128 head가 GPU 1개에 있다는 가정(DP attention). TP로 나누면 K8 · 같아지는 B가 달라진다

- 커널 묶음은 SGLang · vLLM 경로를 대표로 단순화했다. 실제 융합 경계와 커널 수는 프레임워크 코드 기준으로 재확인 필요
- W_UK · W_UV 저장 정밀도(BF16 가정)와 FlashMLA FP8 KV 형식(656 B)의 공식 문서 대조
- GLM-5.2는 같은 DSA를 쓰면서 층 간 색인 공유(`indexer_types`)를 둔다 — 색인 비용을 층 사이에 나누는 변형으로 다음 비교 후보

## 출처

- HF transformers 5.14 `models/deepseek_v32/modular_deepseek_v32.py` — `DeepseekV32Indexer.forward`(L206–271), `DeepseekV32Attention.forward`(L284–356)
- DeepSeek-V3.2-Exp 공식 `inference/model.py` — Indexer(`rotate_activation` · `act_quant` · `fp8_index` · `k_cache` · `k_scale_cache`)
- 기준 보고서 [§1.4 · §1.5 · 부록 B](../../../sparse-hybrid-hw-report-v2-refined.md)
