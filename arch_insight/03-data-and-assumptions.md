# 발표 자료의 수치·출처·비교 조건

현재 [발표 자료](02-deck-draft.md)는 **요약 + 본문 8장**이다. 수치는 [revised-deck-costs.csv](data/revised-deck-costs.csv), 표본 집계는 [revised-deck-summary.json](data/revised-deck-summary.json)에 기록했다.

## 1. 발표에서 비교하는 범위

- 기준은 **층 1개·요청 1개·Decode 1스텝**이며, 데이터 성격 표에서는 모델 전체 묶음을 별도로 표시한다.
- **1M=1,000,000토큰, 1 MB=10⁶ B, 1 GB=10⁹ B**다. 8K=8,192, 128K=131,072다.
- Main KV는 BF16으로 맞추고, **색인은 DSA FP8+scale / CSA FP4 payload / QSA BF16 가정**을 사용한다. 순수한 구조만의 절감률이 아니라, 명시된 구조·색인 정밀도 시나리오 비교다.
- 보유 데이터와 조회 payload를 한 번씩 센 **계산값**이다. 실제 HBM 트래픽·GPU 속도·모델 전체 성능이 아니다.

## 2. 구성과 출처

| 대상 | 파라미터·근거 |
|---|---|
| DSA·Full MLA | latent 512 + RoPE 64, index dim 128, top-k 2,048, backbone 61층. [config 스냅샷](../reports/llm-memory-requirements/data/raw/cfg_deepseek-ai_DeepSeek-V3.2.json), [공식 코드](https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/inference/model.py) |
| CSA·HCA | shared KV 512차원, 압축 4/128, CSA top-k 1,024, 최근 창 128. backbone은 `compress_ratios[:61]`의 CSA 30층·HCA 31층. [config](../reports/llm-memory-requirements/data/raw/cfg_deepseek-ai_DeepSeek-V4-Pro.json), [구조 §2.3·FP4 §5.2.1](https://arxiv.org/html/2606.19348v1) |
| QSA·Full GQA | KV heads 2 × dim 256, index dim 128, block 4, 2,048토큰 예산. [config](../reports/llm-memory-requirements/data/raw/cfg_Qwen_Qwen3.8-Flash-Next.json), [QSA §2.1.2](https://arxiv.org/html/2608.30320v1) |
| Qwen Hybrid | `layer_types`에 GDN 36층·조회층 12층. 조회층은 최종 학습에서 QSA로 교체. GDN은 value heads 48, key heads 16, key/value dim 128, 행렬 state FP32, conv kernel 4. 같은 config·원문 §2.1 |
| 적용 추이 | [기존 조사 CSV](../reports/llm-memory-requirements/data/lineage-kv1m.csv)의 9개 시리즈 55개 모델. 각 모델은 정해진 대표 분류 하나로 집계. 분류가 복합 구조의 모든 속성을 표현하는 것은 아니며 시장 전체 표본도 아님 |

반기별 검산:

| 반기 | Full만 | Sparse·압축 | State 혼합 | Window 혼합 | 합계 |
|---|---:|---:|---:|---:|---:|
| 2024 H2 | 1 | 0 | 0 | 0 | 1 |
| 2025 H1 | 2 | 0 | 3 | 0 | 5 |
| 2025 H2 | 7 | 2 | 4 | 1 | 14 |
| 2026 H1 | 5 | 6 | 7 | 2 | 20 |
| 2026 H2 | 1 | 4 | 6 | 4 | 15 |

## 3. 1M에서의 층당 수치

| 방식 | 읽기 = 색인 + 본 KV·state | 보유량 | 기준 Full 대비 읽기 절감 |
|---|---:|---:|---:|
| Full MLA | 0 + 1,152 MB | 1,152 MB | 1× |
| DSA | 132 + 2.359 MB | 1,284 MB | 8.57× ≈ 9× |
| CSA | 16 + 1.180 MB | 272.131 MB + 미산정 항목 | 67.06× |
| HCA | 0 + 8.131 MB | 8.131 MB + 미산정 항목 | 141.69× |
| Full GQA | 0 + 2,048 MB | 2,048 MB | 1× |
| QSA | 64 + 4.194 MB | 2,112 MB | 30.03× |
| GDN | 0 + 3.207 MB | 3.207 MB | 읽기 외 같은 양의 상태 쓰기 |

DeepSeek 계열의 기준은 Full MLA, Qwen 계열의 기준은 같은 KV heads·차원의 Full GQA다. 서로 다른 폭·압축 손실의 모델이므로 절감 배수가 같은 정확도나 성능 순위를 뜻하지 않는다.

색인이 차지하는 읽기 비중은 DSA **98.2%**, CSA **93.1%**, QSA **93.8%**다. 이는 시간 비중이 아니다.

## 4. 계산식과 미산정 항목

`S`: 유효 토큰 수. Main KV는 MLA 1,152 B/토큰, GQA 2,048 B/토큰, V4 엔트리 1,024 B다.

| 방식 | 조회 읽기량 식 |
|---|---|
| Full MLA | `S × 1,152` |
| DSA | `S × 132 + min(S,2048) × 1,152` |
| CSA | `floor(S/4) × 64 + min(floor(S/4),1024) × 1,024 + min(S,128) × 1,024` |
| HCA | `floor(S/128) × 1,024 + min(S,128) × 1,024` |
| Full GQA | `S × 2,048` |
| QSA | `floor(S/4) × 256 + (선택한 완성 블록의 토큰 수 + tail) × 2,048` |
| GDN | `48×128×128×4 + (2×16×128 + 48×128)×3×2` |

- **DSA scale 포함:** 128차원 FP8 128 B + 참조 코드의 FP32 scale 4 B = 132 B/토큰. KV 본체는 비교용 BF16이다.
- **CSA scale 미산정:** FP4 index payload를 64 B로 센다. 양자화 scale·패딩과 압축 대기 상태는 미산정이므로 CSA/HCA 값은 전체 캐시 할당량보다 작을 수 있다.
- **QSA dtype 가정:** 블록 키의 저장 dtype을 확정하지 못해 BF16을 가정한다. 불완전 블록의 tail은 포함하며 1M은 block 4의 경계라 tail이 0이다. pooling 보조 상태는 미산정이다.
- **GDN 상태 모델:** 행렬 state FP32 + conv 직전 3칸 BF16을 한 번 읽고 한 번 쓴다고 둔다. 온칩 상주·fusion이 있으면 실제 HBM 이동은 달라진다.
- **공통 제외:** 가중치·projection·FFN·통신·임시 score/top-k 버퍼·페이지 과다 읽기·KV 추가 쓰기·할당 예약. 공유 latent가 여러 번 읽히는 커널의 비용도 별도다.

읽기 절감의 점근 상한은 DSA `1152/132=8.73×`, QSA `2048/(256/4)=32×`, CSA `1152/(64/4)=72×`다. 본문은 발표용 반올림을 사용한다.

## 5. 길이·동시성·속도 해석

- DSA config 길이는 163,840, QSA는 262,144, V4는 1,048,576이다. 1M의 막대는 **같은 길이에서 방식을 비교하는 계산**이며 DSA·QSA의 기본 config 범위를 넘는다. 곡선에서 해당 구간을 점선으로 표시했다.
- 선택된 KV 안에는 원본 정보가 남지만, 선택에서 빠진 정보를 회수할 수 있다는 뜻은 아니다. 압축·state의 정보 손실도 달라 별도 품질 검증이 필요하다.
- 상태는 독립 요청 수에 비례한다. GDN 36층 합계 **115.46 MB/요청**, 한 층은 **3.207 MB/요청**이다. B=8일 때 전체 상태는 약 924 MB, 현재 층 8요청은 약 25.66 MB다.
- MoE의 활성 파라미터 비율은 실제 HBM 읽기 비율과 동일하지 않다. 배치의 전문가 합집합·재사용·캐시·실행 경로에 따라 달라져 메모리 동작 표에서는 고정 읽기 비율을 쓰지 않았다.
- 읽기 payload 감소를 속도 향상으로 환산하지 않았다. 선택·주소·gather와 연산 처리량을 포함한 단계별 측정이 다음 작업이다. Prefill은 query 간 재사용이 달라 이 decode 식을 그대로 적용하지 않는다.

## 6. 재현·그림 출처

```powershell
python arch_insight/figures/make_revised_figures.py
```

- 그림 1~4는 [Claude 초안](02-deck-draft-claude.md)의 추이·개념·층 구성 표현을 별도 파일로 재사용했다. 원본 폴더를 수정하지 않는다.
- 그림 5~7은 같은 계열별 색과 직접 숫자 표기 방식으로 다시 작성했다. [그림 생성기](figures/make_revised_figures.py)는 이 계산 CSV와 그림을 함께 재생성한다.
- 현재 본문은 `figures/revised-*`, `data/revised-deck-*`를 사용한다. 기존의 `cost_model.py`와 BF16 통제 데이터는 이전 분석 시나리오이며 현재 본문 수치와 혼합하지 않는다.
