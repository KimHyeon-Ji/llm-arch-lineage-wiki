# 아키텍처 설계 조건 분석 — 계산 근거

현재 [6장 발표 자료](02-deck-draft.md)는 기존 기법 설명에 이어 **어떤 조건에서 설계 아이디어가 성립하는지**를 분석한다. 아래 전환점·필요 BW·I/O 비중·SRAM 작업 집합은 원문에 실린 실측값이 아니라 **이 문서의 계산 결과**다.

## 1. 구성과 공통 가정

| 대상 | 사용한 구성 | 근거 |
|---|---|---|
| DSA | MLA latent 512 + RoPE 64, top-k 2,048, 61층. 본 KV BF16=1,152 B/토큰, 색인 FP8 128 B+FP32 scale 4 B | [공식 config 스냅샷](../reports/llm-memory-requirements/data/raw/cfg_deepseek-ai_DeepSeek-V3.2.json), [공식 코드 Indexer](https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/inference/model.py) |
| QSA | KV heads 2×256, index dim 128, block 4, top-k 2,048토큰. KV 2,048 B/토큰, 색인 BF16 256 B/블록 가정 | [공식 config 스냅샷](../reports/llm-memory-requirements/data/raw/cfg_Qwen_Qwen3.8-Flash-Next.json), [원문 §2.1.2](https://arxiv.org/html/2608.30320v1) |
| CSA | shared KV 512차원, 4:1 압축, top-k 1,024, 최근 창 128. KV BF16 1,024 B/엔트리, 색인 FP4 payload 64 B/블록 | [공식 config 스냅샷](../reports/llm-memory-requirements/data/raw/cfg_deepseek-ai_DeepSeek-V4-Pro.json), [구조 §2.3·FP4 §5.2.1](https://arxiv.org/html/2606.19348v1) |
| GDN·Hybrid | 36 GDN+12 QSA. GDN 행렬 state FP32, value heads 48, key heads 16, key/value dim 128, conv kernel 4 | 같은 Qwen config와 원문 §2.1.1 |

- **Decode 분석**이다. 한 쿼리는 한 요청의 새 토큰 한 개이며 모든 query head를 포함한다.
- 32K=32,768, 128K=131,072, 256K=262,144다. 1 MB=10⁶ B, 1 GB=10⁹ B, 1 MiB=2²⁰ B다.
- 본문 주요 비교는 config 지원 범위 안이다. DSA는 128K까지, QSA hybrid는 256K까지 계산했다.
- KV를 한 번 읽고 공유 헤드가 재사용한다고 센 논리 payload다. 실제 HBM 트래픽은 커널·cache·fusion에 따라 달라진다.
- QSA 색인의 BF16은 가정이고, CSA 색인의 양자화 scale·패딩·압축 대기는 미산정이다. 정밀도가 바뀌면 수치도 바뀐다.
- Prefill·가중치·FFN·통신·projection·softmax/top-k 임시 공간·할당 예약은 포함하지 않았다. Byte 비중으로 시간 병목을 단정하지 않는다.

## 2. 색인과 선택 KV의 읽기량이 같아지는 길이

선택 예산이 차 있는 구간에서 다음과 같다. `S`는 토큰 수다.

| 방식 | 색인 읽기 | 선택 KV 읽기 | 두 값이 같아지는 S |
|---|---|---|---:|
| DSA | `132×S` | `2048×1152` | 17,873 |
| QSA | `256×floor(S/4)` | `2048×2048` | 65,536 |
| CSA | `64×floor(S/4)` | `(1024+128)×1024` | 73,728 |

CSA의 선택 KV 항에는 최근 원본 창을 포함했다. 블록 반올림의 미세한 차이는 발표용 전환 길이에서 무시했다. 128K의 색인 비중은 DSA 88.0%, QSA 66.7%, CSA 64.0%다.

이는 **읽기 byte의 교차점**이다. selector의 내적·비교 처리량, gather의 낮은 실효 BW, kernel 호출 지연은 별도다. 메모리 근처 selector의 필요 연산 처리량도 함께 산정해야 한다.

## 3. 용량 계층의 BW 필요조건

DSA·128K·61층·KV BF16에서 본 KV 보유량은 `61×131072×1152 = 9.211 GB/요청`, 색인 보유량은 `61×131072×132 = 1.055 GB/요청`이다.

선택된 본 KV 조회량은 `P = 61×2048×1152 = 143.917 MB/스텝/요청`이다. 모든 선택 KV를 외부 계층에서 읽고 fast-tier hit가 없다고 가정한다.

```text
외부 BW 필요량 = P × 동시 요청 B × 요청당 생성 속도 r × 읽기 증폭 A
```

**r=50 token/s/요청**이라는 설계 목표에서:

| 동시 요청 | A=1 | A=4 |
|---|---:|---:|
| 1 | 7.20 GB/s | 28.78 GB/s |
| 8 | 57.57 GB/s | 230.27 GB/s |
| 32 | 230.27 GB/s | 921.07 GB/s |

50 token/s, 증폭 1·4배, 외부 예산 100 GB/s는 **설계 시나리오**다. 실제 제품 사양·측정 결과를 뜻하지 않는다. 여기의 A는 페이지 과다 읽기·추가 전송 등에 대한 감도 변수다.

외부 공급 비율을 f라 두면, 같은 증폭 가정에서 `f ≤ 가용 BW / (P×B×r×A)`다. B=8, A=4, 가용 BW=100 GB/s이면 **f≤43.43%**다. 이는 보유 KV를 얼마나 외부에 둘 수 있는지가 아니라, **그 시점의 선택 payload 중 외부에서 공급할 수 있는 최대 비율**이다. 공간 배치를 결정하려면 선택의 시간적 재사용·hot-set 추적이 추가로 필요하다.

이는 모델 복제본 전체의 **합산 BW 필요조건**이며, 특정 GPU 한 링크의 요구량이 아니다. 왕복 지연·동기화·분산 배치·다른 트래픽·온칩 cache 효과를 더해야 실제 성립 여부를 판단할 수 있다.

## 4. Hybrid의 조회·상태 I/O

GDN 한 층의 상태:

```text
행렬 state = 48×128×128×4 B = 3,145,728 B
conv 상태 = (2×16×128 + 48×128)×3×2 B = 61,440 B
합계 = 3,207,168 B = 3.058594 MiB
```

행렬 state는 config의 FP32, conv는 직전 3칸 BF16으로 모델링했다. 한 스텝에 상태를 한 번 읽고 한 번 쓰는 기준에서는 36개 GDN 층의 R+W가 `36×2×3,207,168 = 230.916 MB`다.

| 문맥 | 12 QSA 조회 | 36 GDN R+W | 모델링한 합 | state 비중 |
|---|---:|---:|---:|---:|
| 32K | 75.497 MB | 230.916 MB | 306.414 MB | 75.4% |
| 128K | 150.995 MB | 230.916 MB | 381.911 MB | 60.5% |
| 256K | 251.658 MB | 230.916 MB | 482.574 MB | 47.9% |

QSA에는 전체 블록 색인과 선택 원본 KV 읽기를 포함했다. GDN state의 실제 HBM 왕복·내부 중복 읽기는 측정하지 않았다. 따라서 **60%는 모델 전체 시간이나 실측 HBM 트래픽의 비중이 아니다**. 상태가 길이와 무관하다는 것이 갱신 이동량도 작다는 뜻은 아님을 보이는 기준 계산이다.

## 5. 현재 층 작업 집합과 다음 토큰까지의 상주

**전용 state SRAM 32 MiB**라는 가상 설계 예산을 둔다. activation·연산 scratch·KV tile 등은 이 예산과 별도다.

| 범위 | 1요청 | 8요청 |
|---|---:|---:|
| 현재 GDN 층 1개의 state | 3.059 MiB | 24.469 MiB |
| GDN 36층의 state 전부 | 110.109 MiB | 880.875 MiB |

현재 층만 보면 최대 `floor(32/3.058594)=10요청`의 state가 맞는다. 하지만 모든 층을 거친 후 다음 토큰에서 다시 쓰려면 다른 층들의 state도 살아 있어야 한다. **현재 층이 맞는다는 사실만으로 매 토큰의 HBM load/store가 없어지지 않는다.**

온칩 처리의 우선 목표는 현재 층 갱신에서의 중복 read/write를 줄이는 것이다. 토큰 사이 영구 상주는 별도 총용량·대체 정책·스케줄링 문제다. 요청 묶음을 작게 만들면 가중치 재사용이나 처리량에 손해가 생길 수 있어 함께 평가해야 한다.

## 6. 재현

```powershell
python arch_insight/analysis/architecture_tradeoffs.py
python arch_insight/figures/make_tradeoff_figures.py
```

현재 본문은 `figures/decision-*`와 아래 데이터만 사용한다.

- [architecture-tradeoffs.json](data/architecture-tradeoffs.json): 주요 값, 입력 파라미터, config SHA-256.
- [tradeoff-selector.csv](data/tradeoff-selector.csv): 길이별 색인·선택 KV 바이트.
- [tradeoff-tier-bandwidth.csv](data/tradeoff-tier-bandwidth.csv): batch·읽기 증폭·필요 BW.
- [tradeoff-hybrid-io.csv](data/tradeoff-hybrid-io.csv): 문맥별 Hybrid I/O.
- [tradeoff-state-sram.csv](data/tradeoff-state-sram.csv): 상태 생존 범위별 용량.

기존 기법·추이 정리는 저장소의 세미나와 다른 초안에 남아 있다. 현재 본문 네 그래프는 위 수치를 계산해 직접 그렸고, 다른 초안의 그림을 불러오거나 복사하지 않는다.
