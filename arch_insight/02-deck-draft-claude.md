# Sparse · Hybrid Attention 구조 분석 및 아키텍처 시사점

차세대 아키텍처 탐색을 위한 KV cache · 색인 동작 특성 정량 분석 · 2026.11

> 슬라이드 원고 (Claude 버전). `---` 하나가 슬라이드 한 장이고, `# Ⅰ.` 같은 장 제목은 간지 한 장이다. 그림은 [figures-claude/](figures-claude/)에 있고 `python arch_insight/figures-claude/make_figures.py`로 다시 만든다 (계산 요약도 함께 출력).
> 계획과 기준: [00-deck-plan-claude.md](00-deck-plan-claude.md) · 비교 기준: **128K = 공통 비교 조건**, **1M = 차세대 설계 시나리오** (V3.2 · Qwen3.8은 지원 길이 밖 — 외삽)

---

## 요약

| 구분 | 핵심 인사이트 | 관련 |
|---|---|---|
| **구조 변화** | 전 층 full attention은 사라지는 중 — 선택 읽기 · 압축 · window · linear state를 **층마다 섞는 구조**가 주류 | 1.1 · 1.3 |
| **KV cache 절감** | 선택 읽기는 **읽기만**, 압축 · hybrid는 **크기까지** 감소 — 동시 요청 수를 늘리는 것은 크기를 줄인 방식 | 1.2 · 1.4 · Ⅱ |
| **Sparse 고유 비용 ① 색인** | 길이에 따라 늘어나는 것은 KV cache가 아니라 **색인 키** — 절감 효과의 상한을 정하고, 키를 줄일수록 대역폭 문제가 **연산 문제**로 이동 | 1.6 · Ⅲ |
| **Sparse 고유 비용 ② 흩어진 읽기** | 선택된 KV는 작은 조각 수천 개 — HBM에서는 **지연**, 대용량 메모리 계층에서는 **읽기 증폭 · 무작위 읽기 횟수**가 제약 | 2.3 |
| **Hybrid 고유 비용** | 고정 크기 state를 매 스텝 **전부 읽고 덮어씀** — 짧은 컨텍스트에서는 KV cache보다 state 갱신이 주된 메모리 트래픽 | 4.1 |
| **Sparse · Hybrid 분담** | Sparse는 **읽기**, Hybrid는 **크기**를 줄임 — 둘을 함께 써야 읽기 · 크기 모두 감소 | 4.2 |
| **아키텍처 시사점** | 병목이 KV cache 읽기에서 **색인 훑기 · top-k · 흩어진 읽기 · state 갱신**으로 이동 — 대역폭뿐 아니라 연산 · 지연 · 쓰기 대역폭을 함께 설계해야 함 | Ⅴ |

---

## 구성

| 장 | 내용 |
|---|---|
| **Ⅰ. 구조 개요** | Attention 구조 동향 · 방식별 KV cache 쓰기 · 읽기 · 층 구성 · 모델별 규모 · MoE 비교 · Sparse attention 디코드 읽기 구성 |
| **Ⅱ. KV cache** | 읽기 · 크기 절감 효과 · 컨텍스트 길이별 효과 · 접근 패턴과 읽기 증폭 |
| **Ⅲ. 색인** | 색인 키 읽기와 절감 포화 · 색인 연산 특성 |
| **Ⅳ. Hybrid** | 모델 전체 읽기 · 쓰기 구성 · Sparse와의 역할 분담 |
| **Ⅴ. 종합** | 데이터별 메모리 접근 특성 · 단계별 특성과 예상 제약 |

---

# Ⅰ. 구조 개요

Attention 방식별 KV cache 쓰기 · 읽기 구조와 모델별 규모, sparse attention의 디코드 읽기 구성

---

## 1.1 Attention 구조 동향

### 조사 대상 2026 하반기 15개 모델 중 전 층 full attention은 1개

![반기별 attention 구성](figures-claude/01-trend.png)

- 전 층 full attention 모델: 2025 하반기 7/14 → 2026 상반기 5/20 → **2026 하반기 1/15**
- 대체 방식은 회사별로 상이 — 선택 읽기(DSA 계열), 압축 저장(DeepSeek-V4), linear state 혼합(Qwen · Kimi · Nemotron), sliding window 혼합
- hybrid 모델도 일부 full 층은 KV cache를 전부 읽음 — "전부 읽기 탈피"는 층 단위로 진행 중

<sub>※ 출처: 주요 9개 시리즈 55개 공개 모델의 공식 config ([메모리 보고서](../reports/llm-memory-requirements.md) §1). 조사 대상 표본이며 '24 하반기 · '25 상반기는 모델 수가 적음. 모델별로 가장 먼저 해당하는 분류 하나로 집계.</sub>

---

## 1.2 방식별 KV cache 쓰기 · 읽기 구조

### 방식마다 KV cache에 쓰는 양 · 읽는 양 · 보관 크기가 다름 — 선택 읽기는 읽기만, 압축 · window는 크기까지, state는 쓰기가 큼

![방식별 KV cache 쓰기 · 읽기](figures-claude/02-kv-write-read.png)

- **선택 읽기:** 쓰기 · 크기는 Full과 같고(색인 키 +11%) 읽기만 감소 — 128K에서 색인 키 전부 + 2,048줄, Full 대비 1/8
- **압축 저장 · sliding window:** 보관하는 줄 수 자체를 줄여 크기 · 읽기 모두 감소 — 압축은 쓰기 시 압축 연산 추가, window는 W 밖 토큰 조회 불가
- **Linear state:** 크기는 길이와 무관하게 고정이지만 매 스텝 state 전체(KV 약 1,600줄 분량)를 읽고 덮어씀 — 쓰기가 가장 큰 방식

<sub>※ 기준: 층 1개 · 128K. 선택 읽기 = DSA(V3.2), 압축 = CSA 4:1 · HCA 128:1(DeepSeek-V4), window = gpt-oss · MiMo 128줄 · Muse 2,048줄, state = Qwen3.8 DeltaNet 층당 3.2 MB(Qwen3.8 full 층 KV 1줄 2,048 B로 환산). 출처: [세미나 자료](../seminar/01-attention-seminar_sum_ver.md) §4.3 그림 재구성, 그림 비율은 실제와 다름.</sub>

---

## 1.3 층 구성의 Hybrid화

### 단일 방식 대신 층마다 다른 방식을 혼합 — 한 모델 안에 서로 다른 메모리 접근 패턴이 공존

![층 구성](figures-claude/03-layer-strips.png)

- 2024년 모델은 전 층 동일 방식 → 2026년 모델은 **층마다 다른 방식을 교대 배치**
- DeepSeek-V4: 4줄 압축 층(선택 읽기)과 128줄 압축 층(전부 읽기)을 **1:1 교대** — 정밀한 부분 조회와 거친 전체 조회를 병행
- Qwen3.8: 48층 중 **12층만 KV cache 보유**, 36층은 고정 크기 linear state — 모델 전체 비용은 Ⅳ장에서 합산

<sub>※ 출처: 각 모델 config의 층별 설정 — V4-Pro `compress_ratios`, Qwen3.8-Flash-Next · gpt-oss `layer_types`.</sub>

---

## 1.4 모델 계열별 가중치와 KV cache 규모

### 1M 요청 1개를 처리하려면 가중치 + KV cache — KV cache 비중은 Full GQA 53%, 선택 읽기 11 ~ 25%, 압축 · linear state 1 ~ 4%

![모델 계열별 가중치와 KV cache](figures-claude/04-weights-vs-kv.png)

- **Full attention:** Hy3(GQA)는 KV cache 344 GB가 가중치 300 GB보다 큼 — 요청 1개가 필요 메모리의 절반 이상. 같은 Full이라도 MLA(V3)는 10%
- **선택 읽기:** GLM-5.3 · Hy4는 1M당 KV cache 100 GB(11 ~ 12%) — 읽기만 줄이고 크기는 그대로라 동시 요청 7 ~ 8개면 가중치와 같아짐
- **압축 · linear state:** KV cache 비중 1 ~ 4% — DeepSeek-V4 계열은 동시 84 ~ 92개가 모여야 가중치와 같아짐. **크기를 줄인 방식만 동시 요청 수가 크게 늘어남**

<sub>※ 출처: [메모리 보고서](../reports/llm-memory-requirements.md) 데이터. 필요 메모리 = 가중치 + 요청 1개의 KV cache (활성값 · 버퍼 제외). 가중치는 공식 체크포인트 크기(정밀도는 모델마다 다름), KV cache는 구조 비교용 BF16 기준, 1M = 1,048,576. 가중치는 모든 요청이 공유, KV cache는 요청마다 따로 필요.</sub>

---

## 1.5 MoE와의 비교

### Sparse attention은 'KV cache의 MoE' — 단, 선택 대상과 제약이 다름

![MoE와 sparse attention](figures-claude/05-moe-analogy.png)

- 공통점: 소형 선택기(router / indexer)가 **일부만 선택해 사용**, 미선택 expert · KV도 **전부 보유**
- 차이점: MoE는 고정된 expert 256개를 **전 요청이 공유**, sparse는 계속 증가하는 KV 100만 줄을 **요청별로 따로** 선택
- 제약: MoE는 소배치에서 가중치 대역폭, 대규모 expert 병렬에서 토큰 activation의 all-to-all 통신. sparse attention은 색인 훑기 · 흩어진 KV 읽기 · KV cache 용량

<sub>※ 기준: expert 1개 = DeepSeek-V3 3 × 7,168 × 2,048 파라미터 · FP8 ≈ 44 MB (가중치 읽기 단위). KV 1줄 = MLA 576차원 · BF16 = 1,152 B (층당). all-to-all이 옮기는 것은 가중치가 아니라 토큰 activation.</sub>

---

## 1.6 Sparse attention 디코드 읽기 구성

### KV cache 읽기는 2,048줄로 고정 — 길이에 비례해 늘어나는 것은 색인 키 읽기 (128K에서 84%)

![Sparse attention 디코드 읽기 구성](figures-claude/06-sparse-read-composition.png)

- **KV cache:** 선택된 2,048줄만 읽어 길이와 무관하게 0.14 GB 고정 — 디코드 1스텝 합계는 Full(MLA) 대비 128K 1/7, 1M 1/8 (Ⅱ장)
- **색인 키:** 컨텍스트 1토큰마다 132 B × 61층씩 늘어나 8K 31% → 128K 84% → 1M 93% — sparse attention 읽기의 대부분이자 절감 포화의 원인 (Ⅲ장)
- **기타:** top-k 점수 기록 · 읽기(토큰당 8 B)와 KV 추가 쓰기 — 5% 안팎이지만 색인 키와 함께 길이에 비례

<sub>※ 기준: DeepSeek-V3.2 61층 · 요청 1개 · 디코드 1스텝 · KV BF16 · 색인 키 132 B 계산값. top-k 점수는 FP32로 HBM에 기록했다가 다시 읽는 구현 기준 — 점수 계산과 top-k를 한 커널로 합치면 줄어듦. 1M은 지원 길이(160K) 밖.</sub>

---

# Ⅱ. KV cache

선택된 KV cache 읽기의 절감 효과, 컨텍스트 길이 의존성, 흩어진 읽기의 접근 특성

---

## 2.1 KV cache 읽기 · 크기 절감 효과

### 128K에서 디코드당 KV cache 읽기 1/8 ~ 1/128 — 선택 읽기는 KV cache 크기를 유지, 압축만 크기도 감소

![KV cache 읽기와 크기](figures-claude/07-kv-read-vs-size.png)

- **선택 읽기(DSA · QSA):** KV cache 읽기 1/8 · 1/21, 크기는 ×1.11 · ×1.03 (색인 키 추가분) — 읽기만 줄이는 방식
- **압축 저장(CSA · HCA):** 읽기 1/46 · 1/128, 크기 1/4 · 1/128 — 모델 전체로는 V4-Pro가 MLA 대비 128K 1/68, 1M 1/92 (CSA · HCA 층 교대 합산)
- **Linear state:** 층당 3.2 MB를 읽고 같은 양을 씀, 크기 고정 — 128K full 층(GQA 268 MB) 대비 읽기 1/84

<sub>※ 기준: 층 1개 · 128K · KV cache BF16 통일(구조 효과 비교), 색인 키는 공개 저장 형식(DSA FP8 + scale 132 B, CSA FP4 64 B, QSA 블록 키는 정밀도 미표기로 BF16 256 B 가정). 비율은 같은 회사의 전부 읽기 구조(V3 MLA · Qwen3.8 full 층 GQA) 대비. 읽기 막대의 옅은 색은 색인 키 — 비중은 Ⅲ장.</sub>

---

## 2.2 컨텍스트 길이별 KV cache 읽기 절감

### 선택형은 길어질수록 효과가 커지다 약 9×(DSA) · 32×(QSA)에서 포화 — 압축형은 짧은 컨텍스트에서도 효과

![컨텍스트 길이별 KV cache 읽기 절감](figures-claude/08-kv-read-saving-by-length.png)

- **읽기량 기준:** 디코드 1스텝에 읽는 바이트(색인 키 포함)의 Full 대비 배수 — 같은 메모리 대역폭이면 KV cache 읽기 시간도 같은 비율로 단축
- **선택형(DSA · QSA):** 8K에서 DSA 2.7× · QSA 3.6×, 128K 7.7× · 21× → 상한에서 포화. 2K 이하는 전부 선택되는 구간이라 색인만큼 손해 (실제로는 색인 생략 가능)
- **압축형(CSA · HCA):** 1K에서도 CSA 2.9× · HCA 8.5× — KV cache 자체가 작아 짧은 길이에서도 효과

<sub>※ 기준: 층 1개 · 디코드 1스텝 계산값. 실선 = 지원 길이 안, 점선 = 밖(외삽), ● = 지원 길이 끝(V3.2 160K · Qwen3.8 256K · V4 1M). 읽기 시간 환산은 대역폭이 같다는 가정 — 흩어진 읽기의 효율 저하는 2.3장.</sub>

---

## 2.3 KV cache 접근 패턴과 읽기 증폭

### 선택된 KV는 1.2 KB 조각 2,048개 — HBM에서는 손실이 없지만, 4 KB 페이지 계층에서는 전송량 3.6× · 초당 250만 회 무작위 읽기

![KV cache 접근 패턴과 읽기 증폭](figures-claude/09-access-pattern.png)

- **읽기 증폭:** 필요한 조각보다 큰 단위로 가져와 생기는 낭비 — 1.2 KB 조각마다 4 KB 페이지를 읽으면 전송량 3.6×(FP8 KV면 6.2×), 그만큼 링크 대역폭 소모
- **HBM:** 조각이 전송 단위(64 B)보다 커서 증폭 없음 — 대신 KV gather가 top-k 이후에만 시작하는 직렬 구조라 **메모리 지연 · 동시 요청 처리 수**가 관건
- **대용량 계층(CXL · SSD):** DSA 요청 1개가 초당 250만 회(61층 × 2,048조각 × 20 토큰/s) 무작위 읽기 — Gen5 NVMe 1개 최대 IOPS 수준. QSA처럼 4줄 블록(8 KB)으로 고르면 조각 수 1/4 · 증폭 1.0×

<sub>※ 기준: 층 1개 · 128K · KV BF16 계산값. 증폭 = 실제 전송 바이트 ÷ 필요한 바이트, 조각이 서로 붙어 있지 않다고 가정. HBM 전송 단위 64 B, 계층 4 KB 페이지, Gen5 NVMe 4 KB 무작위 읽기 약 250만 IOPS는 대표값. 초당 요청 수 = 요청 1개 · 20 토큰/s · 모델 전체(V3.2 61층, Qwen3.8 12층).</sub>

---

# Ⅲ. 색인

Sparse attention 고유 비용인 색인 키 — 절감 포화의 원인과 연산 · 대역폭 특성

---

## 3.1 색인 키 읽기와 절감 포화

### 길어질수록 선택된 KV 몫은 0에 수렴하고 토큰당 색인 키 132 B가 바닥으로 남음 — DSA 절감은 1,152 ÷ 132 = 8.7×에서 포화

![색인 키 읽기와 절감 포화](figures-claude/10-index-saturation.png)

- **포화 구조:** 토큰 1개당 읽기 = 색인 키 132 B + 선택된 KV 몫(1,152 B × 2,048 ÷ 길이) — 8K 420 B(2.7×) → 128K 150 B(7.7×) → 바닥 132 B(8.7×)
- **바닥을 낮추는 설계:** 색인 키 FP4 66 B(17×, 가정), 4줄 블록 색인 64 B(QSA 32×), 블록 + FP4 16 B(CSA 72×) — 차기 경쟁 포인트는 **토큰당 색인 바이트**
- **정밀도 효과:** KV만 FP8(656 B)로 줄이면 같은 색인에서 상한 5.0× — 색인 키가 KV 1줄의 20%가 되므로 KV와 색인 키를 함께 줄여야 함

<sub>※ 기준: DSA(V3.2) 층 1개 · 디코드 1스텝 · KV BF16 계산값. 상한은 컨텍스트 → ∞ 극한. 블록 색인은 블록 키 1개가 4토큰을 대표하므로 토큰당 색인 바이트 = 블록 키 ÷ 4. QSA 상한은 GQA 2,048 B 기준. 2K 이하는 전부 선택되어 Full보다 읽기가 큼 (그림 왼쪽 위).</sub>

---

## 3.2 색인 연산 특성

### 색인 키 훑기는 대역폭과 연산을 함께 요구 — DSA는 읽기 시간의 30%만큼 FP8 연산, 본 attention 연산의 3.8배

![색인 연산 특성](figures-claude/11-index-compute.png)

- **대역폭:** 128K 층당 색인 키 17 MB를 매 스텝 순차로 읽음 — H200 HBM으로 3.6 µs, 같은 층의 선택된 KV 읽기(2.4 MB)의 7배
- **연산:** 키 1 B당 124 FLOP(DSA) · 256 FLOP(CSA) — indexer head 64개가 같은 키를 각각 곱하기 때문. HBM 속도로 훑으려면 FP8 피크의 30 ~ 62%, MTP 2토큰이면 CSA는 **연산 병목(124%)**. QSA는 head 4개라 2%
- **3.1과의 연결:** 색인 키를 FP4 · 블록으로 줄이면 읽기는 줄지만 바이트당 연산은 커짐 — **대역폭 문제가 연산 문제로 이동**. prefill에서도 색인 연산이 dense의 1/5이라 절감 상한 약 5배

<sub>※ 기준: 층 1개 · 128K 디코드 1스텝, 색인 키 저장 바이트 기준(DSA 132 B, CSA 64 B, QSA 블록 키 256 B). H200 HBM 4.8 TB/s, FP8 1,979 · BF16 989 TFLOPS(dense) — QSA는 BF16 키라 BF16 피크, CSA FP4 키는 H200에 FP4 연산기가 없어 FP8 피크 적용. 본 attention = head 128 × (576 + 512) × 2 FLOP × 2,048줄. prefill은 61층 합계, dense = head 128 × (K 192 + V 128).</sub>

---

# Ⅳ. Hybrid

Linear state 층과 attention 층을 섞은 모델의 전체 읽기 · 쓰기, sparse와의 역할 분담

---

## 4.1 Hybrid 모델 디코드 읽기 · 쓰기 구성

### Hybrid 고유 비용은 state 갱신 — Qwen3.8은 128K까지 디코드 I/O의 60% 이상이 state 읽기 · 쓰기

![Hybrid 모델 디코드 읽기 · 쓰기 구성](figures-claude/12-hybrid-total.png)

- **state 비중:** 8K 80%, 32K 75%, 128K 60% → 1M 22% — 짧은 컨텍스트일수록 KV cache보다 state가 주된 메모리 트래픽
- **전 층 full GQA 대비:** 128K에서 디코드당 읽기 12.9 GB → 382 MB (1/34)
- **쓰기:** state 덮어쓰기 115 MB vs KV · 색인 추가 25 KB — 쓰기 대역폭과 온칩 처리가 hybrid의 별도 요구 사항

<sub>※ 기준: Qwen3.8-Flash-Next 모델 전체(QSA 12층 + DeltaNet 36층), 요청 1개 · 디코드 1스텝, KV BF16 · state FP32(config `mamba_ssm_dtype`) 계산값. 1M은 지원 길이(256K) 밖.</sub>

---

## 4.2 Sparse와 Hybrid의 역할 분담

### Sparse는 읽기를, Hybrid는 크기를 줄임 — 128K에서 둘을 합쳐야 읽기 3% · 크기 27%

![Sparse와 Hybrid 비교](figures-claude/13-sparse-vs-hybrid.png)

- **Sparse만 (전 층 QSA, 가정):** 읽기 + 쓰기 5%로 크게 줄지만 KV cache 크기는 103% — 색인 키만큼 오히려 증가
- **Hybrid만 (GQA 12층 + state 36층, 가정):** 크기 26% · 읽기 27% — KV cache를 가진 층이 1/4로 줄어든 만큼 둘 다 감소하지만, 읽기는 여전히 길이에 비례
- **결합 (Qwen3.8 실제):** 읽기 3% · 크기 27%. 단, 32K에서는 Sparse만(302 MB)과 읽기가 거의 같음 — state 읽기 · 쓰기 231 MB가 KV 절감분을 상쇄, hybrid의 읽기 이득은 128K 1.6배 · 256K 2.1배로 긴 컨텍스트에서 나타남

<sub>※ 기준: Qwen3.8-Flash-Next 차원(48층, full 층 KV head 2 × 256, QSA 블록 키 BF16 256 B 가정, DeltaNet state FP32 3.2 MB/층)으로 층 구성만 바꾼 계산값. 요청 1개 · 디코드 1스텝 · KV BF16. Sparse만 · Hybrid만은 가정 구성. %는 전 층 Full 대비.</sub>

---

# Ⅴ. 종합

데이터별 메모리 접근 특성과 sparse attention 단계별 특성 · 예상 제약

---

## 5.1 데이터별 메모리 접근 특성

### 가중치는 공유, KV cache · 색인 키 · state는 요청마다 — 매 스텝 읽고 쓰는 비율이 데이터마다 다름

![데이터별 메모리 접근 특성](figures-claude/14-memory-behavior.png)

- **가중치 (V3.2):** 689 GB 중 배치 1은 활성분 38 GB(5%)만 읽고, 배치 32면 expert 합집합이 커져 434 GB(63%) — 쓰기 없음, 모든 요청이 공유
- **KV cache:** 보유량 중 매 스텝 1.6%만 읽고(V3.2 82 MB / 5.24 GB) 끝에 1줄만 추가 — 크지만 드물게 읽는 데이터 → 용량 계층 후보
- **색인 키 · state:** 색인 키는 매번 100% 읽고 쓰기는 1개, state는 매번 100% 읽고 100% 덮어씀(Qwen3.8 115 MB) — 둘 다 요청마다 따로라 동시 요청 수만큼 대역폭 증가

<sub>※ 기준: 요청 1개 · 128K · 디코드 1스텝 계산값. V3.2는 실제 배포 형식(가중치 FP8, KV FP8 656 B/줄, 색인 132 B/줄), Qwen3.8은 KV BF16 · state FP32(config `mamba_ssm_dtype`) · 블록 키 BF16 가정. 가중치 읽기는 균등 routing의 expert 합집합.</sub>

---

## 5.2 단계별 특성과 예상 제약

### 디코드 attention의 이론 시간은 색인 키 훑기가 대부분 — 나머지 단계는 바이트 · FLOP에 잡히지 않는 top-k · 지연이 관건

![단계별 특성과 예상 제약](figures-claude/15-stage-summary.png)

- **색인 키 훑기:** 17 MB 순차 읽기 + 2.1 GFLOP — H200 이론 시간 3.6 µs로 나머지 단계 합(약 1.1 µs)의 3배 이상, 대역폭과 FP8 연산을 함께 요구
- **top-k · KV gather:** 바이트로는 작지만(0.2 ~ 0.3 µs) 비교 · 정렬과 직렬 대기는 이론 시간에 잡히지 않음 — 온칩 SRAM · 동기화 · 메모리 동시 요청 수가 관련 요소
- **검증 계획:** 128K에서 단계별 실측 시간(색인 훑기 / top-k / gather · attention)과 HBM 트래픽을 측정해 이론 시간과 차이가 큰 단계를 우선 최적화 대상으로 선정

<sub>※ 기준: DSA(V3.2) 층 1개 · 128K · 실제 배포 형식(KV FP8 656 B, 색인 132 B). 시간 = 바이트 ÷ 4.8 TB/s 또는 FLOP ÷ 피크(FP8 1,979 · BF16 989 TFLOPS)의 이론값. attention은 FP8 KV를 BF16으로 역양자화해 계산하는 구현 기준 — 역양자화 처리량이 병목이 되는 사례가 FlashMLA 분석 문서에 보고됨.</sub>

---

## 부록 A. 방식별 비교

| | 전부 읽기 (MLA · GQA) | 선택 읽기 (DSA · QSA) | 압축 저장 (CSA · HCA) | Sliding window | Linear state |
|---|---|---|---|---|---|
| 매 스텝 쓰기 | 1줄 | 1줄 + 색인 키 1개 | 4 · 128토큰마다 1줄 | 1줄 추가 · 1줄 폐기 | state 전체 |
| KV cache 크기 | `S` | `S` + 색인 키 (+3 ~ 11%) | `S/4` · `S/128` | `W` 고정 | 고정 (state) |
| 색인 비용 (쿼리당) | 없음 | 색인 키 `S`개 (QSA `S/4`) | CSA 색인 키 `S/4` · HCA 없음 | 없음 | 없음 |
| 색인 산술 강도 | — | DSA 124 · QSA 4 FLOP/B | CSA 256 FLOP/B | — | — |
| attention 계산 (쿼리당) | `S` | 2,048 고정 | CSA 1,024 + 최근 128 · HCA `S/128` + 128 | `W` | 고정 |
| 원본 KV 정확 조회 | 가능 | 가능 (선택 시) | 불가 (최근 128줄만 원본) | `W` 안만 가능 | 불가 |
| 디코드당 읽기 (128K, 층당) | 151 ~ 268 MB | 13 ~ 20 MB | 1.2 ~ 3.3 MB | `W`줄 (W = 128 ~ 2,048) | 3.2 MB + 동일량 쓰기 |
| 절감 상한 | — | DSA 9× · QSA 32× | CSA 72× · HCA 144× | `S/W` | — |
| 추가 연산 | — | 색인 훑기 · top-k · 흩어진 읽기 | 쓰기 시 압축 · top-k · 흩어진 읽기 | 없음 (full 층과 혼합) | 매 스텝 state 갱신 |
| 기존 모델 적용 | — | 추가 학습으로 전환 가능 (V3.1 → V3.2) | 설계 단계부터 적용 | 설계 단계부터 적용 | 설계 단계부터 적용 |

---

## 부록 B. 계산 기준

| 항목 | 기준 |
|---|---|
| 단위 | 층 1개 · 요청 1개 · 디코드 1스텝. 1M = 100만 토큰, 1 GB = 10⁹ B (1.4장만 메모리 보고서 기준 1M = 1,048,576) |
| 비교 길이 | 128K = 공통 비교 조건, 1M = 차세대 설계 시나리오. 지원 길이: V3 · V3.2 163,840 · Qwen3.8-Flash-Next 262,144 · V4-Pro 1,048,576 — 밖은 점선 · * 표시 |
| 구조 비교 (1.2 · 1.6 · Ⅱ ~ Ⅳ장) | KV cache BF16 통일. 색인 키는 공개 저장 형식 — DSA FP8 128 B + FP32 scale 4 B = 132 B, CSA FP4 64 B, QSA 블록 키는 정밀도 미표기로 BF16 256 B 가정 |
| 배포 형식 (5.1 · 5.2) | DeepSeek-V3.2 실제 형식 — 가중치 FP8 689 GB, KV FP8 656 B/줄(FlashMLA 형식), 색인 132 B/줄 |
| 이론 시간 (3.2 · 5.2) | H200 HBM 4.8 TB/s, FP8 1,979 · BF16 989 TFLOPS (dense). 바이트 ÷ 대역폭, FLOP ÷ 피크 |
| top-k 점수 | FP32로 HBM에 기록했다가 다시 읽는 구현 기준 (토큰당 8 B · 층) |
| 가중치 읽기 | 균등 routing에서 배치 N의 expert 합집합 (비 expert 17.1B + 58층 × 합집합 × 44 MB) |
| state | Qwen3.8-Flash-Next DeltaNet 층당 3.2 MB (FP32 state 48 × 128 × 128 + conv), 36층 115 MB |
| 초당 요청 수 (2.3) | 요청 1개 · 20 토큰/s · 모델 전체 층 |
| 미포함 | top-k · 흩어진 읽기의 커널 비효율, 활성값 · 통신 버퍼, FFN 연산 · 통신 시간. 읽기는 KV를 한 번씩 읽는 하한 |
| 정확도 | 분석 범위 외 |

---

## 부록 C. 확인이 필요한 항목

| 항목 | 현재 처리 | 확인 방법 |
|---|---|---|
| QSA 블록 키 정밀도 | BF16 256 B 가정 | Qwen3.8 기술 보고서 · 추론 코드 |
| top-k 점수 기록 방식 | FP32 HBM 기록 가정 | 실제 색인 커널(점수 계산 · top-k 결합 여부) 확인 |
| 대용량 계층 사양 (2.3) | 4 KB 페이지 · Gen5 NVMe 약 250만 IOPS 대표값 | 제품 사양서 (CXL 모듈 · NVMe) |
| 실측 비용 | 없음 (계산값만) | V3.2 기술 보고서의 토큰 위치별 prefill · decode 비용 그래프 확인 후 2.2장 옆 패널로 추가 |
| 단계별 시간 | 없음 (이론 시간만) | 128K에서 색인 훑기 / top-k / gather · attention 시간과 HBM 트래픽 실측 |
