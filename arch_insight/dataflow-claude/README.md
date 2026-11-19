# 데이터 플로우 — Sparse · Hybrid attention 아키텍처 아이디에이션

[기준 보고서](../sparse-hybrid-hw-report-v2-refined.md)에서 출발해, 구조별 연산과 데이터 이동을 커널 단위로 그린다.
목표는 "데이터가 어디서 읽혀, 어디에 머물고, 무엇이 계산하는지"를 보면서 HW 배치를 바꿔 보는 것이다.

| 모델 | 상태 | 자세한 버전 | 약식 (흐름 한눈에) | 보고서 |
|---|---|---|---|---|
| [DSA · DeepSeek-V3.2](models/dsa-v32/) | 완료 — 참조 구현 검증 · 수치 · 뷰어 (split-KV combine 포함) | [viewer/dsa-v32.html](viewer/dsa-v32.html) · [Pages](https://kimhyeon-ji.github.io/llm-arch-lineage-wiki/arch_insight/dataflow-claude/viewer/dsa-v32.html) · [claude.ai](https://claude.ai/artifact/3YVxmZwKAEJXpf8sYtZ42k) | [viewer/dsa-v32-lite.html](viewer/dsa-v32-lite.html) · [Pages](https://kimhyeon-ji.github.io/llm-arch-lineage-wiki/arch_insight/dataflow-claude/viewer/dsa-v32-lite.html) · [claude.ai](https://claude.ai/artifact/PbR2Jbzr7ePzxU1KviAFPH) | [report/dsa-v32.md](report/dsa-v32.md) |
| [Gated DeltaNet · Qwen3.6-27B](models/gdn-qwen36/) | 완료 — 참조 구현 검증 · 수치 · 뷰어 (Hybrid의 linear 층) | [viewer/gdn-qwen36.html](viewer/gdn-qwen36.html) · [Pages](https://kimhyeon-ji.github.io/llm-arch-lineage-wiki/arch_insight/dataflow-claude/viewer/gdn-qwen36.html) · [claude.ai](https://claude.ai/artifact/W2dGw8hpqSRW7Aj5JKPtz5) | [viewer/gdn-qwen36-lite.html](viewer/gdn-qwen36-lite.html) · [Pages](https://kimhyeon-ji.github.io/llm-arch-lineage-wiki/arch_insight/dataflow-claude/viewer/gdn-qwen36-lite.html) · [claude.ai](https://claude.ai/artifact/G5BhyqZ3qwNwHtNEAGHZ1f) | [report/gdn-qwen36.md](report/gdn-qwen36.md) |
| QSA 블록 선택 (Qwen3.8) | 예정 | | | |
| 층 조합 (Qwen3.8 · QSA 12 + DeltaNet 36) | 예정 | | | |
| 비교 기준: 전 층 Full (MLA · GQA) | 예정 | | | |

보고서 = 모델 전체 decode 1스텝 중 attention 비중 + 요청 수 · 문맥 길이별 '흐름 한눈에' + 약식 화면 커널별 이동 스냅샷 분석 + 커널별 특성(병목 위치 · 접근 모양 · 재사용) → 필요한 HW 특성 + 계산값과 실제의 차이(다시 읽기 · 패딩 · L2) + 아키텍처 아이디어 후보.

---

## 원칙

1. **HF 코드를 그대로 그리지 않는다.** HF는 sparse 선택을 dense attention + 마스크로, MLA는 KV를 head별로 풀어서 구현한다.
   배포 커널(FlashMLA sparse, DeepGEMM 색인 커널)과 데이터 이동이 전혀 다르다.
   → 커널 동작대로 짠 **참조 구현**을 만들고, HF와 **출력이 같은지 테스트**한다.
2. **알고리즘과 배치를 나눈다.**
   - `algo.yaml` — HW와 무관. 텐서(shape · 바이트 식 · 종류)와 단계(입력 · 출력 · FLOP 식).
   - `map-*.yaml` — 배치. 커널 묶음, 커널별 HBM 읽기 · 쓰기, 온칩에 머무는 텐서, HW 메모.
   - 아이디어 = 새 `map-*.yaml`. 같은 알고리즘에 배치만 바꿔 이동량 차이를 본다.
3. **식은 실행으로 검증한다.** `build.py`가 참조 구현을 실제 차원(T = 131,072)의 meta device로 돌려
   yaml의 shape · GEMM FLOP 식이 실행 결과와 같은지 검사한다. 메모리는 쓰지 않는다.
4. **수치는 손으로 쓰지 않는다.** `numbers.md`와 뷰어 수치는 yaml 식에서 계산한다.

## 표기 규약

| 항목 | 규약 |
|---|---|
| 기본 조건 | 층 1개 · decode 1스텝 · 요청 1개 · 128K. 보고서 부록 B와 같은 저장 형식 |
| 데이터 종류 (색) | 가중치(회색, 공유) · KV cache(파랑) · 색인 K cache(주황) · 활성값(청록) · 중간 점수(노랑) · 선택 위치(자홍) |
| 연산기 (배지) | `GEMM` 행렬곱 · `VEC` 원소별 · 정규화 · 축약 · `TOP-K` 비교 · 선택 · `GATHER` 흩어진 행 읽기 · `WRITE` 캐시 쓰기 |
| 위치 | HBM(커널 경계를 넘는 값 · 상주 데이터) / 칩 안 메모리 — 공유 메모리 SMEM(타일 · 모은 줄) · 레지스터(누산값 · 원소별 값) / 연산기. L2 캐시는 HBM 이동에 포함 |
| View A | 커널을 위→아래(시간 순)로 나열. 왼쪽 HBM 수명선(● 읽기 ○ 쓰기), 띠 굵기 ∝ √바이트, 오른쪽 이론 하한 |
| View B | 선택 단계의 입력 → 연산 → 출력 텐서 격자. 토큰 축(T · k)은 항상 가로, 긴 축은 40칸으로 줄임. B 축 생략 |
| L2 표시 | 약식 화면에서 상주 데이터(가중치 · cache · state)가 아니고 L2 절반(25 MB) 이하인 값에 `L2` — 앞뒤 커널이 주고받는 작은 값이라 실제로는 L2에서 끝날 수 있음. 화살표 굵기는 그대로 |
| 약식 | 결론 · 설명 없이 흐름만. 위 "흐름 한눈에" = 단계별 HBM 읽기(진한 막대) · 쓰기(옅은 막대), 높이 ∝ 바이트. 아래 "커널별 이동" = HBM 레인 · 화살표(→ 읽기 · ← 쓰기, 굵기 ∝ √바이트) · 칩 레인. 커널 칸을 누르면 텐서 모양(`[B, T, d_I] = [1, 131072, 128]`)이 펼쳐짐 |
| 이론 하한 | max(바이트 ÷ HBM 대역폭, FLOP ÷ 피크). 실행 지연 · 동기화 · gather 비효율 미포함 |

## 구조

```
dataflow-claude/
  README.md              이 문서 — 원칙 · 규약 · 진행 상황
  tools/dflow.py         공용: 단계 · 텐서 기록기(Rec), aten 연산 · GEMM FLOP 로그, 식 계산
  tools/buildlib.py      공용 빌드: yaml 읽기 · 차원 · 커널 수치 · numbers 표 · 뷰어 출력 (모델 build.py가 사용)
  viewer/template.html   뷰어 템플릿 — 모델 데이터가 /*__DATA__*/ 자리에 들어감. 제목 · 문구 · 색 · 조건 버튼은 algo.yaml의 page · kinds · formats에서 읽음
  viewer/template-lite.html  약식 뷰어 템플릿 — 같은 데이터, 흐름 한눈에 + 커널별 이동만. 제목은 algo.yaml page.lite
  viewer/<model>.html    build.py가 만든 공유용 뷰어 — 완전한 HTML 문서, 브라우저 · GitHub Pages에서 바로 열림 (<model>-lite.html = 약식)
  viewer/artifact/       같은 뷰어의 claude.ai artifact 게시용 본문 (직접 고치지 않음)
  report/<model>.md      모델별 보고서 — 그림 report/figures/, 모델 전체 수치 model_totals.py, 그림 · 스냅샷 make_report.py
  models/<model>/
    README.md            차원 · 범위 · 출처 · 관찰
    ref_decode.py        커널 동작 기준 참조 구현
    test_ref.py          HF 구현과 출력 비교
    algo.yaml            HW 무관 연산 그래프
    map-gpu.yaml         현재 GPU 배치
    build.py             검증 → numbers.md · dataflow.json · viewer
    numbers.md           계산 결과 (생성물)
    dataflow.json        뷰어 데이터 (생성물)
```

## 실행

transformers 5.x(`deepseek_v32` 포함) + torch 환경. 지금은 llm-arch-tracer의 `.venv`를 쓴다.

```
cd models/dsa-v32
python test_ref.py     # 참조 구현 = HF 출력 확인
python build.py        # 검증 + numbers.md + viewer/dsa-v32.html
```
