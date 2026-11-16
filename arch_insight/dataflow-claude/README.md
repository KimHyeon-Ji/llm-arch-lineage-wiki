# 데이터 플로우 — Sparse · Hybrid attention 아키텍처 아이디에이션

[기준 보고서](../sparse-hybrid-hw-report-v2-refined.md)에서 출발해, 구조별 연산과 데이터 이동을 커널 단위로 그린다.
목표는 "데이터가 어디서 읽혀, 어디에 머물고, 무엇이 계산하는지"를 보면서 HW 배치를 바꿔 보는 것이다.

| 모델 | 상태 | 뷰어 |
|---|---|---|
| [DSA · DeepSeek-V3.2](models/dsa-v32/) | 파일럿 완료 — 참조 구현 검증 · 수치 · 뷰어 | [viewer/dsa-v32.html](viewer/dsa-v32.html) · [게시본](https://claude.ai/artifact/3YVxmZwKAEJXpf8sYtZ42k) |
| Gated DeltaNet (Qwen3.8 · Qwen3-Next) | 예정 | |
| QSA 블록 선택 (Qwen3.8) | 예정 | |
| 층 조합 (Qwen3.8 · QSA 12 + DeltaNet 36) | 예정 | |
| 비교 기준: 전 층 Full (MLA · GQA) | 예정 | |

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
| 위치 | HBM(커널 경계를 넘는 값 · 상주 데이터) / 온칩(커널 안에서만 쓰는 값) / 연산기 |
| View A | 커널을 위→아래(시간 순)로 나열. 왼쪽 HBM 수명선(● 읽기 ○ 쓰기), 띠 굵기 ∝ √바이트, 오른쪽 이론 하한 |
| View B | 선택 단계의 입력 → 연산 → 출력 텐서 격자. 토큰 축(T · k)은 항상 가로, 긴 축은 40칸으로 줄임. B 축 생략 |
| 이론 하한 | max(바이트 ÷ HBM 대역폭, FLOP ÷ 피크). 실행 지연 · 동기화 · gather 비효율 미포함 |

## 구조

```
dataflow-claude/
  README.md              이 문서 — 원칙 · 규약 · 진행 상황
  tools/dflow.py         공용: 단계 · 텐서 기록기(Rec), aten 연산 · GEMM FLOP 로그, 식 계산
  viewer/template.html   뷰어 템플릿 (모델 데이터가 /*__DATA__*/ 자리에 들어감)
  viewer/<model>.html    build.py가 만든 공유용 뷰어 — 완전한 HTML 문서, 브라우저 · GitHub Pages에서 바로 열림
  viewer/artifact/       같은 뷰어의 claude.ai artifact 게시용 본문 (직접 고치지 않음)
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
