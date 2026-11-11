"""HW · 시스템 아이디어 정리(arch_insight/hw-arch-ideas-claude.md)용 그림 생성기.
실행: python arch_insight/figures-claude/hw-ideas/make_hw_ideas_figures.py [그림 이름 ...]  → 같은 폴더에 *.svg, *.png
SVG 도구 · 색 · 비용 함수는 상위 폴더 make_figures.py(덱 그림)를 그대로 쓴다 — 구조 특징 수치는 덱과 같은 계산값.
"""
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(HERE))
from make_figures import (Fig, INK, MUTED, AXIS, SLATE, BOX, CHROME,  # noqa: E402
                          FULL, FULL_L, SEL, SEL_L, CMP, CMP_L, STATE, STATE_L,
                          MLA_B, DSA_IDX, TOPK, K128, GDN_STATE, hybrid_parts, _cfg)

MB = 1e6


def text_w(s, size):
    return sum(size if ord(c) > 0x2000 else size * 0.55 for c in s)


def head(f, x, y, tag, title, col):
    """패널 머리 — 왼쪽 띠 + '구조 특징' / '아이디어' 꼬리표 + 제목."""
    f.rect(x, y - 21, 6, 28, col, rx=2)
    f.text(x + 16, y, tag, 16, weight=700, color=col)
    f.text(x + 16 + text_w(tag, 16) + 14, y, title, 18, weight=700)


def big_arrow(f, x1, x2, y):
    f.raw(f'<polygon points="{x1},{y - 9} {x2 - 18},{y - 9} {x2 - 18},{y - 20} {x2},{y} {x2 - 18},{y + 20} '
          f'{x2 - 18},{y + 9} {x1},{y + 9}" fill="{SLATE}"/>')
    f.text((x1 + x2) / 2 - 6, y - 30, "그래서", 14, "middle", 700, SLATE)


def thick(f, x1, y, x2, col, sw=7):
    """가로 굵은 화살표."""
    d = 1 if x2 > x1 else -1
    f.line(x1, y, x2 - d * 14, y, col, sw)
    f.raw(f'<polygon points="{x2},{y} {x2 - d * 18},{y - 11} {x2 - d * 18},{y + 11}" fill="{col}"/>')


def legend(f, x, y, items, size=14, gap=26):
    for col, lab in items:
        f.rect(x, y - 12, 16, 16, col, rx=3)
        f.text(x + 24, y + 2, lab, size)
        x += 24 + text_w(lab, size) + gap


# ---------------------------------------------------------------- 그림 0. 구조 특징 → HW 아이디어 지도
def fig_map():
    f = Fig("00-map", 1700, 800)
    f.text(40, 44, "Sparse · Hybrid 구조 특징 → HW 아이디어 — 특징마다 맞는 아이디어가 다름", 26, weight=700)
    f.text(40, 78, "구조 특징 수치 = 덱 계산값 (128K · 요청 1개 · 디코드 1스텝) · 스텝 간 재사용 = HiSparse 실측 · 제안 상태 = 2026-11-11 검색 기준",
           17, color=MUTED)
    for x, h in ((40, "대상"), (150, "구조 특징 (덱 근거)"), (690, "HW 아이디어"), (1130, "대표 제안"), (1490, "상태")):
        f.text(x, 128, h, 16, weight=700, color=MUTED)
    f.line(40, 142, 1660, 142, AXIS, 1)
    rows = [
        ("Sparse", SEL_L, "KV cache 크기 그대로 · 매 스텝 1.6%만 읽음", "고른 위치는 스텝 간 약 70% 재사용", "덱 2.1 · 5.1 · HiSparse",
         "① 계층형 KV 배치", "전체 KV는 host DRAM · CXL, HBM은 hot 캐시", "HiSparse · ECHO · ESS · SAC", "SGLang 반영 · OSDI 2026", 0),
        ("Sparse", SEL_L, "색인 키를 매 스텝 전부 읽고 위치 2,048개만 출력", "선택된 KV는 1.2 KB 조각 2,048개로 흩어짐", "덱 2.3 · 3.2",
         "② 메모리 근처 선택 · attention", "메모리 옆에서 고르고 모아 결과만 이동", "NELSSA · LongSight · PAM", "SK hynix PNM · MICRO 2025 · 2026", 1),
        ("Hybrid", STATE_L, "state는 고정 크기지만 매 스텝 전부 읽고 덮어씀", "115 MB 읽기 + 같은 양 쓰기 · 원소 단위 연산", "덱 Ⅳ · 5.1",
         "③ state 갱신 PIM", "state를 메모리 밖으로 꺼내지 않고 갱신", "Pimba · NOVA · HyPIM", "MICRO 2025 · 2026", 1),
        ("공통", FULL_L, "한 모델 안에서 층마다 방식 · 병목이 다름", "prefill = 연산, decode = 메모리 · 지연", "덱 1.3 · 6.5",
         "④ 층별 · 단계별 이종 가속기", "연산 종류 · 단계마다 다른 HW", "HLX · NOVA · DUET · NELSSA", "MICRO 2025 · 2026 · DAC 2026", 1),
        ("Sparse", SEL_L, "색인 연산이 본 attention의 3.8배", "top-k는 비교 · 정렬 · 동기화 중심", "덱 3.2 · 5.2",
         "색인 · top-k 전용 HW", "점수 계산과 top-k를 한 유닛에서", "sparse 모델 대상 제안 미확인", "dense 대상 PADE(HPCA 2026)만", 2),
    ]
    status = [("실서비스 진입", SEL, "#ffffff", None), ("학회 제안", SEL_L, INK, None), ("공백", "#ffffff", CMP, CMP)]
    y, RH = 154, 120
    for tag, tcol, a1, a2, ref, idea, isub, props, psub, st in rows:
        f.rect(40, y + 14, 92, 32, tcol, rx=6)
        f.text(86, y + 36, tag, 15, "middle", 700)
        f.rect(150, y + 4, 500, RH - 14, BOX[0], BOX[1], 8, 1)
        f.text(168, y + 36, a1, 17, weight=700)
        f.text(168, y + 62, a2, 15, color=INK)
        f.text(168, y + 88, ref, 13, weight=700, color=SLATE)
        f.arrow(656, y + RH / 2 - 4, 684, y + RH / 2 - 4, MUTED, 2.5)
        if st == 2:
            f.rect(690, y + 4, 420, RH - 14, "#ffffff", CMP, 8, 1.6, "6 4")
        else:
            f.rect(690, y + 4, 420, RH - 14, "#ffffff", BOX[1], 8, 1.2)
            f.rect(690, y + 4, 7, RH - 14, SEL if tag != "Hybrid" else STATE, rx=3)
        f.text(712, y + 44, idea, 19, weight=700, color=CMP if st == 2 else INK)
        f.text(712, y + 74, isub, 15, color=MUTED)
        f.text(1130, y + 44, props, 16, weight=700, color=CMP if st == 2 else INK)
        f.text(1130, y + 70, psub, 13, color=MUTED)
        lab, fill, tc, dash = status[st]
        f.rect(1490, y + 26, 170, 36, fill, dash or "none", 18, 1.6, "5 3" if dash else None)
        f.text(1575, y + 50, lab, 15, "middle", 700, tc)
        y += RH
    return f


# ---------------------------------------------------------------- 그림 1. 계층형 KV 배치
def fig_tiered():
    f = Fig("01-tiered-kv", 1700, 530)
    f.text(40, 44, "① 계층형 KV 배치 — 안 읽는 KV는 큰 용량 계층으로, 다시 고를 KV만 HBM에", 26, weight=700)
    f.text(40, 78, "DeepSeek-V3.2 · 128K · 요청 1개 · KV FP8 656 B/줄 (덱 5.1) · 스텝 간 재사용 = GLM-5.1 측정 (HiSparse)", 17, color=MUTED)
    head(f, 40, 136, "구조 특징", "KV cache는 크지만 매 스텝 일부만 읽고, 고른 위치는 반복", SLATE)
    x0, W = 240, 520
    # 매 스텝 읽는 KV — 1.6%
    f.text(40, 196, "매 스텝 읽는 KV", 16, weight=700)
    f.text(40, 218, "요청당 5.24 GB 중 1.6%", 13, color=MUTED)
    f.rect(x0, 178, W, 36, FULL_L, rx=3)
    rnd = random.Random(5)
    for p in rnd.sample(range(10, 230), 4):
        f.rect(x0 + p * W / 240, 178, 2.2, 36, SEL, rx=0)
    f.text(x0, 234, "진한 선 = 이번 스텝에 읽는 KV (폭 비율 그대로)", 13, color=MUTED)
    # 스텝 t / t+1 — 선택 위치 70% 겹침
    pos = sorted(rnd.sample(range(4, 96), 10))
    keep = set(rnd.sample(pos, 7))
    new = sorted(rnd.sample([p for p in range(4, 96) if all(abs(p - q) > 2 for q in pos)], 3))
    for k, (lab, items) in enumerate((("스텝 t", [(p, SEL) for p in pos]),
                                      ("스텝 t+1", [(p, SEL) for p in pos if p in keep] + [(p, CMP) for p in new]))):
        y = 270 + k * 46
        f.text(40, y + 21, lab, 15, weight=700)
        f.rect(x0, y, W, 30, FULL_L, rx=3)
        for p, col in items:
            f.rect(x0 + p * W / 100 - 3, y, 6, 30, col, rx=1)
    f.text(x0, 380, "선택 위치만 확대 · 파랑 = 직전 스텝과 같은 위치, 주황 = 새로 고른 위치", 13, color=MUTED)
    f.text(40, 418, "연속 스텝 top-k 약 70% 겹침 → 최근 고른 KV를 두는 LRU 캐시면 hit 87%", 15)
    f.text(40, 470, "→ 안 읽는 98%가 HBM을 차지해 동시 요청 수를 결정 — HBM에 있을 필요가 없음", 16, weight=700)
    big_arrow(f, 800, 872, 300)
    # 오른쪽 — 아이디어
    head(f, 900, 136, "아이디어", "HBM은 hot 캐시, 전체 KV는 큰 용량 계층", SEL)
    f.rect(900, 168, 760, 102, "#ffffff", AXIS, 8, 1.2)
    f.text(916, 194, "GPU HBM", 15, weight=700)
    f.rect(916, 210, 110, 46, SEL, rx=3)
    f.text(1044, 230, "hot 캐시 — 최근 고른 KV만", 15, weight=700)
    f.text(1044, 252, "요청 · 층마다 2k ~ 4k줄 · 길이와 무관", 13, color=MUTED)
    f.rect(900, 330, 760, 102, "#ffffff", AXIS, 8, 1.2)
    f.text(960, 356, "Host DRAM · CXL 메모리 — 전체 KV (용량 계층)", 15, weight=700)
    f.rect(916, 372, 728, 44, FULL_L, rx=3)
    for p in rnd.sample(range(5, 360), 12):
        f.rect(916 + p * 2, 372, 2.4, 44, CMP, rx=0)
    f.arrow(938, 368, 938, 262, CMP, 2.5)
    f.text(960, 296, "miss만 전송 — 선택의 약 13%", 14, weight=700, color=CMP)
    f.text(960, 318, "새 KV는 host로 write-through", 13, color=MUTED)
    f.text(900, 470, "효과: 요청당 HBM이 길이와 무관 → 동시 요청 증가 (HiSparse 장문 처리량 2.1 ~ 4.7×)", 15, weight=700)
    f.text(900, 498, "필요 HW: host · CXL 용량 · CPU–GPU 링크 대역폭 (전송 1회 PCIe 112 µs → NVLink-C2C 29 µs)", 14, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 2. 메모리 근처 선택 · attention
def fig_near():
    S = K128
    idx, pos, kv = DSA_IDX * S, TOPK * 4, MLA_B * TOPK
    f = Fig("02-near-memory", 1700, 570)
    f.text(40, 44, "② 메모리 근처 선택 · attention — 많이 읽고 적게 내보내는 단계를 메모리 옆으로", 26, weight=700)
    f.text(40, 78, "DeepSeek-V3.2(DSA) 층 1개 · 128K · 디코드 1스텝 · KV BF16 1,152 B · 색인 키 132 B · 위치 int32 (덱 2.3 · 3.2 계산값)", 17, color=MUTED)
    head(f, 40, 136, "구조 특징", "색인 훑기는 많이 읽고 적게 내보내고, gather는 작고 흩어짐", SLATE)
    x0, W = 250, 500
    sc = W / idx
    rows = [("① 색인 키 훑기", "매 스텝 전부 · 순차 읽기", f"{idx / MB:.0f} MB 읽기"),
            ("② top-k", "출력 = 위치 2,048개", f"{pos / 1e3:.0f} KB 출력"),
            ("③ KV gather", "1.2 KB 조각 2,048개", f"{kv / MB:.1f} MB · 흩어진 읽기")]
    rnd = random.Random(9)
    for i, (nm, sub, val) in enumerate(rows):
        y = 168 + i * 76
        f.text(40, y + 22, nm, 16, weight=700)
        f.text(40, y + 44, sub, 13, color=MUTED)
        if i == 0:
            f.rect(x0, y + 4, W, 36, SEL_L, rx=3)
            f.text(x0 + 14, y + 28, val, 15, weight=700)
        elif i == 1:
            f.rect(x0, y + 4, 3, 36, SEL, rx=0)
            f.text(x0 + 14, y + 28, val, 15, weight=700)
        else:
            f.rect(x0, y + 4, W, 36, FULL_L, rx=3)
            tw = kv * sc / 20
            for p in rnd.sample(range(58, 140), 20):
                f.rect(x0 + p * W / 142, y + 4, tw, 36, SEL, rx=0)
            f.text(x0 + 14, y + 28, val, 15, weight=700)
    f.text(40, 418, f"→ {idx / MB:.0f} MB를 읽어 {pos / 1e3:.0f} KB만 다음 단계로 — 메모리 옆에서 처리할수록 이동량 감소", 16, weight=700)
    f.text(40, 446, "gather는 작은 조각 · top-k 뒤에야 정해지는 위치라 GPU 메모리 효율이 낮음 (덱 2.3)", 14, color=MUTED)
    f.text(40, 472, "색인 연산 강도: 키 1 B당 DSA 124 FLOP · QSA 4 FLOP (덱 3.2 · 부록 A)", 14, color=MUTED)
    big_arrow(f, 800, 872, 300)
    head(f, 900, 136, "아이디어", "메모리 장치 안에서 고르고 모아, 결과만 GPU로", SEL)
    f.rect(900, 168, 530, 274, BOX[0], BOX[1], 10, 1.2)
    f.text(918, 196, "메모리 장치 (PNM · CXL 연산 메모리)", 15, weight=700)
    f.rect(918, 212, 494, 48, FULL_L, rx=4)
    f.text(934, 242, "KV cache · 색인 키 (요청 전체)", 14, weight=700)
    f.arrow(1165, 264, 1165, 296, SLATE, 3)
    f.text(1180, 286, "내부 대역폭으로 훑기", 13, color=MUTED)
    f.rect(918, 302, 494, 122, "#ffffff", BOX[1], 6, 1.2)
    f.rect(918, 302, 7, 122, SEL, rx=3)
    f.text(940, 334, "근처 연산기", 16, weight=700)
    f.text(940, 362, "색인 훑기 · top-k · KV gather", 15)
    f.text(940, 390, "NELSSA는 sparse attention까지 장치 안에서 실행", 13, color=MUTED)
    f.rect(1500, 250, 160, 100, "#ffffff", AXIS, 8, 1.2)
    f.text(1580, 308, "GPU", 20, "middle", 700)
    f.arrow(1434, 300, 1494, 300, SEL, 3)
    f.text(1580, 376, f"선택 KV {kv / MB:.1f} MB", 13, "middle", 700, SEL)
    f.text(1580, 396, "또는 attention 출력만", 13, "middle", color=MUTED)
    f.text(900, 480, "효과: 긴 요청 decode를 메모리 장치가 분담 — NELSSA decode 처리량 최대 5.5× (SK hynix PNM 실장치)", 15, weight=700)
    f.text(900, 508, "필요 HW: 메모리 근처 저정밀 연산 · top-k · 작은 단위 gather", 14, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 3. state 갱신 PIM
def fig_state():
    f = Fig("03-state-pim", 1700, 560)
    f.text(40, 44, "③ state 갱신 PIM — 매 스텝 전부 읽고 덮어쓰는 state를 메모리 안에서 갱신", 26, weight=700)
    f.text(40, 78, "Qwen3.8-Flash-Next (DeltaNet 36층 + QSA 12층) · 요청 1개 · 디코드 1스텝 · state FP32 · KV BF16 (덱 4.1 계산값)", 17, color=MUTED)
    head(f, 40, 136, "구조 특징", "state는 고정 크기지만 매 스텝 전부 읽고 같은 양을 씀", SLATE)
    legend(f, 40, 182, [(STATE, "state 읽기"), (STATE_L, "state 쓰기"), (SEL, "KV 읽기"), (SEL_L, "색인 키 읽기")])
    x0, sc = 130, 520 / 400e6
    for k, (lab, S) in enumerate((("32K", 32768), ("128K", K128))):
        y = 214 + k * 62
        sr, sw, kv, ix = hybrid_parts(S)
        f.text(40, y + 26, lab, 18, weight=700)
        x = x0
        for v, col in ((sr, STATE), (sw, STATE_L), (kv, SEL), (ix, SEL_L)):
            f.rect(x, y + 4, v * sc - 2, 34, col, rx=2)
            x += v * sc
        tot = sr + sw + kv + ix
        f.text(x + 10, y + 27, f"{tot / MB:.0f} MB · state {(sr + sw) / tot * 100:.0f}%", 15, weight=700)
    f.text(40, 372, f"state {36 * GDN_STATE / MB:.0f} MB(36층 × {GDN_STATE / MB:.1f} MB)를 매 스텝 전부 읽고 덮어씀 — KV · 색인 추가 쓰기는 25 KB", 15)
    f.text(40, 398, "갱신은 원소 단위 곱 · 합 (바이트당 연산 적음) · 요청마다 따로라 동시 요청 수만큼 증가", 14, color=MUTED)
    f.text(40, 470, "→ 연산보다 데이터 이동이 대부분인 연산 — state를 연산기로 옮기는 비용이 핵심", 16, weight=700)
    big_arrow(f, 800, 872, 300)
    head(f, 900, 136, "아이디어", "state를 메모리 밖으로 꺼내지 않고 PIM 안에서 갱신", SEL)
    st = 36 * GDN_STATE / MB
    # 기존
    f.text(900, 192, "기존 — GPU에서 갱신", 15, weight=700, color=MUTED)
    f.rect(900, 206, 210, 74, "#ffffff", AXIS, 8, 1.2)
    f.text(1005, 250, "HBM — state", 15, "middle", 700)
    f.rect(1450, 206, 210, 74, "#ffffff", AXIS, 8, 1.2)
    f.text(1555, 250, "GPU — 갱신 연산", 15, "middle", 700)
    thick(f, 1116, 228, 1444, STATE)
    thick(f, 1444, 262, 1116, STATE_L)
    f.text(1280, 220, f"읽기 {st:.0f} MB", 13, "middle", 700, halo=True)
    f.text(1280, 254, f"쓰기 {st:.0f} MB", 13, "middle", 700, halo=True)
    # PIM
    f.text(900, 338, "PIM — 메모리 안에서 갱신", 15, weight=700, color=SEL)
    f.rect(900, 352, 400, 84, "#ffffff", SEL, 8, 1.6)
    f.text(916, 400, "HBM-PIM", 15, weight=700)
    f.rect(1050, 364, 236, 60, STATE_L, rx=6)
    f.text(1168, 390, "state 갱신 유닛", 15, "middle", 700)
    f.text(1168, 410, "뱅크 근처 원소 곱 · 합", 12, "middle", color=MUTED)
    f.rect(1450, 352, 210, 84, "#ffffff", AXIS, 8, 1.2)
    f.text(1555, 400, "GPU", 15, "middle", 700)
    f.arrow(1306, 384, 1444, 384, SLATE, 1.6)
    f.arrow(1444, 404, 1306, 404, SLATE, 1.6)
    f.text(1375, 372, "입력 · 출력 벡터만", 13, "middle", 700, SLATE)
    f.text(1375, 428, "층당 수십 KB", 13, "middle", color=MUTED)
    f.text(900, 492, "효과: Pimba 생성 처리량 GPU 대비 최대 4.1× · NOVA는 SSM · GQA · MoE를 모두 메모리 근처로", 15, weight=700)
    f.text(900, 520, "필요 HW: PIM 원소 연산 유닛 · 쓰기 대역폭 · state 온칩 상주용 SRAM", 14, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 4. 층별 · 단계별 이종 가속기
def fig_hetero():
    f = Fig("04-hetero", 1700, 590)
    f.text(40, 44, "④ 층별 · 단계별 이종 가속기 — 한 모델 안에서 병목이 바뀌는 만큼 HW도 나눔", 26, weight=700)
    f.text(40, 78, "층 구성 = 공식 config의 층별 설정 (덱 1.3) · 단계별 특성 = 덱 6.5 · 제안 결과는 저자 보고값", 17, color=MUTED)
    head(f, 40, 136, "구조 특징", "층마다, 단계마다 병목이 다름", SLATE)
    q38 = _cfg("Qwen_Qwen3.8-Flash-Next")["layer_types"]
    v4 = _cfg("deepseek-ai_DeepSeek-V4-Pro")
    ratios = v4["compress_ratios"][:v4["num_hidden_layers"]]
    strips = [("Qwen3.8-Flash-Next (48층)", [SEL if t == "full_attention" else STATE for t in q38],
               [(SEL, "QSA 12층 → 색인 · gather"), (STATE, "DeltaNet 36층 → state 읽기 · 쓰기")]),
              ("DeepSeek-V4-Pro (61층)", [CMP if r == 4 else CMP_L for r in ratios],
               [(CMP, "CSA → 색인 · gather"), (CMP_L, "HCA → 압축 KV 연속 읽기")])]
    x0, x1 = 40, 760
    for i, (nm, cells, leg) in enumerate(strips):
        y = 172 + i * 112
        f.text(x0, y + 4, nm, 15, weight=700)
        cw = (x1 - x0) / len(cells)
        for k, col in enumerate(cells):
            f.rect(x0 + k * cw, y + 16, cw - 2, 34, col, rx=2)
        legend(f, x0, y + 78, leg)
    for k, (t1, t2) in enumerate((("prefill", "큰 행렬곱 — 연산 병목"), ("decode", "토큰 단위 — 메모리 · 지연 병목"))):
        x = 40 + k * 370
        f.rect(x, 402, 350, 64, BOX[0], BOX[1], 8, 1)
        f.text(x + 18, 430, t1, 16, weight=700)
        f.text(x + 18, 454, t2, 14, color=MUTED)
    f.text(40, 520, "→ 층 · 단계마다 쓰는 자원이 달라 한 종류 HW로는 활용률이 낮음", 16, weight=700)
    big_arrow(f, 800, 872, 300)
    head(f, 900, 136, "아이디어", "연산 종류 · 단계별로 맞는 HW를 한 시스템에", SEL)
    cards = [("층별 · 한 장치", "HLX", "MICRO 2025", ["attention · SSM 커널을 파이프라인으로", "한 HW에서 데이터 재사용"], "1K ~ 128K에서 GPU 대비 향상"),
             ("층별 · 메모리 근처", "NOVA", "MICRO 2026", ["GQA · SSM · MoE를 모두 near-memory에서", "GPU로 되돌리는 연산 없음"], "Nemotron3 · Falcon-H1R 대상"),
             ("단계별 분리", "DUET", "DAC 2026", ["prefill = systolic + 외부 메모리", "decode = vector + 고대역 메모리"], "B200 대비 TTFT 4× · 처리량 1.4×"),
             ("길이별 분리", "NELSSA", "MICRO 2026", ["짧은 요청은 GPU, 긴 요청은 PNM", "길이가 늘면 재계산 없이 tier 이동"], "decode 처리량 최대 5.5×")]
    for i, (kind, nm, venue, lines, res) in enumerate(cards):
        x, y = 900 + (i % 2) * 390, 166 + (i // 2) * 180
        f.rect(x, y, 370, 166, "#ffffff", BOX[1], 8, 1.2)
        f.rect(x, y, 7, 166, SEL, rx=3)
        f.text(x + 24, y + 30, f"{kind}  ·  {venue}", 13, weight=700, color=MUTED)
        f.text(x + 24, y + 60, nm, 20, weight=700)
        for j, s in enumerate(lines):
            f.text(x + 24, y + 90 + j * 23, s, 14)
        f.text(x + 24, y + 148, res, 13, weight=700, color=SEL)
    f.text(900, 556, "필요: 층별 커널 · 자원 배분, 장치 간 KV · state 전송 (압축 · hybrid 구조는 전송량도 작음)", 14, color=MUTED)
    return f


FIGS = [fig_map, fig_tiered, fig_near, fig_state, fig_hetero]


def main():
    browser = next((c for c in CHROME if os.path.exists(c)), None)
    only = sys.argv[1:]
    for make in FIGS:
        f = make()
        if only and not any(o in f.name for o in only):
            continue
        with open(os.path.join(HERE, f.name + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(f.svg())
        if browser:
            html = os.path.join(HERE, "_tmp.html")
            with open(html, "w", encoding="utf-8") as fh:
                fh.write(f'<html><body style="margin:0;background:#fff">{f.svg()}</body></html>')
            subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            "--force-device-scale-factor=2", f"--window-size={f.w},{f.h}",
                            f"--screenshot={os.path.join(HERE, f.name + '.png')}", "file:///" + html.replace("\\", "/")],
                           check=False, capture_output=True, timeout=120)
            os.remove(html)
        print("ok", f.name)


if __name__ == "__main__":
    main()
