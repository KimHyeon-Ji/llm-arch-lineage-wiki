"""Sparse · Hybrid 구조 특징과 HW 아키텍처 방향 보고서용 그림.
- 00-summary : 1차 초안(sparse-hybrid-hw-report-claude.md)과 2차본(sparse-hybrid-hw-report-v2-claude.md) 공통 요약
- 01 ~ 07    : 2차본 Ⅰ장(구조 특징) 그림 — 한 그림에 메시지 하나 (동향 · 원리 · 줄어드는 것 · 색인 읽기 · 고르는 과정 · state · 데이터 성격)
실행: python arch_insight/figures-claude/hw-report/make_hw_report_figures.py [그림 이름 ...]  → 같은 폴더에 *.svg, *.png
나머지 그림은 덱(figures-claude/)과 HW 아이디어(figures-claude/hw-ideas/) 그림을 그대로 쓴다.
수치는 모두 덱 계산 함수(make_figures.py)를 그대로 불러 계산한다.
"""
import csv
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(HERE))
from make_figures import (Fig, INK, MUTED, AXIS, GRID, SLATE, OTHER, BOX, CHROME, LOAD,  # noqa: E402
                          FULL, FULL_L, SEL, SEL_L, NAVY, CMP, CMP_L, STATE, STATE_L, K128, M, chip, qwen_config,
                          read_parts, store_parts, total, dsa_model_io, hybrid_parts, stage_reads, fmt_b,
                          MLA_B, DSA_IDX, TOPK, GQA_B, QSA_IDX, CSA_IDX, GDN_STATE, HBM_BW, FP8_PK, BF16_PK,
                          DSA_FLOP, MLA_ATT_FLOP, LINEAGE, HALVES, CAT, _half, _cat)


def text_w(s, size):
    return sum(size if ord(c) > 0x2000 else size * 0.55 for c in s)


STATUS = {"real": ("실서비스 진입", SEL, "#ffffff", None),
          "conf": ("학회 제안", SEL_L, INK, None),
          "gap": ("공백", "#ffffff", CMP, CMP)}


def status_chip(f, x, y, key):
    lab, fill, tc, dash = STATUS[key]
    w = text_w(lab, 13) + 26
    f.rect(x, y, w, 26, fill, dash or "none", 13, 1.5, "4 3" if dash else None)
    f.text(x + w / 2, y + 18, lab, 13, "middle", 700, tc)
    return w


def idea_box(f, x, y, w, h, bar, title, lines, key):
    """대응 아이디어 칸 — 왼쪽 띠 + 제목 + 단계 표시 + 설명 줄."""
    f.rect(x, y, w, h, "#ffffff", BOX[1], 8, 1.2)
    f.rect(x, y, 7, h, bar, rx=3)
    f.text(x + 22, y + 34, title, 18, weight=700)
    status_chip(f, x + 22 + text_w(title, 18) + 12, y + 15, key)
    for j, (s, size, col) in enumerate(lines):
        f.text(x + 22, y + 62 + j * 25, s, size, color=col)


# ---------------------------------------------------------------- 그림 0. 구조 특징 → HW 요구 → HW 대응
def fig_summary():
    S = K128
    hy = qwen_config("hybrid", S)
    hy_r = (hy[0][0] + hy[0][1] + hy[0][2] / 2) / sum(qwen_config("full", S)[0])   # 읽기만 (state 쓰기는 state 열)
    f = Fig("00-summary", 1700, 872)
    f.text(40, 44, "Sparse · Hybrid 구조 특징 → HW 요구 → HW 대응 — 요소마다 대응 방향과 성숙도가 다름", 26, weight=700)
    f.text(40, 78, "구조 = 128K · 요청 1개 · 디코드 1스텝 계산값 (비율 = 같은 회사의 전 층 Full 대비) · "
                   "대응 = 대표 제안과 단계 (2026-11-11 검색 기준)", 17, color=MUTED)
    f.text(1268, 128, "부담", 15, weight=700, color=MUTED)
    for k, lab in enumerate(("없음", "작음", "중간", "큼")):
        x = 1316 + k * 88
        f.rect(x, 113, 24, 18, LOAD[k], "#c9ccd6" if k == 0 else "none", 3, 1)
        f.text(x + 30, 128, lab, 15, color=MUTED)
    X0, W, G = 296, 264, 8
    X = [X0 + i * (W + G) for i in range(5)]
    cols = [("KV cache 크기", "요청당 보유량"), ("KV cache 읽기", "디코드 1스텝 (색인 키 · state 포함)"),
            ("색인 · top-k", "읽을 KV를 고르는 비용"), ("흩어진 읽기", "선택된 KV를 모으는 읽기"),
            ("state 갱신", "매 스텝 읽고 덮어쓰기")]
    for x, (nm, sub) in zip(X, cols):
        f.text(x + 4, 180, nm, 19, weight=700)
        f.text(x + 4, 206, sub, 14, color=MUTED)
    f.line(40, 222, 1660, 222, AXIS, 1)
    rows = [
        ("Full attention", "전 층 MLA · GQA (기준)", FULL, FULL_L,
         [(3, "길이에 비례", ("기준 100%",)), (3, "매 스텝 전부", ("기준 100%",)), (0, "없음", ()),
          (0, "없음", ("한 구간 연속 읽기",)), (0, "없음", ("KV 1줄 추가",))]),
        ("Sparse attention", "선택 읽기 — DSA · QSA", SEL, SEL_L,
         [(3, "그대로", ("103 ~ 111% (색인 키 추가)",)),
          (1, "▼ 1/8 ~ 1/21 감소", ("남은 읽기의 대부분이 색인 키",)),
          (3, "▲ KV보다 큰 새 비용", ("연산 = attention의 3.8배",)),
          (3, "▲ 작고 흩어진 조각", ("1.2 KB × 2,048줄",)),
          (0, "없음", ("KV 1줄 + 색인 키 1개",))]),
        ("Hybrid attention", "state 36 + full 12층 (가정)", STATE, STATE_L,
         [(1, "▼ 약 1/4 감소", ("KV 가진 층이 1/4 (26%)",)),
          (2, "▼ 약 1/4 감소", (f"{hy_r * 100:.0f}% · full 층은 길이에 비례",)),
          (0, "없음", ()), (0, "없음", ("state · KV 모두 연속 읽기",)),
          (3, "▲ 매 스텝 전부 덮어씀", ("115 MB 읽고 같은 양 쓰기",))]),
    ]
    y, RH = 234, 84
    for nm, sub, dark, light, cells in rows:
        chip(f, 40, y + 8, 236, 38, nm, dark, light, 18)
        f.text(40, y + 70, sub, 14, color=MUTED)
        for x, (lv, t1, subs) in zip(X, cells):
            f.rect(x, y, W, RH, LOAD[lv], rx=8)
            dk = lv == 3
            f.text(x + 16, y + 36, t1, 19, weight=700, color="#ffffff" if dk else INK)
            for j, t2 in enumerate(subs):
                f.text(x + 16, y + 62 + j * 22, t2, 14, color="#e6e9f2" if dk else (INK if lv == 2 else MUTED))
        y += RH + 8
    # HW 요구 — 요소마다 필요한 HW 특성
    y += 22
    req = [("메모리 용량", "동시 요청 수를 결정"), ("읽기 대역폭", "주 소비가 KV → 색인 키 · state"),
           ("저정밀 연산 · 온칩 정렬", "FP8 · FP4 연산 밀도 · SRAM"), ("불규칙 접근 · 지연", "메모리 동시 요청 수 · gather"),
           ("쓰기 대역폭 · 온칩", "state 온칩 상주 · 근처 갱신")]
    f.text(40, y + 34, "→ HW 요구", 19, weight=700)
    f.text(40, y + 60, "요소마다 필요한 HW 특성", 14, color=MUTED)
    for x, (t1, t2) in zip(X, req):
        f.arrow(x + W / 2, y - 18, x + W / 2, y - 4, MUTED, 2)
        f.rect(x, y, W, 78, BOX[0], BOX[1], 8, 1)
        f.text(x + 16, y + 34, t1, 17, weight=700)
        f.text(x + 16, y + 60, t2, 14, color=MUTED)
    # HW 대응 — 아이디어 · 대표 제안 · 단계
    y += 78 + 26
    f.text(40, y + 34, "→ HW 대응", 19, weight=700)
    f.text(40, y + 60, "아이디어 · 대표 제안 · 단계", 14, color=MUTED)
    for x in X:
        f.arrow(x + W / 2, y - 22, x + W / 2, y - 4, MUTED, 2)
    BH = 128
    idea_box(f, X[0], y, 2 * W + G, BH, SEL, "① 계층형 KV 배치",
             [("전체 KV는 host DRAM · CXL, HBM은 hot KV · 색인 키만", 15, INK),
              ("HiSparse (SGLang 반영) · ECHO (OSDI 2026) · ESS · SAC", 13, MUTED)], "real")
    idea_box(f, X[2], y, 2 * W + G, BH, SEL, "② 메모리 근처 선택 · gather",
             [("색인 훑기 · top-k · gather를 메모리 옆에서, GPU로는 결과만", 15, INK),
              ("NELSSA (SK hynix PNM) · LongSight · PAM", 13, MUTED),
              ("공백 — 색인 연산 · top-k 전용 유닛 (sparse 모델 대상 미확인)", 13, CMP)], "conf")
    idea_box(f, X[4], y, W, BH, STATE, "③ state PIM",
             [("state를 PIM 안에서 갱신", 15, INK), ("Pimba · NOVA · HyPIM", 13, MUTED)], "conf")
    y += BH + 10
    f.rect(X[0], y, X[4] + W - X[0], 70, "#ffffff", BOX[1], 8, 1.2)
    f.rect(X[0], y, 7, 70, SLATE, rx=3)
    t = "④ 층별 · 단계별 이종 가속기"
    f.text(X[0] + 22, y + 30, t, 18, weight=700)
    status_chip(f, X[0] + 22 + text_w(t, 18) + 12, y + 11, "conf")
    f.text(X[0] + 22, y + 56, "공통 — 층마다 위 병목이 바뀌고 prefill = 연산, decode = 메모리 · 지연 → 연산 종류 · 단계마다 다른 HW를 한 시스템에",
           14, color=INK)
    f.text(X[4] + W - 16, y + 30, "HLX · DUET · NOVA · NELSSA", 13, "end", color=MUTED)
    return f


# ---------------------------------------------------------------- 2차본 Ⅰ장 공통 도구
def head(f, x1, y1, x2, y2, color):
    a = math.atan2(y2 - y1, x2 - x1)
    p1 = (x2 - 12 * math.cos(a - 0.45), y2 - 12 * math.sin(a - 0.45))
    p2 = (x2 - 12 * math.cos(a + 0.45), y2 - 12 * math.sin(a + 0.45))
    f.raw(f'<polygon points="{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>')


def legend(f, items, x, y, gap=36):
    """범례 한 줄 — (이름, 채움, 테두리, 점선 여부)."""
    for nm, fill, stroke, dash in items:
        f.rect(x, y - 14, 18, 18, fill, stroke or "none", 3, 1.6, "4 3" if dash else None)
        f.text(x + 26, y + 1, nm, 15, color=MUTED)
        x += 26 + text_w(nm, 15) + gap


MIX_L = "#c9d1e3"     # 층 혼합 구조 (방식 구분 없이)


# ---------------------------------------------------------------- 그림 1. 전 층 full attention 비중
def fig_trend_full():
    rows = list(csv.DictReader(open(LINEAGE, encoding="utf-8-sig")))
    n = {h: 0 for h in HALVES}
    full = {h: 0 for h in HALVES}
    mix = {h: {} for h in HALVES}
    for r in rows:
        h, c = _half(r["date"]), _cat(r)
        n[h] += 1
        if c == "full":
            full[h] += 1
        else:
            mix[h][c] = mix[h].get(c, 0) + 1
    pc = {h: full[h] / n[h] * 100 for h in HALVES}
    lab = lambda h: f"'{h[2:4]} {'상반기' if h.endswith('H1') else '하반기'}"
    f = Fig("01-trend-full", 1600, 640)
    f.text(40, 44, f"전 층 full attention 모델 비중 — {lab(HALVES[2])} {pc[HALVES[2]]:.0f}% → {lab(HALVES[-1])} {pc[HALVES[-1]]:.0f}%",
           26, weight=700)
    f.text(40, 78, f"조사 대상: 주요 9개 시리즈 {len(rows)}개 공개 모델 · 반기별 공개 모델 중 비중 · 공식 config 기준", 17, color=MUTED)
    legend(f, [("전 층 full attention", FULL, None, False), ("층마다 방식을 섞는 구조 (Sparse · Hybrid · 압축 · window)", MIX_L, None, False)],
           40, 130)
    x0, x1, top, bot = 110, 1040, 180, 530
    Y = lambda p: bot - p / 100 * (bot - top)
    for p in (0, 50, 100):
        f.line(x0, Y(p), x1, Y(p), GRID, 1)
        f.text(x0 - 12, Y(p) + 5, f"{p}%", 15, "end", color=MUTED)
    slot = (x1 - x0) / len(HALVES)
    bw = 120
    pts = []
    for i, h in enumerate(HALVES):
        bx = x0 + slot * i + (slot - bw) / 2
        f.rect(bx, Y(100), bw, Y(0) - Y(100), MIX_L, rx=4)
        f.rect(bx, Y(pc[h]), bw, Y(0) - Y(pc[h]), FULL, rx=4)
        pts.append((bx + bw / 2, Y(pc[h])))
        last = i == len(HALVES) - 1
        f.text(bx + bw / 2, Y(pc[h]) - 16, f"{pc[h]:.0f}%", 24 if last else 20, "middle", 700, halo=True)
        f.text(bx + bw / 2, bot + 30, lab(h), 18, "middle", 700, INK if i >= 2 else MUTED)
        f.text(bx + bw / 2, bot + 54, f"{full[h]}/{n[h]}개" + (" · 표본 적음" if n[h] < 10 else ""), 14, "middle", color=MUTED)
    f.line(x0, bot, x1, bot, AXIS, 1.5)
    # 오른쪽 — 최근 반기의 혼합 구조 내역 (보조 정보)
    h = HALVES[-1]
    rx = 1110
    f.rect(rx - 20, 170, 490, 370, BOX[0], BOX[1], 10, 1.2)
    f.text(rx, 210, f"참고 — {lab(h)} 혼합 구조 {sum(mix[h].values())}개 내역", 18, weight=700)
    f.text(rx, 236, "모델별로 가장 먼저 해당하는 분류 하나로 집계", 13, color=MUTED)
    order = [(c, nm, col) for c, nm, col, _ in CAT if c != "full" and mix[h].get(c)]
    order.sort(key=lambda t: -mix[h][t[0]])
    for k, (c, nm, col) in enumerate(order):
        y = 270 + k * 62
        f.text(rx, y + 18, nm, 16, weight=700)
        f.rect(rx, y + 28, mix[h][c] * 46, 20, col, rx=3)
        f.text(rx + mix[h][c] * 46 + 10, y + 44, f"{mix[h][c]}개", 15, weight=700)
    return f


# ---------------------------------------------------------------- 그림 2. 작동 원리 — Full · Sparse · Hybrid
def fig_how():
    f = Fig("02-how-it-works", 1600, 700)
    f.text(40, 44, "작동 원리 — Sparse는 '전부 보관 · 일부만 읽기', Hybrid는 '대부분 층을 고정 크기 state로'", 26, weight=700)
    f.text(40, 78, "층 1개 · 디코드 1스텝 · 한 줄 = 토큰 하나의 KV · 그림 비율은 실제와 다름 (수치는 128K 기준)", 17, color=MUTED)
    PX, PW = [190, 660, 1130], 440
    heads = [("Full attention", "전 층 MLA · GQA (기준)", FULL, FULL_L), ("Sparse attention", "선택 읽기 — DSA · QSA", SEL, SEL_L),
             ("Hybrid attention", "state 층 + 일부 full 층 — Qwen3.8 · Kimi", STATE, STATE_L)]
    for x, (nm, sub, dark, light) in zip(PX, heads):
        chip(f, x, 108, PW, 44, nm, dark, light, 21)
        f.text(x + 4, 178, sub, 15, color=MUTED)
    top, rh, nrow = 200, 18, 10
    # Full — 전부 읽기
    x = PX[0] + 30
    for k in range(nrow):
        f.rect(x, top + k * rh + 1, 150, rh - 3, FULL, rx=2)
    f.rect(x, top + nrow * rh + 6, 150, rh - 3, "#ffffff", INK, 2, 1.8, "5 3")
    f.text(x + 172, top + 70, "KV 전부", 17, weight=700)
    f.text(x + 172, top + 94, "매 스텝 전부 읽기", 15, color=MUTED)
    f.text(x + 172, top + nrow * rh + 18, "새 KV 1줄 추가", 14, color=MUTED)
    # Sparse — 전부 보관 · 일부만 읽기
    x = PX[1] + 30
    pick = {1, 4, 6, 9}
    for k in range(nrow):
        f.rect(x, top + k * rh + 1, 130, rh - 3, SEL if k in pick else FULL_L, rx=2)
        f.rect(x + 138, top + k * rh + 1, 18, rh - 3, SEL_L, SEL, 2, 1)
    f.rect(x, top + nrow * rh + 6, 130, rh - 3, "#ffffff", INK, 2, 1.8, "5 3")
    f.rect(x + 138, top + nrow * rh + 6, 18, rh - 3, "#ffffff", INK, 2, 1.8, "5 3")
    f.text(x + 176, top + 14, "색인 키", 17, weight=700)
    f.text(x + 176, top + 36, "전부 읽어 점수 계산", 15, color=MUTED)
    f.text(x + 176, top + 92, "고른 KV만 읽기", 17, weight=700, color=SEL)
    f.text(x + 176, top + 114, "128K줄 중 2,048줄", 15, color=MUTED)
    f.text(x + 176, top + 160, "나머지 KV", 17, weight=700, color=MUTED)
    f.text(x + 176, top + 182, "보관만 (다음에 고를 수 있음)", 15, color=MUTED)
    # Hybrid — state 층 + full 층
    x = PX[2] + 20
    cs, cg = 18, 3
    for r in range(5):
        for c in range(5):
            f.rect(x + c * (cs + cg), top + 4 + r * (cs + cg), cs, cs, STATE, rx=2)
    g = 5 * (cs + cg)
    f.rect(x - 4, top, g + 5, g + 5, "none", INK, 4, 1.8, "5 3")
    f.text(x + g / 2, top + g + 40, "state 층 (36/48)", 15, "middle", 700)
    f.text(x + g / 2, top + g + 62, "고정 크기 · 층당 3.2 MB", 13, "middle", color=MUTED)
    f.text(x + g / 2, top + g + 82, "전부 읽고 다시 씀", 13, "middle", 700, STATE)
    kx = x + g + 70
    for k in range(5):
        f.rect(kx, top + 4 + k * rh, 110, rh - 3, FULL, rx=2)
    f.rect(kx, top + 4 + 5 * rh + 2, 110, rh - 3, "#ffffff", INK, 2, 1.8, "5 3")
    f.text(kx + 55, top + g + 40, "full 층 (12/48)", 15, "middle", 700)
    f.text(kx + 55, top + g + 62, "KV 보관 · 읽기", 13, "middle", color=MUTED)
    f.text(kx + 55, top + g + 82, "KV 가진 층이 1/4", 13, "middle", 700)
    # 범례
    legend(f, [("보관만 (이번 스텝에 안 읽음)", FULL_L, None, False), ("이번 스텝에 읽음", SEL, None, False),
               ("이번 스텝에 씀", "#ffffff", INK, True)], 190, 440)
    # 아래 — 보관 · 읽기 · 쓰기 (핵심 차이만 강조)
    rows = [("보관량", [("길이에 비례", "128K줄", None), ("Full과 같음 + 색인 키", "+3 ~ 11%", None),
                      ("full 층 KV만 길이에 비례", "KV 가진 층 1/4 + state 고정", STATE_L)]),
            ("매 스텝 읽기", [("KV 전부", "128K줄", None), ("색인 키 전부 + 고른 KV", "고른 KV 2,048줄", SEL_L),
                          ("full 층 KV + state 전부", "", None)]),
            ("매 스텝 쓰기", [("KV 1줄 추가", "", None), ("KV 1줄 + 색인 키 1개", "", None),
                          ("state 전부 덮어씀", "층당 3.2 MB", STATE_L)])]
    y = 470
    for lab_, cells in rows:
        f.text(40, y + 44, lab_, 18, weight=700, color=MUTED)
        for x, (a, b, hl) in zip(PX, cells):
            f.rect(x, y + 4, PW, 64, hl or BOX[0], rx=6)
            f.text(x + 18, y + 32, a, 18, weight=700)
            if b:
                f.text(x + 18, y + 56, b, 14, color=INK if hl else MUTED)
        y += 74
    return f


# ---------------------------------------------------------------- 그림 3. 줄어드는 것 — 이동량 · 보유량 (같은 모델 조건)
def fig_shrink():
    S = K128
    f = Fig("03-what-shrinks", 1600, 640)
    f.text(40, 44, "줄어드는 것 — Sparse는 매 스텝 이동량만, Hybrid는 이동량과 보유량을 함께 줄임", 26, weight=700)
    f.text(40, 78, "Qwen3.8-Flash-Next 차원(48층)에서 층 구성만 바꾼 비교 · 요청 1개 · 128K · KV BF16 · state FP32 (계산값) · 이동량 = 읽기 + 쓰기",
           17, color=MUTED)
    legend(f, [("Full KV cache", FULL, None, False), ("Sparse KV cache", SEL, None, False), ("색인 키", SEL_L, SEL, False),
               ("state 읽기 · 보관", STATE, None, False), ("state 쓰기", STATE_L, None, False)], 40, 128, 30)
    X1, X2, pw, vmax = 420, 1030, 400, 14e9
    sc = pw / vmax
    f.text(X1, 186, "디코드 1스텝 이동량 (읽기 + 쓰기)", 20, weight=700)
    f.text(X2, 186, "요청당 보유량 (KV cache + 색인 키 + state)", 20, weight=700)
    for X in (X1, X2):
        for v in (0, 5e9, 10e9):
            f.line(X + v * sc, 204, X + v * sc, 590, GRID, 1)
            f.text(X + v * sc, 612, f"{v / 1e9:.0f} GB" if v else "0", 14, "middle", color=MUTED)
    cfgs = [("full", "전 층 Full", "GQA 48층 (기준)"), ("sparse", "Sparse만", "QSA 48층 (가정)"),
            ("hybrid", "Hybrid만", "GQA 12층 + state 36층 (가정)"), ("both", "Sparse + Hybrid", "QSA 12층 + state 36층 (Qwen3.8 실제)")]
    ref_io, ref_sz = (sum(x) for x in qwen_config("full", S))
    for i, (kind, nm, sub) in enumerate(cfgs):
        y = 214 + i * 94
        if kind == "both":
            f.rect(30, y - 6, 1540, 90, BOX[0], rx=8)
        f.text(40, y + 34, nm, 20, weight=700)
        f.text(40, y + 60, sub, 14, color=MUTED)
        (kv_rw, idx_r, st_rw), (kv_s, idx_s, st_s) = qwen_config(kind, S)
        kvcol = FULL if kind in ("full", "hybrid") else SEL
        io = [(kv_rw, kvcol, None), (idx_r, SEL_L, SEL), (st_rw / 2, STATE, None), (st_rw / 2, STATE_L, None)]
        sz = [(kv_s, kvcol, None), (idx_s, SEL_L, SEL), (st_s, STATE, None)]
        for X, parts, ref in ((X1, io, ref_io), (X2, sz, ref_sz)):
            x = X
            for v, col, st in parts:
                if v <= 0:
                    continue
                w = max(v * sc, 2.5)
                f.rect(x, y + 14, w, 38, col, st or "none", 3, 1.2)
                x += w + 1
            tot = sum(v for v, _, _ in parts)
            t1 = fmt_b(tot)
            f.text(x + 12, y + 40, t1, 18, weight=700)
            f.text(x + 12 + text_w(t1, 18) + 10, y + 40, "기준" if kind == "full" else f"({tot / ref * 100:.0f}%)",
                   16, weight=700 if kind != "full" else 400, color=INK if kind != "full" else MUTED)
    return f


# ---------------------------------------------------------------- 그림 4. Sparse에 남는 것 ① — 색인 키 읽기
def fig_index_read():
    f = Fig("04-sparse-index-read", 1600, 660)
    f.text(40, 44, "Sparse에 남는 것 ① — 고른 KV 읽기는 고정, 색인 키 읽기는 길이에 비례해 커짐", 26, weight=700)
    f.text(40, 78, "DeepSeek-V3.2 (DSA 61층) · 요청 1개 · 디코드 1스텝 읽기 · KV BF16 비교 기준 · 색인 키 FP8 + scale 132 B (계산값)",
           17, color=MUTED)
    segs = [("고른 KV 2,048줄 (길이 무관)", SEL, "none"), ("top-k 점수 · KV 추가", OTHER, "none"),
            ("색인 키 (토큰마다 132 B)", SEL_L, SEL)]
    legend(f, [(nm, col, None if st == "none" else st, False) for nm, col, st in segs], 40, 128)
    x0, top, bot = 120, 200, 570
    vmax, cap = 1.6e9, 1.3e9
    Y = lambda v: bot - v / vmax * (bot - top)
    for v in (0, 0.4e9, 0.8e9, 1.2e9, 1.6e9):
        f.line(x0, Y(v), 1540, Y(v), GRID, 1)
        f.text(x0 - 12, Y(v) + 6, f"{v / 1e9:.1f} GB" if v else "0", 15, "end", color=MUTED)
    lens = [(8192, "8K"), (32768, "32K"), (65536, "64K"), (K128, "128K"), (163840, "160K"), (M, "1M*")]
    bw, slot = 130, 236
    for j, (S, lab) in enumerate(lens):
        x = x0 + 50 + j * slot
        parts = dsa_model_io(S)
        tot = sum(parts)
        broken = tot > vmax
        acc = 0
        for (nm, col, st), v in zip(segs, parts):
            hi = acc + v
            y_hi = Y(min(hi, cap)) if broken else Y(hi)
            f.rect(x, y_hi + 1, bw, Y(acc) - y_hi - 2, col, AXIS if broken else st, 3, 1.5, "5 4" if broken else None)
            if col == SEL_L and Y(acc) - y_hi > 40:
                f.text(x + bw / 2, (Y(acc) + y_hi) / 2 + 7, f"색인 {v / tot * 100:.0f}%", 19, "middle", 700)
            acc = hi
        top_y = Y(tot) if not broken else Y(vmax) + 30
        if broken:
            yb = Y(cap) + 2
            f.raw(f'<polygon points="{x - 4:.1f},{yb + 6:.1f} {x + bw + 4:.1f},{yb - 6:.1f} {x + bw + 4:.1f},{yb + 6:.1f} '
                  f'{x - 4:.1f},{yb + 18:.1f}" fill="#ffffff"/>')
            f.rect(x, top_y, bw, Y(cap) - top_y, SEL_L, AXIS, 3, 1.5, "5 4")
        f.text(x + bw / 2, top_y - 34, fmt_b(tot), 19, "middle", 700)
        f.text(x + bw / 2, top_y - 11, f"Full 대비 1/{61 * MLA_B * S / tot:.0f}", 15, "middle", color=MUTED)
        f.text(x + bw / 2, bot + 30, lab, 18, "middle", 700, INK if S == K128 else MUTED)
        f.text(x + bw / 2, Y(parts[0] / 2) + 5, "고정", 14, "middle", 700, "#ffffff")
    f.text(x0 + 50 + 4 * slot + bw / 2, bot + 52, "V3.2 지원 길이", 13, "middle", color=MUTED)
    f.rect(x0 + 30, top - 6, 520, 74, "#ffffff", BOX[1], 8, 1.2)
    f.text(x0 + 48, top + 24, "토큰 1개가 늘 때 늘어나는 읽기 (61층)", 16, weight=700)
    f.text(x0 + 48, top + 50, f"색인 키 132 B × 61 = {132 * 61 / 1e3:.1f} KB  ·  고른 KV 0 B", 16, color=INK)
    f.text(x0, 640, "* 1M은 V3.2 지원 길이(160K) 밖 — 외삽 · Full = 같은 길이의 MLA 61층 읽기 (KV BF16)", 14, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 5. Sparse에 남는 것 ② — 고르는 과정
def fig_steps():
    S = K128
    f = Fig("05-sparse-steps", 1600, 660)
    f.text(40, 44, "Sparse에 남는 것 ② — 고르는 과정: 점수 계산 → top-k → KV 모으기 → attention", 26, weight=700)
    f.text(40, 78, "DeepSeek-V3.2 층 1개 · 128K · 디코드 1스텝 · KV BF16 비교 기준 · 시간 = H200 사양으로 나눈 이론 하한 (실제 시간 아님)",
           17, color=MUTED)
    idx_b, sc_b = DSA_IDX * S, 2 * 4 * S
    ga_b = MLA_B * TOPK
    idx_fl, att_fl = DSA_FLOP * S, MLA_ATT_FLOP * TOPK
    t = [idx_b / HBM_BW * 1e6, sc_b / HBM_BW * 1e6, ga_b / HBM_BW * 1e6, att_fl / BF16_PK * 1e6]
    t_ic = idx_fl / FP8_PK * 1e6
    boxes = [("①", "점수 계산", "색인 키 128K개를 전부 읽고 점수 계산", f"읽기 {fmt_b(idx_b)} · {idx_fl / 1e9:.2f} GFLOP",
              "대역폭 + 저정밀 연산", f"연산 = 본 attention의 {idx_fl / att_fl:.1f}배", SEL_L, SEL),
             ("②", "top-k 선택", "점수 128K개 중 높은 2,048개 고르기", f"점수 기록 · 읽기 {fmt_b(sc_b)}",
              "비교 · 정렬 · 동기화", "행렬곱 아님 — FLOP · 바이트에 안 잡힘", OTHER, None),
             ("③", "KV 모으기 (gather)", "고른 위치의 KV 조각을 모아 읽기", f"{MLA_B / 1e3:.1f} KB × 2,048개 = {fmt_b(ga_b)}",
              "작고 흩어진 읽기", "top-k가 끝나야 위치를 앎", SEL, None),
             ("④", "attention", "고른 2,048줄로 attention 계산", f"{att_fl / 1e9:.2f} GFLOP",
              "작음", "Full이면 128K줄 전부", FULL, None)]
    BX, BW, BY, BH = [40, 430, 820, 1210], 350, 112, 250
    for i, (num, nm, what, data, burden, note, col, st) in enumerate(boxes):
        x = BX[i]
        f.rect(x, BY, BW, BH, "#ffffff", BOX[1], 10, 1.4)
        f.rect(x, BY, BW, 8, col, st or "none", 3, 1)
        f.text(x + 20, BY + 46, f"{num} {nm}", 21, weight=700)
        f.text(x + 20, BY + 76, what, 15, color=MUTED)
        f.text(x + 20, BY + 114, data, 18, weight=700)
        cw = text_w(burden, 15) + 28
        f.rect(x + 20, BY + 134, cw, 32, SEL_L if i < 3 else FULL_L, rx=6)
        f.text(x + 34, BY + 156, burden, 15, weight=700)
        f.text(x + 20, BY + 200, note, 14, color=INK if i == 0 else MUTED, weight=700 if i == 0 else 400)
        if i < 3:
            f.arrow(x + BW + 6, BY + BH / 2, BX[i + 1] - 6, BY + BH / 2, SLATE, 3)
    # 아래 — 이론 하한 시간 (한 줄)
    ty = 430
    f.text(40, ty + 4, "층당 이론 하한 시간", 18, weight=700)
    f.text(40, ty + 28, "H200 · 실제 시간 아님", 14, color=MUTED)
    tx0, tmax, tw = 300, 6.0, 1200
    T = lambda v: tx0 + v / tmax * tw
    for k in range(7):
        f.line(T(k), ty - 24, T(k), ty + 84, GRID, 1)
        f.text(T(k), ty + 104, f"{k} µs", 13, "middle", color=MUTED)
    acc = 0
    cols = [(SEL_L, SEL), (OTHER, "none"), (SEL, "none"), (FULL, "none")]
    for k, (v, (c, st)) in enumerate(zip(t, cols)):
        f.rect(T(acc), ty - 14, T(acc + v) - T(acc) - 1.5, 36, c, st, 3, 1.4)
        lab = f"① 읽기 {v:.2f} µs" if k == 0 else f"{v:.2f}"
        f.text((T(acc) + T(acc + v)) / 2 if k else T(acc) + 14, ty + 10, lab, 15 if k == 0 else 13,
               "middle" if k else "start", 700, INK if k < 2 else "#ffffff")
        if k:
            f.text((T(acc) + T(acc + v)) / 2, ty - 24, "②③④"[k - 1], 15, "middle", 700)
        acc += v
    f.rect(T(acc) + 2, ty - 14, 150, 36, "#ffffff", AXIS, 4, 1.4, "5 4")
    f.text(T(acc) + 77, ty + 9, "+ 정렬 · 흩어진 읽기 지연", 13, "middle", 700, MUTED)
    f.rect(T(0), ty + 30, T(t_ic) - T(0), 14, SLATE, rx=3)
    f.text(T(t_ic) + 10, ty + 43, f"① 연산 {t_ic:.2f} µs — 읽기와 겹쳐 실행 가능", 14, color=MUTED)
    rest = sum(t[1:])
    f.text(40, 590, f"→ 이론 하한으로는 ① 점수 계산이 나머지 합({rest:.1f} µs)의 약 {t[0] / rest:.0f}배 — 대역폭과 저정밀 연산을 함께 씀",
           18, weight=700)
    f.text(40, 620, "②③은 바이트로는 작지만 앞 단계가 끝나야 시작하는 직렬 작업 — 비교 · 정렬 · 흩어진 읽기의 지연은 이론값에 없어 실측 필요",
           15, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 6. Hybrid에 남는 것 — state 읽기 · 쓰기
def fig_state():
    f = Fig("06-hybrid-state", 1600, 490)
    f.text(40, 44, "Hybrid에 남는 것 — state는 크기가 고정이지만 매 스텝 115 MB를 읽고 115 MB를 다시 씀", 26, weight=700)
    f.text(40, 78, "Qwen3.8-Flash-Next (QSA 12층 + DeltaNet 36층) · 요청 1개 · 디코드 1스텝 · KV BF16 · state FP32 (계산값)", 17, color=MUTED)
    st = 36 * GDN_STATE
    # 왼쪽 — 매 스텝 state 왕복
    ax, bx_, by, bh, bwid = 50, 500, 150, 120, 230
    f.rect(ax, by, bwid, bh, STATE_L, rx=10)
    f.text(ax + bwid / 2, by + 48, "HBM — state", 19, "middle", 700)
    f.text(ax + bwid / 2, by + 76, f"{fmt_b(st)} (36층 × 3.2 MB)", 15, "middle")
    f.text(ax + bwid / 2, by + 100, "요청마다 따로", 13, "middle", color=MUTED)
    f.rect(bx_, by, bwid, bh, "#ffffff", BOX[1], 10, 1.4)
    f.text(bx_ + bwid / 2, by + 48, "연산기 — 갱신", 19, "middle", 700)
    f.text(bx_ + bwid / 2, by + 76, "원소 단위 곱 · 합", 15, "middle")
    f.text(bx_ + bwid / 2, by + 100, "연산은 적음", 13, "middle", color=MUTED)
    f.line(ax + bwid + 8, by + 38, bx_ - 20, by + 38, STATE, 6)
    head(f, ax + bwid, by + 38, bx_ - 6, by + 38, STATE)
    f.text((ax + bwid + bx_) / 2, by + 24, f"① 읽기 {fmt_b(st)}", 16, "middle", 700)
    f.line(bx_ - 8, by + 84, ax + bwid + 20, by + 84, STATE_L, 6)
    head(f, bx_, by + 84, ax + bwid + 6, by + 84, "#7fcfb0")
    f.text((ax + bwid + bx_) / 2, by + 112, f"② 다시 쓰기 {fmt_b(st)}", 16, "middle", 700)
    f.text(ax, 340, f"→ 매 스텝 약 {round(2 * st / 1e6, -1):.0f} MB를 옮김 — 길이와 무관한 고정 비용", 21, weight=700)
    kvw = 12 * (GQA_B + QSA_IDX / 4)
    f.text(ax, 372, f"KV · 색인 추가 쓰기는 {fmt_b(kvw)} — state 쓰기가 약 {round(st / kvw, -2):,.0f}배", 15, color=MUTED)
    f.text(ax, 398, "비용 대부분이 연산이 아닌 데이터 이동", 15, color=MUTED)
    # 오른쪽 — 이동량 중 state 몫 (32K · 128K)
    rx = 850
    f.line(810, 110, 810, 480, GRID, 2, "6 6")
    f.text(rx, 134, "디코드 1스텝 이동량 — 모델 전체", 19, weight=700)
    legend(f, [("state 읽기", STATE, None, False), ("state 쓰기", STATE_L, None, False), ("KV 읽기", SEL, None, False),
               ("색인 키 읽기", SEL_L, SEL, False)], rx, 172, 22)
    sc = 520 / 400e6
    for k, (L, lab) in enumerate(((32768, "32K"), (K128, "128K"))):
        y = 210 + k * 110
        p = hybrid_parts(L)
        f.text(rx, y + 32, lab, 19, weight=700)
        x = rx + 70
        for v, col, stc in zip(p, (STATE, STATE_L, SEL, SEL_L), (None, None, None, SEL)):
            w = v * sc
            f.rect(x, y + 8, w - 1.5, 40, col, stc or "none", 3, 1.2)
            if w > 60:
                f.text(x + w / 2, y + 34, f"{v / 1e6:.0f} MB", 14, "middle", 700, "#ffffff" if col in (STATE, SEL) else INK)
            x += w
        tot = sum(p)
        f.text(x + 12, y + 26, fmt_b(tot), 18, weight=700)
        f.text(x + 12, y + 48, f"state {(p[0] + p[1]) / tot * 100:.0f}%", 16, weight=700, color=STATE)
    f.text(rx, 450, "state 몫은 길이와 무관한 고정값이라 짧을수록 큼 (8K 80% · 256K 48%)", 14, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 7. 데이터별 성격 — 보관 · 읽기 · 쓰기
def fig_data():
    S = K128
    f = Fig("07-data-access", 1600, 600)
    f.text(40, 44, "데이터별 성격 — KV cache는 '많이 보관 · 조금 읽기', 색인 키 · state는 '매 스텝 전부 읽기'", 26, weight=700)
    f.text(40, 78, "요청 1개 · 128K · 디코드 1스텝 · 모델 전체 합계 (계산값) · KV · 색인 키 = DeepSeek-V3.2 실제 배포 형식 (KV FP8 656 B), "
                   "state = Qwen3.8 FP32", 17, color=MUTED)
    C1, W1, V1 = 420, 300, 5.3e9
    C2, C3, WB = 900, 1250, 200
    f.text(40, 132, "데이터", 17, weight=700, color=MUTED)
    f.text(C1, 132, "요청당 보유량", 17, weight=700, color=MUTED)
    f.text(C2, 132, "매 스텝 읽기 (보유량 대비)", 17, weight=700, color=MUTED)
    f.text(C3, 132, "매 스텝 쓰기 (보유량 대비)", 17, weight=700, color=MUTED)
    f.line(40, 146, 1560, 146, AXIS, 1)
    kv, idx, st = 61 * 656 * S, 61 * DSA_IDX * S, 36 * GDN_STATE
    rows = [("KV cache", "DeepSeek-V3.2 · 61층", "용량형", SEL, None, kv, 61 * 656 * TOPK, 61 * 656),
            ("색인 키", "DeepSeek-V3.2 · 132 B/토큰", "대역폭형", SEL_L, SEL, idx, idx, 61 * DSA_IDX),
            ("state", "Qwen3.8 · DeltaNet 36층", "대역폭형 · 쓰기", STATE, None, st, st, st)]
    for i, (nm, sub, kind, col, stc, size, rd, wr) in enumerate(rows):
        y = 160 + i * 110
        if i % 2 == 0:
            f.rect(30, y, 1540, 100, BOX[0], rx=8)
        f.text(40, y + 40, nm, 22, weight=700)
        f.text(40, y + 68, sub, 14, color=MUTED)
        cw = text_w(kind, 15) + 26
        f.rect(210, y + 22, cw, 30, SEL_L if kind == "용량형" else ("#e4eefb" if nm == "색인 키" else STATE_L), rx=6)
        f.text(223, y + 43, kind, 15, weight=700)
        w = max(size / V1 * W1, 5)
        f.rect(C1, y + 34, w, 30, col, stc or "none", 4, 1.2)
        f.text(C1 + w + 12, y + 56, fmt_b(size), 19, weight=700)
        for X, v in ((C2, rd), (C3, wr)):
            frac = v / size
            f.rect(X, y + 34, WB, 30, "#e3e6ee", rx=4)
            if frac > 0.001:
                f.rect(X, y + 34, max(WB * frac, 4), 30, col if X == C2 else (STATE_L if nm == "state" else col), stc or "none", 4, 1.2)
            t1 = f"{frac * 100:.0f}%" if frac >= 0.995 or frac >= 0.1 else (f"{frac * 100:.1f}%" if frac >= 0.001 else "≈ 0%")
            f.text(X + WB + 12, y + 50, t1, 19, weight=700)
            f.text(X + WB + 12, y + 72, fmt_b(v), 14, color=MUTED)
    f.text(40, 520, f"→ 보유량은 KV cache가 가장 크지만, 매 스텝 옮기는 양은 색인 키 · state가 더 큼 — "
                    f"V3.2 색인 키 읽기 {fmt_b(idx)} = KV 읽기 {fmt_b(61 * 656 * TOPK)}의 {idx / (61 * 656 * TOPK):.0f}배", 18, weight=700)
    f.text(40, 550, "읽는 비율 1.6% = 2,048줄 ÷ 128K — 길이에 따라 바뀜 (32K 6.3% · 1M 0.2%) · 색인 키 · state는 요청마다 따로라 동시 요청 수만큼 늘어남",
           15, color=MUTED)
    return f


FIGS = [fig_summary, fig_trend_full, fig_how, fig_shrink, fig_index_read, fig_steps, fig_state, fig_data]



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
