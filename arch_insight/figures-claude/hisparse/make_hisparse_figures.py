"""HiSparse 논문 정리(arch_insight/paper-hisparse-claude.md)용 그림 생성기.
실행: python arch_insight/figures-claude/hisparse/make_hisparse_figures.py [그림 이름 ...]  → 같은 폴더에 *.svg, *.png
SVG 도구 · 색 · 글꼴은 상위 폴더 make_figures.py(덱 그림)를 그대로 쓴다.

수치 기준
- 논문 측정값: arXiv 2608.07009v1 (§1 · §2 · §4, Fig. 1 · 4 · 5 · 6 · 8) — 그림마다 basis 줄에 표기
- 요청당 HBM 분해(그림 0 · 4)는 계산값: GLM-5.1 config(78층, MLA 512 + RoPE 64)
  층 · 토큰당 KV BF16 1,152 B + 색인 키 FP8 128 B = 1,280 B → 78층 × 131,072 = 13.09 GB (논문 수치와 일치)
  HiSparse = GPU 캐시 78층 × B(4,096) × 1,152 B + 색인 키 전체 (색인 키는 HBM에 남음)
"""
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True             # 상위 폴더에 __pycache__를 만들지 않음
sys.path.insert(0, os.path.dirname(HERE))
from make_figures import (Fig, chip, INK, MUTED, GRID, AXIS, SLATE, BOX, OTHER,  # noqa: E402
                          FULL, FULL_L, SEL, SEL_L, NAVY, CMP, CMP_L, CHROME)

LAYERS = 78
KV_ROW = 576 * 2           # MLA latent 512 + RoPE 64, BF16
IDX_ROW = 128              # 색인 키 128차원 FP8 (13.09 GB를 재현하는 값, scale 포함 시 132 B)
TOPK = 2048
B_SLOT = 4096
K40, K128, M1 = 40960, 131072, 1048576
GB = 1e9
SYNC = "#8db8ee"           # 동기 Resolve (HiSparse의 옅은 계열)
HOST_BG = "#ffffff"


def req_hbm(S, hisparse):
    """요청 1개의 HBM 사용량 (KV 또는 GPU 캐시, 색인 키) 바이트."""
    rows = min(S, B_SLOT) if hisparse else S
    return LAYERS * rows * KV_ROW, LAYERS * S * IDX_ROW


def text_w(s, size):
    """대략적인 글자 폭 — 한글은 글자 크기, 나머지는 0.55배."""
    return sum(size if ord(c) > 0x2000 else size * 0.55 for c in s)


def legend(f, x, y, items, size=15, gap=32):
    for col, lab, *rest in items:
        dash = rest[0] if rest else None
        if dash:
            f.rect(x, y - 13, 18, 18, "#ffffff", col, 3, 1.6, "4 3")
        else:
            f.rect(x, y - 13, 18, 18, col, rx=3)
        f.text(x + 26, y + 2, lab, size, color=INK)
        x += 26 + text_w(lab, size) + gap
    return x


def comma(v):
    return f"{v:,}"


# ---------------------------------------------------------------- 그림 0. 요약 — 기준 vs HiSparse 배치 + 핵심 수치
def fig_summary():
    f = Fig("00-summary", 1700, 800)
    f.text(40, 44, "HiSparse — 전체 KV는 host DRAM에, HBM에는 요청 · 층별 GPU 캐시와 색인 키만", 26, weight=700)
    f.text(40, 78, "GLM-5.1 · 128K 요청 15개를 담을 때 · KV BF16 · 막대 폭 = 사용량 비율 (계산값) · 아래 수치는 논문 측정값 (1.68 GB만 계산값)", 17, color=MUTED)
    legend(f, 190, 116, [(FULL, "KV cache"), (SEL, "GPU 캐시 (HiSparse)"), (SEL_L, "색인 키"), (CMP, "miss 전송")])
    n = 15
    kv_b, idx = req_hbm(K128, False)
    kv_h, _ = req_hbm(K128, True)
    full = n * (kv_b + idx)
    W = 710
    inner = W - 16
    hb_y, hb_h = 190, 120         # HBM 상자
    ho_y, ho_h = 424, 84          # host 상자
    f.text(40, hb_y + 56, "GPU HBM", 19, weight=700)
    f.text(40, hb_y + 82, "KV 영역", 15, color=MUTED)
    f.text(40, ho_y + 40, "Host DRAM", 19, weight=700)
    f.text(40, ho_y + 66, "약 2 TB", 15, color=MUTED)
    for px, nm, sub, dark, light in ((190, "기준 (SGLang)", "전체 KV를 HBM에 상주", FULL, FULL_L),
                                     (950, "HiSparse", "전체 KV는 host, HBM에는 GPU 캐시 + 색인 키", SEL, SEL_L)):
        chip(f, px, 134, 200, 40, nm, dark, light, 19)
        f.text(px + 214, 160, sub, 16, color=MUTED)
        f.rect(px, hb_y, W, hb_h, "#ffffff", AXIS, 6, 1.2)
        f.rect(px, ho_y, W, ho_h, HOST_BG, AXIS, 6, 1.2)
    # 기준 — 요청 15개가 HBM을 채움
    px = 190
    bw = (inner - (n - 1) * 3) / n
    top, h = hb_y + 8, hb_h - 16
    cap = h * idx / (kv_b + idx)
    for i in range(n):
        x = px + 8 + i * (bw + 3)
        f.rect(x, top, bw, cap - 1, SEL_L, rx=2)
        f.rect(x, top + cap, bw, h - cap, FULL, rx=2)
    f.text(px, hb_y + hb_h + 28, f"요청당 {(kv_b + idx) / GB:.2f} GB = KV {kv_b / GB:.2f} + 색인 키 {idx / GB:.2f} → 15개에서 HBM 가득",
           16, weight=700)
    f.text(px + W / 2, ho_y + 50, "KV에 쓰지 않음", 16, "middle", color=MUTED)
    # HiSparse — GPU 캐시 + 색인 키만 HBM
    px = 950
    wc = inner * n * kv_h / full
    wi = inner * n * idx / full
    f.rect(px + 8, top, wc, h, SEL, rx=2)
    f.rect(px + 8 + wc + 2, top, wi, h, SEL_L, rx=2)
    fx = px + 8 + wc + wi + 10
    f.rect(fx, top, px + W - 8 - fx, h, "none", AXIS, 4, 1.2, "6 4")
    f.text(fx + 20, top + 42, f"요청당 {(kv_h + idx) / GB:.2f} GB = GPU 캐시 {kv_h / GB:.2f} + 색인 키 {idx / GB:.2f}", 16, weight=700)
    f.text(fx + 20, top + 72, f"15개 합 {n * (kv_h + idx) / GB:.0f} GB — 나머지 HBM은 동시 요청 확대 · prefill에 사용", 15, color=MUTED)
    # host — 15개 요청의 KV 원본
    hw = inner * kv_b / 2000e9
    for i in range(n):
        f.rect(px + 8 + i * hw, ho_y + 10, max(hw - 0.6, 1), ho_h - 20, FULL, rx=1)
    hx = px + 8 + n * hw + 16
    f.text(hx, ho_y + 38, f"15개 요청의 전체 KV {n * kv_b / GB:.0f} GB (2 TB의 {n * kv_b / 2000e9 * 100:.0f}%)", 16, weight=700)
    f.text(hx, ho_y + 64, "KV만 host가 원본 보관 — 색인 키는 HBM에 유지", 14, color=MUTED)
    # 사이 화살표
    f.arrow(px + 14, ho_y - 4, px + 14, hb_y + hb_h + 6, CMP, 2.5)
    f.arrow(px + 30, hb_y + hb_h + 6, px + 30, ho_y - 4, SLATE, 2)
    f.text(px + 48, hb_y + hb_h + 48, "↑ miss만 host → HBM (선택의 약 13%)", 15, weight=700, color=CMP)
    f.text(px + 48, hb_y + hb_h + 74, "↓ 새 KV는 host로 write-through", 15, color=MUTED)
    # 아래 — 핵심 수치 4칸
    tiles = [("요청당 HBM (128K)", f"13.09 → {(kv_h + idx) / GB:.2f} GB", ["KV만 보면 0.4 GB (약 1/30, 논문 수치)",
                                                                       "남는 사용량의 대부분은 색인 키"], SEL),
             ("근거 — 선택 위치 재사용", "hit 87%", ["GPU 캐시 4,096줄 · LRU 교체", "host 전송은 선택의 13%만"], SEL),
             ("효과 — 처리량 (32K 이상)", "2.1 ~ 4.7×", ["전부 '같은 HBM에 큰 배치' 효과", "출력 동일 · DSA · NSA · Quest 적용"], SEL),
             ("대가 — 토큰당 지연", "+3.0 ms", ["miss 전송이 TPOT에 노출 (동시 8)", "층간 prefetch 없으면 +7.7 ms"], CMP)]
    tw = (1620 - 3 * 20) / 4
    for k, (head, big, subs, col) in enumerate(tiles):
        x, y = 40 + k * (tw + 20), 572
        f.rect(x, y, tw, 196, BOX[0], BOX[1], 10, 1.2)
        f.rect(x, y, 8, 196, col, rx=3)
        f.text(x + 28, y + 40, head, 16, weight=700, color=MUTED)
        f.text(x + 28, y + 96, big, 36, weight=700, color=INK)
        for j, s in enumerate(subs):
            f.text(x + 28, y + 138 + j * 28, s, 15, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 1. 문제 — HBM 용량 · 단순 offload의 링크
def fig_capacity():
    f = Fig("01-capacity-wall", 1700, 640)
    f.text(40, 44, "기준 SGLang의 한계 — 연산보다 HBM 용량이 먼저, 단순 offload는 host 링크가 먼저 바닥", 26, weight=700)
    f.text(40, 78, "GLM-5.1-FP8 · 8 × H200 · KV BF16 · 논문 §1 · §2.2 · §2.3 (오른쪽 HiSparse 줄은 miss율 13.4%를 적용한 계산값)", 17, color=MUTED)
    # 왼쪽 — 요청당 KV와 동시 요청 수
    f.text(40, 136, "① HBM 용량 — 요청당 KV cache와 동시 요청 수", 20, weight=700)
    f.text(40, 162, "가중치 · activation · CUDA graph 뒤 남는 KV 여유 약 240 GB", 15, color=MUTED)
    x0, scale = 170, 320 / 110
    rows = [("32K + 8K", 4.0, "4 GB", "동시 약 60개", "여기서 처리량 정체"),
            ("128K", 13.09, "13.09 GB", "약 15개", "32K 대비 1/4"),
            ("1M", 110, "100 GB 이상", "수용 불가", "KV만으로 H200 1장 HBM의 70% 이상")]
    for i, (lab, v, vl, cnt, why) in enumerate(rows):
        y = 200 + i * 92
        f.text(40, y + 30, lab, 19, weight=700)
        w = v * scale
        f.rect(x0, y + 8, w, 34, FULL, rx=3)
        if v > 100:   # 축 밖으로 이어짐
            f.raw(f'<polygon points="{x0 + w:.1f},{y + 8} {x0 + w + 14:.1f},{y + 25} {x0 + w:.1f},{y + 42}" fill="{FULL}"/>')
        f.text(x0 + w + (24 if v > 100 else 10), y + 31, vl, 16, weight=700, color=INK)
        f.text(636, y + 24, cnt, 19, weight=700, color=CMP if v > 100 else INK)
        f.text(636, y + 50, why, 14, color=MUTED)
    f.text(40, 500, "구조적 원인 — top-k 위치는 매 스텝 · 층마다 바뀌어, 지금 안 읽는 KV도 HBM에 상주", 15, color=INK)
    f.text(40, 526, "매 스텝 읽기는 2,048줄에 비례 · HBM 점유는 전체 길이에 비례", 15, color=MUTED)
    f.line(880, 120, 880, 560, GRID, 2)
    # 오른쪽 — 단순 offload의 링크 수요
    X0 = 920
    f.text(X0, 136, "② 단순 offload — host 링크 수요 (GPU 1장)", 20, weight=700)
    f.text(X0, 162, "매 스텝 선택된 KV를 전부 host에서 가져올 때 · TPOT 30 ms 가정", 15, color=MUTED)
    bx, bs = 1100, 5.6          # x 시작, GB/s 당 px
    per = TOPK * 13.09e9 / K128 / 0.030 / GB      # 요청당 GB/s
    n = 12
    f.text(X0, 206, f"요청당 토큰마다 약 200 MB (2,048줄 × 토큰당 KV 약 100 KB) → 요청당 약 {per:.0f} GB/s · 막대 1칸 = 요청 1개",
           14, color=MUTED)
    for i, (lab, sub, rate, col) in enumerate((("캐시 없음", "선택 KV 전부 전송", per, CMP),
                                               ("GPU 캐시 hit 87%", "HiSparse (LRU, 4,096줄)", per * 0.134, SEL))):
        y = 236 + i * 110
        f.text(X0, y + 22, lab, 18, weight=700)
        f.text(X0, y + 46, sub, 14, color=MUTED)
        for r in range(n):
            x = bx + r * rate * bs
            over = (r + 1) * rate > 64
            f.rect(x, y + 4, rate * bs - 2, 40, CMP_L if over else col, rx=2)
        tot = n * rate
        f.text(bx + tot * bs + 10, y + 22, f"요청 12개", 14, color=MUTED)
        f.text(bx + tot * bs + 10, y + 42, f"{tot:.0f} GB/s", 16, weight=700, color=INK)
    lx = bx + 64 * bs
    f.line(lx, 226, lx, 470, INK, 2, "6 4")
    f.text(lx, 492, "PCIe Gen5 x16 한 방향 약 64 GB/s", 15, "middle", 700)
    for v in (0, 20, 40, 60, 80):
        f.text(bx + v * bs, 524, f"{v}", 14, "middle", color=MUTED)
    f.text(bx + 80 * bs + 20, 524, "GB/s", 14, color=MUTED)
    f.text(X0, 590, "→ 요청 10여 개면 miss 전송만으로 링크 포화 — 다시 선택될 KV를 HBM에 남겨 전송 자체를 줄여야 함", 17, weight=700)
    return f


# ---------------------------------------------------------------- 그림 2. 구조 — 층별 decode 흐름
def fig_structure():
    f = Fig("02-structure", 1700, 860)
    f.text(40, 44, "HiSparse 층별 decode 흐름 — 색인이 고른 2,048줄만 HBM 캐시에 맞춘 뒤 attention", 26, weight=700)
    f.text(40, 78, "층 1개 · 요청 1개 · 디코드 1스텝 · k = 2,048 · GPU 캐시 B = 4,096줄 · hit · miss 비율 = GLM-5.1 측정 평균 (LRU, 논문 Fig. 6)",
           17, color=MUTED)
    f.rect(40, 110, 1620, 450, BOX[0], BOX[1], 10, 1.2)
    f.text(64, 146, "GPU · HBM", 19, weight=700)
    f.text(186, 146, "연산은 전부 GPU — 옮기는 것은 attention KV뿐", 15, color=MUTED)
    boxes = [(70, 360, "① 색인", ["색인 키 전부 훑기 → top-k", "출력: 논리 위치 2,048개"]),
             (520, 560, "② Resolve — 층당 커널 1개", ["hit 판정 · LRU 교체 · miss만 host에서 읽기",
                                                   "출력: 물리 슬롯 2,048개 (논리 위치 → 슬롯 변환)"]),
             (1170, 460, "③ sparse attention", ["GPU 캐시 슬롯에서 2,048줄 읽기", "attention 커널은 그대로 — 슬롯 목록만 받음"])]
    for x, w, t, lines in boxes:
        f.rect(x, 170, w, 124, "#ffffff", BOX[1], 8, 1.2)
        f.rect(x, 170, 8, 124, SEL, rx=3)
        f.text(x + 26, 206, t, 19, weight=700)
        for j, s in enumerate(lines):
            f.text(x + 26, 240 + j * 28, s, 15, color=INK if j == 0 else MUTED)
    f.arrow(436, 232, 512, 232, MUTED, 2.5)
    f.arrow(1086, 232, 1162, 232, MUTED, 2.5)
    f.rect(70, 308, 300, 30, SEL_L, rx=6)
    f.text(84, 329, "색인 키 — HBM 상주 · 길이에 비례", 14, weight=700)
    f.text(536, 326, "decode CUDA graph 안에 포함 · '소프트웨어 TLB' 역할", 14, color=MUTED)
    # GPU 캐시 격자 — 칸 1개 = 16줄
    gx, gy, cw, ch, cols, rws = 300, 392, 17, 28, 64, 4
    f.text(gx, gy - 16, "GPU 캐시 — 요청 · 층마다 4,096줄, 길이와 무관 (칸 1개 = 16줄)", 17, weight=700)
    cells = ["hit"] * 111 + ["miss"] * 17 + ["keep"] * 128
    random.Random(7).shuffle(cells)
    colmap = {"hit": SEL, "miss": CMP, "keep": FULL_L}
    for i, c in enumerate(cells):
        cx, cy = gx + (i % cols) * cw, gy + (i // cols) * ch
        f.rect(cx, cy, cw - 2, ch - 3, colmap[c], rx=2)
    lx = gx + cols * cw + 24
    for k, (col, a, b) in enumerate(((SEL, "hit 약 1,770줄 (87%)", "이번 선택 중 이미 HBM에 있음"),
                                     (CMP, "miss 약 270줄 (13%)", "host에서 가져와 교체"),
                                     (FULL_L, "이번엔 미선택 2,048줄", "최근 선택분 · LRU 교체 후보"))):
        y = gy + 14 + k * 38
        f.rect(lx, y - 13, 16, 16, col, rx=3)
        f.text(lx + 24, y + 1, a, 15, weight=700)
        f.text(lx + 24, y + 19, b, 13, color=MUTED)
    # host
    f.rect(40, 640, 1620, 180, HOST_BG, AXIS, 10, 1.2)
    f.text(64, 678, "Host DRAM (pinned)", 19, weight=700)
    f.text(264, 678, "모든 요청의 전체 KV 원본 · 길이에 비례", 15, color=MUTED)
    rnd = random.Random(3)
    nh = 272
    hw = cols * cw / nh
    miss = set(rnd.sample(range(nh), 17))
    for i in range(nh):
        f.rect(gx + i * hw, 726, max(hw - 0.8, 1), 50, CMP if i in miss else FULL_L, rx=1)
    f.text(64, 748, "요청 1개 · 층 1개", 14, color=MUTED)
    f.text(64, 768, "KV 128K줄", 14, color=MUTED)
    f.text(gx, 800, "선택된 위치는 매 스텝 바뀜 — 그중 캐시에 없는 줄만 가져옴", 14, color=MUTED)
    # 화살표
    f.arrow(700, 720, 700, gy + rws * ch + 6, CMP, 2.5)
    f.text(716, 594, "miss만 host에서 전송 — 스텝 · 층당 약 270줄", 15, weight=700, color=CMP)
    f.text(716, 616, "GPU 스레드가 pinned host 메모리를 직접 읽음", 14, color=MUTED)
    f.arrow(1200, gy + rws * ch + 6, 1200, 720, SLATE, 2)
    f.text(1216, 594, "④ 새 토큰 KV — GPU 캐시에 바로 기록", 15, weight=700)
    f.text(1216, 616, "host로 write-through (별도 stream, 다음 스텝과 겹침)", 14, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 3. 근거 — 캐시 크기 · 교체 정책별 miss율
def fig_locality():
    f = Fig("03-locality", 1600, 690)
    f.text(40, 44, "GPU 캐시 miss율 — 스텝마다 선택된 2,048줄 중 HBM에 없는 비율", 26, weight=700)
    f.text(40, 78, "GLM-5.1 · k = 2,048 · LongBenchV2 요청 1개(프롬프트 약 10만 토큰) 선택 기록 재생 · 처음 1,000스텝 · 78층 평균 (논문 Fig. 6)",
           17, color=MUTED)
    for x, h in ((40, "캐시 크기 · 교체 정책"), (440, "스텝당 miss율"), (1120, "해석")):
        f.text(x, 128, h, 16, weight=700, color=MUTED)
    f.line(40, 142, 1560, 142, AXIS, 1)
    x0, sc = 440, 20
    rows = [("B = 2,048 (= k)", "이번 top-k만 보관", 30.0, FULL, None, "연속 스텝 선택의 약 70%가 직전과 겹침"),
            ("B = 4,096 · FIFO", "먼저 들어온 줄부터 교체", 17.2, OTHER, None, "여유 슬롯(B > k)만으로 miss 약 절반"),
            ("B = 4,096 · random", "무작위 교체", 16.1, OTHER, None, "정책 차이보다 캐시 크기 효과가 큼"),
            ("B = 4,096 · LRU", "HiSparse 기본", 13.4, SEL, None, "hit 87% — 스텝 · 층당 약 270줄만 host 전송"),
            ("B = 4,096 · Bélády", "미래를 아는 사후 최적 (구현 불가)", 8.2, SEL, "dash", "LRU가 최적의 추세를 따라감 — 최근성이 좋은 예측 지표"),
            ("B = 8,192 · LRU", "캐시 2배", 6.7, NAVY, None, "캐시 2배 → miss 절반 · 대신 요청당 HBM 2배")]
    top = 150
    RH = 72
    for v in (0, 10, 20, 30):
        f.line(x0 + v * sc, top, x0 + v * sc, top + len(rows) * RH, GRID, 1)
        f.text(x0 + v * sc, top + len(rows) * RH + 24, f"{v}%", 15, "middle", color=MUTED)
    for i, (nm, sub, v, col, dash, note) in enumerate(rows):
        y = top + i * RH
        if i == 3:
            f.rect(30, y + 2, 1540, RH - 4, BOX[0], BOX[1], 8, 1.2)
        if i in (1, 5):
            f.line(40, y, 1560, y, GRID, 1)
        f.text(40, y + 32, nm, 18, weight=700)
        f.text(40, y + 56, sub, 14, color=MUTED)
        if dash:
            f.rect(x0, y + 18, v * sc, 34, "#ffffff", col, 3, 2, "5 4")
        else:
            f.rect(x0, y + 18, v * sc, 34, col, rx=3)
        f.text(x0 + v * sc + 12, y + 42, f"{v:.1f}%" if v != 30 else "30%", 18, weight=700)
        f.text(1120, y + 42, note, 15, weight=700 if i == 3 else 400, color=INK if i in (0, 3, 5) else MUTED)
    f.text(40, 650, "→ 최근 선택된 KV가 다시 선택됨 — 캐시는 '이번 top-k 대기소'가 아니라 k보다 큰 hot KV 캐시로 운영", 17, weight=700)
    return f


# ---------------------------------------------------------------- 그림 4. 효과 ① — 요청당 HBM 사용량
def fig_hbm():
    f = Fig("04-hbm-per-request", 1600, 640)
    f.text(40, 44, "요청당 HBM 사용량 — 기준 = 100%, 컨텍스트 길이별", 26, weight=700)
    f.text(40, 78, "GLM-5.1 (78층) · 층 · 토큰당 KV BF16 1,152 B + 색인 키 FP8 128 B · GPU 캐시 B = 4,096줄 · 계산값 (128K 기준 13.09 GB = 논문 수치)",
           17, color=MUTED)
    legend(f, 40, 122, [(FULL, "KV cache (기준 · HBM 상주)"), (SEL, "GPU 캐시 (HiSparse)"), (SEL_L, "색인 키 (양쪽 모두 HBM 유지)")])
    f.text(1300, 168, "KV만", 15, "middle", 700, MUTED)
    f.text(1450, 168, "색인 키 포함", 15, "middle", 700, MUTED)
    f.text(1375, 146, "HiSparse 감소 배율", 15, "middle", 700, MUTED)
    x0, W = 300, 700
    groups = [("32K + 8K", "40K", K40), ("128K", "", K128), ("1M", "GLM-5.2 기준", M1)]
    for g, (lab, sub, S) in enumerate(groups):
        y = 190 + g * 120
        kv_b, idx = req_hbm(S, False)
        kv_h, _ = req_hbm(S, True)
        base = kv_b + idx
        f.text(40, y + 44, lab, 20, weight=700)
        if sub:
            f.text(40, y + 68, sub, 14, color=MUTED)
        for k, (nm, parts) in enumerate((("기준", ((kv_b, FULL), (idx, SEL_L))),
                                         ("HiSparse", ((kv_h, SEL), (idx, SEL_L))))):
            yy = y + k * 42
            f.text(x0 - 16, yy + 24, nm, 15, "end", 700 if k else 400, INK if k else MUTED)
            x = x0
            for v, col in parts:
                w = W * v / base
                f.rect(x, yy + 2, max(w, 2), 32, col, rx=2)
                x += max(w, 2) + 1
            tot = sum(v for v, _ in parts)
            lab_v = f"{tot / GB:.2f} GB" if tot < 20 * GB else f"{tot / GB:.1f} GB"
            f.text(x + 10, yy + 24, lab_v + ("" if k == 0 else f"  ({tot / base * 100:.0f}%)"), 16, weight=700 if k else 400)
            if k:
                f.text(1300, yy + 24, f"1/{kv_b / kv_h:.0f}", 18, "middle", 700)
                f.text(1450, yy + 24, f"1/{base / tot:.0f}", 18, "middle", 700, SEL)
    f.line(1230, 180, 1230, 540, GRID, 1)
    f.text(40, 570, "→ HiSparse 이후 요청당 HBM의 대부분은 색인 키 (128K 78% · 1M 97%) — 다음 용량 한계는 토큰당 색인 바이트", 17, weight=700)
    f.text(40, 600, "논문의 '약 1/30'(128K)은 KV만 비교한 값 · 1M은 같은 차원의 GLM-5.2 기준(GLM-5.1 지원 길이 약 200K) · 색인 키 scale(4 B) · 메타데이터 제외",
           14, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 5. 효과 ② — 처리량 · TTFT
def fig_throughput():
    f = Fig("05-throughput", 1700, 720)
    f.text(40, 44, "최대 생성 처리량 — 기준 vs HiSparse, 컨텍스트 길이별", 26, weight=700)
    f.text(40, 78, "기준 = SGLang v0.5.11 전체 KV HBM 상주 · 그 외 설정 동일 · KV BF16 · closed-loop 부하의 최대값 (논문 Fig. 4 · 5)", 17, color=MUTED)
    legend(f, 40, 122, [(FULL, "기준"), (SEL, "HiSparse")])
    panels = [("Qwen3-30B-A3B + Quest", "GH200 · NVLink-C2C", "tokens/s", 3000, (0, 1000, 2000, 3000),
               [("4K", 2430, 2668), ("32K", 511, 1824), ("200K", 111, 520)], "×"),
              ("GLM-5.1 + DSA", "8 × H200 · PCIe Gen5", "tokens/s", 3000, (0, 1000, 2000, 3000),
               [("4K", 2288, 2280), ("32K", 624, 1919), ("160K", 232, 680)], "×"),
              ("DeepSeek-V4-Flash", "2 × B200 · 32K / 8K · 동시 64 · 축 다름", "tokens/s", 5000, (0, 2500, 5000),
               [("PD 동거", 600, 1257), ("decode 전용", 1511, 4308)], "×"),
              ("TTFT — V4-Flash", "동시 64 · PD 동거 · 낮을수록 좋음", "초", 1000, (0, 500, 1000),
               [("평균 TTFT", 829, 171)], "/")]
    widths = [450, 450, 380, 280]
    x = 40
    top, bot = 250, 560
    for (nm, sub, unit, vmax, ticks, groups, op), w in zip(panels, widths):
        f.text(x, 172, nm, 19, weight=700)
        f.text(x, 198, sub, 14, color=MUTED)
        px0 = x + 56
        Y = lambda v: bot - v / vmax * (bot - top)  # noqa: E731
        for t in ticks:
            f.line(px0, Y(t), x + w - 10, Y(t), GRID, 1)
            f.text(px0 - 8, Y(t) + 5, f"{t:,}", 13, "end", color=MUTED)
        f.text(px0 - 8, top - 18, unit, 13, "end", color=MUTED)
        slot = (x + w - 10 - px0) / len(groups)
        for i, (lab, b, h) in enumerate(groups):
            cx = px0 + slot * i + slot / 2
            bw = 46
            for k, (v, col) in enumerate(((b, FULL), (h, SEL))):
                bx = cx - bw - 3 + k * (bw + 6)
                f.rect(bx, Y(v), bw, bot - Y(v), col, rx=3)
                f.text(bx + bw / 2, Y(v) - 8, f"{v:,}", 13, "middle", color=MUTED)
            ratio = h / b if op == "×" else b / h
            txt = f"×{ratio:.1f}" if op == "×" else f"1/{ratio:.1f}"
            hi = max(b, h)
            f.text(cx, Y(hi) - 32, txt, 19, "middle", 700, SEL if ratio >= 1.5 else MUTED)
            f.text(cx, bot + 28, lab, 16, "middle", 700)
        f.line(px0, bot, x + w - 10, bot, AXIS, 1.5)
        x += w + 20
    f.line(40 + 450 + 450 + 380 + 50, 160, 40 + 450 + 450 + 380 + 50, 600, GRID, 2)
    f.text(40, 640, "→ 4K는 HBM 여유가 있어 효과 없음(1.0 ~ 1.1×) — 길어질수록 같은 HBM에 더 많은 요청을 담아 처리량 증가", 17, weight=700)
    f.text(40, 670, "스텝 자체는 빨라지지 않음 — 같은 처리량 구간의 TPOT는 비슷 (V4-Flash 동시 16: 15.9 vs 16.0 ms), 더 높은 처리량 지점에서는 TPOT 상승",
           15, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 6. 대가 — host IO와 층간 prefetch
def fig_prefetch():
    f = Fig("06-prefetch", 1700, 900)
    f.text(40, 44, "host IO 숨기기 — 층간 색인 공유 모델의 정확한 prefetch", 26, weight=700)
    f.text(40, 78, "GLM-5.2-FP8 (IndexShare: 78층 중 21층만 색인 실행) · 8 × H200 · PD 동거 · 32K / 8K · k = 2,048 · B = 4,096 (논문 §3.5 · §4.6 · Fig. 8)",
           17, color=MUTED)
    f.text(40, 132, "층별 실행 순서 — 개념도 (길이 비율은 실제와 다름)", 19, weight=700)
    legend(f, 500, 132, [(SEL_L, "색인 · top-k"), (CMP, "host IO 대기"), (CMP_L, "미리 복사 (별도 stream)"),
                         (SEL, "attention"), (FULL_L, "MoE · 기타 연산")], 14, 26)
    x0 = 300
    U = {"idx": 50, "io": 70, "att": 50, "moe": 100}
    C = {"idx": SEL_L, "io": CMP, "att": SEL, "moe": FULL_L}
    seq_sync = [("1층 · anchor", ["idx", "io", "att", "moe"]), ("2층 · shared", ["io", "att", "moe"]),
                ("3층 · shared", ["io", "att", "moe"]), ("4층 · anchor", ["idx", "io", "att", "moe"])]
    seq_pf = [("1층 · anchor", ["idx", "io", "att", "moe"]), ("2층 · shared", ["att", "moe"]),
              ("3층 · shared", ["att", "moe"]), ("4층 · anchor", ["idx", "io", "att", "moe"])]

    def row(seq, y, label, sub):
        f.text(40, y + 26, label, 18, weight=700)
        f.text(40, y + 50, sub, 14, color=MUTED)
        x = x0
        starts = []
        for nm, parts in seq:
            sx = x
            for p in parts:
                f.rect(x, y + 4, U[p] - 2, 40, C[p], rx=3)
                x += U[p]
            f.text((sx + x) / 2, y - 8, nm, 13, "middle", color=MUTED)
            f.line(x - 1, y - 2, x - 1, y + 50, AXIS, 1, "3 3")
            starts.append(sx)
        return x, starts

    end_a, _ = row(seq_sync, 196, "동기 Resolve", "층마다 miss를 기다림")
    end_b, st = row(seq_pf, 316, "정확한 prefetch", "shared 층은 Resolve 생략")
    # 복사 stream — anchor Resolve 직후 다음 shared 층 KV를 미리 채움
    f.text(x0 - 12, 384, "복사 stream", 13, "end", color=MUTED)
    for a, labs in ((st[0], ("2층", "3층")), (st[3], ("5층", "6층"))):
        cx = a + U["idx"] + U["io"]
        for lab in labs:
            f.rect(cx, 368, 48, 24, CMP_L, rx=3)
            f.text(cx + 24, 385, lab, 12, "middle", 700, INK)
            cx += 50
    for xe in (end_b, end_a):
        f.line(xe, 196, xe, 400, INK, 1.2, "5 4")
    f.arrow(end_a - 2, 286, end_b + 4, 286, INK, 1.8)
    f.text((end_a + end_b) / 2, 278, "IO 대기 감소", 14, "middle", 700, halo=True)
    f.text(1320, 222, "IO 대기가 매 층 노출", 16, weight=700)
    f.text(1320, 246, "동시 8에서 토큰당 +7.7 ms", 14, color=MUTED)
    f.text(1320, 342, "shared 층 IO는 다른 층 연산 뒤에 숨김", 16, weight=700)
    f.text(1320, 366, "anchor 층 miss는 동기로 남음 (IO의 약 27%)", 14, color=MUTED)
    f.line(40, 430, 1660, 430, GRID, 1.5)
    # 아래 왼쪽 — 최대 처리량
    f.text(40, 474, "최대 처리량 (tokens/s)", 19, weight=700)
    f.text(40, 498, "PD 동거 · closed-loop 최대값", 14, color=MUTED)
    bx0, bsc = 320, 330 / 2034
    rows = [("기준", "전체 KV HBM 상주", 618, FULL, None, "618"),
            ("HiSparse · 동기 Resolve", "prefetch 없음", 1515, SYNC, None, "1,515 · 상한의 74%"),
            ("HiSparse · 정확한 prefetch", "anchor 선택을 shared 층에 미리 적용", 1727, SEL, None, "1,727 · 기준의 2.8× · 상한의 85%"),
            ("IO 없는 상한", "oracle — 출력 무효, 비교용", 2034, SEL, "dash", "2,034")]
    for i, (nm, sub, v, col, dash, lab) in enumerate(rows):
        y = 522 + i * 62
        f.text(40, y + 24, nm, 16, weight=700)
        f.text(40, y + 46, sub, 13, color=MUTED)
        if dash:
            f.rect(bx0, y + 8, v * bsc, 34, "#ffffff", col, 3, 2, "5 4")
        else:
            f.rect(bx0, y + 8, v * bsc, 34, col, rx=3)
        f.text(bx0 + v * bsc + 12, y + 31, lab, 15, weight=700 if i == 2 else 400)
    f.line(960, 456, 960, 770, GRID, 1.5)
    # 아래 오른쪽 — 토큰당 노출된 IO
    X = 1000
    f.text(X, 474, "토큰당 노출된 IO (ms)", 19, weight=700)
    f.text(X, 498, "IO 없는 상한 대비 TPOT 증가분 · 낮을수록 좋음", 14, color=MUTED)
    rx0, rsc = 1200, 380 / 25
    for g, (lab, a, b) in enumerate((("동시 8", 7.7, 3.0), ("동시 256", 22.0, 11.2))):
        y = 532 + g * 116
        f.text(X, y + 40, lab, 18, weight=700)
        for k, (nm, v, col) in enumerate((("동기 Resolve", a, SYNC), ("정확한 prefetch", b, SEL))):
            yy = y + k * 44
            f.text(rx0 - 12, yy + 26, nm, 14, "end", color=MUTED)
            f.rect(rx0, yy + 6, v * rsc, 30, col, rx=3)
            extra = f"  (−{(1 - b / a) * 100:.0f}%)" if k else ""
            f.text(rx0 + v * rsc + 10, yy + 28, f"{v:.1f} ms{extra}", 15, weight=700 if k else 400)
    f.text(40, 816, "→ Resolve 자체 비용은 측정 불가 수준 (동시 8: IO 없는 상한 TPOT 24.1 ms ≈ 기준 24.8 ms) — 남은 overhead는 전부 host IO", 17, weight=700)
    f.text(40, 846, "prefetch는 전송 시점만 앞당길 뿐 양은 그대로 — 고부하에서는 demand miss와 같은 host 링크를 경쟁 (동시 256에서 11.2 ms 잔존)", 15, color=MUTED)
    f.text(40, 872, "색인 공유가 없는 모델의 추측 prefetch(ℓ층 선택을 ℓ+1층 힌트로 사용)는 hit률만 소폭 상승, 처리량 변화 없음", 15, color=MUTED)
    return f


FIGS = [fig_summary, fig_capacity, fig_structure, fig_locality, fig_hbm, fig_throughput, fig_prefetch]


def print_summary():
    print("요청당 HBM (GB): 기준(KV + 색인) → HiSparse(GPU 캐시 + 색인)")
    for lab, S in (("40K", K40), ("128K", K128), ("1M", M1)):
        kb, ix = req_hbm(S, False)
        kh, _ = req_hbm(S, True)
        print(f"  {lab:5s} {kb / GB:7.2f} + {ix / GB:5.2f} = {(kb + ix) / GB:7.2f} → {kh / GB:.2f} + {ix / GB:5.2f} = {(kh + ix) / GB:6.2f}"
              f" | KV만 1/{kb / kh:.0f} · 색인 포함 1/{(kb + ix) / (kh + ix):.1f} · HiSparse 중 색인 {ix / (kh + ix) * 100:.0f}%")
    per = TOPK * 13.09e9 / K128 / 0.030 / GB
    print(f"단순 offload: 요청당 {per:.1f} GB/s → 64 GB/s 포화 {64 / per:.1f}개 | hit 87%면 {per * 0.134:.2f} GB/s")


def main():
    print_summary()
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
