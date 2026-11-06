"""Sparse · Hybrid attention 발표 자료(02-deck-draft-claude.md)용 그림 생성기.
실행: python arch_insight/figures-claude/make_figures.py [그림 이름 ...]  → 같은 폴더에 *.svg, *.png
그림 양식은 reports/llm-memory-requirements/figures/make_figures.py와 같다 (SVG를 직접 그리고 headless 브라우저로 PNG).

수치 기준
- 층 구성 · head 수 · 차원 : 공식 config.json (reports/llm-memory-requirements/data/raw/)
- 1M = 1,000,000 토큰, 1 GB = 10^9 B
- 층당 비교(그림 4 · 5)는 "구조 효과" 기준: KV cache는 BF16으로 통일, 색인 키는 각 방식의 공개 정밀도
  DSA 색인 키 128차원 FP8 = 128 B · CSA 색인 키 128차원 FP4 = 64 B · QSA 블록 키 128차원은 정밀도 표기가 없어 BF16(256 B)로 가정
- KV cache 읽기는 한 번씩만 읽는다고 본 하한이다. top-k · 흩어진 읽기의 비효율은 넣지 않았다
"""
import csv
import json
import math
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(ROOT, "reports", "llm-memory-requirements", "data", "raw")
LINEAGE = os.path.join(ROOT, "reports", "llm-memory-requirements", "data", "lineage-kv1m.csv")

FONT = "'Noto Sans KR','Malgun Gothic',sans-serif"
INK = "#2b2f3a"
MUTED = "#6b7080"
GRID = "#e6e8f1"
AXIS = "#9aa0b0"
SLATE = "#4a5573"
BOX = ("#f6f7fb", "#dfe2ec")

# 방식별 색 — 덱 전체에서 고정 (진한 색, 옅은 색)
FULL, FULL_L = "#9c9a92", "#dddcd6"      # 전부 읽기
SEL, SEL_L = "#2a78d6", "#b7d3f6"        # 골라 읽기 (DSA · QSA)
CMP, CMP_L = "#eb6834", "#f8c6b1"        # 묶어 저장 (CSA · HCA)
STATE, STATE_L = "#1baf7a", "#a9e0cb"    # linear state
WIN, WIN_L = "#eda100", "#f7d98f"        # 최근만 (sliding window)

GB, MB = 1e9, 1e6
M = 1_000_000


# ---------------------------------------------------------------- 비용 모델 (층 1개 · 요청 1개 · decode 1스텝)
MLA_B = 576 * 2            # DeepSeek-V3 / V3.2 latent 512 + RoPE 64, BF16
DSA_IDX = 128              # V3.2 색인 키 128차원 FP8
TOPK = 2048
GQA_B = 2 * 2 * 256 * 2    # Qwen3.8-Flash-Next full 층: KV head 2 × 256 × (K, V) × BF16
QSA_IDX = 128 * 2          # 블록 키 128차원, BF16 가정
QSA_BLOCK = 4
V4_B = 512 * 2             # V4 엔트리 512차원 (K = V), BF16
CSA_IDX = 64               # V4 CSA 색인 키 128차원 FP4
CSA_K = 1024
RECENT = 128               # V4 최근 원본 토큰
GDN_STATE = 48 * 128 * 128 * 4 + (2 * 16 * 128 + 48 * 128) * 3 * 2   # Qwen3.8-Flash-Next DeltaNet 층 하나: FP32 state + conv


def read_parts(kind, S):
    """(색인 키 읽기, KV cache · state 읽기) 바이트."""
    if kind == "mla":
        return 0, MLA_B * S
    if kind == "dsa":
        return DSA_IDX * S, MLA_B * min(S, TOPK)
    if kind == "gqa":
        return 0, GQA_B * S
    if kind == "qsa":
        return QSA_IDX * (S // QSA_BLOCK), GQA_B * min(S, TOPK)
    if kind == "csa":
        return CSA_IDX * S / 4, V4_B * min(S / 4, CSA_K) + V4_B * min(S, RECENT)
    if kind == "hca":
        return 0, V4_B * S / 128 + V4_B * min(S, RECENT)
    if kind == "gdn":
        return 0, GDN_STATE
    raise KeyError(kind)


def store_parts(kind, S):
    """(색인 키 저장, KV cache · state 저장) 바이트."""
    if kind == "mla":
        return 0, MLA_B * S
    if kind == "dsa":
        return DSA_IDX * S, MLA_B * S
    if kind == "gqa":
        return 0, GQA_B * S
    if kind == "qsa":
        return QSA_IDX * (S // QSA_BLOCK), GQA_B * S
    if kind == "csa":
        return CSA_IDX * S / 4, V4_B * S / 4 + V4_B * min(S, RECENT)
    if kind == "hca":
        return 0, V4_B * S / 128 + V4_B * min(S, RECENT)
    if kind == "gdn":
        return 0, GDN_STATE
    raise KeyError(kind)


def total(parts):
    return sum(parts)


# ---------------------------------------------------------------- SVG 도구
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Fig:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.o = name, w, h, []

    def raw(self, s):
        self.o.append(s)

    def rect(self, x, y, w, h, fill, stroke="none", rx=4, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=20, anchor="start", weight=400, color=INK, halo=False):
        h = ' stroke="#ffffff" stroke-width="7" stroke-linejoin="round" paint-order="stroke"' if halo else ""
        self.raw(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" text-anchor="{anchor}" '
                 f'font-weight="{weight}" fill="{color}"{h}>{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, color=AXIS, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                 f'stroke-width="{sw}" stroke-linecap="round"{d}/>')

    def poly(self, pts, color, sw=3, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.raw(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" '
                 f'stroke-linejoin="round" stroke-linecap="round"{d}/>')

    def arrow(self, x1, y1, x2, y2, color=MUTED, sw=2):
        self.line(x1, y1, x2, y2, color, sw)
        a = math.atan2(y2 - y1, x2 - x1)
        p1 = (x2 - 12 * math.cos(a - 0.45), y2 - 12 * math.sin(a - 0.45))
        p2 = (x2 - 12 * math.cos(a + 0.45), y2 - 12 * math.sin(a + 0.45))
        self.raw(f'<polygon points="{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>')

    def svg(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}"><rect width="100%" height="100%" fill="#ffffff"/>' + "".join(self.o) + "</svg>")


def fmt_b(v):
    if v >= 1e9:
        return f"{v / 1e9:.2f} GB"
    if v >= 10e6:
        return f"{v / 1e6:.0f} MB"
    if v >= 1e6:
        return f"{v / 1e6:.1f} MB"
    if v >= 1e3:
        return f"{v / 1e3:.0f} KB"
    return f"{v:.0f} B"


def basis(f, s, x=40, y=40):
    f.text(x, y, s, 19, color=MUTED)


# ---------------------------------------------------------------- 그림 1. 반기별 attention 구성
CAT = [("full", "Full attention만", FULL), ("sparse", "Sparse · 압축 (+ full)", SEL),
       ("state", "Linear state + attention (hybrid)", STATE), ("swa", "Sliding window + full (hybrid)", WIN)]
HALVES = ["2024 H2", "2025 H1", "2025 H2", "2026 H1", "2026 H2"]


def _half(d):
    return f"{d[:4]} H{1 if int(d[5:7]) <= 6 else 2}"


def fig_trend():
    rows = list(csv.DictReader(open(LINEAGE, encoding="utf-8-sig")))
    cnt = {h: {c: 0 for c, _, _ in CAT} for h in HALVES}
    for r in rows:
        cnt[_half(r["date"])][r["category"]] += 1
    f = Fig("01-trend", 1500, 700)
    basis(f, f"공개 모델의 attention 구성 비율 · 반기별 · 9개 시리즈 {len(rows)}개 모델 (공식 config 기준)")
    x0, x1, top, bot = 110, 1000, 130, 600
    Y = lambda p: bot - p * (bot - top)
    for p in (0, 0.25, 0.5, 0.75, 1):
        f.line(x0, Y(p), x1, Y(p), GRID, 1)
        f.text(x0 - 12, Y(p) + 6, f"{p * 100:.0f}%", 16, "end", color=MUTED)
    slot = (x1 - x0) / len(HALVES)
    bw = 112
    last_mid = {}
    for i, h in enumerate(HALVES):
        n = sum(cnt[h].values())
        bx = x0 + slot * i + (slot - bw) / 2
        acc = 0
        for c, _, col in CAT:
            k = cnt[h][c]
            if not k:
                continue
            y_hi, y_lo = Y((acc + k) / n), Y(acc / n)
            f.rect(bx, y_hi + 1, bw, y_lo - y_hi - 2, col, rx=3)
            if y_lo - y_hi >= 26:
                f.text(bx + bw / 2, (y_hi + y_lo) / 2 + 7, str(k), 18, "middle", 700,
                       INK if col == WIN else "#ffffff")
            if i == len(HALVES) - 1:
                last_mid[c] = (y_hi + y_lo) / 2
            acc += k
        share = cnt[h]["full"] / n
        f.text(bx + bw / 2, top - 38, f"Full만 {share * 100:.0f}%", 20, "middle", 700)
        f.text(bx + bw / 2, bot + 32, f"'{h[2:4]} {'상반기' if h.endswith('H1') else '하반기'}", 19, "middle", 700)
        f.text(bx + bw / 2, bot + 58, f"{n}개", 16, "middle", color=MUTED)
    f.line(x0, bot, x1, bot, AXIS, 1.5)
    lx = x1 + 30
    bx_end = x0 + slot * (len(HALVES) - 1) + (slot + bw) / 2
    used = []
    for c, label, col in CAT:
        y = last_mid[c]
        for u in used:
            if abs(y - u) < 30:
                y = u + 30
        used.append(y)
        f.line(bx_end + 4, last_mid[c], lx - 6, y, AXIS, 1.2)
        f.rect(lx, y - 9, 16, 16, col, rx=3)
        f.text(lx + 24, y + 6, label, 18)
    return f


# ---------------------------------------------------------------- 그림 2. MoE vs Sparse attention
def fig_moe():
    f = Fig("02-moe-analogy", 1600, 860)
    rnd = random.Random(7)
    # 왼쪽 — MoE
    f.text(395, 50, "FFN → MoE", 28, "middle", 700)
    f.text(395, 82, "토큰마다 expert 일부만 쓴다", 19, "middle", color=MUTED)
    for j, nm in enumerate(["요청 A", "요청 B", "요청 C"]):
        y = 170 + j * 90
        f.rect(50, y, 100, 44, BOX[0], BOX[1], 8)
        f.text(100, y + 29, nm, 18, "middle")
        f.arrow(154, y + 22, 206, 282, AXIS, 1.6)
    f.rect(212, 256, 104, 52, SLATE, rx=8)
    f.text(264, 289, "router", 19, "middle", 700, "#ffffff")
    f.arrow(320, 282, 366, 282, AXIS, 2)
    gx, gy, cs, gp = 380, 128, 14, 4
    pick = set(rnd.sample(range(256), 10))
    for k in range(256):
        r, c = divmod(k, 16)
        f.rect(gx + c * (cs + gp), gy + r * (cs + gp), cs, cs, SLATE if k in pick else FULL_L, rx=2)
    f.text(gx + 8 * (cs + gp), gy + 16 * (cs + gp) + 30, "expert 256개 — 모든 요청이 같은 가중치를 쓴다", 18, "middle", color=MUTED)
    f.line(800, 40, 800, 450, GRID, 2)
    # 오른쪽 — Sparse
    f.text(1200, 50, "Full → Sparse attention", 28, "middle", 700)
    f.text(1200, 82, "토큰마다 KV cache 일부만 읽는다", 19, "middle", color=MUTED)
    f.rect(840, 256, 112, 52, SEL, rx=8)
    f.text(896, 289, "indexer", 19, "middle", 700, "#ffffff")
    sx, sw_ = 1010, 420
    for j, nm in enumerate(["요청 A", "요청 B", "요청 C"]):
        y = 165 + j * 100
        f.arrow(956, 282, sx - 10, y + 17, AXIS, 1.6)
        f.text(sx + sw_ + 14, y + 24, nm, 18, color=MUTED)
        n = 140
        cw = sw_ / n
        hit = set(rnd.sample(range(n), 9))
        for k in range(n):
            f.rect(sx + k * cw, y + (0 if k in hit else 8), max(cw - 0.8, 1), 34 if k in hit else 18,
                   SEL if k in hit else FULL_L, rx=0.8)
    f.text(1200, 430, "KV cache 100만 줄 — 요청마다 따로, 토큰마다 한 줄씩 늘어난다", 18, "middle", color=MUTED)
    rows = [("고를 후보", "expert 256개 · 고정", "KV 100만 줄 (1M) · 계속 증가"),
            ("한 번에 가져오는 단위", "expert 1개 ≈ 44 MB", "KV 1줄 ≈ 1 KB"),
            ("여러 요청이 나눠 쓰나", "예 — 같은 가중치", "아니오 — 요청마다 다른 KV cache"),
            ("시스템 병목", "네트워크 (all-to-all)", "메모리 (찾기 · 흩어진 읽기 · 용량)")]
    y0 = 490
    for i, (lab, a, b) in enumerate(rows):
        y = y0 + i * 88
        f.rect(40, y, 1520, 78, BOX[0] if i % 2 == 0 else "#ffffff", "none", 8)
        f.text(60, y + 48, lab, 19, weight=700, color=MUTED)
        f.text(520, y + 48, a, 23, "middle", 700 if i == 3 else 400)
        f.text(1200, y + 48, b, 23, "middle", 700 if i == 3 else 400)
    return f


# ---------------------------------------------------------------- 그림 3. KV cache를 표로 보면 — 토큰 수 × 토큰당 차원
def _rows(f, x, y, w, h, n, fill, edge, pick=(), pick_fill=None, rx=3):
    rh = h / n
    for k in range(n):
        f.rect(x, y + k * rh + 1, w, rh - 2, pick_fill if k in pick else fill, rx=2)
    f.rect(x - 1, y - 1, w + 2, h + 2, "none", edge, rx, 1.5)


def fig_axes():
    f = Fig("03-kv-axes", 1600, 720)
    basis(f, "층 1개의 KV cache를 표로 본 그림 · 한 줄 = 토큰 하나")
    top, H, n = 200, 360, 14
    # 왼쪽 — 기준 표 (MHA)
    tx, tw = 230, 270
    f.text(tx + tw / 2, 120, "토큰당 차원 = KV 1줄 크기", 21, "middle", 700)
    f.text(tx + tw / 2, 148, "K · V 2 × 헤드 수 × 헤드 차원", 16, "middle", color=MUTED)
    f.line(tx, 170, tx + tw, 170, INK, 2)
    f.line(tx, 162, tx, 178, INK, 2)
    f.line(tx + tw, 162, tx + tw, 178, INK, 2)
    f.line(tx - 26, top, tx - 26, top + H, INK, 2)
    f.line(tx - 34, top, tx - 18, top, INK, 2)
    f.line(tx - 34, top + H, tx - 18, top + H, INK, 2)
    f.text(tx - 44, top + H / 2 - 8, "토큰 수 S", 21, "end", 700)
    f.text(tx - 44, top + H / 2 + 20, "토큰마다 1줄씩 추가", 16, "end", color=MUTED)
    _rows(f, tx, top, tw, H, n, FULL_L, FULL, pick={6}, pick_fill=FULL)
    f.text(tx + tw / 2, top + H / n * 6.5 + 7, "KV 1줄 = 토큰 1개", 18, "middle", 700, "#ffffff")
    f.line(560, 90, 560, 680, GRID, 2, "6 6")

    def stats(x, a, b):
        f.text(x, 615, "1줄 크기", 16, color=MUTED)
        f.text(x + 82, 615, a, 19, weight=700)
        f.text(x, 650, "줄 수", 16, color=MUTED)
        f.text(x + 82, 650, b, 19, weight=700)

    stats(tx - 10, "32,768 (MHA 가정)", "S · 매번 전부 읽음")
    # MLA — 가로를 줄인다
    cx = 650
    f.text(cx + 105, 120, "MLA", 28, "middle", 700)
    f.text(cx + 105, 152, "가로(1줄 크기)를 줄인다", 18, "middle", color=MUTED)
    f.rect(cx, top, 210, H, "none", AXIS, 3, 1.5, "6 5")
    _rows(f, cx, top, 46, H, n, FULL_L, FULL)
    f.arrow(cx + 196, top + H / 2, cx + 60, top + H / 2, MUTED, 2.5)
    stats(cx, "576", "S · 매번 전부 읽음")
    # DSA — 읽을 줄만 고른다
    dx = 960
    f.text(dx + 90, 120, "DSA", 28, "middle", 700)
    f.text(dx + 90, 152, "표는 그대로, 읽을 줄만 고른다", 18, "middle", color=MUTED)
    _rows(f, dx, top, 46, H, n, FULL_L, FULL, pick={1, 4, 5, 9, 12}, pick_fill=SEL)
    _rows(f, dx + 58, top, 14, H, n, SEL_L, SEL)
    f.text(dx + 84, top + 16, "색인 키", 16, color=MUTED)
    f.rect(dx + 84, top + H - 52, 16, 16, SEL, rx=3)
    f.text(dx + 108, top + H - 38, "이번 스텝에 읽는 줄", 16, color=MUTED)
    stats(dx, "576 + 색인 128", "S · 2,048줄만 읽음")
    # CSA · HCA — 세로를 줄인다
    sx = 1260
    f.text(sx + 150, 120, "CSA · HCA", 28, "middle", 700)
    f.text(sx + 150, 152, "세로(줄 수)를 줄인다", 18, "middle", color=MUTED)
    for x, h_fill, col, edge, nm, ratio in ((sx, H / 4, CMP, CMP, "CSA", "4줄 → 1줄"),
                                            (sx + 170, 10, CMP_L, CMP, "HCA", "128줄 → 1줄")):
        f.rect(x, top, 46, H, "none", AXIS, 3, 1.5, "6 5")
        k = 3 if nm == "CSA" else 1
        _rows(f, x, top, 46, h_fill, k, col, edge)
        f.arrow(x + 23, top + H - 12, x + 23, top + h_fill + 18, MUTED, 2.5)
        f.text(x + 58, top + 24, nm, 21, weight=700)
        f.text(x + 58, top + 50, ratio, 16, color=MUTED)
    stats(sx, "512", "S/4 · S/128")
    f.text(1570, 700, "그림의 비율은 실제와 다르다", 15, "end", color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 4. 층 구성 띠
def _cfg(name):
    with open(os.path.join(RAW, f"cfg_{name}.json"), encoding="utf-8") as fh:
        c = json.load(fh)
    return c.get("text_config", c)


def fig_layers():
    v4 = _cfg("deepseek-ai_DeepSeek-V4-Pro")
    ratios = v4["compress_ratios"][:v4["num_hidden_layers"]]
    q38 = _cfg("Qwen_Qwen3.8-Flash-Next")["layer_types"]
    rows = [
        ("DeepSeek-V3", "2024-12", [FULL] * 61, "61 / 61층", "KV cache를 매번 전부 읽음"),
        ("gpt-oss-120b", "2025-08", [WIN, FULL] * 18, "18 / 36층", "절반은 최근 128토큰 KV만"),
        ("DeepSeek-V3.2", "2025-12", [SEL] * 61, "61 / 61층", "KV cache는 그대로, 읽기만 골라서"),
        ("DeepSeek-V4-Pro", "2026-04", [CMP if r == 4 else CMP_L for r in ratios], "61 / 61층",
         "진한 = KV 4줄을 1줄로 · 옅은 = 128줄을 1줄로"),
        ("Qwen3.8-Flash-Next", "2026-08", [SEL if t == "full_attention" else STATE for t in q38], "12 / 48층",
         "나머지 36층은 고정 크기 state"),
    ]
    f = Fig("A1-layer-strips", 1600, 760)
    legend = [(FULL, "전부 읽기", "KV cache를 매번 전부"), (SEL, "골라 읽기", "색인으로 찾아 KV 일부만"),
              (CMP, "묶어 저장", "KV 여러 줄을 한 줄로"), (STATE, "Linear state", "KV cache 대신 고정 크기 state"),
              (WIN, "최근만", "최근 W토큰 KV만 보관")]
    lx = 40
    for col, nm, d in legend:
        f.rect(lx, 30, 22, 22, col, rx=4)
        f.text(lx + 32, 49, nm, 21, weight=700)
        f.text(lx + 32, 78, d, 16, color=MUTED)
        lx += 312
    x0, x1 = 330, 1170
    f.text(1200, 150, "KV cache가 길이에 따라 자라는 층", 17, weight=700, color=MUTED)
    for i, (nm, date, cells, ann, sub) in enumerate(rows):
        y = 180 + i * 108
        f.text(40, y + 26, nm, 22, weight=700)
        f.text(40, y + 52, date, 16, color=MUTED)
        n = len(cells)
        cw = (x1 - x0) / n
        for k, col in enumerate(cells):
            f.rect(x0 + k * cw, y, cw - 2, 48, col, rx=2)
        f.text(1200, y + 24, ann, 22, weight=700)
        f.text(1200, y + 50, sub, 16, color=MUTED)
    f.text(x0, 180 + 5 * 108 - 10, "1층 →", 15, color=MUTED)
    f.text(x1, 180 + 5 * 108 - 10, "→ 마지막 층", 15, "end", color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 5. KV cache 읽기 vs 크기 (층당, 1M)
FAMILIES = [
    ("DeepSeek", [("전부 읽기 · MLA", "V3", "mla", FULL, FULL_L), ("골라 읽기 · DSA", "V3.2", "dsa", SEL, SEL_L),
                  ("묶기 4줄 → 1 · CSA", "V4", "csa", CMP, CMP_L), ("묶기 128줄 → 1 · HCA", "V4", "hca", CMP, CMP_L)], "mla"),
    ("Qwen", [("전부 읽기 · GQA", "Qwen3.8 full 층", "gqa", FULL, FULL_L),
              ("골라 읽기 · QSA", "Qwen3.8", "qsa", SEL, SEL_L),
              ("Linear state · DeltaNet", "Qwen3.8", "gdn", STATE, STATE_L)], "gqa"),
]


def fig_read_store():
    S = M
    f = Fig("04-kv-read-vs-size", 1600, 760)
    basis(f, "층 1개 · 요청 1개 · 컨텍스트 1M 토큰 · KV cache는 BF16으로 통일 (계산값)")
    pw, gap = 420, 170
    X1, X2 = 390, 390 + pw + gap
    vmax = 2.2e9
    sc = pw / vmax
    f.text(X1, 100, "디코드 1스텝의 KV cache 읽기", 24, weight=700)
    f.text(X2, 100, "KV cache 크기", 24, weight=700)
    for X in (X1, X2):
        for v in (0, 1e9, 2e9):
            f.line(X + v * sc, 120, X + v * sc, 705, GRID, 1)
            f.text(X + v * sc, 732, f"{v / 1e9:.0f} GB", 15, "middle", color=MUTED)
    y = 140
    for fam, items, base in FAMILIES:
        f.text(40, y + 6, fam, 17, weight=700, color=MUTED)
        y += 18
        base_read = total(read_parts(base, S))
        base_store = total(store_parts(base, S))
        for label, model, kind, dark, light in items:
            f.text(40, y + 30, label, 21, weight=700)
            f.text(40, y + 54, model, 15, color=MUTED)
            for X, parts, ref in ((X1, read_parts(kind, S), base_read), (X2, store_parts(kind, S), base_store)):
                idx, main = parts
                w_idx, w_main = idx * sc, max(main * sc, 3)
                if idx:
                    f.rect(X, y + 12, w_idx, 36, light, rx=3)
                f.rect(X + w_idx + (2 if idx else 0), y + 12, w_main, 36, dark, rx=3)
                tot = idx + main
                lab = fmt_b(tot)
                if kind == "gdn":
                    lab += "  + 같은 양 다시 씀 (길이와 무관)" if X == X1 else "  (고정)"
                elif kind != base:
                    r = tot / ref
                    lab += f"  ({'1/' + format(1 / r, '.0f') if r < 0.95 else '×' + format(r, '.2f')})"
                f.text(X + w_idx + w_main + 12, y + 37, lab, 18, weight=700 if X == X1 else 400)
            y += 74
        y += 14
    lx = 1010
    f.rect(lx, 22, 18, 18, SEL_L, rx=3)
    f.text(lx + 26, 38, "옅은 색 = indexer 색인 키", 17, color=MUTED)
    f.rect(lx + 270, 22, 18, 18, SEL, rx=3)
    f.text(lx + 296, 38, "진한 색 = KV cache · state", 17, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 6. 컨텍스트 길이에 따른 KV cache 읽기 절감
CURVES = [("dsa", "mla", "DSA (V3.2)", SEL, None), ("qsa", "gqa", "QSA (Qwen3.8)", SEL, "10 6"),
          ("csa", "mla", "CSA (V4)", CMP, None), ("hca", "mla", "HCA (V4)", CMP, "10 6")]


def fig_gain():
    f = Fig("05-kv-read-saving-by-length", 1600, 780)
    basis(f, "층 1개 · 디코드 1스텝 · Full 대비 KV cache 읽기가 몇 분의 1인가 (계산값, 속도 배수가 아니다)")
    x0, x1, top, bot = 130, 1240, 90, 680
    lo, hi = 1024, 10 * M
    X = lambda s: x0 + (math.log10(s) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)
    ylo, yhi = 0.5, 300
    Y = lambda v: bot - (math.log10(v) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo)) * (bot - top)
    f.rect(X(131072), top, X(10 * M) - X(131072), bot - top, "#f3f4f8", rx=0)
    f.text(X(131072) + 14, top + 28, "긴 컨텍스트 구간", 17, weight=700, color=MUTED)
    for v, lab in ((0.5, "0.5×"), (1, "1×"), (3, "3×"), (10, "10×"), (30, "30×"), (100, "100×"), (300, "300×")):
        f.line(x0, Y(v), x1, Y(v), GRID, 1)
        f.text(x0 - 12, Y(v) + 6, lab, 16, "end", color=MUTED)
    for s, lab in ((1024, "1K"), (8192, "8K"), (32768, "32K"), (131072, "128K"), (M, "1M"), (10 * M, "10M")):
        f.line(X(s), top, X(s), bot, GRID, 1)
        f.text(X(s), bot + 30, lab, 18, "middle", color=MUTED)
    f.text((x0 + x1) / 2, bot + 62, "컨텍스트 길이 (토큰)", 18, "middle", color=MUTED)
    f.line(x0, Y(1), x1, Y(1), INK, 2, "6 5")
    f.text(x0 + 10, Y(1) + 28, "1× = Full과 같음 · 2K 이하에서 DSA는 0.9× (오히려 더 읽음)", 16, color=MUTED)
    ends = []
    for kind, base, label, col, dash in CURVES:
        pts = []
        for i in range(0, 241):
            s = int(lo * (hi / lo) ** (i / 240))
            r = total(read_parts(base, s)) / total(read_parts(kind, s))
            pts.append((X(s), Y(max(min(r, yhi), ylo))))
        f.poly(pts, col, 3.5, dash)
        ends.append((pts[-1][1], label, col, total(read_parts(base, hi)) / total(read_parts(kind, hi))))
    ends.sort()
    used = []
    for yy, label, col, r in ends:
        y = yy
        for u in used:
            if abs(y - u) < 30:
                y = u + 30
        used.append(y)
        f.line(x1 + 4, yy, x1 + 22, y, AXIS, 1.2)
        f.text(x1 + 28, y + 7, f"{label}  {r:.0f}×", 19, weight=700)
    f.text(X(2.2 * M), Y(9) + 40, "DSA는 약 9×에서 멈춘다", 18, "middle", 700, halo=True)
    f.text(X(2.2 * M), Y(9) + 64, "= KV 1줄 크기 ÷ 색인 키 1줄 크기", 16, "middle", color=MUTED, halo=True)
    return f


# ---------------------------------------------------------------- 그림 7. 데이터별 읽기 · 쓰기 성격
def fig_behavior():
    S = M
    kv_main = 61 * MLA_B * S
    idx = 61 * DSA_IDX * S
    rows = [
        ("가중치", "DeepSeek-V3.2 · FP8", "689 GB", (37 / 671, "약 5%", "활성 expert만 · 배치가 크면 ↑"),
         (0, "0", "추론 중 바뀌지 않음"), "HBM — 대역폭"),
        ("KV cache", "V3.2 · sparse", f"{kv_main / GB:.0f} GB", (TOPK / S, "0.2%", "고른 2,048줄만"),
         (1 / S, "≈ 0", "끝에 한 줄 추가"), "용량 계층 후보|LPDDR · CXL 등"),
        ("색인 키", "V3.2 · indexer", f"{idx / GB:.1f} GB", (1, "100%", "매번 전부 훑음"),
         (1 / S, "≈ 0", "끝에 한 줄 추가"), "HBM — 순차 스캔"),
        ("Linear state", "Qwen3.8 · DeltaNet 36층", fmt_b(36 * GDN_STATE), (1, "100%", "매번 전부"),
         (1, "100%", "매번 전부 덮어씀"), "온칩 + 쓰기 대역폭"),
    ]
    f = Fig("07-memory-behavior", 1600, 640)
    basis(f, "요청 1개 · 컨텍스트 1M · 배치 1 · 디코드 1스텝 · 보유량 대비 비율 (계산값)")
    cols = [(40, "데이터"), (330, "보유량"), (520, "매 스텝 읽는 비율"), (920, "매 스텝 다시 쓰는 비율"), (1320, "어울리는 메모리")]
    for x, h in cols:
        f.text(x, 100, h, 19, weight=700, color=MUTED)
    f.line(40, 116, 1560, 116, AXIS, 1)
    bw = 300
    for i, (nm, sub, size, rd, wr, mem) in enumerate(rows):
        y = 140 + i * 118
        if i % 2 == 0:
            f.rect(30, y - 8, 1540, 108, BOX[0], rx=8)
        f.text(40, y + 36, nm, 24, weight=700)
        f.text(40, y + 66, sub, 15, color=MUTED)
        f.text(330, y + 46, size, 24, weight=700)
        for x, (frac, lab, note) in ((520, rd), (920, wr)):
            f.rect(x, y + 18, bw, 30, "#eceef4", rx=4)
            if frac > 0:
                f.rect(x, y + 18, max(bw * frac, 4), 30, SLATE, rx=4)
            f.text(x + bw + 14, y + 41, lab, 22, weight=700)
            f.text(x, y + 76, note, 15, color=MUTED)
        m1, _, m2 = mem.partition("|")
        f.text(1320, y + 46, m1, 20, weight=700)
        if m2:
            f.text(1320, y + 76, m2, 15, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 5. KV 접근 패턴 — 조각 수
def fig_access():
    S = M
    f = Fig("06-access-pattern", 1600, 540)
    basis(f, "층 1개 · 컨텍스트 1M · 디코드 1스텝에 읽는 KV 위치 (개략도, 조각 위치는 예시 · 계산값)")
    x0, x1 = 330, 1090
    cols = [(1140, "조각 수"), (1290, "조각 1개 크기"), (1450, "읽는 양")]
    for x, h in cols:
        f.text(x, 100, h, 19, weight=700, color=MUTED)
    f.text(x0, 100, "KV cache (왼쪽 = 오래된 토큰, 오른쪽 = 최근 토큰)", 19, weight=700, color=MUTED)
    f.line(40, 116, 1560, 116, AXIS, 1)
    rnd = random.Random(11)
    n = 190
    cw = (x1 - x0) / n
    dsa = set(rnd.sample(range(n), 26))
    qsa = set()
    for st in rnd.sample(range(0, n - 4, 6), 9):
        qsa.update(range(st, st + 4))
    rows = [
        ("DSA", "토큰 단위로 선택", dsa, SEL, "2,048", "1.2 KB", fmt_b(MLA_B * TOPK)),
        ("QSA", "4줄 블록 단위로 선택", qsa, SEL, "512", fmt_b(4 * GQA_B), fmt_b(GQA_B * TOPK)),
        ("HCA", "압축 KV 전체를 연속으로", set(range(n)), CMP, "1 (연속)", fmt_b(total(read_parts("hca", S))),
         fmt_b(total(read_parts("hca", S)))),
    ]
    for i, (nm, sub, hit, col, pieces, size, amount) in enumerate(rows):
        y = 140 + i * 130
        if i % 2 == 0:
            f.rect(30, y - 6, 1540, 118, BOX[0], rx=8)
        f.text(40, y + 46, nm, 24, weight=700)
        f.text(40, y + 76, sub, 16, color=MUTED)
        for k in range(n):
            on = k in hit
            f.rect(x0 + k * cw, y + (24 if on else 34), max(cw - 1, 1), 40 if on else 20, col if on else FULL_L, rx=1)
        f.text(1140, y + 60, pieces, 24, weight=700)
        f.text(1290, y + 60, size, 22, weight=700)
        f.text(1450, y + 60, amount, 22)
    return f


# ---------------------------------------------------------------- 그림 8. 동시 1M 요청 수 — 용량 한계 vs 대역폭 한계
NODE_HBM_GB, NODE_HBM_BW = 8 * 141, 8 * 4.8e3        # H200 × 8: GB, GB/s
WEIGHTS_GB = 689                                     # DeepSeek-V3.2 공식 체크포인트 (FP8)
TPS = 20                                             # 요청당 목표 생성 속도 (토큰/s)
TIER_BW = 8 * 64                                     # 대용량 계층: GPU당 64 GB/s 링크 1개 가정 (GB/s)


def concurrency():
    S = M
    kv = 61 * MLA_B * S / GB                         # 요청당 KV cache (GB, BF16)
    idx = 61 * DSA_IDX * S / GB                      # 요청당 색인 키 (GB)
    kv_sel = 61 * MLA_B * TOPK / GB                  # 스텝당 선택된 KV 읽기 (GB)
    hbm_free = NODE_HBM_GB - WEIGHTS_GB
    step_bw = NODE_HBM_BW / TPS - WEIGHTS_GB         # 스텝당 남는 HBM 읽기 예산 (GB), 가중치 전체를 매 스텝 읽는다고 보수적으로 가정
    tier_step = TIER_BW / TPS
    sc = {
        "full_hbm": dict(cap=hbm_free / kv, bw=step_bw / kv),
        "dsa_hbm": dict(cap=hbm_free / (kv + idx), bw=step_bw / (idx + kv_sel)),
        "full_tier": dict(cap=float("inf"), bw=step_bw / 1e-9, tier=tier_step / kv),
        "dsa_tier": dict(cap=hbm_free / idx, bw=step_bw / idx, tier=tier_step / kv_sel),
    }
    for v in sc.values():
        v["n"] = min(v["cap"], v["bw"], v.get("tier", float("inf")))
    info = dict(kv=kv, idx=idx, kv_sel=kv_sel, tier_need_full=kv * TPS, tier_need_dsa=kv_sel * TPS,
                tier_cap=sc["dsa_tier"]["n"] * kv)
    return sc, info


def fig_concurrency():
    sc, info = concurrency()
    f = Fig("08-capacity-vs-bandwidth", 1600, 690)
    basis(f, f"H200 × 8 (HBM {NODE_HBM_GB:,} GB · {NODE_HBM_BW / 1e3:.1f} TB/s) · DeepSeek-V3/V3.2 · 요청당 1M · {TPS} 토큰/s (계산값)")
    # 왼쪽 — 동시 처리 가능한 1M 요청 수
    x0, xw, vmax = 340, 560, 60
    X = lambda v: x0 + min(v, vmax) / vmax * xw
    f.text(40, 104, "노드 1대가 동시에 처리할 수 있는 1M 요청 수", 23, weight=700)
    for v in (0, 20, 40, 60):
        f.line(X(v), 130, X(v), 600, GRID, 1)
        f.text(X(v), 625, f"{v}개", 15, "middle", color=MUTED)
    rows = [
        ("HBM만 사용", None),
        ("Full attention (V3)", ("full_hbm", FULL)),
        ("Sparse DSA (V3.2)", ("dsa_hbm", SEL)),
        ("HBM + 대용량 계층 (KV cache를 내림)", None),
        ("Full attention (V3)", ("full_tier", FULL)),
        ("Sparse DSA (V3.2)", ("dsa_tier", SEL)),
    ]
    y = 136
    for label, spec in rows:
        if spec is None:
            f.text(40, y + 22, label, 17, weight=700, color=MUTED)
            y += 36
            continue
        key, col = spec
        d = sc[key]
        n = d["n"]
        f.text(40, y + 36, label, 20, weight=700)
        f.rect(x0, y + 12, max(X(n) - x0, 4), 34, col, rx=4)
        if key == "full_tier":
            lab, sub = "1개 미만", f"대용량 계층 대역폭 부족 — 1개당 {info['tier_need_full'] / 1e3:.1f} TB/s 필요"
        elif key == "dsa_tier":
            lab, sub = f"{math.floor(n)}개", f"HBM의 색인 키 용량이 한계 · 계층 대역폭으로는 {math.floor(d['tier'])}개까지"
        else:
            lab, sub = f"{math.floor(n)}개", f"용량이 먼저 참 · HBM 대역폭으로는 {math.floor(d['bw'])}개까지"
        f.text(X(n) + 14, y + 38, lab, 22, weight=700)
        f.text(x0, y + 70, sub, 15, color=MUTED)
        y += 92
    # 오른쪽 — 대용량 계층에 필요한 대역폭
    rx0, rw, rmax = 1250, 300, 1500
    R = lambda v: rx0 + v / rmax * rw
    f.line(1000, 90, 1000, 660, GRID, 2, "6 6")
    f.text(1040, 104, "대용량 계층에 필요한 대역폭", 23, weight=700)
    f.text(1040, 132, f"KV cache를 내렸을 때 · 요청 1개 · {TPS} 토큰/s", 16, color=MUTED)
    for v in (0, 750, 1500):
        f.line(R(v), 170, R(v), 360, GRID, 1)
        f.text(R(v), 385, f"{v / 1e3:.2f} TB/s".replace(".00", "").replace(".50", ".5") if v else "0", 15, "middle", color=MUTED)
    for i, (nm, val, col) in enumerate((("Full (V3)", info["tier_need_full"], FULL),
                                        ("DSA (V3.2)", info["tier_need_dsa"], SEL))):
        yy = 210 + i * 90
        f.text(1040, yy + 27, nm, 19, weight=700)
        f.rect(rx0, yy, max(R(val) - rx0, 4), 38, col, rx=4)
        if val >= 100:
            f.text(R(val), yy - 10, f"{val / 1e3:.2f} TB/s", 20, "end", 700)
        else:
            f.text(R(64) + 12, yy + 27, f"{val:.1f} GB/s", 20, weight=700)
    cx = R(64)
    f.line(cx, 175, cx, 360, INK, 2, "5 4")
    f.text(cx - 6, 172, "링크 1개 ≈ 64 GB/s", 15, "end", color=INK)
    ratio = info["tier_need_full"] / info["tier_need_dsa"]
    f.text(1040, 460, f"선택 읽기로 약 {ratio:.0f}배 감소", 22, weight=700)
    f.text(1040, 492, "→ 느린 대용량 메모리로도 KV cache 보관 가능", 17, color=MUTED)
    f.text(1040, 545, f"단, 대용량 계층 용량은 {info['tier_cap'] / 1e3:.1f} TB 필요", 17, weight=700)
    f.text(1040, 573, f"({math.floor(sc['dsa_tier']['n'])}개 × 요청당 KV cache {info['kv']:.0f} GB)", 16, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 9. Sparse attention 단계별 특성과 관련 시스템 요소
def fig_implications():
    S = M
    idx_b, kv_b = read_parts("dsa", S)
    f = Fig("09-sparse-stage-summary", 1600, 730)
    basis(f, "Sparse attention(DSA) 디코드 1스텝 · 층 1개 · 컨텍스트 1M (계산값)")
    cols = [(40, "단계"), (300, "하는 일"), (620, "메모리 특성"), (960, "병목"), (1150, "관련 HW · 시스템 요소")]
    for x, h in cols:
        f.text(x, 100, h, 19, weight=700, color=MUTED)
    f.line(40, 116, 1560, 116, AXIS, 1)
    share = idx_b / (idx_b + kv_b)
    rows = [
        ("1", "색인 키 훑기", "색인 키 100만 줄과 점수 계산", f"{fmt_b(idx_b)} 순차 읽기", share, "메모리 대역폭", True,
         ("HBM 순차 대역폭", "저정밀(FP8 · FP4) 내적 연산기 · 점수 버퍼 (1M개 ≈ 4 MB)")),
        ("2", "top-k 선택", "점수 100만 개 중 2,048개 선택", "비교 · 정렬 연산 (FLOPs에 안 잡힘)", None, "비교 처리", True,
         ("비교 · 정렬 처리 (행렬 연산 외)", "온칩 SRAM · 스레드 간 reduction · 동기화")),
        ("3", "KV gather", "선택된 KV 2,048줄 읽기", f"{fmt_b(kv_b)} · 2,048조각 분산 읽기", 1 - share, "실효 대역폭 저하", True,
         ("메모리 동시 요청 수 · 접근 단위", "DMA 주소 생성 · paged KV 주소 변환")),
        ("4", "attention", "선택된 2,048줄로 계산", "dense 행렬 연산 · 2,048줄 고정", None, "부담 작음", False,
         ("텐서 코어 (dense 행렬 연산)", "온칩 버퍼 — 선택된 KV 2.4 MB 타일 처리")),
    ]
    y = 132
    for num, step, does, mem, frac, neck, hot, (rel, sub) in rows:
        f.rect(30, y, 1540, 104, BOX[0] if int(num) % 2 else "#ffffff", rx=8)
        f.rect(44, y + 30, 40, 40, SEL if hot else FULL, rx=8)
        f.text(64, y + 58, num, 20, "middle", 700, "#ffffff")
        f.text(100, y + 58, step, 22, weight=700)
        f.text(300, y + 58, does, 18)
        f.text(620, y + 44 if frac is not None else y + 58, mem, 18, weight=700 if frac is not None else 400)
        if frac is not None:
            f.rect(620, y + 60, 200, 16, "#e3e6ee", rx=3)
            f.rect(620, y + 60, max(200 * frac, 4), 16, SEL, rx=3)
            f.text(832, y + 74, f"읽기의 {frac * 100:.0f}%", 16, color=MUTED)
        f.text(960, y + 58, neck, 19, weight=700 if hot else 400, color=INK if hot else MUTED)
        f.text(1150, y + 46, rel, 18, weight=700 if hot else 400, color=INK if hot else MUTED)
        f.text(1150, y + 76, sub, 15, color=MUTED)
        y += 112
    # 보관 — 단계가 아니라 상시 부담
    y += 14
    f.line(40, y - 8, 1560, y - 8, AXIS, 1.2, "6 5")
    f.rect(30, y + 4, 1540, 104, BOX[0], rx=8)
    f.rect(44, y + 34, 40, 40, SEL, rx=8)
    f.text(64, y + 61, "+", 22, "middle", 700, "#ffffff")
    f.text(100, y + 62, "KV cache 보관", 22, weight=700)
    f.text(300, y + 62, "선택과 무관하게 전체 KV 유지", 18)
    kv_all = 61 * MLA_B * S
    f.text(620, y + 48, f"{kv_all / GB:.0f} GB (61층) 중 매 스텝 0.2%만 읽음", 18, weight=700)
    f.rect(620, y + 64, 200, 16, "#e3e6ee", rx=3)
    f.rect(620, y + 64, 4, 16, SEL, rx=3)
    f.text(832, y + 78, "읽기 0.2%", 16, color=MUTED)
    f.text(960, y + 62, "용량", 19, weight=700)
    f.text(1150, y + 50, "HBM 용량 · 메모리 계층", 18, weight=700)
    f.text(1150, y + 80, "LPDDR · CXL 등 대용량 계층 · 계층 간 대역폭 · KV 할당 관리", 15, color=MUTED)
    return f


FIGS = [fig_trend, fig_moe, fig_axes, fig_layers, fig_read_store, fig_gain, fig_access, fig_behavior, fig_concurrency, fig_implications]
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


def print_summary():
    S = M
    print("층당 · 1M · 1스텝 KV cache 읽기 / 크기")
    for k in ("mla", "dsa", "csa", "hca", "gqa", "qsa", "gdn"):
        r, s = read_parts(k, S), store_parts(k, S)
        print(f"  {k:4s} read idx {fmt_b(r[0]):>9s} main {fmt_b(r[1]):>9s} total {fmt_b(sum(r)):>9s} | "
              f"store {fmt_b(sum(s)):>9s} | idx share {r[0] / sum(r) * 100:5.1f}%")
    for kind, base, label, _, _ in CURVES:
        print("  ", label, " ".join(f"{s}:{total(read_parts(base, s)) / total(read_parts(kind, s)):.1f}"
                                   for s in (1024, 8192, 131072, M, 10 * M)))


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
