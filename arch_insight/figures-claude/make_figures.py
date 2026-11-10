"""Sparse · Hybrid attention 보고 자료(02-deck-draft-claude.md)용 계산 · 그림 생성기.
실행: python arch_insight/figures-claude/make_figures.py [그림 이름 ...]  → 같은 폴더에 *.svg, *.png
그림 양식은 reports/llm-memory-requirements/figures/make_figures.py와 같다 (SVG를 직접 그리고 headless 브라우저로 PNG).

수치 기준
- 층 구성 · head 수 · 차원 · 지원 길이 : 공식 config.json (reports/llm-memory-requirements/data/raw/)
- 1M = 1,000,000 토큰, 1 GB = 10^9 B. 그림 4만 메모리 보고서 기준(1M = 1,048,576)을 그대로 쓴다
- 구조 비교(그림 2 · 6 ~ 13): KV cache는 BF16으로 통일, 색인 키는 공개 저장 형식
  DSA 색인 키 = FP8 128 B + FP32 scale 4 B = 132 B · CSA 색인 키 = FP4 64 B · QSA 블록 키 = 정밀도 표기가 없어 BF16 256 B로 가정
- 배포 형식(그림 14 · 15): DeepSeek-V3.2 실제 형식 — 가중치 FP8 689 GB, KV FP8 656 B/줄(FlashMLA), 색인 132 B/줄
- 지원 길이: V3 · V3.2 163,840 · Qwen3.8-Flash-Next 262,144 · V4-Pro 1,048,576. 지원 길이 밖은 점선 · * (외삽)
- 읽기량은 KV를 한 번씩만 읽는다고 본 하한이다. 시간은 H200 사양(HBM 4.8 TB/s, FP8 1,979 · BF16 989 TFLOPS dense)으로 나눈 이론값
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
OTHER = "#c3c7d2"                        # 기타
BOX = ("#f6f7fb", "#dfe2ec")

# 방식별 색 — 덱 전체에서 고정 (진한 색, 옅은 색)
FULL, FULL_L = "#9c9a92", "#dddcd6"      # 전부 읽기
SEL, SEL_L = "#2a78d6", "#b7d3f6"        # 선택 읽기 (DSA · QSA)
NAVY = "#184f95"                         # 선택 읽기의 둘째 계열 (QSA)
CMP, CMP_L = "#eb6834", "#f8c6b1"        # 압축 저장 (CSA · HCA)
RUST = "#9c3d14"                         # 압축의 둘째 계열 (HCA)
STATE, STATE_L = "#1baf7a", "#a9e0cb"    # linear state
WIN, WIN_L = "#eda100", "#f7d98f"        # sliding window

GB, MB = 1e9, 1e6
M = 1_000_000
K128 = 131072

SUPPORT = {"mla": 163840, "dsa": 163840, "gqa": 262144, "qsa": 262144, "csa": 1048576, "hca": 1048576}

# H200 사양 (dense)
HBM_BW = 4.8e12
FP8_PK = 1979e12
BF16_PK = 989e12

# ---------------------------------------------------------------- 구조 비용 모델 (층 1개 · 요청 1개 · decode 1스텝)
MLA_B = 576 * 2            # DeepSeek-V3 / V3.2 latent 512 + RoPE 64, BF16
DSA_IDX = 132              # V3.2 색인 키 128차원 FP8 + FP32 scale
TOPK = 2048
GQA_B = 2 * 2 * 256 * 2    # Qwen3.8-Flash-Next full 층: KV head 2 × 256 × (K, V) × BF16
QSA_IDX = 128 * 2          # 블록 키 128차원, BF16 가정
QSA_BLOCK = 4
V4_B = 512 * 2             # V4 엔트리 512차원 (K = V), BF16
CSA_IDX = 64               # V4 CSA 색인 키 128차원 FP4
CSA_K = 1024
RECENT = 128               # V4 최근 원본 토큰
GDN_STATE = 48 * 128 * 128 * 4 + (2 * 16 * 128 + 48 * 128) * 3 * 2   # Qwen3.8-Flash-Next DeltaNet 층 하나: FP32 state + conv ≈ 3.2 MB
SCORE_B = 4                # top-k 점수 FP32 (점수를 HBM에 기록하는 구현 기준)

DSA_FLOP = 64 * 128 * 2    # 색인 키 1개당 연산: indexer head 64 × 128차원 × 2
CSA_FLOP = 64 * 128 * 2
QSA_FLOP = 4 * 128 * 2
MLA_ATT_FLOP = 128 * (576 + 512) * 2   # 선택된 KV 1줄당 attention 연산 (head 128, 흡수형 MLA decode)


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


def ratio_vs(kind, base, S):
    return total(read_parts(base, S)) / total(read_parts(kind, S))


def dsa_model_io(S):
    """DeepSeek-V3.2 모델 전체(61층) 디코드 1스텝 — (선택된 KV 읽기, 기타, 색인 키 읽기) 바이트.
    기타 = top-k 점수 FP32 기록 + 읽기, KV 1줄 + 색인 키 1개 추가 쓰기."""
    kv = 61 * MLA_B * min(S, TOPK)
    other = 61 * (2 * SCORE_B * S + MLA_B + DSA_IDX)
    idx = 61 * DSA_IDX * S
    return kv, other, idx


def hybrid_parts(S):
    st = 36 * GDN_STATE
    qi, qk = read_parts("qsa", S)
    return st, st, 12 * qk, 12 * qi            # state 읽기, state 쓰기, QSA KV 읽기, QSA 색인 읽기


def qwen_config(kind, S):
    """Qwen3.8-Flash-Next 차원(48층)으로 본 구성별 (KV, 색인 키, state) — (읽기 + 쓰기, 크기) 바이트.
    kind: full = 전 층 GQA, sparse = 전 층 QSA, hybrid = GQA 12층 + state 36층, both = QSA 12층 + state 36층(실제)."""
    n_att = 48 if kind in ("full", "sparse") else 12
    n_st = 0 if kind in ("full", "sparse") else 36
    a = "gqa" if kind in ("full", "hybrid") else "qsa"
    ri, rk = read_parts(a, S)
    si, sk = store_parts(a, S)
    io = (n_att * rk, n_att * ri, n_st * 2 * GDN_STATE)
    size = (n_att * sk, n_att * si, n_st * GDN_STATE)
    return io, size


def weight_read(n):
    """디코드 1스텝에 읽는 가중치 (GB). 균등 routing일 때 배치 n의 expert 합집합 (V3.2: 비 expert 17.1B + 58층 × expert 44 MB)."""
    union = 256 * (1 - (1 - 8 / 256) ** max(n, 1))
    return 17.1 + 58 * union * 0.04404


# ---------------------------------------------------------------- SVG 도구
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Fig:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.o = name, w, h, []

    def raw(self, s):
        self.o.append(s)

    def rect(self, x, y, w, h, fill, stroke="none", rx=4, sw=1.5, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' fill-opacity="{opacity}"' if opacity is not None else ""
        self.raw(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{op}/>')

    def text(self, x, y, s, size=20, anchor="start", weight=400, color=INK, halo=False):
        h = ' stroke="#ffffff" stroke-width="7" stroke-linejoin="round" paint-order="stroke"' if halo else ""
        self.raw(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" text-anchor="{anchor}" '
                 f'font-weight="{weight}" fill="{color}"{h}>{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, color=AXIS, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                 f'stroke-width="{sw}" stroke-linecap="round"{d}/>')

    def poly(self, pts, color, sw=3, dash=None):
        if len(pts) < 2:
            return
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.raw(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" '
                 f'stroke-linejoin="round" stroke-linecap="round"{d}/>')

    def area(self, pts, fill):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.raw(f'<polygon points="{p}" fill="{fill}" stroke="none"/>')

    def dot(self, x, y, r, fill, stroke="#ffffff"):
        self.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

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
    if v >= 100e9:
        return f"{v / 1e9:.0f} GB"
    if v >= 1e9:
        return f"{v / 1e9:.2f} GB" if v < 10e9 else f"{v / 1e9:.1f} GB"
    if v >= 10e6:
        return f"{v / 1e6:.0f} MB"
    if v >= 1e6:
        return f"{v / 1e6:.1f} MB"
    if v >= 1e3:
        return f"{v / 1e3:.0f} KB"
    return f"{v:.0f} B"


def pct(x):
    if x >= 0.995:
        return "100%"
    if x >= 0.095:
        return f"{x * 100:.0f}%"
    if x >= 0.01:
        return f"{x * 100:.1f}%"
    return "<0.01%" if x < 1e-4 else f"{x * 100:.2f}%"


def basis(f, s, x=40, y=40):
    f.text(x, y, s, 19, color=MUTED)


def chip(f, x, y, w, h, label, dark, light, size=19, color=INK):
    """방식 이름표 — 옅은 바탕 + 왼쪽 진한 띠."""
    f.rect(x, y, w, h, light, rx=6)
    f.rect(x, y, 8, h, dark, rx=3)
    f.text(x + 20, y + h / 2 + size * 0.36, label, size, weight=700, color=color)


def split_by_support(kind, pts_s, X, Y, fn):
    """지원 길이 안(실선) · 밖(점선) 두 구간의 점 목록."""
    lim = SUPPORT[kind]
    a = [(X(s), Y(fn(s))) for s in pts_s if s <= lim]
    b = [(X(s), Y(fn(s))) for s in pts_s if s >= lim * 0.999] or []
    if a and b and lim < pts_s[-1]:
        b = [(X(lim), Y(fn(lim)))] + b
    return a, b


def logspace(lo, hi, n):
    return [int(lo * (hi / lo) ** (i / (n - 1))) for i in range(n)]


# ---------------------------------------------------------------- 그림 1. 반기별 attention 구성 (모델 수)
CAT = [("full", "Full attention만", FULL, FULL_L), ("select", "선택 읽기 (+ full)", SEL, SEL_L),
       ("compress", "압축 저장 (+ full)", CMP, CMP_L), ("state", "Linear state + attention", STATE, STATE_L),
       ("swa", "Sliding window + full", WIN, WIN_L)]
HALVES = ["2024 H2", "2025 H1", "2025 H2", "2026 H1", "2026 H2"]


def _half(d):
    return f"{d[:4]} H{1 if int(d[5:7]) <= 6 else 2}"


def _cat(r):
    if r["category"] == "sparse":
        return "compress" if "DeepSeek-V4" in r["model"] else "select"
    return r["category"]


def fig_trend():
    rows = list(csv.DictReader(open(LINEAGE, encoding="utf-8-sig")))
    cnt = {h: {c: 0 for c, _, _, _ in CAT} for h in HALVES}
    for r in rows:
        cnt[_half(r["date"])][_cat(r)] += 1
    f = Fig("01-trend", 1500, 700)
    basis(f, f"조사 대상: 주요 9개 시리즈 {len(rows)}개 공개 모델 · 반기별 공개 모델 수 · 공식 config 기준")
    x0, x1, top, bot = 110, 1000, 130, 600
    vmax = 20
    Y = lambda v: bot - v / vmax * (bot - top)
    for v in (0, 5, 10, 15, 20):
        f.line(x0, Y(v), x1, Y(v), GRID, 1)
        f.text(x0 - 12, Y(v) + 6, f"{v}개", 16, "end", color=MUTED)
    slot = (x1 - x0) / len(HALVES)
    bw = 112
    last_mid = {}
    for i, h in enumerate(HALVES):
        n = sum(cnt[h].values())
        bx = x0 + slot * i + (slot - bw) / 2
        acc = 0
        for c, _, col, _ in CAT:
            k = cnt[h][c]
            if not k:
                continue
            y_hi, y_lo = Y(acc + k), Y(acc)
            f.rect(bx, y_hi + 1, bw, y_lo - y_hi - 2, col, rx=3)
            if y_lo - y_hi >= 26:
                f.text(bx + bw / 2, (y_hi + y_lo) / 2 + 7, str(k), 18, "middle", 700, INK if col == WIN else "#ffffff")
            if i == len(HALVES) - 1:
                last_mid[c] = (y_hi + y_lo) / 2
            acc += k
        f.text(bx + bw / 2, Y(n) - 34, f"전 층 full {cnt[h]['full']}/{n}", 18, "middle", 700)
        f.text(bx + bw / 2, Y(n) - 12, f"{n}개", 15, "middle", color=MUTED)
        f.text(bx + bw / 2, bot + 32, f"'{h[2:4]} {'상반기' if h.endswith('H1') else '하반기'}", 19, "middle", 700)
    f.line(x0, bot, x1, bot, AXIS, 1.5)
    lx = x1 + 30
    bx_end = x0 + slot * (len(HALVES) - 1) + (slot + bw) / 2
    used = []
    for c, label, col, _ in CAT:
        if c not in last_mid:
            continue
        y = last_mid[c]
        for u in used:
            if abs(y - u) < 30:
                y = u - 30 if y < u else u + 30
        used.append(y)
        f.line(bx_end + 4, last_mid[c], lx - 6, y, AXIS, 1.2)
        f.rect(lx, y - 9, 16, 16, col, rx=3)
        f.text(lx + 24, y + 6, label, 18)
    return f


# ---------------------------------------------------------------- 그림 2. 방식별 KV cache 쓰기 · 읽기
METHODS = [("전부 읽기", "Full · MLA · GQA", FULL, FULL_L),
           ("선택 읽기", "DSA · QSA", SEL, SEL_L),
           ("압축 저장", "CSA · HCA (DeepSeek-V4)", CMP, CMP_L),
           ("Sliding window", "gpt-oss · MiMo · Muse", WIN, WIN_L),
           ("Linear state", "DeltaNet · Mamba · KDA", STATE, STATE_L)]


def _cells(f, x, y, w, rh, fills):
    for k, c in enumerate(fills):
        yy = y + k * rh
        if c is None:
            f.rect(x, yy + 1, w, rh - 3, "none", AXIS, 2, 1.2, "4 3")
        else:
            f.rect(x, yy + 1, w, rh - 3, c, rx=2)


def _write_row(f, x, y, w, rh, fill):
    f.rect(x, y + 1, w, rh - 3, fill, INK, 2, 2, "5 3")


def fig_write_read():
    S = K128
    f = Fig("02-kv-write-read", 1700, 930)
    basis(f, "층 1개 · 요청 1개 · 디코드 1스텝 · 한 줄 = 토큰 하나의 KV · 수치는 컨텍스트 128K 기준 (그림 비율은 실제와 다름)")
    X0, CW, GAP = 214, 288, 8
    cols = [X0 + i * (CW + GAP) for i in range(5)]
    for i, (nm, ex, dark, light) in enumerate(METHODS):
        chip(f, cols[i], 80, CW, 46, nm, dark, light, 22)
        f.text(cols[i] + 20, 152, ex, 15, color=MUTED)
    top, n, rh = 190, 12, 20
    tw = 108
    wy = top + n * rh + 8                      # 이번 스텝에 쓰는 줄
    # 전부 읽기
    cx = cols[0] + 24
    _cells(f, cx, top, tw, rh, [FULL] * n)
    _write_row(f, cx, wy, tw, rh, FULL)
    f.text(cx + tw + 16, top + 110, "S줄 전부", 16, weight=700)
    f.text(cx + tw + 16, top + 130, "매번 읽기", 14, color=MUTED)
    # 선택 읽기
    cx = cols[1] + 24
    pick = {1, 4, 8, 10}
    _cells(f, cx, top, 92, rh, [SEL if k in pick else FULL_L for k in range(n)])
    _cells(f, cx + 100, top, 16, rh, [SEL_L] * n)
    _write_row(f, cx, wy, 92, rh, FULL_L)
    _write_row(f, cx + 100, wy, 16, rh, SEL_L)
    f.text(cx + 128, top + 18, "색인 키", 16, weight=700)
    f.text(cx + 128, top + 38, "전부 읽기", 14, color=MUTED)
    f.text(cx + 128, top + 168, "고른 k줄만", 16, weight=700)
    f.text(cx + 128, top + 188, "읽기", 14, color=MUTED)
    # 압축 저장
    cx = cols[2] + 24
    f.rect(cx, top, tw, n * rh, "none", AXIS, 3, 1.4, "6 5")
    _cells(f, cx, top, tw, rh, [CMP] * 3 + [CMP_L] * 2)
    _write_row(f, cx, top + 5 * rh, tw, rh, CMP_L)
    f.arrow(cx + tw / 2, top + n * rh - 10, cx + tw / 2, top + 6 * rh + 16, MUTED, 2.5)
    f.text(cx + tw + 16, top + 18, "4줄 → 1줄", 16, weight=700)
    f.text(cx + tw + 16, top + 38, "압축된 줄 읽기", 14, color=MUTED)
    f.text(cx + tw + 16, top + 88, "최근 토큰", 16, weight=700)
    f.text(cx + tw + 16, top + 108, "모이면 압축", 14, color=MUTED)
    f.text(cx + tw + 16, top + 200, "점선 = 원래 크기", 14, color=MUTED)
    # Sliding window
    cx = cols[3] + 24
    _cells(f, cx, top, tw, rh, [None] * 8 + [WIN] * 4)
    _write_row(f, cx, wy, tw, rh, WIN)
    f.text(cx + tw + 16, top + 70, "W 밖은", 16, weight=700, color=MUTED)
    f.text(cx + tw + 16, top + 90, "버림", 14, color=MUTED)
    f.text(cx + tw + 16, top + 190, "최근 W줄만", 16, weight=700)
    f.text(cx + tw + 16, top + 210, "읽기", 14, color=MUTED)
    # Linear state
    cx = cols[4] + 24
    cs, cg = 21, 3
    for r in range(6):
        for c in range(6):
            f.rect(cx + c * (cs + cg), top + 6 + r * (cs + cg), cs, cs, STATE, rx=2)
    g = 6 * (cs + cg)
    f.rect(cx - 4, top + 2, g + 5, g + 5, "none", INK, 4, 2, "5 3")
    by = top + g + 70
    f.rect(cx, by, g - 3, 40, STATE_L, rx=6)
    f.text(cx + g / 2 - 2, by + 26, "새 토큰으로 갱신", 15, "middle", 700)
    f.arrow(cx + 30, top + g + 10, cx + 30, by - 6, MUTED, 2)
    f.arrow(cx + g - 34, by - 6, cx + g - 34, top + g + 10, MUTED, 2)
    f.text(cx + g + 16, top + 40, "고정 크기", 16, weight=700)
    f.text(cx + g + 16, top + 60, "전부 읽고", 14, color=MUTED)
    f.text(cx + g + 16, top + 80, "전부 씀", 14, color=MUTED)
    # 범례
    ly = 482
    f.rect(214, ly, 18, 18, FULL_L, rx=3)
    f.text(240, ly + 15, "보관만 (이번 스텝에 안 읽음)", 16, color=MUTED)
    for k, c in enumerate((FULL, SEL, CMP, WIN, STATE)):
        f.rect(540 + k * 22, ly, 18, 18, c, rx=3)
    f.text(540 + 5 * 22 + 6, ly + 15, "이번 스텝에 읽음", 16, color=MUTED)
    f.rect(820, ly, 30, 18, "#ffffff", INK, 3, 2, "5 3")
    f.text(858, ly + 15, "이번 스텝에 씀", 16, color=MUTED)
    # 표
    r_dsa, r_csa, r_hca = ratio_vs("dsa", "mla", S), ratio_vs("csa", "mla", S), ratio_vs("hca", "mla", S)
    lines_state = round(GDN_STATE / GQA_B, -2)
    rows = [
        ("보유 (128K)", [("128K줄", "길이에 비례", None), ("128K줄 + 색인 키", "크기는 Full과 같음 (+11%)", SEL_L),
                        ("32K줄 · 1K줄", "4:1 · 128:1 압축", None), ("W줄 고정", "W = 128 ~ 2,048", None),
                        ("고정 크기 행렬", f"층당 3.2 MB ≈ KV 약 {lines_state:,.0f}줄", None)]),
        ("매 스텝 쓰기", [("1줄 추가", "", None), ("1줄 + 색인 키 1개", "", None),
                       ("4 · 128토큰마다 1줄", "압축 계산 후 기록", None), ("1줄 추가 · 1줄 폐기", "고리형 버퍼", None),
                       ("state 전체 덮어쓰기", f"KV 약 {lines_state:,.0f}줄 분량", STATE_L)]),
        ("매 스텝 읽기 (128K)", [("128K줄 전부", "기준", None), ("색인 키 전부 + 2,048줄", f"Full 대비 1/{r_dsa:.0f}", SEL_L),
                              ("압축된 줄 (선택 · 전부)", f"Full 대비 1/{r_csa:.0f} · 1/{r_hca:.0f}", None),
                              ("W줄", f"Full 대비 1/{S // 128:,} (W = 128)", None), ("state 전체", "길이와 무관", None)]),
        ("추가 연산 · 제약", [("—", "읽기 · 크기 모두 길이에 비례", None), ("색인 훑기 · top-k", "흩어진 KV 읽기", None),
                          ("쓰기 시 압축 연산", "원본 KV 정확 조회 불가", None), ("W 밖 토큰은 못 봄", "full 층과 혼합해 사용", None),
                          ("매 스텝 갱신 연산", "쓰기 대역폭 필요", None)]),
    ]
    y = 528
    for j, (lab, cells) in enumerate(rows):
        if j % 2 == 0:
            f.rect(30, y, 1650, 90, BOX[0], rx=8)
        f.text(40, y + 52, lab, 18, weight=700, color=MUTED)
        for i, (a, b, hl) in enumerate(cells):
            x = cols[i]
            if hl:
                f.rect(x + 4, y + 8, CW - 8, 74, hl, rx=6)
            f.text(x + 20, y + 40, a, 18, weight=700)
            if b:
                f.text(x + 20, y + 66, b, 14, color=INK if hl else MUTED)
        y += 96
    return f


# ---------------------------------------------------------------- 그림 3. 층 구성 띠
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
    f = Fig("03-layer-strips", 1600, 760)
    legend = [(FULL, "전부 읽기", "KV cache를 매번 전부"), (SEL, "선택 읽기", "색인으로 찾아 KV 일부만"),
              (CMP, "압축 저장", "KV 여러 줄을 한 줄로"), (STATE, "Linear state", "KV cache 대신 고정 크기 state"),
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


# ---------------------------------------------------------------- 그림 4. 모델 계열별 가중치 + KV cache (1M 요청 1개)
FLAGSHIPS = [
    ("full", ["DeepSeek-V3", "Hy3"]),
    ("select", ["GLM-5.3", "MiniMax-M3", "Hy4-preview"]),
    ("compress", ["DeepSeek-V4-Pro", "DeepSeek-V4.1-Flash"]),
    ("state", ["Qwen3.8-2.4T-A95B", "Kimi K3", "Nemotron 3 Ultra 550B-A55B"]),
    ("swa", ["MiMo-V2.6-Pro", "Muse Spark Open 816B-A42B"]),
]


def fig_weights_kv():
    rows = {r["model"]: r for r in csv.DictReader(open(LINEAGE, encoding="utf-8-sig"))}
    dark = {c: d for c, _, d, _ in CAT}
    light = {c: l for c, _, _, l in CAT}
    short = {"Nemotron 3 Ultra 550B-A55B": "Nemotron 3 Ultra", "Muse Spark Open 816B-A42B": "Muse Spark Open"}
    n_rows = sum(len(m) for _, m in FLAGSHIPS)
    H = 200 + n_rows * 52 + (len(FLAGSHIPS) - 1) * 12 + 60
    f = Fig("04-weights-vs-kv", 1600, H)
    basis(f, "주요 시리즈 대표 모델 · 가중치 = 공식 체크포인트 크기 · KV cache = 요청 1개, BF16 구조 기준 (메모리 보고서, 1M = 1,048,576)")
    for (c, lab, d, l), lx in zip(CAT, (40, 270, 500, 730, 1010)):
        chip(f, lx, 64, 34, 26, "", d, l)
        f.text(lx + 44, 84, lab, 16, color=MUTED)
    X1, W1, V1 = 400, 680, 3000
    f.text(40, 140, "모델", 18, weight=700, color=MUTED)
    f.text(X1, 132, "1M 요청 1개를 처리하는 데 필요한 메모리", 20, weight=700)
    f.rect(X1, 146, 14, 14, SLATE, rx=2)
    f.text(X1 + 20, 159, "가중치 (모든 요청이 공유)", 15, color=MUTED)
    f.rect(X1 + 230, 146, 14, 14, SEL, rx=2)
    f.text(X1 + 250, 159, "+ KV cache 1M (요청마다, 방식 색)", 15, color=MUTED)
    f.text(1300, 132, "KV 비중", 18, weight=700)
    f.text(1300, 156, "필요 메모리 중", 14, color=MUTED)
    f.text(1440, 132, "가중치와 같아지는", 16, weight=700)
    f.text(1440, 156, "동시 1M 요청 수", 16, weight=700)
    y_end = H - 50
    for v in range(0, V1 + 1, 500):
        f.line(X1 + v / V1 * W1, 176, X1 + v / V1 * W1, y_end, GRID, 1)
        f.text(X1 + v / V1 * W1, y_end + 24, f"{v:,} GB" if v else "0", 14, "middle", color=MUTED)
    y = 182
    for c, models in FLAGSHIPS:
        for m in models:
            r = rows[m]
            w, kv, n = float(r["weights_GB"]), float(r["kv1m_state1_GB"]), float(r["N_equal_1M"])
            chip(f, 36, y + 2, 340, 46, "", dark[c], light[c])
            f.text(56, y + 23, short.get(m, m), 17, weight=700)
            f.text(56, y + 41, f"{r['date'][:7]} · 가중치 {r['weights_format']}", 12, color=MUTED)
            ww, wk = w / V1 * W1, max(kv / V1 * W1, 2)
            f.rect(X1, y + 12, ww, 26, SLATE, rx=3)
            f.rect(X1 + ww + 2, y + 12, wk, 26, dark[c], rx=3)
            kvs = f"{kv:.0f}" if kv >= 10 else f"{kv:.1f}"
            f.text(X1 + ww + wk + 10, y + 31, f"{w:,.0f} + {kvs} = {w + kv:,.0f} GB", 15)
            share = kv / (w + kv) * 100
            f.text(1300, y + 32, f"{share:.0f}%" if share >= 1.5 else f"{share:.1f}%", 20, weight=700,
                   color=INK if share >= 10 else MUTED)
            f.text(1440, y + 32, f"{n:.1f}개" if n < 10 else f"{n:.0f}개", 18, weight=700)
            y += 52
        y += 12
    return f


# ---------------------------------------------------------------- 그림 5. MoE vs Sparse attention
def fig_moe():
    f = Fig("05-moe-analogy", 1600, 860)
    rnd = random.Random(7)
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
            ("한 번에 읽는 단위", "expert 1개 가중치 ≈ 44 MB", "KV 1줄 ≈ 1 KB"),
            ("여러 요청이 나눠 쓰나", "예 — 같은 가중치", "아니오 — 요청마다 다른 KV cache"),
            ("주된 제약", "소배치: 가중치 대역폭 · 대규모 EP: 토큰 all-to-all", "색인 훑기 · 흩어진 KV 읽기 · KV cache 용량")]
    y0 = 490
    for i, (lab, a, b) in enumerate(rows):
        y = y0 + i * 88
        f.rect(40, y, 1520, 78, BOX[0] if i % 2 == 0 else "#ffffff", "none", 8)
        f.text(60, y + 48, lab, 19, weight=700, color=MUTED)
        sz = 20 if i == 3 else 23
        f.text(560, y + 48, a, sz, "middle", 700 if i == 3 else 400)
        f.text(1210, y + 48, b, sz, "middle", 700 if i == 3 else 400)
    return f


# ---------------------------------------------------------------- 그림 6. Sparse attention 디코드 읽기 구성 (V3.2 모델 전체)
def fig_sparse_mix():
    f = Fig("06-sparse-read-composition", 1600, 740)
    basis(f, "DeepSeek-V3.2 모델 전체 (DSA 61층) · 요청 1개 · 디코드 1스텝 · KV BF16, 색인 키 FP8 + scale 132 B, top-k 점수 FP32 (계산값)")
    segs = [("선택된 KV 읽기 (2,048줄)", SEL), ("기타 (top-k 점수 기록 · 읽기, KV 추가)", OTHER), ("색인 키 읽기", SEL_L)]
    for (nm, col), lx in zip(reversed(segs), (140, 330, 720)):
        f.rect(lx, 74, 18, 18, col, rx=3)
        f.text(lx + 26, 89, nm, 16, color=MUTED)
    lens = [(8192, "8K"), (32768, "32K"), (65536, "64K"), (K128, "128K"), (163840, "160K"), (M, "1M*")]
    x0, top, bot = 140, 150, 620
    vmax = 1.6e9
    Y = lambda v: bot - v / vmax * (bot - top)
    for v in (0, 0.4e9, 0.8e9, 1.2e9, 1.6e9):
        f.line(x0, Y(v), 1110, Y(v), GRID, 1)
        f.text(x0 - 12, Y(v) + 6, f"{v / 1e9:.1f} GB" if v else "0", 15, "end", color=MUTED)
    bw, slot = 100, 160
    cap = 1.30e9
    for j, (S, lab) in enumerate(lens):
        x = x0 + 30 + j * slot
        parts = dsa_model_io(S)
        tot = sum(parts)
        broken = tot > vmax
        out = S > SUPPORT["dsa"]
        acc = 0
        for (nm, col), v in zip(segs, parts):
            hi = acc + v
            y_hi = Y(min(hi, cap)) if broken else Y(hi)
            f.rect(x, y_hi + 1, bw, Y(acc) - y_hi - 2, col, AXIS if out else "none", 3, 1.5, "5 4" if out else None)
            if nm == "색인 키 읽기" and Y(acc) - y_hi > 40:
                f.text(x + bw / 2, (Y(acc) + y_hi) / 2 + 6, f"색인 {v / tot * 100:.0f}%", 16, "middle", 700)
            acc = hi
        top_y = Y(min(tot, cap)) if broken else Y(tot)
        if broken:
            yb = Y(cap) + 2
            f.raw(f'<polygon points="{x - 4:.1f},{yb + 6:.1f} {x + bw + 4:.1f},{yb - 6:.1f} {x + bw + 4:.1f},{yb + 6:.1f} '
                  f'{x - 4:.1f},{yb + 18:.1f}" fill="#ffffff"/>')
            f.rect(x, Y(vmax) + 30, bw, Y(cap) - Y(vmax) - 30, SEL_L, AXIS, 3, 1.5, "5 4")
            top_y = Y(vmax) + 30
        f.text(x + bw / 2, top_y - 34, fmt_b(tot), 18, "middle", 700)
        full = 61 * MLA_B * S
        f.text(x + bw / 2, top_y - 12, f"Full 대비 1/{full / tot:.0f}" if full > tot else "Full보다 큼", 15, "middle", color=MUTED)
        f.text(x + bw / 2, bot + 30, lab, 18, "middle", 700, INK if S == K128 else MUTED)
        f.text(x + bw / 2, Y(parts[0] / 2) + 5, "2,048줄 고정" if j == 0 else "고정", 14, "middle", 700, "#ffffff")
    f.text(x0 + 30 + 4 * slot + bw / 2, bot + 54, "V3.2 지원 길이", 14, "middle", color=MUTED)
    f.text(x0, 690, "* 지원 길이(160K) 밖 — 외삽 · 막대 위 = 디코드 1스텝 합계, Full(MLA 61층) 대비", 14, color=MUTED)
    # 오른쪽 상자 — 길이 1토큰당 늘어나는 읽기
    bx = 1160
    f.rect(bx - 20, 150, 440, 470, BOX[0], BOX[1], 10, 1.2)
    f.text(bx, 192, "컨텍스트가 1토큰 늘 때", 19, weight=700)
    f.text(bx, 218, "디코드 1스텝 읽기 증가분 (61층)", 15, color=MUTED)
    items = [("색인 키", f"132 B × 61 = {132 * 61 / 1e3:.1f} KB", SEL_L, True),
             ("top-k 점수", f"8 B × 61 = {8 * 61 / 1e3:.1f} KB", OTHER, False),
             ("선택된 KV", "0 (2,048줄 고정)", SEL, False)]
    yy = 262
    for nm, v, col, strong in items:
        f.rect(bx, yy - 14, 16, 16, col, rx=3)
        f.text(bx + 26, yy, nm, 17, weight=700)
        f.text(bx + 150, yy, v, 17, weight=700 if strong else 400)
        yy += 44
    f.line(bx, yy - 12, bx + 400, yy - 12, BOX[1], 1.5)
    f.text(bx, yy + 20, "Full(MLA)이라면", 16, color=MUTED)
    f.text(bx, yy + 50, f"1,152 B × 61 = {1152 * 61 / 1e3:.0f} KB", 20, weight=700)
    f.text(bx, yy + 92, "→ 길이에 비례해 늘어나는 것은", 16, weight=700)
    f.text(bx, yy + 118, "   KV cache가 아니라 색인 키", 16, weight=700)
    return f


# ---------------------------------------------------------------- 그림 7. KV cache 읽기 vs 크기 (층당, 128K)
FAMILIES = [
    ("DeepSeek", [("전부 읽기", "V3 · MLA", "mla", FULL, FULL_L), ("선택 읽기", "V3.2 · DSA", "dsa", SEL, SEL_L),
                  ("압축 4줄 → 1", "V4 · CSA", "csa", CMP, CMP_L), ("압축 128줄 → 1", "V4 · HCA", "hca", CMP, CMP_L)], "mla"),
    ("Qwen3.8", [("전부 읽기", "full 층 · GQA", "gqa", FULL, FULL_L),
                 ("선택 읽기", "full 층 · QSA", "qsa", SEL, SEL_L),
                 ("Linear state", "linear 층 · DeltaNet", "gdn", STATE, STATE_L)], "gqa"),
]


def fig_read_store():
    S = K128
    f = Fig("07-kv-read-vs-size", 1600, 740)
    basis(f, "층 1개 · 요청 1개 · 컨텍스트 128K · KV cache BF16 통일, 색인 키는 공개 저장 형식 (계산값) · 비율 = 같은 회사의 전부 읽기 대비")
    X1, X2, pw = 450, 1040, 360
    vmax = 0.3e9
    sc = pw / vmax
    f.text(40, 112, "방식", 18, weight=700, color=MUTED)
    f.text(X1, 104, "디코드 1스텝의 KV cache 읽기", 22, weight=700)
    f.text(X2, 104, "KV cache 크기", 22, weight=700)
    f.rect(X1, 118, 16, 16, SEL_L, rx=3)
    f.text(X1 + 22, 132, "옅은 색 = 색인 키", 15, color=MUTED)
    f.rect(X1 + 170, 118, 16, 16, SEL, rx=3)
    f.text(X1 + 192, 132, "진한 색 = KV cache · state", 15, color=MUTED)
    for X in (X1, X2):
        for v in (0, 0.1e9, 0.2e9, 0.3e9):
            f.line(X + v * sc, 150, X + v * sc, 690, GRID, 1)
            f.text(X + v * sc, 714, f"{v / 1e6:.0f} MB" if v else "0", 14, "middle", color=MUTED)
    y = 150
    for fam, items, base in FAMILIES:
        f.text(40, y + 22, fam, 18, weight=700)
        f.line(40 + len(fam) * 13 + 20, y + 16, X1 - 20, y + 16, BOX[1], 1.5)
        y += 34
        base_read = total(read_parts(base, S))
        base_store = total(store_parts(base, S))
        for label, model, kind, dark, light in items:
            chip(f, 40, y + 8, 200, 42, label, dark, light, 18)
            f.text(256, y + 35, model, 15, color=MUTED)
            for X, parts, ref in ((X1, read_parts(kind, S), base_read), (X2, store_parts(kind, S), base_store)):
                idx, main = parts
                w_idx, w_main = idx * sc, max(main * sc, 3)
                if idx:
                    f.rect(X, y + 12, max(w_idx, 2), 34, light, rx=3)
                f.rect(X + w_idx + (2 if idx else 0), y + 12, w_main, 34, dark, rx=3)
                tot = idx + main
                ex = X + w_idx + w_main + 12
                if kind == "gdn":
                    f.text(ex, y + 36, fmt_b(tot) + (" + 쓰기 같은 양" if X == X1 else " 고정"), 17, weight=700)
                elif kind == base:
                    f.text(ex, y + 36, f"{fmt_b(tot)} · 기준", 17, color=MUTED)
                else:
                    r = tot / ref
                    big = ("1/" + format(1 / r, ".0f")) if r < 0.95 else ("×" + format(r, ".2f"))
                    f.text(ex, y + 37, big, 21, weight=700)
                    f.text(ex + len(big) * 12 + 14, y + 36, fmt_b(tot), 15, color=MUTED)
            y += 62
        y += 14
    return f


# ---------------------------------------------------------------- 그림 8. 컨텍스트 길이에 따른 KV cache 읽기 절감
CURVES = [("dsa", "mla", "DSA (V3.2)", SEL), ("qsa", "gqa", "QSA (Qwen3.8)", NAVY),
          ("csa", "mla", "CSA (V4)", CMP), ("hca", "mla", "HCA (V4)", RUST)]


def fig_gain():
    f = Fig("08-kv-read-saving-by-length", 1600, 820)
    f.text(40, 44, "KV cache 읽기량 절감 배수 — Full 대비 몇 분의 1로 줄었나", 26, weight=700)
    f.text(40, 78, "층 1개 · 디코드 1스텝 읽기 바이트 기준 (색인 키 읽기 포함) · 같은 대역폭이면 읽기 시간도 같은 비율 · 점선 = 모델 지원 길이 밖", 18, color=MUTED)
    x0, x1, top, bot = 130, 1240, 130, 720
    lo, hi = 1024, 10 * M
    X = lambda s: x0 + (math.log10(s) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)
    ylo, yhi = 0.5, 300
    Y = lambda v: bot - (math.log10(max(min(v, yhi), ylo)) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo)) * (bot - top)
    for v, lab in ((0.5, "0.5×"), (1, "1×"), (3, "3×"), (10, "10×"), (30, "30×"), (100, "100×"), (300, "300×")):
        f.line(x0, Y(v), x1, Y(v), GRID, 1)
        f.text(x0 - 12, Y(v) + 6, lab, 16, "end", color=MUTED)
    f.text(x0 - 12, top - 16, "읽기 절감", 15, "end", 700, MUTED)
    for s, lab in ((1024, "1K"), (8192, "8K"), (32768, "32K"), (K128, "128K"), (M, "1M"), (10 * M, "10M")):
        f.line(X(s), top, X(s), bot, GRID, 1)
        f.text(X(s), bot + 30, lab, 18, "middle", color=MUTED)
    f.text((x0 + x1) / 2, bot + 62, "컨텍스트 길이 (토큰)", 18, "middle", color=MUTED)
    f.line(x0, Y(1), x1, Y(1), INK, 2, "6 5")
    f.text(x0 + 10, Y(1) + 28, "1× = Full과 같음", 16, color=MUTED)
    pts = logspace(lo, hi, 241)
    ends = []
    for kind, base, label, col in CURVES:
        fn = lambda s, k=kind, b=base: total(read_parts(b, s)) / total(read_parts(k, s))
        a, b = split_by_support(kind, pts, X, Y, fn)
        f.poly(a, col, 3.5)
        f.poly(b, col, 3.5, "9 7")
        if SUPPORT[kind] < hi:
            f.dot(X(SUPPORT[kind]), Y(fn(SUPPORT[kind])), 6, col)
        ends.append((Y(fn(hi)), label, fn(hi)))
    ends.sort()
    used = []
    for yy, label, r in ends:
        y = yy
        for u in used:
            if abs(y - u) < 30:
                y = u + 30
        used.append(y)
        f.line(x1 + 4, yy, x1 + 22, y, AXIS, 1.2)
        f.text(x1 + 28, y + 7, f"{label}  {r:.0f}×", 19, weight=700)
    f.text(X(1.6 * M), Y(8.7) + 42, "선택형은 길어져도 상한에서 멈춘다", 18, "middle", 700, halo=True)
    f.text(X(1.6 * M), Y(8.7) + 66, "상한 = KV 1줄 ÷ 토큰당 색인 바이트 (Ⅲ장)", 16, "middle", color=MUTED, halo=True)
    f.text(X(1500), Y(150), "● = 모델 지원 길이 끝", 15, color=MUTED, halo=True)
    return f


# ---------------------------------------------------------------- 그림 9. KV cache 접근 패턴과 읽기 증폭
def fig_access():
    S = K128
    f = Fig("09-access-pattern", 1600, 740)
    basis(f, "읽는 모양 = 층 1개 · 컨텍스트 128K · KV BF16 · 초당 요청 수 = 요청 1개 · 모델 전체 · 20 토큰/s (계산값)")
    x0, x1 = 250, 720
    heads = [(x0, "읽는 모양 (왼쪽 = 오래된 토큰)", ""), (760, "조각 × 크기", "선택 단위"), (960, "필요한 양", "층당"),
             (1110, "4 KB 단위로 읽으면", "실제 전송량 (증폭)"), (1360, "초당 무작위 읽기", "요청 1개 · 모델 전체")]
    for x, h, s in heads:
        f.text(x, 100, h, 17, weight=700, color=MUTED)
        if s:
            f.text(x, 122, s, 14, color=MUTED)
    f.line(40, 136, 1560, 136, AXIS, 1)
    rnd = random.Random(11)
    n = 150
    cw = (x1 - x0) / n
    dsa = set(rnd.sample(range(n), 22))
    qsa = set()
    for st in rnd.sample(range(0, n - 4, 6), 7):
        qsa.update(range(st, st + 4))
    amp4k = lambda size: math.ceil(size / 4096) * 4096 / size
    hca_amt = total(read_parts("hca", S))
    iops_dsa = 61 * TOPK * 20
    iops_qsa = 12 * (TOPK // 4) * 20
    rows = [
        ("DSA", "토큰 단위 선택", dsa, SEL, "2,048 × 1.2 KB", fmt_b(MLA_B * TOPK),
         f"{fmt_b(4096 * TOPK)} ({amp4k(MLA_B):.1f}×)", f"FP8 KV면 {amp4k(656):.1f}×", f"{iops_dsa / 1e4:,.0f}만 회", "61층 × 2,048조각 × 20"),
        ("QSA", "4줄 블록 단위 선택", qsa, NAVY, "512 × 8 KB", fmt_b(GQA_B * TOPK),
         f"{fmt_b(GQA_B * TOPK)} ({amp4k(4 * GQA_B):.1f}×)", "블록이 페이지보다 큼", f"{iops_qsa / 1e4:,.0f}만 회", "12층 × 512블록 × 20"),
        ("HCA", "압축 KV 전체를 연속으로", set(range(n)), RUST, "연속 1구간", fmt_b(hca_amt),
         f"{fmt_b(hca_amt)} (1.0×)", "순차 읽기", "층당 1회", "연속 스트림"),
    ]
    for i, (nm, sub, hit, col, pieces, need, xfer, xnote, iops, inote) in enumerate(rows):
        y = 152 + i * 118
        if i % 2 == 0:
            f.rect(30, y - 4, 1540, 110, BOX[0], rx=8)
        f.text(40, y + 44, nm, 24, weight=700)
        f.text(40, y + 72, sub, 15, color=MUTED)
        for k in range(n):
            on = k in hit
            f.rect(x0 + k * cw, y + (24 if on else 34), max(cw - 1, 1), 40 if on else 20, col if on else FULL_L, rx=1)
        f.text(760, y + 56, pieces, 19, weight=700)
        f.text(960, y + 56, need, 19)
        f.text(1110, y + 52, xfer, 20, weight=700, color=INK if "1.0×" not in xfer else MUTED)
        f.text(1110, y + 80, xnote, 14, color=MUTED)
        f.text(1360, y + 52, iops, 20, weight=700, color=INK if i == 0 else MUTED)
        f.text(1360, y + 80, inote, 14, color=MUTED)
    # 아래 — 무엇에 영향을 주나
    boxes = [("읽기 증폭이란", ["필요한 조각보다 큰 단위로 가져와 생기는 낭비",
                              "1.2 KB가 필요해도 4 KB 페이지 단위면 4 KB 전송",
                              "→ 전송량 3.6배, 그만큼 링크 대역폭 소모"], None),
             ("HBM (전송 단위 64 B)", ["조각(1.2 KB)이 전송 단위보다 커서 증폭 1.0×",
                                    "대신 top-k가 끝나야 읽기 시작 — 직렬 대기",
                                    "→ 영향: 메모리 지연 · 동시 요청 처리 수"], SEL),
             ("대용량 계층 (CXL · SSD, 4 KB 페이지)", ["증폭만큼 링크 대역폭 소모 (DSA 3.6×)",
                                                 f"요청 1개가 초당 {iops_dsa / 1e4:,.0f}만 회 무작위 읽기",
                                                 "→ Gen5 NVMe 1개 최대 IOPS(약 250만) 수준"], CMP)]
    bx, bw_ = 30, 503
    for k, (title, lines, col) in enumerate(boxes):
        x = bx + k * (bw_ + 15)
        f.rect(x, 520, bw_, 196, BOX[0], BOX[1], 10, 1.2)
        if col:
            f.rect(x, 520, 8, 196, col, rx=3)
        f.text(x + 24, 560, title, 19, weight=700)
        for j, t in enumerate(lines):
            f.text(x + 24, 604 + j * 40, t, 16, weight=700 if t.startswith("→") else 400,
                   color=INK if t.startswith("→") else MUTED)
    return f


# ---------------------------------------------------------------- 그림 10. 색인 키 읽기와 절감 포화
def fig_saturation():
    f = Fig("10-index-saturation", 1600, 740)
    basis(f, "DSA(V3.2) 층 1개 · 디코드 1스텝 · KV BF16 1,152 B/줄 · 색인 키 132 B/줄 (계산값)")
    x0, x1, top, bot = 130, 960, 140, 620
    lo, hi = 1024, M
    X = lambda s: x0 + (math.log10(s) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)
    vmax = 1400
    Y = lambda v: bot - v / vmax * (bot - top)
    f.text(40, 102, "컨텍스트 토큰 1개당 읽는 바이트", 21, weight=700)
    for v in (0, 200, 400, 600, 800, 1000, 1200, 1400):
        f.line(x0, Y(v), x1, Y(v), GRID, 1)
        f.text(x0 - 12, Y(v) + 5, f"{v:,} B" if v else "0", 14, "end", color=MUTED)
    for s, lab in ((1024, "1K"), (8192, "8K"), (32768, "32K"), (K128, "128K"), (M, "1M")):
        f.line(X(s), top, X(s), bot, GRID, 1)
        f.text(X(s), bot + 28, lab, 17, "middle", color=MUTED)
    f.text((x0 + x1) / 2, bot + 58, "컨텍스트 길이 (토큰, 로그)", 16, "middle", color=MUTED)
    pts = logspace(lo, hi, 200)
    sel = lambda s: MLA_B * min(s, TOPK) / s
    f.area([(X(lo), Y(0))] + [(X(s), Y(DSA_IDX)) for s in pts] + [(X(hi), Y(0))], SEL_L)
    f.area([(X(s), Y(DSA_IDX)) for s in pts] + [(X(s), Y(DSA_IDX + sel(s))) for s in reversed(pts)], SEL)
    f.poly([(X(s), Y(DSA_IDX + sel(s))) for s in pts], NAVY, 3)
    f.line(x0, Y(MLA_B), x1, Y(MLA_B), FULL, 4)
    f.text(X(30000), Y(MLA_B) - 14, "Full (MLA) — 토큰마다 KV 1,152 B를 전부 읽음", 17, "middle", 700, halo=True)
    lim = SUPPORT["dsa"]
    f.rect(X(lim), top, x1 - X(lim), bot - top, "#ffffff", opacity=0.55)
    f.line(X(lim), top, X(lim), bot, AXIS, 1.5, "5 4")
    f.text(X(lim) + 8, top + 20, "V3.2 지원 길이 밖 (외삽)", 14, color=MUTED)
    for s, lab in ((8192, "8K"), (K128, "128K")):
        v = DSA_IDX + sel(s)
        f.dot(X(s), Y(v), 7, NAVY)
        if s == 8192:
            f.text(X(s) + 12, Y(v) - 14, f"{lab}: {v:.0f} B → {MLA_B / v:.1f}×", 17, weight=700, halo=True)
        else:
            f.text(X(s) - 8, Y(v) - 46, f"{lab}: {v:.0f} B → {MLA_B / v:.1f}×", 17, "end", 700, halo=True)
    f.text(X(1100), Y(560), "선택된 2,048줄 ÷ 길이", 16, weight=700, color="#ffffff")
    f.text(X(1100), Y(560) + 22, "길어질수록 0에 수렴", 14, color="#ffffff")
    f.text(X(30000), Y(DSA_IDX / 2) + 6, "바닥 = 색인 키 132 B — 길이와 무관하게 남음", 16, "middle", 700)
    f.text(x1 - 6, Y(330), f"상한 1,152 ÷ 132 = {MLA_B / DSA_IDX:.1f}×", 18, "end", 700, NAVY, halo=True)
    f.arrow(x1 - 60, Y(310), x1 - 60, Y(DSA_IDX) - 6, NAVY, 2)
    # 오른쪽 — 바닥을 낮추는 설계
    rx0 = 1040
    f.line(1000, 90, 1000, 700, GRID, 2, "6 6")
    f.text(rx0, 102, "바닥(토큰당 색인 바이트)을 낮추는 설계", 21, weight=700)
    f.text(rx0, 128, "상한 = KV 1줄 ÷ 토큰당 색인 바이트", 15, color=MUTED)
    rows = [("DSA · V3.2", "FP8 키 + scale", DSA_IDX, MLA_B, SEL, False),
            ("색인 키 FP4", "가정", 66, MLA_B, SEL_L, True),
            ("QSA · 4줄 블록 색인", "블록 키 256 B ÷ 4", QSA_IDX / 4, GQA_B, NAVY, False),
            ("CSA · 블록 + FP4", "64 B ÷ 4", CSA_IDX / 4, MLA_B, CMP, False)]
    bx, bwid, bmax = rx0, 250, 140
    for i, (nm, sub, b, kv, col, hyp) in enumerate(rows):
        y = 160 + i * 96
        f.text(bx, y + 20, nm, 18, weight=700, color=MUTED if hyp else INK)
        f.text(bx + 210, y + 20, sub, 14, color=MUTED)
        f.rect(bx, y + 34, b / bmax * bwid, 28, col, SEL if hyp else "none", 4, 1.5, "5 4" if hyp else None)
        f.text(bx + b / bmax * bwid + 10, y + 55, f"{b:.0f} B → 상한 {kv / b:.0f}×" if kv / b >= 10 else
               f"{b:.0f} B → 상한 {kv / b:.1f}×", 18, weight=700)
    f.rect(rx0 - 10, 556, 540, 150, BOX[0], BOX[1], 8, 1.2)
    f.text(rx0 + 6, 590, "KV를 FP8(656 B)로 줄이면 상한은 오히려 감소", 16, weight=700)
    f.text(rx0 + 6, 620, f"656 ÷ 132 = {656 / DSA_IDX:.1f}× — 색인 키가 KV 1줄의 {DSA_IDX / 656 * 100:.0f}%", 16, color=MUTED)
    f.text(rx0 + 6, 656, "→ KV와 색인 키를 함께 줄여야 상한 유지", 16, weight=700)
    f.text(rx0 + 6, 684, "QSA 상한은 GQA 2,048 B, 나머지는 MLA 1,152 B 기준", 13, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 11. 색인 연산 특성 — 읽기 시간 vs 연산 시간
def fig_index_compute():
    S = K128
    f = Fig("11-index-compute", 1600, 760)
    basis(f, "층 1개 · 컨텍스트 128K · 디코드 1스텝 · H200 (HBM 4.8 TB/s, FP8 1,979 · BF16 989 TFLOPS dense) 이론 시간 (계산값)")
    rows = [("DSA (V3.2)", "head 64 · FP8 키", DSA_IDX * S, DSA_FLOP * S, FP8_PK, SEL),
            ("DSA + MTP 2토큰", "쿼리 2개가 같은 키를 읽음", DSA_IDX * S, 2 * DSA_FLOP * S, FP8_PK, SEL),
            ("CSA (V4)", "head 64 · FP4 키 · 1/4 압축", CSA_IDX * S / 4, CSA_FLOP * S / 4, FP8_PK, CMP),
            ("CSA + MTP 2토큰", "쿼리 2개가 같은 키를 읽음", CSA_IDX * S / 4, 2 * CSA_FLOP * S / 4, FP8_PK, CMP),
            ("QSA (Qwen3.8)", "head 4 · BF16 블록 키", QSA_IDX * S / 4, QSA_FLOP * S / 4, BF16_PK, NAVY)]
    x0, xw, tmax = 330, 440, 4.0
    X = lambda t: x0 + t / tmax * xw
    f.text(40, 104, "디코드 — 색인 키 읽기 시간 vs 연산 시간", 21, weight=700)
    f.rect(40, 122, 16, 16, SEL, rx=3)
    f.text(62, 136, "HBM 읽기", 15, color=MUTED)
    f.rect(160, 122, 16, 16, SLATE, rx=3)
    f.text(182, 136, "연산", 15, color=MUTED)
    f.text(880, 104, "연산 ÷ 읽기", 17, weight=700, color=MUTED)
    f.text(880, 126, "100% 넘으면 연산 병목", 13, color=MUTED)
    for t in (0, 1, 2, 3, 4):
        f.line(X(t), 150, X(t), 640, GRID, 1)
        f.text(X(t), 664, f"{t} µs", 14, "middle", color=MUTED)
    for i, (nm, sub, b, fl, pk, col) in enumerate(rows):
        y = 160 + i * 96
        tr, tc = b / HBM_BW * 1e6, fl / pk * 1e6
        f.text(40, y + 30, nm, 19, weight=700)
        f.text(40, y + 54, sub, 14, color=MUTED)
        f.text(40, y + 76, f"{fmt_b(b)} · {fl / 1e9:.2f} GFLOP · {fl / b:.0f} FLOP/B", 13, color=MUTED)
        f.rect(x0, y + 12, max(X(tr) - x0, 3), 26, col, rx=3)
        f.text(X(tr) + 8, y + 31, f"{tr:.2f} µs", 15)
        f.rect(x0, y + 44, max(X(tc) - x0, 3), 26, SLATE, rx=3)
        f.text(X(tc) + 8, y + 63, f"{tc:.2f} µs", 15)
        r = tc / tr * 100
        f.text(880, y + 50, f"{r:.0f}%", 24, weight=700, color=INK if r >= 20 else MUTED)
        if r > 100:
            f.text(950, y + 50, "연산 병목", 16, weight=700)
    f.text(40, 712, "MTP = 한 스텝에 토큰 2개 예측 · 바이트당 연산이 2배가 되어 CSA는 H200 경계(412 FLOP/B)를 넘음", 15, color=MUTED)
    # 오른쪽 — 색인 연산 vs 본 attention
    rx0 = 1090
    f.line(1050, 90, 1050, 730, GRID, 2, "6 6")
    f.text(rx0, 104, "색인 연산 vs 본 attention 연산 (DSA)", 20, weight=700)
    idx_fl, att_fl = DSA_FLOP * S, MLA_ATT_FLOP * TOPK
    f.text(rx0, 146, "디코드 · 층당 · 128K", 16, weight=700, color=MUTED)
    sc = 210 / idx_fl
    for k, (nm, v, col) in enumerate((("색인 (128K개 키)", idx_fl, SEL_L), ("attention (2,048줄)", att_fl, SEL))):
        y = 166 + k * 50
        f.text(rx0, y + 24, nm, 15)
        f.rect(rx0 + 160, y + 6, v * sc, 28, col, rx=3)
        f.text(rx0 + 168 + v * sc, y + 26, f"{v / 1e9:.2f} GFLOP", 15, weight=700)
    f.text(rx0, 296, f"→ 색인 연산이 본 attention의 {idx_fl / att_fl:.1f}배", 17, weight=700)
    f.text(rx0, 360, "Prefill · 61층 합계 · Full dense = 100", 16, weight=700, color=MUTED)
    dense_pair = 128 * (192 + 128) * 2
    idx_share = DSA_FLOP / dense_pair * 100
    psc = 300 / 100
    items = [("Full dense", [(100, FULL)], "100"), ]
    for S_, lab in ((K128, "DSA · 128K"), (M, "DSA · 1M*")):
        sel_share = (S_ * TOPK) / (S_ * S_ / 2) * 100
        items.append((lab, [(idx_share, SEL_L), (sel_share, SEL)], f"{idx_share + sel_share:.1f} → {100 / (idx_share + sel_share):.1f}배 감소"))
    for k, (lab, segs, txt) in enumerate(items):
        y = 380 + k * 56
        f.text(rx0, y + 26, lab, 15)
        x = rx0 + 120
        for v, col in segs:
            f.rect(x, y + 8, v * psc, 28, col, rx=3)
            x += v * psc + 2
        f.text(x + 8, y + 28, txt, 15, weight=700)
    f.text(rx0, 580, f"색인 {idx_share:.0f} (전 쌍) + 선택 attention", 15, color=MUTED)
    f.text(rx0, 612, "→ prefill도 색인이 절감 상한(약 5배)을 정함", 17, weight=700)
    f.text(rx0, 650, "* 1M은 V3.2 지원 길이 밖 (외삽)", 13, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 12. Hybrid 모델 전체 합산 (Qwen3.8-Flash-Next)
def fig_hybrid():
    f = Fig("12-hybrid-total", 1600, 720)
    basis(f, "Qwen3.8-Flash-Next 모델 전체 (QSA 12층 + DeltaNet 36층) · 요청 1개 · 디코드 1스텝 · KV BF16 · state FP32 (계산값)")
    lens = [(8192, "8K"), (32768, "32K"), (K128, "128K"), (262144, "256K"), (M, "1M")]
    x0, top, bot = 150, 130, 600
    vmax = 1.2e9
    Y = lambda v: bot - v / vmax * (bot - top)
    for v in (0, 0.4e9, 0.8e9, 1.2e9):
        f.line(x0, Y(v), 1120, Y(v), GRID, 1)
        f.text(x0 - 12, Y(v) + 6, f"{v / 1e9:.1f} GB" if v else "0", 15, "end", color=MUTED)
    segs = [("state 읽기", STATE), ("state 쓰기", STATE_L), ("QSA KV 읽기", SEL), ("QSA 색인 키 읽기", SEL_L)]
    bw, slot = 110, 190
    last = None
    for j, (S, lab) in enumerate(lens):
        x = x0 + 40 + j * slot
        parts = hybrid_parts(S)
        acc = 0
        mids = []
        out = S > 262144
        for (nm, col), v in zip(segs, parts):
            f.rect(x, Y(acc + v) + 1, bw, Y(acc) - Y(acc + v) - 2, col, AXIS if out else "none", 3, 1.5, "5 4" if out else None)
            mids.append((Y(acc) + Y(acc + v)) / 2)
            acc += v
        state_share = (parts[0] + parts[1]) / acc * 100
        f.text(x + bw / 2, Y(acc) - 34, fmt_b(acc), 18, "middle", 700)
        f.text(x + bw / 2, Y(acc) - 12, f"state {state_share:.0f}%", 16, "middle", color=MUTED)
        f.text(x + bw / 2, bot + 30, lab + ("*" if out else ""), 18, "middle", 700)
        last = (x + bw, mids)
    used = []
    for (nm, col), my in zip(segs, last[1]):
        y = my
        for u in used:
            if abs(y - u) < 26:
                y = u - 26
        used.append(y)
        f.line(last[0] + 4, my, last[0] + 24, y, AXIS, 1.2)
        f.rect(last[0] + 30, y - 8, 14, 14, col, rx=2)
        f.text(last[0] + 50, y + 5, nm, 15)
    f.text(150, 670, "* 지원 길이(256K) 밖 — 외삽", 14, color=MUTED)
    bx = 1260
    f.rect(bx - 20, 120, 340, 470, BOX[0], BOX[1], 10, 1.2)
    full128 = 48 * GQA_B * K128
    hyb128 = sum(hybrid_parts(K128))
    f.text(bx, 165, "128K에서 비교", 19, weight=700)
    f.text(bx, 205, "전 층 full GQA(48층)라면", 16, color=MUTED)
    f.text(bx, 237, fmt_b(full128), 26, weight=700)
    f.text(bx, 277, "hybrid (현재 구성)", 16, color=MUTED)
    f.text(bx, 309, f"{fmt_b(hyb128)} (1/{full128 / hyb128:.0f})", 26, weight=700)
    f.text(bx, 365, "디코드 1스텝 쓰기량", 19, weight=700)
    f.text(bx, 403, "state 덮어쓰기", 16, color=MUTED)
    f.text(bx, 433, fmt_b(36 * GDN_STATE), 24, weight=700)
    kv_write = 12 * (GQA_B + QSA_IDX / 4)
    f.text(bx, 473, "KV · 색인 추가 (12층)", 16, color=MUTED)
    f.text(bx, 503, fmt_b(kv_write), 24, weight=700)
    f.text(bx, 545, f"→ 쓰기는 state가 약 {round(36 * GDN_STATE / kv_write, -2):,.0f}배", 16, weight=700, color=INK)
    return f


# ---------------------------------------------------------------- 그림 13. Sparse와 Hybrid의 역할 — 읽기 vs 크기
CONFIGS = [("full", "전 층 Full", "GQA 48층 (가정)"), ("sparse", "Sparse만", "QSA 48층 (가정)"),
           ("hybrid", "Hybrid만", "GQA 12층 + state 36층 (가정)"), ("both", "Sparse + Hybrid", "QSA 12층 + state 36층 (실제)")]


def fig_sparse_hybrid():
    S = K128
    f = Fig("13-sparse-vs-hybrid", 1700, 700)
    basis(f, "Qwen3.8-Flash-Next 차원으로 층 구성만 바꾼 비교 · 48층 · 요청 1개 · 128K · KV BF16, state FP32 (계산값)")
    for (nm, col), lx in zip((("Full KV cache", FULL), ("Sparse KV cache", SEL), ("색인 키", SEL_L), ("Linear state", STATE)),
                             (40, 210, 400, 510)):
        f.rect(lx, 66, 16, 16, col, rx=3)
        f.text(lx + 22, 80, nm, 15, color=MUTED)
    X1, X2, pw, vmax = 380, 850, 280, 14e9
    sc = pw / vmax
    f.text(X1, 126, "디코드 1스텝 읽기 + 쓰기", 20, weight=700)
    f.text(X2, 126, "KV cache + state 크기", 20, weight=700)
    for X in (X1, X2):
        for v in (0, 5e9, 10e9):
            f.line(X + v * sc, 146, X + v * sc, 560, GRID, 1)
            f.text(X + v * sc, 584, f"{v / 1e9:.0f} GB" if v else "0", 14, "middle", color=MUTED)
    ref_io = sum(qwen_config("full", S)[0])
    ref_sz = sum(qwen_config("full", S)[1])
    for i, (kind, nm, sub) in enumerate(CONFIGS):
        y = 160 + i * 100
        if kind == "both":
            f.rect(30, y - 6, 1280, 96, BOX[0], rx=8)
        f.text(40, y + 34, nm, 20, weight=700)
        f.text(40, y + 60, sub, 14, color=MUTED)
        io, size = qwen_config(kind, S)
        kvcol = FULL if kind in ("full", "hybrid") else SEL
        for X, parts, ref in ((X1, io, ref_io), (X2, size, ref_sz)):
            x = X
            for v, col in zip(parts, (kvcol, SEL_L, STATE)):
                if v <= 0:
                    continue
                w = max(v * sc, 2)
                f.rect(x, y + 14, w, 36, col, rx=3)
                x += w + 1
            tot = sum(parts)
            f.text(x + 10, y + 39, f"{fmt_b(tot)}", 17, weight=700)
            f.text(x + 10 + len(fmt_b(tot)) * 10 + 8, y + 39, f"({tot / ref * 100:.0f}%)" if kind != "full" else "기준", 15, color=MUTED)
    # 오른쪽 — 길이별 읽기 + 쓰기
    tx = 1360
    f.line(1335, 100, 1335, 660, GRID, 2, "6 6")
    f.text(tx, 126, "읽기 + 쓰기 길이별", 19, weight=700)
    lens = [(32768, "32K"), (K128, "128K"), (262144, "256K")]
    for j, (_, lab) in enumerate(lens):
        f.text(tx + 170 + j * 70, 162, lab, 15, "middle", 700, color=MUTED)
    for i, (kind, nm, _) in enumerate(CONFIGS):
        y = 200 + i * 100
        f.text(tx, y + 20, nm, 15, weight=700)
        for j, (S_, _) in enumerate(lens):
            v = sum(qwen_config(kind, S_)[0])
            f.text(tx + 170 + j * 70, y + 20, f"{v / 1e9:.2f}" if v < 10e9 else f"{v / 1e9:.1f}", 16, "middle",
                   700 if kind in ("sparse", "both") else 400, INK if kind in ("sparse", "both") else MUTED)
    f.text(tx, 620, "단위 GB · 256K = 지원 길이 끝", 13, color=MUTED)
    s32, b32 = sum(qwen_config("sparse", 32768)[0]), sum(qwen_config("both", 32768)[0])
    f.text(40, 640, f"32K에서는 Sparse만({fmt_b(s32)})과 Sparse + Hybrid({fmt_b(b32)})의 읽기가 거의 같음 — state 읽기 · 쓰기가 KV 절감분을 상쇄", 16, weight=700)
    f.text(40, 670, "크기는 KV cache를 가진 층 수(48 → 12)가 정함 — Sparse만으로는 크기가 줄지 않음", 16, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 14. 데이터별 메모리 접근 특성 (기준 모델 2개, 128K)
def fig_behavior():
    S = K128
    v_kv, v_idx, v_sel = 61 * 656 * S, 61 * DSA_IDX * S, 61 * 656 * TOPK
    q_kv, q_idx, q_sel = 12 * GQA_B * S, 12 * QSA_IDX * (S // 4), 12 * GQA_B * TOPK
    st = 36 * GDN_STATE
    w1, w32 = weight_read(1), weight_read(32)
    groups = [
        ("Sparse 기준 모델 — DeepSeek-V3.2 (DSA 61층 · 실제 배포 형식)", [
            ("가중치", "FP8 체크포인트", SLATE, 689e9,
             (w1 * 1e9, f"{w1:.0f} / 689 GB ({w1 / 689 * 100:.0f}%)", f"배치 1 · 배치 32면 {w32:.0f} / 689 GB ({w32 / 689 * 100:.0f}%)"),
             (0, "0", "바뀌지 않음"), "모든 요청 공유", "HBM — 대역폭"),
            ("KV cache", "FP8 656 B/줄", SEL, v_kv, (v_sel, None, "선택된 2,048줄 × 61층"),
             (61 * 656, None, "끝에 1줄 추가"), "요청마다 × N", "용량 계층 후보"),
            ("색인 키", "FP8 + scale 132 B/줄", SEL_L, v_idx, (v_idx, None, "매번 전부 훑음"),
             (61 * DSA_IDX, None, "끝에 1개 추가"), "요청마다 × N", "HBM — 순차 스캔")]),
        ("Hybrid 기준 모델 — Qwen3.8-Flash-Next (QSA 12층 + DeltaNet 36층)", [
            ("KV cache", "BF16 2,048 B/줄 · 12층", SEL, q_kv, (q_sel, None, "선택된 2,048줄 × 12층"),
             (12 * GQA_B, None, "끝에 1줄 추가"), "요청마다 × N", "용량 계층 후보"),
            ("블록 색인 키", "256 B / 4토큰 (BF16 가정)", SEL_L, q_idx, (q_idx, None, "매번 전부 훑음"),
             (12 * QSA_IDX / 4, None, "4토큰마다 1개 (평균)"), "요청마다 × N", "HBM — 순차 스캔"),
            ("Linear state", "FP32 · 36층", STATE, st, (st, None, "매번 전부"),
             (st, None, "매번 전부 덮어씀"), "요청마다 × N", "온칩 + 쓰기 대역폭")]),
    ]
    f = Fig("14-memory-behavior", 1700, 860)
    basis(f, "요청 1개 · 컨텍스트 128K · 디코드 1스텝 · 막대 = 보유량 중 매 스텝 읽거나 쓰는 비율 (계산값)")
    cols = [(40, "데이터"), (300, "보유량"), (450, "매 스텝 읽기 (읽기 / 보유)"), (940, "매 스텝 쓰기 (쓰기 / 보유)"),
            (1360, "공유 범위"), (1510, "어울리는 메모리")]
    for x, h in cols:
        f.text(x, 100, h, 18, weight=700, color=MUTED)
    f.line(40, 116, 1660, 116, AXIS, 1)
    y = 128
    for title, rows in groups:
        f.text(40, y + 30, title, 19, weight=700)
        y += 46
        for i, (nm, sub, col, size, (rv, rlab, rnote), (wv, wlab, wnote), share, mem) in enumerate(rows):
            if i % 2 == 0:
                f.rect(30, y - 6, 1640, 96, BOX[0], rx=8)
            f.text(40, y + 34, nm, 22, weight=700)
            f.text(40, y + 62, sub, 14, color=MUTED)
            f.text(300, y + 42, fmt_b(size), 21, weight=700)
            for X, v, lab, note, strong in ((450, rv, rlab, rnote, True), (940, wv, wlab, wnote, False)):
                frac = v / size
                f.rect(X, y + 22, 150, 24, "#e3e6ee", rx=4)
                if v > 0:
                    f.rect(X, y + 22, max(150 * frac, 3), 24, col, rx=4)
                txt = lab if lab else f"{fmt_b(v)} / {fmt_b(size)} ({pct(frac)})"
                f.text(X + 164, y + 41, txt, 17, weight=700 if (strong or frac > 0.5) else 400)
                f.text(X, y + 72, note, 14, color=MUTED)
            f.text(1360, y + 42, share, 17, weight=700 if share != "모든 요청 공유" else 400)
            f.text(1510, y + 42, mem, 17, weight=700)
            y += 100
        y += 10
    return f


# ---------------------------------------------------------------- 그림 15. 단계별 특성과 예상 제약
def fig_stage():
    S = K128
    f = Fig("15-stage-summary", 1700, 790)
    basis(f, "DSA(V3.2) 디코드 1스텝 · 층 1개 · 컨텍스트 128K · 실제 배포 형식 (KV FP8 656 B, 색인 132 B) · 시간 = H200 사양으로 나눈 이론값")
    cols = [(40, "단계"), (250, "특징 — 데이터 · 연산량"), (680, "이론 시간 (H200)"), (990, "예상 제약"),
            (1225, "검증 지표"), (1440, "관련 HW · 시스템 요소")]
    for x, h in cols:
        f.text(x, 100, h, 18, weight=700, color=MUTED)
    f.line(40, 116, 1660, 116, AXIS, 1)
    idx_b, idx_fl = DSA_IDX * S, DSA_FLOP * S
    score_b = 2 * SCORE_B * S
    gat_b = 656 * TOPK
    att_fl = MLA_ATT_FLOP * TOPK
    t = lambda b: b / HBM_BW * 1e6
    rows = [
        ("1", "색인 키 훑기", "대역폭 + 연산", SEL, SEL_L,
         (f"{fmt_b(idx_b)} 순차 읽기 · {idx_fl / 1e9:.1f} GFLOP", f"키 1 B당 {idx_fl / idx_b:.0f} FLOP · 본 attention 연산의 {idx_fl / att_fl:.1f}배"),
         [("HBM", t(idx_b), SEL), ("FP8", idx_fl / FP8_PK * 1e6, SLATE)],
         ("HBM 대역폭 · FP8 연산", f"연산이 읽기 시간의 {idx_fl / FP8_PK / (idx_b / HBM_BW) * 100:.0f}%"),
         ("단계 시간 · 달성 대역폭", "연산기 사용률"), ("HBM 대역폭 · FP8 연산기", "메모리 근처 처리는 연산 밀도가 관건")),
        ("2", "top-k 선택", "비교 · 동기화", SEL, SEL_L,
         (f"점수 128K개 기록 · 읽기 ({fmt_b(score_b)})", "비교 · 정렬 — FLOP에 안 잡힘"),
         [("HBM", t(score_b), OTHER)],
         ("비교 처리 · 동기화", "정렬 시간은 실측 필요"), ("top-k 시간 비중", ""), ("온칩 SRAM · reduction", "스레드 간 동기화")),
        ("3", "KV gather", "지연 · 불규칙 접근", SEL, SEL_L,
         (f"656 B 조각 2,048개 = {fmt_b(gat_b)}", "top-k가 끝나야 시작 (직렬)"),
         [("HBM", t(gat_b), SEL)],
         ("지연 · 의존성", "계층에서는 증폭 · IOPS"), ("달성 대역폭 · 메모리 요청 수", ""), ("메모리 동시 요청 수 · DMA", "paged KV 주소 변환")),
        ("4", "attention", "연산 소형", FULL, FULL_L,
         (f"선택된 2,048줄 · {att_fl / 1e9:.2f} GFLOP", f"색인 연산의 1/{idx_fl / att_fl:.1f}"),
         [("BF16", att_fl / BF16_PK * 1e6, SLATE)],
         ("FP8 KV 역양자화", "연산량 자체는 작음"), ("커널 시간 · 역양자화 비중", ""), ("텐서 코어 · 역양자화 경로", "")),
    ]
    tx, tw, tmax = 680, 150, 4.0
    y = 128
    for num, step, tag, dark, light, (a1, a2), bars, (b1, b2), (c1, c2), (d1, d2) in rows:
        f.rect(30, y, 1640, 116, BOX[0] if int(num) % 2 else "#ffffff", rx=8)
        f.rect(44, y + 38, 40, 40, dark, rx=8)
        f.text(64, y + 66, num, 20, "middle", 700, "#ffffff")
        f.text(100, y + 66, step, 20, weight=700)
        chip(f, 250, y + 12, len(tag) * 16 + 44, 32, tag, dark, light, 16)
        f.text(250, y + 74, a1, 17, weight=700)
        f.text(250, y + 100, a2, 14, color=MUTED)
        for k, (lab, tv, col) in enumerate(bars):
            by = y + 22 + k * 36 if len(bars) > 1 else y + 40
            f.text(tx, by + 17, lab, 13, color=MUTED)
            f.rect(tx + 44, by, max(tv / tmax * tw, 3), 24, col, rx=3)
            f.text(tx + 52 + tv / tmax * tw, by + 18, f"{tv:.2f} µs", 15, weight=700)
        for x, t1, t2 in ((990, b1, b2), (1225, c1, c2), (1440, d1, d2)):
            f.text(x, y + 52, t1, 16, weight=700)
            if t2:
                f.text(x, y + 80, t2, 14, color=MUTED)
        y += 124
    y += 10
    f.line(40, y - 4, 1660, y - 4, AXIS, 1.2, "6 5")
    f.rect(30, y + 6, 1640, 116, BOX[0], rx=8)
    f.rect(44, y + 44, 40, 40, SEL, rx=8)
    f.text(64, y + 71, "+", 22, "middle", 700, "#ffffff")
    f.text(100, y + 72, "KV cache 보관", 20, weight=700)
    chip(f, 250, y + 18, 16 * 2 + 44, 32, "용량", SEL, SEL_L, 16)
    kv_req, idx_req = 61 * 656 * S, 61 * DSA_IDX * S
    f.text(250, y + 80, f"요청당 KV {fmt_b(kv_req)} + 색인 {fmt_b(idx_req)}", 17, weight=700)
    f.text(250, y + 106, "61층 합계 · × 동시 요청 수", 14, color=MUTED)
    f.text(tx, y + 72, "— (용량 문제)", 15, color=MUTED)
    for x, t1, t2 in ((990, "HBM 용량", "분산 방식(복제 여부)에 좌우"), (1225, "동시 요청 수 · HBM 사용량", ""),
                      (1440, "HBM 용량 · 메모리 계층", "KV 할당 · 계층 간 이동")):
        f.text(x, y + 58, t1, 16, weight=700)
        if t2:
            f.text(x, y + 86, t2, 14, color=MUTED)
    return f


FIGS = [fig_trend, fig_write_read, fig_layers, fig_weights_kv, fig_moe, fig_sparse_mix, fig_read_store, fig_gain,
        fig_access, fig_saturation, fig_index_compute, fig_hybrid, fig_sparse_hybrid, fig_behavior, fig_stage]
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


def print_summary():
    S = K128
    print("층당 · 128K · 1스텝 KV cache 읽기 / 크기 (BF16 KV)")
    for k in ("mla", "dsa", "csa", "hca", "gqa", "qsa", "gdn"):
        r, s = read_parts(k, S), store_parts(k, S)
        print(f"  {k:4s} read {fmt_b(sum(r)):>9s} (idx {r[0] / sum(r) * 100:4.0f}%) | store {fmt_b(sum(s)):>9s}")
    print("  비율 dsa/csa/hca vs mla:", [round(ratio_vs(k, "mla", S), 1) for k in ("dsa", "csa", "hca")],
          "qsa vs gqa:", round(ratio_vs("qsa", "gqa", S), 1))
    for k, b in (("dsa", "mla"), ("qsa", "gqa"), ("csa", "mla"), ("hca", "mla")):
        print(f"  {k} 길이별 절감:", {s: round(ratio_vs(k, b, s), 1) for s in (1024, 8192, K128, M)})
    v4_1m = 30 * total(read_parts("csa", M)) + 31 * total(read_parts("hca", M))
    v4_128 = 30 * total(read_parts("csa", S)) + 31 * total(read_parts("hca", S))
    print(f"  V4-Pro 전체 vs MLA: 128K 1/{61 * MLA_B * S / v4_128:.0f} · 1M 1/{61 * MLA_B * M / v4_1m:.0f}")
    print("V3.2 모델 전체 디코드 구성 (KV, 기타, 색인):")
    for s in (8192, 32768, K128, 163840, M):
        p = dsa_model_io(s)
        print(f"  {s:>8d} total {fmt_b(sum(p)):>8s} KV {p[0] / sum(p) * 100:3.0f}% 기타 {p[1] / sum(p) * 100:3.0f}% "
              f"색인 {p[2] / sum(p) * 100:3.0f}% | Full 1/{61 * MLA_B * s / sum(p):.1f}")
    print("하이브리드 Qwen3.8 (state 비중):", {lab: f"{(p[0] + p[1]) / sum(p) * 100:.0f}% {fmt_b(sum(p))}"
                                         for lab, p in (("8K", hybrid_parts(8192)), ("32K", hybrid_parts(32768)),
                                                        ("128K", hybrid_parts(S)), ("1M", hybrid_parts(M)))})
    print("구성별 (읽기+쓰기 | 크기):")
    for kind, nm, _ in CONFIGS:
        print(f"  {nm:16s}", [f"{fmt_b(sum(qwen_config(kind, s)[0]))}" for s in (32768, S, 262144)],
              "| 크기 128K", fmt_b(sum(qwen_config(kind, S)[1])))
    print("색인 연산 (128K, 층당): DSA read µs", round(DSA_IDX * S / HBM_BW * 1e6, 2), "comp µs", round(DSA_FLOP * S / FP8_PK * 1e6, 2),
          "| attention GFLOP", round(MLA_ATT_FLOP * TOPK / 1e9, 2), "ratio", round(DSA_FLOP * S / (MLA_ATT_FLOP * TOPK), 2))
    print("가중치 읽기 GB: 배치 1", round(weight_read(1), 1), "배치 32", round(weight_read(32), 1))


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
