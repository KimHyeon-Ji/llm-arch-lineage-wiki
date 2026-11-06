"""Sparse · Hybrid attention 보고 자료(02-deck-draft-claude.md)용 계산 · 그림 생성기.
실행: python arch_insight/figures-claude/make_figures.py [그림 이름 ...]  → 같은 폴더에 *.svg, *.png
그림 양식은 reports/llm-memory-requirements/figures/make_figures.py와 같다 (SVG를 직접 그리고 headless 브라우저로 PNG).

수치 기준
- 층 구성 · head 수 · 차원 · 지원 길이 : 공식 config.json (reports/llm-memory-requirements/data/raw/)
- 1M = 1,000,000 토큰, 1 GB = 10^9 B. 그림 2만 메모리 보고서 기준(1M = 1,048,576)을 그대로 쓴다
- 구조 효과 비교(그림 6 · 7 · 8 · 10): KV cache는 BF16으로 통일, 색인 키는 공개 저장 형식
  DSA 색인 키 = FP8 128 B + FP32 scale 4 B = 132 B · CSA 색인 키 = FP4 64 B · QSA 블록 키 = 정밀도 표기가 없어 BF16 256 B로 가정
- 배포 시나리오(그림 12 · 14 · 15 · 16 · 17): DeepSeek-V3.2 실제 형식 — 가중치 FP8 689 GB, KV FP8 656 B/줄(FlashMLA), 색인 132 B/줄
- 지원 길이: V3 · V3.2 163,840 · Qwen3.8-Flash-Next 262,144 · V4-Pro 1,048,576. 지원 길이 밖은 점선(외삽)
- 읽기량은 KV를 한 번씩만 읽는다고 본 하한이다. top-k · 흩어진 읽기의 비효율은 따로 표시한 곳 외에는 넣지 않았다
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
SEL, SEL_L = "#2a78d6", "#b7d3f6"        # 선택 읽기 (DSA · QSA)
NAVY = "#184f95"                         # 선택 읽기의 둘째 계열 (QSA, 계층 시나리오)
CMP, CMP_L = "#eb6834", "#f8c6b1"        # 압축 저장 (CSA · HCA)
RUST = "#9c3d14"                         # 압축의 둘째 계열 (HCA)
STATE, STATE_L = "#1baf7a", "#a9e0cb"    # linear state
WIN, WIN_L = "#eda100", "#f7d98f"        # sliding window
VIOLET = "#6250d6"                       # 민감도 시나리오

GB, MB = 1e9, 1e6
M = 1_000_000
K128 = 131072

SUPPORT = {"mla": 163840, "dsa": 163840, "gqa": 262144, "qsa": 262144, "csa": 1048576, "hca": 1048576}


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


# ---------------------------------------------------------------- 배포 시나리오 모델 (DeepSeek-V3.2 · H200 × 8)
DEP = dict(gpus=8, hbm=141.0, hbm_bw=4800.0, reserve=0.10, weights=689.0, tps=20.0, tier_bw=64.0,
           amp=1.0, link_eff=1.0, kv=656, idx=132, layers=61, replicate=False)


def weight_read(n):
    """디코드 1스텝에 읽는 가중치 (GB). 균등 routing일 때 배치 n의 expert 합집합 (V3.2: 비 expert 17.1B + 58층 × expert 44 MB)."""
    union = 256 * (1 - (1 - 8 / 256) ** max(n, 1))
    return 17.1 + 58 * union * 0.04404


def solve(S, mode, **over):
    """동시 처리 가능한 요청 수와 먼저 막히는 제약. mode = full · dsa_hbm · dsa_tier."""
    p = dict(DEP, **over)
    g = 1 if p["replicate"] else p["gpus"]          # 복제면 GPU 1장이 모든 요청을 들고 있는 것과 같다
    hbm = p["hbm"] * g
    bw = p["hbm_bw"] * g
    wts = p["weights"] / p["gpus"] * g
    tier = p["tier_bw"] * g * p["link_eff"]
    kv = p["layers"] * p["kv"] * S / GB
    idx = p["layers"] * p["idx"] * S / GB
    sel = p["layers"] * p["kv"] * min(S, TOPK) / GB
    free = hbm * (1 - p["reserve"]) - wts
    budget = 1 / p["tps"]

    def check(n):
        w = weight_read(n) / p["gpus"] * g
        if mode == "full":
            cap, t = n * kv, (w + n * kv) / bw
        elif mode == "dsa_hbm":
            cap, t = n * (kv + idx), (w + n * (idx + sel)) / bw
        else:
            cap, t = n * idx, (w + n * idx) / bw + n * sel * p["amp"] / tier
        return cap <= free, t <= budget

    n = 0
    while n < 5000:
        c, t = check(n + 1)
        if not (c and t):
            why = "용량" if not c else "시간"
            return n, why
        n += 1
    return n, "-"


def step_times(S, mode, n, **over):
    """요청 n개일 때 디코드 1스텝의 메모리 시간 구성 (ms): 가중치 · 색인 · 선택 KV."""
    p = dict(DEP, **over)
    bw, tier = p["hbm_bw"] * p["gpus"], p["tier_bw"] * p["gpus"] * p["link_eff"]
    idx = p["layers"] * p["idx"] * S / GB
    sel = p["layers"] * p["kv"] * min(S, TOPK) / GB
    w = weight_read(n) / bw * 1e3
    i = n * idx / bw * 1e3
    s = n * sel * p["amp"] / (tier if mode == "dsa_tier" else bw) * 1e3
    return w, i, s


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
        if len(pts) < 2:
            return
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.raw(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" '
                 f'stroke-linejoin="round" stroke-linecap="round"{d}/>')

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
    if v >= 1e9:
        return f"{v / 1e9:.2f} GB" if v < 10e9 else f"{v / 1e9:.1f} GB"
    if v >= 10e6:
        return f"{v / 1e6:.0f} MB"
    if v >= 1e6:
        return f"{v / 1e6:.1f} MB"
    if v >= 1e3:
        return f"{v / 1e3:.0f} KB"
    return f"{v:.0f} B"


def basis(f, s, x=40, y=40):
    f.text(x, y, s, 19, color=MUTED)


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
CAT = [("full", "Full attention만", FULL), ("select", "선택 읽기 (+ full)", SEL), ("compress", "압축 저장 (+ full)", CMP),
       ("state", "Linear state + attention", STATE), ("swa", "Sliding window + full", WIN)]
HALVES = ["2024 H2", "2025 H1", "2025 H2", "2026 H1", "2026 H2"]


def _half(d):
    return f"{d[:4]} H{1 if int(d[5:7]) <= 6 else 2}"


def _cat(r):
    if r["category"] == "sparse":
        return "compress" if "DeepSeek-V4" in r["model"] else "select"
    return r["category"]


def fig_trend():
    rows = list(csv.DictReader(open(LINEAGE, encoding="utf-8-sig")))
    cnt = {h: {c: 0 for c, _, _ in CAT} for h in HALVES}
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
        for c, _, col in CAT:
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
    for c, label, col in CAT:
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


# ---------------------------------------------------------------- 그림 2. 모델 계열별 가중치 vs KV cache
FLAGSHIPS = [
    ("full", ["DeepSeek-V3", "Hy3"]),
    ("select", ["GLM-5.3", "MiniMax-M3", "Hy4-preview"]),
    ("compress", ["DeepSeek-V4-Pro", "DeepSeek-V4.1-Flash"]),
    ("state", ["Qwen3.8-2.4T-A95B", "Kimi K3", "Nemotron 3 Ultra 550B-A55B"]),
    ("swa", ["MiMo-V2.6-Pro", "Muse Spark Open 816B-A42B"]),
]


def fig_weights_kv():
    rows = {r["model"]: r for r in csv.DictReader(open(LINEAGE, encoding="utf-8-sig"))}
    col = {c: k for c, _, k in CAT}
    lab = {c: l for c, l, _ in CAT}
    short = {"Nemotron 3 Ultra 550B-A55B": "Nemotron 3 Ultra", "Muse Spark Open 816B-A42B": "Muse Spark Open"}
    n_rows = sum(len(m) for _, m in FLAGSHIPS)
    H = 150 + len(FLAGSHIPS) * 30 + n_rows * 48 + 70
    f = Fig("02-weights-vs-kv", 1600, H)
    basis(f, "주요 시리즈 대표 모델 · 가중치 = 공식 체크포인트 크기 · KV cache = 요청 1개, BF16 구조 기준 (메모리 보고서, 1M = 1,048,576)")
    X1, W1, V1 = 400, 380, 2600
    X2, W2, V2 = 980, 330, 400
    f.text(X1, 100, "가중치 (모든 요청이 공유)", 21, weight=700)
    f.text(X2, 100, "KV cache · 요청 1개, 1M", 21, weight=700)
    f.text(1430, 92, "가중치와 같아지는", 16, weight=700)
    f.text(1430, 114, "동시 1M 요청 수", 16, weight=700)
    y_end = H - 60
    for X, W, V, ticks in ((X1, W1, V1, (0, 1000, 2000)), (X2, W2, V2, (0, 200, 400))):
        for v in ticks:
            f.line(X + v / V * W, 130, X + v / V * W, y_end, GRID, 1)
            f.text(X + v / V * W, y_end + 24, f"{v:,} GB" if v else "0", 14, "middle", color=MUTED)
    y = 136
    for c, models in FLAGSHIPS:
        f.rect(40, y + 8, 12, 12, col[c], rx=2)
        f.text(60, y + 20, lab[c], 16, weight=700, color=MUTED)
        y += 30
        for m in models:
            r = rows[m]
            w, kv, n = float(r["weights_GB"]), float(r["kv1m_state1_GB"]), float(r["N_equal_1M"])
            f.text(40, y + 22, short.get(m, m), 17, weight=700)
            f.text(40, y + 40, f"{r['date'][:7]} · {r['weights_format']}", 12, color=MUTED)
            f.rect(X1, y + 10, max(w / V1 * W1, 3), 24, SLATE, rx=3)
            f.text(X1 + w / V1 * W1 + 8, y + 28, f"{w:,.0f} GB", 15)
            f.rect(X2, y + 10, max(kv / V2 * W2, 3), 24, col[c], rx=3)
            f.text(X2 + kv / V2 * W2 + 8, y + 28, f"{kv:.0f} GB" if kv >= 10 else f"{kv:.1f} GB", 15, weight=700)
            f.text(1430, y + 28, f"{n:.1f}개" if n < 10 else f"{n:.0f}개", 17, weight=700)
            y += 48
    return f


# ---------------------------------------------------------------- 그림 3. MoE vs Sparse attention
def fig_moe():
    f = Fig("03-moe-analogy", 1600, 860)
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


# ---------------------------------------------------------------- 그림 4. KV cache를 표로 보면
def _rows(f, x, y, w, h, n, fill, edge, pick=(), pick_fill=None, rx=3):
    rh = h / n
    for k in range(n):
        f.rect(x, y + k * rh + 1, w, rh - 2, pick_fill if k in pick else fill, rx=2)
    f.rect(x - 1, y - 1, w + 2, h + 2, "none", edge, rx, 1.5)


def fig_axes():
    f = Fig("04-kv-axes", 1600, 720)
    basis(f, "층 1개의 KV cache를 표로 본 그림 · 한 줄 = 토큰 하나 · 1줄 크기는 BF16 기준 바이트 (색인 키는 FP8 + scale)")
    top, H, n = 200, 360, 14
    tx, tw = 230, 270
    f.text(tx + tw / 2, 120, "KV 1줄 크기 (토큰당)", 21, "middle", 700)
    f.text(tx + tw / 2, 148, "K · V × 헤드 수 × 헤드 차원", 16, "middle", color=MUTED)
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

    stats(tx - 10, "80 KB (MHA 환산)", "S · 매번 전부 읽음")
    cx = 650
    f.text(cx + 105, 120, "MLA", 28, "middle", 700)
    f.text(cx + 105, 152, "가로(1줄 크기)를 줄인다", 18, "middle", color=MUTED)
    f.rect(cx, top, 210, H, "none", AXIS, 3, 1.5, "6 5")
    _rows(f, cx, top, 46, H, n, FULL_L, FULL)
    f.arrow(cx + 196, top + H / 2, cx + 60, top + H / 2, MUTED, 2.5)
    stats(cx, "1,152 B", "S · 매번 전부 읽음")
    dx = 960
    f.text(dx + 90, 120, "DSA", 28, "middle", 700)
    f.text(dx + 90, 152, "표는 그대로, 읽을 줄만 고른다", 18, "middle", color=MUTED)
    _rows(f, dx, top, 46, H, n, FULL_L, FULL, pick={1, 4, 5, 9, 12}, pick_fill=SEL)
    _rows(f, dx + 58, top, 14, H, n, SEL_L, SEL)
    f.text(dx + 84, top + 16, "색인 키", 16, color=MUTED)
    f.rect(dx + 84, top + H - 52, 16, 16, SEL, rx=3)
    f.text(dx + 108, top + H - 38, "이번 스텝에 읽는 줄", 16, color=MUTED)
    stats(dx, "1,152 B + 색인 132 B", "S · 2,048줄만 읽음")
    sx = 1260
    f.text(sx + 150, 120, "CSA · HCA", 28, "middle", 700)
    f.text(sx + 150, 152, "세로(줄 수)를 줄인다", 18, "middle", color=MUTED)
    for x, h_fill, col, edge, nm, ratio, note in ((sx, H / 4, CMP, CMP, "CSA", "4줄 → 1줄", "+ 색인 · top-k"),
                                                  (sx + 170, 10, CMP_L, CMP, "HCA", "128줄 → 1줄", "전부 읽음")):
        f.rect(x, top, 46, H, "none", AXIS, 3, 1.5, "6 5")
        k = 3 if nm == "CSA" else 1
        _rows(f, x, top, 46, h_fill, k, col, edge)
        f.arrow(x + 23, top + H - 12, x + 23, top + h_fill + 18, MUTED, 2.5)
        f.text(x + 58, top + 24, nm, 21, weight=700)
        f.text(x + 58, top + 50, ratio, 16, color=MUTED)
        f.text(x + 58, top + 74, note, 15, color=MUTED)
    stats(sx, "1,024 B", "S/4 · S/128")
    f.text(1570, 700, "그림의 비율은 실제와 다름 · 80 KB = V3 헤드 128 × (K 192 + V 128) × BF16", 15, "end", color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 5. 층 구성 띠
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
    f = Fig("05-layer-strips", 1600, 760)
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


# ---------------------------------------------------------------- 그림 6. KV cache 읽기 vs 크기 (층당, 128K) + 색인 비중
FAMILIES = [
    ("DeepSeek", [("전부 읽기 · MLA", "V3", "mla", FULL, FULL_L), ("선택 읽기 · DSA", "V3.2", "dsa", SEL, SEL_L),
                  ("압축 4줄 → 1 · CSA", "V4", "csa", CMP, CMP_L), ("압축 128줄 → 1 · HCA", "V4", "hca", CMP, CMP_L)], "mla"),
    ("Qwen", [("전부 읽기 · GQA", "Qwen3.8 full 층", "gqa", FULL, FULL_L),
              ("선택 읽기 · QSA", "Qwen3.8", "qsa", SEL, SEL_L),
              ("Linear state · DeltaNet", "Qwen3.8", "gdn", STATE, STATE_L)], "gqa"),
]


def fig_read_store():
    S = K128
    f = Fig("06-kv-read-vs-size", 1600, 760)
    basis(f, "층 1개 · 요청 1개 · 컨텍스트 128K · KV cache BF16 통일, 색인 키는 공개 저장 형식 (계산값)")
    pw = 260
    X1, X2 = 330, 830
    vmax = 0.3e9
    sc = pw / vmax
    f.text(X1, 100, "디코드 1스텝의 KV cache 읽기", 22, weight=700)
    f.text(X2, 100, "KV cache 크기", 22, weight=700)
    f.text(1310, 100, "읽기 중 색인 키 비중", 22, weight=700)
    for j, s in enumerate(("32K", "128K", "1M")):
        f.text(1330 + j * 80, 128, s, 16, "middle", 700 if s == "128K" else 400, color=INK if s == "128K" else MUTED)
    for X in (X1, X2):
        for v in (0, 0.1e9, 0.2e9, 0.3e9):
            f.line(X + v * sc, 140, X + v * sc, 705, GRID, 1)
            f.text(X + v * sc, 732, f"{v / 1e6:.0f} MB" if v else "0", 14, "middle", color=MUTED)
    y = 140
    for fam, items, base in FAMILIES:
        f.text(40, y + 6, fam, 17, weight=700, color=MUTED)
        y += 18
        base_read = total(read_parts(base, S))
        base_store = total(store_parts(base, S))
        for label, model, kind, dark, light in items:
            f.text(40, y + 30, label, 20, weight=700)
            f.text(40, y + 54, model, 15, color=MUTED)
            for X, parts, ref in ((X1, read_parts(kind, S), base_read), (X2, store_parts(kind, S), base_store)):
                idx, main = parts
                w_idx, w_main = idx * sc, max(main * sc, 3)
                if idx:
                    f.rect(X, y + 12, max(w_idx, 2), 36, light, rx=3)
                f.rect(X + w_idx + (2 if idx else 0), y + 12, w_main, 36, dark, rx=3)
                tot = idx + main
                lab = fmt_b(tot)
                if kind == "gdn":
                    lab += " + 쓰기 같은 양" if X == X1 else " (고정)"
                elif kind != base:
                    r = tot / ref
                    lab += f" ({'1/' + format(1 / r, '.0f') if r < 0.95 else '×' + format(r, '.2f')})"
                f.text(X + w_idx + w_main + 10, y + 37, lab, 17, weight=700 if X == X1 else 400)
            if kind in ("dsa", "qsa", "csa"):
                for j, s in enumerate((32768, K128, M)):
                    i_, m_ = read_parts(kind, s)
                    out = s > SUPPORT[kind]
                    f.text(1330 + j * 80, y + 37, f"{i_ / (i_ + m_) * 100:.0f}%" + ("*" if out else ""), 19, "middle",
                           700 if s == K128 else 400, INK if s == K128 else MUTED)
            y += 74
        y += 14
    f.text(1310, 690, "* 모델 지원 길이 밖 (외삽)", 14, color=MUTED)
    lx = 1010
    f.rect(lx, 22, 18, 18, SEL_L, rx=3)
    f.text(lx + 26, 38, "옅은 색 = 색인 키", 17, color=MUTED)
    f.rect(lx + 200, 22, 18, 18, SEL, rx=3)
    f.text(lx + 226, 38, "진한 색 = KV cache · state", 17, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 7. 컨텍스트 길이에 따른 KV cache 읽기 절감
CURVES = [("dsa", "mla", "DSA (V3.2)", SEL), ("qsa", "gqa", "QSA (Qwen3.8)", NAVY),
          ("csa", "mla", "CSA (V4)", CMP), ("hca", "mla", "HCA (V4)", RUST)]


def fig_gain():
    f = Fig("07-kv-read-saving-by-length", 1600, 780)
    basis(f, "층 1개 · 디코드 1스텝 · Full 대비 KV cache 읽기가 몇 분의 1인가 (계산값, 속도 배수 아님) · 점선 = 모델 지원 길이 밖")
    x0, x1, top, bot = 130, 1240, 90, 680
    lo, hi = 1024, 10 * M
    X = lambda s: x0 + (math.log10(s) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)
    ylo, yhi = 0.5, 300
    Y = lambda v: bot - (math.log10(max(min(v, yhi), ylo)) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo)) * (bot - top)
    for v, lab in ((0.5, "0.5×"), (1, "1×"), (3, "3×"), (10, "10×"), (30, "30×"), (100, "100×"), (300, "300×")):
        f.line(x0, Y(v), x1, Y(v), GRID, 1)
        f.text(x0 - 12, Y(v) + 6, lab, 16, "end", color=MUTED)
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
    f.text(X(1.6 * M), Y(8.7) + 66, "상한 = KV 1줄 ÷ 토큰당 색인 바이트", 16, "middle", color=MUTED, halo=True)
    f.text(X(1500), Y(150), "● = 모델 지원 길이 끝", 15, color=MUTED, halo=True)
    return f


# ---------------------------------------------------------------- 그림 8. 절감 상한을 정하는 요인
def fig_ceiling():
    f = Fig("08-saving-ceiling", 1600, 640)
    basis(f, "선택 · 압축형 KV cache 읽기 절감의 이론 상한 (컨텍스트 → ∞) · 상한 = KV 1줄 바이트 ÷ 토큰당 색인 바이트 (계산값)")
    rows = [
        ("DSA · 실제 FP8 KV 기준", "KV 656 B ÷ 색인 132 B", 656 / 132, SEL, False),
        ("DSA · V3.2 (BF16 KV 기준)", "KV 1,152 B ÷ 색인 132 B", 1152 / 132, SEL, False),
        ("DSA · 색인 키 FP4 (가정)", "KV 1,152 B ÷ 색인 66 B", 1152 / 66, SEL_L, True),
        ("QSA · 4줄 블록 색인", "KV 2,048 B ÷ (블록 키 256 B ÷ 4)", 2048 / 64, NAVY, False),
        ("CSA · 4줄 블록 + FP4 색인", "KV 1,152 B ÷ (색인 64 B ÷ 4)", 1152 / 16, CMP, False),
        ("HCA · 색인 없음, KV 1/128", "KV 1,152 B ÷ (엔트리 1,024 B ÷ 128)", 1152 / 8, RUST, False),
    ]
    x0, xw, vmax = 640, 760, 150
    X = lambda v: x0 + v / vmax * xw
    f.text(40, 100, "방식", 18, weight=700, color=MUTED)
    f.text(330, 100, "계산식 (토큰당)", 18, weight=700, color=MUTED)
    f.text(x0, 100, "절감 상한", 18, weight=700, color=MUTED)
    for v in (0, 50, 100, 150):
        f.line(X(v), 120, X(v), 590, GRID, 1)
        f.text(X(v), 615, f"{v}×", 15, "middle", color=MUTED)
    for i, (nm, formula, v, col, hyp) in enumerate(rows):
        y = 130 + i * 76
        f.text(40, y + 34, nm, 20, weight=700, color=MUTED if hyp else INK)
        f.text(330, y + 34, formula, 17, color=MUTED)
        f.rect(x0, y + 14, X(v) - x0, 32, col, SEL if hyp else "none", 4, 1.5, "5 4" if hyp else None)
        f.text(X(v) + 12, y + 37, f"{v:.1f}×" if v < 20 else f"{v:.0f}×", 21, weight=700)
    return f


# ---------------------------------------------------------------- 그림 9. 색인 연산 — 산술 강도와 prefill 연산 구성
def fig_intensity():
    f = Fig("09-indexer-intensity", 1600, 680)
    basis(f, "색인 키 1바이트당 연산 (decode, 층당) · prefill은 61층 합계, Full dense attention = 100% (계산값)")
    rows = [("QSA (Qwen3.8)", 4 * 128 * 2 / 256, NAVY, "head 4개 · BF16 블록 키"),
            ("DSA (V3.2)", 64 * 128 * 2 / 132, SEL, "head 64개 · FP8 키"),
            ("DSA + MTP 2토큰", 2 * 64 * 128 * 2 / 132, SEL_L, "쿼리 2개가 같은 키를 읽음"),
            ("CSA (V4)", 64 * 128 * 2 / 64, CMP, "head 64개 · FP4 키"),
            ("CSA + MTP 2토큰", 2 * 64 * 128 * 2 / 64, CMP_L, "쿼리 2개가 같은 키를 읽음")]
    x0, xw = 290, 520
    X = lambda v: x0 + math.log10(v) / 3 * xw
    f.text(40, 100, "색인 연산의 산술 강도 (FLOP/B)", 22, weight=700)
    f.text(850, 100, "HBM 속도로 훑을 때", 17, weight=700, color=MUTED)
    f.text(850, 122, "H200 FP8 피크 대비", 17, weight=700, color=MUTED)
    for v in (1, 10, 100, 1000):
        f.line(X(v), 140, X(v), 560, GRID, 1)
        f.text(X(v), 585, f"{v:,}", 15, "middle", color=MUTED)
    for v, nm in ((1979 / 4.8, "H200 FP8 경계 412"), (4500 / 8, "B200 FP8 경계 562")):
        f.line(X(v), 140, X(v), 560, INK, 1.8, "5 4")
        f.text(X(v) + (-6 if v < 500 else 6), 158, nm, 14, "end" if v < 500 else "start", color=INK, halo=True)
    for i, (nm, ai, col, sub) in enumerate(rows):
        y = 175 + i * 76
        f.text(40, y + 26, nm, 19, weight=700)
        f.text(40, y + 48, sub, 14, color=MUTED)
        f.rect(x0, y + 10, X(ai) - x0, 30, col, rx=4)
        f.text(X(ai) + 10, y + 32, f"{ai:.0f}", 19, weight=700, halo=True)
        need = ai * 4.8 / 1979 * 100
        f.text(850, y + 32, f"{need:.0f}%" + (" → 연산 병목" if need > 100 else ""), 20, weight=700,
               color=INK if need > 20 else MUTED)
    # 오른쪽 — prefill 연산 구성
    f.line(1080, 90, 1080, 640, GRID, 2, "6 6")
    f.text(1110, 100, "Prefill attention 연산 (DSA)", 22, weight=700)
    f.text(1110, 126, "Full dense attention = 100%", 16, color=MUTED)
    bx, bw, top, bot = 1150, 110, 170, 560
    Yp = lambda v: bot - v / 30 * (bot - top)
    for v in (0, 10, 20, 30):
        f.line(1120, Yp(v), 1540, Yp(v), GRID, 1)
        f.text(1112, Yp(v) + 5, f"{v}%", 14, "end", color=MUTED)
    dense_pair = 128 * (192 + 128) * 2
    idx_pair = 64 * 128 * 2
    for j, (S, lab) in enumerate(((K128, "128K"), (M, "1M*"))):
        x = bx + j * 200
        idx_share = idx_pair / dense_pair * 100
        sel_share = (S * TOPK) / (S * S / 2) * 100
        f.rect(x, Yp(idx_share), bw, Yp(0) - Yp(idx_share), SEL_L, rx=3)
        f.rect(x, Yp(idx_share + sel_share), bw, Yp(idx_share) - Yp(idx_share + sel_share) - 2, SEL, rx=3)
        f.text(x + bw / 2, (Yp(idx_share) + Yp(0)) / 2 + 6, f"색인 {idx_share:.0f}%", 17, "middle", 700)
        f.text(x + bw + 8, Yp(idx_share + sel_share / 2) + 5, f"선택 {sel_share:.1f}%", 15, color=MUTED)
        f.text(x + bw / 2, Yp(idx_share + sel_share) - 12, f"{100 / (idx_share + sel_share):.1f}배 감소", 17, "middle", 700)
        f.text(x + bw / 2, bot + 28, lab, 18, "middle", 700)
    f.text(1110, 612, "색인 연산 = dense attention의 1/5 → prefill 절감 상한 약 5배", 14, weight=700)
    f.text(1110, 634, "* 1M은 V3.2 지원 길이 밖 (외삽)", 14, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 10. KV 접근 패턴과 읽기 증폭
def fig_access():
    S = K128
    f = Fig("10-access-pattern", 1600, 560)
    basis(f, "층 1개 · 컨텍스트 128K · 디코드 1스텝 · KV BF16 · 조각 = 논리적 선택 구간 (물리 요청 수는 전송 단위 · 병합에 따라 다름)")
    x0, x1 = 300, 930
    cols = [(960, "조각 수"), (1070, "조각 크기"), (1180, "읽는 양"), (1300, "증폭 · HBM"), (1440, "증폭 · 4 KB 단위")]
    for x, h in cols:
        f.text(x, 100, h, 17, weight=700, color=MUTED)
    f.text(x0, 100, "KV cache (왼쪽 = 오래된 토큰, 오른쪽 = 최근 토큰)", 17, weight=700, color=MUTED)
    f.line(40, 116, 1560, 116, AXIS, 1)
    rnd = random.Random(11)
    n = 170
    cw = (x1 - x0) / n
    dsa = set(rnd.sample(range(n), 24))
    qsa = set()
    for st in rnd.sample(range(0, n - 4, 6), 8):
        qsa.update(range(st, st + 4))
    amp = lambda size, unit: math.ceil(size / unit) * unit / size
    hca_amt = total(read_parts("hca", S))
    rows = [
        ("DSA", "토큰 단위로 선택", dsa, SEL, "2,048", "1.2 KB", fmt_b(MLA_B * TOPK), amp(MLA_B, 64), amp(MLA_B, 4096),
         f"FP8 KV면 {amp(656, 4096):.1f}×"),
        ("QSA", "4줄 블록 단위로 선택", qsa, NAVY, "512", "8 KB", fmt_b(GQA_B * TOPK), amp(4 * GQA_B, 64), amp(4 * GQA_B, 4096), ""),
        ("HCA", "압축 KV 전체를 연속으로", set(range(n)), RUST, "연속", "—", fmt_b(hca_amt), 1.0, 1.0, ""),
    ]
    for i, (nm, sub, hit, col, pieces, size, amount, a_h, a_4k, note) in enumerate(rows):
        y = 140 + i * 128
        if i % 2 == 0:
            f.rect(30, y - 6, 1540, 116, BOX[0], rx=8)
        f.text(40, y + 46, nm, 24, weight=700)
        f.text(40, y + 76, sub, 16, color=MUTED)
        for k in range(n):
            on = k in hit
            f.rect(x0 + k * cw, y + (24 if on else 34), max(cw - 1, 1), 40 if on else 20, col if on else FULL_L, rx=1)
        f.text(960, y + 58, pieces, 22, weight=700)
        f.text(1070, y + 58, size, 20, weight=700)
        f.text(1180, y + 58, amount, 20)
        f.text(1300, y + 58, f"{a_h:.1f}×", 22, weight=700, color=MUTED)
        f.text(1440, y + 58, f"{a_4k:.1f}×", 22, weight=700, color=INK if a_4k > 1.5 else MUTED)
        if note:
            f.text(1440, y + 84, note, 14, color=MUTED)
    f.text(40, 535, "증폭 = 실제 전송 바이트 ÷ 필요한 바이트 (조각이 서로 붙어 있지 않다고 가정) · HBM 접근 단위 64 B, 대용량 계층 4 KB 페이지 가정", 15, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 11. Hybrid 모델 전체 합산 (Qwen3.8-Flash-Next)
def hybrid_parts(S):
    st = 36 * GDN_STATE
    qi, qk = read_parts("qsa", S)
    return st, st, 12 * qk, 12 * qi            # state 읽기, state 쓰기, QSA KV 읽기, QSA 색인 읽기


def fig_hybrid():
    f = Fig("11-hybrid-total", 1600, 720)
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
    # 오른쪽 상자 — 전 층 full GQA와 비교, 쓰기량
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
    f.text(bx, 545, f"→ 쓰기는 state가 약 {36 * GDN_STATE / kv_write:,.0f}배", 16, weight=700, color=INK)
    return f


# ---------------------------------------------------------------- 그림 12. 데이터별 메모리 접근 특성 (배포 정밀도, 128K)
def fig_behavior():
    S = K128
    kv = 61 * 656 * S
    idx = 61 * DSA_IDX * S
    sel = 61 * 656 * TOPK
    st = 36 * GDN_STATE
    w1, w51 = weight_read(1), weight_read(51)
    rows = [
        ("가중치", "V3.2 · FP8", "689 GB", (w1 / 689, f"{w1:.0f} GB ({w1 / 689 * 100:.0f}%)", f"배치 51이면 {w51:.0f} GB ({w51 / 689 * 100:.0f}%)"),
         ("0", "바뀌지 않음"), "모든 요청 공유", "HBM — 대역폭"),
        ("KV cache", "V3.2 · FP8 656 B/줄", fmt_b(kv), (sel / kv, f"{fmt_b(sel)} ({sel / kv * 100:.1f}%)", "선택된 2,048줄"),
         (fmt_b(61 * 656), "끝에 1줄 추가"), "요청마다 × N", "용량 계층 후보"),
        ("색인 키", "V3.2 · 132 B/줄", fmt_b(idx), (1, f"{fmt_b(idx)} (100%)", "매번 전부 훑음"),
         (fmt_b(61 * DSA_IDX), "끝에 1줄 추가"), "요청마다 × N", "HBM — 순차 스캔"),
        ("Linear state", "Qwen3.8 · 36층 FP32", fmt_b(st), (1, f"{fmt_b(st)} (100%)", "매번 전부"),
         (fmt_b(st) + " (100%)", "매번 전부 덮어씀"), "요청마다 × N", "온칩 + 쓰기 대역폭"),
    ]
    f = Fig("12-memory-behavior", 1600, 640)
    basis(f, "요청 1개 · 컨텍스트 128K · 디코드 1스텝 · 실제 배포 정밀도 (가중치 FP8, KV FP8, 색인 FP8 + scale, state FP32) · 계산값")
    cols = [(40, "데이터"), (290, "보유량"), (450, "매 스텝 읽기"), (880, "매 스텝 쓰기"), (1110, "공유 범위"), (1320, "어울리는 메모리")]
    for x, h in cols:
        f.text(x, 100, h, 19, weight=700, color=MUTED)
    f.line(40, 116, 1560, 116, AXIS, 1)
    for i, (nm, sub, size, (frac, rlab, rnote), (wlab, wnote), share, mem) in enumerate(rows):
        y = 140 + i * 118
        if i % 2 == 0:
            f.rect(30, y - 8, 1540, 108, BOX[0], rx=8)
        f.text(40, y + 36, nm, 23, weight=700)
        f.text(40, y + 66, sub, 15, color=MUTED)
        f.text(290, y + 46, size, 22, weight=700)
        f.rect(450, y + 22, 150, 24, "#e3e6ee", rx=4)
        f.rect(450, y + 22, max(150 * frac, 4), 24, SLATE, rx=4)
        f.text(612, y + 41, rlab, 18, weight=700)
        f.text(450, y + 76, rnote, 15, color=MUTED)
        f.text(880, y + 41, wlab, 18, weight=700)
        f.text(880, y + 76, wnote, 15, color=MUTED)
        f.text(1110, y + 46, share, 18, weight=700 if share != "모든 요청 공유" else 400)
        f.text(1320, y + 46, mem, 19, weight=700)
    return f


# ---------------------------------------------------------------- 그림 13. 접근 밀도 — 데이터 요구 vs 메모리 기술
MEMS = [("NVMe SSD", 14 / 8000, "Gen5 x4 · 8 TB 가정"), ("CXL 메모리", 64 / 512, "x16 64 GB/s · 512 GB 가정"),
        ("LPDDR5X", 512 / 480, "Grace 480 GB · 약 512 GB/s"), ("HBM3e", 4800 / 141, "H200 141 GB · 4.8 TB/s")]


def fig_density():
    tps = DEP["tps"]
    f = Fig("13-access-density", 1600, 700)
    basis(f, f"접근 밀도 = 보유 1 GB당 필요한 대역폭 (GB/s per GB) = 매 스텝 읽는 비율 × {tps:.0f} 토큰/s · 메모리 쪽 = 대역폭 ÷ 용량 (사양은 대표값)")
    x0, x1 = 420, 1480
    lo, hi = 0.001, 100
    X = lambda v: x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)
    top, bot = 210, 600
    for v in (0.001, 0.01, 0.1, 1, 10, 100):
        f.line(X(v), top - 10, X(v), bot, GRID, 1)
        f.text(X(v), bot + 26, f"{v:g}", 15, "middle", color=MUTED)
    f.text((x0 + x1) / 2, bot + 56, "GB/s per GB (로그)  ·  왼쪽일수록 드물게 읽음", 16, "middle", color=MUTED)
    for j, (nm, d, spec) in enumerate(MEMS):
        x = X(d)
        f.line(x, top - 10, x, bot, INK, 1.8, "5 4")
        ytxt = 104 + (j % 2) * 46
        f.text(x, ytxt, f"{nm} {d:.3g}" if d < 1 else f"{nm} {d:.0f}" if d > 10 else f"{nm} {d:.1f}", 17, "middle", 700)
        f.text(x, ytxt + 20, spec, 13, "middle", color=MUTED)
    w1, w64 = weight_read(1) / 689 * tps, weight_read(64) / 689 * tps
    data = [("Linear state (읽기 + 쓰기)", 2 * tps, None, STATE, "요청마다"),
            ("Full KV cache · 색인 키", tps, None, FULL, "요청마다"),
            ("가중치 (배치 1 → 64)", w1, w64, SLATE, "공유"),
            ("Sparse KV cache · 128K", TOPK / K128 * tps, None, SEL, "요청마다"),
            ("Sparse KV cache · 1M", TOPK / M * tps, None, SEL, "요청마다")]
    for i, (nm, v, v2, col, share) in enumerate(data):
        y = top + 20 + i * 78
        f.text(40, y + 6, nm, 19, weight=700)
        f.text(40, y + 30, share, 14, color=MUTED)
        if v2:
            f.line(X(v), y, X(v2), y, col, 8)
            f.text(X(v2) + 14, y + 6, f"{v:.1f} → {v2:.0f}", 17, weight=700, halo=True)
        else:
            f.dot(X(v), y, 10, col)
            f.text(X(v) + 16, y + 6, f"{v:.2g}" if v < 1 else f"{v:.0f}", 17, weight=700, halo=True)
    f.text(40, 680, "점이 어떤 메모리 선보다 왼쪽이면 그 메모리의 대역폭으로 감당 가능 (용량을 꽉 채워 써도)", 16, weight=700)
    return f


# ---------------------------------------------------------------- 그림 14. 동시 요청 수 × 컨텍스트 길이
MAP_LINES = [("Full attention · HBM만", "full", {}, FULL), ("DSA · HBM만", "dsa_hbm", {}, SEL),
             ("DSA + 대용량 계층 (이상적)", "dsa_tier", {}, NAVY),
             ("DSA + 대용량 계층 (4 KB 단위 읽기)", "dsa_tier", {"amp": 4096 / 656}, VIOLET)]


def fig_concurrency_map():
    f = Fig("14-concurrency-map", 1600, 740)
    basis(f, "H200 × 8 · DeepSeek-V3.2 실제 형식 (KV FP8) · HBM 10% 예비 · 20 토큰/s · 대용량 계층 512 GB/s · 요청은 GPU에 분산 (계산값) · 점선 = 지원 길이 밖")
    x0, x1, top, bot = 130, 1180, 90, 660
    lo, hi = 16384, M
    X = lambda s: x0 + (math.log10(s) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)
    ylo, yhi = 1, 1000
    Y = lambda v: bot - (math.log10(max(min(v, yhi), ylo)) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo)) * (bot - top)
    for v in (1, 3, 10, 30, 100, 300, 1000):
        f.line(x0, Y(v), x1, Y(v), GRID, 1)
        f.text(x0 - 12, Y(v) + 6, f"{v}", 15, "end", color=MUTED)
    f.text(x0 - 12, top - 20, "동시 요청 수", 15, "end", color=MUTED)
    for s, lab in ((16384, "16K"), (32768, "32K"), (K128, "128K"), (163840, "160K"), (524288, "512K"), (M, "1M")):
        f.line(X(s), top, X(s), bot, GRID if s != 163840 else AXIS, 1 if s != 163840 else 1.5)
        f.text(X(s), bot + 28, lab, 16, "middle", color=MUTED if s != 163840 else INK)
    f.text(X(163840) + 6, top + 18, "V3.2 지원 길이", 14, color=INK)
    f.text((x0 + x1) / 2, bot + 58, "요청당 컨텍스트 길이 (토큰)", 17, "middle", color=MUTED)
    pts = logspace(lo, hi, 60)
    ends = []
    for label, mode, over, col in MAP_LINES:
        vals = [(s, solve(s, mode, **over)) for s in pts]
        a = [(X(s), Y(max(n, 1))) for s, (n, _) in vals if s <= 163840]
        b = [(X(s), Y(max(n, 1))) for s, (n, _) in vals if s >= 163840 * 0.98]
        f.poly(a, col, 3.5)
        f.poly(b, col, 3.5, "9 7")
        for s_mark in (K128, M):
            n, why = solve(s_mark, mode, **over)
            f.dot(X(s_mark), Y(max(n, 1)), 6, col)
        ends.append((Y(max(vals[-1][1][0], 1)), label, vals[-1][1][0], col))
    ends.sort()
    used = []
    for yy, label, n, col in ends:
        y = yy
        for u in used:
            if abs(y - u) < 30:
                y = u + 30
        used.append(y)
        f.line(x1 + 4, yy, x1 + 22, y, AXIS, 1.2)
        f.text(x1 + 28, y + 6, f"{label}  {n}", 16, weight=700)
    # 계층 시나리오에서 먼저 막히는 곳이 바뀌는 지점
    prev = None
    for s_ in pts:
        n_, why_ = solve(s_, "dsa_tier")
        if prev == "시간" and why_ == "용량":
            f.dot(X(s_), Y(n_), 7, "#ffffff", NAVY)
            f.text(X(s_) - 10, Y(n_) - 40, "← 디코드 시간 예산", 15, "end", 700, NAVY, halo=True)
            f.text(X(s_) + 10, Y(n_) - 40, "HBM 색인 키 용량 →", 15, "start", 700, NAVY, halo=True)
            f.text(X(s_), Y(n_) - 18, f"약 {s_ // 1000}K에서 제약 전환", 14, "middle", color=MUTED, halo=True)
            break
        prev = why_
    return f


# ---------------------------------------------------------------- 그림 15. 병목 사다리 (1M)
LADDER = [("① HBM만 사용", "dsa_hbm", {}),
          ("② KV cache → 대용량 계층", "dsa_tier", {}),
          ("③ + 색인 키 FP4", "dsa_tier", {"idx": 66}),
          ("④ + 색인 1/4 블록 압축 (CSA형)", "dsa_tier", {"idx": 16.5}),
          ("⑤ + 계층 대역폭 2배", "dsa_tier", {"idx": 16.5, "tier_bw": 128})]
WHY = {"용량": "HBM 용량 (색인 키)", "시간": "디코드 시간 예산 (계층 읽기 직렬)"}


def fig_ladder():
    S = M
    f = Fig("15-bottleneck-ladder", 1600, 620)
    basis(f, "1M 차세대 시나리오 (V3.2 지원 160K 밖 — 외삽) · H200 × 8 · KV FP8 · HBM 10% 예비 · 20 토큰/s · 계층 512 GB/s (계산값)")
    x0, xw, vmax = 520, 640, 350
    X = lambda v: x0 + v / vmax * xw
    f.text(40, 100, "단계 (누적 적용)", 18, weight=700, color=MUTED)
    f.text(x0, 100, "동시 1M 요청 수", 18, weight=700, color=MUTED)
    f.text(1250, 100, "먼저 막히는 곳", 18, weight=700, color=MUTED)
    for v in (0, 100, 200, 300):
        f.line(X(v), 120, X(v), 560, GRID, 1)
        f.text(X(v), 585, f"{v}", 15, "middle", color=MUTED)
    for i, (nm, mode, over) in enumerate(LADDER):
        y = 130 + i * 84
        n, why = solve(S, mode, **over)
        if why == "용량" and mode == "dsa_hbm":
            wtxt = "HBM 용량 (KV + 색인)"
        else:
            wtxt = WHY.get(why, why)
        f.text(40, y + 36, nm, 20, weight=700)
        f.rect(x0, y + 14, max(X(n) - x0, 4), 36, SEL if i == 0 else NAVY, rx=4)
        f.text(X(n) + 12, y + 40, f"{n}개", 22, weight=700)
        f.text(1250, y + 40, wtxt, 18, color=INK)
    return f


# ---------------------------------------------------------------- 그림 16. 부분 최적화의 전체 효과 (Amdahl)
def fig_amdahl():
    f = Fig("16-partial-speedup", 1600, 680)
    basis(f, "디코드 1스텝 메모리 시간 중 attention 경로(색인 + 선택 KV)의 비중 p · 그 경로만 k배 빨라질 때 전체 = 1 ÷ ((1 − p) + p ÷ k) (계산값)")
    x0, x1, top, bot = 140, 1000, 100, 600
    pmax, vmax = 0.8, 5
    X = lambda p: x0 + p / pmax * (x1 - x0)
    Y = lambda v: bot - (v - 1) / (vmax - 1) * (bot - top)
    for v in (1, 2, 3, 4, 5):
        f.line(x0, Y(v), x1, Y(v), GRID, 1)
        f.text(x0 - 12, Y(v) + 6, f"{v}×", 15, "end", color=MUTED)
    for p in (0, 0.2, 0.4, 0.6, 0.8):
        f.line(X(p), top, X(p), bot, GRID, 1)
        f.text(X(p), bot + 28, f"{p * 100:.0f}%", 15, "middle", color=MUTED)
    f.text((x0 + x1) / 2, bot + 58, "attention 경로가 차지하는 시간 비중 p", 17, "middle", color=MUTED)
    for k, col, lab in ((2, SEL_L, "경로 2배"), (4, "#6da7ec", "경로 4배"), (10, SEL, "경로 10배"), (1e9, NAVY, "경로 무한대")):
        pts = [(X(i / 200 * pmax), Y(1 / ((1 - i / 200 * pmax) + (i / 200 * pmax) / k))) for i in range(0, 201)]
        f.poly(pts, col, 3)
        f.text(x1 + 12, pts[-1][1] + 6, lab, 16, weight=700)
    n1, _ = solve(K128, "dsa_hbm")
    w, i, s_ = step_times(K128, "dsa_hbm", n1)
    p1 = (i + s_) / (w + i + s_)
    n2, _ = solve(M, "dsa_tier")
    w2, i2, s2 = step_times(M, "dsa_tier", n2)
    p2 = (i2 + s2) / (w2 + i2 + s2)
    for p, tag in ((p1, "A"), (p2, "B")):
        v = 1 / ((1 - p) + p / 10)
        f.dot(X(p), Y(v), 13, INK)
        f.text(X(p), Y(v) + 6, tag, 15, "middle", 700, "#ffffff")
    bx = 1160
    f.rect(bx - 16, 100, 430, 520, BOX[0], BOX[1], 8, 1.2)
    for j, (tag, title, pv, n, comp) in enumerate((
            ("A", f"128K · HBM만 · 동시 {n1}개", p1, n1, f"가중치 {w:.1f} · 색인 {i:.1f} · 선택 KV {s_:.1f} ms"),
            ("B", f"1M · KV cache를 계층에 · 동시 {n2}개", p2, n2, f"가중치 {w2:.1f} · 색인 {i2:.1f} · 계층 KV {s2:.1f} ms"))):
        y = 150 + j * 190
        f.dot(bx + 14, y - 6, 13, INK)
        f.text(bx + 14, y, tag, 15, "middle", 700, "#ffffff")
        f.text(bx + 38, y + 1, title, 17, weight=700)
        f.text(bx, y + 44, f"p = {pv * 100:.0f}%", 24, weight=700)
        f.text(bx + 120, y + 44, f"→ 경로 10배 가속해도 전체 {1 / ((1 - pv) + pv / 10):.2f}×", 16)
        f.text(bx, y + 78, comp, 15, color=MUTED)
    f.text(bx, 540, "가중치 읽기 = 균등 routing의 expert 합집합", 14, color=MUTED)
    f.text(bx, 562, "FFN 연산 · 통신 미포함 (메모리 시간만)", 14, color=MUTED)
    f.text(bx, 584, "B는 V3.2 지원 길이 밖 (외삽)", 14, color=MUTED)
    return f


# ---------------------------------------------------------------- 그림 17. 단계별 특성 · 검증할 제약
def fig_stage():
    S = K128
    f = Fig("17-sparse-stage-summary", 1600, 760)
    basis(f, "DSA(V3.2) 디코드 1스텝 · 층 1개 · 컨텍스트 128K · 실제 배포 정밀도 (KV FP8 656 B, 색인 132 B) · 계산값")
    cols = [(40, "단계"), (270, "읽기 · 연산 (128K)"), (640, "예상 제약"), (950, "검증 지표"), (1200, "관련 HW · 시스템 요소")]
    for x, h in cols:
        f.text(x, 100, h, 18, weight=700, color=MUTED)
    f.line(40, 116, 1560, 116, AXIS, 1)
    idx_b = DSA_IDX * S
    idx_fl = 64 * 128 * 2 * S
    att_fl = 128 * (2 * 512 + 64) * 2 * TOPK
    rows = [
        ("1", "색인 키 훑기", True, (f"{fmt_b(idx_b)} · {idx_fl / 1e9:.1f} GFLOP", "FP8 · 124 FLOP/B · 1M이면 132 MB · 16 GFLOP"),
         ("HBM 대역폭 + FP8 연산", "HBM 속도로 훑으려면 H200 피크의 약 30%"), ("단계 시간 · 달성 대역폭", "연산기 사용률"),
         ("HBM 대역폭 · FP8 연산기", "메모리 근처 처리는 연산 밀도가 관건")),
        ("2", "top-k 선택", True, (f"점수 {S // 1000}K개 ({fmt_b(4 * S)})", "비교 · 정렬 — FLOPs에 안 잡힘"),
         ("비교 처리 · 동기화", "행렬 연산기로 빨라지지 않음"), ("top-k 시간 비중", ""), ("온칩 SRAM · reduction", "스레드 간 동기화")),
        ("3", "KV gather", True, (f"{fmt_b(656 * TOPK)} · 2,048조각", "HBM에서는 읽기 증폭 ≈ 1.0"),
         ("지연 · 의존성", "top-k 이후 직렬 · 계층에서는 증폭"), ("달성 대역폭 · 메모리 요청 수", ""),
         ("메모리 동시 요청 수 · DMA", "paged KV 주소 변환 · FP8 역양자화")),
        ("4", "attention", False, (f"{att_fl / 1e9:.2f} GFLOP", "선택된 2,048줄 · dense"),
         (f"작음 — 색인 연산의 약 1/{idx_fl / att_fl:.0f}", ""), ("—", ""), ("텐서 코어", "FP8 역양자화 처리량")),
    ]
    y = 132
    for num, step, hot, (a1, a2), (b1, b2), (c1, c2), (d1, d2) in rows:
        f.rect(30, y, 1540, 104, BOX[0] if int(num) % 2 else "#ffffff", rx=8)
        f.rect(44, y + 30, 40, 40, SEL if hot else FULL, rx=8)
        f.text(64, y + 58, num, 20, "middle", 700, "#ffffff")
        f.text(100, y + 58, step, 21, weight=700)
        for x, t1, t2 in ((270, a1, a2), (640, b1, b2), (950, c1, c2), (1200, d1, d2)):
            f.text(x, y + 46, t1, 17, weight=700 if hot else 400, color=INK if hot else MUTED)
            if t2:
                f.text(x, y + 76, t2, 14, color=MUTED)
        y += 112
    y += 14
    f.line(40, y - 8, 1560, y - 8, AXIS, 1.2, "6 5")
    f.rect(30, y + 4, 1540, 104, BOX[0], rx=8)
    f.rect(44, y + 34, 40, 40, SEL, rx=8)
    f.text(64, y + 61, "+", 22, "middle", 700, "#ffffff")
    f.text(100, y + 62, "KV cache 보관", 21, weight=700)
    kv_req, idx_req = 61 * 656 * S, 61 * DSA_IDX * S
    for x, t1, t2 in ((270, f"요청당 KV {fmt_b(kv_req)} + 색인 {fmt_b(idx_req)}", "× 동시 요청 수 (61층)"),
                      (640, "HBM 용량", "분산 방식(복제 여부)에 크게 좌우"), (950, "동시 요청 수", "HBM 사용량"),
                      (1200, "HBM 용량 · 메모리 계층", "계층 간 대역폭 · KV 할당 관리")):
        f.text(x, y + 50, t1, 17, weight=700)
        f.text(x, y + 80, t2, 14, color=MUTED)
    return f


FIGS = [fig_trend, fig_weights_kv, fig_moe, fig_axes, fig_layers, fig_read_store, fig_gain, fig_ceiling, fig_intensity,
        fig_access, fig_hybrid, fig_behavior, fig_density, fig_concurrency_map, fig_ladder, fig_amdahl, fig_stage]
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


SENSITIVITY = [("기준 (KV FP8 · 예비 10% · 20 토큰/s · 증폭 1 · 요청 분산)", {}),
               ("KV BF16 (1,152 B/줄)", {"kv": 1152}),
               ("HBM 예비 0%", {"reserve": 0.0}),
               ("HBM 예비 20%", {"reserve": 0.20}),
               ("목표 50 토큰/s", {"tps": 50}),
               ("계층 읽기 증폭 4×", {"amp": 4.0}),
               ("계층 4 KB 단위 읽기 (증폭 6.2×)", {"amp": 4096 / 656}),
               ("계층 링크 실효 50%", {"link_eff": 0.5}),
               ("색인 scale 제외 (128 B)", {"idx": 128}),
               ("KV · 색인을 GPU마다 복제 (TP 참조 구현)", {"replicate": True})]


def print_summary():
    S = K128
    print("층당 · 128K · 1스텝 KV cache 읽기 / 크기 (BF16 KV)")
    for k in ("mla", "dsa", "csa", "hca", "gqa", "qsa", "gdn"):
        r, s = read_parts(k, S), store_parts(k, S)
        print(f"  {k:4s} read {fmt_b(sum(r)):>9s} (idx {r[0] / sum(r) * 100:4.0f}%) | store {fmt_b(sum(s)):>9s}")
    v4_1m = 30 * total(read_parts("csa", M)) + 31 * total(read_parts("hca", M))
    v4_128 = 30 * total(read_parts("csa", S)) + 31 * total(read_parts("hca", S))
    print(f"  V4-Pro 전체 vs MLA: 128K 1/{61 * MLA_B * S / v4_128:.0f} · 1M 1/{61 * MLA_B * M / v4_1m:.0f}")
    print("하이브리드 Qwen3.8 (state 비중):", {lab: f"{(p[0] + p[1]) / sum(p) * 100:.0f}% {fmt_b(sum(p))}"
                                         for lab, p in (("32K", hybrid_parts(32768)), ("128K", hybrid_parts(S)), ("1M", hybrid_parts(M)))})
    print("동시 요청 수 (128K | 1M):")
    for label, mode, over, _ in MAP_LINES:
        print(f"  {label:34s} {solve(S, mode, **over)} | {solve(M, mode, **over)}")
    print("민감도 (DSA · HBM만 | DSA + 계층) @128K , @1M:")
    for label, over in SENSITIVITY:
        r = [solve(s, m, **over) for s in (S, M) for m in ("dsa_hbm", "dsa_tier")]
        print(f"  {label:44s} 128K {r[0]} {r[1]} | 1M {r[2]} {r[3]}")
    print("사다리 (1M):", [(nm, solve(M, mode, **over)) for nm, mode, over in LADDER])
    n2, _ = solve(M, "dsa_tier")
    print("1M 계층 시간 구성 (ms):", [round(x, 1) for x in step_times(M, "dsa_tier", n2)], "n", n2)


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
