"""MiniMax M2.5 · M3 구조 스터디용 그림 생성기. 실행: python make_figures.py [그림 이름 ...]  → 같은 폴더에 *.svg, *.png

수치의 근거
- MiniMax-M2.5 : HF MiniMaxAI/MiniMax-M2.5 config.json · modeling_minimax_m2.py · safetensors 헤더
                 (62층, d 3072, 쿼리 48 · KV 8 · 헤드 128, expert 256 top-8 폭 1536, qk_norm per_layer, rotary 64)
- MiniMax-M3   : HF MiniMaxAI/MiniMax-M3 config.json(text_config) · safetensors 헤더,
                 transformers models/minimax_m3_vl/modular_minimax_m3_vl.py (MiniMaxM3VLIndexer, SparseMoeBlock)
                 (60층, d 6144, 쿼리 64 · KV 4 · 헤드 128, 0~2층 dense FFN 12288, 3~59층 MoE 128 top-4 + shared 1 · MSA)
- MSA          : sparse_block_size 128, sparse_topk_blocks 16, sparse_local_block 1, index head 4 × 128, 색인 키 1 × 128
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "'Noto Sans KR','Malgun Gothic',sans-serif"
MONO = "Consolas,'Courier New',monospace"

INK = "#343a4d"
MUTED = "#7b8196"
PURPLE = "#6b4fc8"
KINDS = {  # fill, stroke
    "q": ("#eceafa", "#6b4fc8"),
    "k": ("#f8e6f1", "#c2479c"),
    "v": ("#e4f0e4", "#4d9a56"),
    "c": ("#fdefd8", "#d18a1c"),
    "i": ("#e2eefc", "#3b7dd8"),
    "n": ("#eff0f6", "#d3d6e2"),
}
SOLID = {"q": "#6b4fc8", "k": "#c2479c", "v": "#4d9a56", "c": "#e0a030", "i": "#3b7dd8", "n": "#c9cddb"}
CHIP = {"g": "#4d9a56", "r": "#c2479c", "n": "#7b8196"}
# 층 배치 색 — attention 종류, FFN 종류
LAYER = {"full": "#8f79d8", "msa": SOLID["c"], "moe": "#4d9a56", "dense": "#9aa0b4"}


class Fig:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.o = name, w, h, []

    def raw(self, s):
        self.o.append(s)

    def rect(self, x, y, w, h, fill, stroke="none", rx=6, sw=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=18, anchor="middle", weight=400, color=INK, mono=False):
        f = MONO if mono else FONT
        self.raw(f'<text x="{x}" y="{y}" font-family="{f}" font-size="{size}" text-anchor="{anchor}" '
                 f'font-weight="{weight}" fill="{color}">{s}</text>')

    def title(self, x, y, s, size=24, anchor="start"):
        self.text(x, y, s, size, anchor, 700, PURPLE)

    def line(self, x1, y1, x2, y2, color="#8b90a3", sw=2, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#ah)"' if arrow else ""
        self.raw(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def arrow(self, x1, y1, x2, y2, dash=None, color="#5d6377"):
        self.line(x1, y1, x2, y2, color, 2.2, dash, True)

    def path(self, d, dash=None, arrow=True, color="#5d6377"):
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#ah)"' if arrow else ""
        self.raw(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"{ds}{m}/>')

    def svg(self):
        defs = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
                'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#5d6377"/></marker></defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}">{defs}<rect width="100%" height="100%" fill="#ffffff"/>' + "".join(self.o) + "</svg>")


def _box(f, x, y, w, h, kind, label="", dims=None, sub=None, solid=False, fs=16):
    fill, stroke = KINDS[kind]
    f.rect(x, y, w, h, SOLID[kind] if solid else fill, stroke, 5, 1.8)
    if label:
        f.text(x + w / 2, y + h / 2 + fs * 0.36, label, fs, mono=True, color="#ffffff" if solid else INK)
    if dims:
        f.text(x + w / 2, y + h + 22, dims, 14, color=MUTED, mono=True)
    if sub:
        f.text(x + w / 2, y + h + 42, sub, 13, color=MUTED)


def _chip(f, cx, y, text, kind):
    w = sum(13 if ord(ch) > 0x2000 else 7.5 for ch in text) + 22
    f.rect(cx - w / 2, y, w, 24, CHIP[kind], "none", 12)
    f.text(cx, y + 17, text, 13, weight=700, color="#ffffff")


def _op(f, x1, x2, y, label, chip=None, kind="g"):
    """화살표 하나와 그 위의 연산 이름, 아래의 연산 종류."""
    f.arrow(x1, y, x2, y)
    if label:
        f.text((x1 + x2) / 2, y - 12, label, 14, weight=700)
    if chip:
        _chip(f, (x1 + x2) / 2, y + 12, chip, kind)


def _legend(f, x, y):
    for i, (t, k) in enumerate((("행렬곱", "g"), ("원소별 연산", "n"), ("행렬곱이 아닌 일", "r"))):
        f.rect(x + i * 190, y - 13, 16, 16, CHIP[k], "none", 4)
        f.text(x + i * 190 + 24, y, t, 15, "start", 600)


def _section(f, y, name, *notes):
    f.text(40, y, name, 20, "start", 700)
    for i, n in enumerate(notes):
        f.text(40, y + 24 + i * 20, n, 15, "start", color=MUTED)


def _divider(f, y):
    f.line(40, y, 1520, y, "#e3e5ee", 2, "6 6")


def _summary(f, y, bold, muted=None, h=56):
    f.rect(40, y, 1480, h, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(64, y + 34 if not muted else y + 30, bold, 17, "start", 700)
    if muted:
        f.text(64, y + 54, muted, 15, "start", color=MUTED)


# ---------------------------------------------------------------- 1. 층 배치
def fig_layers():
    f = Fig("01-layer-map", 1560, 470)
    f.title(40, 44, "층 배치 — config.json의 attn_type_list · moe_layer_freq · sparse_attention_freq 그대로")

    def row(y, name, attn, ffn, note):
        f.text(40, y + 30, name, 19, "start", 700)
        f.text(206, y + 17, "attn", 13, "end", color=MUTED, mono=True)
        f.text(206, y + 47, "FFN", 13, "end", color=MUTED, mono=True)
        for i, (a, m) in enumerate(zip(attn, ffn)):
            x = 220 + i * 21
            f.rect(x, y, 18, 26, LAYER[a], "none", 3)
            f.rect(x, y + 30, 18, 26, LAYER[m], "none", 3)
        f.text(229, y + 80, "0", 14, color=MUTED, mono=True)
        f.text(220 + (len(attn) - 1) * 21 + 9, y + 80, str(len(attn) - 1), 14, color=MUTED, mono=True)
        f.text(220, y + 106, note, 16, "start", color=MUTED)

    row(84, "M2.5 · 62층", ["full"] * 62, ["moe"] * 62,
        "62층 전부 같은 층 — GQA full attention + MoE (expert 256개 중 8개)")
    row(244, "M3 · 60층", ["full"] * 3 + ["msa"] * 57, ["dense"] * 3 + ["moe"] * 57,
        "앞 3층만 full attention + dense FFN · 나머지 57층은 MSA(블록 선택) + MoE (expert 128개 중 4개 + shared 1개)")
    lx = 220
    for k, label in (("full", "full attention (컨텍스트 전부 읽는다)"), ("msa", "MSA (128토큰 블록 16개만 읽는다)"),
                     ("moe", "MoE FFN"), ("dense", "dense FFN")):
        f.rect(lx, 412, 18, 24, LAYER[k], "none", 3)
        f.text(lx + 28, 430, label, 16, "start")
        lx += {"full": 370, "msa": 370, "moe": 170}.get(k, 0)
    return f


# ---------------------------------------------------------------- 2. decoder 층 하나
def _block_column(f, cx, title, rows, side):
    """층 하나를 세로로 그린다. rows: (y, h, kind, label, sub, residual_add)"""
    f.text(cx, 100, title, 21, weight=700)
    w = 420
    x = cx - w / 2
    f.rect(cx - 80, 124, 160, 36, KINDS["n"][0], KINDS["n"][1], 5, 1.8)
    f.text(cx, 148, "h  1 × " + side["d"], 15, mono=True)
    prev = 160
    res_from = 142
    for y, h, kind, label, sub, add in rows:
        if add:
            f.arrow(cx, prev, cx, y - 2)
            f.raw(f'<circle cx="{cx}" cy="{y + 14}" r="14" fill="#fff" stroke="#5d6377" stroke-width="2"/>')
            f.text(cx, y + 21, "+", 22, weight=700)
            # residual
            rx = x - 30
            f.path(f"M{cx - 80},{res_from} L{rx},{res_from} L{rx},{y + 14} L{cx - 16},{y + 14}", dash="5 5")
            res_from = y + 14
            prev = y + 28
            continue
        f.arrow(cx, prev, cx, y - 2)
        bw = w if h > 40 else 200
        fill, stroke = KINDS[kind]
        f.rect(cx - bw / 2, y, bw, h, fill, stroke, 6, 2)
        if h > 40:
            f.text(cx, y + 30, label, 18, weight=700)
            for i, s in enumerate(sub):
                f.text(cx, y + 56 + i * 22, s, 14, color=MUTED, mono=i == 0)
        else:
            f.text(cx, y + h / 2 + 6, label, 15, mono=True)
        prev = y + h
    f.arrow(cx, prev, cx, prev + 34)
    f.text(cx, prev + 56, "다음 층으로  1 × " + side["d"], 15, color=MUTED, mono=True)


def fig_block():
    f = Fig("02-decoder-layer", 1560, 880)
    f.title(40, 44, "decoder 층 하나 — pre-norm, residual 두 번 (두 모델 모두 같은 골격)")
    m25 = [(186, 34, "n", "RMSNorm (× w)", None, False),
           (246, 118, "q", "GQA attention", ["쿼리 48 · KV 8 · 헤드 128",
                                              "QK-norm: 층 단위 (헤드 48개를 한 벡터로)",
                                              "RoPE: 128차원 중 앞 64"], False),
           (392, 0, None, None, None, True),
           (446, 34, "n", "RMSNorm (× w)", None, False),
           (506, 118, "v", "MoE", ["expert 256 → top-8 · 폭 1,536",
                                   "sigmoid 점수 + bias로 고른다",
                                   "shared expert 없음 · SwiGLU(SiLU)"], False),
           (652, 0, None, None, None, True)]
    m3 = [(186, 34, "n", "RMSNorm (× (1+w))", None, False),
          (246, 118, "c", "MSA  (3~59층)", ["쿼리 64 · KV 4 · 헤드 128",
                                            "indexer가 128토큰 블록 16개를 고른다",
                                            "QK-norm: 헤드 단위 · RoPE 앞 64"], False),
          (392, 0, None, None, None, True),
          (446, 34, "n", "RMSNorm (× (1+w))", None, False),
          (506, 118, "v", "MoE  (3~59층)", ["expert 128 → top-4 · 폭 3,072",
                                            "+ shared expert 1개 (항상) · routed × 2.0",
                                            "SwiGLU(OAI식: clamp 7, α 1.702)"], False),
          (652, 0, None, None, None, True)]
    _block_column(f, 400, "M2.5 — 62층 모두", m25, {"d": "3072"})
    _block_column(f, 1160, "M3 — 3~59층 (57층)", m3, {"d": "6144"})
    f.text(1160, 790, "0~2층은 MSA 대신 full attention, MoE 대신 dense FFN (폭 12,288)", 15, color=MUTED)
    f.line(780, 90, 780, 760, "#e3e5ee", 2, "6 6")
    _summary(f, 810, "M3는 d를 2배(3072 → 6144)로 키우고, expert는 절반 수 · 2배 폭으로 바꾸고, attention 읽기를 블록 선택으로 줄였다.")
    return f


# ---------------------------------------------------------------- 3. M2.5 attention 연산
def fig_m25_attn():
    f = Fig("03-m25-attention-ops", 1560, 920)
    f.title(40, 44, "M2.5 attention의 텐서 연산 — 층 하나, 토큰 하나를 만들 때")
    _legend(f, 960, 44)

    # ---- ① projection
    y = 96
    _section(f, y + 22, "① projection", "가중치를 읽는다", "S와 무관")
    _box(f, 200, y + 20, 120, 24, "n", dims="x  1 × 3072")
    for i, (lbl, w, kind, dims) in enumerate((("q", 300, "q", "× W_Q  →  1 × 6144 (48 × 128)"),
                                               ("k", 100, "k", "× W_K  →  1 × 1024 (8 × 128)"),
                                               ("v", 100, "v", "× W_V  →  1 × 1024 (8 × 128)"))):
        yy = y + i * 60 - 8
        f.path(f"M326,{y + 32} C390,{y + 32} 390,{yy + 14} 462,{yy + 14}")
        _box(f, 470, yy, w, 28, kind, lbl, fs=15)
        f.text(470 + w + 14, yy + 19, dims, 14, "start", color=MUTED, mono=True)
    _chip(f, 388, y + 160, "행렬 × 벡터", "g")
    f.text(1040, y + 78, "q 폭(6,144)이 d(3,072)의 두 배다 — 헤드 수 × 헤드 차원이 d보다 크다", 15, "start", color=MUTED)
    f.text(1040, y + 102, "가중치: W_Q 3072×6144 · W_K, W_V 3072×1024 · W_O 6144×3072", 15, "start", color=MUTED, mono=False)
    f.text(1040, y + 126, "층당 44.0M 파라미터 (FP8 체크포인트)", 15, "start", color=MUTED)
    _divider(f, 290)

    # ---- ② QK-norm · RoPE
    y = 326
    _section(f, y + 26, "② QK-norm", "RoPE까지", "원소별")
    _box(f, 200, y, 300, 40, "q", "q  1 × 6144", fs=15)
    _op(f, 508, 680, y + 20, "RMSNorm(6144 전체)", "원소별 + 합 1번", "n")
    _box(f, 688, y, 300, 40, "q", "q̂  1 × 6144", fs=15)
    _op(f, 996, 1120, y + 20, "48 × 128로", None)
    f.text(1058, y + 52, "reshape", 13, color=MUTED, mono=True)
    # 헤드 하나: 앞 64는 회전, 뒤 64는 통과
    f.rect(1128, y, 120, 40, SOLID["q"], "none", 5)
    f.rect(1248, y, 120, 40, KINDS["q"][0], KINDS["q"][1], 5, 1.8)
    f.text(1188, y + 26, "RoPE 64", 15, mono=True, color="#ffffff")
    f.text(1308, y + 26, "통과 64", 15, mono=True)
    f.text(1248, y + 64, "헤드마다 앞 64차원만 회전 (θ = 5,000,000)", 14, color=MUTED)
    f.text(200, y + 88, "k도 같다: RMSNorm(1024 전체) → 8 × 128 → 앞 64만 RoPE", 15, "start", color=MUTED)
    f.text(200, y + 112, "norm을 헤드마다가 아니라 층 전체 벡터에 건다 — 헤드를 TP로 나누면 제곱합을 GPU끼리 all-reduce 해야 한다",
           15, "start", 700, "#c2479c")
    _divider(f, 470)

    # ---- ③ 캐시 쓰기
    y = 504
    _section(f, y + 22, "③ 쓰기", "한 줄 붙인다")
    _box(f, 200, y + 6, 100, 28, "k", "k̂", fs=15)
    _box(f, 320, y + 6, 100, 28, "v", "v", fs=15)
    _op(f, 428, 560, y + 20, "끝에 붙인다", "쓰기", "r")
    for i, kind in enumerate(("k", "v")):
        x0 = 568 + i * 330
        f.rect(x0, y - 8, 300, 56, KINDS[kind][0], KINDS[kind][1], 5, 1.8)
        for r in range(1, 4):
            f.line(x0 + 1, y - 8 + r * 12, x0 + 299, y - 8 + r * 12, KINDS[kind][1], 0.8)
        f.rect(x0 + 1, y + 37, 298, 10, SOLID[kind], "none", 0)
        f.text(x0 + 150, y + 70, ("K" if kind == "k" else "V") + " 캐시  8 × S × 128", 14, color=MUTED, mono=True)
    f.text(1240, y + 8, "토큰당 2 × 8 × 128 × 2 B", 15, "start", color=MUTED)
    f.text(1240, y + 32, "= 4,096 B · 62층이면 248 KiB", 15, "start", color=MUTED)
    _divider(f, 604)

    # ---- ④ 읽기
    y = 640
    _section(f, y + 26, "④ 읽기", "캐시 전체를", "두 번 읽는다")
    _box(f, 200, y, 90, 64, "q", "q", "8 × 6 × 128", "KV 헤드별로 6개씩")
    f.text(308, y + 40, "×", 26)
    _box(f, 326, y + 4, 200, 56, "k", "Kᵀ", "8 × 128 × S", "K 캐시 전체")
    f.text(544, y + 40, "=", 26)
    _box(f, 562, y, 170, 64, "q", "점수", "8 × 6 × S")
    _op(f, 740, 840, y + 32, "softmax", "원소별", "n")
    _box(f, 848, y, 170, 64, "q", "p", "8 × 6 × S", "/√128 후 S 방향")
    f.text(1036, y + 40, "×", 26)
    _box(f, 1054, y + 4, 130, 56, "v", "V", "8 × S × 128", "V 캐시 전체")
    _op(f, 1192, 1272, y + 32, "× W_O", None)
    _box(f, 1280, y + 12, 110, 40, "n", "y", "1 × 3072")
    _chip(f, 308, y + 124, "행렬 × 행렬", "g")
    _chip(f, 1036, y + 124, "행렬 × 행렬", "g")
    _summary(f, 820, "48헤드가 8개 KV 헤드를 6개씩 나눠 읽는다. 읽는 양은 KV 헤드 수 × S — 컨텍스트에 그대로 비례한다.",
             "128K에서 토큰 하나당 62층 × 4,096 B × 131,072 ≈ 33.3 GB를 읽는다", 76)
    return f


# ---------------------------------------------------------------- 4. M3 MSA 연산
def fig_m3_msa():
    f = Fig("04-m3-msa-ops", 1560, 1140)
    f.title(40, 44, "M3 MSA의 텐서 연산 — sparse 층 하나, 토큰 하나를 만들 때")
    _legend(f, 960, 44)

    # ---- 쓰기
    y = 96
    _section(f, y + 22, "쓰기", "토큰마다", "캐시 두 개")
    _box(f, 180, y + 40, 120, 24, "n", dims="x  1 × 6144")
    # 본 KV
    f.path(f"M306,{y + 52} C350,{y + 52} 350,{y + 14} 400,{y + 14}")
    _box(f, 408, y, 90, 28, "k", "k̂", fs=15)
    _box(f, 506, y, 90, 28, "v", "v", fs=15)
    f.text(450, y - 10, "4 × 128 씩", 13, color=MUTED, mono=True)
    _op(f, 604, 700, y + 14, "", None)
    f.rect(708, y - 6, 300, 40, KINDS["k"][0], KINDS["k"][1], 5, 1.8)
    f.text(858, y + 20, "K, V 캐시  4 × S × 128 각각", 14, mono=True)
    # 색인 키
    f.path(f"M306,{y + 52} C350,{y + 52} 350,{y + 92} 400,{y + 92}")
    f.text(354, y + 120, "× W_ik · norm · RoPE", 13, "start", color=MUTED, mono=True)
    _box(f, 408, y + 78, 48, 28, "i", solid=True)
    f.text(468, y + 98, "k_I  1 × 128", 14, "start", color=MUTED, mono=True)
    _op(f, 604, 700, y + 92, "", None)
    f.rect(708, y + 70, 300, 44, KINDS["i"][0], KINDS["i"][1], 5, 1.8)
    for r in range(1, 3):
        f.line(709, y + 70 + r * 12, 1007, y + 70 + r * 12, KINDS["i"][1], 0.8)
    f.rect(709, y + 104, 298, 9, SOLID["i"], "none", 0)
    f.text(858, y + 136, "색인 키 캐시  S × 128 · BF16", 14, color=MUTED, mono=True)
    f.text(1050, y + 18, "본 K, V: 토큰당 2 × 4 × 128 × 2 B = 2,048 B", 15, "start", color=MUTED)
    f.text(1050, y + 42, "색인 키: 헤드 하나뿐 — 토큰당 256 B (value 없음)", 15, "start", color=MUTED)
    f.text(1050, y + 66, "sparse 층(57층)에만 있다", 15, "start", color=MUTED)
    _divider(f, 270)

    # ---- ① 점수
    y = 306
    _section(f, y + 26, "① 점수", "색인 키 전체를", "읽는다")
    _box(f, 200, y, 56, 64, "q", "q_I", "4 × 128", "index 헤드 4개")
    f.text(276, y + 40, "×", 26)
    _box(f, 300, y + 4, 280, 56, "i", "K_Iᵀ", "128 × S", "색인 키 캐시 전체")
    f.text(600, y + 40, "=", 26)
    _box(f, 624, y, 280, 64, "q", "q_I · K_Iᵀ", "4 × S", "미래 토큰은 −∞")
    _chip(f, 276, y + 124, "행렬 × 행렬", "g")
    f.text(960, y + 26, "index 헤드 수 = KV 헤드 수 = 4", 15, "start", 700)
    f.text(960, y + 50, "헤드 h의 점수로 KV 헤드 h가 읽을 곳을 고른다", 15, "start", color=MUTED)
    f.text(960, y + 74, "ReLU · 헤드 가중치 · 헤드 합이 없다 (DSA와 다르다)", 15, "start", color=MUTED)
    _divider(f, 474)

    # ---- ② 블록 점수
    y = 510
    _section(f, y + 26, "② 블록 점수", "128토큰마다", "최댓값 하나")
    _box(f, 200, y, 280, 64, "q", "4 × S", None)
    _op(f, 488, 600, y + 32, "128씩 묶어", None)
    f.text(544, y + 64, "reshape", 13, color=MUTED, mono=True)
    # 4 × (S/128) × 128 그림
    for b in range(6):
        f.rect(608 + b * 50, y + 4, 44, 56, KINDS["q"][0], KINDS["q"][1], 4, 1.5)
        f.text(630 + b * 50, y + 38, "128", 12, color=MUTED, mono=True)
    f.text(758, y + 86, "4 × S/128 × 128", 14, color=MUTED, mono=True)
    _op(f, 916, 1036, y + 32, "max", "원소별 (블록 안)", "n")
    hot = ["#d9d2f3", "#6b4fc8", "#eceafa", "#b1a2e6", "#eceafa", "#8f79d8"]
    for b, c in enumerate(hot):
        f.rect(1044 + b * 32, y + 18, 28, 28, c, "none", 3)
    f.text(1136, y + 72, "4 × S/128", 14, color=MUTED, mono=True)
    f.text(1260, y + 26, "블록 안 어느 한 토큰이라도", 15, "start", color=MUTED)
    f.text(1260, y + 50, "점수가 높으면 블록이 뽑힌다", 15, "start", color=MUTED)
    _divider(f, 640)

    # ---- ③ 고르기
    y = 676
    _section(f, y + 26, "③ 고르기", "헤드마다", "블록 16개")
    for b, c in enumerate(hot):
        f.rect(200 + b * 32, y + 18, 28, 28, c, "none", 3)
    f.rect(200 + 6 * 32, y + 18, 28, 28, "#fff", SOLID["k"], 3, 2.5)
    f.text(214 + 6 * 32, y + 38, "∞", 14, weight=700, color=SOLID["k"])
    f.text(310, y + 72, "현재 블록(local 1개)은 +∞로", 13, color=MUTED)
    _op(f, 440, 560, y + 32, "top-16", "비교 · 선택", "r")
    end = 568
    for i, t in enumerate(("2", "5", "9", "…", "현재")):
        w = 52 if i < 4 else 64
        f.rect(end, y + 18, w, 28, "#fff", "#9aa0b4", 4, 1.5)
        f.text(end + w / 2, y + 38, t, 13, mono=True)
        end += w + 6
    f.text(706, y + 72, "블록 번호  4 × 16", 14, color=MUTED, mono=True)
    f.text(900, y + 26, "현재 블록은 16개 '안에' 포함된다 — 17번째로 더하지 않는다", 15, "start", 700)
    f.text(900, y + 50, "앞쪽 고정 블록(init)은 0개 · 블록이 16개 이하면 전부 고른다", 15, "start", color=MUTED)
    f.text(900, y + 74, "검증(1+K 토큰)에서는 query 토큰마다 따로 고른다", 15, "start", color=MUTED)
    _divider(f, 806)

    # ---- ④ 본 attention
    y = 842
    _section(f, y + 26, "④ 본 attention", "고른 블록만", "읽는다")
    _box(f, 200, y, 100, 64, "q", "q_h", "16 × 128", "KV 헤드 h의 쿼리 16개")
    f.text(318, y + 40, "×", 26)
    _box(f, 336, y + 4, 230, 56, "k", "K_hᵀ (고른 것)", "128 × 2,048", "16블록 × 128토큰")
    f.text(584, y + 40, "=", 26)
    _box(f, 602, y, 150, 64, "q", "점수", "16 × 2,048")
    _op(f, 760, 850, y + 32, "softmax", "원소별", "n")
    _box(f, 858, y, 120, 64, "q", "p", "16 × 2,048")
    f.text(996, y + 40, "×", 26)
    _box(f, 1014, y + 4, 140, 56, "v", "V_h (고른 것)", "2,048 × 128")
    _op(f, 1162, 1290, y + 32, "4헤드 이어 붙여", None)
    f.text(1226, y + 64, "64 × 128 → × W_O", 13, color=MUTED, mono=True)
    _box(f, 1298, y + 12, 110, 40, "n", "y", "1 × 6144")
    _chip(f, 451, y + 124, "gather · 블록 단위 읽기", "r")
    _chip(f, 996, y + 124, "행렬 × 행렬", "g")
    _summary(f, 1030, "S가 아무리 길어도 본 attention은 KV 헤드마다 2,048토큰만 읽는다. S에 비례하는 것은 색인 키 읽기(①)뿐이다.",
             "S ≤ 2,048이면 블록 16개 안에 전부 들어가 full attention과 같다 — 짧은 컨텍스트에서는 이득이 없다", 76)
    return f


# ---------------------------------------------------------------- 7. MSA 개념 — 블록 단위로 고른다
# 토큰마다 indexer 점수 (블록 9개 × 토큰 6개로 줄여 그린다). 마지막 블록이 현재 블록, 뒤 두 토큰은 아직 없다.
MSA_SCORES = [[.2, .3, .1, .25, .2, .15], [.2, .9, .3, .2, .1, .2], [.3, .2, .35, .1, .2, .3],
              [.1, .2, .15, .4, .2, .1], [.2, .3, .2, .25, .8, .3], [.3, .2, .1, .2, .35, .2],
              [.7, .4, .3, .2, .5, .3], [.2, .1, .3, .45, .2, .1], [.3, .2, .4, .3, None, None]]
MSA_PICK = {1, 4, 6}  # 점수로 뽑힌 블록 — 현재 블록(8)은 강제로 들어간다


def fig_msa_concept():
    f = Fig("07-msa-concept", 1560, 820)
    f.title(40, 44, "블록 단위로 고른다 — MSA (MiniMax-M3, sparse 층 하나 · KV 헤드 하나)")
    bx = [270 + b * 138 for b in range(9)]
    cur = 8

    def tok_x(b, i):
        return bx[b] + 8 + i * 20

    # 원본 토큰
    f.text(40, 128, "원본 토큰", 20, "start", 700)
    f.text(160, 128, "S개", 15, "start", color=MUTED)
    for b in range(9):
        last = b == cur
        f.rect(bx[b], 100, 128, 40, "#fff" if not last else "#f6f5fd", SOLID["q"] if last else "#b9bdd0", 6, 1.8,
               "5 4" if last else None)
        for i, s in enumerate(MSA_SCORES[b]):
            f.rect(tok_x(b, i), 108, 16, 24, "#c9cddb" if s is not None else "none", "none" if s is not None else "#c9cddb",
                   3, 1.2, None if s is not None else "3 3")
    f.text(bx[cur] + 64, 162, "현재 블록", 14, weight=700, color=SOLID["q"])
    f.text(bx[0], 162, "128토큰 = 블록 하나", 14, "start", color=MUTED)

    # 추가된 단계 띠
    f.rect(30, 186, 1500, 330, "#f6f5fd", "#d9d2f3", 12, 1.5)
    f.rect(48, 198, 120, 26, PURPLE, "none", 13)
    f.text(108, 216, "추가된 단계", 13, weight=700, color="#ffffff")

    # ① 토큰 점수
    f.text(48, 262, "① 토큰 점수", 20, "start", 700, SOLID["i"])
    f.text(48, 286, "q_I · k_I — 색인 키 전부", 15, "start", color=MUTED)
    base = 300
    for b in range(9):
        for i, s in enumerate(MSA_SCORES[b]):
            if s is None:
                continue
            h = 8 + s * 56
            f.rect(tok_x(b, i), base - h, 16, h, "#a9c8f0", "none", 2)

    # ② 블록 max
    f.text(48, 372, "② 블록 점수", 20, "start", 700, SOLID["i"])
    f.text(48, 396, "블록 안 토큰 점수의 max", 15, "start", color=MUTED)
    base = 410
    for b in range(9):
        if b == cur:
            f.rect(bx[b] + 34, base - 64, 60, 64, "#fff", SOLID["k"], 4, 2.2)
            f.text(bx[b] + 64, base - 24, "+∞", 18, weight=700, color=SOLID["k"])
            continue
        m = max(MSA_SCORES[b])
        h = 8 + m * 56
        f.rect(bx[b] + 34, base - h, 60, h, SOLID["i"] if b in MSA_PICK else "#a9c8f0", "none", 3)
    f.text(bx[cur] + 64, base - 74, "현재 블록은 +∞로", 13, weight=700, color=SOLID["k"])

    # ③ top-k
    f.text(48, 470, "③ top-16", 20, "start", 700, SOLID["i"])
    f.text(48, 494, "블록 번호만 넘긴다", 15, "start", color=MUTED)
    for b in sorted(MSA_PICK | {cur}):
        f.arrow(bx[b] + 64, 418, bx[b] + 64, 540)
    f.text(bx[2] + 10, 470, "그림은 블록 9개 중 4개", 13, "start", color=MUTED)
    f.text(bx[2] + 10, 490, "실제로는 S/128개 중 16개 (현재 블록 포함)", 13, "start", color=MUTED)

    # ④ attention
    f.text(40, 580, "④ attention", 20, "start", 700, "#c07a12")
    f.text(40, 604, "고른 블록의 원본 KV", 15, "start", color=MUTED)
    for b in range(9):
        on = b in MSA_PICK or b == cur
        f.rect(bx[b], 552, 128, 40, SOLID["c"] if on else "#e6e8f1", "#c98a1c" if on else "none", 6, 1.8)
        if on:
            for i, s in enumerate(MSA_SCORES[b]):
                if s is not None:
                    f.rect(tok_x(b, i), 560, 16, 24, "#f3c879", "none", 3)
    f.text(bx[0], 622, "블록 안 128토큰을 통째로 읽는다 — 블록 하나는 이어 읽기, 블록 사이는 건너뛰기", 15, "start", 700)
    f.text(bx[0], 646, "압축하지 않는다 — 블록은 고르는 단위일 뿐, 읽는 것은 원본 토큰이다", 15, "start", color=MUTED)

    # 1M 토큰이면
    f.text(40, 724, "1M 토큰이면", 20, "start", 700)
    f.rect(270, 690, 1250, 60, "#fff", "#d3d6e2", 10, 1.5)
    f.text(300, 728, "1,048,576 토큰", 18, "start")
    f.text(520, 728, "→", 20)
    f.text(560, 728, "8,192 블록", 18, "start", 700)
    f.text(720, 728, "→", 20)
    f.text(760, 728, "16블록 = 2,048토큰 읽는다", 18, "start", 700)
    f.text(1080, 728, "KV 헤드 4개가 각자 고른다", 16, "start", color=MUTED)
    f.text(40, 790, "색인 키는 1M개 전부 훑는다 (①). 본 KV는 2,048개만 읽는다 (④).", 15, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 8. MSA 텐서 연산 카드
CARD = {"g": ("#f1f8f1", "#cfe6d1"), "r": ("#fdf0f7", "#f2c6df"), "n": ("#f6f7fb", "#dfe2ec")}


def _card(f, x, y, title, lines, chip, kind, draw=None, w=262, h=210):
    fill, stroke = CARD[kind]
    f.rect(x, y, w, h, fill, stroke, 10, 1.8)
    f.text(x + 16, y + 34, title, 19, "start", 700)
    if draw:
        draw(x + 16, y + 52)
    for i, t in enumerate(lines):
        f.text(x + 16, y + 128 + i * 22, t, 14, "start", 700 if i else 400, mono=True)
    if chip:
        w2 = sum(13 if ord(ch) > 0x2000 else 7.5 for ch in chip) + 22
        f.rect(x + 16, y + 170, w2, 24, CHIP[kind], "none", 12)
        f.text(x + 16 + w2 / 2, y + 187, chip, 13, weight=700, color="#ffffff")


def _empty_card(f, x, y, name, w=262, h=210):
    f.rect(x, y, w, h, "#fff", "#d3d6e2", 10, 1.5, "6 5")
    f.text(x + w / 2, y + h / 2 - 2, name, 21, weight=700, color="#a3a8b8")
    f.text(x + w / 2, y + h / 2 + 26, "없다", 16, color="#a3a8b8")


def fig_msa_tensor():
    f = Fig("08-msa-tensor", 1560, 680)
    f.title(40, 44, "텐서 연산으로 보면 — MSA 층과 full attention 층 (M3)")
    xs = [130 + i * 280 for i in range(5)]

    def d_index(x, y):
        _box(f, x, y + 4, 40, 36, "q", "q_I", fs=13)
        f.text(x + 52, y + 28, "·", 22)
        f.rect(x + 66, y + 12, 160, 20, KINDS["i"][0], KINDS["i"][1], 4, 1.8)

    def d_max(x, y):
        for g in range(3):
            for i in range(4):
                h = [10, 26, 14, 18, 30, 12, 16, 8, 14, 20, 34, 12][g * 4 + i]
                f.rect(x + g * 40 + i * 9, y + 40 - h, 7, h, "#a9c8f0", "none", 1)
        f.arrow(x + 124, y + 26, x + 160, y + 26)
        for g, h in enumerate((26, 30, 34)):
            f.rect(x + 170 + g * 20, y + 40 - h, 14, h, SOLID["i"], "none", 2)

    def d_topk(x, y):
        hs = [8, 12, 30, 10, 14, 34, 9, 12, 28, 10, 13, 9, 11, 38]
        for i, h in enumerate(hs):
            f.rect(x + i * 16, y + 40 - h, 12, h, SOLID["i"] if h > 25 else "#a9c8f0", "none", 2)
        f.rect(x + 14 * 16, y + 2, 12, 38, "#fff", SOLID["k"], 2, 1.8)

    def d_gather(x, y):
        on = {2, 5, 8, 13}
        for i in range(14):
            f.rect(x + i * 16, y, 14, 14, SOLID["c"] if i in on else "#fbe7c4", "none", 2)
        for j, i in enumerate(sorted(on)):
            f.line(x + i * 16 + 7, y + 15, x + 72 + j * 22 + 10, y + 32, "#8b90a3", 1.2)
            f.rect(x + 72 + j * 22, y + 32, 20, 16, SOLID["c"], "none", 2)

    def d_attn(x, y, wide=False):
        _box(f, x, y + 4, 40, 36, "q", "q", fs=13)
        f.text(x + 52, y + 28, "·", 22)
        f.rect(x + 66, y + 2, 150 if wide else 54, 40, SOLID["c"], "#c98a1c", 6, 1.8)

    # MSA 행
    y = 96
    f.text(40, y + 104, "MSA", 26, "start", 700)
    f.text(40, y + 132, "3~59층", 15, "start", color=MUTED)
    _card(f, xs[0], y, "① indexer 점수", ["(4 × 128) × (128 × S)", "→ (4 × S)"], "행렬 × 행렬", "g", d_index)
    _card(f, xs[1], y, "② 블록 max", ["(4 × S/128 × 128)", "→ (4 × S/128)"], "블록 안 최댓값", "n", d_max)
    _card(f, xs[2], y, "③ top-k", ["(4 × S/128)", "→ 헤드마다 16블록"], "비교 · 선택", "r", d_topk)
    _card(f, xs[3], y, "④ gather", ["16블록 × 128토큰", "→ (2,048 × 128) K, V"], "블록 단위 읽기", "r", d_gather)
    _card(f, xs[4], y, "⑤ attention", ["(16 × 128) × (128 × 2,048)", "→ 16 × 2,048 · 헤드 4번"], "행렬 × 행렬", "g", d_attn)
    for i in range(4):
        f.arrow(xs[i] + 264, y + 105, xs[i + 1] - 4, y + 105)
    f.line(40, 334, 1520, 334, "#e3e5ee", 2, "6 6")

    # full 행
    y = 360
    f.text(40, y + 104, "full", 26, "start", 700)
    f.text(40, y + 132, "0~2층", 15, "start", color=MUTED)
    for i, name in enumerate(("indexer", "블록 max", "top-k", "gather")):
        _empty_card(f, xs[i], y, name)
    _card(f, xs[4], y, "attention", ["(16 × 128) × (128 × S)", "→ 16 × S · 헤드 4번"], "행렬 × 행렬", "g",
          lambda x, yy: d_attn(x, yy, True))
    for i in range(4):
        f.arrow(xs[i] + 264, y + 105, xs[i + 1] - 4, y + 105)

    # 요약
    f.rect(40, 596, 1480, 70, "#f6f5fd", "#d9d2f3", 10, 1.5)
    for i, (t, k) in enumerate((("행렬곱", "g"), ("원소별", "n"), ("행렬곱이 아닌 일", "r"))):
        f.rect(64 + i * 130, 610, 14, 14, CHIP[k], "none", 3)
        f.text(84 + i * 130, 622, t, 14, "start", 600)
    f.text(480, 622, "점수 · 블록 max · top-k는 KV 헤드 4개가 따로 한다 (index 헤드 수 = KV 헤드 수)", 14, "start",
           color=MUTED, mono=False)
    f.text(64, 652, "MSA는 ⑤의 크기가 S와 무관하다 — S에 비례하는 것은 ①뿐이다.", 17, "start", 700)
    f.text(820, 652, "full 층은 단계가 없지만 attention이 S에 비례한다.", 17, "start", 700)
    return f


# ---------------------------------------------------------------- 9. 한 층 안의 두 캐시 · KV 헤드마다 다른 선택
MSA_HEAD_PICK = [{2, 7, 11, 18}, {0, 5, 13, 20}, {3, 7, 15, 16}, {9, 10, 14, 21}]


def fig_msa_heads():
    f = Fig("09-msa-heads", 1560, 720)
    f.title(40, 44, "한 층 안의 두 캐시 — 색인 키로 고르고, 원본 KV를 읽는다 (M3 sparse 층)")
    n, x0, cw = 24, 230, 42
    f.text(x0, 96, "토큰 0", 14, "start", color=MUTED)
    f.text(x0 + n * cw - 4, 96, "지금", 14, "end", color=MUTED)
    f.line(x0, 106, x0 + n * cw - 4, 106, "#8b90a3", 2)

    # 색인 키 캐시
    y = 126
    f.text(40, y + 22, "색인 키 캐시", 19, "start", 700, SOLID["i"])
    f.text(40, y + 44, "S × 128 · 헤드 하나", 14, "start", color=MUTED)
    f.rect(x0, y, n * cw - 4, 30, KINDS["i"][0], KINDS["i"][1], 5, 1.8)
    for b in range(1, n):
        f.line(x0 + b * cw - 2, y + 1, x0 + b * cw - 2, y + 29, KINDS["i"][1], 0.8)
    f.text(x0, y + 54, "① 점수에서 전부 훑는다 — 길이 S에 비례", 14, "start", color=MUTED)
    f.text(x0 + n * cw + 20, y + 20, "4헤드 점수 → 헤드마다 top-16", 14, "start", 700, SOLID["i"])

    # KV 헤드 4개
    for h in range(4):
        y = 236 + h * 74
        f.text(40, y + 24, f"KV 헤드 {h}", 19, "start", 700)
        f.text(40, y + 46, f"쿼리 헤드 {16 * h}~{16 * h + 15}", 13, "start", color=MUTED)
        for b in range(n):
            x = x0 + b * cw
            if b == n - 1:
                f.rect(x, y, cw - 4, 36, SOLID["q"], "none", 4)
                f.rect(x - 2, y - 2, cw, 40, "none", SOLID["q"], 5, 1.5, "4 3")
            elif b in MSA_HEAD_PICK[h]:
                f.rect(x, y, cw - 4, 36, SOLID["c"], "#c98a1c", 4, 1.5)
            else:
                f.rect(x, y, cw - 4, 36, "#fdf3e1", "#ecd3a4", 4, 1.2)
        f.arrow(x0 + n * cw + 8, y + 18, x0 + n * cw + 70, y + 18)
        f.rect(x0 + n * cw + 78, y, 150, 36, KINDS["c"][0], KINDS["c"][1], 5, 1.8)
        f.text(x0 + n * cw + 153, y + 24, "2,048 × 128", 14, mono=True)
    f.text(x0, 552, "본 K, V 캐시 — 칸 하나가 128토큰 블록 · 4 × S × 128 각각", 14, "start", color=MUTED)
    f.text(x0 + n * cw - 4, 552, "현재 블록 — 늘 포함", 14, "end", 700, SOLID["q"])
    f.text(x0 + n * cw + 153, 552, "헤드마다 softmax", 14, color=MUTED)

    _summary(f, 586, "고르는 것은 색인 키, 읽는 것은 원본 KV다. KV 헤드 4개가 서로 다른 블록을 읽고, 쿼리 헤드 16개는 자기 KV 헤드의 선택을 따른다.",
             "그림은 블록 24개 중 5개를 골랐다 — 실제로는 16개다. 1+K개 토큰을 검증할 때는 토큰마다 따로 고른다.", 76)
    return f


# ---------------------------------------------------------------- 5. MoE 연산
def fig_moe():
    f = Fig("05-moe-ops", 1560, 900)
    f.title(40, 44, "MoE의 텐서 연산 — 층 하나, 토큰 하나 (값은 M2.5 | M3)")
    _legend(f, 960, 44)

    # ---- ① 라우터
    y = 100
    _section(f, y + 26, "① 라우터", "expert 고르기")
    _box(f, 180, y + 20, 110, 24, "n", dims="x  1 × d")
    _op(f, 296, 380, y + 32, "× W_g", "행렬 × 벡터", "g")
    _box(f, 388, y + 12, 130, 40, "i", "logits", "1 × E")
    _op(f, 526, 620, y + 32, "sigmoid", "원소별", "n")
    _box(f, 628, y + 12, 100, 40, "i", "s", "1 × E", "0~1 점수")
    _op(f, 736, 830, y + 32, "+ bias", "원소별", "n")
    _box(f, 838, y + 12, 100, 40, "i", "s + b", "1 × E", "고를 때만 쓴다")
    _op(f, 946, 1050, y + 32, "top-k", "비교 · 선택", "r")
    end = 1058
    for t in ("17", "40", "…", "203"):
        f.rect(end, y + 18, 46, 28, "#fff", "#9aa0b4", 4, 1.5)
        f.text(end + 23, y + 38, t, 13, mono=True)
        end += 52
    f.text(1160, y + 72, "expert 번호  1 × k", 14, color=MUTED, mono=True)
    f.text(180, y + 128, "가중치는 bias 없는 s에서 가져와 합이 1이 되게 나눈다:  w = s[idx] / Σ s[idx]", 15, "start", 700)
    f.text(180, y + 152, "bias(e_score_correction_bias)는 부하 균형용 — 고르는 데만 쓰고 출력 크기에는 들어가지 않는다. 라우터는 FP32",
           15, "start", color=MUTED)
    _divider(f, 290)

    # ---- ② expert
    y = 330
    _section(f, y + 26, "② expert", "고른 k개만", "계산한다")
    f.text(180, y + 30, "x", 16, mono=True)
    ys = [y, y + 64, y + 150]
    for i, yy in enumerate(ys):
        lab = "expert " + ("17" if i == 0 else "40" if i == 1 else "203")
        f.path(f"M196,{y + 26} C230,{y + 26} 230,{yy + 22} 270,{yy + 22}")
        _box(f, 278, yy + 4, 170, 36, "v", "W1 · W3", fs=14)
        _op(f, 456, 560, yy + 22, "SwiGLU", None)
        _box(f, 568, yy + 4, 110, 36, "v", "1 × f", fs=14)
        _op(f, 686, 760, yy + 22, "× W2", None)
        _box(f, 768, yy + 4, 100, 36, "n", "1 × d", fs=14)
        _op(f, 876, 960, yy + 22, "× w_" + str(i + 1), None)
        f.path(f"M968,{yy + 22} C1000,{yy + 22} 1000,{y + 100} 1036,{y + 100}")
        if i == 1:
            f.text(570, yy + 70, "⋮  (M2.5 8개 | M3 4개)", 15, color=MUTED)
    f.text(363, y + 214, "d × 2f → 1 × 2f", 13, color=MUTED, mono=True)
    _chip(f, 363, y + 226, "행렬 × 벡터 · expert마다", "g")
    f.raw(f'<circle cx="1056" cy="{y + 100}" r="18" fill="#fff" stroke="#5d6377" stroke-width="2"/>')
    f.text(1056, y + 108, "Σ", 20, weight=700)
    _op(f, 1076, 1170, y + 100, "× 2.0 (M3만)", None)
    f.raw(f'<circle cx="1192" cy="{y + 100}" r="18" fill="#fff" stroke="#5d6377" stroke-width="2"/>')
    f.text(1192, y + 108, "+", 22, weight=700)
    _op(f, 1212, 1290, y + 100, "", None)
    _box(f, 1298, y + 82, 100, 36, "n", "1 × d", fs=14)
    # shared
    _box(f, 1110, y + 196, 164, 40, "c", "shared expert", fs=14)
    f.text(1290, y + 222, "M3만 · 폭 3,072 · 항상 계산", 13, "start", color=MUTED)
    f.arrow(1192, y + 194, 1192, y + 120)
    f.path(f"M206,{y + 38} L206,{y + 276} L1192,{y + 276} L1192,{y + 240}", dash="5 5")
    _divider(f, 630)

    # ---- 표
    y = 664
    _section(f, y + 4, "두 모델")
    cols = [("", 180), ("E", 340), ("k", 470), ("폭 f", 580), ("shared", 720), ("routed 배율", 880),
            ("SwiGLU", 1030), ("토큰당 expert 계산", 1290)]
    for t, x in cols:
        f.text(x, y + 4, t, 15, "start", 700, MUTED)
    rows = (("M2.5", "256", "8", "1,536", "없음", "1", "silu(g) · u", "8 × 14.2M = 113M 파라미터"),
            ("M3", "128", "4", "3,072", "1개 · 3,072", "2.0", "OAI식 (clamp, α 1.702)", "5 × 56.6M = 283M 파라미터"))
    for r, row in enumerate(rows):
        for (t, x), v in zip(cols, row):
            f.text(x, y + 40 + r * 32, v, 16, "start", 700 if x == 180 else 400, mono=x in (340, 470, 580, 880))
    f.text(180, y + 128, "M3의 SwiGLU:  g ← min(g, 7) · u ← clamp(u, ±7) · out = (u + 1) · g · σ(1.702 g)   (gpt-oss와 같은 식)",
           15, "start", color=MUTED)
    _summary(f, 820, "고르는 규칙(sigmoid + bias → top-k → 정규화)은 같다. M3는 expert를 줄이고 넓혀 shared를 붙였고, 합에 2.0을 곱한다.")
    return f


# ---------------------------------------------------------------- 6. 캐시 크기와 읽는 양
M25_KV = 62 * 2 * 8 * 128 * 2          # 253,952 B
M3_KV = 60 * 2 * 4 * 128 * 2           # 122,880 B
M3_IDX = 57 * 128 * 2                  # 14,592 B


def _read(model, S):
    """토큰 하나를 만들 때 읽는 캐시 바이트 (전 층, BF16). 반환: [(이름, 바이트, 색)]"""
    if model == "m25":
        return [("K, V 전부 · 62층", 62 * 4096 * S, LAYER["full"])]
    return [("full 3층 K, V", 3 * 2048 * S, LAYER["full"]),
            ("색인 키 · 57층", 57 * 256 * S, SOLID["i"]),
            ("고른 블록 K, V · 57층", 57 * 2048 * min(S, 2048), SOLID["c"])]


def _gb(b):
    return f"{b / 1e9:.1f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB"


def fig_kv():
    f = Fig("06-kv-cache-read", 1560, 760)
    f.title(40, 44, "KV cache — 토큰 하나가 남기는 양, 토큰 하나를 만들 때 읽는 양 (전 층, BF16)")

    # ---- 저장
    f.text(40, 104, "저장 — 토큰 하나", 20, "start", 700)
    sx = 260 / 1024 * 3  # px per KiB
    for r, (name, parts) in enumerate((("M2.5", [("K, V  62층", M25_KV, LAYER["full"])]),
                                       ("M3", [("K, V  60층", M3_KV, LAYER["full"]), ("색인 키  57층", M3_IDX, SOLID["i"])]))):
        y = 136 + r * 70
        f.text(40, y + 28, name, 18, "start", 700)
        x = 160
        for lbl, b, col in parts:
            w = b / 1024 * sx
            f.rect(x, y, w, 40, col, "none", 4)
            if w > 120:
                f.text(x + w / 2, y + 26, f"{lbl} · {b / 1024:.0f} KiB", 14, color="#ffffff", weight=700)
            x += w + 2
        tot = sum(b for _, b, _ in parts)
        f.text(x + 12, y + 26, f"{tot / 1024:.1f} KiB", 16, "start", 700)
        if r == 1:
            f.text(x + 110, y + 26, f"← 색인 키 {M3_IDX / 1024:.2f} KiB", 14, "start", color=SOLID["i"])
    f.text(160, 298, "M2.5: 62 × 2 × 8 × 128 × 2 B   ·   M3: 60 × 2 × 4 × 128 × 2 B + 57 × 128 × 2 B", 14, "start", color=MUTED, mono=True)
    _divider(f, 330)

    # ---- 읽기
    f.text(40, 376, "읽기 — 토큰 하나를 만들 때", 20, "start", 700)
    rows = (("M2.5 · 128K", "m25", 131072), ("M3 · 128K", "m3", 131072), ("M3 · 1M", "m3", 1048576))
    scale = 1080 / 34e9
    for r, (name, m, S) in enumerate(rows):
        y = 404 + r * 76
        f.text(40, y + 28, name, 18, "start", 700)
        x = 200
        for lbl, b, col in _read(m, S):
            w = max(b * scale, 3)
            f.rect(x, y, w, 40, col, "none", 4)
            if w > 180:
                f.text(x + w / 2, y + 26, f"{lbl} · {_gb(b)}", 14, color="#ffffff", weight=700)
            x += w + 2
        tot = sum(b for _, b, _ in _read(m, S))
        f.text(x + 12, y + 26, _gb(tot), 16, "start", 700)
        if m == "m3":
            parts = _read(m, S)
            f.text(x + 100, y + 26, "  ·  ".join(f"{lbl} {_gb(b)}" for lbl, b, _ in parts), 13, "start", color=MUTED)
    f.text(200, 650, "M2.5의 최대 길이는 196,608이라 1M 행이 없다. 같은 128K에서 M3는 M2.5의 약 1/11을 읽는다.", 15, "start", color=MUTED)
    _summary(f, 680, "M3가 길게 읽는 것은 본 KV가 아니라 색인 키(57층 × 256 B × S)와 full 3층이다. 본 KV 읽기는 S와 무관하게 약 239 MB다.")
    return f


# ---------------------------------------------------------------- 10. EAGLE-3 draft 모듈
# Inferact/MiniMax-M3-EAGLE3 config.json · safetensors 헤더, vLLM llama_eagle3.py (fc_norm, norm_output)
AUX = (2, 30, 57)  # eagle_config가 없어 vLLM 기본값


def fig_eagle_draft():
    f = Fig("10-eagle3-draft", 1560, 1390)
    f.title(40, 44, "EAGLE-3 draft 모듈 — decoder 층 1개 (Inferact/MiniMax-M3-EAGLE3, BF16)")
    cx, w = 820, 420
    x0 = cx - w / 2

    # ---- aux 입력 (pass 1만)
    f.text(1100, 100, "pass 1만 — target의 중간 hidden 3개", 16, "start", 700)
    f.text(1122, 136, "target", 13, "end", color=MUTED)
    for L in range(60):
        f.rect(1130 + L * 6, 116, 5, 26, SOLID["q"] if L in AUX else "#d3d6e2", "none", 1)
    for i, L in enumerate(AUX):
        bx, c = 1132.5 + L * 6, 1160 + i * 152
        f.arrow(bx, 146, c, 172)
        _box(f, c - 56, 174, 112, 28, "i", f"h_L{L}", fs=14)
        f.arrow(c, 202, c, 222)
        _box(f, c - 56, 224, 112, 26, "n", "fc_norm", fs=13)
        t = 1312 + (i - 1) * 60
        f.path(f"M{c},252 C{c},272 {t},268 {t},288")
    _box(f, 1192, 290, 240, 34, "n", "concat  1 × 18,432", fs=14)
    f.arrow(1312, 324, 1312, 344)
    _box(f, 1192, 346, 240, 36, "i", "fc  18,432 → 6,144", solid=True, fs=14)
    f.text(1312, 404, "ReplicatedLinear — TP로 안 나눈다", 13, color=MUTED)
    f.path("M1190,364 C1110,364 1110,396 1024,396")

    # ---- 입력 두 개
    _box(f, 620, 250, 160, 36, "c", "token", fs=15)
    f.arrow(700, 286, 700, 318)
    _box(f, 590, 320, 220, 36, "c", "embed_tokens", fs=14)
    f.arrow(700, 356, 700, 438)
    f.text(560, 330, "embed_tokens 200,064 × 6,144", 14, "end", color=MUTED, mono=True)
    f.text(560, 352, "target과 값이 같다 → vLLM이 공유", 14, "end", color=MUTED)
    f.text(940, 346, "pass 1: fc 출력", 13, color=MUTED)
    f.text(940, 366, "pass 2~: 이전 pass의 층 출력", 13, color=MUTED)
    _box(f, 860, 378, 160, 36, "i", "g  1 × 6,144", fs=14)
    f.arrow(940, 414, 940, 438)
    _box(f, 600, 440, 200, 32, "n", "input_layernorm", fs=14)
    _box(f, 840, 440, 200, 32, "n", "hidden_norm", fs=14)
    f.path("M700,472 C700,492 760,488 760,508")
    f.path("M940,472 C940,492 880,488 880,508")
    _box(f, x0, 510, w, 40, "n", "concat  [ê ; ĝ]  1 × 12,288", fs=15)
    f.text(560, 526, "토큰과 hidden을 따로 norm 하고", 14, "end", color=MUTED)
    f.text(560, 548, "이어 붙인다 — 폭이 2H", 14, "end", color=MUTED)

    # ---- attention
    f.arrow(cx, 550, cx, 588)
    f.rect(x0, 590, w, 130, KINDS["q"][0], KINDS["q"][1], 6, 2)
    f.text(cx, 620, "self-attention  (draft 자체 KV 캐시)", 18, weight=700)
    f.text(cx, 648, "q 12,288 → 64 × 128", 14, mono=True)
    f.text(cx, 670, "k, v 12,288 → 64 × 128  (GQA판 4 × 128)", 14, mono=True)
    f.text(cx, 696, "causal · RoPE θ 5M · o 8,192 → 6,144", 14, color=MUTED)
    f.text(1100, 620, "sparse가 아니다 — context 전체를 읽는다", 15, "start", 700)
    f.text(1100, 644, "MHA: KV 헤드 64 · 토큰당 KV 32 KiB", 15, "start", color=MUTED)
    f.text(1100, 668, "GQA: KV 헤드 4 · 토큰당 KV 2 KiB", 15, "start", color=MUTED)
    f.text(1100, 692, "target의 MSA · 색인 키는 쓰지 않는다", 15, "start", color=MUTED)

    def add(y):
        f.raw(f'<circle cx="{cx}" cy="{y}" r="14" fill="#fff" stroke="#5d6377" stroke-width="2"/>')
        f.text(cx, y + 7, "+", 22, weight=700)

    f.arrow(cx, 720, cx, 738)
    add(754)
    f.path(f"M940,426 L1060,426 L1060,754 L{cx + 16},754", dash="5 5")
    f.text(1068, 742, "residual = g", 13, "start", color=MUTED)
    f.arrow(cx, 768, cx, 798)
    _box(f, cx - 110, 800, 220, 32, "n", "RMSNorm", fs=14)

    # ---- MLP
    f.arrow(cx, 832, cx, 858)
    f.rect(x0, 860, w, 110, KINDS["v"][0], KINDS["v"][1], 6, 2)
    f.text(cx, 890, "MLP — SwiGLU (dense 하나)", 18, weight=700)
    f.text(cx, 918, "gate, up 6,144 → 18,432", 14, mono=True)
    f.text(cx, 942, "down 18,432 → 6,144", 14, mono=True)
    f.text(1100, 900, "target의 MoE가 아니다", 15, "start", 700)
    f.text(1100, 924, "expert · 라우터 없이 18,432 폭 하나", 15, "start", color=MUTED)
    f.arrow(cx, 970, cx, 998)
    add(1014)
    f.path(f"M{cx},784 L1080,784 L1080,1014 L{cx + 16},1014", dash="5 5")
    f.arrow(cx, 1028, cx, 1058)
    _box(f, cx - 110, 1060, 220, 32, "n", "norm", fs=14)

    # ---- 출력
    f.path(f"M{cx + 110},1076 L1150,1076", dash="5 5")
    f.rect(1158, 1056, 340, 40, "#fff", SOLID["i"], 6, 1.8, "5 4")
    f.text(1328, 1082, "→ 다음 pass의 g  (norm_output)", 14, color=SOLID["i"], weight=700)
    f.arrow(cx, 1092, cx, 1128)
    _box(f, x0, 1130, w, 40, "c", "lm_head  6,144 → 200,064", fs=15)
    f.text(560, 1146, "draft 자기 lm_head — target과 값이 다르다", 14, "end", color=MUTED)
    f.text(560, 1168, "어휘 축소 없음 · 매 pass 200,064 전부", 14, "end", color=MUTED)
    f.arrow(cx, 1170, cx, 1198)
    f.rect(cx - 110, 1200, 220, 36, "#fff", SOLID["k"], 6, 2)
    f.text(cx, 1224, "argmax → draft 토큰", 15, weight=700)
    f.path(f"M{cx + 110},1218 L1150,1218", dash="5 5")
    f.rect(1158, 1198, 340, 40, "#fff", SOLID["c"], 6, 1.8, "5 4")
    f.text(1328, 1224, "→ 다음 pass의 token", 14, color="#b5741a", weight=700)
    _summary(f, 1270, "target 층 3개를 이어 붙여 6,144로 줄이고(fc), 토큰 embedding과 합쳐 일반 Llama 층 하나를 지난다.",
             "draft 고유 파라미터 805M (MHA) · 616M (GQA) + 자기 lm_head 1.23B — 체크포인트 합 3.26B · 3.07B", 76)
    return f


# ---------------------------------------------------------------- 11. EAGLE-3 한 step과 비용
GOLDEN_AL = {"thinking off": [1.84, 2.52, 2.93, 3.41, 3.59, 3.57, 3.85, 4.06],
             "thinking on": [1.78, 2.38, 2.83, 3.11, 3.35, 3.45, 3.53, 3.74]}
# TP8 GPU 한 장이 pass마다 읽는 가중치 (BF16, GB)
EG_FC = 6144 * 18432 * 2 / 1e9                              # 0.226 — 복제, pass 1만
EG_LAYER = (8192 * 12288 * 3 + 6144 * 8192 + 3 * 6144 * 18432) * 2 / 8 / 1e9   # 0.173
EG_HEAD = 200064 * 6144 * 2 / 8 / 1e9                       # 0.307


def fig_eagle_step():
    f = Fig("11-eagle3-step", 1560, 1190)
    f.title(40, 44, "EAGLE-3 decode 한 step (K = 3) — 흐름, GPU당 읽는 양, 수락 길이")

    # ---- ① 흐름
    _section(f, 100, "① 흐름", "draft는 K번 차례로", "target은 한 번에")
    xs = [200, 470, 700, 930]
    _box(f, 200, 120, 200, 70, "n")
    f.text(300, 150, "target verify", 16, weight=700)
    f.text(300, 174, "(이전 step)", 14, color=MUTED)
    for i, x in enumerate(xs[1:]):
        f.rect(x, 120, 180, 70, KINDS["c"][0], KINDS["c"][1], 6, 2)
        f.text(x + 90, 150, f"draft pass {i + 1}", 16, weight=700)
        f.text(x + 90, 174, "fc + 층 + lm_head" if i == 0 else "층 + lm_head", 14, color=MUTED)
        f.rect(x + 60, 214, 60, 30, "#fff", SOLID["c"], 5, 2)
        f.text(x + 90, 235, f"t{i + 1}", 15, mono=True, weight=700)
        f.arrow(x + 90, 190, x + 90, 212)
    f.arrow(400, 155, 468, 155)
    f.text(434, 112, "aux × 3", 13, color=MUTED, mono=True)
    f.arrow(650, 155, 698, 155)
    f.arrow(880, 155, 928, 155)
    for x in (674, 904):
        f.text(x, 112, "g, t", 13, color=MUTED, mono=True)
    f.rect(1180, 120, 260, 70, SOLID["q"], "none", 6)
    f.text(1310, 150, "target verify", 16, weight=700, color="#ffffff")
    f.text(1310, 174, "1 + 3 토큰을 한 번에", 14, color="#ffffff")
    f.path("M1050,229 L1150,229 L1150,170 L1178,170")
    for x in (560, 790):
        f.path(f"M{x + 30},244 L{x + 30},262 L1150,262 L1150,229", arrow=False)
    f.text(1310, 222, "수락 길이(AL) ≈ 2.9", 15, weight=700)
    f.text(1310, 244, "+ 다음 step의 aux · 마지막 토큰", 13, color=MUTED)
    f.text(200, 300, "pass 1만 target의 hidden 3개를 fc로 받는다. pass 2부터는 직전 pass의 층 출력(g)과 토큰을 다시 넣는다.", 15, "start", color=MUTED)
    f.text(200, 324, "draft 토큰은 K개가 차례로 나온다 — pass끼리 병렬이 안 된다. lm_head는 pass마다 요청당 1행만 계산한다.", 15, "start", color=MUTED)
    _divider(f, 356)

    # ---- ② GPU당 가중치 읽기
    y = 392
    _section(f, y + 4, "② 가중치")
    f.text(200, y + 4, "TP8 GPU 한 장이 step마다 읽는 가중치 (BF16, MHA draft)", 16, "start", 700)
    sc = 1100 / 8.0
    parts = [("fc", EG_FC, SOLID["i"]), ("층", EG_LAYER, SOLID["c"]), ("lm_head", EG_HEAD, SOLID["k"])]
    x = 200
    for p in range(3):
        for lbl, b, col in parts:
            if lbl == "fc" and p:
                continue
            wb = b * sc
            f.rect(x, y + 24, wb, 40, col, "none", 3)
            if wb > 30:
                f.text(x + wb / 2, y + 49, lbl if wb > 50 else "", 13, color="#ffffff", weight=700)
            x += wb + 1
        f.text((x - (EG_LAYER + EG_HEAD) * sc / 2 - (EG_FC * sc / 2 if p == 0 else 0)), y + 84, f"pass {p + 1}", 13, color=MUTED)
        x += 6
    tot = EG_FC + 3 * (EG_LAYER + EG_HEAD)
    f.text(x + 10, y + 50, f"{tot:.2f} GB", 16, "start", 700)
    f.rect(200, y + 104, 7.5 * sc, 40, "#e6e2f8", "none", 3)
    f.rect(200, y + 104, 7.0 * sc, 40, "#d9d2f3", "none", 3)
    f.text(200 + 3.5 * sc, y + 129, "target verify (batch 1, 추정 7~8 GB — 깨어난 expert 가중치)", 14, weight=700)
    f.text(200 + 7.5 * sc + 12, y + 130, "7~8 GB", 16, "start", 700)
    f.text(200, y + 176, f"pass 1 = fc {EG_FC:.2f} + 층 {EG_LAYER:.2f} + lm_head {EG_HEAD:.2f} GB · pass 2~3 = {EG_LAYER + EG_HEAD:.2f} GB"
           " — lm_head가 절반 이상이다", 15, "start", color=MUTED)
    f.text(200, y + 200, "pass마다 AllReduce 3번 (embed 뒤 · o_proj 뒤 · down 뒤) + logits gather — 작은 batch에서는 호출 횟수가 지연을 만든다",
           15, "start", color=MUTED)
    _divider(f, 626)

    # ---- ③ draft KV 읽기 vs batch
    y = 662
    _section(f, y + 4, "③ draft KV", "batch에 비례")
    f.text(200, y + 4, "TP8 GPU 한 장이 pass마다 읽는 것 — context 8K, BF16", 16, "start", 700)
    w_pass = EG_LAYER + EG_HEAD
    sc2 = 520 / 2.7
    rows = [(1, "batch 1"), (16, "batch 16"), (64, "batch 64")]
    for r, (B, name) in enumerate(rows):
        yy = y + 30 + r * 54
        f.text(200, yy + 24, name, 15, "start", 700)
        for c, (lbl, perTok, col) in enumerate((("MHA", 4096, SOLID["k"]), ("GQA", 512, SOLID["v"]))):
            bx = 300 + c * 620
            kv = B * 8192 * perTok / 1e9
            f.rect(bx, yy + 6, w_pass * sc2, 26, "#d3d6e2", "none", 3)
            f.rect(bx + w_pass * sc2 + 1, yy + 6, max(kv * sc2, 2), 26, col, "none", 3)
            f.text(bx + w_pass * sc2 + kv * sc2 + 10, yy + 25, f"KV {_gb(kv * 1e9)}", 14, "start", 700 if kv > w_pass else 400)
    f.text(300 + w_pass * sc2 / 2, y + 196, f"가중치 {w_pass:.2f} GB", 13, color=MUTED)
    f.text(300, y + 224, "MHA draft (GPU당 KV 헤드 8)", 15, "start", 700, SOLID["k"])
    f.text(920, y + 224, "GQA draft (GPU당 KV 헤드 1, 복제)", 15, "start", 700, SOLID["v"])
    f.text(300, y + 248, "batch 14쯤부터 KV가 가중치보다 커진다", 14, "start", color=MUTED)
    f.text(920, y + 248, "같은 조건에서 KV가 1/8 — 장문맥 · 대배치용", 14, "start", color=MUTED)
    _divider(f, 930)

    # ---- ④ 수락 길이
    y = 966
    _section(f, y + 4, "④ 수락 길이", "InferenceX", "golden AL")
    px0, py0, pw, ph = 300, y + 150, 560, 150
    f.line(px0, py0, px0 + pw, py0, "#9aa0b4", 1.5)
    for v in (1, 2, 3, 4):
        yy = py0 - (v - 1) / 3.2 * ph
        f.line(px0, yy, px0 + pw, yy, "#eceef4", 1)
        f.text(px0 - 10, yy + 5, str(v), 13, "end", color=MUTED, mono=True)
    for k in range(1, 9):
        f.text(px0 + (k - 1) / 7 * pw, py0 + 22, str(k), 13, color=MUTED, mono=True)
    f.text(px0 + pw / 2, py0 + 44, "K (num_speculative_tokens)", 13, color=MUTED)
    for name, col in (("thinking off", SOLID["q"]), ("thinking on", SOLID["c"])):
        pts = [(px0 + i / 7 * pw, py0 - (a - 1) / 3.2 * ph) for i, a in enumerate(GOLDEN_AL[name])]
        f.raw(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for a, b in pts:
            f.raw(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{col}"/>')
        f.text(pts[-1][0] + 10, pts[-1][1] + 5, f"{name} {GOLDEN_AL[name][-1]}", 13, "start", 700, col)
    kx = px0 + 2 / 7 * pw
    f.line(kx, py0, kx, py0 - ph, "#c2479c", 1.5, "4 4")
    f.text(kx + 6, py0 - ph + 10, "K = 3: 2.93 · 2.83", 13, "start", 700, "#c2479c")
    f.text(1000, y + 30, "B300 · vLLM · SPEED-Bench coding · temperature 1.0", 15, "start", color=MUTED)
    f.text(1000, y + 60, "AL은 K를 늘려도 빠르게 포화한다", 15, "start", 700)
    f.text(1000, y + 84, "draft 비용은 K에 비례 — 최적 K가 있다", 15, "start", color=MUTED)
    f.text(1000, y + 116, "모델 카드 (TP4, K=3): MT-Bench 2.70", 15, "start", color=MUTED)
    f.text(1000, y + 140, "GSM8K · HumanEval · MATH500 약 3.5", 15, "start", color=MUTED)
    f.text(1000, y + 164, "수락률 56.6 ~ 83.9% — 데이터셋에 크게 좌우", 15, "start", color=MUTED)
    return f


FIGS = [fig_layers, fig_block, fig_m25_attn, fig_m3_msa, fig_moe, fig_kv, fig_msa_concept, fig_msa_tensor, fig_msa_heads,
        fig_eagle_draft, fig_eagle_step]
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


def main():
    browser = next((c for c in CHROME if os.path.exists(c)), None)
    only = sys.argv[1:]
    for make in FIGS:
        f = make()
        if only and not any(o in f.name for o in only):
            continue
        svg = os.path.join(HERE, f.name + ".svg")
        with open(svg, "w", encoding="utf-8") as fh:
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
