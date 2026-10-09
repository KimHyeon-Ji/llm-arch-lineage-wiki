"""세미나용 그림 생성기. 실행: python make_figures.py  → 같은 폴더에 *.svg, *.png

수치의 근거
- Qwen3-235B-A22B  : HF config (94층, num_attention_heads 64, num_key_value_heads 4, head_dim 128)
- GPU              : H100 80 GB · 3.35 TB/s · BF16 dense 989 TFLOPS, B200 192 GB · 8 TB/s · 2.25 PFLOPS
- MLA              : deepseek-ai/DeepSeek-V3 inference/model.py (kv_cache 512 + pe_cache 64)
- DSA              : deepseek-ai/DeepSeek-V3.2-Exp inference/model.py, config_671B_v3.2.json
                     (index_n_heads 64, index_head_dim 128, index_topk 2048, indexer 키 캐시 FP8)
- CSA / HCA        : arXiv:2606.19348 2.3절·4.2.1절, DeepSeek-V4-Pro / Flash config.json
- QSA              : Qwen3.8-Flash-Next 공개 자료 (4토큰 micro-block, 블록 512개, GDN 36 + QSA 12)
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
OFF = "#e6e8f1"


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

    def box(self, x, y, w, h, kind, label="", sub=None, fs=20, mono=True, subsize=16):
        fill, stroke = KINDS[kind]
        self.rect(x, y, w, h, fill, stroke)
        if label:
            self.text(x + w / 2, y + h / 2 + fs * 0.35, label, fs, mono=mono)
        if sub:
            self.text(x + w / 2, y + h + subsize + 8, sub, subsize, color=MUTED, mono=True)

    def line(self, x1, y1, x2, y2, color="#8b90a3", sw=2, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#ah)"' if arrow else ""
        self.raw(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def arrow(self, x1, y1, x2, y2, dash=None, color="#5d6377"):
        self.line(x1, y1, x2, y2, color, 2.2, dash, True)

    def curve(self, x1, y1, cx1, cy1, cx2, cy2, x2, y2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<path d="M{x1},{y1} C{cx1},{cy1} {cx2},{cy2} {x2},{y2}" fill="none" stroke="#5d6377" '
                 f'stroke-width="2.2"{d} marker-end="url(#ah)"/>')

    def cells(self, x, y, colors, cw=26, ch=26, gap=2, rx=3):
        for i, c in enumerate(colors):
            self.rect(x + i * (cw + gap), y, cw, ch, c, "none", rx)
        return x + len(colors) * (cw + gap) - gap

    def stack(self, x, y, w, rows, kind, rh=18, hi_last=True):
        """S줄짜리 캐시를 세로로 그린다. 마지막 줄 = 방금 붙인 줄."""
        fill, stroke = KINDS[kind]
        for r in range(rows):
            last = hi_last and r == rows - 1
            self.rect(x, y + r * (rh + 3), w, rh, SOLID[kind] if last else fill, stroke, 3, 1.5)
        return y + rows * (rh + 3) - 3

    def svg(self):
        defs = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
                'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#5d6377"/></marker></defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}">{defs}<rect width="100%" height="100%" fill="#ffffff"/>' + "".join(self.o) + "</svg>")


# ---------------------------------------------------------------- 1. decode 한 스텝
def fig_decode():
    f = Fig("01-decode-step", 1560, 600)
    f.title(40, 44, "decode 한 스텝을 모양으로 따라가기 (헤드 하나)")
    y = 120
    f.box(40, y, 80, 40, "n", "x", "1 × d")
    f.arrow(124, y + 20, 196, y + 20)
    f.text(160, y + 8, "W_Q", 14, color=MUTED, mono=True)
    f.box(200, y, 120, 40, "q", "q", "1 × d_h")
    f.text(350, y + 28, "·", 30)
    f.box(380, y - 22, 190, 84, "k", "Kᵀ", "d_h × S")
    f.text(600, y + 27, "=", 24)
    hot = ["#d9d2f3", "#eceafa", "#8f79d8", "#eceafa", "#c4b8ec", "#eceafa", "#6b4fc8", "#b1a2e6"]
    end = f.cells(630, y + 7, hot)
    f.text((630 + end) / 2, y + 64, "점수 1 × S", 16, color=MUTED, mono=True)
    f.arrow(end + 12, y + 20, end + 108, y + 20)
    f.text(end + 60, y + 8, "softmax", 15, color=MUTED)
    px = end + 118
    end2 = f.cells(px, y + 7, hot)
    f.text((px + end2) / 2, y + 64, "p 1 × S", 16, color=MUTED, mono=True)
    f.text(end2 + 24, y + 28, "·", 30)
    vx = end2 + 48
    f.box(vx, y - 50, 120, 140, "v", "V", "S × d_h")
    f.text(vx + 150, y + 27, "=", 24)
    f.box(vx + 180, y, 120, 40, "q", "o", "1 × d_h")
    f.arrow(vx + 240, y + 76, vx + 240, y + 150)
    f.text(vx + 252, y + 120, "W_O", 14, "start", color=MUTED, mono=True)
    f.box(vx + 200, y + 154, 80, 40, "n", "y", "1 × d")

    # 캐시
    cy = 330
    kb = f.stack(250, cy, 150, 7, "k")
    vb = f.stack(430, cy, 150, 7, "v")
    f.text(250, kb + 30, "K  S × d_h", 16, "start", color=MUTED, mono=True)
    f.text(580, vb + 30, "V  S × d_h", 16, "end", color=MUTED, mono=True)
    f.text(610, cy + 18, "KV cache", 18, "start", 700)
    f.text(610, cy + 44, "지금까지 모든 토큰의 k, v", 16, "start", color=MUTED)
    # 쓰기
    f.box(40, kb - 18, 80, 30, "k", "k", fs=17)
    f.box(40, kb + 26, 80, 30, "v", "v", fs=17)
    f.line(80, y + 78, 80, kb - 22, dash="5 5")
    f.text(92, y + 150, "W_K, W_V", 14, "start", color=MUTED, mono=True)
    f.arrow(124, kb - 3, 244, kb - 9)
    f.curve(124, kb + 41, 260, kb + 78, 420, kb + 70, 440, kb + 8)
    f.text(40, kb + 100, "쓰기 — 한 줄만 붙인다", 17, "start", 700, "#c2479c")
    # 읽기
    f.curve(325, cy - 6, 325, cy - 70, 475, y + 170, 475, y + 96)
    f.curve(584, cy + 100, 900, cy + 100, vx + 60, cy + 20, vx + 60, y + 124, dash="7 6")
    f.text(486, y + 150, "읽기 ①  K 전체", 17, "start", 700, PURPLE)
    f.text(vx - 130, cy + 60, "읽기 ②  V 전체", 17, "start", 700, PURPLE)
    # 메모
    nx = 760
    f.text(nx, cy + 150, "• 입력도 한 줄(1 × d), 출력도 한 줄(1 × d)", 18, "start")
    f.text(nx, cy + 182, "• 그 사이에 길이 S짜리 텐서를 두 번 지난다 — 읽기 ①, ②", 18, "start")
    f.text(nx, cy + 214, "• 컨텍스트가 길어져도 쓰기는 한 줄, 읽기만 커진다", 18, "start")
    return f


# ---------------------------------------------------------------- 2. MHA / GQA / MQA
def fig_heads():
    f = Fig("02-head-sharing", 1560, 470)
    panels = [("MHA · KV 헤드 8개", 8, 60), ("GQA · KV 헤드 2개", 2, 580), ("MQA · KV 헤드 1개", 1, 1100)]
    for name, nkv, x0 in panels:
        f.title(x0 + 200, 50, name, 24, "middle")
        qx = [x0 + i * 52 for i in range(8)]
        for x in qx:
            f.box(x, 90, 42, 42, "q", "q", fs=18)
        g = 8 // nkv
        for j in range(nkv):
            cx = sum(qx[j * g:(j + 1) * g]) / g
            for i in range(j * g, (j + 1) * g):
                f.line(qx[i] + 21, 134, cx + 21, 278, "#a3a7b8", 1.8)
            f.box(cx, 280, 42, 42, "k", "k", fs=18)
            f.box(cx, 330, 42, 42, "v", "v", fs=18)
        f.text(x0 + 200, 410, f"캐시 {nkv}장  ·  ( S × {nkv} × d_h )", 17, color=MUTED, mono=True)
    f.text(780, 452, "쿼리 헤드 8개 기준. 쿼리 헤드 수는 그대로 두고, 캐시에 남길 K · V만 줄인다.  "
                     "(Qwen3-235B: 쿼리 64 · KV 4)", 17, color=INK)
    return f


# ---------------------------------------------------------------- 3. MLA가 저장하는 것
def fig_mla_cache():
    f = Fig("03-mla-cache", 1560, 620)
    f.title(40, 44, "MLA — 무엇을 캐시에 남기나 (DeepSeek-V3)")
    y = 110
    f.box(40, y + 50, 100, 44, "n", "x", "7168")
    # 위 갈래 : 내용
    f.arrow(144, y + 62, 256, y + 22)
    f.text(190, y + 22, "W_DKV", 14, color=MUTED, mono=True)
    f.box(260, y, 230, 44, "c", "c_KV", "512")
    f.arrow(494, y + 22, 596, y + 22)
    # 아래 갈래 : RoPE
    f.arrow(144, y + 82, 256, y + 122)
    f.text(190, y + 126, "W_KR", 14, color=MUTED, mono=True)
    f.box(260, y + 100, 70, 44, "k", "k_R", "64 · RoPE 적용 · 전 헤드 공유", subsize=15)
    f.arrow(334, y + 122, 596, y + 122)
    # 캐시
    f.rect(600, y - 26, 250, 200, "#fafbfd", "#b9bdcc", 10, 2, "7 6")
    f.text(725, y - 36, "캐시 — 토큰 · 층마다 576개 값", 17, weight=700)
    f.box(620, y, 210, 44, "c", "c_KV  512", fs=18)
    f.box(620, y + 100, 80, 44, "k", "k_R 64", fs=16)
    # 복원
    f.arrow(854, y + 14, 986, y - 6, dash="7 6")
    f.arrow(854, y + 30, 986, y + 96, dash="7 6")
    f.text(912, y - 14, "W_UK", 14, color=MUTED, mono=True)
    f.text(906, y + 86, "W_UV", 14, color=MUTED, mono=True)
    f.box(990, y - 30, 250, 44, "k", "k_C  헤드당 128", fs=17)
    f.box(990, y + 76, 250, 44, "v", "v  헤드당 128", fs=17)
    f.text(1260, y - 2, "× 128 헤드", 16, "start", color=MUTED)
    f.text(1260, y + 104, "× 128 헤드", 16, "start", color=MUTED)
    f.text(1115, y + 160, "쓸 때 만들어 내는 값 — 저장하지 않는다", 16, color=MUTED)

    # 크기 비교 막대 (길이 비례)
    by = 380
    f.text(40, by, "토큰 하나가 층 하나에 남기는 값의 개수", 19, "start", 700)
    full = 1300
    f.text(40, by + 44, "MHA", 18, "start", 600)
    f.rect(130, by + 24, full / 2, 30, KINDS["k"][0], KINDS["k"][1], 4)
    f.rect(130 + full / 2, by + 24, full / 2, 30, KINDS["v"][0], KINDS["v"][1], 4)
    f.text(130 + full / 4, by + 45, "K  128 헤드 × 128", 16, mono=True)
    f.text(130 + full * 3 / 4, by + 45, "V  128 헤드 × 128", 16, mono=True)
    f.text(130 + full, by + 78, "32,768", 18, "end", 700)
    w = full * 576 / 32768
    f.text(40, by + 134, "MLA", 18, "start", 600)
    f.rect(130, by + 114, w * 512 / 576, 30, SOLID["c"], KINDS["c"][1], 2)
    f.rect(130 + w * 512 / 576, by + 114, w * 64 / 576, 30, SOLID["k"], KINDS["k"][1], 2)
    f.text(130 + w + 16, by + 135, "576  =  c_KV 512 + k_R 64", 18, "start", 700)
    f.text(40, by + 200, "막대 길이는 실제 비율이다.  같은 헤드 구성(128 헤드 · 헤드 차원 128)의 MHA와 비교.", 16, "start",
           color=MUTED)
    return f


# ---------------------------------------------------------------- 4. MLA 흡수
def fig_mla_absorb():
    f = Fig("04-mla-absorb", 1560, 740)
    # 왼쪽 : 펴서 비교
    f.title(40, 44, "펴서 비교 — 논문 수식 그대로")
    f.stack(60, 110, 110, 7, "c", hi_last=False)
    f.text(115, 290, "캐시 c", 17, weight=600)
    f.text(115, 312, "S × 512", 16, color=MUTED, mono=True)
    f.arrow(176, 180, 286, 180)
    f.text(231, 166, "× W_UK", 15, color=MUTED, mono=True)
    for i, dx in enumerate((16, 8, 0)):
        f.stack(290 + dx, 94 + dx * 2, 250, 7, "k", hi_last=False)
    f.text(415 + 0, 306, "K  ( S × 128 )  × 128 헤드", 16, color=MUTED, mono=True)
    f.text(415, 340, "매 스텝, S개 전부를 편다", 18, weight=700, color="#c2479c")
    f.text(596, 190, "·", 30)
    f.box(622, 162, 110, 36, "q", "q_C", "1 × 128")

    f.line(780, 30, 780, 430, "#d3d6e2", 2, "6 6")

    # 오른쪽 : 쿼리를 옮겨서
    x0 = 830
    f.title(x0, 44, "쿼리를 옮겨서 비교 — 실제 추론 (absorb)")
    f.box(x0, 150, 110, 36, "q", "q_C", "1 × 128")
    f.arrow(x0 + 114, 168, x0 + 226, 168)
    f.text(x0 + 170, 154, "× W_UKᵀ", 15, color=MUTED, mono=True)
    f.box(x0 + 230, 150, 230, 36, "q", "q̃", "1 × 512  ·  헤드마다 다르다")
    f.text(x0 + 345, 250, "쿼리 1개만 옮긴다", 18, weight=700, color=PURPLE)
    f.text(x0 + 492, 176, "·", 30)
    f.stack(x0 + 530, 96, 150, 7, "c", hi_last=False)
    f.text(x0 + 605, 276, "캐시 c", 17, weight=600)
    f.text(x0 + 605, 298, "S × 512  ·  그대로", 16, color=MUTED, mono=True)
    f.text(x0 + 345, 330, "K를 만들지 않는다", 17, color=MUTED)

    f.rect(40, 450, 1480, 270, "#f6f5fd", "#d9d2f3", 10, 1.5)
    ex, nx = 150, 800
    # 1행 : 원래 식 + 키의 정체
    f.text(ex + 34, 500, "q_C · k_Cᵀ", 24, "start", 700, mono=True)
    f.text(ex + 196, 500, "원래의 점수", 16, "start", color=MUTED)
    f.rect(nx - 16, 470, 300, 46, KINDS["k"][0], KINDS["k"][1], 8, 2.5)
    f.text(nx + 134, 501, "k_C = c · W_UK", 23, weight=700, color="#a8327f", mono=True)
    f.text(nx + 300, 500, "키는 캐시의 c에서 만들어진다", 16, "start", color=MUTED)
    rows = [(552, "= q_C · (c · W_UK)ᵀ", "k_C 자리에 넣는다"),
            (604, "= q_C · W_UKᵀ · cᵀ", "전치하면 곱의 순서가 뒤집힌다   (A · B)ᵀ = Bᵀ · Aᵀ"),
            (656, "= (q_C · W_UKᵀ) · cᵀ  =  q̃ · cᵀ", "앞의 둘을 먼저 곱한다 (결합 법칙)")]
    for y, eq, note in rows:
        f.text(ex, y, eq, 24, "start", 700, mono=True)
        f.text(nx - 16, y, note, 16, "start", color=MUTED)
    f.text(ex, 700, "값은 그대로다. 곱하는 순서만 바뀌었다.    전체 점수 = q̃ · cᵀ + q_R · k_Rᵀ (RoPE 64차원 몫)", 16, "start",
           color=MUTED)
    return f


# ---------------------------------------------------------------- 5. 읽는 패턴
def fig_patterns():
    f = Fig("05-read-patterns", 1560, 420)
    f.title(40, 44, "지금 토큰이 캐시에서 읽는 위치 — 개수보다 조각 수")
    on, off = "#6b4fc8", OFF
    rows = [
        ("전부 읽기", [1] * 16, "연속 1구간"),
        ("최근 W개 (SWA, W = 4)", [0] * 12 + [1] * 4, "연속 1구간"),
        ("블록 단위로 고름 (블록 4 · 2개)", [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], "2구간"),
        ("토큰 단위로 고름", [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], "6구간"),
    ]
    for r, (name, pat, note) in enumerate(rows):
        y = 90 + r * 70
        f.text(40, y + 28, name, 19, "start", 600)
        end = f.cells(400, y, [on if p else off for p in pat], 44, 40, 5, 5)
        f.text(end + 30, y + 28, note, 19, "start", 700 if r >= 2 else 400, "#c2479c" if r == 3 else INK)
    f.text(400, 388, "0", 15, color=MUTED, mono=True)
    f.text(400 + 15 * 49 + 22, 388, "15 (지금)", 15, color=MUTED, mono=True)
    f.text(40, 388, "메모리는 이어진 구간을 한 번에 읽을 때 가장 빠르다.", 16, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 6. DSA
def fig_dsa():
    f = Fig("06-dsa", 1560, 600)
    f.title(40, 44, "DSA — 같은 컨텍스트를 두 번 훑는다 (DeepSeek-V3.2)")
    n = 28
    sel = {3, 9, 12, 19, 24}
    x0, cw = 420, 34
    # ① indexer
    y = 100
    f.text(40, y + 22, "① lightning indexer", 20, "start", 700, "#3b7dd8")
    f.text(40, y + 48, "모든 토큰에 값싼 점수를 매긴다", 16, "start", color=MUTED)
    f.cells(x0, y, [KINDS["i"][0]] * n, cw, 34, 4, 4)
    heights = [5, 9, 4, 30, 8, 6, 11, 5, 7, 27, 6, 9, 25, 5, 8, 10, 4, 7, 6, 31, 9, 5, 8, 6, 28, 7, 5, 9]
    for i, h in enumerate(heights):
        f.rect(x0 + i * (cw + 4) + 9, y + 78 - h, 16, h, "#3b7dd8" if i in sel else "#b9d3f5", "none", 2)
    f.text(40, y + 76, "I(t,s) = Σ w_h · ReLU( q_h · k_s )", 15, "start", color=MUTED, mono=True)
    # ② top-k
    y2 = 230
    f.text(40, y2 + 22, "② top-k", 20, "start", 700)
    f.text(40, y2 + 48, "점수가 높은 2,048개의 위치", 16, "start", color=MUTED)
    f.cells(x0, y2, [SOLID["i"] if i in sel else OFF for i in range(n)], cw, 34, 4, 4)
    for i in sel:
        f.arrow(x0 + i * (cw + 4) + 17, y2 + 40, x0 + i * (cw + 4) + 17, y2 + 92)
    # ③ MLA
    y3 = 330
    f.text(40, y3 + 22, "③ MLA attention", 20, "start", 700, "#b87410")
    f.text(40, y3 + 48, "고른 위치의 latent만 읽는다", 16, "start", color=MUTED)
    f.cells(x0, y3, [SOLID["c"] if i in sel else OFF for i in range(n)], cw, 34, 4, 4)
    f.text(x0, y3 + 62, "0", 14, color=MUTED, mono=True)
    f.text(x0 + (n - 1) * (cw + 4) + 17, y3 + 62, "S (지금)", 14, color=MUTED, mono=True)

    # 비교 표
    ty = 450
    f.rect(40, ty, 720, 120, "#f1f6fe", "#b9d3f5", 10, 1.5)
    f.text(64, ty + 34, "indexer — 넓고 얕게", 19, "start", 700, "#3b7dd8")
    f.text(64, ty + 64, "범위  S 전부   ·   키  128차원 1개 (FP8)", 17, "start")
    f.text(64, ty + 94, "쿼리  64 헤드 × 128   ·   토큰마다 키 캐시가 따로 든다", 17, "start")
    f.rect(800, ty, 720, 120, "#fef7ea", "#f0cf93", 10, 1.5)
    f.text(824, ty + 34, "MLA attention — 좁고 깊게", 19, "start", 700, "#b87410")
    f.text(824, ty + 64, "범위  k = 2,048   ·   캐시  c_KV 512 + k_R 64", 17, "start")
    f.text(824, ty + 94, "쿼리  128 헤드 × 512   ·   출력은 여기서만 만든다", 17, "start")
    return f


# ---------------------------------------------------------------- 7. CSA / HCA
def _tokens(f, x, y, n, r=9, gap=6, color="#b7bccd"):
    for i in range(n):
        f.raw(f'<circle cx="{x + i * (2 * r + gap) + r}" cy="{y}" r="{r}" fill="{color}"/>')
    return 2 * r + gap


def fig_v4():
    f = Fig("07-v4-csa-hca", 1560, 660)
    # CSA
    f.title(40, 44, "CSA — 4개씩 묶고, 그 위에서 고른다")
    step, pitch = 24, 112
    for j in range(5):
        _tokens(f, 60 + j * pitch, 110, 4)
    f.text(60 + 5 * pitch + 8, 116, "원본 토큰  S", 16, "start", color=MUTED)
    ex = []
    for j in range(5):
        gx = 60 + j * pitch
        cx = gx + 45
        f.rect(gx - 6, 95, 102, 30, "none", "#c9cddb", 8, 1.5)
        bx = cx - 34
        ex.append(bx)
        for i in range(4):  # 자기 블록 (C^a)
            f.line(gx + i * step + 9, 126, cx, 196, "#8b90a3", 1.6)
        if j > 0:  # 앞 블록과 겹침 (C^b)
            for i in range(4):
                f.line(gx - pitch + i * step + 9, 126, cx, 196, "#c9cddb", 1.4, "4 4")
    picked = {1, 3}
    for j, bx in enumerate(ex):
        f.rect(bx, 200, 68, 34, SOLID["c"] if j in picked else KINDS["c"][0], KINDS["c"][1], 5)
    f.text(60 + 5 * pitch + 8, 222, "압축 엔트리  S / 4", 16, "start", color=MUTED)
    f.text(60, 270, "엔트리 하나에 토큰 8개가 반영된다 (자기 4개 + 앞의 4개, 점선).  가중치는 학습된다.", 15, "start",
           color=MUTED)
    f.text(60, 304, "→  indexer(FP4)가 엔트리마다 점수를 매겨 상위 k개만 고른다  (V4-Pro 1,024 · V4-Flash 512)", 17,
           "start", 600)

    # HCA
    y0 = 360
    f.title(40, y0, "HCA — 128개씩 묶고, 전부 본다")
    for j in range(2):
        gx = 60 + j * 300
        f.rect(gx, y0 + 34, 270, 28, "#dfe2ec", "#c9cddb", 8, 1.5)
        f.text(gx + 135, y0 + 54, "토큰 128개", 15, color=INK)
        f.arrow(gx + 135, y0 + 66, gx + 135, y0 + 106)
        f.rect(gx + 101, y0 + 110, 68, 34, SOLID["c"], KINDS["c"][1], 5)
    f.text(640, y0 + 54, "원본 토큰  S", 16, "start", color=MUTED)
    f.text(640, y0 + 132, "압축 엔트리  S / 128  — 고르지 않는다. 겹침도 없다.", 16, "start", color=MUTED)

    # 공통 : sliding window
    f.rect(840, 70, 680, 190, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(864, 104, "두 층 모두에 있는 것", 19, "start", 700, PURPLE)
    f.text(864, 138, "• 압축하지 않은 최근 128토큰 (sliding window 가지)", 17, "start")
    f.text(864, 168, "   — 자기 블록 안의 토큰은 압축 엔트리로 볼 수 없어서", 15, "start", color=MUTED)
    f.text(864, 200, "• 엔트리 하나가 K이자 V  (512차원 · KV 헤드 1개 · MQA)", 17, "start")
    f.text(864, 232, "• RoPE는 마지막 64차원에만", 17, "start")

    # 1M 숫자
    ny = 560
    f.rect(40, ny - 30, 1480, 110, "#fafbfd", "#d3d6e2", 10, 1.5)
    f.text(64, ny + 4, "1M 토큰이면", 18, "start", 700)
    for dy, cells in ((4, ("CSA", "1,000,000", "→  250,000 엔트리", "→  상위 1,024개 + 최근 128토큰", "좁고 정밀")),
                      (44, ("HCA", "1,000,000", "→  7,812 엔트리", "→  전부 + 최근 128토큰", "넓고 거침"))):
        for cx_, c_ in zip((240, 320, 470, 720, 1130), cells):
            f.text(cx_, ny + dy, c_, 18, "start", 700 if c_ in ("CSA", "HCA") else 400)
    return f


# ---------------------------------------------------------------- 8. V4 층 배치
def fig_v4_layers():
    f = Fig("08-v4-layers", 1560, 400)
    f.title(40, 44, "DeepSeek-V4의 층 배치 — config.json의 compress_ratios 그대로")
    col = {"csa": SOLID["c"], "hca": "#8f79d8", "swa": "#9aa0b4"}

    def row(y, name, kinds, note):
        f.text(40, y + 24, name, 19, "start", 700)
        for i, k in enumerate(kinds):
            f.rect(230 + i * 21, y, 18, 34, col[k], "none", 3)
        f.text(230, y + 58, "0", 14, color=MUTED, mono=True)
        f.text(230 + (len(kinds) - 1) * 21 + 9, y + 58, str(len(kinds) - 1), 14, color=MUTED, mono=True)
        f.text(230, y + 84, note, 16, "start", color=MUTED)

    pro = ["hca", "hca"] + ["csa" if i % 2 == 0 else "hca" for i in range(2, 61)]
    flash = ["swa", "swa"] + ["csa" if i % 2 == 0 else "hca" for i in range(2, 43)]
    row(90, "V4-Pro · 61층", pro, f"CSA {pro.count('csa')}층  ·  HCA {pro.count('hca')}층  —  첫 두 층은 HCA")
    row(220, "V4-Flash · 43층", flash,
        f"CSA {flash.count('csa')}층  ·  HCA {flash.count('hca')}층  —  첫 두 층은 순수 sliding window")
    lx = 230
    for k, label in (("csa", "CSA  (압축률 4, 고른다)"), ("hca", "HCA  (압축률 128, 전부 본다)"),
                     ("swa", "sliding window만")):
        f.rect(lx, 350, 18, 24, col[k], "none", 3)
        f.text(lx + 28, 368, label, 16, "start")
        lx += 330
    return f


# ---------------------------------------------------------------- 9. QSA
def fig_qsa():
    f = Fig("09-qsa", 1560, 560)
    f.title(40, 44, "QSA — 4토큰 블록 단위로 고른다 (Qwen3.8-Flash-Next)")
    step = 30
    n = 22
    picked = {1, 3}
    # 토큰
    pitch = 4 * step + 14
    for i in range(n):
        b = i // 4
        on = b in picked or i >= 20
        f.rect(60 + b * pitch + (i % 4) * step, 100, 24, 30, "#6b4fc8" if on else OFF, "none", 4)
    for b in range(5):
        gx = 60 + b * pitch
        f.rect(gx - 6, 94, 4 * step + 6, 42, "none", "#b9bdcc", 8, 1.5)
        f.arrow(gx + 2 * step - 3, 140, gx + 2 * step - 3, 188)
        f.rect(gx + 2 * step - 37, 192, 68, 32, SOLID["i"] if b in picked else KINDS["i"][0], KINDS["i"][1], 5)
    f.rect(60 + 5 * pitch - 6, 94, 2 * step + 6, 42, "none", "#6b4fc8", 8, 1.5, "5 4")
    f.text(60 + 5 * pitch + step - 3, 160, "아직 덜 찬 블록", 15, color=PURPLE)
    f.text(60 + 5 * pitch + step - 3, 180, "항상 읽는다", 15, color=PURPLE)
    f.text(860, 122, "토큰 — 본 attention의 KV는 토큰 단위 그대로 둔다", 16, "start", color=MUTED)
    f.text(860, 214, "블록 키 = 색인 키 4개의 평균.  색인 쿼리 헤드는 4개", 16, "start", color=MUTED)
    f.text(60, 270, "→  indexer가 블록마다 점수를 매겨 상위 512개 블록(= 2,048토큰)을 고른다", 17, "start", 600)
    f.text(60, 302, "→  고른 블록 안의 토큰 4개를 전부 읽는다.  읽는 위치가 4개씩 이어져 있다.", 17, "start", 600)

    # 층 배치
    y = 380
    f.text(40, y, "층 배치 · 48층", 19, "start", 700)
    for i in range(48):
        q = i % 4 == 3
        f.rect(230 + i * 26, y + 20, 22, 34, "#6b4fc8" if q else "#cfd3e0", "none", 3)
    f.rect(230, y + 80, 22, 24, "#cfd3e0", "none", 3)
    f.text(262, y + 98, "Gated DeltaNet 36층 — 고정 크기 상태를 고쳐 쓴다. KV cache 없음", 16, "start")
    f.rect(880, y + 80, 22, 24, "#6b4fc8", "none", 3)
    f.text(912, y + 98, "QSA 12층 — KV cache 있음", 16, "start")
    f.text(230, y + 140, "3 : 1 반복.  KV cache가 길이에 따라 자라는 층은 넷 중 하나뿐이다.", 16, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 10. 캐시의 실제 크기
def fig_cache_size():
    f = Fig("10-cache-size", 1560, 600)
    f.title(40, 44, "KV cache의 크기 — Qwen3-235B-A22B")
    chips = [("2", "K, V", "n"), ("94", "층  L", "q"), ("S", "길이", "c"), ("4", "KV 헤드  n_kv", "k"),
             ("128", "헤드 차원  d_h", "v"), ("2 B", "BF16", "n")]
    x = 60
    for i, (big, small, kind) in enumerate(chips):
        f.box(x, 90, 150, 60, kind, big, fs=26)
        f.text(x + 75, 178, small, 16, color=MUTED)
        if i < len(chips) - 1:
            f.text(x + 172, 130, "×", 24, color=MUTED)
        x += 194
    f.text(x + 10, 130, "×  요청 수", 22, "start", 600)
    rows = [("토큰 하나", "2 × 94 × 4 × 128 × 2 B  =  192,512 B  ≈  192.5 KB"),
            ("32K 대화 하나", "192.5 KB × 32,768 토큰  ≈  6.3 GB"),
            ("1M 토큰이라면", "192.5 KB × 1,000,000 토큰  ≈  192.5 GB")]
    for i, (k, v) in enumerate(rows):
        f.text(60, 240 + i * 38, k, 19, "start", 700)
        f.text(240, 240 + i * 38, v, 19, "start", mono=True)

    gx, gy, gw = 60, 400, 1440
    f.text(gx, gy - 14, "H100 한 장 · 80 GB", 18, "start", 700)
    f.rect(gx, gy, gw, 70, "#f4f5f9", "#b9bdcc", 8, 2)
    bw = gw * 6.308 / 80
    for i in range(12):
        f.rect(gx + i * bw + 3, gy + 6, bw - 6, 58, KINDS["q"][0], KINDS["q"][1], 6, 1.5)
        f.text(gx + i * bw + bw / 2, gy + 42, "32K", 16)
    f.text(gx + 12 * bw + (gw - 12 * bw) / 2, gy + 42, "4.3 GB", 13, color=MUTED)
    f.text(gx, gy + 110, "가중치를 하나도 올리지 않아도 32K 대화 열두 개에서 찬다.  "
                         "1M 토큰 대화는 하나가 B200 한 장(192 GB)을 넘는다.", 18, "start")
    f.text(gx, gy + 142, "KV 헤드를 4개로 줄인 GQA의 숫자다. 헤드마다 따로 뒀다면(KV 헤드 64) 여기에 16을 곱해야 한다.", 16, "start",
           color=MUTED)
    return f


# ---------------------------------------------------------------- 11. 줄일 수 있는 곳 셋
def fig_knobs():
    f = Fig("11-three-knobs", 1560, 500)
    f.title(40, 44, "KV cache를 줄이는 세 방향")
    # 층이 겹친 캐시 블록
    for d in (40, 20, 0):
        f.rect(110 + d, 160 - d, 260, 240, "#f4f2fc" if d else "#eceafa", "#6b4fc8", 6, 1.8)
    for r in range(9):
        f.rect(122, 174 + r * 24, 236, 16, "#6b4fc8" if r in (2, 6) else "#cfc6ef", "none", 3)
    # 치수선
    f.line(110, 424, 370, 424, PURPLE, 2)
    f.line(110, 416, 110, 432, PURPLE, 2)
    f.line(370, 416, 370, 432, PURPLE, 2)
    f.text(240, 454, "① 토큰당 저장량", 18, weight=700, color=PURPLE)
    f.line(84, 160, 84, 400, "#3b7dd8", 2)
    f.line(76, 160, 92, 160, "#3b7dd8", 2)
    f.line(76, 400, 92, 400, "#3b7dd8", 2)
    f.raw(f'<text transform="translate(64,280) rotate(-90)" font-family="{FONT}" font-size="18" text-anchor="middle" '
          f'font-weight="700" fill="#3b7dd8">② 길이 S 중 읽는 줄</text>')
    f.line(376, 154, 412, 118, "#4d9a56", 2)
    f.text(424, 112, "③ 층 수", 18, "start", 700, "#4d9a56")

    rows = [("①", "토큰당 저장량을 줄인다", "Compressed Attention", "MQA · GQA · MLA", "쓰기에서 무엇을 남길지 바꾼다", PURPLE, "q"),
            ("②", "읽는 위치를 줄인다", "Sparse Attention", "SWA · DSA · CSA · HCA · QSA", "읽기에서 얼마나 훑을지 바꾼다",
             "#3b7dd8", "i"),
            ("③", "층 사이 중복을 줄인다", "Layer Sharing", "CLA · YOCO · IndexShare", "①②와 겹쳐 쓸 수 있다",
             "#4d9a56", "v")]
    for i, (n, what, fam, who, note, col, kind) in enumerate(rows):
        y = 110 + i * 118
        f.rect(560, y, 940, 96, KINDS[kind][0], KINDS[kind][1], 10, 1.5)
        f.text(590, y + 40, f"{n}  {what}", 22, "start", 700, col)
        f.text(1476, y + 40, fam, 18, "end", 600)
        f.text(590, y + 74, who, 17, "start", mono=True)
        f.text(1476, y + 74, note, 16, "end", color=MUTED)
    return f


# ---------------------------------------------------------------- 12. 오늘 따라갈 길
def fig_lineage():
    f = Fig("12-lineage", 1560, 640)
    f.title(40, 44, "기법의 계보")

    def node(x, y, name, year, who, kind, w=250, dash=None):
        f.rect(x, y, w, 62, KINDS[kind][0], KINDS[kind][1], 10, 2, dash)
        f.text(x + 18, y + 28, name, 21, "start", 700)
        f.text(x + w - 16, y + 28, year, 16, "end", color=MUTED, mono=True)
        f.text(x + 18, y + 51, who, 14, "start", color=MUTED)

    node(655, 80, "MHA", "2017", "GPT-2/3 · Llama 2 7B", "n")
    lx, rx = 250, 1010
    f.text(lx + 125, 196, "저장을 줄인다", 19, weight=700, color=PURPLE)
    f.text(rx + 125, 196, "읽기를 줄인다", 19, weight=700, color="#3b7dd8")
    f.curve(700, 146, 640, 200, 560, 150, lx + 200, 176)
    f.curve(860, 146, 920, 200, 1000, 150, rx + 50, 176)
    node(lx, 216, "MQA", "2019", "PaLM · Falcon", "q")
    node(lx, 316, "GQA", "2023", "Llama 3 · Gemma 3 · Qwen3", "q")
    node(lx, 416, "MLA", "2024", "DeepSeek-V2/V3 · Kimi K2 · GLM-5", "q")
    for y in (278, 378):
        f.arrow(lx + 125, y + 2, lx + 125, y + 34)
    node(rx, 216, "SWA", "2023", "Mistral 7B · Gemma 2/3", "i")
    node(rx, 416, "DSA", "2025", "DeepSeek-V3.2 · GLM-5", "i")
    node(rx, 516, "CSA · HCA", "2026", "DeepSeek-V4", "i")
    node(rx + 300, 516, "QSA", "2026", "Qwen3.8-Flash-Next", "i", 210, "6 5")
    f.arrow(rx + 125, 280, rx + 125, 412)
    f.arrow(rx + 125, 480, rx + 125, 512)
    f.line(rx + 252, 547, rx + 298, 547, "#8b90a3", 2, "5 5")
    f.arrow(lx + 254, 447, rx - 6, 447)
    f.text((lx + 250 + rx) / 2, 436, "MLA 위에 얹는다", 16, color=MUTED)
    f.text(rx + 268, 252, "위치로 고른다", 17, "start")
    f.text(rx + 268, 452, "내용으로 고른다", 17, "start")
    f.text(rx + 125, 606, "묶고 나서 고른다", 17)
    f.text(rx + 405, 606, "다른 회사, 같은 결론", 17)
    return f


# ---------------------------------------------------------------- 13. MHA
def fig_mha():
    f = Fig("13-mha", 1560, 500)
    f.title(40, 44, "MHA — 헤드마다 Q, K, V가 따로 있다")
    f.box(40, 190, 80, 44, "n", "x", "d")
    lanes = [(90, "헤드 1"), (190, "헤드 2"), (330, "헤드 n_h")]
    for y, name in lanes:
        f.curve(124, 212, 180, 212, 180, y + 22, 236, y + 22)
        f.text(240, y - 10, name, 15, "start", color=MUTED)
        f.box(240, y, 60, 44, "q", "q", fs=18)
        f.box(308, y, 60, 44, "k", "k", fs=18)
        f.box(376, y, 60, 44, "v", "v", fs=18)
        f.arrow(440, y + 22, 516, y + 22)
        f.rect(520, y, 190, 44, "#ffffff", "#8b90a3", 8, 1.8)
        f.text(615, y + 28, "softmax(q·Kᵀ)·V", 16, mono=True)
        f.curve(714, y + 22, 780, y + 22, 780, 212, 836, 212)
        f.stack(1160, y - 6, 110, 3, "k", 12)
        f.stack(1280, y - 6, 110, 3, "v", 12)
    f.text(338, 286, "⋮", 26, color=MUTED)
    f.text(1275, 286, "⋮", 26, color=MUTED)
    f.box(840, 190, 100, 44, "n", "concat", fs=16, mono=False)
    f.arrow(944, 212, 996, 212)
    f.text(970, 198, "W_O", 13, color=MUTED, mono=True)
    f.box(1000, 190, 70, 44, "n", "y", "d")
    f.text(1275, 60, "캐시 — 헤드마다 K, V 한 장씩", 17, weight=700)
    f.text(1275, 410, "( S × n_h × d_h ) × 2", 16, color=MUTED, mono=True)
    f.text(40, 450, "d = n_h × d_h 이므로 헤드를 늘려도 전체 계산량은 그대로다.  늘어나는 것은 캐시의 장수다.", 18, "start")
    f.text(40, 480, "예: GPT-2 small  768 = 12 헤드 × 64", 16, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 14 / 17. 읽기의 모양
def _read_row(f, y, name, kw, klabel, hi):
    f.text(40, y + 38, name, 20, "start", 700)
    for dx in (12, 6, 0):
        f.rect(200 + dx, y + 12 - dx, 190, 34, KINDS["q"][0], KINDS["q"][1], 6)
    f.text(295, y + 35, "q̃", 19, mono=True)
    f.text(301, y + 78, "n_h × 1 × 512", 15, color=MUTED, mono=True)
    f.text(430, y + 40, "·", 30)
    f.rect(460, y - 6, kw, 70, KINDS["c"][0], KINDS["c"][1], 6)
    f.text(460 + kw / 2, y + 36, "cᵀ", 19, mono=True)
    f.text(460 + kw / 2, y + 90, klabel, 15, weight=700 if hi else 400, color="#b87410" if hi else MUTED, mono=True)
    f.text(460 + kw + 34, y + 38, "=", 24)
    sx = 460 + kw + 66
    for dx in (12, 6, 0):
        f.rect(sx + dx, y + 12 - dx, kw, 34, KINDS["q"][0], KINDS["q"][1], 6)
    f.text(sx + kw / 2 + 6, y + 78, klabel.replace("512", "n_h × 1"), 15, color=MUTED, mono=True)


def fig_mla_shape():
    f = Fig("14-mla-read-shape", 1560, 250)
    f.title(40, 44, "MLA의 읽기 — 쿼리는 헤드마다 다르고, 캐시는 하나다")
    _read_row(f, 100, "MLA", 420, "512 × S", False)
    f.text(1500, 138, "+ q_R · k_Rᵀ", 17, "end", color=MUTED, mono=True)
    f.text(1500, 164, "RoPE 64차원 몫", 14, "end", color=MUTED)
    return f


def fig_dsa_shape():
    f = Fig("17-dsa-read-shape", 1560, 380)
    f.title(40, 44, "MLA → DSA — 읽기에서 바뀐 것은 글자 하나")
    _read_row(f, 100, "MLA", 420, "512 × S", False)
    _read_row(f, 240, "DSA", 70, "512 × k", True)
    f.text(800, 282, "k = 2,048.  컨텍스트가 100만이어도 비싼 계산은 2,048줄만 본다.", 18, "start")
    return f


# ---------------------------------------------------------------- 15. SWA 마스크
def fig_swa():
    f = Fig("15-swa-mask", 1560, 520)
    on, skip, fut = "#6b4fc8", "#ebe8f8", "#e3e5ee"

    def panel(x0, name, cap, rule):
        f.title(x0 + 144, 50, name, 22, "middle")
        for t in range(8):
            f.text(x0 - 14, 96 + t * 36 + 22, str(t), 14, "end", color=MUTED, mono=True)
            for j in range(8):
                c = fut if j > t else (on if rule(t, j) else skip)
                f.rect(x0 + j * 36, 96 + t * 36, 33, 33, c, "none", 4)
        f.text(x0 + 144, 414, cap, 16, color=MUTED)

    panel(110, "전부 읽기", "캐시  S줄", lambda t, j: True)
    panel(620, "최근 W개만 (W = 3)", "캐시  min(S, W)줄", lambda t, j: t - j < 3)
    panel(1130, "+ 맨 앞 토큰은 남긴다", "캐시  W줄 + 앞 몇 줄", lambda t, j: t - j < 3 or j == 0)
    lx = 110
    for c, label, adv in ((on, "읽는다", 150), (skip, "지나간 토큰 — 읽지 않는다", 300), (fut, "아직 오지 않은 토큰", 0)):
        f.rect(lx, 456, 24, 24, c, "none", 4)
        f.text(lx + 34, 475, label, 16, "start")
        lx += adv
    f.text(1450, 475, "행 = 지금 만드는 토큰,  열 = 과거 토큰", 15, "end", color=MUTED)
    return f


# ---------------------------------------------------------------- 16. 저장 × 읽기
def fig_store_read():
    f = Fig("16-store-read", 1560, 500)
    f.title(40, 44, "지금까지 — 토큰당 저장량과 읽는 줄 수")
    rows = 14

    def panel(x, w, kind, name, cap, picked=None):
        for r in range(rows):
            on = picked is None or r in picked
            f.rect(x, 110 + r * 20, w, 15, SOLID[kind] if on else OFF, "none", 2)
        f.text(x, 426, name, 20, "start", 700)
        f.text(x, 454, cap, 16, "start", color=MUTED)

    panel(60, 570, "k", "MHA", "크게 저장  ×  전부 읽음")
    panel(790, 10, "c", "MLA", "작게 저장  ×  전부 읽음")
    panel(1130, 10, "c", "원하는 것", "작게 저장  ×  일부만 읽음", {2, 7, 8, 12})
    f.line(40, 106, 40, 386, "#8b90a3", 2)
    f.raw(f'<text transform="translate(28,246) rotate(-90)" font-family="{FONT}" font-size="15" text-anchor="middle" '
          f'fill="{MUTED}">길이 S</text>')
    f.text(60, 96, "토큰당 32,768개 값", 15, "start", color=MUTED)
    f.text(790, 96, "576개 값", 15, "start", color=MUTED)
    f.text(1130, 96, "576개 값", 15, "start", color=MUTED)
    f.text(830, 240, "토큰 하나는 얇아졌다.", 17, "start")
    f.text(830, 268, "줄 수는 그대로다.", 17, "start", 700)
    f.text(1170, 254, "관련 있는 줄만.", 17, "start", 700, "#b87410")
    f.text(60, 486, "막대의 가로 길이는 실제 비율이다 (32,768 : 576).", 15, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 18. DSA 구조
def fig_dsa_structure():
    f = Fig("18-dsa-structure", 1560, 620)
    f.title(40, 44, "DSA의 구조 — 고르는 길과 보는 길 (DeepSeek-V3.2)")
    f.box(40, 250, 90, 50, "n", "h_t", "지금 토큰", mono=True)
    # 고르는 길
    f.text(210, 96, "고르는 길 — lightning indexer", 19, "start", 700, "#3b7dd8")
    f.curve(134, 262, 170, 262, 170, 152, 206, 152)
    f.box(210, 126, 190, 52, "i", "색인 쿼리", "64 헤드 × 128", mono=False, fs=18)
    f.text(428, 162, "·", 30)
    f.box(456, 116, 330, 72, "i", "색인 키 캐시", "S × 128  ·  FP8  ·  헤드 공유", mono=False, fs=18)
    f.arrow(790, 152, 846, 152)
    sc = ["#b9d3f5"] * 10
    for i in (2, 5, 8):
        sc[i] = "#3b7dd8"
    end = f.cells(850, 139, sc, 24, 26, 3)
    f.text((850 + end) / 2, 196, "점수 1 × S", 15, color=MUTED, mono=True)
    f.arrow(end + 8, 152, end + 62, 152)
    f.box(end + 66, 126, 170, 52, "i", "top-k", "2,048개의 위치", mono=False, fs=18)
    tx = end + 66 + 85
    # 보는 길
    f.text(210, 340, "보는 길 — MLA attention", 19, "start", 700, "#b87410")
    f.curve(134, 288, 170, 288, 170, 402, 206, 402)
    f.box(210, 376, 190, 52, "q", "쿼리 q̃", "128 헤드 × ( 512 + 64 )", mono=False, fs=18)
    f.box(456, 366, 330, 72, "c", "MLA 캐시", "S × ( 512 + 64 )", mono=False, fs=18)
    f.arrow(790, 402, 846, 402)
    f.box(850, 376, 250, 52, "c", "고른 k줄만", "k × ( 512 + 64 )", mono=False, fs=18)
    f.arrow(1104, 402, 1166, 402)
    f.box(1170, 376, 150, 52, "n", "attention", mono=False, fs=18)
    f.arrow(1324, 402, 1376, 402)
    f.box(1380, 376, 80, 52, "q", "o")
    f.raw('<path d="M404,402 C440,402 424,522 560,522 L1150,522 C1230,522 1245,500 1245,436" fill="none" '
          'stroke="#5d6377" stroke-width="2.2" marker-end="url(#ah)"/>')
    f.text(855, 510, "쿼리는 고른 k줄하고만 곱한다", 16, color=MUTED)
    f.curve(tx, 212, tx, 300, 975, 290, 975, 370, dash="7 6")
    f.text(tx + 14, 268, "위치만 넘겨준다", 16, "start", color=MUTED)
    f.text(40, 566, "새 토큰이 오면 두 캐시에 각각 한 줄씩 쓴다.  출력을 만드는 것은 아래 길뿐이고, 위 길은 어디를 볼지만 정한다.", 17,
           "start")
    f.text(40, 596, "점수  I(t,s) = Σ_h w_h · ReLU( q_h · k_s )   —  w_h는 지금 토큰에서 나오는 헤드별 가중치", 15, "start",
           color=MUTED, mono=True)
    return f


# ---------------------------------------------------------------- 19. 토큰 하나가 지나가는 길
def fig_journey():
    f = Fig("19-token-journey", 1560, 520)
    f.title(40, 44, "decode 한 스텝 — 토큰 하나가 지나가는 길")
    y = 170
    f.box(40, y, 130, 56, "n", "Embedding", mono=False, fs=17)
    f.arrow(174, y + 28, 226, y + 28)
    # 레이어 묶음
    f.rect(230, 96, 1010, 250, "#fafbfd", "#b9bdcc", 12, 2, "7 6")
    f.text(250, 124, "레이어 — L번 반복", 17, "start", 700)
    f.rect(258, 142, 590, 112, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(274, 164, "Attention", 16, "start", 700, PURPLE)
    f.box(274, 178, 150, 56, "q", "q, k, v 만들기", mono=False, fs=16)
    f.arrow(428, 206, 456, 206)
    f.box(460, 178, 150, 56, "k", "캐시에 쓰기", mono=False, fs=16)
    f.arrow(614, 206, 642, 206)
    f.rect(646, 178, 186, 56, SOLID["q"], SOLID["q"], 6)
    f.text(739, 213, "캐시 전체 읽기", 17, weight=700, color="#ffffff")
    f.arrow(852, 198, 906, 198)
    f.rect(910, 142, 300, 112, KINDS["v"][0], KINDS["v"][1], 10, 1.5)
    f.text(1060, 190, "FFN", 22, weight=700)
    f.text(1060, 222, "파라미터의 대부분", 15, color=MUTED)
    f.arrow(1244, y + 28, 1296, y + 28)
    f.box(1300, y, 130, 56, "n", "LM Head", mono=False, fs=17)
    f.text(1365, y + 84, "다음 토큰", 15, color=MUTED)
    # 주석
    f.line(739, 238, 739, 384, PURPLE, 2, "5 5")
    f.text(739, 410, "컨텍스트 길이 S에 비례하는 유일한 단계", 19, weight=700, color=PURPLE)
    f.text(739, 440, "길어질수록 이 칸의 비중이 커진다", 16, color=MUTED)
    f.line(1060, 258, 1060, 384, "#4d9a56", 2, "5 5")
    f.text(1150, 410, "짧은 컨텍스트에서 가장 비싼 단계", 19, weight=700, color="#4d9a56")
    f.text(1150, 440, "토큰마다 가중치를 전부 읽는다 — S와 무관", 16, color=MUTED)
    f.text(349, 300, "가중치 읽기", 15, color=MUTED)
    f.text(535, 300, "한 줄", 15, color=MUTED)
    return f


# ---------------------------------------------------------------- 20. 나눠 쓰는 것과 못 나눠 쓰는 것
def fig_amortize():
    f = Fig("20-weights-vs-kv", 1560, 640)
    f.title(40, 44, "가중치는 요청들이 나눠 쓰고, KV cache는 나눠 쓰지 못한다")
    # 왼쪽 : 가중치
    f.text(380, 96, "가중치", 21, weight=700)
    for i in range(4):
        f.box(120 + i * 140, 130, 100, 40, "q", f"요청 {i + 1}", mono=False, fs=16)
        f.line(170 + i * 140, 172, 380, 248, "#a3a7b8", 1.8)
    f.rect(200, 250, 360, 80, KINDS["v"][0], KINDS["v"][1], 10, 2)
    f.text(380, 298, "가중치 한 벌", 19, weight=600)
    f.text(380, 366, "한 번 읽어 네 요청에 쓴다", 17, weight=700, color="#4d9a56")
    f.text(380, 394, "요청이 늘어도 읽는 양은 그대로", 16, color=MUTED)
    f.line(780, 80, 780, 410, "#d3d6e2", 2, "6 6")
    # 오른쪽 : KV
    f.text(1170, 96, "KV cache", 21, weight=700)
    for i in range(4):
        x = 910 + i * 140
        f.box(x, 130, 100, 40, "q", f"요청 {i + 1}", mono=False, fs=16)
        f.line(x + 50, 172, x + 50, 204, "#a3a7b8", 1.8)
        f.stack(x, 208, 100, 6 if i != 2 else 4, "k" if i % 2 == 0 else "c", 14, hi_last=False)
    f.text(1170, 366, "요청마다 자기 것을 따로 읽는다", 17, weight=700, color="#c2479c")
    f.text(1170, 394, "요청이 늘면 읽는 양도 그만큼 늘어난다", 16, color=MUTED)

    # 균형점 눈금 (로그)
    import math
    ax, aw, ay = 120, 1320, 520
    f.text(40, ay - 46, "바이트 하나를 읽는 동안 할 수 있는 계산 (FLOP / byte, 로그 눈금)", 18, "start", 700)
    f.line(ax, ay, ax + aw, ay, "#8b90a3", 2)

    def px(v):
        return ax + aw * math.log10(v) / 3

    for v in (1, 10, 100, 1000):
        f.line(px(v), ay - 6, px(v), ay + 6, "#8b90a3", 2)
        f.text(px(v), ay + 28, str(v), 15, color=MUTED, mono=True)
    f.rect(px(1), ay - 16, px(2) - px(1), 32, SOLID["k"], "none", 6)
    f.text(px(1.4), ay + 62, "decode가 시키는 일", 17, weight=700, color="#c2479c")
    f.text(px(1.4), ay + 88, "바이트당 1 ~ 2", 15, color=MUTED)
    for v, name in ((281, "B200 ≈ 281"), (295, "H100 ≈ 295")):
        f.rect(px(v) - 3, ay - 18, 6, 36, PURPLE, "none", 2)
    f.text(px(288), ay + 62, "GPU가 할 수 있는 일", 17, weight=700, color=PURPLE)
    f.text(px(288), ay + 88, "H100 ≈ 295  ·  B200 ≈ 281", 15, color=MUTED)
    f.text((px(2) + px(281)) / 2, ay - 22, "두 자릿수 넘게 남는다 — 속도를 정하는 것은 메모리 대역폭이다", 16, color=INK)
    return f


# ---------------------------------------------------------------- 21. MHA vs MQA
def fig_mha_mqa():
    f = Fig("21-mha-vs-mqa", 1560, 520)
    panels = [("MHA · KV 헤드 8개", 8, 150, "쿼리 헤드마다 자기 K, V를 읽는다"),
              ("MQA · KV 헤드 1개", 1, 930, "모든 쿼리 헤드가 같은 K, V를 읽는다")]
    for name, nkv, x0, note in panels:
        f.title(x0 + 232, 50, name, 24, "middle")
        qx = [x0 + i * 60 for i in range(8)]
        for x in qx:
            f.box(x, 90, 46, 46, "q", "q", fs=19)
        g = 8 // nkv
        for j in range(nkv):
            cx = sum(qx[j * g:(j + 1) * g]) / g
            for i in range(j * g, (j + 1) * g):
                f.line(qx[i] + 23, 138, cx + 23, 288, "#a3a7b8", 1.8)
            f.box(cx, 290, 46, 46, "k", "k", fs=19)
            f.box(cx, 344, 46, 46, "v", "v", fs=19)
        f.text(x0 + 232, 432, note, 18, weight=600)
        f.text(x0 + 232, 462, f"캐시  ( S × {nkv} × d_h ) × 2", 16, color=MUTED, mono=True)
    f.line(780, 70, 780, 470, "#d3d6e2", 2, "6 6")
    f.text(780, 504, "쿼리 헤드 8개 기준.  점수 계산은 양쪽 모두 8번이다.  읽어 오는 K, V만 8장에서 1장으로 줄어든다.", 17)
    return f


# ---------------------------------------------------------------- 22. MLA 전체 흐름
def fig_mla_flow():
    f = Fig("22-mla-flow", 1560, 800)
    f.title(40, 44, "MLA 전체 흐름 — decode 한 스텝 (DeepSeek-V3)")
    PINK, AMB = "#c2479c", "#b87410"
    f.box(40, 300, 90, 54, "n", "x", "지금 토큰 · 7168")

    # ---- 저장하는 쪽
    f.text(200, 96, "저장하는 쪽 — 한 줄 쓴다", 19, "start", 700, PINK)
    f.curve(134, 314, 170, 314, 170, 146, 216, 146)
    f.box(220, 120, 200, 52, "c", "c_KV", "512", fs=20)
    f.curve(134, 322, 180, 322, 180, 236, 216, 236)
    f.box(220, 210, 110, 52, "k", "k_R", "64 · RoPE", fs=20)
    f.text(192, 136, "W_DKV", 13, "end", color=MUTED, mono=True)
    f.text(222, 200, "W_KR", 13, "start", color=MUTED, mono=True)

    # ---- 캐시
    cx = 900
    f.rect(cx, 78, 420, 208, "#fafbfd", "#b9bdcc", 10, 2, "7 6")
    f.text(cx + 210, 66, "캐시 — 과거 토큰 전부", 17, weight=700)
    f.stack(cx + 24, 104, 250, 6, "c", 18)
    f.stack(cx + 306, 104, 90, 6, "k", 18)
    f.text(cx + 149, 258, "c_KV  S × 512", 15, color=MUTED, mono=True)
    f.text(cx + 351, 258, "k_R  S × 64", 15, color=MUTED, mono=True)
    f.arrow(424, 146, cx - 6, 146)
    f.arrow(334, 236, cx - 6, 236)
    f.text(660, 134, "끝에 붙인다", 15, color=MUTED)

    # ---- 묻는 쪽
    f.text(200, 404, "묻는 쪽 — 지금 토큰의 쿼리", 19, "start", 700, PURPLE)
    f.curve(134, 340, 170, 340, 150, 480, 196, 480)
    f.box(200, 454, 110, 52, "n", "c_Q", "1536", fs=19)
    f.curve(314, 470, 350, 470, 340, 446, 376, 446)
    f.curve(314, 490, 350, 490, 340, 566, 376, 566)
    f.box(380, 420, 150, 52, "q", "q_C", "128 헤드 × 128", fs=19)
    f.arrow(534, 446, 626, 446)
    f.text(580, 432, "× W_UKᵀ", 13, color=MUTED, mono=True)
    f.box(630, 420, 190, 52, "q", "q̃", "128 헤드 × 512", fs=20)
    f.box(380, 540, 150, 52, "q", "q_R", "128 헤드 × 64 · RoPE", fs=19)

    # ---- 점수
    f.rect(930, 420, 170, 52, "#ffffff", AMB, 8, 2.2)
    f.text(1015, 452, "내용 점수", 18, weight=700, color=AMB)
    f.text(1015, 496, "q̃ · c_KVᵀ", 15, color=MUTED, mono=True)
    f.rect(1130, 540, 170, 52, "#ffffff", PINK, 8, 2.2)
    f.text(1215, 572, "위치 점수", 18, weight=700, color=PINK)
    f.text(1215, 616, "q_R · k_Rᵀ", 15, color=MUTED, mono=True)
    f.arrow(824, 446, 926, 446)
    f.arrow(534, 566, 1126, 566)
    f.arrow(1015, 292, 1015, 414)
    f.text(1027, 362, "읽기 ①", 15, "start", 700, PURPLE)
    f.arrow(1251, 292, 1251, 534, dash="5 5")
    # ---- 합, softmax
    f.curve(1104, 446, 1380, 446, 1400, 470, 1400, 494)
    f.curve(1304, 566, 1350, 566, 1350, 524, 1378, 520)
    f.raw('<circle cx="1400" cy="517" r="19" fill="#ffffff" stroke="#5d6377" stroke-width="2"/>')
    f.text(1400, 525, "+", 24)
    f.arrow(1400, 540, 1400, 654)
    f.text(1412, 606, "softmax", 15, "start", color=MUTED)
    f.box(1340, 658, 120, 52, "q", "p", fs=20)
    f.text(1476, 690, "128 × S", 15, "start", color=MUTED, mono=True)

    # ---- 섞기 → 값 펴기 → 출력 (오른쪽에서 왼쪽으로)
    f.arrow(1336, 684, 1154, 684)
    f.box(900, 658, 250, 52, "c", "섞기  p · c_KV", "128 × 512", mono=False, fs=18)
    f.text(1250, 670, "읽기 ②", 15, weight=700, color=PURPLE)
    f.arrow(896, 684, 714, 684)
    f.text(805, 670, "× W_UV", 13, color=MUTED, mono=True)
    f.box(500, 658, 210, 52, "v", "값 펴기", "128 헤드 × 128", mono=False, fs=18)
    f.arrow(496, 684, 334, 684)
    f.text(415, 670, "W_O", 13, color=MUTED, mono=True)
    f.box(220, 658, 110, 52, "n", "출력", "7168", mono=False, fs=18)
    f.text(40, 780, "캐시를 읽는 곳은 두 군데다 — 점수를 낼 때(읽기 ①)와 섞을 때(읽기 ②).  K와 V는 어디에서도 만들지 않는다.", 16, "start",
           color=MUTED)
    return f


FIGS = [fig_decode, fig_heads, fig_mla_cache, fig_mla_absorb, fig_patterns, fig_dsa, fig_v4, fig_v4_layers, fig_qsa,
        fig_cache_size, fig_knobs, fig_lineage, fig_mha, fig_mla_shape, fig_swa, fig_store_read, fig_dsa_shape,
        fig_dsa_structure, fig_journey, fig_amortize, fig_mha_mqa, fig_mla_flow]
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
