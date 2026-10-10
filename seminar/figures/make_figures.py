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
    f.text(1115, y + 160, "정의상의 K · V — 저장하지 않고, 실제 decode에서는 만들지도 않는다", 16, color=MUTED)

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
    f.title(40, 44, "펴서 비교 — 정의식을 그대로 실행 (naive)")
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
    f.text(x0 + 345, 250, "쿼리만 옮긴다 — 헤드마다 한 번", 18, weight=700, color=PURPLE)
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
    f = Fig("06-dsa", 1560, 720)
    f.title(40, 44, "DSA — MLA 앞에 indexer를 둔다 (DeepSeek-V3.2)")
    BLUE, AMB = "#3b7dd8", "#b87410"
    n, x0, cw = 28, 420, 34
    sel = [3, 9, 12, 19, 24]

    def cx(i):
        return x0 + i * (cw + 4) + cw / 2

    # ---- 지금까지 : MLA만
    y = 92
    f.text(40, y + 22, "지금까지 — MLA만", 20, "start", 700)
    f.text(40, y + 48, "100만 줄을 전부, 비싸게 읽는다", 16, "start", color=MUTED)
    f.cells(x0, y, [SOLID["c"]] * n, cw, 34, 4, 4)
    f.line(40, 176, 1520, 176, "#e3e5ee", 2, "6 6")

    # ---- ① 후보 찾기
    y1 = 216
    f.rect(36, y1 - 14, 360, 132, "#f1f6fe", "#b9d3f5", 10, 1.5)
    f.rect(52, y1 - 2, 96, 26, BLUE, "none", 13)
    f.text(100, y1 + 16, "추가된 단계", 14, weight=700, color="#ffffff")
    f.text(52, y1 + 52, "① lightning indexer", 20, "start", 700, BLUE)
    f.text(52, y1 + 78, "전부 훑어 후보를 찾는다 — 싸게", 16, "start", color=MUTED)
    f.text(52, y1 + 102, "점수만 매겨 큰 것을 고른다", 16, "start", color=MUTED)
    f.cells(x0, y1, [KINDS["i"][0]] * n, cw, 34, 4, 4)
    heights = [5, 9, 4, 30, 8, 6, 11, 5, 7, 27, 6, 9, 25, 5, 8, 10, 4, 7, 6, 31, 9, 5, 8, 6, 28, 7, 5, 9]
    for i, h in enumerate(heights):
        f.rect(cx(i) - 8, y1 + 82 - h, 16, h, BLUE if i in sel else "#b9d3f5", "none", 2)
    f.text(x0 + n * (cw + 4) + 6, y1 + 80, "점수", 14, "start", color=MUTED)
    for i in sel:
        f.arrow(cx(i), y1 + 90, cx(i), y1 + 150)
    f.text(cx(sel[2]) + 16, y1 + 126, "위치만 넘긴다 — 2,048개", 15, "start", color=MUTED)

    # ---- ② MLA attention
    y2 = y1 + 158
    f.text(52, y2 + 22, "② MLA attention", 20, "start", 700, AMB)
    f.text(52, y2 + 48, "고른 줄만 읽는다 — 원래 하던 계산", 16, "start", color=MUTED)
    f.cells(x0, y2, [SOLID["c"] if i in sel else OFF for i in range(n)], cw, 34, 4, 4)
    f.text(x0, y2 + 60, "0", 14, color=MUTED, mono=True)
    f.text(cx(n - 1), y2 + 60, "S (지금)", 14, color=MUTED, mono=True)

    # ---- 두 단계 비교
    ty = 496
    f.rect(40, ty, 720, 124, "#f1f6fe", "#b9d3f5", 10, 1.5)
    f.text(64, ty + 34, "① indexer — 넓고 얕게", 19, "start", 700, BLUE)
    f.text(64, ty + 66, "읽는 범위   S줄 전부", 17, "start")
    f.text(64, ty + 96, "한 줄        128바이트  ·  점수만 낸다", 17, "start")
    f.rect(800, ty, 720, 124, "#fef7ea", "#f0cf93", 10, 1.5)
    f.text(824, ty + 34, "② MLA attention — 좁고 깊게", 19, "start", 700, AMB)
    f.text(824, ty + 66, "읽는 범위   고른 2,048줄", 17, "start")
    f.text(824, ty + 96, "한 줄        1,152바이트  ·  출력을 만든다", 17, "start")
    f.text(780, 672, "MLA만   S × 비싼 단가          →          DSA   S × 싼 단가  +  k × 비싼 단가", 18, weight=700,
           mono=False)
    return f


# ---------------------------------------------------------------- 7. CSA / HCA
def _tokens(f, x, y, n, r=9, gap=6, color="#b7bccd"):
    for i in range(n):
        f.raw(f'<circle cx="{x + i * (2 * r + gap) + r}" cy="{y}" r="{r}" fill="{color}"/>')
    return 2 * r + gap


def fig_v4():
    f = Fig("07-v4-csa-hca", 1560, 620)
    f.title(40, 44, "DSA 앞에 압축 단계를 둔다 — CSA와 HCA (DeepSeek-V4)")
    BLUE, AMB = "#3b7dd8", "#b87410"
    n, cw, gap, GG = 24, 18, 4, 12
    pitch = cw + gap
    CX, HX = 270, 930

    # 열 제목
    f.text(CX, 96, "CSA", 26, "start", 700)
    f.text(CX + 70, 96, "4개씩 묶고 → 고른다", 18, "start", color=MUTED)
    f.text(HX, 96, "HCA", 26, "start", 700)
    f.text(HX + 70, 96, "128개씩 묶고 → 전부 읽는다", 18, "start", color=MUTED)
    f.line(HX - 44, 76, HX - 44, 590, "#e3e5ee", 2, "6 6")

    # 추가된 단계 띠
    f.rect(30, 168, 1500, 112, "#f6f5fd", "#d9d2f3", 10, 1.5)

    # 행 이름
    f.text(48, 138, "원본 토큰", 19, "start", 700)
    f.text(160, 138, "S개", 15, "start", color=MUTED, mono=True)
    f.rect(48, 180, 92, 24, PURPLE, "none", 12)
    f.text(94, 197, "추가된 단계", 13, weight=700, color="#ffffff")
    f.text(48, 236, "① 압축", 21, "start", 700, PURPLE)
    f.text(48, 262, "여러 토큰 → 엔트리 1개", 15, "start", color=MUTED)
    f.text(48, 336, "② indexer · top-k", 21, "start", 700, BLUE)
    f.text(48, 362, "DSA와 같다", 15, "start", color=MUTED)
    f.text(48, 432, "③ attention", 21, "start", 700, AMB)
    f.text(48, 458, "+ 최근 128토큰 (압축 안 함)", 15, "start", color=MUTED)
    f.text(48, 546, "1M 토큰이면", 19, "start", 700)

    def column(x0, m, sel, all_read):
        k = n // m
        ew = min(m * pitch - gap - 16, 120)
        cs = []
        for j in range(k):
            bx = x0 + j * (m * pitch + GG)
            bw = m * pitch - gap
            f.cells(bx, 120, ["#c9cddb"] * m, cw, 24, gap, 4)
            c = bx + bw / 2
            cs.append(c)
            f.rect(bx - 3, 116, bw + 6, 32, "none", "#b0a3e6", 7, 1.5)
            for i in range(m):
                f.line(bx + i * pitch + cw / 2, 150, c, 222, "#9a9fb3", 1.3)
            f.rect(c - ew / 2, 226, ew, 34, KINDS["c"][0], KINDS["c"][1], 6)
            on = all_read or j in sel
            f.rect(c - ew / 2, 410, ew, 34, SOLID["c"] if on else OFF, KINDS["c"][1] if on else "none", 6)
        return cs, k * (m * pitch + GG) - GG - gap

    # ---- CSA
    cs, W = column(CX, 4, {1, 4}, False)
    hs = [9, 34, 7, 12, 30, 8]
    for j, c in enumerate(cs):
        on = j in (1, 4)
        f.rect(c - 11, 358 - hs[j], 22, hs[j], BLUE if on else "#b9d3f5", "none", 3)
        if on:
            f.arrow(c, 366, c, 404)
    f.text(cs[4] + 14, 390, "위치만 넘긴다", 14, "start", color=MUTED)
    f.text(CX, 474, "고른 엔트리만 읽는다 — 흩어진 읽기", 16, "start", 600)

    # ---- HCA
    cs, W = column(HX, 12, set(), True)
    f.text(HX + W, 170 + 126, "그림은 12개씩 그렸다 — 실제로는 128개", 13, "end", color=MUTED)
    f.rect(HX, 306, W, 60, "#ffffff", "#c9cddb", 10, 1.6, "6 5")
    f.text(HX + W / 2, 332, "고르지 않는다", 18, weight=700, color="#8b90a3")
    f.text(HX + W / 2, 355, "indexer · top-k · gather 없음", 15, color="#8b90a3")
    for c in cs:
        f.arrow(c, 370, c, 404)
    f.text(HX, 474, "전부 읽는다 — 이어 읽기", 16, "start", 600)

    # ---- 숫자
    for x0, a_, b_ in ((CX, "250,000 엔트리", "1,024개 읽는다"), (HX, "7,812 엔트리", "전부 읽는다")):
        f.rect(x0, 512, 580, 56, "#fafbfd", "#d3d6e2", 10, 1.5)
        f.text(x0 + 24, 547, "1,000,000 토큰", 18, "start")
        f.text(x0 + 180, 547, "→", 18)
        f.text(x0 + 206, 547, a_, 18, "start", 700)
        f.text(x0 + 390, 547, "→", 18)
        f.text(x0 + 416, 547, b_, 18, "start", 700)
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
    f = Fig("09-qsa", 1560, 640)
    f.title(40, 44, "무엇을 묶나 — CSA와 QSA (Qwen3.8-Flash-Next)")
    BLUE, AMB, TOK = "#3b7dd8", "#b87410", "#6b4fc8"
    cw, gap, GG, m, k = 18, 4, 12, 4, 6
    pitch = cw + gap
    bw = m * pitch - gap
    CX, QX = 270, 930
    W = k * (bw + GG) - GG
    sel = {1, 4}

    f.text(CX, 96, "CSA", 26, "start", 700)
    f.text(CX + 70, 96, "KV를 묶는다 → 엔트리를 읽는다", 18, "start", color=MUTED)
    f.text(QX, 96, "QSA", 26, "start", 700)
    f.text(QX + 70, 96, "색인 키만 묶는다 → 원본 토큰을 읽는다", 18, "start", color=MUTED)
    f.line(QX - 44, 76, QX - 44, 610, "#e3e5ee", 2, "6 6")
    f.rect(30, 168, 1500, 112, "#f6f5fd", "#d9d2f3", 10, 1.5)

    f.text(48, 138, "원본 토큰", 19, "start", 700)
    f.text(160, 138, "S개", 15, "start", color=MUTED, mono=True)
    f.text(48, 222, "① 묶기", 21, "start", 700, PURPLE)
    f.text(48, 248, "4토큰씩", 15, "start", color=MUTED)
    f.text(48, 336, "② indexer · top-k", 21, "start", 700, BLUE)
    f.text(48, 362, "S/4개에 점수를 매긴다", 15, "start", color=MUTED)
    f.text(48, 432, "③ attention", 21, "start", 700, AMB)
    f.text(48, 458, "무엇을 읽나", 15, "start", color=MUTED)
    f.text(48, 566, "1M 토큰이면", 19, "start", 700)

    hs = [9, 34, 7, 12, 30, 8]
    for x0, qsa in ((CX, False), (QX, True)):
        for j in range(k):
            bx = x0 + j * (bw + GG)
            c = bx + bw / 2
            f.cells(bx, 120, ["#c9cddb"] * m, cw, 24, gap, 4)
            f.rect(bx - 3, 116, bw + 6, 32, "none", "#b0a3e6", 7, 1.5)
            if qsa:
                f.arrow(c, 152, c, 222)
                f.rect(c - 36, 226, 72, 34, KINDS["i"][0], KINDS["i"][1], 6)
            else:
                for i in range(m):
                    f.line(bx + i * pitch + cw / 2, 150, c, 222, "#9a9fb3", 1.3)
                f.rect(c - 36, 226, 72, 34, KINDS["c"][0], KINDS["c"][1], 6)
            on = j in sel
            f.rect(c - 11, 358 - hs[j], 22, hs[j], BLUE if on else "#b9d3f5", "none", 3)
            if on:
                f.arrow(c, 366, c, 404)
            if qsa:
                f.cells(bx, 412, [TOK if on else OFF] * m, cw, 28, gap, 4)
                if on:
                    f.rect(bx - 3, 408, bw + 6, 36, "none", TOK, 7, 1.6)
            else:
                f.rect(c - 36, 410, 72, 34, SOLID["c"] if on else OFF, KINDS["c"][1] if on else "none", 6)
    f.text(CX + W + 6, 249, "압축 엔트리", 14, "start", color=MUTED)
    f.text(QX + W + 6, 249, "블록 키", 14, "start", color=MUTED)

    f.text(CX, 474, "압축 엔트리를 읽는다", 16, "start", 700)
    f.text(CX, 498, "원본 토큰은 남지 않는다 · 캐시 길이 S/4", 15, "start", color=MUTED)
    f.text(QX, 474, "블록 안의 원본 토큰 4개를 통째로 읽는다", 16, "start", 700)
    f.text(QX, 498, "KV cache는 토큰 단위 그대로 · 캐시 길이 S", 15, "start", color=MUTED)

    for x0, a_, b_ in ((CX, "250,000 엔트리", "1,024개 읽는다"), (QX, "250,000 블록", "512블록 = 2,048토큰")):
        f.rect(x0, 532, 590, 56, "#fafbfd", "#d3d6e2", 10, 1.5)
        f.text(x0 + 20, 567, "1,000,000 토큰", 18, "start")
        f.text(x0 + 172, 567, "→", 18)
        f.text(x0 + 194, 567, a_, 18, "start", 700)
        f.text(x0 + 362, 567, "→", 18)
        f.text(x0 + 384, 567, b_, 18, "start", 700)
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


# ---------------------------------------------------------------- 23. KV cache의 모양
def fig_kv_cache():
    f = Fig("23-kv-cache", 1560, 560)
    f.title(40, 44, "KV cache — 토큰마다 한 줄, 층마다 한 벌")
    x0, y0, w, rows, rh = 470, 130, 230, 8, 26
    # 뒤에 겹친 층
    for d in (28, 14):
        f.rect(x0 + d, y0 - d, 2 * w + 30, rows * (rh + 4) + 6, "#fafbfd", "#c9cddb", 8, 1.5)
    f.rect(x0 - 12, y0 - 12, 2 * w + 54, rows * (rh + 4) + 20, "#ffffff", "#b9bdcc", 8, 1.5)
    f.stack(x0, y0, w, rows, "k", rh)
    f.stack(x0 + w + 30, y0, w, rows, "v", rh)
    f.text(x0 + w / 2, y0 - 44, "K", 20, weight=700)
    f.text(x0 + w + 30 + w / 2, y0 - 44, "V", 20, weight=700)
    f.text(x0 + 2 * w + 80, y0 - 44, "층마다 한 벌", 16, "start", color=MUTED)
    # 행 번호
    labels = ["토큰 1", "토큰 2", "토큰 3", "", "⋮", "", "", "토큰 S"]
    for r, lab in enumerate(labels):
        if lab:
            f.text(x0 - 26, y0 + r * (rh + 4) + 19, lab, 15, "end", color=MUTED)
    # 쓰기
    ly = y0 + (rows - 1) * (rh + 4) + rh / 2
    f.box(60, ly - 62, 150, 40, "k", "k", fs=18)
    f.box(60, ly - 14, 150, 40, "v", "v", fs=18)
    f.text(135, ly - 82, "지금 토큰", 16, color=MUTED)
    f.arrow(214, ly - 20, 370, ly - 2)
    f.text(60, ly + 60, "쓰기 — 끝에 한 줄 붙인다", 18, "start", 700, "#c2479c")
    f.text(60, ly + 86, "쓴 줄은 고치지 않는다", 15, "start", color=MUTED)
    # 읽기
    bx = x0 + 2 * w + 80
    top, bot = y0, y0 + rows * (rh + 4) - 4
    f.raw(f'<path d="M{bx},{top} C{bx + 22},{top} {bx + 22},{top} {bx + 22},{top + 20} L{bx + 22},{bot - 20} '
          f'C{bx + 22},{bot} {bx + 22},{bot} {bx},{bot}" fill="none" stroke="{PURPLE}" stroke-width="2.4"/>')
    f.text(bx + 44, (top + bot) / 2 - 6, "읽기 — 전부", 18, "start", 700, PURPLE)
    f.text(bx + 44, (top + bot) / 2 + 20, "토큰을 하나 만들 때마다", 15, "start", color=MUTED)
    f.text(bx + 44, (top + bot) / 2 + 44, "처음부터 끝까지 이어서 읽는다", 15, "start", color=MUTED)
    f.text(40, 520, "길이가 S이면 읽기와 쓰기의 비는 S : 1이다.  수명은 요청이 끝날 때까지다.", 17, "start")
    return f


# ---------------------------------------------------------------- 24. KV cache의 크기 (막대)
def fig_kv_bars():
    f = Fig("24-kv-size", 1560, 640)
    f.title(40, 44, "KV cache — 살아 있는 토큰 수가 크기를 정한다 (BF16)")
    f.text(1520, 44, "모든 층의 캐시 합", 16, "end", color=MUTED)
    ax, aw, top, vmax = 440, 980, 168, 300.0

    def px(v):
        return ax + aw * v / vmax

    models = [
        ("Qwen3-32B", "GQA · 64층 · KV 헤드 8", 34.4, 274.9),
        ("Qwen3-235B-A22B", "GQA · 94층 · KV 헤드 4", 25.2, 201.9),
        ("DeepSeek-V3", "MLA · 61층 · 576값", 9.2, 73.7),
    ]
    bot = 518
    f.rect(ax, 78, 20, 14, "#b8abe8", "none", 3)
    f.text(ax + 30, 91, "토큰 13만 — 128K 요청 1개, 또는 8K 요청 16개", 14, "start", color=MUTED)
    f.rect(ax + 420, 78, 20, 14, SOLID["q"], "none", 3)
    f.text(ax + 450, 91, "토큰 105만 — 128K 요청 8개, 또는 8K 요청 128개", 14, "start", color=MUTED)
    # GPU 한 장의 메모리
    for v, name in ((80, "H100 · 80 GB"), (192, "B200 · 192 GB")):
        f.line(px(v), top - 24, px(v), bot, "#8b90a3", 1.6, "5 5")
        f.text(px(v), top - 32, name, 14, color=INK)
    # 눈금
    for v in (0, 50, 100, 150, 200, 250, 300):
        f.line(px(v), top - 6, px(v), bot, "#eceef4", 1.5)
        f.text(px(v), bot + 24, f"{v}", 14, color=MUTED, mono=True)
    f.text(px(vmax) + 38, bot + 24, "GB", 14, "start", color=MUTED)

    for i, (name, desc, one, eight) in enumerate(models):
        y = top + i * 116
        f.text(ax - 24, y + 18, name, 18, "end", 700)
        f.text(ax - 24, y + 42, desc, 14, "end", color=MUTED)

        for j, (label, value, color) in enumerate((("요청 1개", one, "#b8abe8"),
                                                   ("동시 8개", eight, SOLID["q"]))):
            by = y + j * 42
            wbar = max(px(value) - ax, 4)
            f.rect(ax, by, wbar, 32, color, "none", 4)
            tx = ax + wbar + 10
            f.raw(f'<text x="{tx}" y="{by + 23}" font-family="{FONT}" font-size="16" font-weight="700" '
                  f'fill="{INK}" stroke="#ffffff" stroke-width="6" paint-order="stroke">{value:.1f} GB</text>')
    f.line(ax, top - 6, ax, bot, "#8b90a3", 2)

    f.rect(40, 566, 1480, 48, "#f4f1fc", "#d8d0f1", 8, 1.5)
    f.text(780, 597,
           "Qwen3-32B · 128K 요청 2개 → KV cache 68.7 GB  ›  BF16 가중치 약 65.6 GB",
           18, weight=700, color=PURPLE)
    return f


# ---------------------------------------------------------------- 25. MHA와 MLA의 점수 계산을 텐서로
def fig_mla_tensor():
    f = Fig("25-mha-vs-mla-tensor", 1560, 700)
    f.title(40, 44, "점수 계산을 텐서로 놓고 보면 — MHA와 MLA")
    U = 0.31          # 차원 1 = 0.31px  (128 → 40px, 512 → 159px)
    SW = 330          # 길이 S를 나타내는 가로
    RED, GRN = "#c2479c", "#4d9a56"

    def sheets(x, y, w, h, kind, n=3, off=7):
        for i in range(n - 1, -1, -1):
            f.rect(x + i * off, y - i * off, w, h, KINDS[kind][0], KINDS[kind][1], 5, 1.8)

    # ---------------- MHA
    y = 150
    f.text(40, y + 30, "MHA", 22, "start", 700)
    f.text(40, y + 56, "헤드마다 따로", 15, "start", color=MUTED)
    qw, kh = 128 * U, 128 * U
    sheets(230, y + 8, qw, 16, "q")
    f.text(230 + qw / 2 + 7, y + 62, "q  1 × 128", 15, color=MUTED, mono=True)
    f.text(230 + qw / 2 + 7, y + 84, "× 128 헤드", 14, color=MUTED)
    f.text(330, y + 26, "·", 30)
    sheets(370, y - 4, SW, kh, "k")
    f.text(370 + SW / 2, y + kh / 2 + 2, "Kᵀ", 18, mono=True)
    f.text(370 + SW / 2 + 7, y + 62, "128 × S", 15, color=MUTED, mono=True)
    f.text(370 + SW / 2 + 7, y + 84, "× 128장 — 헤드마다 한 장", 14, weight=700, color=RED)
    f.text(750, y + 26, "=", 24)
    sheets(790, y + 8, SW, 16, "q")
    f.text(790 + SW / 2 + 7, y + 62, "점수  128 × S", 15, color=MUTED, mono=True)
    f.text(1160, y + 20, "행렬 × 벡터를 128번", 17, "start", 600)
    f.text(1160, y + 46, "읽은 값을 한 번씩만 쓴다", 15, "start", color=MUTED)

    f.line(40, 290, 1520, 290, "#e3e5ee", 2, "6 6")

    # ---------------- MLA
    y = 340
    f.text(40, y + 80, "MLA", 22, "start", 700)
    f.text(40, y + 106, "캐시는 하나", 15, "start", color=MUTED)
    qh, qw2, ch = 42, 512 * U, 512 * U
    f.rect(130, y + 58, 60, 36, KINDS["q"][0], KINDS["q"][1], 5, 1.8)
    f.text(160, y + 82, "q_C", 15, mono=True)
    f.arrow(194, y + 76, 226, y + 76)
    f.text(210, y + 50, "① 새로 생긴 변환", 14, weight=700, color=RED)
    f.text(210, y + 122, "× W_UKᵀ", 13, color=MUTED, mono=True)
    f.text(210, y + 142, "토큰당 한 번", 13, color=MUTED)
    f.rect(230, y + 55, qw2 - 30, qh, KINDS["q"][0], KINDS["q"][1], 5, 1.8)
    f.text(230 + (qw2 - 30) / 2, y + 82, "q̃", 18, mono=True)
    f.text(230 + (qw2 - 30) / 2, y + 122, "128 × 512", 15, color=MUTED, mono=True)
    f.text(372, y + 82, "·", 30)
    f.rect(400, y, SW, ch, KINDS["c"][0], KINDS["c"][1], 5, 1.8)
    f.text(400 + SW / 2, y + ch / 2 + 6, "c_KVᵀ", 18, mono=True)
    f.text(400 + SW / 2, y + ch + 26, "512 × S", 15, color=MUTED, mono=True)
    f.text(400 + SW / 2, y + ch + 48, "한 장 — 전 헤드가 같이 읽는다", 14, weight=700, color=GRN)
    # 내적 길이 표시
    bx = 400 + SW + 14
    f.line(bx, y, bx, y + ch, RED, 2.2)
    f.line(bx - 6, y, bx + 6, y, RED, 2.2)
    f.line(bx - 6, y + ch, bx + 6, y + ch, RED, 2.2)
    f.text(bx + 14, y + ch / 2 - 6, "② 내적이 길어진다", 14, "start", 700, RED)
    f.text(bx + 14, y + ch / 2 + 16, "128 → 512 (+ 위치 64)", 14, "start", color=MUTED)
    f.text(bx + 14, y + ch / 2 + 36, "캐시 한 줄마다 치른다", 14, "start", color=MUTED)
    f.text(960, y + 82, "=", 24)
    f.rect(1000, y + 55, SW, qh, KINDS["q"][0], KINDS["q"][1], 5, 1.8)
    f.text(1000 + SW / 2, y + 122, "점수  128 × S", 15, color=MUTED, mono=True)
    f.text(1000, y + 20, "행렬 × 행렬 한 번", 17, "start", 600)
    f.text(1000, y + 42, "읽은 값을 128번 쓴다", 15, "start", color=MUTED)

    # ---------------- 요약
    f.rect(40, 600, 1480, 76, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(70, 632, "줄어든 것", 17, "start", 700, GRN)
    f.text(180, 632, "캐시 128장 → 1장.  한 줄에 저장하는 값 2 · n_h · d_h → d_c + d_r", 17, "start")
    f.text(70, 660, "늘어난 것", 17, "start", 700, RED)
    f.text(180, 660, "① 쿼리를 옮기고 값을 펴는 변환 (길이와 무관)    ② 길어진 내적 (길이에 비례)", 17, "start")
    return f


# ---------------------------------------------------------------- 26. 줄이는 축
def fig_axes():
    f = Fig("26-axes", 1560, 600)
    f.title(40, 44, "KV cache를 표로 보면 — 토큰 수 × 토큰당 차원")
    top, RH, NR = 200, 20, 13
    H = RH * NR
    RED = "#c2479c"

    def table(x, y, w, rows, kind, solid=False):
        fill, stroke = KINDS[kind]
        if kind == "n":
            fill, stroke = "#eef0f5", "#aeb3c4"
        f.rect(x, y, w, rows * RH, SOLID[kind] if solid else fill, stroke, 5, 1.8)
        for r in range(1, rows):
            f.line(x + 1, y + r * RH, x + w - 1, y + r * RH, stroke, 0.8)

    def ghost(x, w):
        f.rect(x, top, w, H, "none", "#c9cddb", 5, 1.4, "5 5")

    def head(cx, name, cap, color):
        f.text(cx, 132, name, 24, weight=700, color=color)
        f.text(cx, 162, cap, 17, weight=600)

    def foot(cx, a_, b_):
        f.text(cx - 120, top + H + 40, "토큰당 차원", 15, "start", 700, MUTED)
        f.text(cx - 22, top + H + 40, a_, 17, "start")
        f.text(cx - 120, top + H + 68, "토큰 수", 15, "start", 700, MUTED)
        f.text(cx - 22, top + H + 68, b_, 17, "start")

    # ---- 기준 : 축의 뜻
    x0, w0 = 230, 250
    table(x0, top, w0, NR, "n")
    f.rect(x0 + 1, top + 5 * RH, w0 - 2, RH, "#c9cddb", "none", 0)
    f.text(x0 + w0 / 2, top + 5 * RH + 15, "한 줄 = 토큰 하나", 14, weight=700)
    # 가로
    f.line(x0, top - 14, x0 + w0, top - 14, RED, 2)
    f.line(x0, top - 20, x0, top - 8, RED, 2)
    f.line(x0 + w0, top - 20, x0 + w0, top - 8, RED, 2)
    f.text(x0 + w0 / 2, 132, "토큰 하나당 차원", 19, weight=700, color=RED)
    f.text(x0 + w0 / 2, 158, "토큰 하나가 남기는 값의 개수", 15, color=MUTED)
    f.text(x0 + w0 / 2, 178, "(K·V 2 × 헤드 수 × 헤드 차원)", 14, color=MUTED)
    # 세로
    f.line(x0 - 16, top, x0 - 16, top + H, RED, 2)
    f.line(x0 - 22, top, x0 - 10, top, RED, 2)
    f.line(x0 - 22, top + H, x0 - 10, top + H, RED, 2)
    f.text(x0 - 30, top + H / 2 - 22, "컨텍스트의 토큰 수", 19, "end", 700, RED)
    f.text(x0 - 30, top + H / 2 + 4, "S (sequence length)", 15, "end", color=MUTED)
    f.text(x0 - 30, top + H / 2 + 26, "토큰마다 한 줄", 14, "end", color=MUTED)
    foot(x0 + w0 / 2, "32,768  (MHA라면)", "S")
    f.line(540, 96, 540, 560, "#e3e5ee", 2, "6 6")

    # ---- MLA
    cx = 700
    head(cx, "MLA", "토큰당 차원을 줄인다", "#b87410")
    ghost(cx - 100, 200)
    table(cx - 100, top, 40, NR, "c")
    f.arrow(cx + 92, top + H / 2, cx - 50, top + H / 2, color=RED)
    foot(cx, "32,768 → 576", "S 그대로")

    # ---- DSA
    cx = 1000
    head(cx, "DSA", "읽는 토큰만 고른다", "#3b7dd8")
    table(cx - 50, top, 40, NR, "c")
    for r in (1, 4, 5, 8, 11):
        f.rect(cx - 50, top + r * RH, 40, RH, SOLID["c"], KINDS["c"][1], 2, 1)
    table(cx, top, 14, NR, "i")
    f.text(cx + 22, top + 16, "색인 키", 14, "start", color=MUTED)
    foot(cx, "576 + 색인 128", "S 그대로 · 읽기 2,048")

    # ---- CSA · HCA
    cx = 1320
    head(cx, "CSA · HCA", "토큰 수 쪽을 줄인다", PURPLE)
    for dx, rows, nm, cap in ((-110, 3, "CSA", "S / 4"), (50, 0, "HCA", "S / 128")):
        ghost(cx + dx, 40)
        if rows:
            table(cx + dx, top, 40, rows, "c", True)
        else:
            f.rect(cx + dx, top, 40, 7, SOLID["c"], KINDS["c"][1], 3, 1.8)
        f.arrow(cx + dx + 20, top + H - 8, cx + dx + 20, top + rows * RH + 22, color=RED)
        f.text(cx + dx + 52, top + 22, nm, 18, "start", 700)
        f.text(cx + dx + 52, top + 46, cap, 15, "start", color=MUTED, mono=True)
    foot(cx, "512", "S → S/4 · S/128")

    f.text(1520, 584, "그림의 비율은 실제와 다르다", 13, "end", color=MUTED)
    return f


# ---------------------------------------------------------------- 27. 묶는 방법
def fig_compress():
    f = Fig("27-v4-compress", 1560, 600)
    f.title(40, 44, "여러 토큰을 엔트리 하나로 — 평균이 아니라 학습된 가중합")
    x0, pitch, bw = 250, 82, 60
    w = [0.04, 0.06, 0.05, 0.14, 0.18, 0.08, 0.34, 0.11]
    SC, base_y = 300, 318
    RED = "#c2479c"

    def cx(i):
        return x0 + i * pitch + bw / 2

    # 블록
    f.rect(x0 - 10, 92, 4 * pitch - 2, 62, "none", "#c9cddb", 8, 1.6, "5 5")
    f.rect(x0 + 4 * pitch - 10, 92, 4 * pitch - 2, 62, "none", "#b0a3e6", 8, 2)
    f.text(x0 + 2 * pitch - 11, 84, "앞 블록 4개", 15, color=MUTED)
    f.text(x0 + 6 * pitch - 11, 84, "자기 블록 4개", 15, weight=700, color=PURPLE)

    # 행 이름
    f.text(40, 122, "① 값", 21, "start", 700)
    f.text(40, 148, "토큰마다 512차원으로 투영", 15, "start", color=MUTED)
    f.text(40, 244, "② 가중치", 21, "start", 700)
    f.text(40, 270, "토큰마다 반영할 비율", 15, "start", color=MUTED)
    f.text(40, 294, "8개의 합 = 1", 15, "start", color=MUTED)
    f.text(40, 412, "③ 합", 21, "start", 700)
    f.text(40, 438, "가중치를 곱해 더한다", 15, "start", color=MUTED)

    mid = x0 + 4 * pitch - 11
    for i in range(8):
        f.rect(x0 + i * pitch, 106, bw, 34, KINDS["c"][0], KINDS["c"][1], 6, 1.8, None if i >= 4 else "4 3")
        f.text(cx(i), 129, "C", 16, mono=True)
        h = w[i] * SC
        f.rect(cx(i) - 15, base_y - h, 30, h, SOLID["q"], "none", 3)
        f.text(cx(i), base_y + 20, f"{w[i]:.2f}", 14, color=INK, mono=True)
        f.line(cx(i), base_y + 28, mid, 384, "#9a9fb3", 1.3)
    f.line(x0 - 10, base_y, x0 + 8 * pitch - 12, base_y, "#aeb3c4", 1.6)
    ya = base_y - 0.125 * SC
    f.line(x0 - 10, ya, x0 + 8 * pitch - 12, ya, RED, 1.8, "7 5")
    f.text(x0 + 8 * pitch - 2, ya - 4, "평균이면", 14, "start", 700, RED)
    f.text(x0 + 8 * pitch - 2, ya + 16, "전부 1/8", 14, "start", 700, RED)

    f.rect(mid - 85, 388, 170, 46, SOLID["c"], KINDS["c"][1], 7)
    f.text(mid, 418, "엔트리 하나", 18, weight=700, color="#ffffff")
    f.text(mid + 100, 418, "512차원 · K이자 V", 15, "start", color=MUTED)

    f.rect(40, 478, 930, 90, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(64, 514, "비율이 입력에 따라, 차원마다 달라진다 — 그 비율을 내는 행렬을 학습한다", 18, "start", 700)
    f.text(64, 546, "값 C = h · W_KV", 15, "start", color=MUTED, mono=True)
    f.text(250, 546, "가중치 = softmax( h · W_Z + B )", 15, "start", color=MUTED, mono=True)
    f.text(590, 546, "엔트리 = Σ 가중치 ⊙ 값", 15, "start", color=MUTED, mono=True)

    # ---- 겹쳐서 묶는다
    f.line(1000, 76, 1000, 572, "#e3e5ee", 2, "6 6")
    f.text(1030, 100, "겹쳐서 묶는다", 22, "start", 700, PURPLE)
    f.text(1178, 100, "overlapped compression", 15, "start", color=MUTED)
    bx0, bp, hi = 1040, 170, 1
    for j in range(3):
        gx = bx0 + j * bp
        ecx = gx + 45
        _tokens(f, gx, 174, 4, color="#8f95ab" if j <= hi else "#c9cddb")
        f.rect(gx - 6, 159, 102, 30, "none", "#b0a3e6" if j == hi else "#c9cddb", 8, 2 if j == hi else 1.5,
               None if j == hi else "5 4")
        on = j == hi
        for i in range(4):
            f.line(gx + i * 24 + 9, 191, ecx, 262, PURPLE if on else "#d3d6e2", 2 if on else 1.3)
        if j > 0:
            for i in range(4):
                f.line(gx - bp + i * 24 + 9, 191, ecx, 262, PURPLE if on else "#d3d6e2", 1.6 if on else 1.2, "5 4")
        f.rect(ecx - 40, 266, 80, 34, SOLID["c"] if on else KINDS["c"][0], KINDS["c"][1], 6)
    f.text(bx0 + 45, 148, "앞 블록", 15, color=MUTED)
    f.text(bx0 + bp + 45, 148, "자기 블록", 15, weight=700, color=PURPLE)
    f.text(bx0 + bp + 45, 324, "이 엔트리", 15, weight=700)
    f.text(1030, 372, "엔트리 하나  =  자기 블록 4개 + 앞 블록 4개", 17, "start", 600)
    f.text(1030, 402, "엔트리는 4토큰마다 하나  →  길이는 S/4", 17, "start", 600)
    f.line(1030, 440, 1520, 440, "#e3e5ee", 1.5)
    f.text(1030, 484, "HCA", 22, "start", 700, PURPLE)
    f.text(1030, 518, "128개를 하나로  →  길이는 S/128", 17, "start", 600)
    f.text(1030, 548, "겹치지 않는다", 17, "start", 600)
    return f


# ---------------------------------------------------------------- 텐서 연산 카드
def _cards(f):
    GRN, RED = "#4d9a56", "#c2479c"
    cw_, ch_, x0, px = 262, 200, 130, 280
    BG = {"g": ("#f3faf4", "#bfe0c4", GRN), "r": ("#fdf3f9", "#efc3de", RED), "n": ("#f7f8fb", "#d3d6e2", "#7b8196")}

    def g_comp(gx, gy, l2, kind="c"):
        f.rect(gx, gy, 228, 16, KINDS[kind][0], KINDS[kind][1], 4, 1.6)
        f.arrow(gx + 12, gy + 19, gx + 12, gy + 31)
        f.rect(gx, gy + 34, l2, 16, SOLID[kind], KINDS[kind][1], 4, 1.6)

    def g_idx(gx, gy):
        f.rect(gx, gy + 8, 34, 34, KINDS["q"][0], KINDS["q"][1], 5, 1.8)
        f.text(gx + 17, gy + 31, "q", 16, mono=True)
        f.text(gx + 47, gy + 33, "·", 26)
        f.rect(gx + 60, gy + 15, 168, 20, KINDS["i"][0], KINDS["i"][1], 4, 1.8)

    def g_topk(gx, gy):
        hs = [8, 12, 34, 9, 14, 7, 11, 38, 10, 8, 13, 30, 9, 12, 7, 10, 9]
        for i, h in enumerate(hs):
            f.rect(gx + i * 13.5, gy + 48 - h, 9, h, "#3b7dd8" if h > 25 else "#b9d3f5", "none", 2)

    def g_gather(gx, gy):
        sel = (2, 7, 11)
        for i in range(17):
            f.rect(gx + i * 13.5, gy, 10, 16, SOLID["c"] if i in sel else KINDS["c"][0], "none", 2)
        for n_, i in enumerate(sel):
            f.line(gx + i * 13.5 + 5, gy + 18, gx + 95 + n_ * 14, gy + 32, "#9a9fb3", 1.3)
            f.rect(gx + 90 + n_ * 14, gy + 34, 11, 16, SOLID["c"], "none", 2)

    def g_att(gx, gy, l2):
        f.rect(gx, gy + 8, 34, 34, KINDS["q"][0], KINDS["q"][1], 5, 1.8)
        f.text(gx + 17, gy + 31, "q", 16, mono=True)
        f.text(gx + 47, gy + 33, "·", 26)
        f.rect(gx + 60, gy + 8, l2, 34, SOLID["c"], KINDS["c"][1], 5, 1.8)

    def card(i, y, title, glyph, l1, l2, kind, tag):
        x = x0 + i * px
        bg, bd, col = BG[kind]
        f.rect(x, y, cw_, ch_, bg, bd, 10, 1.6)
        f.text(x + 16, y + 32, title, 19, "start", 700)
        glyph(x + 16, y + 46)
        f.text(x + 16, y + 126, l1, 15, "start", mono=True)
        f.text(x + 16, y + 148, l2, 15, "start", 700, mono=True)
        tw = len(tag) * 15 + 22
        f.rect(x + 16, y + 162, tw, 26, col, "none", 13)
        f.text(x + 16 + tw / 2, y + 180, tag, 14, weight=700, color="#ffffff")

    def none(i, y, name):
        x = x0 + i * px
        f.rect(x, y, cw_, ch_, "#ffffff", "#d3d6e2", 10, 1.5, "6 5")
        f.text(x + cw_ / 2, y + ch_ / 2 - 2, name, 20, weight=700, color="#a9aec0")
        f.text(x + cw_ / 2, y + ch_ / 2 + 26, "없다", 17, color="#a9aec0")

    def arrows(y, color="#5d6377"):
        for i in range(4):
            f.arrow(x0 + i * px + cw_ + 1, y + ch_ / 2, x0 + (i + 1) * px - 2, y + ch_ / 2, color=color)

    return card, none, arrows, g_comp, g_idx, g_topk, g_gather, g_att, (cw_, ch_, x0, px)


# ---------------------------------------------------------------- 28. CSA · HCA 텐서 연산
def fig_v4_tensor():
    f = Fig("28-v4-tensor", 1560, 680)
    f.title(40, 44, "텐서 연산으로 보면 — CSA와 HCA")
    GRN, RED = "#4d9a56", "#c2479c"
    card, none, arrows, g_comp, g_idx, g_topk, g_gather, g_att, (cw_, ch_, x0, px) = _cards(f)

    # ---- CSA
    y = 78
    f.text(40, y + 96, "CSA", 26, "start", 700)
    f.text(40, y + 122, "4개씩", 15, "start", color=MUTED)
    card(0, y, "① 압축", lambda gx, gy: g_comp(gx, gy, 57), "(S × 512)", "→ (S/4 × 512)", "n", "4토큰마다 한 번")
    card(1, y, "② indexer 점수", g_idx, "(64 × 128) × (128 × S/4)", "→ (1 × S/4)", "g", "행렬 × 행렬")
    card(2, y, "③ top-k", g_topk, "(1 × S/4)", "→ 위치 1,024개", "r", "비교 · 선택")
    card(3, y, "④ gather", g_gather, "위치 1,024개", "→ (1,024 × 512)", "r", "흩어진 읽기")
    card(4, y, "⑤ attention", lambda gx, gy: g_att(gx, gy, 44), "(128 × 512) × (512 × k)", "→ (128 × k)", "g",
         "행렬 × 행렬")
    arrows(y)
    f.line(40, 302, 1520, 302, "#e3e5ee", 2, "6 6")

    # ---- HCA
    y = 326
    f.text(40, y + 96, "HCA", 26, "start", 700)
    f.text(40, y + 122, "128개씩", 15, "start", color=MUTED)
    card(0, y, "① 압축", lambda gx, gy: g_comp(gx, gy, 9), "(S × 512)", "→ (S/128 × 512)", "n", "128토큰마다 한 번")
    none(1, y, "indexer")
    none(2, y, "top-k")
    none(3, y, "gather")
    card(4, y, "② attention", lambda gx, gy: g_att(gx, gy, 110), "(128 × 512) × (512 × n)", "→ (128 × n)", "g",
         "행렬 × 행렬")
    arrows(y)

    f.rect(40, 552, 1480, 96, "#fafbfd", "#d3d6e2", 10, 1.5)
    f.rect(64, 572, 16, 16, GRN, "none", 4)
    f.text(90, 586, "행렬곱", 16, "start", 700)
    f.rect(170, 572, 16, 16, RED, "none", 4)
    f.text(196, 586, "행렬곱이 아닌 일", 16, "start", 700)
    f.text(420, 586, "k = 1,024 + 최근 128", 15, "start", color=MUTED, mono=True)
    f.text(680, 586, "n = S/128 + 최근 128", 15, "start", color=MUTED, mono=True)
    f.text(64, 626, "CSA는 top-k와 gather가 토큰마다 · 층마다 돈다.", 18, "start", 600)
    f.text(560, 626, "HCA는 처음부터 끝까지 이어 읽는 행렬곱 하나다.", 18, "start", 600)
    return f


# ---------------------------------------------------------------- 29. QSA 읽기 패턴
def fig_qsa_read():
    f = Fig("29-qsa-read", 1560, 470)
    f.title(40, 44, "같은 양을 읽어도 — 토큰 단위와 블록 단위")
    TOK, RED, GRN = "#6b4fc8", "#c2479c", "#4d9a56"
    n, cw, gap, x0 = 64, 14, 4, 330
    pitch = cw + gap
    dsa = {2, 7, 11, 12, 19, 23, 28, 30, 35, 41, 44, 46, 51, 55, 58, 62}
    blocks = {1, 6, 9, 13}

    y = 96
    f.text(40, y + 16, "DSA", 24, "start", 700)
    f.text(40, y + 42, "토큰 하나씩 고른다", 16, "start", color=MUTED)
    f.cells(x0, y, [TOK if i in dsa else OFF for i in range(n)], cw, 30, gap, 3)
    f.text(x0, y + 58, "읽는 토큰 2,048개  ·  읽는 곳 최대 2,048군데", 17, "start", 600)

    y = 210
    f.text(40, y + 16, "QSA", 24, "start", 700)
    f.text(40, y + 42, "4토큰 블록으로 고른다", 16, "start", color=MUTED)
    f.cells(x0, y, [TOK if i // 4 in blocks else OFF for i in range(n)], cw, 30, gap, 3)
    for b in range(n // 4):
        on = b in blocks
        f.rect(x0 + b * 4 * pitch - 2, y - 4, 4 * pitch - gap + 4, 38, "none", TOK if on else "#d3d6e2", 6,
               1.8 if on else 1)
    f.text(x0, y + 62, "읽는 토큰 2,048개  ·  읽는 곳 512군데", 17, "start", 600)
    f.text(x0 + 420, y + 62, "— 블록 안은 메모리에서 이어져 있다", 16, "start", color=MUTED)

    f.rect(40, 330, 1480, 108, "#fafbfd", "#d3d6e2", 10, 1.5)
    f.text(64, 368, "얻는 것", 17, "start", 700, GRN)
    f.text(160, 368, "읽는 곳이 1/4이다. 이어진 4토큰을 한 번에 가져온다", 17, "start")
    f.text(64, 406, "내는 것", 17, "start", 700, RED)
    f.text(160, 406, "필요한 토큰이 하나뿐이어도 블록 4개를 다 읽는다 — 블록이 클수록 읽기는 쉬워지고 선택은 거칠어진다", 17, "start")
    return f


# ---------------------------------------------------------------- 30. QSA 텐서 연산
def fig_qsa_tensor():
    f = Fig("30-qsa-tensor", 1560, 420)
    f.title(40, 44, "텐서 연산으로 보면 — QSA")
    GRN, RED, TOK = "#4d9a56", "#c2479c", "#6b4fc8"
    card, none, arrows, g_comp, g_idx, g_topk, g_gather, g_att, (cw_, ch_, x0, px) = _cards(f)

    def g_blocks(gx, gy):
        sel = (1, 3)
        for i in range(16):
            on = i // 4 in sel
            f.rect(gx + i * 14, gy, 11, 16, TOK if on else "#e6e8f1", "none", 2)
        for n_, b in enumerate(sel):
            f.line(gx + b * 56 + 26, gy + 18, gx + 78 + n_ * 64, gy + 32, "#9a9fb3", 1.3)
            for i in range(4):
                f.rect(gx + 52 + n_ * 64 + i * 13, gy + 34, 11, 16, TOK, "none", 2)

    def g_gqa(gx, gy):
        f.rect(gx, gy + 8, 34, 34, KINDS["q"][0], KINDS["q"][1], 5, 1.8)
        f.text(gx + 17, gy + 31, "q", 16, mono=True)
        f.text(gx + 47, gy + 33, "·", 26)
        f.rect(gx + 66, gy + 2, 70, 34, TOK, "none", 5)
        f.rect(gx + 60, gy + 10, 70, 34, TOK, "#ffffff", 5, 1.5)

    y = 78
    f.text(40, y + 96, "QSA", 26, "start", 700)
    f.text(40, y + 122, "4개씩", 15, "start", color=MUTED)
    card(0, y, "① 블록 키", lambda gx, gy: g_comp(gx, gy, 57, "i"), "(S × 128)", "→ (S/4 × 128)", "n", "4토큰마다 평균")
    card(1, y, "② indexer 점수", g_idx, "(4 × 128) × (128 × S/4)", "→ (1 × S/4)", "g", "행렬 × 행렬")
    card(2, y, "③ top-k", g_topk, "(1 × S/4)", "→ 블록 512개", "r", "비교 · 선택")
    card(3, y, "④ gather", g_blocks, "블록 512개", "→ 토큰 2,048개", "r", "블록째 읽기")
    card(4, y, "⑤ attention (GQA)", g_gqa, "(12 × 256) × (256 × 2,048)", "→ (12 × 2,048)  × 2그룹", "g", "행렬 × 행렬")
    arrows(y)

    f.rect(40, 310, 1480, 78, "#fafbfd", "#d3d6e2", 10, 1.5)
    f.rect(64, 324, 16, 16, GRN, "none", 4)
    f.text(90, 338, "행렬곱", 16, "start", 700)
    f.rect(170, 324, 16, 16, RED, "none", 4)
    f.text(196, 338, "행렬곱이 아닌 일", 16, "start", 700)
    f.text(64, 372, "DSA와 같은 다섯 단계다.  달라진 것은 ②③의 길이(S → S/4)와 ④의 단위(토큰 → 블록)다.", 18, "start", 600)
    return f


# ---------------------------------------------------------------- 31. QSA 층 배치
def fig_qsa_layers():
    f = Fig("31-qsa-layers", 1560, 300)
    f.title(40, 44, "Qwen3.8-Flash-Next의 층 배치 — 48층")
    TOK = "#6b4fc8"
    x0 = 60
    for i in range(48):
        q = i % 4 == 3
        f.rect(x0 + i * 30, 84, 25, 44, TOK if q else "#cfd3e0", "none", 4)
    f.text(x0, 154, "1", 14, "start", color=MUTED, mono=True)
    f.text(x0 + 47 * 30 + 25, 154, "48", 14, "end", color=MUTED, mono=True)

    f.rect(60, 186, 700, 84, "#f7f8fb", "#d3d6e2", 10, 1.5)
    f.rect(84, 204, 22, 22, "#cfd3e0", "none", 4)
    f.text(118, 222, "Gated DeltaNet — 36층", 19, "start", 700)
    f.text(84, 254, "과거를 고정 크기 상태 하나에 담는다  ·  KV cache 없음", 16, "start", color=MUTED)
    f.rect(800, 186, 700, 84, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.rect(824, 204, 22, 22, TOK, "none", 4)
    f.text(858, 222, "QSA — 12층", 19, "start", 700, PURPLE)
    f.text(824, 254, "원본 토큰을 직접 찾아 읽는다  ·  KV cache 있음", 16, "start", color=MUTED)
    return f


FIGS = [fig_decode, fig_heads, fig_mla_cache, fig_mla_absorb, fig_patterns, fig_dsa, fig_v4, fig_v4_layers, fig_qsa,
        fig_cache_size, fig_knobs, fig_lineage, fig_mha, fig_mla_shape, fig_swa, fig_store_read, fig_dsa_shape,
        fig_dsa_structure, fig_journey, fig_amortize, fig_mha_mqa, fig_mla_flow, fig_kv_cache, fig_kv_bars, fig_mla_tensor,
        fig_axes, fig_compress, fig_v4_tensor, fig_qsa_read, fig_qsa_tensor, fig_qsa_layers]
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
