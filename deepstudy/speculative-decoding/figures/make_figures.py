"""speculative decoding / MTP 스터디용 그림 생성기. 실행: python make_figures.py [그림 이름 ...]  → 같은 폴더에 *.svg, *.png

수치의 근거
- 채택 규칙                : Leviathan et al. arXiv:2211.17192, Chen et al. arXiv:2302.01318
- EAGLE-3                  : arXiv:2503.01840 (decoder 1층, 3개 층 feature 융합, SGLang 배치별 처리량 표)
- MTP 모듈                 : DeepSeek-V3 arXiv:2412.19437 2.2절 (RMSNorm · concat · M_k · TRM_k · 공유 head)
- DFlash                   : arXiv:2602.06036 (draft 5층, 블록 16, target 5개 층 feature를 모든 draft 층의 K/V로 주입,
                             Qwen3-8B 평균 4.86× · τ 6.49, EAGLE-3 트리 60은 2.02× · τ 3.40, SGLang 동시 요청 수별 표)
- DSpark                   : arXiv:2607.05147 (마르코프 head rank 256, confidence head)
- DFlash 2                 : inco.ai/blog/dflash2 (상위 16개 후보의 경로 선택, 2-tap conv)
- Qwen3-8B 파라미터         : HF config (36층, d 4096, d_ff 12288, 쿼리 헤드 32 · KV 헤드 8 · 헤드 차원 128, vocab 151,936)
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
GRID = "#e6e8f1"
KINDS = {  # fill, stroke
    "t": ("#fdeadf", "#eb6834"),   # target
    "d": ("#e2eefc", "#2a78d6"),   # drafter
    "c": ("#eceafa", "#6b4fc8"),   # 확정된 캐시
    "ok": ("#e4f0e4", "#4d9a56"),  # 채택
    "no": ("#f8e6f1", "#c2479c"),  # 거부 · 버림
    "a": ("#fdefd8", "#d18a1c"),   # target에서 꺼낸 hidden
    "n": ("#eff0f6", "#c9cddb"),
}
SOLID = {"t": "#eb6834", "d": "#2a78d6", "c": "#6b4fc8", "ok": "#4d9a56", "no": "#c2479c", "a": "#e0a030", "n": "#c9cddb"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # 차트 계열 — 이 순서로만 쓴다


def esc(s):
    s = str(s)
    lead = len(s) - len(s.lstrip(" "))
    s = " " * lead + s[lead:]   # SVG는 앞쪽 공백을 지운다 — 들여쓰기를 살린다
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
                 f'font-weight="{weight}" fill="{color}">{esc(s)}</text>')

    def lines(self, x, y, rows, size=17, lh=28, anchor="start", color=INK, weight=400):
        for i, r in enumerate(rows):
            self.text(x, y + i * lh, r, size, anchor, weight, color)

    def title(self, x, y, s, size=24, anchor="start"):
        self.text(x, y, s, size, anchor, 700, PURPLE)

    def box(self, x, y, w, h, kind, label="", sub=None, fs=18, mono=False, subsize=15, weight=400):
        fill, stroke = KINDS[kind]
        self.rect(x, y, w, h, fill, stroke)
        if label:
            self.text(x + w / 2, y + h / 2 + fs * 0.35, label, fs, mono=mono, weight=weight)
        if sub:
            self.text(x + w / 2, y + h + subsize + 8, sub, subsize, color=MUTED)

    def strike(self, x, y, w, h):
        self.line(x + 8, y + h - 8, x + w - 8, y + 8, KINDS["no"][1], 2)

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

    def poly(self, pts, color, sw=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.raw(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" '
                 f'stroke-linejoin="round" stroke-linecap="round"{d}/>')

    def dot(self, x, y, r, fill):
        self.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')

    def svg(self):
        defs = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
                'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#5d6377"/></marker></defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}">{defs}<rect width="100%" height="100%" fill="#ffffff"/>' + "".join(self.o) + "</svg>")


# ---------------------------------------------------------------- 공통 예시 · 도우미
# 모든 그림이 같은 예시를 쓴다: 프롬프트 "The cat sat", 직전에 생성된 "on"(아직 KV 없음),
# 초안 "the mat and slept", target은 세 번째 자리에서 "."을 고른다 → 채택 2개 + "." = 3개 확정.
PROMPT = ["The", "cat", "sat"]
DRAFT = ["the", "mat", "and", "slept"]
TOUT = ["the", "mat", ".", "…", "…"]


def pend(f, x, y, w, h, label, fs=18):
    """확정됐지만 아직 KV가 없는 토큰 — 점선 상자."""
    f.rect(x, y, w, h, "#ffffff", "#7b8196", 6, 2, "6 4")
    f.text(x + w / 2, y + h / 2 + fs * 0.35, label, fs, mono=True)


def legend(f, x, y, items=("c", "p", "d", "t", "ok", "no")):
    names = {"c": "KV가 있는 확정 토큰", "p": "확정됐지만 KV가 아직 없는 토큰", "d": "초안", "t": "target이 고른 토큰",
             "ok": "채택된 초안", "no": "버린 초안"}
    for k in items:
        if k == "p":
            f.rect(x, y - 13, 26, 18, "#ffffff", "#7b8196", 4, 1.8, "4 3")
        else:
            f.rect(x, y - 13, 26, 18, KINDS[k][0], KINDS[k][1], 4, 1.8)
        f.text(x + 34, y + 1, names[k], 14, "start", color=MUTED)
        x += 48 + len(names[k]) * 13


def summary(f, x, y, w, bold, muted=None):
    h = 80 if muted else 52
    f.rect(x, y, w, h, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(x + 24, y + 33, bold, 17, "start", 700)
    if muted:
        f.text(x + 24, y + 62, muted, 15, "start", color=MUTED)


# ---------------------------------------------------------------- 16. 시간 흐름
def fig_time_flow():
    f = Fig("16-time-flow", 1560, 600)
    f.title(40, 44, "같은 세 토큰을 만드는 시간 — 일반 생성 vs speculative decoding")
    y0 = 84
    f.text(40, y0 + 28, "지금 상태", 17, "start", 700)
    for i, t in enumerate(PROMPT):
        f.box(200 + i * 92, y0, 82, 42, "c", t, fs=16, mono=True)
    f.text(200 + 133, y0 + 64, "프롬프트 · KV 있음", 14, color=MUTED)
    pend(f, 486, y0, 82, 42, "on", 16)
    f.text(527, y0 + 64, "방금 생성 · KV 없음", 14, color=MUTED)
    legend(f, 760, y0 + 26, ("c", "p", "d", "t"))

    x0, u = 230, 290
    y1 = 220
    f.text(40, y1 + 30, "일반 생성", 18, "start", 700)
    f.text(40, y1 + 54, "토큰 하나에 forward 한 번", 14, "start", color=MUTED)
    for i, (a, b) in enumerate([("on", "the"), ("the", "mat"), ("mat", ".")]):
        x = x0 + i * (u + 10)
        f.box(x, y1, u, 50, "t", "target forward", fs=16)
        f.text(x + u / 2, y1 + 76, f"입력 {a}  →  출력 {b}", 15, color=MUTED)
    xe = x0 + 3 * (u + 10) - 10

    y2 = 360
    f.text(40, y2 + 30, "speculative", 18, "start", 700)
    f.text(40, y2 + 54, "초안 4개 + forward 한 번", 14, "start", color=MUTED)
    for i, t in enumerate(DRAFT):
        f.box(x0 + i * 44, y2, 38, 50, "d")
    f.text(x0 + 85, y2 + 76, "초안 the mat and slept", 15, color=MUTED)
    xv = x0 + 4 * 44 + 6
    f.box(xv, y2, u + 40, 50, "t", "target forward — 5자리를 한 번에", fs=16)
    f.text(xv + (u + 40) / 2, y2 + 76, "확정  the  mat  .", 15, weight=700)
    f.line(xv + u + 54, y2 + 25, xe, y2 + 25, "#4d9a56", 2, "6 6")
    f.text((xv + u + 54 + xe) / 2, y2 + 16, "아낀 시간", 16, weight=700, color="#3d7c45")
    f.arrow(x0, 470, xe + 20, 470)
    f.text(xe + 20, 494, "시간", 14, "end", color=MUTED)
    summary(f, 40, 504, 1480, "같은 세 토큰(the mat .)을 만드는 데 target forward가 3번에서 1번으로 줄었다.",
            "대신 초안을 만드는 시간이 들고, 5자리를 한 번에 처리하는 forward는 한 자리일 때보다 조금 길다.")
    return f


# ---------------------------------------------------------------- 17. 입력과 출력의 한 칸 차이
def fig_shift():
    f = Fig("17-shift-by-one", 1560, 600)
    f.title(40, 44, "verify의 입력과 출력은 한 칸씩 어긋나 있다")
    cw, step, x0 = 150, 200, 360
    yi, yb, yo = 120, 214, 330
    f.text(40, yi + 30, "target에 넣는 5개", 17, "start", 700)
    f.text(40, yo + 30, "자리마다 고른", 17, "start", 700)
    f.text(40, yo + 54, "다음 토큰", 17, "start", 700)
    for i, t in enumerate(["on"] + DRAFT):
        x = x0 + i * step
        if i == 0:
            pend(f, x, yi, cw, 46, t)
        else:
            f.box(x, yi, cw, 46, "d", t, fs=18, mono=True)
        f.text(x + cw / 2, yi - 12, f"자리 {i}", 14, color=MUTED)
    marks = [("#4d9a56", "✓ d1과 같다"), ("#4d9a56", "✓ d2와 같다"), ("#c2479c", "✗ d3과 다르다"), ("#b9bdcc", "보지 않는다"),
             ("#b9bdcc", "")]
    for i, t in enumerate(TOUT):
        x = x0 + i * step
        f.box(x, yo, cw, 46, "t", t, fs=18, mono=True)
        col, s = marks[i]
        if i < 4:
            f.line(x + cw / 2 + 10, yo - 2, x + step + cw / 2 - 10, yi + 50, col, 2.5, None if i < 3 else "6 5", True)
        if s:
            f.text(x + cw / 2, yo + 74, s, 15, weight=700 if i < 3 else 400, color=col if i < 3 else MUTED)
    f.rect(x0 - 10, yb, 4 * step + cw + 20, 56, KINDS["t"][0], KINDS["t"][1], 8, 2)
    f.text(x0 + 2 * step + cw / 2, yb + 35, "target forward 1번 — 다섯 자리 모두 '여기까지 보고 다음 토큰'을 고른다", 16, weight=700)
    f.text(x0 + 4 * step + cw / 2, yo + 74, "비교할 초안이 없다", 15, color=MUTED)
    f.text(x0 + 4 * step + cw / 2, yo + 96, "다 맞으면 보너스 토큰", 15, color=MUTED)
    summary(f, 40, 492, 1480, "자리 i의 출력은 '자리 i까지 보고 고른 다음 토큰'이다. 그래서 바로 다음 칸(자리 i+1)의 초안과 비교한다.",
            "화살표는 비교하는 짝이다. 자리 2에서 target은 '.'을 골랐고 초안은 'and'였다 — 여기서 멈춘다.")
    return f


# ---------------------------------------------------------------- 18. hidden과 토큰을 둘 다 받는 이유
def fig_hidden_token():
    f = Fig("18-hidden-and-token", 1560, 640)
    f.title(40, 44, "drafter가 hidden과 토큰을 둘 다 받는 이유")
    # target 쪽
    f.text(40, 104, "target이 자리 'sat'을 처리했다 (prefill)", 18, "start", 700)
    for i, t in enumerate(PROMPT):
        f.box(60 + i * 128, 130, 118, 40, "c", t, fs=16, mono=True)
    f.rect(60, 196, 374, 60, KINDS["t"][0], KINDS["t"][1], 8, 2)
    f.text(247, 232, "target의 층들을 모두 지난다", 16, weight=700)
    f.arrow(247, 259, 247, 293)
    f.box(150, 296, 194, 44, "a", "hidden  h_sat", fs=17, mono=True)
    f.text(247, 362, "'The cat sat까지 읽은 이해'", 15, color=MUTED)
    f.text(247, 384, "벡터 한 줄 (8B 모델이면 숫자 4,096개)", 14, color=MUTED)
    f.arrow(247, 396, 247, 428)
    f.box(150, 431, 194, 40, "t", "출력 head", fs=16)
    f.text(247, 494, "어휘 전체에 대한 확률 → 'on' 0.6 · 'down' 0.3 …", 14, color=MUTED)
    f.text(247, 516, "여기서 'on'이 뽑혔다", 15, weight=700)
    # 가운데 질문
    f.line(520, 90, 520, 540, "#d3d6e2", 2, "6 6")
    f.text(560, 104, "hidden만 주면 drafter가 모르는 것", 18, "start", 700)
    f.lines(560, 144, ["h_sat은 'on'을 고르기 전의 상태다.",
                       "다음 칸에 'on'이 뽑혔는지 'down'이 뽑혔는지",
                       "hidden에는 들어 있지 않다.",
                       "",
                       "그다음 토큰은 그 선택에 따라 달라진다."], 16, 30)
    f.box(560, 316, 150, 40, "n", "on  →  the", fs=16, mono=True)
    f.box(730, 316, 150, 40, "n", "down  →  on", fs=16, mono=True)
    f.text(720, 380, "같은 h_sat이어도 다음 칸의 토큰이 다르면", 15, color=MUTED)
    f.text(720, 402, "맞혀야 할 답이 다르다", 15, color=MUTED)
    # drafter 쪽
    f.line(940, 90, 940, 540, "#d3d6e2", 2, "6 6")
    f.text(980, 104, "그래서 drafter는 둘을 함께 받는다", 18, "start", 700)
    f.box(990, 150, 220, 46, "a", "hidden  h_sat", fs=16, mono=True)
    f.text(1100, 218, "여기까지의 이해", 14, color=MUTED)
    f.box(1250, 150, 230, 46, "n", "임베딩  Emb(on)", fs=16, mono=True)
    f.text(1365, 218, "그다음에 실제로 온 토큰", 14, color=MUTED)
    f.arrow(1100, 232, 1200, 284)
    f.arrow(1365, 232, 1265, 284)
    f.box(1060, 288, 350, 56, "d", "drafter — 얇은 층 하나", fs=17, weight=700)
    f.arrow(1235, 347, 1235, 383)
    f.box(1160, 386, 150, 44, "d", "the", fs=18, mono=True)
    f.text(1235, 456, "on 다음 토큰을 맞힌다 — 첫 초안 d1", 15, weight=700)
    f.text(1235, 480, "임베딩: 토큰 ID를 벡터로 바꾼 것 (target의 표를 함께 쓴다)", 13, color=MUTED)
    summary(f, 40, 548, 1480, "hidden은 '여기까지 읽은 이해', 임베딩은 '그다음에 실제로 온 토큰'이다. drafter는 둘로 그다음 토큰을 맞힌다.",
            "문맥을 읽는 일은 target이 이미 했다. 그 결과를 받아 쓰니 drafter는 층 하나로도 꽤 잘 맞힌다.")
    return f


# ---------------------------------------------------------------- 19. 순차 초안과 병렬 초안
def fig_serial_parallel():
    f = Fig("19-serial-vs-parallel", 1560, 620)
    f.title(40, 44, "순차 초안과 병렬 초안 — forward를 몇 번 도나")
    f.text(40, 104, "순차 — 초안 하나에 forward 한 번", 18, "start", 700)
    x, y = 60, 128
    for i, t in enumerate(["on"] + DRAFT):
        if i == 0:
            pend(f, x, y, 80, 40, t, 16)
        else:
            f.box(x, y, 80, 40, "d", t, fs=16, mono=True)
        if i < 4:
            f.arrow(x + 83, y + 20, x + 99, y + 20)
            f.box(x + 102, y + 2, 50, 36, "d", "1층", fs=14, weight=700)
            f.arrow(x + 155, y + 20, x + 171, y + 20)
        x += 174
    f.text(60, 200, "앞에서 뽑은 토큰을 다음 스텝의 입력으로 넣는다. 그래서 차례로 돌 수밖에 없다", 15, "start", color=MUTED)

    f.text(40, 262, "병렬 — 블록 전체에 forward 한 번", 18, "start", 700)
    bx = 60
    for i, t in enumerate(["on", "[M]", "[M]", "[M]", "[M]"]):
        if i == 0:
            pend(f, bx + i * 130, 286, 116, 38, t, 15)
        else:
            f.box(bx + i * 130, 286, 116, 38, "n", t, fs=15, mono=True)
        f.arrow(bx + i * 130 + 58, 327, bx + i * 130 + 58, 343)
    f.box(bx, 346, 636, 42, "d", "draft 층 여러 개 — 블록 안의 자리들은 서로를 다 본다", fs=15, weight=700)
    for i, t in enumerate(DRAFT):
        f.arrow(bx + (i + 1) * 130 + 58, 391, bx + (i + 1) * 130 + 58, 407)
        f.box(bx + (i + 1) * 130, 410, 116, 38, "d", t, fs=15, mono=True)
    f.text(60, 474, "[M]은 '여기를 채워라'는 빈칸 표시(마스크 토큰)다. 네 자리를 동시에 채운다", 15, "start", color=MUTED)

    # 시간 비교
    tx = 860
    f.rect(tx - 20, 90, 680, 400, "#fafbfd", "#d3d6e2", 10, 1.5)
    f.text(tx, 126, "초안 4개를 만드는 시간", 18, "start", 700)
    f.text(tx, 176, "순차", 16, "start", 600)
    for i in range(4):
        f.rect(tx + 70 + i * 70, 158, 64, 28, SERIES[0], "none", 4)
    f.text(tx + 70, 214, "1층 × 4번 — 초안이 늘면 그만큼 는다", 14, "start", color=MUTED)
    f.text(tx, 276, "병렬", 16, "start", 600)
    f.rect(tx + 70, 258, 130, 28, SERIES[0], "none", 4)
    f.text(tx + 70, 314, "여러 층 × 1번 — 초안이 늘어도 거의 그대로", 14, "start", color=MUTED)
    f.lines(tx, 370, ["• 병렬은 한 번이 길다 (층이 더 많다)",
                      "• 그래도 초안을 길게 뽑을수록 순차보다 유리하다",
                      "• 대신 뒤 자리는 앞 자리에 무엇이 뽑혔는지 모르고 고른다"], 15, 28)
    summary(f, 40, 510, 1480, "순차는 초안 하나에 forward 한 번, 병렬은 블록 전체에 forward 한 번이다.",
            "병렬의 약점은 자리끼리의 의존이다 — 'New York' / 'San Francisco'가 반반이면 'New Francisco'가 나올 수 있다.")
    return f


# ---------------------------------------------------------------- 20. 컨텍스트를 받는 경로
def fig_context_paths():
    f = Fig("20-context-paths", 1560, 760)
    f.title(40, 44, "drafter는 컨텍스트를 어떻게 받나 — 받는 길이 캐시와 prefill을 정한다")
    rows = [("별도 draft 모델", "토큰만 받는다", "tok"), ("EAGLE-3 · MTP", "hidden → 층 하나를 돌려 캐시를 만든다", "fwd"),
            ("DFlash · DSpark", "hidden → 층마다 K · V로 바로 넣는다", "inj"), ("Gemma 4 MTP", "target의 KV를 직접 읽는다", "read")]
    for r, (name, how, kind) in enumerate(rows):
        y = 92 + r * 146
        f.rect(40, y, 1480, 128, "#fafbfd", "#e3e5ee", 10, 1.2)
        f.text(64, y + 40, name, 18, "start", 700)
        f.text(64, y + 66, how, 14, "start", color=MUTED)
        # target 탑과 KV
        f.rect(530, y + 16, 120, 96, KINDS["t"][0], KINDS["t"][1], 8, 2)
        f.text(590, y + 70, "target", 16, weight=700)
        for i in range(5):
            f.rect(440, y + 18 + i * 19, 70, 14, KINDS["c"][0], KINDS["c"][1], 3, 1.2)
        f.text(475, y + 124, "target KV", 12, color=MUTED)
        dx = 980
        if kind == "tok":
            f.box(700, y + 44, 120, 40, "n", "토큰 열", fs=15)
            f.arrow(824, y + 64, dx - 6, y + 64)
            f.text(900, y + 54, "토큰 ID", 13, color=MUTED)
            f.box(dx, y + 34, 200, 60, "d", "작은 LLM (전 층)", fs=15, weight=700)
            for i in range(5):
                f.rect(1220, y + 18 + i * 19, 60, 14, KINDS["d"][0], KINDS["d"][1], 3, 1.2)
            f.text(1250, y + 124, "자기 KV — 전 층", 12, color=MUTED)
            f.text(1300, y + 58, "prefill: 프롬프트를", 14, "start", color=MUTED)
            f.text(1300, y + 78, "자기 층으로 다시 읽는다", 14, "start", color=MUTED)
        elif kind == "fwd":
            f.arrow(654, y + 50, dx - 6, y + 62)
            f.text(760, y + 40, "hidden (마지막 층 · 또는 세 층)", 13, color="#b87410", weight=700)
            f.box(dx, y + 34, 200, 60, "d", "drafter 층 1개", fs=15, weight=700)
            for i in range(5):
                f.rect(1220, y + 18 + i * 19, 16, 14, KINDS["d"][0], KINDS["d"][1], 3, 1.2)
            f.text(1228, y + 124, "1층 KV", 12, color=MUTED)
            f.text(1300, y + 58, "prefill: hidden으로", 14, "start", color=MUTED)
            f.text(1300, y + 78, "층 하나를 한 번 돌린다", 14, "start", color=MUTED)
        elif kind == "inj":
            for i in range(3):
                f.arrow(654, y + 34 + i * 30, 700, y + 64)
            f.box(704, y + 44, 150, 40, "a", "섞어서 투영", fs=14)
            for i in range(5):
                f.arrow(858, y + 64, dx + 196, y + 20 + i * 19)
            for i in range(5):
                f.rect(dx + 200, y + 14 + i * 19, 40, 14, KINDS["d"][0], KINDS["d"][1], 3, 1.2)
            f.text(dx + 220, y + 124, "층마다 K · V", 12, color=MUTED)
            f.text(1300, y + 58, "prefill: 투영만 한다", 14, "start", color=MUTED)
            f.text(1300, y + 78, "(attention · FFN 없음)", 14, "start", color=MUTED)
        else:
            f.curve(475, y + 112, 475, y + 126, 820, y + 120, dx - 6, y + 70)
            f.text(810, y + 96, "KV를 읽기만 한다", 13, color=PURPLE, weight=700)
            f.box(dx, y + 34, 200, 60, "d", "drafter 4층", fs=15, weight=700)
            f.text(1300, y + 58, "자기 캐시도,", 14, "start", color=MUTED)
            f.text(1300, y + 78, "따로 하는 prefill도 없다", 14, "start", color=MUTED)
    summary(f, 40, 684, 1480, "위에서 아래로 갈수록 drafter가 target의 결과를 더 직접 쓴다. 받는 길이 캐시 크기와 prefill 비용을 정한다.")
    return f


# ---------------------------------------------------------------- 21. 사이클 시간과 확정 토큰
def fig_cycle_tokens():
    f = Fig("21-cycle-tokens", 1560, 570)
    f.title(40, 44, "사이클 하나의 시간은 거의 같고, 확정되는 토큰 수만 달라진다")
    x0 = 420
    rows = [("일반 생성 — forward 한 번", False, ["the"], "1개"),
            ("speculative · 첫 초안부터 틀림 (a = 0)", True, ["the"], "1개"),
            ("speculative · 앞의 2개 채택 (a = 2) — 우리 예", True, ["the", "mat", "."], "3개"),
            ("speculative · 전부 채택 (a = 4, 가정)", True, ["d1", "d2", "d3", "d4", "+1"], "5개")]
    for r, (name, spec, toks, n) in enumerate(rows):
        y = 100 + r * 84
        f.text(40, y + 30, name, 16, "start", 700 if r == 2 else 600)
        x = x0
        if spec:
            for i in range(4):
                f.rect(x + i * 30, y + 6, 26, 40, KINDS["d"][0], KINDS["d"][1], 4, 1.5)
            x += 128
            f.rect(x, y + 6, 300, 40, KINDS["t"][0], KINDS["t"][1], 6, 2)
            f.text(x + 150, y + 32, "verify (5자리)", 15)
            x += 300
        else:
            f.rect(x, y + 6, 280, 40, KINDS["t"][0], KINDS["t"][1], 6, 2)
            f.text(x + 140, y + 32, "target forward (1자리)", 15)
            x += 280
        tx = 920
        for i, t in enumerate(toks):
            last = i == len(toks) - 1
            kind = "t" if last else "ok"
            f.box(tx + i * 92, y + 6, 84, 40, kind, t, fs=15, mono=True)
        f.text(1400, y + 32, n, 18, weight=700)
    f.text(x0 + 200, 92, "사이클 시간", 15, color=MUTED)
    f.text(1100, 92, "확정되는 토큰", 15, color=MUTED)
    f.lines(40, 448, ["• 마지막 칸(주황)은 언제나 target이 직접 고른 토큰이다 — 그래서 최소 1개는 확정된다",
                      "• a = 4일 때 다섯째 토큰(+1)은 마지막 자리의 출력, 즉 보너스 토큰이다"], 15, 28)
    summary(f, 40, 496, 1480, "속도는 '사이클 하나의 시간'과 '평균 몇 개가 확정되나'의 비다.")
    return f


# ---------------------------------------------------------------- 22. 갈래별 target과 draft 모듈
def fig_anatomy():
    f = Fig("22-drafter-anatomy", 1560, 900)
    f.title(40, 44, "갈래별로 나란히 — target은 같고, draft 모듈만 다르다")
    AMB, GRAY = "#d18a1c", "#8b90a3"
    # 범례
    lx = 1010
    f.rect(lx, 26, 26, 18, KINDS["t"][0], KINDS["t"][1], 5, 2)
    f.text(lx + 34, 41, "target", 14, "start", color=MUTED)
    f.rect(lx + 104, 26, 26, 18, KINDS["d"][0], KINDS["d"][1], 5, 2)
    f.text(lx + 138, 41, "draft 모듈", 14, "start", color=MUTED)
    f.line(lx + 236, 35, lx + 266, 35, AMB, 4)
    f.text(lx + 274, 41, "hidden", 14, "start", color=MUTED)
    f.line(lx + 348, 35, lx + 378, 35, PURPLE, 4)
    f.text(lx + 386, 41, "target KV", 14, "start", color=MUTED)

    cw, ch = 474, 330

    def card(cx, cy, name, sub, flow, flow_col, draft, cap, mode, foot):
        f.rect(cx, cy, cw, ch, "#ffffff", "#dfe2ec", 14, 1.5)
        f.text(cx + 24, cy + 38, name, 20, "start", 700)
        f.text(cx + 24, cy + 62, sub, 14, "start", color=MUTED)
        # target — 모든 카드에서 같은 모양
        f.rect(cx + 28, cy + 92, 112, 112, KINDS["t"][0], KINDS["t"][1], 16, 2.5)
        f.text(cx + 84, cy + 154, "target", 17, weight=700)
        # 받는 것
        if flow:
            f.line(cx + 146, cy + 148, cx + 286, cy + 148, flow_col, 4, None, True)
            w = 14 * len(flow) * 0.62 + 24
            f.rect(cx + 216 - w / 2, cy + 118, w, 24, "#ffffff", flow_col, 12, 1.8)
            f.text(cx + 216, cy + 135, flow, 13, weight=700, color=flow_col)
        # draft 모듈
        dx, dy, dw, dh = cx + 296, cy + 92, 146, 112
        if draft == "llm":
            f.rect(dx, dy, dw, dh, KINDS["d"][0], KINDS["d"][1], 16, 2.5)
            for j in range(5):
                f.rect(dx + 18, dy + 16 + j * 18, dw - 36, 10, "#ffffff", KINDS["d"][1], 3, 1.2)
        elif draft == "none":
            f.rect(dx, dy + 22, dw, dh - 44, "#f4f5f9", GRAY, 14, 2, "6 5")
            f.text(dx + dw / 2, dy + dh / 2 + 6, "문자열 매칭", 15, weight=700, color="#5d6377")
        else:
            n = draft
            gap = 6
            bh = min(30, (dh - (n - 1) * gap) / n)
            top = dy + (dh - (n * bh + (n - 1) * gap)) / 2
            for j in range(n):
                f.rect(dx, top + j * (bh + gap), dw, bh, KINDS["d"][0], KINDS["d"][1], 7, 2.2)
        f.text(dx + dw / 2, cy + 226, cap, 13, color=MUTED)
        # 초안을 뽑는 방식
        y = cy + 252
        if mode == "seq":
            for i in range(4):
                f.rect(cx + 28 + i * 58, y, 38, 26, SOLID["d"], "none", 7)
                if i < 3:
                    f.line(cx + 70 + i * 58, y + 13, cx + 82 + i * 58, y + 13, KINDS["d"][1], 2, None, True)
            f.text(cx + 270, y + 19, "1개씩 · γ번 돈다", 15, "start", 700, KINDS["d"][1])
        elif mode == "par":
            f.rect(cx + 22, y - 6, 192, 38, "none", KINDS["d"][1], 10, 2)
            for i in range(4):
                f.rect(cx + 30 + i * 46, y, 38, 26, SOLID["d"], "none", 7)
            f.text(cx + 270, y + 19, "한 번에 γ개 · 1번", 15, "start", 700, KINDS["d"][1])
        else:
            for i in range(4):
                f.rect(cx + 28 + i * 46, y, 38, 26, "#c9cddb", "none", 7)
            f.text(cx + 270, y + 19, "찾아서 베낀다", 15, "start", 700, "#5d6377")
        f.text(cx + 24, cy + 312, foot, 14, "start")

    xs, ys = [40, 543, 1046], [86, 436]
    card(xs[0], ys[0], "별도 draft 모델", "옆에 둔 작은 LLM", "토큰만", GRAY, "llm", "작은 LLM 전 층 · 자기 KV", "seq",
         "문맥을 처음부터 다시 읽는다")
    card(xs[1], ys[0], "EAGLE-3", "나중에 따로 학습", "hidden × 3층", AMB, 1, "1층 + 작은 head", "seq",
         "target이 읽은 결과를 받아 1층으로 맞힌다")
    card(xs[2], ys[0], "MTP 모듈", "본 학습 때 함께 (DeepSeek · Qwen)", "hidden × 1층", AMB, 1, "본체와 같은 블록 1개", "seq",
         "EAGLE과 같은 모양 — 모델이 들고 나온다")
    card(xs[0], ys[1], "DFlash · DSpark", "나중에 따로 학습", "hidden × 5층", AMB, 5, "5층 · 층마다 target 정보", "par",
         "블록을 한 번에 — 길게 뽑아도 시간이 같다")
    card(xs[1], ys[1], "Gemma 4 MTP", "별도 체크포인트로 배포", "KV 직접 읽기", PURPLE, 4, "4층 · 자기 캐시 없음", "seq",
         "target의 KV를 읽어 캐시와 prefill이 없다")
    card(xs[2], ys[1], "n-gram · suffix", "모델 없음", None, GRAY, "none", "CPU 또는 GPU", "copy",
         "이미 나온 글에서 이어지는 구간을 찾는다")

    f.rect(40, 792, 1480, 80, "#f6f5fd", "#d9d2f3", 12, 1.5)
    f.text(64, 826, "target은 모두 같다 — 초안을 받아 한 번에 검증하는 forward 하나.", 18, "start", 700)
    f.text(64, 856, "다른 것은 세 가지 — 무엇을 받나 (토큰 · hidden · KV), draft가 얼마나 큰가 (층 수), 몇 번 도나 (γ번 · 1번).", 15, "start",
           color=MUTED)
    return f


# ---------------------------------------------------------------- 1. 가중치는 한 번 읽는다
def fig_weights_once():
    f = Fig("01-weights-once", 1560, 520)
    f.title(40, 44, "target forward 한 번의 시간은 '읽기'가 정한다 — 그래서 여러 줄을 한 번에 태운다")
    x0, u = 60, 190
    y1 = 124
    f.text(x0, y1 - 20, "일반 decode — 토큰 4개에 target forward 4번", 19, "start", 700)
    for i in range(4):
        x = x0 + i * (u + 8)
        f.box(x, y1, u, 54, "t", "가중치 전체 읽기", fs=16)
        f.text(x + u / 2, y1 + 80, f"토큰 {i + 1}", 16, color=MUTED)
    y2 = 300
    f.text(x0, y2 - 20, "speculative decoding — 초안 4개 + target forward 1번", 19, "start", 700)
    for i in range(4):
        f.box(x0 + i * 26, y2, 20, 54, "d")
    f.text(x0 + 49, y2 + 80, "초안 4번", 16, color=MUTED)
    xv = x0 + 4 * 26 + 6
    f.box(xv, y2, u + 30, 54, "t", "가중치 전체 읽기 — 한 번", fs=16)
    f.text(xv + (u + 30) / 2, y2 + 80, "5줄을 함께 검증 → 토큰 1 ~ 5개", 16, color=MUTED)
    xe = x0 + 4 * (u + 8) - 8
    f.line(xv + u + 44, y2 + 27, xe, y2 + 27, "#4d9a56", 2, "6 6")
    f.text((xv + u + 44 + xe) / 2, y2 + 18, "아낀 시간", 16, weight=700, color="#3d7c45")
    f.arrow(x0, 440, xe + 30, 440)
    f.text(xe + 30, 466, "시간", 15, "end", color=MUTED)

    px = 960
    f.rect(px - 24, 80, 600, 400, "#fafbfd", "#d3d6e2", 10, 1.5)
    f.text(px, 116, "target forward 한 번을 뜯어보면", 19, "start", 700)
    for y, name, cw in ((150, "1줄을 태울 때 (decode)", 7), (268, "5줄을 태울 때 (verify)", 35)):
        f.text(px, y + 16, name, 17, "start", 600)
        f.text(px, y + 50, "읽기", 15, "start", color=MUTED)
        f.rect(px + 52, y + 32, 420, 24, KINDS["t"][0], KINDS["t"][1], 4, 1.5)
        f.text(px, y + 82, "연산", 15, "start", color=MUTED)
        f.rect(px + 52, y + 64, cw, 24, SOLID["c"], "none", 3)
    f.lines(px, 400, ["• 읽기 — 가중치를 메모리에서 가져온다. 줄 수와 무관하다",
                      "• 연산 — 줄 수에 비례한다. 배치가 작으면 읽기보다 훨씬 짧다",
                      "• 걸리는 시간은 둘 중 긴 쪽이 정한다 → 5줄이어도 거의 그대로"], 16, 27)
    return f


# ---------------------------------------------------------------- 2. 한 사이클
def fig_cycle():
    f = Fig("02-cycle", 1560, 736)
    f.title(40, 44, "한 사이클 — draft → verify → accept  (γ = 4)")
    cw, ch, gap = 130, 46, 30

    def cx(i):
        return 640 + i * (cw + gap)

    lab = 40
    # 지금 상태
    y = 92
    f.text(lab, y + 30, "지금 상태", 19, "start", 700)
    for i, t in enumerate(PROMPT):
        f.box(310 + i * 104, y, 92, ch, "c", t, fs=18, mono=True)
    f.text(310 + 150, y + ch + 24, "프롬프트 · KV 있음", 15, color=MUTED)
    pend(f, cx(0), y, cw, ch, "on")
    f.text(cx(0) + cw + 16, y + 22, "마지막으로 확정된 토큰", 15, "start", color=MUTED)
    f.text(cx(0) + cw + 16, y + 42, "아직 target에 넣지 않았다 (KV 없음)", 15, "start", color=MUTED)

    # ① draft
    y = 208
    f.text(lab, y + 22, "① draft", 19, "start", 700)
    f.text(lab, y + 46, "drafter가 다음 γ개를 추측한다", 15, "start", color=MUTED)
    draft = ["the", "mat", "and", "slept"]
    for i, t in enumerate(draft):
        f.box(cx(i + 1), y, cw, ch, "d", t, fs=18, mono=True)
        f.text(cx(i + 1) + cw / 2, y - 10, f"d{i + 1}", 14, color=MUTED, mono=True)
        if i:
            f.arrow(cx(i) + cw + 3, y + ch / 2, cx(i + 1) - 4, y + ch / 2)
    f.curve(cx(0) + cw / 2, 92 + ch + 4, cx(0) + cw / 2, y + ch / 2, cx(0) + cw, y + ch / 2, cx(1) - 4, y + ch / 2)

    # ② verify
    y = 322
    f.text(lab, y + 22, "② verify", 19, "start", 700)
    f.text(lab, y + 46, "γ + 1개를 target에 한꺼번에 넣는다", 15, "start", color=MUTED)
    for i, t in enumerate(["on"] + draft):
        if i == 0:
            pend(f, cx(i), y, cw, ch, t)
        else:
            f.box(cx(i), y, cw, ch, "d", t, fs=18, mono=True)
        f.arrow(cx(i) + cw / 2, y + ch + 3, cx(i) + cw / 2, y + ch + 25)
    yt = y + ch + 28
    f.box(cx(0), yt, cx(4) + cw - cx(0), 44, "t", "target forward 1번  ·  가중치는 한 번만 읽는다", fs=17, weight=700)
    yo = yt + 44 + 28
    outs = ["the", "mat", ".", "…", "…"]
    for i, t in enumerate(outs):
        f.arrow(cx(i) + cw / 2, yt + 47, cx(i) + cw / 2, yo - 3)
        f.box(cx(i), yo, cw, ch, "t", t, fs=18, mono=True)
    f.text(lab, yo + 22, "target이 각 자리에서", 15, "start", color=MUTED)
    f.text(lab, yo + 42, "고른 '다음 토큰'", 15, "start", color=MUTED)
    marks = [("d1 'the'와 같다", "ok", "✓"), ("d2 'mat'과 같다", "ok", "✓"), ("d3 'and'와 다르다", "no", "✗"),
             ("볼 필요 없다", "n", ""), ("볼 필요 없다", "n", "")]
    for i, (s, k, m) in enumerate(marks):
        col = {"ok": "#3d7c45", "no": "#a8327f", "n": MUTED}[k]
        f.text(cx(i) + cw / 2, yo + ch + 24, (m + " " + s).strip(), 15, weight=700 if k != "n" else 400, color=col)

    # ③ accept
    y = 614
    f.text(lab, y + 22, "③ accept", 19, "start", 700)
    f.text(lab, y + 46, "맞는 데까지 + target의 토큰 하나", 15, "start", color=MUTED)
    f.box(cx(0), y, cw, ch, "ok", "the", fs=18, mono=True)
    f.box(cx(1), y, cw, ch, "ok", "mat", fs=18, mono=True)
    f.box(cx(2), y, cw, ch, "t", ".", fs=18, mono=True)
    f.text(cx(2) + cw / 2, y + ch + 22, "target이 고친 토큰", 15, color=MUTED)
    for i in (3, 4):
        f.box(cx(i), y, cw, ch, "no", draft[i - 1], fs=18, mono=True)
        f.strike(cx(i), y, cw, ch)
    f.text(cx(3) + cw + gap / 2, y + ch + 22, "버린다", 15, color=MUTED)
    f.text(cx(0) + cw + gap / 2, y + ch + 22, "채택 a = 2", 15, weight=700, color="#3d7c45")
    f.text(310, y + 30, "→  이번 사이클에 3개 확정", 18, "start", 700)
    legend(f, 40, 712)
    return f


# ---------------------------------------------------------------- 3. 채택 규칙
def fig_accept_rule():
    f = Fig("03-accept-rule", 1560, 560)
    f.title(40, 44, "채택 규칙 — 겹치는 만큼 받고, 모자란 만큼 다시 뽑는다")
    q = [0.5, 0.3, 0.1, 0.1]
    p = [0.3, 0.4, 0.2, 0.1]
    base, sc, bw = 400, 520, 44
    x0 = 110
    f.line(70, base, 640, base, "#8b90a3", 1.5)
    for v in (0.1, 0.2, 0.3, 0.4, 0.5):
        yy = base - v * sc
        f.line(70, yy, 640, yy, GRID, 1)
        f.text(60, yy + 5, f"{v:.1f}", 13, "end", color=MUTED, mono=True)
    for i, name in enumerate("ABCD"):
        x = x0 + i * 136
        hq, hp = q[i] * sc, p[i] * sc
        f.rect(x, base - hq, bw, hq, SERIES[0], "none", 4)
        f.rect(x + bw + 2, base - hp, bw, hp, SERIES[1], "none", 4)
        f.text(x + bw / 2, base - hq - 8, f"{q[i]:.1f}", 14, mono=True)
        f.text(x + bw + 2 + bw / 2, base - hp - 8, f"{p[i]:.1f}", 14, mono=True)
        m = min(q[i], p[i]) * sc
        f.rect(x - 5, base - m, 2 * bw + 12, m, "none", "#343a4d", 3, 2, "5 4")
        f.text(x + bw + 1, base + 24, f"토큰 {name}", 16)
    lx = 110
    f.rect(lx, 76, 16, 16, SERIES[0], "none", 3)
    f.text(lx + 24, 90, "q — drafter의 분포", 15, "start")
    f.rect(lx + 190, 76, 16, 16, SERIES[1], "none", 3)
    f.text(lx + 214, 90, "p — target의 분포", 15, "start")
    f.rect(lx + 380, 76, 16, 16, "none", "#343a4d", 3, 2, "5 4")
    f.text(lx + 404, 90, "겹치는 부분 min(p, q)", 15, "start")
    f.text(70, 470, "겹치는 부분의 합 = 0.3 + 0.3 + 0.1 + 0.1 = 0.8  →  이 위치의 채택 확률", 17, "start", 700)
    f.text(70, 500, "두 분포가 같으면 1, 전혀 안 겹치면 0이다", 15, "start", color=MUTED)

    rx = 720
    f.rect(rx, 76, 800, 150, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(rx + 24, 112, "drafter가 q에서 토큰 x를 뽑았다", 18, "start", 700)
    f.text(rx + 24, 146, "① 확률 min(1, p(x) / q(x)) 로 채택한다", 18, "start")
    f.text(rx + 24, 178, "② 거부되면  max(0, p − q) 를 정규화한 분포에서 다시 뽑는다", 18, "start")
    f.text(rx + 24, 208, "    여기서는 [0, 0.1, 0.1, 0] → [0, 0.5, 0.5, 0]", 15, "start", color=MUTED)

    ty = 268
    cols = [rx + 40, rx + 220, rx + 440, rx + 680]
    heads = ["토큰", "q로 뽑혀 채택", "거부 뒤 다시 뽑힘", "합"]
    for cxx, h in zip(cols, heads):
        f.text(cxx, ty, h, 16, weight=700)
    f.line(rx, ty + 12, rx + 800, ty + 12, "#b9bdcc", 1.5)
    rows = [("A", "0.5 × 0.6 = 0.30", "0.2 × 0 = 0", "0.30"), ("B", "0.3 × 1 = 0.30", "0.2 × 0.5 = 0.10", "0.40"),
            ("C", "0.1 × 1 = 0.10", "0.2 × 0.5 = 0.10", "0.20"), ("D", "0.1 × 1 = 0.10", "0.2 × 0 = 0", "0.10")]
    for r, row in enumerate(rows):
        yy = ty + 44 + r * 36
        for cxx, v in zip(cols, row):
            f.text(cxx, yy, v, 16, mono=True, weight=700 if cxx == cols[3] else 400)
    f.line(rx, ty + 162, rx + 800, ty + 162, "#b9bdcc", 1.5)
    f.text(rx + 24, ty + 196, "합 열이 target의 분포 p = [0.3, 0.4, 0.2, 0.1] 과 정확히 같다", 17, "start", 700)
    f.text(rx + 24, ty + 226, "0.2는 거부될 확률이다 (1 − 0.8). 깎은 몫을 거부했을 때 그대로 돌려준다", 15, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 차트 도우미
def axes(f, x0, y0, w, h, xlim, ylim, xticks, yticks, xlabel, ylabel, xfmt=str, yfmt=str):
    def X(v):
        return x0 + (v - xlim[0]) / (xlim[1] - xlim[0]) * w

    def Y(v):
        return y0 + h - (v - ylim[0]) / (ylim[1] - ylim[0]) * h

    for t in yticks:
        f.line(x0, Y(t), x0 + w, Y(t), GRID, 1)
        f.text(x0 - 10, Y(t) + 5, yfmt(t), 14, "end", color=MUTED, mono=True)
    f.line(x0, y0 + h, x0 + w, y0 + h, "#8b90a3", 1.5)
    for t in xticks:
        f.line(X(t), y0 + h, X(t), y0 + h + 5, "#8b90a3", 1.5)
        f.text(X(t), y0 + h + 24, xfmt(t), 14, color=MUTED, mono=True)
    f.text(x0 + w / 2, y0 + h + 52, xlabel, 15, color=MUTED)
    f.text(x0 - 44, y0 - 16, ylabel, 15, "start", color=MUTED)
    return X, Y


# ---------------------------------------------------------------- 4. τ 곡선
def fig_tau():
    f = Fig("04-tau-curve", 1560, 580)
    f.title(40, 44, "초안을 길게 뽑아도 τ는 1 / (1 − α) 에서 멈춘다")
    X, Y = axes(f, 100, 100, 820, 380, (0, 16), (0, 10), [1, 2, 4, 6, 8, 10, 12, 14, 16], [0, 2, 4, 6, 8, 10],
                "γ — 한 번에 뽑는 초안 토큰 수", "τ — 사이클 한 번에 확정되는 토큰 수")
    f.poly([(X(g), Y(g + 1)) for g in range(0, 10)], "#8b90a3", 2, "6 6")
    f.text(X(6.3), Y(9.0), "α = 1  (전부 맞을 때)  τ = γ + 1", 15, "end", color=MUTED)
    for a, col in zip((0.9, 0.8, 0.6), SERIES):
        pts = [(X(g), Y((1 - a ** (g + 1)) / (1 - a))) for g in range(0, 17)]
        f.poly(pts, col, 2.5)
        for g in (1, 2, 4, 8, 16):
            f.dot(X(g), Y((1 - a ** (g + 1)) / (1 - a)), 5, col)
        cap = 1 / (1 - a)
        f.line(X(16) + 8, Y(cap), X(16) + 30, Y(cap), col, 2, "4 4")
        f.text(X(16) + 38, Y(cap) + 5, f"천장 {cap:g}", 15, "start", mono=True)
        end = (1 - a ** 17) / (1 - a)
        f.text(X(16) - 6, Y(end) - 12, f"α = {a}", 16, "end", weight=700)
    nx = 1080
    f.rect(nx - 24, 96, 480, 392, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(nx, 134, "α — 한 자리가 채택될 확률", 18, "start", 700)
    f.text(nx, 160, "(앞 자리가 모두 채택됐다는 조건에서)", 15, "start", color=MUTED)
    f.text(nx, 206, "τ = 1 + α + α² + … + α^γ", 20, "start", 700, mono=True)
    f.text(nx, 240, "  = (1 − α^(γ+1)) / (1 − α)", 20, "start", 700, mono=True)
    f.lines(nx, 290, ["• 맨 앞의 1은 target이 직접 주는 토큰이다",
                      "• k번째 초안은 앞의 k−1개가 다 맞아야 산다",
                      "  그래서 α^k 씩 줄어든다",
                      "• α 0.8 → 0.9면 천장이 5 → 10이다",
                      "• γ를 늘리는 것보다 α를 올리는 쪽이 크다"], 16, 30)
    f.text(100, 560, "점은 γ = 1, 2, 4, 8, 16.  α가 자리마다 같다고 둔 단순 모형이다 (실제로는 자리마다 다르다 — 2장).", 15, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 5. drafter 지도
def fig_map():
    f = Fig("05-drafter-map", 1560, 660)
    f.title(40, 44, "drafter를 두 축으로 놓으면 — 어디에 사나 × 초안을 어떻게 뽑나")
    c1, c2, cwid = 330, 940, 590
    f.text(c1 + cwid / 2, 92, "순차 — 초안 토큰마다 forward 한 번  (γ번)", 18, weight=700)
    f.text(c2 + cwid / 2, 92, "병렬 — 블록 전체를 forward 한 번에", 18, weight=700)

    def pill(x, y, w, kind, s, sub=None):
        f.rect(x, y, w, 38, KINDS[kind][0], KINDS[kind][1], 19, 1.8)
        f.text(x + w / 2, y + 25, s, 16, weight=600)
        if sub:
            f.text(x + w + 12, y + 25, sub, 14, "start", color=MUTED)

    rows = [(112, 118, "별도 모델", "자기 임베딩 · 자기 층 · 자기 KV"),
            (250, 236, "target에 붙는 얇은 모듈", "target의 hidden과 LM head를 쓴다"),
            (506, 96, "모델 없음", "토큰 열에서 찾아 온다")]
    for y, h, name, sub in rows:
        f.text(40, y + h / 2 - 4, name, 18, "start", 700)
        f.text(40, y + h / 2 + 22, sub, 14, "start", color=MUTED)
    # 별도 모델
    f.rect(c1, 112, cwid, 118, "#fafbfd", "#d3d6e2", 10, 1.5)
    f.rect(c2, 112, cwid, 118, "#fafbfd", "#d3d6e2", 10, 1.5)
    pill(c1 + 20, 132, 250, "n", "draft model  (2022)")
    f.text(c1 + 20, 204, "작은 LLM을 통째로 옆에 둔다 · llama.cpp에서 흔하다", 14, "start", color=MUTED)
    pill(c2 + 20, 132, 190, "n", "PARD  (2025)")
    f.text(c2 + 20, 204, "작은 LLM에 마스크 토큰을 넣어 한 번에 여러 개", 14, "start", color=MUTED)
    # 얇은 모듈
    f.rect(c1 - 10, 242, 2 * cwid + 40, 252, "#f6f5fd", PURPLE, 12, 2.5)
    f.text(c1 + 8, 268, "지금의 주류", 15, "start", 700, PURPLE)
    f.rect(c1, 280, cwid, 200, "#ffffff", "#d3d6e2", 10, 1.5)
    f.rect(c2, 280, cwid, 200, "#ffffff", "#d3d6e2", 10, 1.5)
    pill(c1 + 20, 294, 160, "d", "Medusa", "2024 · 서로 독립인 head")
    pill(c1 + 20, 340, 260, "d", "EAGLE → EAGLE-3", "2024 – 25 · decoder 1층")
    pill(c1 + 20, 386, 160, "d", "MTP 모듈", "2024 – · DeepSeek · Qwen · GLM")
    pill(c1 + 20, 432, 230, "d", "Gemma 4 MTP", "2026 · target의 KV를 읽는다")
    pill(c2 + 20, 294, 160, "d", "P-EAGLE", "EAGLE-3의 병렬판")
    pill(c2 + 20, 340, 160, "d", "DFlash", "2026.02 · 5층 · 블록 diffusion")
    pill(c2 + 20, 386, 160, "d", "DSpark", "2026.07 · + 마르코프 head")
    pill(c2 + 20, 432, 160, "d", "DFlash 2", "2026.08 · + 후보 경로 선택")
    f.arrow(c1 + cwid - 150, 268, c2 + 150, 268, color=PURPLE)
    f.text((c1 + cwid + c2) / 2, 262, "2026년의 이동", 15, weight=700, color=PURPLE)
    # 모델 없음
    f.rect(c1, 506, 2 * cwid + 20, 96, "#fafbfd", "#d3d6e2", 10, 1.5)
    pill(c1 + 20, 522, 260, "n", "n-gram  (prompt lookup)")
    pill(c1 + 300, 522, 210, "n", "suffix decoding")
    f.text(c1 + 20, 586, "초안 비용이 거의 없다 · 입력을 베끼는 작업(코드 수정, RAG)에서만 잘 맞는다 · 다른 방법의 보조로 쓴다", 14, "start",
           color=MUTED)
    f.text(40, 640, "파란색 = target의 hidden state를 받아 쓰는 drafter.  이름 옆은 공개 시점과 한 줄 특징.", 15, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 6. 연표
def fig_timeline():
    f = Fig("06-timeline", 1560, 440)
    f.title(40, 44, "누가 초안을 만드나 — 2022 → 2026")
    ev = [("2022.11", "Speculative", "Decoding", "별도 draft 모델"),
          ("2024.01", "Medusa", "EAGLE", "target 위에 head"),
          ("2024.06", "EAGLE-2", "", "문맥마다 다른 트리"),
          ("2024.12", "DeepSeek-V3", "MTP", "본 학습에 넣는다"),
          ("2025.03", "EAGLE-3", "", "여러 층의 hidden"),
          ("2025.09", "Qwen3-Next MTP", "FastMTP", "1층을 여러 스텝"),
          ("2026.02", "DFlash", "", "블록을 한 번에"),
          ("2026.05", "Gemma 4", "MTP", "target KV를 읽는다"),
          ("2026.07", "DSpark", "", "부하에 맞춘 검증"),
          ("2026.08", "DFlash 2", "", "후보 경로 선택")]
    xs = [100 + i * 151 for i in range(10)]
    yl = 250
    eras = [(0, 0, "n", "별도 모델"), (1, 5, "d", "target에 붙는 얇은 모듈 · 초안은 순차"), (6, 9, "ok", "초안을 병렬로 · 검증 길이를 부하에 맞춘다")]
    for a, b, k, name in eras:
        x1, x2 = xs[a] - 66, xs[b] + 66
        f.rect(x1, 78, x2 - x1, 34, KINDS[k][0], KINDS[k][1], 8, 1.5)
        f.text((x1 + x2) / 2, 101, name, 15, weight=700)
    f.line(40, yl, 1520, yl, "#8b90a3", 2.5)
    for i, (d, n1, n2, note) in enumerate(ev):
        x = xs[i]
        up = i % 2 == 0
        f.dot(x, yl, 7, PURPLE)
        f.text(x, yl + (30 if up else -18), d, 14, color=MUTED, mono=True)
        if up:
            f.line(x, yl - 10, x, yl - 44, "#b9bdcc", 1.5)
            yy = yl - 102 if n2 else yl - 78
            f.text(x, yy, n1, 17, weight=700)
            if n2:
                f.text(x, yy + 24, n2, 17, weight=700)
            f.text(x, yl - 54, note, 14, color=MUTED)
        else:
            f.line(x, yl + 10, x, yl + 44, "#b9bdcc", 1.5)
            f.text(x, yl + 68, n1, 17, weight=700)
            if n2:
                f.text(x, yl + 92, n2, 17, weight=700)
            f.text(x, yl + (116 if n2 else 92), note, 14, color=MUTED)
    f.text(40, 420, "간격은 시간에 비례하지 않는다.  날짜는 논문이나 공개 글이 나온 달.", 15, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 7. 얇은 순차 drafter 둘
def fig_thin_ar():
    f = Fig("07-thin-ar-drafters", 1560, 740)
    f.title(40, 44, "얇은 순차 drafter 둘 — EAGLE-3와 MTP 모듈은 거의 같은 모양이다")

    def panel(px, name, sub, in1, mid_l, cat, layer, head, head_kind, out):
        f.rect(px, 70, 720, 500, "#fafbfd", "#d3d6e2", 12, 1.5)
        f.text(px + 24, 104, name, 20, "start", 700)
        f.text(px + 24, 130, sub, 15, "start", color=MUTED)
        y = 164
        f.box(px + 40, y, 300, 44, "a", in1, fs=15)
        f.box(px + 390, y, 280, 44, "n", "토큰 임베딩 Emb(t)  ·  d", fs=15)
        y2 = y + 74
        f.arrow(px + 190, y + 47, px + 190, y2 - 3)
        f.arrow(px + 530, y + 47, px + 530, y2 - 3)
        f.box(px + 70, y2, 240, 38, "d", mid_l, fs=15)
        f.box(px + 430, y2, 200, 38, "d", "RMSNorm", fs=15)
        y3 = y2 + 70
        f.arrow(px + 190, y2 + 41, px + 300, y3 - 3)
        f.arrow(px + 530, y2 + 41, px + 430, y3 - 3)
        f.box(px + 150, y3, 420, 38, "d", cat, fs=15)
        y4 = y3 + 66
        f.arrow(px + 360, y3 + 41, px + 360, y4 - 3)
        f.box(px + 40, y4, 600, 44, "d", layer, fs=15, weight=700)
        y5 = y4 + 72
        f.arrow(px + 360, y4 + 47, px + 360, y5 - 3)
        f.box(px + 150, y5, 420, 38, head_kind, head, fs=15)
        f.arrow(px + 360, y5 + 41, px + 360, y5 + 60)
        f.text(px + 360, y5 + 82, out, 16, weight=700)
        # 되먹임 — 층의 출력이 다음 스텝의 hidden 입력 자리로
        f.poly([(px + 644, y4 + 22), (px + 700, y4 + 22), (px + 700, y - 20), (px + 190, y - 20)], "#5d6377", 2.2, "6 5")
        f.arrow(px + 190, y - 20, px + 190, y - 4, dash="6 5")

    panel(40, "EAGLE-3", "target은 그대로 두고 나중에 따로 학습한다",
          "target hidden — 낮은 · 중간 · 높은 층  ·  3d", "선형 3d → d → RMSNorm",
          "이어 붙이기  →  2d", "decoder layer × 1  —  Q · K · V 투영이 2d를 받는다 · 자기 KV",
          "자기 LM head — 자주 나오는 3.2만 토큰만", "d", "다음 토큰의 분포")
    panel(800, "MTP 모듈 — DeepSeek-V3", "본 학습 때 보조 손실로 함께 학습한다",
          "target 마지막 층 hidden  ·  d", "RMSNorm",
          "이어 붙이기 → 선형 M  2d → d", "Transformer 블록 × 1  —  본체와 같은 블록 (V3: MLA + MoE)",
          "target의 출력 head (공유)  d → V", "t", "그다음 토큰의 분포")
    f.text(400, 596, "점선 — 두 번째 스텝부터는 자기가 낸 hidden이 target hidden 자리에 들어간다", 15, color=MUTED)
    f.text(1160, 596, "점선 — 더 뽑으려면 같은 블록을 다시 돈다 (V3는 모듈이 하나다)", 15, color=MUTED)

    ty = 640
    cols = [60, 360, 960]
    f.line(40, ty - 26, 1520, ty - 26, "#d3d6e2", 1.5)
    for cxx, h in zip(cols, ["", "EAGLE-3", "MTP 모듈"]):
        f.text(cxx, ty, h, 16, "start", 700)
    rows = [("언제 학습하나", "target을 다 학습한 뒤, target은 얼려 두고", "target과 함께 처음부터 (보조 손실)"),
            ("target에서 받는 것", "세 층의 hidden을 섞은 것", "마지막 층의 hidden"),
            ("target과 함께 쓰는 것", "임베딩 표  (LM head는 작은 어휘로 따로 둔다)", "임베딩 표 · 출력 head")]
    for r, row in enumerate(rows):
        for cxx, v in zip(cols, row):
            f.text(cxx, ty + 30 + r * 27, v, 15, "start", color=MUTED if cxx == cols[0] else INK)
    return f


# ---------------------------------------------------------------- 8. 병렬 drafter
def fig_parallel():
    f = Fig("08-parallel-drafters", 1560, 530)
    f.title(40, 44, "DFlash 계열의 장치 — target 정보를 모든 층에 넣고, 블록 안의 의존을 되살린다")
    # target 정보 주입
    y0 = 70
    f.text(40, y0 + 6, "target의 정보는 어디로 들어가나 — 입력이 아니라 모든 draft 층의 K · V로", 19, "start", 700)
    f.rect(60, y0 + 34, 150, 200, KINDS["t"][0], KINDS["t"][1], 8, 2)
    f.text(135, y0 + 60, "target", 17, weight=700)
    for i in range(5):
        yy = y0 + 84 + i * 30
        f.rect(74, yy, 122, 20, "#ffffff", KINDS["t"][1], 3, 1.2)
        f.arrow(214, yy + 10, 300, y0 + 134)
    f.text(135, y0 + 256, "5개 층의 hidden을 꺼낸다", 14, color=MUTED)
    f.box(304, y0 + 106, 300, 56, "a", "이어 붙이기 5d → 선형 → d · RMSNorm", fs=14)
    f.text(454, y0 + 186, "확정된 토큰마다 한 줄", 14, color=MUTED)
    for i in range(5):
        yy = y0 + 46 + i * 38
        f.arrow(608, y0 + 134, 716, yy + 14)
        f.rect(720, yy, 92, 28, KINDS["c"][0], KINDS["c"][1], 4, 1.5)
        f.text(766, yy + 20, "K · V", 14, mono=True)
        f.rect(820, yy, 250, 28, KINDS["d"][0], KINDS["d"][1], 4, 1.5)
        f.text(945, yy + 20, f"draft 층 {i + 1}", 14)
    f.text(895, y0 + 256, "층마다 자기 투영으로 K · V를 만들어 캐시에 둔다", 14, color=MUTED)
    f.lines(1110, y0 + 62, ["• 초안 블록의 자리들이 이 K · V를 읽는다",
                            "• target이 이미 계산한 표현을",
                            "  모든 층에서 직접 참조하는 셈이다",
                            "• EAGLE-3는 같은 정보를 입력 한 곳에만 넣는다",
                            "• 그래서 draft를 깊게 쌓을수록 잘 맞는다"], 15, 28)

    # 세대
    y1 = 370
    f.line(40, y1 - 24, 1520, y1 - 24, "#d3d6e2", 1.5)
    gens = [("DFlash", ["블록의 모든 자리를 동시에 뽑는다",
                        "뒤 자리는 앞 자리가 무엇으로 뽑혔는지 모른다",
                        "→ 블록 뒤쪽으로 갈수록 덜 맞는다"]),
            ("DSpark", ["직전에 실제로 뽑힌 토큰으로 다음 자리의 로짓을 고친다",
                        "(표 조회 + 작은 곱 한 번 · rank 256)",
                        "자리마다 통과 확률도 함께 낸다 → 검증 길이를 정한다"]),
            ("DFlash 2", ["자리마다 상위 16개 후보를 남긴다",
                          "이웃한 후보 쌍에 점수를 매겨 이어지는 경로 하나를 고른다",
                          "층 안에 앞 자리를 섞는 2-tap conv를 넣는다"])]
    for i, (name, rows) in enumerate(gens):
        gx = 40 + i * 500
        f.rect(gx, y1, 480, 136, "#fafbfd", "#d3d6e2", 10, 1.5)
        f.text(gx + 20, y1 + 32, name, 18, "start", 700, KINDS["d"][1])
        f.lines(gx + 20, y1 + 62, rows, 14, 25)
        if i < 2:
            f.arrow(gx + 482, y1 + 68, gx + 498, y1 + 68)
    return f


# ---------------------------------------------------------------- 9. verify의 모양
def fig_verify_shapes():
    f = Fig("09-verify-shapes", 1560, 670)
    f.title(40, 44, "verify forward — decode에서 한 줄이던 것이 K줄이 된다")
    S, K = 12, 5
    cs, cg = 24, 3

    def panel(px, name, rows):
        f.text(px, 96, name, 20, "start", 700)
        # 가중치 곱
        yb = 190
        for r in range(rows):
            f.rect(px, yb - (rows * 20) / 2 + r * 20, 130, 16, KINDS["n"][0], KINDS["n"][1], 3, 1.2)
        f.text(px + 65, yb + 76, f"입력  {rows} × d", 15, color=MUTED, mono=True)
        f.text(px + 152, yb + 8, "·", 30)
        f.rect(px + 176, yb - 62, 124, 124, KINDS["t"][0], KINDS["t"][1], 8, 2)
        f.text(px + 238, yb - 2, "가중치", 17, weight=700)
        f.text(px + 238, yb + 22, "d × n", 15, color=MUTED, mono=True)
        f.text(px + 324, yb + 8, "=", 24)
        for r in range(rows):
            f.rect(px + 350, yb - (rows * 20) / 2 + r * 20, 130, 16, KINDS["n"][0], KINDS["n"][1], 3, 1.2)
        f.text(px + 415, yb + 76, f"출력  {rows} × n", 15, color=MUTED, mono=True)
        # 점수 표
        gy = 330
        f.text(px, gy - 14, "attention 점수 — 가로는 읽는 캐시 줄, 세로는 새 토큰", 16, "start", 600)
        for r in range(rows):
            for c in range(S):
                f.rect(px + c * (cs + cg), gy + r * (cs + cg), cs, cs, SOLID["c"], "none", 3)
            if rows > 1:
                for c in range(K):
                    on = c <= r
                    f.rect(px + (S + c) * (cs + cg) + 14, gy + r * (cs + cg), cs, cs, SOLID["d"] if on else GRID, "none", 3)
        f.text(px + S * (cs + cg) / 2, gy + rows * (cs + cg) + 20, "확정된 캐시  S줄", 15, color=MUTED)
        if rows > 1:
            f.text(px + (S + K / 2) * (cs + cg) + 14, gy + rows * (cs + cg) + 20, "이번에 쓴 K줄", 15, color=MUTED)

    panel(60, "decode — 새 토큰 1줄", 1)
    f.line(770, 76, 770, 500, "#d3d6e2", 2, "6 6")
    panel(820, f"verify — 새 토큰 K = {K}줄", K)
    f.text(820 + (S + K) * (cs + cg) + 30, 330 + 58, "계단 모양", 15, "start", 700)
    f.text(820 + (S + K) * (cs + cg) + 30, 330 + 80, "j번째 줄은", 14, "start", color=MUTED)
    f.text(820 + (S + K) * (cs + cg) + 30, 330 + 100, "j번째까지만 본다", 14, "start", color=MUTED)
    f.rect(40, 524, 1480, 126, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.lines(64, 556, ["• 가중치 — 같은 행렬을 한 번 읽는다. 곱하는 줄만 K배다  (행렬 × 벡터가 행렬 × 행렬이 된다)",
                      "• 캐시 — K개 쿼리가 같은 S줄을 읽는다. 커널이 쿼리 사이에서 재사용하면 한 번만 읽는다 (이상적인 경우).  연산은 K·(S+K)에 비례한다",
                      "• 새로 쓰는 KV는 1줄에서 K줄이 된다"], 16, 30)
    return f


# ---------------------------------------------------------------- 10. KV cache의 한 사이클
def fig_kv_lifecycle():
    f = Fig("10-kv-lifecycle", 1560, 640)
    f.title(40, 44, "target의 KV cache — K줄을 쓰고, 맞은 만큼만 남긴다  (γ = 4, K = 5)")
    rh, rg, sw_ = 26, 4, 230
    y0 = 150

    def stack(x, base, extra):
        for r, t in enumerate(base):
            f.rect(x, y0 + r * (rh + rg), sw_, rh, KINDS["c"][0], KINDS["c"][1], 4, 1.5)
            f.text(x + sw_ / 2, y0 + r * (rh + rg) + 19, t, 15, mono=True)
        yy = y0 + len(base) * (rh + rg)
        for t, k in extra:
            f.rect(x, yy, sw_, rh, KINDS[k][0], KINDS[k][1], 4, 1.5)
            f.text(x + sw_ / 2, yy + 19, t, 15, mono=True)
            if k == "no":
                f.line(x + 10, yy + rh / 2, x + sw_ - 10, yy + rh / 2, KINDS["no"][1], 2)
            yy += rh + rg
        return yy

    xs = [50, 430, 810, 1190]
    names = ["① 사이클 시작", "② verify가 K줄을 쓴다", "③ accept — 남길 줄을 정한다", "④ 다음 사이클 시작"]
    for x, n in zip(xs, names):
        f.text(x, 110, n, 19, "start", 700)
    # ①
    yb = stack(xs[0], PROMPT, [])
    f.text(xs[0] + sw_ + 14, y0 + 50, "프롬프트", 14, "start", color=MUTED)
    pend(f, xs[0], yb + 14, 96, 34, "on", 15)
    f.text(xs[0] + 108, yb + 30, "방금 생성 · 아직", 14, "start", color=MUTED)
    f.text(xs[0] + 108, yb + 48, "KV가 없다", 14, "start", color=MUTED)
    # ②
    stack(xs[1], PROMPT, [("on", "c")] + [(t, "d") for t in DRAFT])
    f.text(xs[1] + sw_ + 14, y0 + 3 * (rh + rg) + 64, "새로 쓴", 14, "start", color=MUTED)
    f.text(xs[1] + sw_ + 14, y0 + 3 * (rh + rg) + 84, "K = 5줄", 15, "start", 700)
    # ③
    stack(xs[2], PROMPT, [("on", "c"), ("the", "ok"), ("mat", "ok"), ("and", "no"), ("slept", "no")])
    f.text(xs[2] + sw_ + 14, y0 + 3 * (rh + rg) + 34, "남긴다", 14, "start", color=MUTED)
    f.text(xs[2] + sw_ + 14, y0 + 3 * (rh + rg) + 54, "a + 1 = 3줄", 15, "start", 700)
    f.text(xs[2] + sw_ + 14, y0 + 6 * (rh + rg) + 20, "버린다", 14, "start", color=MUTED)
    f.text(xs[2] + sw_ + 14, y0 + 6 * (rh + rg) + 40, "γ − a = 2줄", 15, "start", 700)
    # ④
    yb = stack(xs[3], PROMPT + ["on", "the", "mat"], [])
    pend(f, xs[3], yb + 14, 96, 34, ".", 15)
    f.text(xs[3] + 108, yb + 30, "target이 고친 토큰", 14, "start", color=MUTED)
    f.text(xs[3] + 108, yb + 48, "다음 verify의 첫 줄이 된다", 14, "start", color=MUTED)
    for i in range(3):
        f.arrow(xs[i] + sw_ + 96, 104, xs[i + 1] - 16, 104)

    by = 440
    f.rect(40, by, 1480, 184, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(64, by + 34, "남는 줄 수 = 이번에 확정된 토큰 수  (채택 a개 + target의 1개).  target이 준 토큰 자신의 줄은 다음 사이클에 쓴다", 17,
           "start", 700)
    f.text(64, by + 74, "버리는 방법 ①  길이만 되돌린다", 16, "start", 700)
    f.text(64, by + 100, "요청의 길이 값을 거부된 수만큼 줄인다. 버린 줄은 그 자리에 그대로 있다가 다음 verify가 덮어쓴다  (vLLM)", 15, "start",
           color=MUTED)
    f.text(64, by + 132, "버리는 방법 ②  자리를 반납한다", 16, "start", 700)
    f.text(64, by + 156, "캐시 자리를 토큰 단위로 나눠 주는 경우. 버린 줄의 자리를 돌려주고 '토큰 → 자리' 표만 고친다. 트리 검증에는 이 방식이 필요하다", 15,
           "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 11. 트리와 마스크
def fig_tree():
    f = Fig("11-tree-mask", 1560, 640)
    f.title(40, 44, "트리 검증 — 갈래를 한 줄로 펴고, 자기 조상만 보게 가린다")
    nodes = [("on", 60, 300, "n"), ("the", 250, 210, "d"), ("a", 250, 410, "d"), ("mat", 440, 150, "d"),
             ("floor", 440, 270, "d"), ("mat", 440, 410, "d"), ("and", 630, 150, "d")]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6)]
    nw, nh = 110, 42
    for a, b in edges:
        f.arrow(nodes[a][1] + nw + 2, nodes[a][2] + nh / 2, nodes[b][1] - 4, nodes[b][2] + nh / 2)
    for i, (t, x, y, k) in enumerate(nodes):
        f.box(x, y, nw, nh, k, t, fs=17, mono=True)
        f.text(x + 8, y - 8, str(i), 14, "start", color=MUTED, mono=True)
    for d, x in enumerate([60, 250, 440, 630]):
        f.text(x + nw / 2, 500, f"위치 n{'' if d == 0 else '+' + str(d)}", 15, color=MUTED, mono=True)
    f.text(60, 100, "초안이 갈래로 뻗는다 (자리마다 후보 여러 개)", 18, "start", 700)
    f.text(60, 540, "숫자는 한 줄로 폈을 때의 순서다.  위치 번호는 깊이로 준다 — 형제는 같은 위치를 받는다", 15, "start", color=MUTED)

    gx, gy, cs, cg = 1010, 170, 44, 4
    anc = {0: [0], 1: [0, 1], 2: [0, 2], 3: [0, 1, 3], 4: [0, 1, 4], 5: [0, 2, 5], 6: [0, 1, 3, 6]}
    f.text(840, 100, "target이 쓰는 attention 마스크 — 칠한 칸만 본다", 18, "start", 700)
    f.rect(gx - 150, gy, 120, 7 * (cs + cg) - cg, SOLID["c"], "none", 6)
    f.text(gx - 90, gy + 3.5 * (cs + cg) - 8, "확정된", 15, color="#ffffff", weight=700)
    f.text(gx - 90, gy + 3.5 * (cs + cg) + 14, "컨텍스트", 15, color="#ffffff", weight=700)
    f.text(gx - 90, gy - 12, "전부 본다", 14, color=MUTED)
    for i, (t, *_r) in enumerate(nodes):
        f.text(gx + i * (cs + cg) + cs / 2, gy - 12, t, 14, mono=True)
        f.text(gx + 7 * (cs + cg) + 10, gy + i * (cs + cg) + 28, f"{i}  {t}", 15, "start", mono=True)
        for j in range(7):
            f.rect(gx + j * (cs + cg), gy + i * (cs + cg), cs, cs, SOLID["d"] if j in anc[i] else GRID, "none", 4)
    f.text(gx + 3.5 * (cs + cg), gy + 7 * (cs + cg) + 22, "읽는 줄 (이번에 쓴 7줄)", 14, color=MUTED)
    f.lines(840, 560, ["• 'floor'는 'on', 'the', 자기만 본다. 옆 갈래의 'mat'은 못 본다",
                       "• 일렬(chain) 초안은 이 표가 계단 모양이 되는 특별한 경우다"], 15, 26)
    return f


# ---------------------------------------------------------------- 12. drafter별 캐시
def fig_drafter_caches():
    f = Fig("12-drafter-caches", 1560, 620)
    f.title(40, 44, "drafter는 컨텍스트를 어디서 읽나 — 종류마다 캐시가 다르다")
    cols = [("별도 draft 모델", 56, "자기 층 전부의 KV", "토큰만 받는다"),
            ("EAGLE-3 · MTP 모듈", 16, "1층 분량의 KV", "target hidden을 받아 만든다"),
            ("Gemma 4 MTP", 0, "컨텍스트 캐시 없음", "target의 KV를 직접 읽는다"),
            ("DFlash · DSpark", 40, "draft 층 수만큼의 KV", "target hidden을 투영한 것"),
            ("n-gram · suffix", -1, "모델 캐시 없음", "토큰 열에서 문자열을 찾는다")]
    rows, rh, rg = 9, 20, 4
    for i, (name, dw, l1, l2) in enumerate(cols):
        x = 40 + i * 300
        f.rect(x, 76, 284, 420, "#fafbfd", "#d3d6e2", 12, 1.5)
        f.text(x + 142, 112, name, 18, weight=700)
        tx = x + 44
        for r in range(rows):
            f.rect(tx, 150 + r * (rh + rg), 80, rh, KINDS["c"][0], KINDS["c"][1], 3, 1.3)
        f.text(tx + 40, 150 + rows * (rh + rg) + 18, "target KV", 14, color=MUTED)
        f.text(tx + 40, 150 + rows * (rh + rg) + 36, "S줄 × 모든 층", 13, color=MUTED)
        dx = x + 170
        if dw > 0:
            for r in range(rows):
                f.rect(dx, 150 + r * (rh + rg), dw, rh, KINDS["d"][0], KINDS["d"][1], 3, 1.3)
            f.text(dx + dw / 2, 150 + rows * (rh + rg) + 18, "drafter KV", 14, color=MUTED)
        elif dw == 0:
            f.box(dx - 6, 226, 92, 60, "d", "drafter", fs=15, weight=700)
            f.arrow(tx + 84, 256, dx - 10, 256)
            f.text(dx + 40, 310, "읽기만 한다", 14, color=MUTED)
        else:
            f.box(dx - 16, 226, 110, 60, "n", "토큰 열", fs=15)
            f.text(dx + 39, 310, "CPU 또는 GPU", 14, color=MUTED)
        f.text(x + 142, 446, l1, 16, weight=700)
        f.text(x + 142, 472, l2, 14, color=MUTED)
    f.lines(40, 536, ["• 파란 막대의 폭 = 캐시를 두는 층 수 (개념적인 비율).  줄 수는 어느 쪽이든 컨텍스트 길이 S다",
                      "• 자기 캐시가 있는 drafter는 prefill도 따로 치른다.  target의 KV를 읽는 drafter와 모델 없는 방법은 그 비용이 없다"], 15, 28)
    return f


# ---------------------------------------------------------------- 13. 사이클 시간
def fig_cycle_time():
    f = Fig("13-cycle-time", 1560, 500)
    f.title(40, 44, "사이클 하나의 시간과 얻는 토큰 — Qwen3-8B, 배치 1  (개략 계산)")
    x0, sc = 330, 480     # target decode 한 스텝 = 480px
    rows = [("일반 decode", 0.0, 1.00, "1", "1.0×", "기준"),
            ("EAGLE-3  (트리 60 · 깊이 8)", 0.38, 1.10, "3.40", "2.3×", "보고값 2.02×"),
            ("DFlash  (블록 16)", 0.22, 1.03, "6.49", "5.2×", "보고값 4.86×")]
    f.text(x0, 96, "사이클 시간  (target decode 한 스텝 = 1)", 16, "start", 600)
    f.text(1150, 96, "확정 토큰 τ", 16, weight=600)
    f.text(1290, 96, "속도 = τ ÷ 시간", 16, weight=600)
    for t in (0, 0.5, 1.0, 1.5):
        f.line(x0 + t * sc, 112, x0 + t * sc, 370, GRID, 1)
        f.text(x0 + t * sc, 392, f"{t:g}", 14, color=MUTED, mono=True)
    for i, (name, d, v, tau, sp, rep) in enumerate(rows):
        y = 134 + i * 78
        f.text(40, y + 30, name, 17, "start", 600)
        x = x0
        if d:
            f.rect(x, y, d * sc - 2, 44, SERIES[0], "none", 4)
            f.text(x + d * sc / 2, y + 62, f"draft {d:.2f}", 14, color=MUTED, mono=True)
            x += d * sc
        f.rect(x, y, v * sc, 44, SERIES[1], "none", 4)
        f.text(x + v * sc / 2, y + 62, f"target forward {v:.2f}", 14, color=MUTED, mono=True)
        f.text(1150, y + 30, tau, 19, weight=700, mono=True)
        f.text(1290, y + 30, sp, 19, weight=700, mono=True)
        f.text(1400, y + 30, rep, 14, "start", color=MUTED)
    f.rect(x0, 418, 16, 16, SERIES[0], "none", 3)
    f.text(x0 + 24, 432, "draft", 15, "start")
    f.rect(x0 + 100, 418, 16, 16, SERIES[1], "none", 3)
    f.text(x0 + 124, 432, "target forward (verify)", 15, "start")
    f.text(40, 472, "막대는 가중치를 읽는 양과 연산량만으로 셈한 값이다 (4장).  τ와 보고값은 DFlash 논문의 Qwen3-8B 평균 (temperature 0).", 15,
           "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 14. 배치와 이득
def fig_batch():
    f = Fig("14-batch-speedup", 1560, 600)
    f.title(40, 44, "배치가 커지면 이득이 줄어든다 — 논문이 보고한 값")
    # 왼쪽 : EAGLE-3 논문
    f.text(100, 92, "Llama-3.1-8B · SGLang · H100 · MT-bench  (EAGLE-3 논문)", 16, "start", 600)
    bx = [2, 4, 8, 16, 24, 32, 48, 56, 64]
    X, Y = axes(f, 100, 130, 560, 340, (0, 64), (0.6, 2.0), [2, 8, 16, 24, 32, 48, 64], [0.6, 1.0, 1.4, 1.8],
                "배치 크기", "속도 배수 (speculation 없음 = 1)", yfmt=lambda v: f"{v:.1f}")
    f.line(100, Y(1.0), 660, Y(1.0), "#8b90a3", 1.5, "6 5")
    for name, ys, col in (("EAGLE", [1.40, 1.38, 1.23, 1.02, 0.93, 0.94, 0.88, 0.99, 0.99], SERIES[0]),
                          ("EAGLE-3", [1.81, 1.82, 1.62, 1.48, 1.39, 1.32, 1.38, 1.34, 1.38], SERIES[1])):
        f.poly([(X(b), Y(v)) for b, v in zip(bx, ys)], col, 2.5)
        for b, v in zip(bx, ys):
            f.dot(X(b), Y(v), 4.5, col)
        f.text(X(64) + 12, Y(ys[-1]) + 5, name, 15, "start", weight=700)
    f.text(X(12), Y(0.70), "점선 아래 = 오히려 느려진다", 14, color=MUTED)
    # 오른쪽 : DFlash 논문
    ox = 860
    f.text(ox, 92, "SGLang · B200  (DFlash 논문)", 16, "start", 600)
    cx_ = [1, 4, 8, 16, 32]
    X2, Y2 = axes(f, ox, 130, 420, 340, (0, 32), (0, 6), [1, 4, 8, 16, 32], [0, 1, 2, 3, 4, 5, 6],
                  "동시 요청 수", "속도 배수 (speculation 없음 = 1)")
    f.line(ox, Y2(1.0), ox + 420, Y2(1.0), "#8b90a3", 1.5, "6 5")
    ser = [("Qwen3-8B · 수학", [5.1, 4.5, 4.5, 3.9, 2.8], SERIES[0], 16),
           ("Qwen3-8B · 코드", [4.2, 3.6, 3.6, 3.0, 2.4], SERIES[1], 14),
           ("Qwen3-Coder-30B-A3B · 코드", [3.5, 3.0, 3.2, 3.2, 3.1], SERIES[2], -4)]
    for name, ys, col, dy in ser:
        f.poly([(X2(b), Y2(v)) for b, v in zip(cx_, ys)], col, 2.5)
        for b, v in zip(cx_, ys):
            f.dot(X2(b), Y2(v), 4.5, col)
        f.text(X2(32) + 12, Y2(ys[-1]) + dy, name, 15, "start", weight=700)
    f.text(40, 572, "두 그림은 모델 · GPU · 측정 방법이 달라 서로 비교할 수 없다.  각 그림 안에서 오른쪽으로 갈수록 내려가는 모양만 본다.", 15, "start",
           color=MUTED)
    return f


# ---------------------------------------------------------------- 15. 시뮬레이터 구조
def fig_sim():
    f = Fig("15-sim-structure", 1560, 620)
    f.title(40, 44, "시뮬레이터에 넣을 것 — 사이클 하나를 네 단계로 센다")
    ins = [("target의 op 표", "decode 한 스텝의 연산 · shape"), ("drafter 명세", "층 수 · 종류 · γ · 순차/병렬"),
           ("채택 모형", "자리별 채택 확률 α₁ … α_γ"), ("하드웨어", "메모리 대역폭 · 연산 성능 · 용량"),
           ("부하", "동시 요청 수 B · 컨텍스트 길이 S")]
    f.text(40, 96, "입력", 18, "start", 700)
    for i, (a, b) in enumerate(ins):
        y = 116 + i * 84
        f.rect(40, y, 300, 68, "#fafbfd", "#b9bdcc", 8, 1.5)
        f.text(60, y + 29, a, 17, "start", 700)
        f.text(60, y + 53, b, 14, "start", color=MUTED)
        f.arrow(344, y + 34, 400, 310)
    f.rect(404, 100, 780, 440, "#f6f5fd", PURPLE, 14, 2.5)
    f.text(428, 136, "사이클 한 번", 18, "start", 700, PURPLE)
    steps = [("d", "① draft", "drafter의 연산을 센다", "순차면 γ번, 병렬이면 1번"),
             ("t", "② verify", "target의 op 표에서 새 토큰 1줄을 K줄로 바꿔 센다", "가중치 읽기는 그대로, 연산과 KV 쓰기는 K배"),
             ("ok", "③ accept", "채택 수 a를 채택 모형에서 뽑는다", "앞에서부터 보다가 처음 틀린 자리에서 멈춘다"),
             ("c", "④ commit", "컨텍스트 S += a + 1,  KV는 K줄 쓰고 γ − a줄 버린다", "drafter의 캐시도 새로 확정된 줄만큼 채운다")]
    for i, (k, n, a, b) in enumerate(steps):
        y = 158 + i * 92
        f.rect(428, y, 732, 74, KINDS[k][0], KINDS[k][1], 8, 2)
        f.text(448, y + 32, n, 18, "start", 700)
        f.text(590, y + 30, a, 16, "start")
        f.text(590, y + 56, b, 14, "start", color=MUTED)
        if i < 3:
            f.arrow(794, y + 77, 794, y + 89)
    f.curve(1160, 471, 1176, 471, 1176, 195, 1163, 195, dash="6 5")
    f.text(794, 528, "요청이 끝날 때까지 되풀이한다", 14, color=MUTED)
    f.text(1240, 96, "출력", 18, "start", 700)
    outs = [("토큰/초", "요청 하나 · 시스템 전체"), ("토큰 간격의 분포", "평균과 p99 — 들쭉날쭉해진다"),
            ("KV 사용량", "target + drafter + 여유 K줄"), ("버려진 일", "거부된 줄의 연산과 쓰기")]
    for i, (a, b) in enumerate(outs):
        y = 116 + i * 106
        f.arrow(1188, 310, 1236, y + 34)
        f.rect(1240, y, 280, 68, "#fafbfd", "#b9bdcc", 8, 1.5)
        f.text(1260, y + 29, a, 17, "start", 700)
        f.text(1260, y + 53, b, 14, "start", color=MUTED)
    f.text(40, 590, "②가 새 코드를 가장 적게 요구한다 — 이미 있는 decode 비용 모형에서 새 토큰의 줄 수만 바꾸면 된다.", 15, "start", color=MUTED)
    return f


FIGS = [fig_anatomy, fig_time_flow, fig_shift, fig_hidden_token, fig_serial_parallel, fig_context_paths, fig_cycle_tokens,
        fig_weights_once, fig_cycle, fig_accept_rule, fig_tau, fig_map, fig_timeline, fig_thin_ar, fig_parallel,
        fig_verify_shapes, fig_kv_lifecycle, fig_tree, fig_drafter_caches, fig_cycle_time, fig_batch, fig_sim]
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


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
