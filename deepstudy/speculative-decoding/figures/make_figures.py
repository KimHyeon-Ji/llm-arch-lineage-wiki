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


# ---------------------------------------------------------------- 25. EAGLE-3 · MTP · DSpark 단계별
def fig_three_methods():
    f = Fig("25-three-methods", 1560, 930)
    f.title(40, 44, "EAGLE-3 · MTP · DSpark — 같은 사이클, 다른 초안과 검증")
    f.text(40, 76, "이번 그림의 설정: EAGLE-3 트리 (topk 2 · 깊이 3 · 예산 6) · MTP-1 · DSpark adaptive (한가할 때, 요청 A)   "
                   "|   직전 확정 토큰 on, target의 답은 'the mat .'", 14, "start", color=MUTED)
    C1, C2, C3 = 230, 820, 1250
    for x, s in ((C1, "① 초안을 어떻게 만드나"), (C2, "② target에 얼마나 넣나"), (C3, "③ 시뮬레이터에서 다르게 셀 것")):
        f.text(x, 118, s, 18, "start", 700, PURPLE)
    f.line(40, 132, 1520, 132, "#e3e5ee", 2)

    def node(x, y, s, kind="d", w=64, h=30, fs=14):
        f.box(x, y, w, h, kind, s, fs=fs, mono=True)

    def rows(x, y, toks, w=50):
        for i, t in enumerate(toks):
            if i == 0:
                pend(f, x + i * (w + 6), y, w, 30, t, 13)
            else:
                f.box(x + i * (w + 6), y, w, 30, "t", t, fs=12, mono=True)

    def label(y, name, sub):
        f.text(40, y, name, 24, "start", 700)
        f.text(40, y + 26, sub, 14, "start", color=MUTED)

    def lines(x, y, items):
        for i, (s, kind) in enumerate(items):
            col = {"b": INK, "m": MUTED, "blue": KINDS["d"][1], "pink": KINDS["no"][1], "green": KINDS["ok"][1]}[kind]
            f.text(x, y + i * 26, s, 15 if kind in ("b",) else 14, "start", 700 if kind in ("b", "blue", "pink", "green") else 400, col)

    # ---------- EAGLE-3
    y0 = 150
    label(y0 + 40, "EAGLE-3", "트리")
    lines(C1, y0 + 24, [("얇은 drafter를 깊이마다 1번 — 노드마다 상위 2개씩 붙인다", "b"),
                        ("깊이마다 점수 높은 노드 2개만 다시 펼친다", "m"),
                        ("다 펼친 10개 중 점수 순 6개만 남긴다 (예산)", "blue")])
    T = {"on": (C1 + 10, y0 + 170), "the": (C1 + 120, y0 + 130), "a": (C1 + 120, y0 + 210),
         "mat1": (C1 + 230, y0 + 105), "floor": (C1 + 230, y0 + 155), "mat2": (C1 + 230, y0 + 210), "and": (C1 + 340, y0 + 105)}
    for p, c in (("on", "the"), ("on", "a"), ("the", "mat1"), ("the", "floor"), ("a", "mat2"), ("mat1", "and")):
        (x1, y1), (x2, y2) = T[p], T[c]
        f.line(x1 + 64, y1 + 15, x2, y2 + 15, "#a3a7b8", 1.5)
    pend(f, T["on"][0], T["on"][1], 64, 30, "on", 14)
    for k, s in (("the", "the"), ("a", "a"), ("mat1", "mat"), ("floor", "floor"), ("mat2", "mat"), ("and", "and")):
        node(T[k][0], T[k][1], s)
    for d, x in ((1, C1 + 120), (2, C1 + 230), (3, C1 + 340)):
        f.text(x + 32, y0 + 256, f"깊이 {d}", 12, color=MUTED)
    rows(C2, y0 + 14, ["on", "the", "a", "mat", "floor", "mat", "and"])
    lines(C2, y0 + 76, [("노드 6개 + 직전 토큰 1 = 7줄을 한 번에", "b"),
                        ("각 노드는 자기 조상만 본다 (트리 마스크)", "m"),
                        ("the ✓ → mat ✓ → 답 '.'은 트리에 없다 → 멈춤", "m"),
                        ("확정 3 · 버림 4", "pink"),
                        ("최대 확정 = 깊이 + 1 (노드 수가 아니다)", "m")])
    lines(C3, y0 + 24, [("검증 줄(노드 수 + 1)과", "b"), ("최대 채택(깊이)을 따로 센다", "b"),
                        ("트리를 넓히면 채택 기회와 함께", "m"), ("검증 비용 · KV도 는다", "m")])
    f.line(40, y0 + 278, 1520, y0 + 278, "#eef0f5", 1.5)

    # ---------- MTP
    y0 = 445
    label(y0 + 40, "MTP", "MTP-1 · 일렬")
    lines(C1, y0 + 24, [("MTP 블록(본체 층과 같은 층)으로 하나씩 만든다", "b"),
                        ("MTP-1은 1개. 더 뽑으려면 같은 블록을 반복한다", "m")])
    pend(f, C1 + 10, y0 + 80, 64, 30, "on", 14)
    f.arrow(C1 + 78, y0 + 95, C1 + 116, y0 + 95)
    node(C1 + 120, y0 + 80, "the")
    rows(C2, y0 + 14, ["on", "the"])
    lines(C2, y0 + 76, [("초안 1 + 직전 토큰 1 = 2줄 — 모든 요청 같다", "b"),
                        ("앞에서부터 비교: the ✓ + 보너스 mat", "m"),
                        ("확정 2 · 버림 0", "green")])
    lines(C3, y0 + 24, [("초안을 늘리면 draft가", "b"), ("직렬로 길어진다", "b"),
                        ("draft ≈ γ × (블록 1개 + head)", "m"), ("뒤 칸 채택률이 빨리 떨어진다", "m")])
    f.line(40, y0 + 140, 1520, y0 + 140, "#eef0f5", 1.5)

    # ---------- DSpark
    y0 = 600
    label(y0 + 40, "DSpark", "일렬 + 검증 길이 고르기")
    lines(C1, y0 + 24, [("블록을 한 번에 → 5칸, 직전 토큰으로 가볍게 보정", "b"),
                        ("칸마다 '여기까지 맞을 확률'도 낸다", "blue")])
    pend(f, C1 + 10, y0 + 86, 56, 30, "on", 13)
    toks = [("the", ".95"), ("mat", ".85"), ("and", ".51"), ("slept", ".18"), ("…", ".04")]
    for i, (t, p) in enumerate(toks):
        x = C1 + 76 + i * 88
        if i < 2:
            node(x, y0 + 86, t, w=80)
        else:
            f.rect(x, y0 + 86, 80, 30, "#f4f5f9", "#c9cddb", 6, 1.5, "4 3")
            f.text(x + 40, y0 + 106, t, 14, color="#9aa0b3", mono=True)
        f.text(x + 40, y0 + 136, p, 12, color=MUTED, mono=True)
    cut = C1 + 76 + 2 * 88 - 4
    f.line(cut, y0 + 76, cut, y0 + 126, KINDS["no"][1], 2.5)
    f.text(cut + 4, y0 + 162, "스케줄러가 여기서 자른다", 13, "start", 700, KINDS["no"][1])
    rows(C2, y0 + 14, ["on", "the", "mat"])
    lines(C2, y0 + 76, [("5칸 중 앞 2칸 + 직전 토큰 1 = 3줄", "b"),
                        ("몇 칸 넣을지는 요청마다 · 스텝마다 다르다", "pink"),
                        ("(배치 전체의 확률과 부하를 보고 정한다)", "m"),
                        ("확정 3 · 버림 0 — 자른 칸은 검증하지 않는다", "green")])
    lines(C3, y0 + 24, [("생성 길이(5칸)와", "b"), ("검증 길이(ℓ)를 따로 센다", "b"),
                        ("검증 길이는 부하 · confidence로", "m"), ("스텝마다 바뀐다 — 스케줄러를 돌린다", "m")])

    summary(f, 40, 818, 1480, "트리는 많이 넣고 많이 버린다 · MTP-1은 적게 넣고 적게 얻는다 · DSpark는 통과할 만큼만 골라 넣는다",
            "넣는 줄 수가 verify 비용을, 초안을 만드는 방식이 draft 비용을 정한다")
    return f


def fig_mtp_chain():
    f = Fig("41-mtp-chain", 1560, 470)
    f.title(40, 44, "MTP — MTP 블록으로 초안을 하나씩, 더 뽑으려면 같은 블록을 반복한다")

    def chain(y, name, sub, n, verify_rows):
        f.text(40, y + 18, name, 18, "start", 700)
        f.text(40, y + 42, sub, 13, "start", color=MUTED)
        x = 260
        f.box(x, y, 150, 46, "a", "h + 임베딩(on)", fs=13)
        x += 150
        toks = ["the", "mat", "and"]
        for i in range(n):
            f.arrow(x + 4, y + 23, x + 30, y + 23)
            x += 34
            f.box(x, y, 120, 46, "d", "MTP 블록", fs=14)
            f.arrow(x + 124, y + 23, x + 150, y + 23)
            x += 154
            f.box(x, y + 6, 64, 34, "d", toks[i], fs=14, mono=True)
            x += 64
        x += 40
        f.rect(x, y, 230, 46, KINDS["t"][0], KINDS["t"][1], 6, 1.5)
        f.text(x + 115, y + 29, f"verify {verify_rows}줄", 15, weight=700, color=KINDS["t"][1])
        return x
    chain(110, "MTP-1", "DeepSeek-V3 기본", 1, 2)
    chain(200, "γ = 3", "같은 블록을 3번", 3, 4)
    f.text(260, 290, "draft 시간 ─────────────────────────▶   블록 수만큼 직렬로 늘어난다", 13, "start", color=MUTED)
    f.text(40, 336, "h = 직전 위치의 target hidden.  블록 = 본체 층 하나와 같은 구조 (DeepSeek: MLA + MoE) + target의 출력 head", 14, "start", color=MUTED)
    summary(f, 40, 362, 1480, "모델에 들어 있는 모듈이라 따로 학습하지 않는다. 대신 블록이 본체 층과 같아 draft 한 번이 무겁다",
            "반복해서 뒤 칸을 뽑으면 학습 때 없던 쓰임이라 채택률이 빨리 떨어진다 (DeepSeek-V3는 MTP 모듈 1개로 학습)")
    return f


def fig_eagle_tree():
    f = Fig("26-eagle-tree-budget", 1560, 760)
    f.title(40, 44, "EAGLE-2 · 3의 트리 초안 — 펼치고, 예산으로 자르고, 맞는 갈래를 따라간다 (topk 2 · 깊이 3 · 예산 6)")
    # 후보: (키, 표시, 부모, 깊이, 점수)
    C = [("the", "the", "on", 1, .60), ("a", "a", "on", 1, .25),
         ("the.mat", "mat", "the", 2, .42), ("the.floor", "floor", "the", 2, .12),
         ("a.mat", "mat", "a", 2, .125), ("a.big", "big", "a", 2, .075),
         ("the.mat.and", "and", "the.mat", 3, .21), ("the.mat.with", "with", "the.mat", 3, .084),
         ("a.mat.and", "and", "a.mat", 3, .05), ("a.mat..", ".", "a.mat", 3, .0375)]
    expanded = {"on", "the", "a", "the.mat", "a.mat"}
    slots = {"the.mat.and": 0, "the.mat.with": 1, "the.floor": 2, "a.mat.and": 3, "a.mat..": 4, "a.big": 5}
    Y = {k: 140 + v * 58 for k, v in slots.items()}
    Y["the.mat"] = (Y["the.mat.and"] + Y["the.mat.with"]) / 2
    Y["a.mat"] = (Y["a.mat.and"] + Y["a.mat.."]) / 2
    Y["the"] = (Y["the.mat"] + Y["the.floor"]) / 2
    Y["a"] = (Y["a.mat"] + Y["a.big"]) / 2
    Y["on"] = (Y["the"] + Y["a"]) / 2
    X = {0: 50, 1: 200, 2: 370, 3: 540}
    depth = {k: d for k, _, _, d, _ in C}
    depth["on"] = 0
    score = {k: s for k, _, _, _, s in C}
    label = {k: t for k, t, _, _, _ in C}
    W, H = 138, 36

    f.text(40, 100, "① 펼치기 — 깊이마다 점수 높은 노드 2개를 골라, 각자 상위 2개를 붙인다", 16, "start", 700)
    for k, _, p, d, s in C:
        f.line(X[depth[p]] + (W if p != "on" else 70), Y[p] + H / 2, X[d], Y[k] + H / 2, "#a3a7b8", 1.5)
    pend(f, X[0], Y["on"], 70, H, "on", 15)
    for k, t, p, d, s in C:
        kind = "d" if k in expanded else "n"
        f.box(X[d], Y[k], W, H, kind, "", fs=14)
        f.text(X[d] + 12, Y[k] + 24, t, 15, "start", 700, mono=True)
        f.text(X[d] + W - 10, Y[k] + 24, f"{s:.4f}".rstrip("0"), 13, "end", color=MUTED, mono=True)
    for d, x in X.items():
        if d:
            f.text(x + W / 2, 130, f"깊이 {d}", 13, color=MUTED)
    f.text(40, 500, "점수 = 경로의 확률 곱 (예: the → mat = 0.60 × 0.70 = 0.42). 파랑 = 다음 깊이로 펼친 노드", 13, "start", color=MUTED)
    f.text(40, 522, "drafter forward는 깊이마다 1번 — 깊이 1은 1줄, 깊이 2 · 3은 펼친 노드 2개를 한 번에", 13, "start", color=MUTED)

    # ② 예산
    bx = 740
    f.line(bx - 20, 90, bx - 20, 600, "#e3e5ee", 2, "6 6")
    f.text(bx, 100, "② 예산으로 자르기 — 점수 순으로 6개", 16, "start", 700)
    order = sorted(C, key=lambda c: -c[4])
    for i, (k, t, p, d, s) in enumerate(order):
        yy = 126 + i * 38
        keep = i < 6
        path = {"the": "the", "the.mat": "the mat", "a": "a", "the.mat.and": "the mat and", "a.mat": "a mat",
                "the.floor": "the floor", "the.mat.with": "the mat with", "a.big": "a big", "a.mat.and": "a mat and",
                "a.mat..": "a mat ."}[k]
        f.text(bx, yy + 20, path, 14, "start", 700 if keep else 400, INK if keep else MUTED, mono=True)
        f.rect(bx + 150, yy + 6, s * 300, 18, SOLID["d"] if keep else "#d6d9e3", "none", 3)
        f.text(bx + 156 + s * 300, yy + 20, f"{s:.4f}".rstrip("0"), 12, "start", color=MUTED, mono=True)
        if i == 5:
            f.line(bx, yy + 33, bx + 350, yy + 33, KINDS["no"][1], 2.5, "6 4")
            f.text(bx + 352, yy + 38, "예산 N = 6", 13, "start", 700, KINDS["no"][1])
    f.text(bx, 530, "부모의 점수 ≥ 자식의 점수 → 위에서 자르면", 13, "start", color=MUTED)
    f.text(bx, 550, "남은 노드는 저절로 트리가 된다", 13, "start", color=MUTED)

    # ③ 검증
    cx = 1150
    f.line(cx - 20, 90, cx - 20, 600, "#e3e5ee", 2, "6 6")
    f.text(cx, 100, "③ 남은 트리를 한 번에 검증", 16, "start", 700)
    kept = {"on": (cx, 280), "the": (cx + 90, 200), "a": (cx + 90, 360), "the.mat": (cx + 180, 160),
            "the.floor": (cx + 180, 240), "a.mat": (cx + 180, 360), "the.mat.and": (cx + 270, 160)}
    edges = [("on", "the"), ("on", "a"), ("the", "the.mat"), ("the", "the.floor"), ("a", "a.mat"), ("the.mat", "the.mat.and")]
    good = {("on", "the"), ("the", "the.mat")}
    for a_, b_ in edges:
        (x1, y1), (x2, y2) = kept[a_], kept[b_]
        col, sw = (KINDS["ok"][1], 3) if (a_, b_) in good else ("#a3a7b8", 1.5)
        f.line(x1 + 72, y1 + 15, x2, y2 + 15, col, sw)
    for k, (x, yy) in kept.items():
        t = "on" if k == "on" else label[k]
        if k == "on":
            pend(f, x, yy, 72, 30, t, 14)
        else:
            kind = "ok" if k in ("the", "the.mat") else "no"
            f.box(x, yy, 72, 30, kind, t, fs=14, mono=True)
    f.box(cx + 270, 210, 72, 30, "t", ".", fs=14, mono=True)
    f.text(cx + 306, 260, "target의 선택", 12, color=MUTED)
    f.text(cx, 430, "K = 7줄 (on + 노드 6개)", 15, "start", 700)
    f.text(cx, 456, "on → the ✓ → mat ✓ → target은 '.'", 14, "start")
    f.text(cx, 480, "'.'이 mat의 자식에 없다 → 멈춤", 14, "start", color=MUTED)
    f.text(cx, 512, "확정 3개 (the mat .) · 버림 4줄", 15, "start", 700, KINDS["no"][1])

    f.rect(40, 620, 1480, 120, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(64, 654, "세 손잡이 — 깊이(최대 채택 길이) · topk(깊이마다 펼치는 갈래) · 예산 N(target이 검증하는 노드 수)", 17, "start", 700)
    f.text(64, 684, "SGLang: --speculative-num-steps · --speculative-eagle-topk · --speculative-num-draft-tokens   (Llama 기본 5 · 4 · 8)", 15, "start",
           color=MUTED)
    f.text(64, 710, "EAGLE-2 · 3 논문: 깊이 6 · 8, 노드 60개 안팎. 예산이 클수록 많이 맞히지만, 버리는 줄도 많아진다", 15, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 27. DSpark의 검증 길이 고르기
DSPARK_REQ = {"A": [.95, .90, .60, .35, .20], "B": [.70, .50, .40, .35, .30], "C": [.97, .95, .92, .85, .70]}


def _dspark_plan(cost):
    surv = []
    for r, cs in DSPARK_REQ.items():
        a = 1.0
        for j, c in enumerate(cs):
            a *= c
            surv.append((a, r, j + 1))
    surv.sort(key=lambda t: -t[0])
    best, best_k, curve, tau = None, 0, [], 3.0
    curve.append(3.0)
    for k, (a, r, j) in enumerate(surv, 1):
        tau += a
        th = tau / (1 + cost * k)
        curve.append(th)
    best_k = max(range(len(curve)), key=lambda k: curve[k])
    ell = {r: 0 for r in DSPARK_REQ}
    for a, r, j in surv[:best_k]:
        ell[r] = max(ell[r], j)
    return surv, curve, best_k, ell


def fig_fixed_vs_dspark():
    f = Fig("32-fixed-vs-dspark", 1560, 640)
    f.title(40, 44, "verify에 무엇이 들어가나 — 고정 K는 요청마다 같은 수, DSpark는 요청마다 자기 초안의 앞부분만")
    _, _, _, ell = _dspark_plan(0.10)
    W, H, G = 62, 38, 8

    # 왼쪽: 고정 K
    lx = 40
    f.text(lx, 100, "MTP · EAGLE-3 일렬 (γ = 3)", 18, "start", 700)
    f.text(lx, 124, "초안을 전부 넣는다. 요청마다 같은 수 — 다른 요청과 상관없다", 14, "start", color=MUTED)
    for i, r in enumerate(DSPARK_REQ):
        y = 160 + i * 96
        f.text(lx, y + 25, f"요청 {r}", 16, "start", 700)
        pend(f, lx + 80, y, W, H, "직전", 13)
        for j in range(3):
            f.box(lx + 80 + (j + 1) * (W + G), y, W, H, "d", f"초안{j + 1}", fs=13)
        f.rect(lx + 72, y - 8, 4 * W + 3 * G + 16, H + 16, "none", KINDS["t"][1], 8, 2, "6 4")
        f.text(lx + 80 + 4 * (W + G) + 18, y + 25, "verify 4줄", 15, "start", 700, KINDS["t"][1])
    f.text(lx, 470, "합계 = 요청 수 × 4 = 12줄", 16, "start", 700)
    f.text(lx, 496, "배치가 커져도 요청당 4줄은 그대로 (정책이 γ를 바꾸면 모두 같이 바뀐다)", 13, "start", color=MUTED)

    # 오른쪽: DSpark
    rx = 700
    f.line(rx - 30, 90, rx - 30, 520, "#e3e5ee", 2, "6 6")
    f.text(rx, 100, "DSpark (한가할 때)", 18, "start", 700)
    f.text(rx, 124, "초안은 5칸을 다 뽑지만, verify에는 앞 ℓ칸만 — 숫자는 '여기까지 다 맞을 확률'", 14, "start", color=MUTED)
    for i, (r, cs) in enumerate(DSPARK_REQ.items()):
        y = 160 + i * 96
        f.text(rx, y + 25, f"요청 {r}", 16, "start", 700)
        pend(f, rx + 80, y, W, H, "직전", 13)
        a = 1.0
        for j, c in enumerate(cs):
            a *= c
            x = rx + 80 + (j + 1) * (W + G)
            if j < ell[r]:
                f.box(x, y, W, H, "d", f"{a:.2f}", fs=14, mono=True)
            else:
                f.rect(x, y, W, H, "#f4f5f9", "#c9cddb", 6, 1.5, "4 3")
                f.text(x + W / 2, y + H / 2 + 5, f"{a:.2f}", 14, color="#b0b4c3", mono=True)
        cut = rx + 80 + (ell[r] + 1) * (W + G) - G / 2
        f.line(cut, y - 10, cut, y + H + 10, KINDS["no"][1], 2.5)
        f.rect(rx + 72, y - 8, (ell[r] + 1) * (W + G) - G + 16, H + 16, "none", KINDS["t"][1], 8, 2, "6 4")
        f.text(rx + 80 + 6 * (W + G) + 6, y + 25, f"verify {ell[r] + 1}줄", 15, "start", 700, KINDS["t"][1])
    total = sum(ell[r] + 1 for r in DSPARK_REQ)
    f.text(rx, 470, f"합계 {total}줄 — 배치 전체의 줄 예산을 '맞을 확률이 높은 칸'부터 나눈다", 16, "start", 700)
    f.text(rx, 496, "점선 칸은 검증하지 않고 버린다. 다른 요청은 ℓ을 정할 때만 영향을 준다", 13, "start", color=MUTED)

    summary(f, 40, 540, 1480, "요청 안에서는 둘 다 '자기 초안의 앞부분'을 그대로 넣는다. 다른 점은 몇 칸을 넣느냐를 누가 정하나 — 설정값(고정) vs 스케줄러(배치 전체를 보고)",
            "분홍 선 = 이번 스텝의 검증 길이 ℓ. 바쁠 때는 줄 예산이 줄어 A 2 · B 0 · C 3칸으로 짧아진다 (그림 27)")
    return f


def fig_dspark_schedule():
    f = Fig("27-dspark-schedule", 1560, 760)
    f.title(40, 44, "DSpark의 검증 길이 — 요청마다 · 매 스텝, 부하를 보고 정한다 (요청 3개 예시)")
    surv, light, kl, ell_l = _dspark_plan(0.10)
    _, busy, kb, ell_b = _dspark_plan(0.20)

    # ① 요청별 생존 확률
    f.text(40, 100, "① 칸마다 '여기까지 다 통과할 확률'", 16, "start", 700)
    f.text(40, 124, "confidence를 앞에서부터 곱한 값", 13, "start", color=MUTED)
    for i, (r, cs) in enumerate(DSPARK_REQ.items()):
        y = 160 + i * 74
        f.text(40, y + 26, f"요청 {r}", 16, "start", 700)
        a = 1.0
        for j, c in enumerate(cs):
            a *= c
            x = 120 + j * 66
            shade = int(235 - a * 160)
            f.rect(x, y, 58, 38, f"rgb({shade},{shade + 8},{min(255, shade + 30)})", KINDS["d"][1], 6, 1.5)
            f.text(x + 29, y + 25, f"{a:.2f}", 14, weight=700, color="#ffffff" if a > .55 else INK, mono=True)
    f.text(40, 400, "C는 끝까지 잘 맞고, B는 첫 칸부터 불확실하다", 13, "start", color=MUTED)

    # ② 한 줄로 세우기
    bx = 500
    f.line(bx - 20, 86, bx - 20, 560, "#e3e5ee", 2, "6 6")
    f.text(bx, 100, "② 모든 요청의 칸을 한 줄로 세운다", 16, "start", 700)
    for i, (a, r, j) in enumerate(surv[:12]):
        yy = 122 + i * 34
        f.text(bx, yy + 18, f"{r}{j}", 14, "start", 700, mono=True)
        f.rect(bx + 40, yy + 4, a * 280, 18, SOLID["d"] if i < kl else "#d6d9e3", "none", 3)
        f.text(bx + 46 + a * 280, yy + 18, f"{a:.2f}", 12, "start", color=MUTED, mono=True)
    for k, lab, col in ((kl, "한가할 때 여기까지", KINDS["ok"][1]), (kb, "바쁠 때 여기까지", KINDS["no"][1])):
        yy = 122 + k * 34 - 6
        f.line(bx, yy, bx + 360, yy, col, 2.5, "6 4")
        f.text(bx + 362, yy + 5, lab, 13, "start", 700, col)

    # ③ Θ 곡선
    cx, cy, cw, chh = 1010, 130, 480, 300
    f.line(cx - 20, 86, cx - 20, 560, "#e3e5ee", 2, "6 6")
    f.text(cx - 6, 100, "③ Θ = τ · SPS(B)가 가장 클 때까지 넣는다", 16, "start", 700)
    ymin, ymax = 2.5, 6.0

    def PX(k):
        return cx + 30 + k * (cw - 50) / 12

    def PY(v):
        return cy + chh - (v - ymin) / (ymax - ymin) * chh

    for v in (3, 4, 5, 6):
        f.line(cx + 30, PY(v), cx + cw - 20, PY(v), GRID, 1)
        f.text(cx + 22, PY(v) + 5, str(v), 12, "end", color=MUTED, mono=True)
    f.line(cx + 30, cy + chh, cx + cw - 20, cy + chh, "#8b90a3", 1.5)
    for k in (0, 3, 6, 9, 12):
        f.text(PX(k), cy + chh + 20, str(k), 12, color=MUTED, mono=True)
    f.text(cx + 250, cy + chh + 44, "넣은 칸 수", 13, color=MUTED)
    for curve, kk, col, name in ((light, kl, SERIES[0], "한가할 때"), (busy, kb, SERIES[1], "바쁠 때")):
        f.poly([(PX(k), PY(v)) for k, v in enumerate(curve[:13])], col, 2.5)
        f.dot(PX(kk), PY(curve[kk]), 6, col)
        f.text(PX(12) + 6, PY(curve[12]) + 5, name, 13, "start", 700)
    f.text(cx + 30, cy + chh + 70, "τ: 기대 확정 토큰 수 · SPS(B): 줄 수 B에서 초당 스텝 수 (가정한 비용 곡선)", 12, "start", color=MUTED)

    # 결과
    def res(ell):
        return " · ".join(f"{r} {ell[r] + 1}줄" for r in DSPARK_REQ) + f"  = {sum(ell[r] + 1 for r in DSPARK_REQ)}줄"
    f.rect(40, 590, 1480, 150, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(64, 624, "검증할 줄 (K = 고른 칸 + 1)", 15, "start", 700)
    f.text(320, 624, "한가할 때   " + res(ell_l), 15, "start", 700, KINDS["ok"][1])
    f.text(320, 652, "바쁠 때     " + res(ell_b), 15, "start", 700, KINDS["no"][1])
    f.text(64, 690, "검증 길이가 요청마다 다르고, 부하가 오르면 짧아진다. 통과할 확률이 낮은 칸은 아예 검증에 넣지 않는다.", 17, "start", 700)
    f.text(64, 718, "서비스에서는 CUDA graph 때문에 총 줄 수를 두 스텝 전의 confidence로 미리 정하고, 지금의 confidence로 순서만 정한다.", 14, "start",
           color=MUTED)
    return f


# ---------------------------------------------------------------- 30. e2e 시뮬레이터 루프에 넣을 것
def fig_sim_loop():
    f = Fig("30-sim-loop", 1560, 800)
    f.title(40, 44, "e2e 서빙 시뮬레이터의 한 스텝 — speculation이 더하는 것 (파랑)")
    W, H, Y0 = 340, 340, 92
    boxes = [
        (40, "① 워크로드", ["LMBenchmark 세션 · InferenceX", "closed loop · JSONL trace"],
         ["요청마다 채택 프로필", "  (작업 종류 · 온도 · 배율)", "세션의 뒤 턴 = 긴 문맥", "  → α가 S에 따라 변한다",
          "closed loop: 빨라지면", "  다음 요청도 빨리 온다"]),
        (420, "② 스케줄러 (Python)", ["continuous batching · 토큰 예산", "KV 블록 · 선점 · prefix cache · P/D"],
         ["γ 정책 → 요청마다 K", "토큰 예산에서 Σ K를 먼저 뗀다", "KV: S + K줄 예약 · 버린 줄의 블록", "prefix 적중: drafter 상태 복원",
          "P/D: drafter 상태도 넘긴다", "drafter 가중치 · 캐시 · 버퍼"]),
        (800, "③ STAGE — ET 생성", ["배치의 연산을 그래프로"],
         ["drafter 문맥 갱신 (직전 a + 1줄)", "초안 op (스텝마다 직렬)", "verify Σ K줄 · 트리는 조상 마스크", "MoE 라우팅에 Σ K줄 (버릴 줄 포함)",
          "로짓 K × 어휘 · 거부 샘플링", "graph 크기: 요청당 1 + γ줄 단위"]),
        (1180, "④ ASTRA-sim", ["ET → 스텝 시간 (연산 + 통신)"],
         ["drafter 갱신 → 초안 → verify", "  직렬 의존 그대로", "drafter TP는 따로", "채택 처리의 host 직렬 몫"]),
    ]
    for x, name, subs, adds in boxes:
        f.rect(x, Y0, W, H, "#ffffff", "#c9cddb", 10, 1.8)
        f.text(x + 20, Y0 + 34, name, 19, "start", 700)
        for i, s in enumerate(subs):
            f.text(x + 20, Y0 + 60 + i * 20, s, 13, "start", color=MUTED)
        f.rect(x + 14, Y0 + 106, W - 28, H - 120, KINDS["d"][0], "none", 8)
        for i, s in enumerate(adds):
            f.text(x + 30, Y0 + 138 + i * 28, ("+ " if not s.startswith(" ") else "") + s, 14, "start",
                   600 if not s.startswith(" ") else 400, KINDS["d"][1] if not s.startswith(" ") else INK)
    for x in (380, 760, 1140):
        f.arrow(x + 2, Y0 + 170, x + 38, Y0 + 170)
    for x, lab in ((380, "도착"), (760, "배치"), (1140, "ET")):
        f.text(x + 20, Y0 + 156, lab, 12, color=MUTED)

    # ⑤ 스텝 끝
    ex, ey, ew, eh = 800, 500, 720, 168
    f.rect(ex, ey, ew, eh, KINDS["ok"][0], KINDS["ok"][1], 10, 2)
    f.text(ex + 20, ey + 34, "⑤ 스텝 끝 — 채택을 뽑고 확정한다 (새로 넣는 단계)", 19, "start", 700)
    for i, s in enumerate(["요청마다 채택 수 a를 분포에서 뽑는다 (자리 · 요청 · 출력 구간에 따라)",
                           "a + 1개 확정 — EOS · max_tokens가 끼면 거기서 자른다",
                           "유효 길이 S += 출력 수 · 버린 줄은 prefix cache에 넣지 않는다",
                           "확정 토큰을 스텝 끝 시각에 한꺼번에 내보낸다"]):
        f.text(ex + 30, ey + 66 + i * 26, "• " + s, 14, "start")
    f.arrow(1350, Y0 + H + 2, 1350, ey - 2)
    f.text(1362, 474, "Δt → 시계 전진", 13, "start", color=MUTED)
    # ⑤ → ② 다음 스텝
    f.line(ex, 560, 590, 560, "#5d6377", 2.2)
    f.arrow(590, 560, 590, Y0 + H + 4)
    f.text(700, 550, "다음 스텝", 13, color=MUTED)

    # ⑥ 지표
    mx, my, mw, mh = 40, 500, 340, 168
    f.rect(mx, my, mw, mh, "#ffffff", "#c9cddb", 10, 1.8)
    f.text(mx + 20, my + 34, "⑥ 지표", 19, "start", 700)
    f.text(mx + 20, my + 60, "TTFT · TPOT · ITL · 처리량", 13, "start", color=MUTED)
    f.rect(mx + 14, my + 76, mw - 28, 78, KINDS["d"][0], "none", 8)
    f.text(mx + 30, my + 106, "+ τ · 채택 히스토그램", 14, "start", 600, KINDS["d"][1])
    f.text(mx + 30, my + 136, "+ 버린 줄 · 사용자당 속도 분포", 14, "start", 600, KINDS["d"][1])
    f.line(ex, 640, mx + mw + 4, 640, "#5d6377", 2.2, arrow=True)
    f.text(600, 630, "토큰 · 시각", 13, color=MUTED)
    f.arrow(210, my - 2, 210, Y0 + H + 4)
    f.text(222, 474, "완료 → 다음 요청 (closed loop)", 13, "start", color=MUTED)

    f.rect(40, 700, 1480, 84, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(64, 732, "결정은 ②(이번 스텝에 무엇을 넣나)와 ⑤(스텝 끝에 무엇을 확정하나)에서 일어난다. ③ · ④는 그 결과를 연산 · 통신으로 센다.", 16,
           "start", 700)
    f.text(64, 762, "e2e라서 보이는 것: 큐 대기 · K줄 예약에 따른 메모리 압박과 선점 · 토큰 예산 경쟁 · closed loop의 되먹임 · 앞뒤 요청의 간섭", 14,
           "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 31. e2e 시간 축의 dynamics (작은 예시 시뮬레이션)
E2E_REQ = {  # 도착 ms, 프롬프트, 출력 길이 L, α, 어려운 구간(출력 위치 [시작, 끝))
    "R1": (0, 300, 30, .92, [(12, 19)]), "R2": (0, 220, 16, .90, []),
    "R3": (3, 360, 24, .80, [(3, 13)]), "R4": (8, 260, 28, .85, [(16, 22)]),
    "R5": (12, 300, 22, .88, []), "R6": (20, 240, 18, .75, [(5, 9)])}
E2E = dict(rho=0.93, hard=0.45, blk=16, cap=1480, dmem=160, budget=320, seed=5)


def _e2e_sim():
    import math
    import random
    P_ = E2E
    rnd = random.Random(P_["seed"])
    U = {k: [rnd.random() for _ in range(80)] for k in E2E_REQ}      # 요청 · 출력 위치마다 미리 뽑은 난수
    st = {k: dict(arr=v[0], P=v[1], L=v[2], al=v[3], hard=v[4], state="wait", gen=0, S=0, done_pre=0, need_pre=v[1],
                  held=0, aprev=0, segs=[], pres=[], order=None, events=[]) for k, v in E2E_REQ.items()}
    nblk = (P_["cap"] - P_["dmem"]) // P_["blk"]

    def blocks(tok):
        return math.ceil(tok / P_["blk"])

    def gamma(B):                                                     # 배치 4 이상이면 초안을 줄인다
        return 3 if B <= 3 else 2

    def ishard(s, pos):
        return any(a <= pos < b for a, b in s["hard"])
    t, steps, order = 0.0, [], 0
    while any(s["state"] != "done" for s in st.values()):
        used = sum(s["held"] if s["state"] == "run" else blocks(s["need_pre"] + 4)
                   for s in st.values() if s["state"] in ("run", "pre"))
        for k, s in st.items():                                       # 입장: 프롬프트 블록이 지금 들어가야 한다
            if s["state"] == "wait" and s["arr"] <= t:
                if used + blocks(s["need_pre"] + 4) <= nblk:
                    order += 1
                    s.update(state="pre", held=0, done_pre=0, order=order)
                    s["events"].append(("adm", t))
                    used += blocks(s["need_pre"] + 4)
                else:
                    break
        run = [k for k, s in st.items() if s["state"] == "run"]
        pre = sorted([k for k, s in st.items() if s["state"] == "pre"], key=lambda k: st[k]["order"])
        if not run and not pre:
            t = min(s["arr"] for s in st.values() if s["state"] == "wait")
            continue
        K = gamma(len(run) + len(pre)) + 1
        for k in run:                                                 # S + K 블록 — 끝날 때까지 줄이지 않는다
            st[k]["held"] = max(st[k]["held"], blocks(st[k]["S"] + K))
        left, chunks = P_["budget"] - K * len(run), {}                # 토큰 예산: 검증 줄을 먼저 뗀다
        for k in pre:
            s = st[k]
            c = min(left, s["need_pre"] - s["done_pre"])
            if c <= 0:
                break
            chunks[k] = c
            left -= c
            s["held"] = max(s["held"], blocks(s["done_pre"] + c))
        preempted = []
        while sum(st[k]["held"] for k in st if st[k]["state"] in ("run", "pre")) > nblk:   # 선점: 가장 늦게 들어온 요청
            v = max([k for k in st if st[k]["state"] in ("run", "pre")], key=lambda k: st[k]["order"])
            s = st[v]
            s.update(state="wait", held=0, need_pre=s["P"] + s["gen"], done_pre=0, S=0)
            s["events"].append(("preempt", t))
            preempted.append(v)
            if v in run:
                run.remove(v)
            chunks.pop(v, None)
        g = gamma(len(run) + len(chunks))
        K = g + 1
        sumK = K * len(run)
        draft = (0.30 * g + 0.03 * sum(st[k]["aprev"] + 1 for k in run)) if run else 0.0
        prefill = sum(chunks.values()) * (0.010 + 0.002)                                  # target + drafter prefill
        verify = 3.0 + 0.10 * sumK + 0.002 * sum(st[k]["S"] for k in run)
        host = 0.4
        dt = draft + prefill + verify + host
        rec = dict(t0=t, t1=t + dt, B=len(run) + len(chunks), K=K if run else 0, sumK=sumK, chunk=sum(chunks.values()),
                   comp=(draft, prefill, verify, host), preempted=preempted)
        for k in run:
            s = st[k]
            g0 = s["gen"]
            hard_any = any(ishard(s, g0 + j) for j in range(g))
            a = 0
            while a < g:
                al = s["al"] * E2E["rho"] ** a * (E2E["hard"] if ishard(s, g0 + a) else 1.0)
                if U[k][g0 + a] >= al:
                    break
                a += 1
            n = min(a + 1, s["L"] - s["gen"])
            s["gen"] += n
            s["S"] += n
            s["aprev"] = a
            s["segs"].append((t, t + dt, n, hard_any))
            if s["gen"] >= s["L"]:
                s.update(state="done", fin=t + dt, held=0)
        for k, c in chunks.items():
            s = st[k]
            s["pres"].append((t, t + dt, c, s["need_pre"] > s["P"]))
            s["done_pre"] += c
            if s["done_pre"] >= s["need_pre"]:
                s.update(state="run", S=s["need_pre"] + 1, aprev=0)
                s["gen"] += 1
                s["held"] = max(s["held"], blocks(s["S"]))
        live = [s for s in st.values() if s["state"] in ("run", "pre")]
        rec["valid"] = sum(s["S"] if s["state"] == "run" else s["done_pre"] for s in live)
        rec["held"] = sum(s["held"] for s in live) * P_["blk"]
        steps.append(rec)
        t += dt
    return st, steps


def fig_e2e_timeline(name="31-e2e-timeline", title=None, note_keys=None, note_text=None):
    st, steps = _e2e_sim()
    keys = note_keys or ["r2fin", "prefill", "hard", "preempt", "kchg", "strip"]
    f = Fig(name, 1560, 1190 - (6 - len(keys)) * 32)
    f.title(40, 44, title or "e2e 시간 축 — 채택이 스텝마다 달라지면, 끝나는 시점 · 배치 · K · KV · 선점이 함께 바뀐다")
    T = steps[-1]["t1"]
    x0, x1 = 170, 1500

    def X(t):
        return x0 + t / T * (x1 - x0)
    GREEN = {1: "#e4f0e4", 2: "#b5d8b8", 3: "#7dbb84", 4: "#4d9a56"}
    DRAFTC, HOSTC = "#9cc3ee", "#b9bdcc"

    # 범례
    lx = 760
    for n in (1, 2, 3, 4):
        f.rect(lx + (n - 1) * 40, 80, 30, 18, GREEN[n], KINDS["ok"][1], 3, 1)
        f.text(lx + (n - 1) * 40 + 15, 94, str(n), 12, weight=700, color="#ffffff" if n >= 3 else INK, mono=True)
    f.text(lx + 166, 94, "확정 토큰 수 n", 12, "start", color=MUTED)
    f.line(lx + 270, 90, lx + 300, 90, KINDS["no"][1], 3)
    f.text(lx + 306, 94, "어려운 구간", 12, "start", color=MUTED)
    f.rect(lx + 390, 80, 30, 18, KINDS["t"][0], KINDS["t"][1], 3, 1.2)
    f.text(lx + 426, 94, "prefill (청크)", 12, "start", color=MUTED)
    f.text(lx + 530, 95, "✕", 15, "start", 700, KINDS["no"][1])
    f.text(lx + 548, 94, "선점", 12, "start", color=MUTED)

    # ① 요청별 진행
    f.text(40, 92, "① 요청마다 — 칸 하나가 스텝 하나", 16, "start", 700)
    rows = {}
    for i, (k, s) in enumerate(st.items()):
        y = 124 + i * 52
        rows[k] = y
        f.text(40, y + 21, k, 15, "start", 700)
        f.text(80, y + 21, f"α {s['al']:.2f}", 13, "start", color=MUTED, mono=True)
        f.raw(f'<path d="M{X(s["arr"]):.1f},{y + 1} l-6,-9 h12 z" fill="#5d6377"/>')
        adm = [tt for e, tt in s["events"] if e == "adm"]
        pre = [tt for e, tt in s["events"] if e == "preempt"]
        if adm[0] > s["arr"] + 0.5:
            f.line(X(s["arr"]), y + 15, X(adm[0]), y + 15, "#a3a7b8", 2, "5 4")
        for tp in pre:
            back = next(tt for tt in adm if tt > tp)
            f.line(X(tp), y + 15, X(back), y + 15, KINDS["no"][1], 2, "5 4")
            f.text(X(tp) + 2, y + 21, "✕", 16, "middle", 700, KINDS["no"][1])
            f.text((X(tp) + X(back)) / 2, y + 9, "선점 → 대기", 12, weight=700, color=KINDS["no"][1])
        for a_, b_, c, again in s["pres"]:
            f.rect(X(a_) + 1, y + 2, X(b_) - X(a_) - 2, 26, KINDS["t"][0], KINDS["t"][1], 4, 1.5)
            lab = "다시 prefill" if again else ("prefill" if X(b_) - X(a_) > 52 else "")
            f.text((X(a_) + X(b_)) / 2, y + 20, lab, 11, color=KINDS["t"][1], weight=700)
        for a_, b_, n, hard in s["segs"]:
            f.rect(X(a_) + 1, y + 2, X(b_) - X(a_) - 2, 26, GREEN[n], KINDS["ok"][1], 4, 1)
            f.text((X(a_) + X(b_)) / 2, y + 20, str(n), 13, weight=700, color="#ffffff" if n >= 3 else INK, mono=True)
            if hard:
                f.line(X(a_) + 2, y + 33, X(b_) - 2, y + 33, KINDS["no"][1], 3)
        f.text(X(s["fin"]) + 6, y + 20, "끝", 12, "start", 700, KINDS["ok"][1])

    # ② 스텝
    top2 = 470
    f.text(40, top2, f"② 스텝마다 — 막대 높이 = verify 줄 Σ K, 주황 숫자 = 같은 스텝의 prefill 토큰 (토큰 예산 {E2E['budget']}에서 Σ K를 먼저 뗀다)", 16,
           "start", 700)
    base2, hmax = top2 + 110, 80
    for r in steps:
        xa, xb = X(r["t0"]), X(r["t1"])
        h = r["sumK"] / 16 * hmax
        f.rect(xa + 1, base2 - h, xb - xa - 2, h, SOLID["d"], "none", 2)
        if r["chunk"]:
            f.text((xa + xb) / 2, base2 - h - 6, f"+{r['chunk']}", 11, weight=700, color=SOLID["t"], mono=True)
        xx = xa + 1                                                   # 스텝 시간의 구성: draft · prefill · verify · host
        for part, col in zip(r["comp"], (DRAFTC, SOLID["t"], SOLID["d"], HOSTC)):
            w = part / (r["t1"] - r["t0"]) * (xb - xa - 2)
            if w > 0:
                f.rect(xx, base2 + 6, w, 12, col, "none", 0)
            xx += w
        f.text((xa + xb) / 2, base2 + 36, str(r["B"]), 12, color=MUTED, mono=True)
        f.text((xa + xb) / 2, base2 + 54, str(r["K"]) if r["K"] else "–", 12, color=MUTED, mono=True)
    f.line(x0, base2, x1, base2, "#8b90a3", 1.2)
    f.text(x0 - 10, base2 + 16, "스텝 시간", 12, "end", color=MUTED)
    f.text(x0 - 10, base2 + 36, "배치 B", 12, "end", color=MUTED)
    f.text(x0 - 10, base2 + 54, "요청당 K", 12, "end", color=MUTED)
    lx2 = 1080
    for i, (lab, col) in enumerate((("draft", DRAFTC), ("prefill", SOLID["t"]), ("verify", SOLID["d"]), ("host", HOSTC))):
        f.rect(lx2 + i * 104, top2 + 16, 16, 12, col, "none", 2)
        f.text(lx2 + i * 104 + 22, top2 + 27, lab, 12, "start", color=MUTED)
    f.text(lx2 - 8, top2 + 27, "스텝 시간의 구성:", 12, "end", color=MUTED)

    # ③ KV
    top3 = 700
    f.text(40, top3, f"③ KV — drafter 몫 + 유효 길이 S + 블록 단위 · 보유분 (블록 {E2E['blk']}토큰, 끝날 때까지 보유)", 16, "start", 700)
    base3, h3 = top3 + 150, 125

    def Y3(v):
        return base3 - v / 1600 * h3

    def area(vals, col):
        pts = []
        for r, v in zip(steps, vals):
            pts += [(X(r["t0"]), Y3(v)), (X(r["t1"]), Y3(v))]
        f.raw('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in [(X(0), base3)] + pts + [(X(T), base3)])
              + f'" fill="{col}" stroke="none"/>')
    dm = E2E["dmem"]
    area([dm + r["held"] for r in steps], "#d9d2f3")
    area([dm + r["valid"] for r in steps], "#a493e0")
    f.rect(X(0), Y3(dm), X(T) - X(0), base3 - Y3(dm), "#c9cddb", "none", 0)
    f.text(X(2), Y3(dm / 2) + 4, "drafter 몫 (가중치 · 캐시)", 11, "start", 700, INK)
    f.line(x0, Y3(E2E["cap"]), x1, Y3(E2E["cap"]), KINDS["no"][1], 1.8, "7 5")
    f.text(x1, Y3(E2E["cap"]) - 6, f"KV 한도 {E2E['cap']:,}토큰", 12, "end", 700, KINDS["no"][1])
    f.line(x0, base3, x1, base3, "#8b90a3", 1.2)
    for v in (0, 500, 1000, 1500):
        f.text(x0 - 10, Y3(v) + 4, f"{v:,}", 11, "end", color=MUTED, mono=True)
    for i, (lab, col) in enumerate((("유효 길이 S", "#a493e0"), ("블록 단위 · 보유분", "#d9d2f3"))):
        f.rect(950 + i * 150, top3 - 12, 16, 12, col, "none", 2)
        f.text(972 + i * 150, top3 - 1, lab, 12, "start", color=MUTED)
    for tt in range(0, int(T) + 1, 20):
        f.text(X(tt), base3 + 22, f"{tt}", 12, color=MUTED, mono=True)
    f.text(x1, base3 + 40, "시뮬레이션 시각 (ms)", 12, "end", color=MUTED)

    # 사건 표시
    def mark(x, y, num):
        f.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{PURPLE}"/>')
        f.text(x, y + 5, str(num), 13, weight=700, color="#ffffff")
    r2, r3, r4, r5 = st["R2"], st["R3"], st["R4"], st["R5"]
    p4 = r4["pres"][1]
    hard3 = [sg for sg in r3["segs"] if sg[3]]
    tp = next(tt for e, tt in r5["events"] if e == "preempt")
    kchg = next(r for i, r in enumerate(steps) if i and r["K"] == 4 and steps[i - 1]["K"] == 3)
    pos = {"r2fin": (X(r2["fin"]) + 40, rows["R2"] + 15), "prefill": (X(p4[1]) + 2, rows["R4"] - 9),
           "hard": ((X(hard3[0][0]) + X(hard3[-1][1])) / 2, rows["R3"] - 9), "preempt": (X(tp) + 2, rows["R5"] - 9),
           "kchg": ((X(kchg["t0"]) + X(kchg["t1"])) / 2, base2 + 76), "strip": (x0 - 120, base2 + 12)}
    readm = [tt for e, tt in r5["events"] if e == "adm"][1]
    texts = {
        "r2fin": f"R2는 잘 맞아 {len(r2['segs'])}스텝 만에 끝난다 → KV가 비어 기다리던 R5가 들어온다",
        "prefill": "토큰 예산에서 verify 줄을 먼저 떼고 남은 만큼만 prefill한다 → R2 · R3 · R4의 prefill이 두 스텝에 나뉜다",
        "hard": "R3의 어려운 구간(밑줄)에서는 확정 수가 연달아 1 — 난이도는 출력 위치에 붙어 있어 정책을 바꿔도 같은 토큰이 어렵다",
        "preempt": f"블록은 끝날 때까지 보유하고 요청은 자란다 → KV 한도를 넘어 가장 늦게 들어온 R5가 선점된다 → {readm:.0f} ms에 프롬프트 + 생성분을 다시 prefill",
        "kchg": "배치가 3 이하로 줄면 정책이 초안을 늘린다 (γ 2 → 3, 요청당 K 3 → 4)",
        "strip": "스텝 시간 = draft(직전 채택 수에 따라) + prefill(drafter prefill 포함) + verify(Σ K · Σ S에 따라) + host",
    }
    texts.update(note_text or {})
    for i, k in enumerate(keys):
        mark(pos[k][0], pos[k][1], i + 1)
    ch = 40 + 32 * len(keys)
    f.rect(40, 902, 1480, ch + 24, "#f6f5fd", "#d9d2f3", 10, 1.5)
    for i, k in enumerate(keys):
        y = 934 + i * 32
        f.raw(f'<circle cx="74" cy="{y - 5}" r="11" fill="{PURPLE}"/>')
        f.text(74, y, str(i + 1), 13, weight=700, color="#ffffff")
        f.text(96, y, texts[k], 15, "start")
    f.text(64, 902 + ch + 8, f"설명용 예시 — γ 3 (배치 4 이상이면 2) · k번째 칸 α × {E2E['rho']}^(k−1) · 어려운 구간 α × {E2E['hard']} · "
                     "난수는 요청 · 출력 위치마다 미리 뽑아 둔 값 · 스텝 시간은 가정한 비용식", 12, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 33 ~ 40. 06의 구현 포인트 — 항목마다 "무엇인가"
def _chip(f, x, y, w, kind, s, fs=13, h=26):
    fill, stroke = KINDS[kind]
    f.rect(x, y, w, h, fill, stroke, 13, 1.5)
    f.text(x + w / 2, y + h / 2 + fs * 0.36, s, fs, weight=700, color=stroke)


def fig_sim_accept():
    f = Fig("33-sim-accept", 1560, 600)
    f.title(40, 44, "① 채택 모형 — 칸마다 '맞을 확률' α를 두고, 앞 칸부터 판정해 처음 틀린 칸에서 멈춘다")

    # 왼쪽: 판정
    f.text(40, 100, "판정 — 난수 u가 α보다 작으면 통과", 17, "start", 700)
    alpha, u = [.9, .8, .6, .5], [.31, .55, .72, None]
    verdict = [("통과", "ok"), ("통과", "ok"), ("틀림 → 멈춤", "no"), ("보지 않음", "n")]
    top, hh = 140, 220
    for j in range(4):
        x = 90 + j * 160
        f.rect(x, top, 70, hh, "#f4f5f9", "#c9cddb", 6, 1.5)
        h = alpha[j] * hh
        f.rect(x, top + hh - h, 70, h, KINDS["d"][0], KINDS["d"][1], 6, 1.5)
        f.text(x + 35, top + hh - h + 22, f"α {alpha[j]}", 14, weight=700, color=KINDS["d"][1], mono=True)
        if u[j] is not None:
            yu = top + hh - u[j] * hh
            col = KINDS["ok"][1] if u[j] < alpha[j] else KINDS["no"][1]
            f.line(x - 12, yu, x + 82, yu, col, 3)
            f.text(x + 88, yu + 5, f"u {u[j]}", 13, "start", 700, col, mono=True)
        f.text(x + 35, top + hh + 26, f"T{j + 1}", 16, weight=700, mono=True)
        lab, kind = verdict[j]
        _chip(f, x - 20, top + hh + 40, 110, kind, lab, 12)
    f.text(40, top + hh + 112, "→ 채택 a = 2, 확정 n = 3  (T1 · T2 + target이 T3 자리에 고른 토큰 1개)", 16, "start", 700)

    # 오른쪽: α 만들기
    rx = 780
    f.line(rx - 30, 86, rx - 30, 520, "#e3e5ee", 2, "6 6")
    f.text(rx, 100, "α 만들기 — 요청 기준 × 자리 감쇠 × 출력 위치 난이도", 17, "start", 700)
    # 출력 위치 띠
    f.text(rx, 134, "요청 R1의 출력 위치 (1 ~ 30)", 13, "start", color=MUTED)
    cw = 23
    for p in range(1, 31):
        x = rx + (p - 1) * cw
        hard = 13 <= p <= 19
        done = p <= 11
        slot = 12 <= p <= 14
        fill = KINDS["ok"][0] if done else ("#fbe3ef" if hard else "#ffffff")
        stroke = KINDS["d"][1] if slot else ("#e7a6c8" if hard else "#d6d9e3")
        f.rect(x, 144, cw - 3, 22, fill, stroke, 3, 2.4 if slot else 1)
    f.text(rx, 186, "초록 = 이미 생성 11개", 12, "start", color=KINDS["ok"][1])
    f.text(rx + 165, 186, "파랑 테두리 = 이번 초안 3칸", 12, "start", color=KINDS["d"][1])
    f.text(rx + 365, 186, "분홍 = 어려운 구간 13 ~ 19", 12, "start", color=KINDS["no"][1])
    # 곱셈 표
    cols = ["칸 1 (12번째)", "칸 2 (13번째)", "칸 3 (14번째)"]
    rows = [("요청 기준", ["0.92", "0.92", "0.92"], INK),
            ("× 자리 감쇠 0.93^(k−1)", ["× 1", "× 0.93", "× 0.86"], INK),
            ("× 출력 위치 난이도", ["× 1 (쉬움)", "× 0.45 (어려움)", "× 0.45 (어려움)"], KINDS["no"][1]),
            ("= 이 칸의 α", ["0.92", "0.39", "0.36"], KINDS["d"][1])]
    tx, ty = rx, 220
    for j, c in enumerate(cols):
        f.text(tx + 250 + j * 150, ty, c, 13, weight=700, color=MUTED)
    for i, (lab, vals, col) in enumerate(rows):
        y = ty + 40 + i * 44
        if i == 3:
            f.line(tx, y - 28, tx + 700, y - 28, "#c9cddb", 1.5)
        f.text(tx, y, lab, 15, "start", 700 if i == 3 else 400)
        for j, v in enumerate(vals):
            f.text(tx + 250 + j * 150, y, v, 15, weight=700 if i == 3 else 400, color=col, mono=True)
    f.text(rx, 450, "기준 α · 어려운 구간은 요청이 도착할 때 정한다. 자리 감쇠는 drafter 종류로 정한 고정 규칙이다", 13, "start", color=MUTED)

    summary(f, 40, 510, 1480, "α는 매 스텝 무작위로 바뀌지 않는다 — 정해진 확률 프로필에서 '이번에 판정할 출력 위치'의 값을 쓰고, 난수와 비교해 a가 정해진다")
    return f


def fig_sim_verify():
    f = Fig("34-sim-verify", 1560, 660)
    f.title(40, 44, "② verify 비용 — target forward 한 번에 요청마다 K줄 (decode는 1줄)")

    def stack(x, y, per):
        for r in range(3):
            for k in range(per):
                yy = y + (r * per + k) * 20
                if k == 0:
                    f.rect(x, yy, 120, 16, "#ffffff", "#7b8196", 3, 1.2, "4 3")
                else:
                    f.rect(x, yy, 120, 16, KINDS["d"][0], KINDS["d"][1], 3, 1.2)
                f.text(x + 60, yy + 12, f"R{r + 1} · {'직전 토큰' if k == 0 else f'초안 {k}'}", 10, color=INK)
        return y + 3 * per * 20
    f.text(40, 100, "decode — 요청 3개 × 1줄 = 3줄", 16, "start", 700)
    stack(40, 130, 1)
    f.box(200, 120, 100, 80, "t", "target", fs=15)
    f.arrow(164, 160, 196, 160)
    f.text(380, 100, "verify — 요청 3개 × 4줄 = 12줄", 16, "start", 700)
    stack(380, 130, 4)
    f.box(540, 190, 110, 120, "t", "target", fs=15)
    f.arrow(504, 250, 536, 250)
    f.text(595, 334, "forward 1번", 13, color=MUTED)
    notes = [("가중치 읽기", "1번 — decode와 같다"), ("곱셈 · KV 쓰기", "줄 수만큼 (× K)"), ("attention", "요청마다 K × (S + K)"),
             ("MoE", "12줄 모두 라우팅 (버릴 줄 포함)"), ("출력", "로짓 12 × 어휘 + 거부 샘플링")]
    for i, (a, b) in enumerate(notes):
        y = 420 + i * 28
        f.text(40, y, a, 14, "start", 700)
        f.text(190, y, b, 14, "start", color=MUTED)

    # 오른쪽: 배치에 따라 verify ÷ decode
    cx = 900
    f.line(cx - 50, 86, cx - 50, 560, "#e3e5ee", 2, "6 6")
    f.text(cx - 20, 100, "verify 시간 ÷ decode 시간 — 배치에 따라 (K = 4)", 17, "start", 700)
    X, Y = axes(f, cx + 40, 150, 520, 280, (0, 8), (1, 4), list(range(0, 9)), [1, 2, 3, 4],
                "배치 (요청 수, 로그 축)", "배수", xfmt=lambda v: str(2 ** v), yfmt=lambda v: f"{v}×")
    pts = []
    for i in range(81):
        b = 2 ** (i / 10)
        pts.append((X(i / 10), Y((1 + 4 * b / 64) / (1 + b / 64))))
    f.poly(pts, SERIES[1], 3)
    f.text(X(0.2), Y(1.42), "작은 배치 → 거의 같다", 14, "start", 700, KINDS["ok"][1])
    f.text(X(4.4), Y(3.75), "큰 배치 → K배에 가까워진다", 14, "start", 700, KINDS["no"][1])
    f.text(cx - 20, 520, "작은 배치는 가중치 읽기, 큰 배치는 곱셈이 스텝 시간의 대부분이다", 13, "start", color=MUTED)
    f.text(cx - 20, 542, "설명용 가정: 스텝 시간 = 가중치 읽기(고정) + 곱셈(줄 수에 비례), 줄 64개에서 둘이 같다", 12, "start", color=MUTED)
    summary(f, 40, 580, 1480, "verify는 decode의 '새 토큰 줄'을 1 → K로 바꾼 것 — 작은 배치에서는 싸고, 큰 배치 · MoE · 긴 문맥에서는 비싸진다")
    return f


def fig_sim_draft():
    f = Fig("35-sim-draft", 1560, 520)
    f.title(40, 44, "③ draft 비용 — 갱신 패스(직전 a + 1줄) + 직렬 초안, 그래서 스텝마다 바뀐다")
    x0 = 330
    rows = [("EAGLE-3 · MTP (γ = 3)", "직전 a = 0 → 갱신 1줄", 1, "seq"),
            ("EAGLE-3 · MTP (γ = 3)", "직전 a = 3 → 갱신 4줄", 4, "seq"),
            ("DSpark (블록 5칸)", "직전 a = 2 → 투영 3줄", 3, "blk")]
    for i, (name, sub, nrow, mode) in enumerate(rows):
        y = 110 + i * 100
        f.text(40, y + 22, name, 16, "start", 700)
        f.text(40, y + 44, sub, 13, "start", color=MUTED)
        w_up = 50 + 26 * nrow
        f.rect(x0, y, w_up, 50, KINDS["a"][0], KINDS["a"][1], 6, 1.5)
        f.text(x0 + w_up / 2, y + 22, "갱신" if mode == "seq" else "투영", 13, weight=700, color=KINDS["a"][1])
        f.text(x0 + w_up / 2, y + 40, f"{nrow}줄", 12, color=KINDS["a"][1], mono=True)
        x = x0 + w_up + 6
        if mode == "seq":
            for k in (1, 2, 3):
                f.box(x, y, 70, 50, "d", f"초안 {k}", fs=13)
                x += 76
        else:
            f.box(x, y, 150, 50, "d", "블록 1번 → 5칸", fs=13)
            x += 156
            f.box(x, y, 60, 50, "d", "보정", fs=12)
            x += 66
        f.rect(x + 6, y, 360, 50, KINDS["t"][0], KINDS["t"][1], 6, 1.5)
        f.text(x + 186, y + 31, "verify (target)", 14, weight=700, color=KINDS["t"][1])
    f.text(x0, 96, "draft 시간 ─────────────▶", 12, "start", color=MUTED)
    f.text(1180, 140, "draft 시간을 바꾸는 것", 15, "start", 700)
    for i, s in enumerate(["배치 B", "γ (트리는 깊이 · topk)", "직전 채택 수 a", "drafter 문맥 길이"]):
        f.text(1190, 170 + i * 26, "• " + s, 14, "start")
    f.text(1180, 300, "갱신 · 투영 패스가 첫 초안도 낸다", 12, "start", color=MUTED)
    summary(f, 40, 420, 1480, "draft 시간은 상수가 아니다 — 갱신 패스의 줄 수가 직전 스텝의 채택 수를 따라 바뀐다",
            "ET에 drafter op로 넣거나, 루프라인으로 계산해 ET 앞에 노드 하나로 넣는다 (GPU 시간으로, 커널 바닥 시간 · 통신 포함)")
    return f


def fig_sim_resource():
    f = Fig("36-sim-resource", 1560, 620)
    f.title(40, 44, "④ 자원 — 한 스텝에서 speculation이 더 쓰는 몫 (파랑)")
    # 토큰 예산
    f.text(40, 100, "토큰 예산 — 스텝당 512토큰, 배치 64 · K 4", 16, "start", 700)
    bx, bw = 40, 1200
    f.rect(bx, 120, bw * 256 / 512, 40, KINDS["d"][0], KINDS["d"][1], 4, 1.5)
    f.text(bx + bw * 128 / 512, 146, "verify Σ K = 64 × 4 = 256 (먼저 뗀다)", 14, weight=700, color=KINDS["d"][1])
    f.rect(bx + bw * 256 / 512, 120, bw * 256 / 512, 40, KINDS["t"][0], KINDS["t"][1], 4, 1.5)
    f.text(bx + bw * 384 / 512, 146, "남은 256 → prefill 청크", 14, weight=700, color=KINDS["t"][1])
    f.text(bx + bw + 16, 146, "모자라면 초안을 자른다 (vLLM)", 13, "start", color=MUTED)

    # KV 블록
    f.text(40, 214, "KV — 요청 하나의 마지막 블록들 (블록 16토큰, S = 302, K = 4)", 16, "start", 700)
    cw, y1 = 22, 240

    def blockrow(y, cells, label):
        f.text(40, y + 30, label, 13, "start", color=MUTED)
        x = 300
        for bi, blk in enumerate(cells):
            for ci, kind in enumerate(blk):
                fill = {"S": "#a493e0", "K": KINDS["d"][0], "e": "#ffffff", "x": "#f1d5e4"}[kind]
                stroke = {"S": "#7c69c9", "K": KINDS["d"][1], "e": "#d6d9e3", "x": "#e7a6c8"}[kind]
                f.rect(x + ci * cw, y, cw - 3, 34, fill, stroke, 3, 1.2)
            f.text(x + 8 * cw, y + 52, f"블록 {19 + bi}", 12, color=MUTED)
            x += 16 * cw + 18
    blk19 = ["S"] * 14 + ["K"] * 2
    blk20 = ["K"] * 2 + ["e"] * 14
    blockrow(y1, [blk19, blk20], "스텝 전 — S + K = 306줄 예약")
    blk19b = ["S"] * 15 + ["x"]
    blk20b = ["x"] * 2 + ["e"] * 14
    blockrow(y1 + 84, [blk19b, blk20b], "채택 0 → S = 303")
    f.text(1100, y1 + 22, "파랑 = 검증할 K줄", 13, "start", 700, KINDS["d"][1])
    f.text(1100, y1 + 106, "분홍 = 버린 줄 (다음 스텝이 덮어쓴다)", 13, "start", 700, KINDS["no"][1])
    f.text(1100, y1 + 130, "블록 20은 비었지만 끝날 때까지 보유 (vLLM)", 13, "start", color=MUTED)

    # GPU 메모리
    f.text(40, 452, "GPU 메모리", 16, "start", 700)
    parts = [("target 가중치", 430, "n"), ("drafter 가중치", 90, "d"), ("KV 블록 (target)", 520, "c"),
             ("drafter KV", 80, "d"), ("로짓 · 버퍼", 84, "d"), ("graph", 60, "d"), ("여유", 136, "e")]
    x = 40
    for lab, w, kind in parts:
        fill, stroke = (("#ffffff", "#d6d9e3") if kind == "e" else KINDS[kind])
        f.rect(x, 468, w, 40, fill, stroke, 3, 1.5)
        f.text(x + w / 2, 493, lab, 12, weight=700, color=stroke if kind != "n" else INK)
        x += w + 4
    summary(f, 40, 536, 1480, "speculation이 더 쓰는 것: 토큰 예산의 Σ K · KV의 K줄 예약과 보유 블록 · drafter의 가중치 · 캐시 · 버퍼",
            None)
    return f


def fig_sim_kpolicy():
    f = Fig("37-sim-kpolicy", 1560, 560)
    f.title(40, 44, "⑤ K 정책 — 몇 줄을 검증할지 무엇을 보고 정하나")
    W, H, y0 = 300, 200, 150
    xs = [60, 440, 820, 1200]
    names = [("고정", "설정값 그대로"), ("vLLM Dynamic SD", "동시 요청 수 구간별"), ("SGLang adaptive", "채택 길이의 이동 평균 + 1"),
             ("DSpark", "요청마다 · 스텝마다 다르게")]
    for (n, sub), x in zip(names, xs):
        f.text(x, 100, n, 17, "start", 700)
        f.text(x, 124, sub, 13, "start", color=MUTED)
        f.line(x, y0 + H, x + W, y0 + H, "#8b90a3", 1.5)
        f.line(x, y0, x, y0 + H, "#8b90a3", 1.5)
        f.text(x - 8, y0 + 10, "K", 13, "end", 700, color=MUTED)

    def KY(k):
        return y0 + H - k / 5 * H
    # 고정
    x = xs[0]
    f.line(x, KY(4), x + W, KY(4), SERIES[0], 3)
    f.text(x + W / 2, KY(4) - 10, "K = 4", 13, weight=700, color=SERIES[0])
    f.text(x + W, y0 + H + 20, "시간 →", 12, "end", color=MUTED)
    # Dynamic SD
    x = xs[1]
    segs = [(0, .3, 4), (.3, .6, 2), (.6, 1, 1)]
    for a, b, k in segs:
        f.line(x + a * W, KY(k), x + b * W, KY(k), SERIES[0], 3)
    for lab, p in (("1", 0), ("64", .3), ("128", .6), ("512", 1)):
        f.text(x + p * W, y0 + H + 20, lab, 12, color=MUTED, mono=True)
    f.text(x + W, y0 + H + 40, "동시 요청 수 →", 12, "end", color=MUTED)
    # adaptive
    x = xs[2]
    acc = [3.5, 3.6, 3.4, 3.5, 1.4, 1.3, 1.5, 1.2, 1.4, 3.2, 3.4, 3.5, 3.3, 3.6]
    ema, ks, e = [], [], 3.0
    for v in acc:
        e = 0.8 * e + 0.2 * v
        ema.append(e)
        ks.append(min(4, max(2, round(e) + 1)))
    n = len(acc)
    f.poly([(x + i / (n - 1) * W, KY(v)) for i, v in enumerate(acc)], "#c9cddb", 2)
    f.poly([(x + i / (n - 1) * W, KY(v)) for i, v in enumerate(ema)], SERIES[2], 2, "5 4")
    pts = []
    for i, k in enumerate(ks):
        xa = x + i / (n - 1) * W
        pts += [(xa, KY(k)), (x + min(i + 1, n - 1) / (n - 1) * W, KY(k))]
    f.poly(pts, SERIES[0], 3)
    f.text(x + 4, y0 + H + 20, "회색 = 실제 채택 길이 · 점선 = 이동 평균", 11, "start", color=MUTED)
    f.text(x + W, y0 + H + 40, "스텝 →", 12, "end", color=MUTED)
    f.text(x + 0.55 * W, KY(4.6), "한발 늦게 따라간다", 12, weight=700, color=KINDS["no"][1])
    # DSpark
    x = xs[3]
    for i, (r, k) in enumerate((("A", 3), ("B", 2), ("C", 5))):
        bx = x + 30 + i * 90
        f.rect(bx, KY(k), 56, y0 + H - KY(k), KINDS["d"][0], KINDS["d"][1], 4, 1.5)
        f.text(bx + 28, KY(k) - 8, f"K {k}", 13, weight=700, color=KINDS["d"][1])
        f.text(bx + 28, y0 + H + 20, f"요청 {r}", 12, color=MUTED)
    f.text(x + W, y0 + H + 40, "confidence + 부하 (그림 32)", 12, "end", color=MUTED)
    f.text(60, KY(4) - 10, "", 10)
    for x in xs:
        f.line(x + W, y0 + H, x + W, y0 + H, "#8b90a3", 1)
    f.blue = SERIES[0]
    f.line(60, 430, 100, 430, SERIES[0], 3)
    f.text(108, 435, "파랑 = 그 스텝의 K", 13, "start", color=MUTED)
    summary(f, 40, 460, 1480, "시뮬레이터에서는 이상적인 K가 아니라 엔진의 정책을 그대로 돌린다 — 반응이 늦는 손해까지 나온다",
            "비동기 스케줄링이면 다음 스텝을 a를 모른 채 짠다 → '다 맞았다'고 가정하고 예약 · 예산을 잡은 뒤 나중에 고친다 (vLLM)")
    return f


def fig_sim_commit():
    f = Fig("38-sim-commit", 1560, 500)
    f.title(40, 44, "⑥ 확정 처리 — 스텝 끝에 n = min(a + 1, 남은 수)개를 한꺼번에 내보낸다")
    x0 = 300

    def tok(x, y, kind):
        f.rect(x, y, 14, 14, KINDS[kind][0], KINDS[kind][1], 3, 1.2)
    # decode only
    y = 130
    f.text(40, y + 4, "speculation 없음", 16, "start", 700)
    f.text(40, y + 26, "스텝마다 1개 → 11개에 11스텝", 13, "start", color=MUTED)
    for i in range(11):
        xa = x0 + i * 62
        f.rect(xa, y - 14, 58, 30, "#f4f5f9", "#c9cddb", 4, 1)
        tok(xa + 44, y - 36, "ok")
    # speculation
    y = 260
    f.text(40, y + 4, "speculation (γ 3, 출력 11개)", 16, "start", 700)
    f.text(40, y + 26, "스텝마다 n개가 한꺼번에 → 5스텝", 13, "start", color=MUTED)
    ns = [(2, 3), (0, 1), (3, 4), (0, 1)]   # (a, n)
    xa = x0
    left = 11
    for a, n in ns:
        f.rect(xa, y - 14, 110, 30, "#f4f5f9", "#c9cddb", 4, 1)
        for j in range(n):
            tok(xa + 92, y - 36 - j * 17, "ok")
        f.text(xa + 50, y + 6, f"a {a} → n {n}", 12, weight=700, mono=True)
        left -= n
        xa += 116
    # 마지막: 남은 수에서 자른다
    f.rect(xa, y - 14, 110, 30, "#f4f5f9", "#c9cddb", 4, 1)
    for j in range(4):
        if j < left:
            tok(xa + 92, y - 36 - j * 17, "ok")
        else:
            f.rect(xa + 92, y - 36 - j * 17, 14, 14, "#ffffff", KINDS["no"][1], 3, 1.2, "3 2")
    f.text(xa + 50, y + 6, f"a 3 → n {left}", 12, weight=700, mono=True)
    f.text(xa + 120, y - 50, f"남은 수 {left}개만 — 나머지는 버린다", 13, "start", 700, KINDS["no"][1])
    f.text(xa + 120, y - 30, "(출력 길이 L에 닿아 끝)", 12, "start", color=MUTED)
    f.text(x0, y + 40, "시간 →", 12, "start", color=MUTED)
    f.text(40, 350, "스텝 끝에 같이 하는 일", 15, "start", 700)
    for i, s in enumerate(["S += n · 생성 수 += n", "a를 다음 스텝으로 — 다음 draft의 갱신 줄 수",
                           "ITL은 묶음 단위 / 토큰 단위 중 비교할 벤치마크와 같은 정의로"]):
        f.text(60, 378 + i * 26, "• " + s, 14, "start")
    summary(f, 800, 352, 720, "출력 길이 L은 워크로드 값 그대로", "출력 분포를 지키는 검증이면 EOS를 따로 뽑지 않는다")
    return f


def fig_sim_drafter_state():
    f = Fig("39-sim-drafter-state", 1560, 480)
    f.title(40, 44, "⑦ drafter 상태 — drafter의 문맥(KV · hidden)을 만들거나 옮겨야 하는 네 순간")
    W, H, y0 = 350, 290, 90
    xs = [40, 410, 780, 1150]
    heads = ["처음 들어올 때", "P/D로 나눠 돌릴 때", "prefix 캐시가 적중할 때", "선점됐다 돌아올 때"]
    for x, h in zip(xs, heads):
        f.rect(x, y0, W, H, "#ffffff", "#c9cddb", 10, 1.5)
        f.text(x + 20, y0 + 32, h, 17, "start", 700)
    # 1 처음
    x = xs[0] + 20
    f.text(x, y0 + 70, "prefill 스텝", 13, "start", color=MUTED)
    f.box(x, y0 + 82, 300, 40, "t", "target prefill", fs=14)
    f.box(x, y0 + 130, 300, 40, "d", "drafter prefill", fs=14)
    f.text(x, y0 + 210, "→ target KV와 drafter KV가", 14, "start")
    f.text(x, y0 + 234, "    함께 생긴다", 14, "start")
    # 2 P/D
    x = xs[1] + 20
    f.rect(x, y0 + 70, 130, 110, KINDS["t"][0], KINDS["t"][1], 8, 1.5)
    f.text(x + 65, y0 + 96, "P", 18, weight=700, color=KINDS["t"][1])
    f.text(x + 65, y0 + 122, "target prefill", 12)
    f.text(x + 65, y0 + 142, "+ drafter prefill", 12, weight=700, color=KINDS["d"][1])
    f.rect(x + 180, y0 + 70, 130, 110, KINDS["d"][0], KINDS["d"][1], 8, 1.5)
    f.text(x + 245, y0 + 96, "D", 18, weight=700, color=KINDS["d"][1])
    f.text(x + 245, y0 + 130, "draft · verify", 12)
    f.arrow(x + 132, y0 + 125, x + 176, y0 + 125)
    for i, s in enumerate(["보내는 것: target KV", "+ drafter KV + 마지막 hidden", "DSpark는 최근 128토큰만"]):
        f.text(x, y0 + 214 + i * 24, s, 13, "start", 700 if i == 1 else 400, KINDS["d"][1] if i == 1 else INK)
    # 3 prefix
    x = xs[2] + 20
    f.text(x, y0 + 70, "프롬프트 블록", 13, "start", color=MUTED)
    for i in range(6):
        kind = "c" if i < 5 else "t"
        f.rect(x + i * 50, y0 + 82, 44, 30, KINDS[kind][0], KINDS[kind][1], 4, 1.5)
    f.text(x + 5 * 50 + 22, y0 + 132, "다시 계산", 12, weight=700, color=KINDS["t"][1])
    f.text(x, y0 + 132, "적중", 12, "start", 700, KINDS["c"][1])
    for i, s in enumerate(["drafter KV도 적중했나?", "아니면 그 구간 drafter prefill", "vLLM EAGLE: 적중을 블록 하나 줄여", "다시 계산한다",
                           "버린 초안 줄은 등록하지 않는다"]):
        f.text(x, y0 + 176 + i * 24, s, 13, "start", 700 if i == 0 else 400)
    # 4 선점
    x = xs[3] + 20
    f.rect(x, y0 + 82, 90, 30, KINDS["ok"][0], KINDS["ok"][1], 4, 1.2)
    f.text(x + 45, y0 + 102, "decode", 12)
    f.text(x + 100, y0 + 104, "✕", 18, weight=700, color=KINDS["no"][1])
    f.line(x + 112, y0 + 97, x + 190, y0 + 97, KINDS["no"][1], 2, "5 4")
    f.text(x + 151, y0 + 88, "대기", 12, weight=700, color=KINDS["no"][1])
    f.rect(x + 196, y0 + 82, 110, 30, KINDS["t"][0], KINDS["t"][1], 4, 1.2)
    f.text(x + 251, y0 + 102, "다시 prefill", 12, weight=700, color=KINDS["t"][1])
    for i, s in enumerate(["프롬프트 + 이미 생성한 토큰을", "다시 prefill한다", "drafter prefill도 다시"]):
        f.text(x, y0 + 176 + i * 24, s, 13, "start", 700 if i == 2 else 400, KINDS["d"][1] if i == 2 else INK)
    summary(f, 40, 404, 1480, "target KV만 생각하면 drafter 쪽 비용이 빠진다 — 네 순간마다 drafter 문맥을 만들 비용과 옮길 양을 더한다")
    return f


def fig_sim_records():
    f = Fig("40-sim-records", 1560, 500)
    f.title(40, 44, "⑧ 기록과 검증 — 값을 나눠 남겨야 오차가 어디서 났는지 찾는다")
    fields = [("제안 수", "draft 비용", "d"), ("검증 수 K", "verify 비용 · 토큰 예산", "t"), ("실행 줄 수 (패딩 포함)", "verify 비용", "t"),
              ("채택 수 a", "채택 모형", "ok"), ("출력 수 n", "지표", "ok"), ("유효 길이 S", "attention 비용", "c"),
              ("할당 블록 수", "메모리 · 선점", "c")]
    f.text(40, 100, "스텝마다 요청별로 남길 값", 16, "start", 700)
    for i, (a, b, kind) in enumerate(fields):
        y = 120 + i * 44
        f.box(40, y, 230, 34, kind, a, fs=14)
        f.arrow(276, y + 17, 320, y + 17)
        f.text(330, y + 23, b, 14, "start", color=MUTED)
    rx = 640
    f.line(rx - 30, 90, rx - 30, 440, "#e3e5ee", 2, "6 6")
    f.text(rx, 100, "맞추는 순서", 16, "start", 700)
    steps = [("1  비용", "vLLM synthetic으로 채택을 고정", "→ 같은 배치 · 같은 K에서 스텝 시간 비교"),
             ("2  채택", "고정 길이 실행의 기록으로 α 곡선", "→ 길이가 바뀌는 기록은 검증한 자리만 센다"),
             ("3  e2e", "같은 워크로드를 엔진과 시뮬레이터에서", "→ TTFT · TPOT · 처리량의 분포 비교")]
    for i, (h, a, b) in enumerate(steps):
        y = 124 + i * 92
        f.rect(rx, y, 860, 76, "#f6f5fd", "#d9d2f3", 8, 1.5)
        f.text(rx + 20, y + 32, h, 17, "start", 700, PURPLE)
        f.text(rx + 130, y + 32, a, 15, "start", 700)
        f.text(rx + 130, y + 58, b, 14, "start", color=MUTED)
    f.text(40, 470, "확인", 15, "start", 700)
    checks = ["drafter 없는 설정 = 기존 결과", "α = 1 → 매 스텝 최대 채택", "α = 0 → 스텝당 1토큰", "n ≤ a + 1 ≤ K"]
    x = 100
    for c in checks:
        w = 26 + len(c) * 12
        _chip(f, x, 452, w, "ok", c, 13, 28)
        x += w + 16
    return f


# ---------------------------------------------------------------- 42 ~ 44. 구현 방안 공유 페이지
def fig_impl_structure():
    f = Fig("42-impl-structure", 1560, 530)
    f.title(40, 44, "구현할 부분 — 한 사이클은 draft → verify → accept")
    W, H, y0 = 440, 360, 90
    cols = [(40, "d", "① Draft", "초안을 만든다", "갈래마다 다르다"),
            (560, "t", "② Verify", "target이 초안을 한 번에 확인한다", "연산은 같다"),
            (1080, "ok", "③ Accept", "어디까지 받을지 정한다", "갈래마다 같다")]
    for x, kind, name, role, chip in cols:
        fill, stroke = KINDS[kind]
        f.rect(x, y0, W, H, fill, stroke, 14, 2.5)
        f.text(x + 28, y0 + 48, name, 28, "start", 700)
        f.text(x + 28, y0 + 84, role, 17, "start", color=MUTED)
        cw = 30 + len(chip) * 17
        f.rect(x + W - cw - 24, y0 + 24, cw, 34, "#ffffff", stroke, 17, 2)
        f.text(x + W - cw / 2 - 24, y0 + 47, chip, 15, weight=700, color=stroke)
    for x in (480, 1000):
        f.arrow(x + 4, y0 + H / 2, x + 76, y0 + H / 2)

    def pill(x, y, w, s, kind):
        fill, stroke = KINDS[kind]
        f.rect(x, y, w, 36, "#ffffff", stroke, 18, 2)
        f.text(x + w / 2, y + 24, s, 15, weight=700)
    # Draft: 갈래 이름만
    names = [("별도 draft 모델", 190), ("EAGLE-3", 190), ("MTP 모듈", 190), ("Gemma 4 MTP", 190),
             ("DFlash", 120), ("DSpark", 120), ("DFlash 2", 120)]
    x, y = 68, y0 + 130
    for i, (n, w) in enumerate(names):
        if i == 4:
            x, y = 68, y + 50
        elif i in (2,):
            x, y = 68, y + 50
        pill(x, y, w, n, "d")
        x += w + 14
    # Verify
    vx = 588
    f.text(vx, y0 + 136, "연산 = target forward 1번 (갈래마다 같다)", 15, "start", 700)
    f.text(vx, y0 + 172, "넣는 모양 — 갈래 · 설정마다 다르다", 14, "start", 700, MUTED)
    for i, (sx, eng) in enumerate([("일렬", "MTP · EAGLE-3 topk 1 · DFlash"), ("트리", "EAGLE-3 topk > 1"),
                                   ("요청마다 다른 길이", "DSpark")]):
        y = y0 + 186 + i * 50
        f.rect(vx, y, 384, 38, "#ffffff", KINDS["t"][1], 19, 2)
        f.text(vx + 18, y + 25, sx, 15, "start", 700)
        f.text(vx + 368, y + 25, eng, 13, "end", 700, KINDS["t"][1])
    # Accept
    ax = 1108
    f.text(ax, y0 + 136, "앞에서부터 비교하고, 처음 틀린 곳에서 멈춘다", 15, "start", 700)
    f.text(ax, y0 + 172, "채택 규칙 — 엔진(vLLM · SGLang 등) 설정으로 고른다", 14, "start", 700, MUTED)
    for i, (sx, eng) in enumerate([("토큰별 거부 샘플링 (기본)", "vLLM · SGLang"), ("block verification", "vLLM · SGLang"),
                                   ("손실 허용", "vLLM (fly)")]):
        y = y0 + 186 + i * 50
        f.rect(ax, y, 384, 38, "#ffffff", KINDS["ok"][1], 19, 2)
        f.text(ax + 18, y + 25, sx, 15, "start", 700)
        f.text(ax + 368, y + 25, eng, 13, "end", 700, KINDS["ok"][1])
    f.line(270, y0 + H + 6, 270, y0 + H + 34, "#5d6377", 2.2, "7 5")
    f.line(270, y0 + H + 34, 1300, y0 + H + 34, "#5d6377", 2.2, "7 5")
    f.line(1300, y0 + H + 34, 1300, y0 + H + 6, "#5d6377", 2.2, "7 5")
    f.raw(f'<path d="M270,{y0 + H + 4} l-7,12 h14 z" fill="#5d6377"/>')
    f.text(785, y0 + H + 58, "확정된 마지막 토큰부터 다음 사이클", 14, color=MUTED)
    return f


def fig_drafter_modules():
    f = Fig("43-drafter-modules", 1560, 790)
    f.title(40, 44, "drafter 모듈 셋 — target에 붙는 얇은 모듈, 가운데 블록이 어떻게 생겼나")
    W, y0 = 480, 90
    xs = [40, 540, 1040]
    heads = [("EAGLE-3", "한 칸씩 · 따로 학습"), ("MTP — DeepSeek-V3", "한 칸씩 · 본 학습 때 함께"), ("DSpark", "한 번에 γ칸 · 따로 학습")]
    for x, (h, sub) in zip(xs, heads):
        f.rect(x, y0, W, 600, "#fafbfd", "#d6d9e3", 12, 1.5)
        f.text(x + 24, y0 + 38, h, 21, "start", 700)
        f.text(x + 24, y0 + 64, sub, 14, "start", color=MUTED)

    def bx(x, y, w, h, kind, s, fs=14, weight=400, sw=1.6):
        fill, stroke = KINDS[kind]
        f.rect(x, y, w, h, fill, stroke, 7, sw)
        f.text(x + w / 2, y + h / 2 + fs * 0.36, s, fs, weight=weight)

    def core(x, y, w, s1, s2):
        f.rect(x - 6, y - 6, w + 12, 92, "#ffffff", PURPLE, 10, 2.5, "7 4")
        bx(x, y, w, 80, "d", "", sw=2.5)
        f.text(x + w / 2, y + 34, s1, 16, weight=700)
        f.text(x + w / 2, y + 60, s2, 13, color=MUTED)

    def down(x, y1, y2):
        f.arrow(x, y1 + 2, x, y2 - 2)

    def out(x, y, w, s, sub):
        f.text(x + w / 2, y, s, 15, weight=700)
        f.text(x + w / 2, y + 24, sub, 13, color=MUTED)

    # EAGLE-3
    x = xs[0] + 24
    bx(x, 180, 250, 44, "a", "target hidden 3개 층 · 3d", 13)
    bx(x + 266, 180, 166, 44, "t", "임베딩 (target 공유)", 13)
    down(x + 125, 224, 260)
    down(x + 349, 224, 260)
    bx(x, 260, 250, 40, "d", "선형 3d → d", 13)
    bx(x + 266, 260, 166, 40, "d", "RMSNorm", 13)
    f.arrow(x + 125, 302, x + 200, 334)
    f.arrow(x + 349, 302, x + 260, 334)
    bx(x + 60, 336, 312, 38, "d", "이어 붙이기 → 2d", 13)
    down(x + 216, 374, 420)
    core(x, 420, 432, "decoder layer × 1", "Q · K · V가 2d를 받는다 · 자기 KV")
    down(x + 216, 506, 540)
    bx(x + 60, 540, 312, 40, "d", "자기 LM head — 어휘 3.2만", 13)
    down(x + 216, 580, 612)
    out(x, 632, 432, "출력: 다음 토큰 1칸의 분포", "다음 칸은 자기 hidden을 다시 넣어 반복")

    # MTP
    x = xs[1] + 24
    bx(x, 180, 250, 44, "a", "target 마지막 층 hidden · d", 13)
    bx(x + 266, 180, 166, 44, "t", "임베딩 (target 공유)", 13)
    down(x + 125, 224, 260)
    down(x + 349, 224, 260)
    bx(x, 260, 250, 40, "d", "RMSNorm", 13)
    bx(x + 266, 260, 166, 40, "d", "RMSNorm", 13)
    f.arrow(x + 125, 302, x + 200, 334)
    f.arrow(x + 349, 302, x + 260, 334)
    bx(x + 60, 336, 312, 38, "d", "이어 붙이기 → 선형 2d → d", 13)
    down(x + 216, 374, 420)
    core(x, 420, 432, "Transformer 블록 × 1", "본체 층과 같은 구조 (V3: MLA + MoE)")
    down(x + 216, 506, 540)
    bx(x + 60, 540, 312, 40, "t", "target의 출력 head (공유) — 어휘 전체", 13)
    down(x + 216, 580, 612)
    out(x, 632, 432, "출력: 그다음 토큰 1칸의 분포", "더 뽑으려면 같은 블록을 다시 돈다")

    # DSpark
    x = xs[2] + 24
    bx(x, 180, 250, 44, "a", "target 여러 층 hidden", 13)
    bx(x + 266, 180, 166, 44, "n", "직전 토큰 + 빈칸 γ개", 13)
    down(x + 125, 224, 260)
    down(x + 349, 224, 260)
    bx(x, 260, 250, 40, "d", "섞어서 층마다 K · V로", 13)
    bx(x + 266, 260, 166, 40, "t", "임베딩 (target 공유)", 13)
    f.line(x + 125, 302, x + 125, 412, KINDS["a"][1], 2, "5 4")
    f.text(x + 8, 392, "모든 층에 넣는다", 12, "start", 700, KINDS["a"][1])
    f.arrow(x + 349, 302, x + 300, 418)
    core(x, 420, 432, "블록 drafter × 3층", "MoE · 최근 128토큰만 · γ칸을 한 번에")
    down(x + 216, 506, 540)
    bx(x, 540, 136, 40, "t", "출력 head (공유)", 12)
    bx(x + 148, 540, 136, 40, "d", "보정 head", 12)
    bx(x + 296, 540, 136, 40, "d", "confidence head", 12)
    down(x + 216, 580, 612)
    out(x, 632, 432, "출력: γ칸의 분포 + 칸마다 맞을 확률", "보정은 직전 토큰으로 왼쪽부터 · 반복하지 않는다")

    f.rect(40, 716, 1480, 56, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(64, 750, "보라 점선 = 모듈의 몸통 — EAGLE-3 · MTP는 층 1개를 초안마다 돌리고, DSpark는 3층을 한 번만 돌린다.  노랑 = target에서 받는 hidden · 주황 = target 것을 함께 씀", 14,
           "start", 700)
    return f


def fig_accept_e2e():
    return fig_e2e_timeline(
        "44-accept-e2e",
        "채택 모형이 만드는 e2e dynamics — 확정 수가 요청 · 스텝마다 달라지면, 끝나는 시점 · 배치 · K · KV가 따라 움직인다",
        ["hard", "r2fin", "kchg", "preempt"],
        {"hard": "같은 요청 안에서도 확정 수가 스텝마다 다르다 — R3는 어려운 구간(밑줄)에서 연달아 1개",
         "r2fin": "요청마다 α가 달라 끝나는 시점이 갈린다 — R2(α 0.90)는 6스텝 만에 끝나고, 비운 자리 · KV로 R5가 들어온다",
         "kchg": "배치가 바뀌면 K 정책이 초안 수를 바꾼다 → 다음 스텝의 시간과 확정 수가 또 바뀐다",
         "preempt": "요청이 자라 KV가 차면 가장 늦게 들어온 R5가 선점된다 — 언제 차는지도 확정 속도가 정한다"})


# ---------------------------------------------------------------- 45 ~ 48. 구현 방안 페이지 — 알고리즘별 · 채택 모형
def fig_eagle_intuitive():
    f = Fig("45-eagle-intuitive", 1560, 640)
    f.title(40, 44, "EAGLE-3 — 트리를 깊이마다 키우고, 예산만큼 남겨 한 번에 검증한다 (topk 2 · 깊이 3 · 예산 6)")
    # ① 트리
    f.text(40, 96, "① 깊이마다 점수 높은 노드 2개를 골라 각자 2개씩 붙인다 → 점수 순 6개만 남긴다", 16, "start", 700)
    C = [("the", "the", "on", 1, .60), ("a", "a", "on", 1, .25),
         ("t.mat", "mat", "the", 2, .42), ("t.floor", "floor", "the", 2, .12),
         ("a.mat", "mat", "a", 2, .125), ("a.big", "big", "a", 2, .075),
         ("tm.and", "and", "t.mat", 3, .21), ("tm.with", "with", "t.mat", 3, .084),
         ("am.and", "and", "a.mat", 3, .05), ("am.dot", ".", "a.mat", 3, .0375)]
    kept = {"the", "a", "t.mat", "t.floor", "a.mat", "tm.and"}
    slot = {"tm.and": 0, "tm.with": 1, "t.floor": 2, "am.and": 3, "am.dot": 4, "a.big": 5}
    Y = {k: 132 + v * 54 for k, v in slot.items()}
    Y["t.mat"] = (Y["tm.and"] + Y["tm.with"]) / 2
    Y["a.mat"] = (Y["am.and"] + Y["am.dot"]) / 2
    Y["the"] = (Y["t.mat"] + Y["t.floor"]) / 2
    Y["a"] = (Y["a.mat"] + Y["a.big"]) / 2
    Y["on"] = (Y["the"] + Y["a"]) / 2
    X = {0: 50, 1: 190, 2: 350, 3: 510}
    depth = {k: d for k, _, _, d, _ in C}
    depth["on"] = 0
    W, H = 128, 34
    for k, t, p, d, s in C:
        col = "#7b8196" if k in kept else "#c9cddb"
        f.line(X[depth[p]] + (W if p != "on" else 70), Y[p] + H / 2, X[d], Y[k] + H / 2, col, 1.5, None if k in kept else "4 3")
    pend(f, X[0], Y["on"], 70, H, "on", 15)
    for k, t, p, d, s in C:
        if k in kept:
            f.box(X[d], Y[k], W, H, "d", "")
            f.text(X[d] + 12, Y[k] + 23, t, 15, "start", 700, mono=True)
            f.text(X[d] + W - 10, Y[k] + 23, f"{s:.4f}".rstrip("0"), 12, "end", color=KINDS["d"][1], mono=True)
        else:
            f.rect(X[d], Y[k], W, H, "#f4f5f9", "#c9cddb", 6, 1.5, "4 3")
            f.text(X[d] + 12, Y[k] + 23, t, 15, "start", color="#a3a7b8", mono=True)
            f.text(X[d] + W - 10, Y[k] + 23, f"{s:.4f}".rstrip("0"), 12, "end", color="#b9bdcc", mono=True)
    for d in (1, 2, 3):
        f.text(X[d] + W / 2, 470, f"깊이 {d}", 13, weight=700, color=MUTED)
        f.text(X[d] + W / 2, 490, "drafter 1번", 12, color=MUTED)
    f.text(40, 528, "숫자 = 점수(경로의 확률 곱).  파랑 = 예산 6개에 든 노드 · 회색 점선 = 탈락", 13, "start", color=MUTED)

    # ② 트리 마스크
    mx0 = 760
    f.line(mx0 - 30, 80, mx0 - 30, 540, "#e3e5ee", 2, "6 6")
    f.text(mx0, 96, "② 7줄을 한 번에 넣고, 각 줄은 자기 조상만 본다 (트리 마스크)", 16, "start", 700)
    order = ["on", "the", "a", "t.mat", "t.floor", "a.mat", "tm.and"]
    lab = {"on": "on", "the": "the", "a": "a", "t.mat": "mat", "t.floor": "floor", "a.mat": "mat", "tm.and": "and"}
    par = {k: p for k, _, p, _, _ in C}

    def anc(k):
        out = {k}
        while k != "on":
            k = par[k]
            out.add(k)
        return out
    cs, gx, gy = 38, mx0 + 70, 150
    path = {"on", "the", "t.mat"}
    for j, k in enumerate(order):
        f.text(gx + j * cs + cs / 2, gy - 10, lab[k], 12, weight=700, color=MUTED, mono=True)
    for i, r in enumerate(order):
        f.text(gx - 8, gy + i * cs + cs / 2 + 5, lab[r], 13, "end", 700, KINDS["ok"][1] if r in ("the", "t.mat") else INK, mono=True)
        A = anc(r)
        for j, k in enumerate(order):
            on = k in A
            fill = (KINDS["ok"][0] if r in path and on else KINDS["c"][0]) if on else "#ffffff"
            f.rect(gx + j * cs, gy + i * cs, cs - 3, cs - 3, fill, "#d6d9e3", 3, 1)
    f.text(gx, gy + 7 * cs + 22, "칠한 칸 = 그 줄이 보는 토큰", 12, "start", color=MUTED)
    tx = gx + 7 * cs + 30
    for i, (s_, kind) in enumerate([("결과", "b"), ("on → the ✓ → mat ✓", "g"), ("답 '.'은 mat의 자식에 없다", "m"),
                                    ("→ 멈춤", "m"), ("확정 3 · 버림 4", "p"), ("", "m"), ("최대 확정 = 깊이 + 1", "m"),
                                    ("(노드 수가 아니다)", "m")]):
        col = {"b": INK, "g": KINDS["ok"][1], "m": MUTED, "p": KINDS["no"][1]}[kind]
        f.text(tx, gy + 14 + i * 28, s_, 15, "start", 700 if kind in ("b", "g", "p") else 400, col)
    summary(f, 40, 556, 1480, "넓게 넣어 맞힐 기회를 늘리는 대신 많이 버린다 — 검증 줄 = 노드 수 + 1, 최대 채택 = 깊이")
    return f


def fig_mtp_intuitive():
    f = Fig("46-mtp-intuitive", 1560, 620)
    f.title(40, 44, "MTP — 본체 층과 같은 블록 하나를 본체 위에 붙여, 그다음 토큰을 미리 맞힌다")
    # 본체
    sx, w = 70, 230
    f.text(sx, 96, "target (본체)", 16, "start", 700)
    stack = [("출력 head", "t", 150), ("층 61", "t", 200), ("⋮", "e", 246), ("층 2", "t", 286), ("층 1", "t", 332), ("임베딩", "t", 378)]
    for s, kind, y in stack:
        if kind == "e":
            f.text(sx + w / 2, y + 10, s, 22, weight=700, color=MUTED)
        else:
            f.box(sx, y, w, 36, kind, s, fs=14)
    f.text(sx + w / 2, 136, "▲ 출력: on (다음 토큰)", 13, weight=700, mono=True)
    f.text(sx + w / 2, 440, "입력: … The cat sat", 13, color=MUTED, mono=True)
    f.text(sx + w / 2, 464, "층 61개", 13, weight=700, color=MUTED)
    # MTP
    mx = 520
    f.text(mx, 96, "MTP 모듈 (초안 1개)", 16, "start", 700)
    f.box(mx, 150, 160, 40, "a", "h — 층 61의 출력", fs=13)
    f.box(mx + 176, 150, 150, 40, "t", "임베딩(on)", fs=13)
    f.line(sx + w + 4, 218, mx - 6, 176, KINDS["a"][1], 2, "5 4")
    f.text((sx + w + mx) / 2 + 6, 166, "본체가 on을 낼 때 쓴 hidden", 12, color=KINDS["a"][1], weight=700)
    f.arrow(mx + 80, 192, mx + 140, 224)
    f.arrow(mx + 251, 192, mx + 190, 224)
    f.rect(mx - 6, 222, 338, 76, "#ffffff", PURPLE, 10, 2.5, "7 4")
    f.box(mx, 228, 326, 64, "d", "")
    f.text(mx + 163, 256, "MTP 블록 × 1", 16, weight=700)
    f.text(mx + 163, 280, "본체 층 1개와 같은 구조 (V3: MLA + MoE)", 12, color=MUTED)
    f.arrow(mx + 163, 300, mx + 163, 324)
    f.box(mx + 40, 326, 246, 36, "t", "출력 head (본체와 공유)", fs=13)
    f.arrow(mx + 163, 364, mx + 163, 388)
    f.box(mx + 113, 390, 100, 36, "d", "the", fs=15, mono=True)
    f.text(mx + 163, 448, "초안 = 그다음 토큰", 13, weight=700, color=KINDS["d"][1])
    # 한 사이클
    rx = 960
    f.line(rx - 30, 80, rx - 30, 500, "#e3e5ee", 2, "6 6")
    f.text(rx, 96, "한 사이클 (MTP-1)", 16, "start", 700)
    f.box(rx, 130, 150, 44, "d", "draft: 블록 1번", fs=13)
    f.arrow(rx + 154, 152, rx + 186, 152)
    f.rect(rx + 190, 130, 210, 44, KINDS["t"][0], KINDS["t"][1], 6, 1.5)
    f.text(rx + 295, 157, "verify: on · the = 2줄", 13, weight=700, color=KINDS["t"][1])
    f.text(rx, 214, "결과는 둘 중 하나", 14, "start", 700)
    _chip(f, rx, 228, 230, "ok", "the 맞음 → 2개 확정", 13, 30)
    f.text(rx + 242, 249, "85 ~ 90% (DeepSeek-V3)", 13, "start", color=MUTED)
    _chip(f, rx, 268, 230, "no", "틀림 → 1개 (target 토큰)", 13, 30)
    f.text(rx, 334, "초안을 3개로 늘리면", 14, "start", 700)
    x = rx
    for i in range(3):
        f.box(x, 350, 92, 36, "d", "블록", fs=13)
        if i < 2:
            f.arrow(x + 94, 368, x + 112, 368)
        x += 116
    f.text(rx, 414, "블록을 직렬로 3번 · verify 4줄", 13, "start", color=MUTED)
    f.text(rx, 438, "뒤 칸일수록 잘 안 맞는다 (학습 때 없던 쓰임)", 13, "start", color=MUTED)
    summary(f, 40, 520, 1480, "draft 한 번 = 본체 층 1개 + head — 가볍지 않다. 더 뽑으려면 같은 블록을 직렬로 반복한다",
            "모델에 들어 있는 모듈이라 따로 학습하지 않는다")
    return f


def fig_dspark_intuitive():
    f = Fig("47-dspark-intuitive", 1560, 640)
    f.title(40, 44, "DSpark — 5칸을 한 번에 뽑고, 배치 전체에서 맞을 확률이 높은 칸만 검증에 넣는다")
    surv, curve, kl, ell = _dspark_plan(0.10)
    # ① 요청마다 5칸
    f.text(40, 96, "① 요청마다 5칸을 한 번에 + 칸마다 '여기까지 맞을 확률'", 16, "start", 700)
    for i, (r, cs) in enumerate(DSPARK_REQ.items()):
        y = 130 + i * 84
        f.text(40, y + 26, f"요청 {r}", 16, "start", 700)
        a = 1.0
        for j, c in enumerate(cs):
            a *= c
            x = 120 + j * 72
            shade = int(235 - a * 160)
            f.rect(x, y, 64, 40, f"rgb({shade},{shade + 8},{min(255, shade + 30)})", KINDS["d"][1], 6, 1.5)
            f.text(x + 32, y + 26, f"{a:.2f}", 14, weight=700, color="#ffffff" if a > .55 else INK, mono=True)
    f.text(40, 400, "진할수록 맞을 확률이 높다", 13, "start", color=MUTED)
    # ② 한 줄로
    bx = 560
    f.line(bx - 30, 80, bx - 30, 540, "#e3e5ee", 2, "6 6")
    f.text(bx, 96, "② 배치의 모든 칸을 확률 순으로 세우고, 줄 예산까지 넣는다", 16, "start", 700)
    for i, (a, r, j) in enumerate(surv[:10]):
        yy = 120 + i * 34
        sel = i < kl
        f.text(bx, yy + 19, f"{r}{j}", 14, "start", 700, INK if sel else MUTED, mono=True)
        f.rect(bx + 40, yy + 5, a * 300, 20, SOLID["d"] if sel else "#d6d9e3", "none", 3)
        f.text(bx + 46 + a * 300, yy + 19, f"{a:.2f}", 12, "start", color=MUTED, mono=True)
    ycut = 120 + kl * 34 - 4
    f.line(bx, ycut, bx + 420, ycut, KINDS["no"][1], 2.5, "6 4")
    f.text(bx + 300, ycut - 8, f"여기까지 {kl}칸", 13, "start", 700, KINDS["no"][1])
    f.text(bx, 120 + 10 * 34 + 24, f"넣은 {kl}칸 + 요청마다 직전 토큰 1줄 = {kl + 3}줄", 14, "start", 700, KINDS["no"][1])
    f.text(bx, 120 + 10 * 34 + 48, "부하가 크면 선이 올라간다 (짧아진다)", 13, "start", color=MUTED)
    # ③ 결과
    rx = 1080
    f.line(rx - 30, 80, rx - 30, 540, "#e3e5ee", 2, "6 6")
    f.text(rx, 96, "③ 요청마다 다른 길이로 verify", 16, "start", 700)
    for i, r in enumerate(DSPARK_REQ):
        y = 130 + i * 84
        f.text(rx, y + 26, f"요청 {r}", 15, "start", 700)
        pend(f, rx + 70, y + 4, 48, 32, "직전", 11)
        for j in range(5):
            x = rx + 124 + j * 54
            if j < ell[r]:
                f.box(x, y + 4, 48, 32, "t", f"{r}{j + 1}", fs=12, mono=True)
            else:
                f.rect(x, y + 4, 48, 32, "#ffffff", "#d6d9e3", 5, 1.2, "3 3")
        f.text(rx + 124 + 5 * 54 + 4, y + 26, f"{ell[r] + 1}줄", 14, "start", 700, KINDS["t"][1])
    f.text(rx, 400, f"합계 {sum(ell[r] + 1 for r in DSPARK_REQ)}줄 · 점선 칸은 검증하지 않고 버린다", 14, "start", 700)
    summary(f, 40, 556, 1480, "생성은 모두 5칸, 검증은 요청마다 다르게 — 몇 칸을 넣을지는 배치 전체를 보고 스케줄러가 정한다",
            None)
    return f


def fig_accept_intuitive():
    f = Fig("48-accept-intuitive", 1560, 560)
    f.title(40, 44, "채택 모형 — 초안 칸마다 '맞을 확률' α를 붙이고, 앞 칸부터 판정한다")
    toks = ["the", "mat", "and", "slept"]
    alpha = [.9, .8, .6, .5]
    us = [.31, .55, .72, None]
    X0, CW, BW = 330, 190, 150
    rows = [("초안", 110), ("α — 맞을 확률", 200), ("난수 u", 290), ("판정 (u < α면 통과)", 380)]
    for lab, y in rows:
        f.text(40, y + 27, lab, 16, "start", 700)
    f.text(40, 252, "앞 칸이 다 맞았을 때", 12, "start", color=MUTED)
    f.text(40, 268, "이 칸이 맞을 확률", 12, "start", color=MUTED)
    pend(f, 210, 110, 80, 40, "on", 15)
    for j in range(4):
        x = X0 + j * CW
        f.box(x, 110, BW, 40, "d", "")
        f.text(x + 14, 136, f"T{j + 1}", 14, "start", 700, KINDS["d"][1], mono=True)
        f.text(x + BW - 14, 136, toks[j], 15, "end", 700, mono=True)
        for y1, y2 in ((152, 196), (242, 286), (332, 376)):
            f.arrow(x + BW / 2, y1, x + BW / 2, y2)
        f.rect(x, 200, BW, 40, "#ffffff", KINDS["d"][1], 6, 2)
        f.text(x + BW / 2, 227, f"{alpha[j]}", 18, weight=700, color=KINDS["d"][1], mono=True)
        if us[j] is None:
            f.rect(x, 290, BW, 40, "#f4f5f9", "#c9cddb", 6, 1.5, "4 3")
            f.text(x + BW / 2, 315, "—", 15, color="#a3a7b8")
            f.rect(x, 380, BW, 40, "#f4f5f9", "#c9cddb", 6, 1.5, "4 3")
            f.text(x + BW / 2, 405, "보지 않음", 14, color="#a3a7b8")
        else:
            okk = us[j] < alpha[j]
            f.rect(x, 290, BW, 40, "#f4f5f9", "#c9cddb", 6, 1.5)
            f.text(x + BW / 2, 316, f"{us[j]}", 16, weight=700, mono=True)
            kind = "ok" if okk else "no"
            f.box(x, 380, BW, 40, kind, "")
            f.text(x + BW / 2, 406, "✓ 통과" if okk else "✗ 틀림 → 멈춤", 15, weight=700, color=KINDS[kind][1])
    # 결과
    rx = 1130
    f.line(rx - 24, 96, rx - 24, 430, "#e3e5ee", 2, "6 6")
    f.text(rx, 130, "결과", 16, "start", 700)
    f.text(rx, 164, "채택 a = 2", 18, "start", 700, KINDS["ok"][1])
    f.text(rx, 196, "확정 n = 3", 18, "start", 700, KINDS["ok"][1])
    f.text(rx, 222, "T1 · T2 + target이 T3 자리에 고른 토큰", 13, "start", color=MUTED)
    f.text(rx, 280, "같은 α라도 난수에 따라 결과가 갈린다", 14, "start", 700)
    p, s = [], 1.0
    for k in range(5):
        if k < 4:
            p.append(s * (1 - alpha[k]))
            s *= alpha[k]
        else:
            p.append(s)
    for k, v in enumerate(p):
        x = rx + k * 76
        f.rect(x, 296, 68, 52, "#ffffff", "#c9cddb", 6, 1.2)
        f.text(x + 34, 318, f"{k + 1}개", 13, weight=700)
        f.text(x + 34, 339, f"{v * 100:.0f}%", 13, color=MUTED, mono=True)
    tau = sum((k + 1) * v for k, v in enumerate(p))
    f.text(rx, 372, f"한 스텝의 확정 수 — 평균 {tau:.2f}개", 13, "start", color=MUTED)
    summary(f, 40, 456, 1480, "α는 판정에 쓰는 확률이다 — 초안 칸마다 하나씩 붙고, 난수와 비교해 통과 여부를 정한다",
            "처음 틀린 칸에서 멈추고, 그 자리에는 target이 고른 토큰이 들어간다 (다 맞으면 보너스 1개)")
    return f


def fig_accept_units():
    f = Fig("49-accept-units", 1560, 790)
    f.title(40, 44, "α를 어떻게 정하나 — 요청 · 자리 · 출력 위치 · 스텝, 단위마다 무엇이 다른가")
    LX, DX = 40, 400
    heads = [("요청마다", "요청이 도착할 때 기준 α를 하나 정한다", "(작업 · 온도 · 요청별 차이)", 96),
             ("자리마다", "한 번의 초안 안에서", "뒤 칸일수록 낮춘다", 216),
             ("출력 위치마다", "출력 중 어려운 구간에 놓인 칸은 낮춘다", "(요청이 도착할 때 구간을 정해 둔다)", 356),
             ("스텝마다", "α는 그대로 — 이번 초안 칸이 놓인", "출력 위치의 값을 꺼내 판정한다", 506)]
    for i, (h, a, b, y) in enumerate(heads):
        f.text(LX, y + 26, h, 20, "start", 700, PURPLE)
        f.text(LX, y + 54, a, 14, "start")
        f.text(LX, y + 76, b, 14, "start", color=MUTED)
        if i:
            f.line(LX, y - 12, 1520, y - 12, "#eef0f5", 1.5)
    # 요청마다
    for j, (r, task, a) in enumerate((("R1", "코드", .92), ("R2", "대화", .75), ("R3", "수학", .88))):
        x = DX + j * 300
        f.rect(x, 104, 270, 64, "#ffffff", "#c9cddb", 10, 1.5)
        f.text(x + 18, 132, f"{r} · {task} 요청", 15, "start", 700)
        f.text(x + 18, 156, "기준 α", 13, "start", color=MUTED)
        f.text(x + 250, 152, f"{a}", 22, "end", 700, KINDS["d"][1], mono=True)
    # 자리마다
    vals = [.92 * .93 ** k for k in range(4)]
    for k, v in enumerate(vals):
        x = DX + k * 230
        f.box(x, 222, 120, 30, "d", f"칸 {k + 1}", fs=13)
        f.rect(x, 260, 120, 40, "#ffffff", KINDS["d"][1], 6, 2)
        f.text(x + 60, 287, f"{v:.2f}", 18, weight=700, color=KINDS["d"][1], mono=True)
        if k < 3:
            f.arrow(x + 124, 280, x + 226, 280)
            f.text(x + 175, 272, "× 0.93", 12, weight=700, color=MUTED)
    f.text(DX, 322, "R1 기준 0.92에서 시작 — 줄어드는 비율은 drafter마다 다르다", 12, "start", color=MUTED)

    # 출력 위치 띠
    def strip(y, done, slots, label=None):
        cw = 30
        for pp in range(1, 31):
            x = DX + (pp - 1) * cw
            hard = 13 <= pp <= 19
            fill = KINDS["ok"][0] if pp <= done else ("#fbe3ef" if hard else "#ffffff")
            stroke = KINDS["d"][1] if pp in slots else ("#e7a6c8" if hard else "#d6d9e3")
            f.rect(x, y, cw - 4, 26, fill, stroke, 3, 2.6 if pp in slots else 1)
        for pp in (1, 13, 19, 30):
            f.text(DX + (pp - 1) * cw + 13, y + 44, str(pp), 11, color=MUTED, mono=True)
        if label:
            f.text(DX, y - 10, label, 14, "start", 700)
    strip(368, 0, set())
    f.text(DX + 12 * 30, 360, "어려운 구간 (13 ~ 19번째) — 이 위치의 칸은 α × 0.45", 13, "start", 700, KINDS["no"][1])
    f.text(DX, 448, "출력 토큰 위치 1 ~ 30 (R1의 출력 30개)", 12, "start", color=MUTED)
    # 스텝마다
    strip(540, 4, {5, 6, 7}, "스텝 A — 4개 생성, 이번 초안은 5 · 6 · 7번째 → α 0.92 · 0.86 · 0.80")
    strip(630, 12, {13, 14, 15}, "스텝 B — 12개 생성, 이번 초안은 13 · 14 · 15번째 (어려운 구간) → α 0.41 · 0.39 · 0.36")
    f.text(DX, 694, "초록 = 이미 생성한 토큰 · 파랑 테두리 = 이번 초안 3칸이 놓인 자리", 12, "start", color=MUTED)
    summary(f, 40, 712, 1480, "한 칸의 α = 요청의 기준 α × 자리 감쇠 × (어려운 위치면 0.45) — 매 스텝 바뀌는 것은 '어느 위치를 판정하나'뿐이다")
    return f


# ---------------------------------------------------------------- 23. 큰 구조 — 한 사이클의 네 칸
def fig_pipeline():
    f = Fig("23-pipeline", 1560, 740)
    f.title(40, 44, "speculative decoding의 큰 구조 — 한 사이클은 네 칸이다")
    cols = [(40, "d", "① draft", "초안을 만든다", "갈래마다 다르다"),
            (420, "t", "② verify", "target이 한 번에 확인한다", "거의 같다"),
            (800, "ok", "③ accept", "어디까지 받을지 정한다", "같다"),
            (1180, "c", "④ 정리", "캐시를 맞추고 다음으로", "캐시에 따라 다르다")]
    W = 340

    def chip(x, y, w, kind, s, fs=14, weight=600):
        f.rect(x, y, w, 32, KINDS[kind][0], KINDS[kind][1], 16, 1.8)
        f.text(x + w / 2, y + 21, s, fs, weight=weight)

    def card(x, y, h, kind, rows, fs=14):
        f.rect(x, y, W, h, "#ffffff", KINDS[kind][1], 10, 2)
        for i, (t, b) in enumerate(rows):
            f.text(x + W / 2, y + 30 + i * 24, t, fs, weight=700 if b else 400, color=INK if b else MUTED)

    for x, kind, name, sub, tag in cols:
        f.rect(x, 78, W, 522, KINDS[kind][0], KINDS[kind][1], 14, 2)
        f.text(x + 24, 116, name, 22, "start", 700)
        f.text(x + 24, 142, sub, 15, "start", color=MUTED)
        f.rect(x + W - 24 - len(tag) * 14 - 20, 96, len(tag) * 14 + 20, 28, "#ffffff", KINDS[kind][1], 14, 1.8)
        f.text(x + W - 24 - (len(tag) * 14 + 20) / 2, 115, tag, 13, weight=700, color=KINDS[kind][1])
    for x in (380, 760, 1140):
        f.arrow(x + 2, 380, x + 38, 380)

    # ① draft
    x = 60
    f.text(x, 186, "순차 — 1개씩 γ번", 14, "start", 700, MUTED)
    chip(x, 196, 145, "d", "별도 draft 모델")
    chip(x + 155, 196, 145, "d", "EAGLE-3")
    chip(x, 236, 145, "d", "MTP 모듈")
    chip(x + 155, 236, 145, "d", "Gemma 4 MTP")
    f.text(x, 300, "병렬 — 한 번에 γ개", 14, "start", 700, MUTED)
    chip(x, 310, 95, "d", "DFlash")
    chip(x + 102, 310, 95, "d", "DSpark")
    chip(x + 204, 310, 96, "d", "DFlash 2")
    f.text(x, 374, "모델 없음 — 베낀다", 14, "start", 700, MUTED)
    chip(x, 384, 145, "n", "n-gram")
    chip(x + 155, 384, 145, "n", "suffix")
    f.line(x, 446, x + 300, 446, "#c9d9f2", 1.5)
    f.text(x, 478, "입력 — 마지막 확정 토큰 (+ target의 hidden)", 14, "start")
    f.text(x, 504, "출력 — 초안 γ개", 14, "start", 700)
    f.text(x, 530, "예: the  mat  and  slept", 14, "start", color=MUTED, mono=True)

    # ② verify
    x = 420
    card(x + 20 - 20, 176, 112, "t", [("target forward 1번", True), ("마지막 확정 토큰 + 초안 = K줄", False),
                                      ("가중치는 한 번만 읽는다", False)])
    f.text(x + 20, 326, "검증 모양", 14, "start", 700, MUTED)
    chip(x + 20, 336, 300, "t", "일렬 — vLLM · SGLang Spec V2")
    chip(x + 20, 376, 300, "t", "트리 — SGLang EAGLE 경로 일부")
    chip(x + 20, 416, 300, "t", "요청마다 다른 길이 — DSpark")
    f.line(x + 20, 466, x + 320, 466, "#f3cdb7", 1.5)
    f.text(x + 20, 498, "출력 — 자리마다 다음 토큰의 분포 K개", 14, "start", 700)
    f.text(x + 20, 524, "예: the  mat  .  …  …", 14, "start", color=MUTED, mono=True)

    # ③ accept
    x = 800
    card(x, 176, 112, "ok", [("앞에서부터 비교한다", True), ("처음 틀린 자리에서 멈춘다", False),
                             ("틀린 자리는 target의 토큰으로", False)])
    f.text(x + 20, 326, "채택 규칙", 14, "start", 700, MUTED)
    chip(x + 20, 336, 300, "ok", "토큰별 거부 샘플링 — 기본")
    chip(x + 20, 376, 300, "ok", "block verification — SGLang")
    chip(x + 20, 416, 300, "n", "typical acceptance — 분포 보존 X")
    f.line(x + 20, 466, x + 320, 466, "#c7e2c9", 1.5)
    f.text(x + 20, 498, "출력 — 채택 a개 + target 토큰 1개", 14, "start", 700)
    f.text(x + 20, 524, "예: the  mat  +  .  →  3개 확정", 14, "start", color=MUTED, mono=True)

    # ④ 정리
    x = 1180
    for i, (t, s) in enumerate([("target KV", "K줄 중 a+1줄만 남긴다"), ("drafter 캐시", "확정된 자리를 다시 쓴다"),
                                ("혼합 모델의 선형 상태", "채택된 자리로 되돌린다"), ("다음 사이클의 입력", "마지막 토큰 → verify 맨 앞")]):
        y = 176 + i * 88
        f.rect(x + 20, y, 300, 72, "#ffffff", KINDS["c"][1], 10, 1.8)
        f.text(x + 170, y + 30, t, 15, weight=700)
        f.text(x + 170, y + 54, s, 14, color=MUTED)
    f.text(x + 20, 556, "버린 줄은 지우지 않는다 —", 14, "start", color=MUTED)
    f.text(x + 20, 580, "길이를 되돌리거나 자리를 반납한다", 14, "start", color=MUTED)

    # 되돌이
    f.raw('<path d="M1350,602 C1350,628 1350,628 1322,628 L238,628 C210,628 210,628 210,608" fill="none" '
          'stroke="#5d6377" stroke-width="2.2" stroke-dasharray="7 5" marker-end="url(#ah)"/>')
    f.text(780, 652, "다음 사이클 — 마지막 확정 토큰(예: .)부터 다시 초안을 만든다", 14, color=MUTED)
    f.rect(40, 668, 1480, 60, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(64, 705, "갈래가 갈리는 곳은 ① draft뿐이다. ②③은 거의 같고, ④는 정리할 캐시의 종류에 따라 달라진다.", 17, "start", 700)
    return f


# ---------------------------------------------------------------- 24. verify의 연산 순서
def fig_verify_ops():
    f = Fig("24-verify-ops", 1560, 700)
    f.title(40, 44, "verify forward의 연산 — 순서대로, decode와 무엇이 다른가")
    TAG = {"same": ("#eff0f6", "#8b90a3", "decode와 같다"), "k": (KINDS["t"][0], KINDS["t"][1], "K배"),
           "warn": (KINDS["no"][0], KINDS["no"][1], "주의")}

    def op(x, y, w, h, name, rows, tag):
        fill, stroke, label = TAG[tag]
        f.rect(x, y, w, h, "#ffffff", stroke, 10, 2)
        f.rect(x, y, w, 30, fill, stroke, 10, 2)
        f.rect(x + 1, y + 20, w - 2, 10, fill, "none", 0)
        f.text(x + 14, y + 21, name, 15, "start", 700)
        f.text(x + w - 12, y + 21, label, 12, "end", 700, stroke)
        for i, r in enumerate(rows):
            f.text(x + 14, y + 54 + i * 22, r, 13, "start", color=INK if i == 0 else MUTED)

    # 입력
    f.text(40, 108, "입력 K줄", 15, "start", 700)
    for i, t in enumerate(["on", "the", "mat", "and", "slept"]):
        if i == 0:
            pend(f, 40, 122 + i * 38, 110, 30, t, 14)
        else:
            f.box(40, 122 + i * 38, 110, 30, "d", t, fs=14, mono=True)
    f.text(95, 330, "확정 토큰 + 초안 4개", 13, color=MUTED)
    f.arrow(154, 213, 176, 213)
    op(180, 138, 190, 150, "임베딩", ["표에서 K줄 꺼낸다", "한 번만 · 가볍다"], "same")
    f.arrow(374, 213, 398, 213)

    # 층 블록
    bx, by, bw, bh = 400, 90, 1120, 252
    f.rect(bx, by, bw, bh, "#fbfaf6", "#d18a1c", 14, 2, "8 5")
    f.text(bx + 18, by + 28, "층마다 반복 (× N층)", 15, "start", 700, "#b87410")
    ops = [("Q · K · V 투영", ["가중치 1번 읽기", "곱셈만 K배"], "k"),
           ("KV 쓰기", ["K줄을 쓴다", "나중에 a+1줄만 남는다"], "k"),
           ("attention", ["캐시 S줄 + 새 K줄 (계단)", "곱셈 K배 · 읽기는 커널 따라", "MLA는 곱셈에 막히기 쉽다"], "warn"),
           ("O 투영", ["가중치 1번 읽기", "곱셈만 K배"], "k"),
           ("FFN · MoE", ["dense: 가중치 1번", "MoE: K줄이 부르는", "expert의 합집합을 읽는다"], "warn")]
    ow, gap = 204, 14
    for i, (n, rows, tag) in enumerate(ops):
        x = bx + 18 + i * (ow + gap)
        op(x, by + 48, ow, 150, n, rows, tag)
        if i < len(ops) - 1:
            f.arrow(x + ow + 1, by + 123, x + ow + gap - 1, by + 123)
    f.text(bx + 18, by + 228, "다섯 연산이 층마다 되풀이된다", 13, "start", color=MUTED)

    # 끝 단계 — 왼쪽에서 오른쪽으로
    y2 = 392
    f.arrow(bx + 160, by + bh + 2, bx + 160, y2 - 4)
    ends = [("hidden 꺼내기", ["drafter에 넘길 hidden", "EAGLE-3 3층 · MTP 1층 · DFlash 5층"], "k", 400),
            ("출력 head", ["로짓 K × V", "큰 어휘 · 긴 블록이면 버퍼가 크다"], "warn", 760),
            ("비교 · 거부 샘플링", ["앞에서부터 비교 → 채택 a", "작은 연산 + 호스트 작업"], "same", 1120)]
    for i, (n, rows, tag, x) in enumerate(ends):
        op(x, y2, 330, 118, n, rows, tag)
    f.arrow(732, y2 + 59, 756, y2 + 59)
    f.arrow(1092, y2 + 59, 1116, y2 + 59)
    f.text(1285, y2 + 146, "→ ③ accept로", 15, weight=700, color=KINDS["ok"][1])
    f.text(180, y2 + 50, "마지막 층의 출력", 13, "start", color=MUTED)
    f.text(180, y2 + 72, "(drafter용으로 일부 층의", 13, "start", color=MUTED)
    f.text(180, y2 + 94, " hidden도 함께 꺼낸다)", 13, "start", color=MUTED)

    # 범례와 요약
    for k, lx in (("same", 40), ("k", 200), ("warn", 290)):
        fill, stroke, label = TAG[k]
        f.rect(lx, 560, 26, 18, fill, stroke, 4, 1.8)
        f.text(lx + 34, 574, label, 14, "start", color=MUTED)
    f.rect(40, 600, 1480, 76, "#f6f5fd", "#d9d2f3", 10, 1.5)
    f.text(64, 632, "verify = decode의 연산 표에서 새 토큰 줄만 1 → K. 가중치는 한 번 읽고, 곱셈과 KV 쓰기가 K배다.", 17, "start", 700)
    f.text(64, 660, "주의할 곳은 셋 — attention (MLA는 곱셈에 막히기 쉽다) · FFN (MoE는 expert 합집합) · 출력 (로짓 K줄).", 15, "start", color=MUTED)
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

    def tw(t, size):
        w = 0
        for c in t:
            o = ord(c)
            if 0xAC00 <= o <= 0xD7A3:
                w += size * 0.95
            elif c == " ":
                w += size * 0.3
            elif c in "×·":
                w += size * 0.62
            elif c.isupper():
                w += size * 0.64
            elif c.isascii():
                w += size * 0.56
            else:
                w += size * 0.9
        return w

    def card(cx, cy, name, sub, flow, flow_col, draft, cap, mode, foot):
        f.rect(cx, cy, cw, ch, "#ffffff", "#dfe2ec", 14, 1.5)
        f.text(cx + 24, cy + 38, name, 20, "start", 700)
        f.text(cx + 24, cy + 62, sub, 14, "start", color=MUTED)
        # target — 모든 카드에서 같은 모양
        f.rect(cx + 28, cy + 92, 104, 112, KINDS["t"][0], KINDS["t"][1], 16, 2.5)
        f.text(cx + 80, cy + 154, "target", 17, weight=700)
        # 받는 것 — 화살표 한가운데에 라벨을 얹는다
        ay = cy + 148
        if flow:
            f.line(cx + 138, ay, cx + 300, ay, flow_col, 4, None, True)
            w = tw(flow, 13) + 24
            mid = cx + 214
            f.rect(mid - w / 2, ay - 13, w, 26, "#ffffff", flow_col, 13, 2)
            f.text(mid, ay + 5, flow, 13, weight=700, color=flow_col)
        # draft 모듈
        dx, dy, dw, dh = cx + 308, cy + 92, 138, 112
        if draft == "llm":
            f.rect(dx, dy, dw, dh, KINDS["d"][0], KINDS["d"][1], 16, 2.5)
            for j in range(5):
                f.rect(dx + 18, dy + 16 + j * 18, dw - 36, 10, "#ffffff", KINDS["d"][1], 3, 1.2)
            bottom = dy + dh
        elif draft == "none":
            f.rect(dx, dy + 22, dw, dh - 44, "#f4f5f9", GRAY, 14, 2, "6 5")
            f.text(dx + dw / 2, dy + dh / 2 + 6, "문자열 매칭", 15, weight=700, color="#5d6377")
            bottom = dy + dh - 22
        else:
            n, gap = draft, 6
            bh = min(30, (dh - (n - 1) * gap) / n)
            top = dy + (dh - (n * bh + (n - 1) * gap)) / 2
            for j in range(n):
                f.rect(dx, top + j * (bh + gap), dw, bh, KINDS["d"][0], KINDS["d"][1], 7, 2.2)
            bottom = top + n * bh + (n - 1) * gap
        f.text(dx + dw / 2, bottom + 22, cap, 13, color=MUTED)
        # 초안을 뽑는 방식
        y = cy + 252
        if mode == "seq":
            for i in range(4):
                f.rect(cx + 28 + i * 58, y, 38, 26, SOLID["d"], "none", 7)
                if i < 3:
                    f.line(cx + 70 + i * 58, y + 13, cx + 82 + i * 58, y + 13, KINDS["d"][1], 2, None, True)
            f.text(cx + 28 + 3 * 58 + 38 + 24, y + 19, "1개씩 · γ번 돈다", 15, "start", 700, KINDS["d"][1])
        elif mode == "par":
            f.rect(cx + 22, y - 6, 192, 38, "none", KINDS["d"][1], 10, 2)
            for i in range(4):
                f.rect(cx + 30 + i * 46, y, 38, 26, SOLID["d"], "none", 7)
            f.text(cx + 214 + 24, y + 19, "한 번에 γ개 · 1번", 15, "start", 700, KINDS["d"][1])
        else:
            for i in range(4):
                f.rect(cx + 28 + i * 46, y, 38, 26, "#c9cddb", "none", 7)
            f.text(cx + 28 + 3 * 46 + 38 + 24, y + 19, "찾아서 베낀다", 15, "start", 700, "#5d6377")
        f.text(cx + 24, cy + 312, foot, 14, "start")

    xs, ys = [40, 543, 1046], [86, 436]
    card(xs[0], ys[0], "별도 draft 모델", "옆에 둔 작은 LLM", "토큰만", GRAY, "llm", "작은 LLM 전 층 · 자기 KV", "seq",
         "문맥을 처음부터 다시 읽는다")
    card(xs[1], ys[0], "EAGLE-3", "나중에 따로 학습", "hidden × 3층", AMB, 1, "1층 + 작은 head", "seq",
         "target이 읽은 결과를 받아 1층으로 맞힌다")
    card(xs[2], ys[0], "MTP 모듈", "본 학습 때 함께 (DeepSeek · Qwen)", "hidden × 1층", AMB, 1, "본체와 같은 블록 1개", "seq",
         "EAGLE과 같은 모양 — 모델이 들고 나온다")
    card(xs[0], ys[1], "DFlash · DSpark", "나중에 따로 학습", "hidden × 5층", AMB, 5, "5층 (서비스용 DSpark는 3층)", "par",
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
    f.text(64, by + 156, "캐시 자리를 토큰 단위로 나눠 주는 경우. 버린 줄의 자리를 돌려주고 '토큰 → 자리' 표만 고친다 (SGLang)", 15,
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


FIGS = [fig_three_methods, fig_mtp_chain, fig_eagle_tree, fig_fixed_vs_dspark, fig_dspark_schedule, fig_sim_loop, fig_e2e_timeline, fig_sim_accept, fig_sim_verify, fig_sim_draft, fig_sim_resource, fig_sim_kpolicy, fig_sim_commit, fig_sim_drafter_state, fig_sim_records, fig_impl_structure, fig_drafter_modules, fig_accept_e2e, fig_eagle_intuitive, fig_mtp_intuitive, fig_dspark_intuitive, fig_accept_intuitive, fig_accept_units, fig_pipeline, fig_verify_ops, fig_anatomy, fig_time_flow, fig_shift, fig_hidden_token, fig_serial_parallel, fig_context_paths, fig_cycle_tokens,
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
