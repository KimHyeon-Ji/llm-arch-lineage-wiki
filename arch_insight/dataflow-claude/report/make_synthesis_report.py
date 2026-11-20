"""종합 보고서 그림 — 모델 5개(full 기준선 · sparse 2 · hybrid 2)를 같은 지표로 비교.

  python report/make_synthesis_report.py            그림 생성
  python report/make_synthesis_report.py print      본문 수치 출력

synth-c1-kind-map   기준 1 — 요청별 데이터가 무엇에 비례해 커지나 (모델별)
synth-c2-life-map   기준 2 — 수명별로 모델의 데이터가 어디에 있나
(synth-c3-char-map  기준 3 — make_criteria_figs.py)
synth-s1-per-req    요청 1개의 요청별 데이터: 보관 vs 스텝당 이동 (문맥 8K · 32K · 128K)
synth-s2-req-mix    요청 1개 · 128K의 요청별 데이터 구성
synth-s3-char-mix   모델 전체 보관 · 이동의 성격 구성 (조건 3개)
synth-s4-cross      역전점 — 요청별 데이터가 가중치를 넘는 요청 수 (128K)
synth-sum           정리 — 특성 → 인사이트 → HW
수치는 tools/traits.py (약식 뷰어와 같은 규칙).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

import make_report as mr  # noqa: E402
import make_traits_report as mt  # noqa: E402
import make_criteria_figs as cf  # noqa: E402
import traits as tr  # noqa: E402

W = cf.W
MODELS = cf.MODELS
TAG, NAME = cf.TAG, {"mla-v31": "V3.1 · full (MLA)", "dsa-v32": "V3.2 · DSA", "csa-v4flash": "V4-Flash · CSA · HCA",
                     "kda-k3": "Kimi K3 · KDA + MLA", "qsa-qwen38": "Qwen3.8 · GDN + QSA"}
MCOL = cf.MCOL
TS = (8192, 32768, 131072)
GRP = [("kv", "KV cache", mr.BLUE), ("idx", "색인 · 블록 키", mr.ORANGE), ("state", "state · conv", mr.GREEN),
       ("ring", "최근 창 · 압축기 state", "#6250d6")]
_M = {}


def M_(mid):
    if mid not in _M:
        _M[mid] = tr.load(mid)
    return _M[mid]


def gkey(r):
    g = r["group"]
    if "창" in g or "압축기" in g:
        return "ring"
    if "색인" in g or "블록" in g:
        return "idx"
    if "state" in g:
        return "state"
    return "kv"


def per_req(mid, T):
    """요청 1개의 요청별 데이터 (B = 1). 종류 묶음별 (보관, 읽기, 쓰기)."""
    out = {}
    for r in tr.items(M_(mid), 1, T, "model"):
        if r["life"] != "req":
            continue
        a = out.setdefault(gkey(r), [0, 0, 0])
        a[0] += r["store"]
        a[1] += r["read"]
        a[2] += r["write"]
    return out


def life_move(mid, B, T):
    return tr.by(tr.items(M_(mid), B, T, "model"), "life", "move")


def cross_move(mid, T=131072):
    """요청별 이동 > 가중치 이동이 되는 최소 요청 수 (이분 탐색, 8,192까지)."""
    def ok(B):
        d = life_move(mid, B, T)
        return d.get("req", 0) > d.get("svc", 0)
    lo, hi = 1, 8192
    if not ok(hi):
        return None
    while lo < hi:
        m = (lo + hi) // 2
        lo, hi = (lo, m) if ok(m) else (m + 1, hi)
    return lo


def cross_store(mid, T=131072):
    rows = tr.items(M_(mid), 1, T, "model")
    w = sum(r["store"] for r in rows if r["life"] == "svc")
    q = sum(r["store"] for r in rows if r["life"] == "req")
    return math.ceil(w / q)


def min_gpu(mid, B, T):
    return math.ceil(sum(r["store"] for r in tr.items(M_(mid), B, T, "model")) / 141e9)


# ------------------------------------------------------------------ 기준 1 · 2 지도 (분류만 눈에 띄게)
def fig_kind_map():
    cols = [[], [], []]
    for mid in MODELS:
        rows = cf.items_of(M_(mid))
        cols[0].append((MCOL[mid], TAG[mid], f"가중치 {cf.fsize(sum(r['store'] for r in rows if r['life'] == 'svc'))}"))
        fx = [r for r in rows if r["life"] == "req" and r["tok"] <= 0]
        tk = [r for r in rows if r["life"] == "req" and r["tok"] > 0]
        if fx:
            cols[1].append((MCOL[mid], TAG[mid], f"{' · '.join(dict.fromkeys(r['group'] for r in fx))} · 요청당 {cf.fsize(sum(r['req'] for r in fx))}"))
        if tk:
            cols[2].append((MCOL[mid], TAG[mid], f"{' · '.join(dict.fromkeys(r['group'] for r in tk))} · 토큰당 {cf.fsize(sum(r['tok'] for r in tk))}"))
    cf.kind_fig("synth-c1-kind-map", "모델 5개 — 데이터 종류", cols)


def fig_life_map():
    lanes = {"svc": [(mr.GRAY, "전 모델", "가중치 (MoE · attention · LM head · 임베딩)")],
             "req": [],
             "step": [(mr.YELLOW, "전 모델", "q · 점수 · 선택 위치 등")],
             "kern": [(None, "전 모델", "attention 점수 · 확률 (sparse는 색인 head별 점수, linear는 state 중간값도)", True)]}
    for mid in MODELS:
        rows = cf.items_of(M_(mid))
        gs = [r["group"] for r in sorted(rows, key=lambda r: -r["store"]) if r["life"] == "req"]
        lanes["req"].append((MCOL[mid], TAG[mid], " · ".join(dict.fromkeys(gs))))
    cf.life_fig("synth-c2-life-map", "모델 5개 — 데이터 수명", lanes)


# ------------------------------------------------------------------ s1 요청 1개: 보관 vs 이동
def fig_per_req():
    PX, PY, PW, PH = 150, 190, 980, 780
    lo, hi = -2.0, 1.0           # 10^-2 ~ 10^1 GB
    def sx(v):
        return PX + PW * (math.log10(v / 1e9) - lo) / (hi - lo)
    def sy(v):
        return PY + PH - PH * (math.log10(v / 1e9) - lo) / (hi - lo)
    f = mr.Fig("synth-s1-per-req", W, PY + PH + 220)
    mr.title(f, "요청 1개의 요청별 데이터 — 보관 vs 스텝당 이동 (문맥 8K · 32K · 128K)",
             "요청별 데이터 = KV · 색인 키 · state 등 요청 동안 남는 것 (모델 전체 층) · 두 축 모두 로그 · 점 크기 = 문맥 길이")
    for e in range(-2, 2):
        for m in (1, 2, 5):
            v = m * 10 ** e * 1e9
            if 10 ** lo * 1e9 <= v <= 10 ** hi * 1e9:
                f.line(sx(v), PY, sx(v), PY + PH, mr.GRID, 1)
                f.line(PX, sy(v), PX + PW, sy(v), mr.GRID, 1)
                lab = cf.fsize(v).replace(".0 ", " ")
                f.text(sx(v), PY + PH + 26, lab, 13, "middle", color=mr.MUTED)
                f.text(PX - 10, sy(v) + 5, lab, 13, "end", color=mr.MUTED)
    f.rect(PX, PY, PW, PH, "none", mr.AXIS, 0, 1.2)
    f.text(PX + PW / 2, PY + PH + 60, "보관 — 요청 1개가 HBM에 들고 있는 요청별 데이터 (용량)", 16, "middle", 700)
    f.raw(f'<text x="{PX - 92}" y="{PY + PH / 2}" font-family="{mr.FONT}" font-size="16" font-weight="700" fill="{mr.INK}" '
          f'text-anchor="middle" transform="rotate(-90 {PX - 92} {PY + PH / 2})">스텝당 이동 — 요청 1개 몫의 읽기 + 쓰기 (대역폭)</text>')
    for k, lab in ((1.0, "이동 = 보관 (보관한 만큼 매 스텝 읽음)"), (0.1, "이동 = 보관의 10%")):
        x1, x2 = 10 ** lo * 1e9, 10 ** hi * 1e9
        a, b = max(x1, 10 ** lo * 1e9 / k), min(x2, 10 ** hi * 1e9 / k)
        f.raw(f'<line x1="{sx(a):.1f}" y1="{sy(a * k):.1f}" x2="{sx(b):.1f}" y2="{sy(b * k):.1f}" stroke="{mr.AXIS}" '
              f'stroke-width="1.6" stroke-dasharray="7 6"/>')
        if k == 1.0:
            f.text(sx(b) - 8, sy(b * k) + 22, lab, 13, "end", color=mr.MUTED)
        else:
            f.text(sx(1.5e9) + 12, sy(1.5e8) + 20, lab, 13, "start", color=mr.MUTED)
    for mid in MODELS:
        pts = []
        for T in TS:
            g = per_req(mid, T)
            S = sum(v[0] for v in g.values())
            mv = sum(v[1] + v[2] for v in g.values())
            pts.append((sx(S), sy(mv), T, S, mv))
        f.raw('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y, *_ in pts) + f'" fill="none" stroke="{MCOL[mid]}" stroke-width="2.4"/>')
        for i, (x, y, T, S, mv) in enumerate(pts):
            f.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5 + 3 * i}" fill="{MCOL[mid]}" stroke="#ffffff" stroke-width="2"/>')
        x, y = pts[-1][:2]
        dx, dy = {"mla-v31": (14, 26), "kda-k3": (-14, -16), "dsa-v32": (14, 6), "qsa-qwen38": (14, 6), "csa-v4flash": (14, 6)}[mid]
        f.text(x + dx, y + dy, TAG[mid], 16, "start" if dx > 0 else "end", 800, MCOL[mid])
        x0, y0 = pts[0][:2]
        f.text(x0 - 12, y0 + (20 if mid == "qsa-qwen38" else 5), "8K", 12, "end", color=mr.MUTED)
    # 오른쪽 표 — 128K 값
    RX = PX + PW + 70
    f.text(RX, PY + 10, "문맥 128K · 요청 1개", 17, weight=800)
    hdr = [("모델", 0), ("보관", 220), ("이동", 330), ("읽는 비율", 440)]
    for t, dx in hdr:
        f.text(RX + dx, PY + 46, t, 14, weight=700, color=mr.MUTED)
    y = PY + 82
    for mid in MODELS:
        g = per_req(mid, 131072)
        S = sum(v[0] for v in g.values())
        R = sum(v[1] for v in g.values())
        mv = sum(v[1] + v[2] for v in g.values())
        f.rect(RX, y - 13, 14, 14, MCOL[mid], "none", 3)
        f.text(RX + 22, y, NAME[mid], 14, weight=700)
        f.text(RX + 220, y, cf.fsize(S), 14)
        f.text(RX + 330, y, cf.fsize(mv), 14)
        f.text(RX + 440, y, f"{R / S:.0%}", 14, weight=700)
        y += 36
    cf.lines(f, RX, y + 14, ["읽는 비율 = 스텝당 읽기 ÷ 보관", "(state의 다시 쓰기는 이동에만 들어감)"], 13)
    y += 90
    f.text(RX, y, "읽는 방법", 15, weight=800)
    cf.lines(f, RX, y + 28, cf.wrap("대각선 위 = 보관한 만큼 매 스텝 옮김 (용량 · 대역폭 함께). 대각선에서 아래로 멀수록 '보관은 크고 이동은 작은' 용량형. "
                                    "선이 오른쪽으로만 길면 문맥이 이동을 거의 안 늘림", 14, W - 40 - RX), 14, mr.INK, lh=22)
    mr.takeaway(f, PY + PH + 120, "full은 보관 = 이동 (대각선) · sparse는 이동을 10 ~ 20%대로 · linear state는 문맥이 짧아도 이동이 남음",
                "128K: V3.1 5.2 GB를 매 스텝 전부 읽음 · V3.2 보관 6.3 GB 중 1.1 GB · Q3.8 3.4 GB 중 0.27 GB · K3는 보관 4.1 GB보다 이동 4.5 GB가 큼")
    mr.save(f)


# ------------------------------------------------------------------ s2 요청 1개 · 128K 구성
def fig_req_mix():
    X1, PW = 330, 560
    X2 = X1 + PW + 160
    RH, TOP = 58, 220
    vmax = max(sum(v[0] for v in per_req(m, 131072).values()) for m in MODELS)
    scale = PW / (vmax * 1.04)
    f = mr.Fig("synth-s2-req-mix", W, TOP + len(MODELS) * RH + 200)
    mr.title(f, "요청 1개 · 문맥 128K의 요청별 데이터 구성 — 무엇이 용량을, 무엇이 대역폭을 쓰나",
             "막대 길이 = 바이트 (두 판 같은 눈금) · 모델 전체 층 합 · 이동 = 스텝당 읽기 + 쓰기")
    mr.legend(f, [(n, c) for _, n, c in GRP], 40, 132)
    for x, t in ((X1, "보관 — 요청 1개가 들고 있는 양"), (X2, "스텝당 이동 — 요청 1개 몫")):
        f.text(x, TOP - 26, t, 17, weight=700)
    y = TOP
    for mid in MODELS:
        g = per_req(mid, 131072)
        f.rect(40, y + 12, 14, 14, MCOL[mid], "none", 3)
        f.text(62, y + RH / 2 + 4, NAME[mid], 16, weight=700)
        for x0, fn in ((X1, lambda v: v[0]), (X2, lambda v: v[1] + v[2])):
            x = x0
            for k, _, c in GRP:
                if k in g and fn(g[k]) > 0:
                    w = fn(g[k]) * scale
                    if w > 0.5:
                        f.rect(x, y + 10, max(w - 2, 0.8), RH - 22, c, "none", 3)
                    x += w
            tot = sum(fn(v) for v in g.values())
            f.text(x + 10, y + RH / 2 + 5, cf.fsize(tot), 15, weight=700)
        y += RH
    mr.takeaway(f, y + 50, "sparse는 같은 KV를 들고 있으면서 읽기만 줄임 — 남는 이동은 전부 읽는 색인 · 블록 키와 state",
                "V3.2 KV 5.2 GB 중 스텝당 82 MB만 읽음 · 색인 K 1.1 GB는 전부 읽음 / Q3.8 GDN state 113 MB는 읽고 같은 양을 다시 씀")
    mr.save(f)


# ------------------------------------------------------------------ s3 모델 전체 성격 구성
CONDS = [(1, 8192, "요청 1개 · 8K"), (32, 131072, "요청 32개 · 128K"), (256, 131072, "요청 256개 · 128K")]


def fig_char_mix():
    X1, PW, TW = 330, 560, 110
    X2 = X1 + PW + TW + 30
    RH, GAP, TOP = 38, 30, 214
    keys = [k for k in tr.CHAR]
    cats = [(k, tr.CHAR[k], mt.CHAR_COL[k]) for k in keys]
    H = TOP + len(CONDS) * (len(MODELS) * RH + GAP) + 130
    f = mr.Fig("synth-s3-char-mix", W, H)
    mr.title(f, "모델 전체 — 데이터 성격별 보관 · 이동 구성 (조건 3개)",
             "decode 1스텝 · 모델 전체(전 층 + MoE + LM head) · MoE expert 읽기는 균등 라우팅 가정 기댓값")
    mr.legend(f, [(c[1], c[2]) for c in cats], 40, 132, 24)
    for x, t in ((X1, "보관 — HBM에 들고 있는 양"), (X2, "스텝당 이동 — HBM 읽기 + 쓰기")):
        f.text(x, TOP - 46, t, 17, weight=700)
        f.text(x, TOP - 22, "구성 비율 (합 = 100%)", 14, color=mr.MUTED)
        f.text(x + PW + TW, TOP - 22, "합", 14, "end", color=mr.MUTED)
    y = TOP
    for B, T, lab in CONDS:
        f.text(40, y + 26, lab, 17, weight=800)
        for mid in MODELS:
            rows = tr.items(M_(mid), B, T, "model")
            f.text(180, y + RH / 2 + 6, TAG[mid], 15, weight=700, color=MCOL[mid])
            for x0, fld in ((X1, "store"), (X2, "move")):
                val = tr.by(rows, "char", fld)
                tot = sum(val.values())
                x = x0
                for ck, _, c in cats:
                    w = PW * val.get(ck, 0) / tot
                    if w > 0.25:
                        f.rect(x, y + 5, max(w - 2, 0.7), RH - 10, c, mr.BOX[1] if c == mr.LMH else "none", 3, 1)
                        p = mt.pct(val[ck] / tot)
                        if w >= mr.text_w(p, 13) + 10:
                            f.text(x + w / 2 - 1, y + RH / 2 + 5, p, 13, "middle", 700, "#ffffff" if c in mt.DARK else mr.INK)
                    x += w
                f.text(x0 + PW + TW, y + RH / 2 + 6, mt.fb(tot), 14, "end", 700)
            y += RH
        f.line(40, y + GAP / 2 - 2, W - 40, y + GAP / 2 - 2, mr.GRID, 1)
        y += GAP
    mr.takeaway(f, y + 36, "이동의 주된 몫은 대부분 조건에서 read-only로 골라 읽는 MoE — full(V3.1)만 많은 요청 · 긴 문맥에서 KV가 넘어섬",
                "요청 256개 · 128K 이동: V3.1 KV(끝에 추가 · 전부 읽음) 66% / V3.2 골라 읽는 MoE 66% · 전부 읽는 색인 K 27% / K3 MoE 54% · MLA KV 34% · state 8.5%")
    mr.save(f)


# ------------------------------------------------------------------ s4 역전점
def fig_cross():
    PX, PW, TOP, RH = 330, 1000, 210, 70
    lo, hi = 0, math.log2(4096)
    def sx(n):
        return PX + PW * (math.log2(min(max(n, 1), 4096)) - lo) / (hi - lo)
    f = mr.Fig("synth-s4-cross", W, TOP + len(MODELS) * RH + 220)
    mr.title(f, "역전점 — 요청 몇 개부터 요청별 데이터가 가중치를 넘나 (문맥 128K)",
             "○ 보관 역전 = 요청별 보관 > 가중치 보관 · ● 이동 역전 = 요청별 이동 > 가중치 이동 · 가로축 = 요청 수 (로그)")
    for n in (1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        f.line(sx(n), TOP - 10, sx(n), TOP + len(MODELS) * RH, mr.GRID if n not in (32, 256) else "#c9ccd6", 1 if n not in (32, 256) else 2)
        f.text(sx(n), TOP + len(MODELS) * RH + 26, str(n), 13, "middle", color=mr.MUTED)
    f.text(sx(32), TOP - 18, "보고서 조건 32", 12, "middle", 700, mr.MUTED)
    f.text(sx(256), TOP - 18, "256", 12, "middle", 700, mr.MUTED)
    y = TOP
    for mid in MODELS:
        cs, cm = cross_store(mid), cross_move(mid)
        f.rect(40, y + 22, 14, 14, MCOL[mid], "none", 3)
        f.text(62, y + 35, NAME[mid], 15, weight=700)
        f.line(PX, y + 30, PX + PW, y + 30, mr.GRID, 1)
        f.raw(f'<circle cx="{sx(cs):.1f}" cy="{y + 30}" r="9" fill="#ffffff" stroke="{MCOL[mid]}" stroke-width="3"/>')
        f.text(sx(cs), y + 62, f"보관 {cs}", 13, "middle", 700, MCOL[mid])
        if cm:
            f.raw(f'<circle cx="{sx(cm):.1f}" cy="{y + 30}" r="9" fill="{MCOL[mid]}"/>')
            f.text(sx(cm), y + 14, f"이동 {cm}", 13, "middle", 700, MCOL[mid])
        y += RH
    RX = PX + PW + 50
    f.text(RX, TOP + 4, "읽는 방법", 15, weight=800)
    cf.lines(f, RX, TOP + 32, cf.wrap("○가 왼쪽일수록 요청별 데이터가 일찍 용량을 차지 · ●가 오른쪽일수록 요청이 많아도 이동은 가중치가 정함. "
                                      "○과 ● 사이가 넓으면 '용량은 막히는데 대역폭은 남는' 구간", 14, W - 40 - RX), 14, mr.INK, lh=22)
    mr.takeaway(f, y + 90, "sparse 선택은 이동 역전만 뒤로 미루고 보관 역전은 그대로 — 용량이 대역폭보다 먼저 막힘",
                "V3.1 보관 129 · 이동 126개 (거의 같이) / V3.2 보관 108 · 이동 592개 / Q3.8 보관 74 · 이동 656개 / V4는 압축으로 둘 다 뒤로")
    mr.save(f)


# ------------------------------------------------------------------ 정리
SUM_ROWS = [
    ("full KV (V3.1)", "blue", "app_all", "끝에 추가 · 전부 읽음 — 토큰당 40 KB, 요청 × 문맥에 비례",
     "보관한 만큼 매 스텝 읽음", "128K에서 요청 126 ~ 129개부터 보관 · 이동 모두 요청별 데이터가 가중치를 넘음",
     "HBM 용량 · 대역폭을 함께, KV 압축 · 저정밀", "아래 계층 메모리는 읽기 대역폭이 모자라 효과 제한", "2.5"),
    ("골라 읽는 KV (V3.2 · V4 · Q3.8)", "blue", "app_sel", "끝에 추가 · 골라 읽음 — 요청당 고정 예산(2,048 · 512)만 읽음",
     "이동은 줄지만 보관은 그대로 — 용량이 먼저 막힘", "128K 읽는 비율 8 ~ 22% · V3.2 보관은 색인 키만큼 오히려 20% 늘어남",
     "전체 KV는 큰 메모리, 고를 KV만 HBM", "선택 겹침 · 블록 단위 전송이 클수록 유리", "2.1"),
    ("전부 읽는 색인 · 블록 키", "orange", "app_all", "끝에 추가 · 전부 읽음 — 토큰당 0.7 ~ 8 KB",
     "sparse 모델의 요청별 이동 중 가장 큰 몫", "요청 256개 · 128K: V3.2 모델 이동 27% · V4 12.5% · Q3.8 7.4%",
     "메모리 근처 선택 · 색인 연산 · top-k", "키 형식 · head 수에 따라 연산 요구가 4 ~ 124 FLOP/B로 갈림", "2.2"),
    ("다시 쓰는 state (K3 · Q3.8)", "green", "rmw_all", "매 스텝 전부 다시 씀 — 요청당 113 ~ 434 MB, 문맥과 무관",
     "많은 요청 · 짧은 문맥에서도 이동이 남음", "요청 256개 · 8K: K3 이동 12.5% · Q3.8 18% — 이동 비중이 보관 비중의 약 2배",
     "메모리 안 갱신으로 왕복 제거", "이득은 갱신 연산을 메모리 쪽이 감당하는지에 달림", "2.4"),
    ("MoE 가중치 (전 모델)", "gray-l", "ro_sel", "read-only · 골라 읽음 — 요청이 늘면 대부분을 읽음",
     "sparse · hybrid에서는 다시 이동의 대부분", "요청 32개 · 128K: 요청별 데이터 이동은 전체의 4 ~ 16%뿐 (V3.1은 28%)",
     "가중치 대역폭 · 요청 수에 맞춘 배치", "attention 쪽 HW 이득의 상한 = 요청별 이동 비중", "-"),
]


def fig_sum():
    cf.SUMMARY["_synth"] = SUM_ROWS
    mt.PREFIX["_synth"] = "synth"
    rows = SUM_ROWS
    X, WS, PAD = [40, 640, 1190], [530, 480, W - 40 - 1190], 24

    def h_of(r):
        a = 64 + len(cf.wrap(r[3], 14, WS[0] - 2 * PAD)) * 22
        b = 30 + len(cf.wrap(r[4], 17, WS[1] - 2 * PAD)) * 26 + len(cf.wrap(r[5], 14, WS[1] - 2 * PAD)) * 22
        c = 64 + len(cf.wrap(r[6], 17, WS[2] - 2 * PAD)) * 26 + len(cf.wrap(r[7], 14, WS[2] - 2 * PAD)) * 22
        return max(a, b, c) + 12
    TOP = 170
    f = mr.Fig("synth-sum", W, TOP + sum(h_of(r) + 18 for r in rows) + 60)
    mr.title(f, "종합 — sparse · hybrid가 바꾼 데이터 특성 → 인사이트 → HW",
             "줄마다 데이터 성격 하나 · 숫자는 본문과 같은 계산 · 괄호 절 번호 = 기준 보고서 Ⅱ장")
    for x, t in zip(X, ("데이터 특성 (어느 모델 · 어떤 성격)", "인사이트 (비교에서 보이는 것)", "HW로 연결")):
        f.text(x, TOP - 22, t, 16, weight=700, color=mr.MUTED)
    y = TOP
    for r in rows:
        h = h_of(r)
        item, c, ck, trait, i1, i2, h1, h2, sec = r
        f.rect(X[0], y, WS[0], h, cf.BOXF, cf.LINE, 12, 1.2)
        f.rect(X[0], y, 7, h, cf.char_col(ck), "none", 3)
        cf.chip(f, X[0] + PAD, y + 36, cf.col(c), item, size=18)
        cf.lines(f, X[0] + PAD, y + 66, cf.wrap(trait, 14, WS[0] - 2 * PAD), 14, mr.MUTED, lh=22)
        f.rect(X[1], y, WS[1], h, "#ffffff", cf.LINE, 12, 1.2)
        cf.tb(f, X[1] + PAD, y + 36, WS[1] - 2 * PAD, i1, i2)
        f.rect(X[2], y, WS[2], h, "#f3f1fc", "#d9d2f5", 12, 1.2)
        tag = cf.SECS[sec]
        tw = mr.text_w(tag, 13) + 20
        f.rect(X[2] + PAD, y + 18, tw, 26, "#7b61c9" if sec not in ("-", "x") else cf.IDLE, "none", 13)
        f.text(X[2] + PAD + tw / 2, y + 36, tag, 13, "middle", 700, "#ffffff")
        cf.tb(f, X[2] + PAD, y + 72, WS[2] - 2 * PAD, h1, h2)
        cf.arrow(f, X[0] + WS[0] + 10, X[1] - 10, y + h / 2)
        cf.arrow(f, X[1] + WS[1] + 10, X[2] - 10, y + h / 2)
        y += h + 18
    mr.save(f)


def show():
    for mid in MODELS:
        print(f"== {mid}")
        for T in TS:
            g = per_req(mid, T)
            S = sum(v[0] for v in g.values())
            R = sum(v[1] for v in g.values())
            Wr = sum(v[2] for v in g.values())
            print(f"  T{T // 1024}K store {S / 1e9:.3f} read {R / 1e9:.3f} write {Wr / 1e6:.2f}MB ratio {R / S:.2f} |",
                  {k: (round(v[0] / 1e9, 3), round((v[1] + v[2]) / 1e9, 3)) for k, v in g.items()})
        print("  cross store", cross_store(mid), "move", cross_move(mid))
        print("  minG", [min_gpu(mid, B, T) for B in (1, 32, 256) for T in TS], "n_gpu", M_(mid)["model"]["deploy"]["n_gpu"])
        for B, T, _ in CONDS:
            d = life_move(mid, B, T)
            tot = sum(d.values())
            print(f"  B{B} T{T // 1024}K req move share {d.get('req', 0) / tot:.3f}")


if __name__ == "__main__":
    if sys.argv[1:] == ["print"]:
        show()
    else:
        fig_kind_map()
        fig_life_map()
        fig_per_req()
        fig_req_mix()
        fig_char_mix()
        fig_cross()
        fig_sum()
