"""종합 보고서 그림 — 모델 5개(기준선 full · sparse 2 · hybrid 2)를 모델별 보고서와 같은 틀로 비교.

  python report/make_synthesis_report.py            그림 생성
  python report/make_synthesis_report.py print      본문 수치 출력

모델별 보고서는 구성 비율(합 = 100%)로 그리지만, 종합은 **바이트 그대로** 그려 모델끼리의 크기 차이가 보이게 함.
조건은 셋만: 요청 1개 · 128K / 요청 256개 · 8K / 요청 256개 · 128K. 모델 줄은 기준선 · Sparse · Hybrid로 묶어 띠를 칠함.

synth-t1-kind-model   종류별 보관 · 이동 (모델 전체, 조건 3개)
synth-t2-kind-req     요청 1개 몫의 요청별 데이터 — 종류별 (문맥 8K · 128K, 같은 눈금)
synth-t3-life-model   수명별 보관 · 이동 (모델 전체, 조건 3개)
synth-t4-life-cross   역전점 — 요청 몇 개부터 '요청 동안' 데이터가 '서비스 내내' 가중치를 넘나 (128K)
synth-t5-char-model   성격별 보관 · 이동 (모델 전체, 조건 3개)
synth-t6-char-req     요청 1개 몫: 보관 vs 스텝당 이동 — 읽는 비율 (문맥 8K · 32K · 128K)
synth-sum             정리 — 특성 → 인사이트 → HW
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
TAG = cf.TAG
MCOL = cf.MCOL
ATT = {"mla-v31": "MLA", "dsa-v32": "DSA", "csa-v4flash": "CSA + HCA", "kda-k3": "KDA + DSA", "qsa-qwen38": "GDN + QSA"}
NAME = {m: f"{TAG[m]} · {ATT[m]}" for m in MODELS}
# 구조 묶음 — 줄 띠 · 알약
FAM = [("full", "기준선", ["mla-v31"]), ("sparse", "Sparse", ["dsa-v32", "csa-v4flash"]),
       ("hybrid", "Hybrid", ["kda-k3", "qsa-qwen38"])]
FAM_OF = {m: k for k, _, ms in FAM for m in ms}
FAM_BG = {"full": "#f3f4f7", "sparse": "#eef3f8", "hybrid": "#f6f0f4"}
FAM_ACC = {"full": "#4a5573", "sparse": "#0e7490", "hybrid": "#a23b72"}
FAM_SUB = {"full": "full attention", "sparse": "골라 읽기 · 압축", "hybrid": "linear + sparse"}
TS = (8192, 32768, 131072)
CONDS = [(1, 131072, "요청 1개 · 128K", "요청이 적을 때"),
         (256, 8192, "요청 256개 · 8K", "요청이 많고 문맥이 짧을 때"),
         (256, 131072, "요청 256개 · 128K", "요청 · 문맥이 모두 클 때 (다섯 모델 모두 HBM 초과)")]
# 종류 — 모델마다 묶음 이름이 달라 하나로 맞춤
KIND = [("w_attn", "attention 가중치", mr.GRAY), ("w_moe", "MoE · FFN 가중치", mr.GRAY_L),
        ("w_etc", "LM head · 임베딩 · 기타 가중치", mr.LMH), ("kv", "KV cache", mr.BLUE),
        ("idx", "색인 · 블록 키", mr.ORANGE), ("state", "state · conv", mr.GREEN),
        ("ring", "최근 창 · 압축기 state", "#6250d6"), ("temp", "커널 사이 중간값", mr.YELLOW)]
REQK = [k for k in KIND if k[0] in ("kv", "idx", "state", "ring")]
LIFEK = [(k, tr.LIFE[k], mt.LIFE_COL[k]) for k in ("svc", "req", "step")]
CHARK = [(k, tr.CHAR[k], mt.CHAR_COL[k]) for k in tr.CHAR]
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


def kkey(r):
    if r["life"] == "svc":
        g = r["group"]
        return "w_attn" if "attention" in g else "w_moe" if ("MoE" in g or "FFN" in g) else "w_etc"
    if r["life"] == "step":
        return "temp"
    return gkey(r)


def agg(mid, B, T, keyf, only_req=False):
    """(보관 dict, 이동 dict) — 키 함수로 묶음."""
    st, mv = {}, {}
    for r in tr.items(M_(mid), B, T, "model"):
        if only_req and r["life"] != "req":
            continue
        k = keyf(r)
        st[k] = st.get(k, 0) + r["store"]
        mv[k] = mv.get(k, 0) + r["move"]
    return st, mv


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


# ------------------------------------------------------------------ 모델 줄 머리 — 구조 알약 + 모델 이름 + 띠
XF, XN = 40, 168


def fam_rows(f, y, RH, FG, x_end, draw_row):
    """구조 묶음별로 띠를 칠하고 줄마다 draw_row(mid, y)를 부름. 마지막 y."""
    for k, lab, ms in FAM:
        h = len(ms) * RH
        f.rect(XF, y, x_end - XF, h, FAM_BG[k], "none", 8)
        f.rect(XF, y, 6, h, FAM_ACC[k], "none", 3)
        pw = mr.text_w(lab, 14) + 22
        f.rect(XF + 16, y + h / 2 - 13 - (8 if len(ms) > 1 else 0), pw, 26, FAM_ACC[k], "none", 13)
        f.text(XF + 16 + pw / 2, y + h / 2 + 5 - (8 if len(ms) > 1 else 0), lab, 14, "middle", 800, "#ffffff")
        if len(ms) > 1:
            f.text(XF + 16, y + h / 2 + 28, FAM_SUB[k], 12, color=mr.MUTED)
        for i, mid in enumerate(ms):
            yy = y + i * RH
            f.text(XN, yy + RH / 2 + 6, NAME[mid], 15, weight=800)
            draw_row(mid, yy)
        y += h + FG
    return y - FG


def fam_legend(f, x, y):
    """구조 범례 한 줄."""
    for k, lab, ms in FAM:
        pw = mr.text_w(lab, 13) + 18
        f.rect(x, y - 16, pw, 22, FAM_ACC[k], "none", 11)
        f.text(x + pw / 2, y, lab, 13, "middle", 800, "#ffffff")
        t = f"{' · '.join(TAG[m] for m in ms)} — {FAM_SUB[k]}"
        f.text(x + pw + 8, y, t, 13, color=mr.MUTED)
        x += pw + 8 + mr.text_w(t, 13) + 26
    return x


# ------------------------------------------------------------------ 바이트 막대 비교 (모델 전체 · 요청 1개 몫 공통)
X1, PW, TW = 380, 540, 110
X2 = X1 + PW + TW + 30


def bar(f, x0, y, RH, vals, cats, scale):
    x = x0
    for k, _, c in cats:
        v = vals.get(k, 0)
        w = v * scale
        if w > 0.4:
            f.rect(x, y + 6, max(w - 2, 0.8), RH - 12, c, mr.BOX[1] if c == mr.LMH else "none", 3, 1)
            lab = mt.fb(v)
            if w >= mr.text_w(lab, 12.5) + 12:
                f.text(x + w / 2 - 1, y + RH / 2 + 5, lab, 12.5, "middle", 700, "#ffffff" if c in mt.DARK else mr.INK)
        x += w
    f.text(x + 8, y + RH / 2 + 6, mt.fb(sum(vals.values())), 14, weight=800)


def cmp_fig(name, ttl, sub, cats, conds, valf, take, take_sub, shared=False, heads=None):
    """조건마다 모델 5줄 (구조별 띠). 보관 | 이동 두 판, 바이트 막대. shared = 조건끼리도 같은 눈금."""
    RH, FG, CH, CG, TOP = 36, 8, 40, 26, 250
    data = {(c[0], c[1], m): valf(m, c[0], c[1]) for c in conds for m in MODELS}
    smax = {c[:2]: max(sum(data[(c[0], c[1], m)][0].values()) for m in MODELS) for c in conds}
    mmax = {c[:2]: max(sum(data[(c[0], c[1], m)][1].values()) for m in MODELS) for c in conds}
    if shared:
        a, b = max(smax.values()), max(mmax.values())
        smax, mmax = {k: a for k in smax}, {k: b for k in mmax}
    block = CH + 5 * RH + 2 * FG + CG
    f = mr.Fig(name, W, TOP + len(conds) * block + 130)
    mr.title(f, ttl, sub)
    mr.legend(f, [(c[1], c[2]) for c in cats], 40, 130, 24)
    fam_legend(f, 40, 172)
    h1, h2 = heads or ("보관 — HBM에 들고 있는 양", "스텝당 이동 — HBM 읽기 + 쓰기")
    for x, t in ((X1, h1), (X2, h2)):
        f.text(x, TOP - 22, t, 17, weight=700)
    y = TOP
    for B, T, lab, note in conds:
        f.text(40, y + 24, lab, 18, weight=800)
        f.text(40 + mr.text_w(lab, 18) + 14, y + 24, note, 14, color=mr.MUTED)
        if not shared or (B, T) == conds[0][:2]:
            f.text(X1 + PW + TW, y + 24, f"막대 끝 = {mt.fb(smax[(B, T)])}", 12.5, "end", color=mr.MUTED)
            f.text(X2 + PW + TW, y + 24, f"막대 끝 = {mt.fb(mmax[(B, T)])}", 12.5, "end", color=mr.MUTED)
        y += CH

        def row(mid, yy, B=B, T=T):
            st, mv = data[(B, T, mid)]
            bar(f, X1, yy, RH, st, cats, PW / smax[(B, T)])
            bar(f, X2, yy, RH, mv, cats, PW / mmax[(B, T)])
        y = fam_rows(f, y, RH, FG, W - 40, row) + CG
    mr.takeaway(f, y + 36, take, take_sub)
    mr.save(f)


def fig_kind_model():
    cmp_fig("synth-t1-kind-model", "모델 5개 — 데이터 종류별 보관 · 이동 (모델 전체 · 바이트)",
            "decode 1스텝 · 전 층 + MoE + LM head · 조건마다 다섯 모델이 같은 눈금 · MoE expert 읽기는 균등 라우팅 가정",
            KIND, CONDS, lambda m, B, T: agg(m, B, T, kkey), *TAKE["t1"])


def fig_kind_req():
    conds = [(1, 8192, "문맥 8K", "요청 1개"), (1, 131072, "문맥 128K", "요청 1개")]
    cmp_fig("synth-t2-kind-req", "모델 5개 — 요청 1개 몫의 요청별 데이터 (종류별 · 바이트)",
            "요청별 데이터 = KV · 색인 키 · state 등 요청 동안 남는 것 · 전 층 합 · 두 문맥이 같은 눈금 → 문맥에 따라 무엇이 커지는지 보임",
            REQK, conds, lambda m, B, T: agg(m, B, T, gkey, only_req=True), *TAKE["t2"], shared=True,
            heads=("보관 — 요청 1개가 들고 있는 양", "스텝당 이동 — 요청 1개 몫"))


def fig_life_model():
    cmp_fig("synth-t3-life-model", "모델 5개 — 수명별 보관 · 이동 (모델 전체 · 바이트)",
            "서비스 내내 = 가중치 · 요청 동안 = KV · 색인 키 · state 등 · 스텝 안 = 커널 사이 값(그린 층만 셈) · 조건마다 같은 눈금",
            LIFEK, CONDS, lambda m, B, T: agg(m, B, T, lambda r: r["life"]), *TAKE["t3"])


def fig_char_model():
    cmp_fig("synth-t5-char-model", "모델 5개 — 데이터 성격별 보관 · 이동 (모델 전체 · 바이트)",
            "성격 라벨은 요청 1개 · 128K 기준 (요청이 많으면 MoE도 대부분을 읽음) · 조건마다 같은 눈금",
            CHARK, CONDS, lambda m, B, T: agg(m, B, T, lambda r: r["char"]), *TAKE["t5"])


# ------------------------------------------------------------------ t4 역전점
def fig_cross():
    PX, PWX, TOP, RH, FG = 380, 980, 250, 64, 10
    lo, hi = 0, math.log2(4096)

    def sx(n):
        return PX + PWX * (math.log2(min(max(n, 1), 4096)) - lo) / (hi - lo)
    H = TOP + 5 * RH + 2 * FG
    f = mr.Fig("synth-t4-life-cross", W, H + 230)
    mr.title(f, "역전점 — 요청 몇 개부터 '요청 동안' 데이터가 '서비스 내내' 가중치를 넘나 (문맥 128K)",
             "○ 보관 역전 = 요청별 보관 > 가중치 보관 · ● 이동 역전 = 요청별 이동 > 가중치 이동 · 가로축 = 요청 수 (로그)")
    fam_legend(f, 40, 140)
    for n in (1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        big = n in (32, 256)
        f.line(sx(n), TOP - 10, sx(n), H, "#c9ccd6" if big else mr.GRID, 2 if big else 1)
        f.text(sx(n), H + 26, str(n), 13, "middle", color=mr.MUTED)
    f.text(sx(32), TOP - 18, "32", 12, "middle", 700, mr.MUTED)
    f.text(sx(256), TOP - 18, "256", 12, "middle", 700, mr.MUTED)
    f.text(PX + PWX / 2, H + 56, "요청 수", 14, "middle", 700, mr.MUTED)

    def row(mid, y):
        cs, cm = cross_store(mid), cross_move(mid)
        f.line(PX, y + RH / 2, PX + PWX, y + RH / 2, "#dcdfe7", 1)
        if cm:
            f.line(sx(cs), y + RH / 2, sx(cm), y + RH / 2, mr.AXIS, 3)
        f.raw(f'<circle cx="{sx(cs):.1f}" cy="{y + RH / 2}" r="9" fill="#ffffff" stroke="{mr.INK}" stroke-width="3"/>')
        f.text(sx(cs), y + RH / 2 + 28, f"보관 {cs}", 13, "middle", 700)
        if cm:
            f.raw(f'<circle cx="{sx(cm):.1f}" cy="{y + RH / 2}" r="9" fill="{mr.INK}"/>')
            f.text(sx(cm), y + RH / 2 - 16, f"이동 {cm:,}", 13, "middle", 700)
    fam_rows(f, TOP, RH, FG, PX + PWX + 30, row)
    RX = PX + PWX + 60
    f.text(RX, TOP + 4, "읽는 방법", 15, weight=800)
    cf.lines(f, RX, TOP + 32, cf.wrap("○가 왼쪽일수록 요청별 데이터가 일찍 용량을 차지 · ●가 오른쪽일수록 요청이 많아도 이동은 가중치가 정함. "
                                      "○ ~ ● 사이(회색 선)가 길면 '용량은 막히는데 대역폭은 남는' 구간", 14, W - 40 - RX), 14, mr.INK, lh=22)
    mr.takeaway(f, H + 110, *TAKE["t4"])
    mr.save(f)


# ------------------------------------------------------------------ t6 요청 1개: 보관 vs 이동 (읽는 비율)
MARK = {"full": "circle", "sparse": "square", "hybrid": "diamond"}


def marker(f, x, y, r, fam, fill):
    if fam == "full":
        f.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
    elif fam == "sparse":
        f.raw(f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" rx="2" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
    else:
        d = r * 1.3
        f.raw(f'<path d="M{x:.1f},{y - d:.1f} L{x + d:.1f},{y:.1f} L{x:.1f},{y + d:.1f} L{x - d:.1f},{y:.1f} Z" fill="{fill}" '
              f'stroke="#ffffff" stroke-width="2"/>')


def fig_per_req():
    PX, PY, PWX, PH = 150, 150, 940, 760
    lo, hi = -2.0, 1.0           # 10^-2 ~ 10^1 GB

    def sx(v):
        return PX + PWX * (math.log10(v / 1e9) - lo) / (hi - lo)

    def sy(v):
        return PY + PH - PH * (math.log10(v / 1e9) - lo) / (hi - lo)
    f = mr.Fig("synth-t6-char-req", W, PY + PH + 220)
    mr.title(f, "요청 1개 몫의 요청별 데이터 — 보관 vs 스텝당 이동 (문맥 8K · 32K · 128K)",
             "두 축 모두 로그 · 점 크기 = 문맥 길이 · 모양 = 구조 (● 기준선 · ■ Sparse · ◆ Hybrid) · 대각선에서 아래로 멀수록 골라 읽음(용량형)")
    for e in range(-2, 2):
        for m in (1, 2, 5):
            v = m * 10 ** e * 1e9
            if 10 ** lo * 1e9 <= v <= 10 ** hi * 1e9:
                f.line(sx(v), PY, sx(v), PY + PH, mr.GRID, 1)
                f.line(PX, sy(v), PX + PWX, sy(v), mr.GRID, 1)
                lab = cf.fsize(v).replace(".0 ", " ")
                f.text(sx(v), PY + PH + 26, lab, 13, "middle", color=mr.MUTED)
                f.text(PX - 10, sy(v) + 5, lab, 13, "end", color=mr.MUTED)
    f.rect(PX, PY, PWX, PH, "none", mr.AXIS, 0, 1.2)
    f.text(PX + PWX / 2, PY + PH + 60, "보관 — 요청 1개가 HBM에 들고 있는 요청별 데이터 (용량)", 16, "middle", 700)
    f.raw(f'<text x="{PX - 92}" y="{PY + PH / 2}" font-family="{mr.FONT}" font-size="16" font-weight="700" fill="{mr.INK}" '
          f'text-anchor="middle" transform="rotate(-90 {PX - 92} {PY + PH / 2})">스텝당 이동 — 요청 1개 몫의 읽기 + 쓰기 (대역폭)</text>')
    for k, lab in ((1.0, "이동 = 보관 (보관한 만큼 매 스텝 읽음)"), (0.1, "이동 = 보관의 10%")):
        x1, x2 = 10 ** lo * 1e9, 10 ** hi * 1e9
        a, b = max(x1, x1 / k), min(x2, x2 / k)
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
            pts.append((sx(sum(v[0] for v in g.values())), sy(sum(v[1] + v[2] for v in g.values()))))
        f.raw('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="none" stroke="{MCOL[mid]}" stroke-width="2.4"/>')
        for i, (x, y) in enumerate(pts):
            marker(f, x, y, 5 + 3 * i, FAM_OF[mid], MCOL[mid])
        x, y = pts[-1]
        dx, dy = {"mla-v31": (16, 28), "kda-k3": (-16, -18), "dsa-v32": (16, 6), "qsa-qwen38": (16, 6), "csa-v4flash": (16, 6)}[mid]
        f.text(x + dx, y + dy, TAG[mid], 16, "start" if dx > 0 else "end", 800)
        x0, y0 = pts[0]
        f.text(x0 - 12, y0 + (20 if mid == "qsa-qwen38" else 5), "8K", 12, "end", color=mr.MUTED)
    # 오른쪽 표 — 128K 값, 구조별로 묶음
    RX = PX + PWX + 60
    f.text(RX, PY + 10, "문맥 128K · 요청 1개", 17, weight=800)
    for t, dx in (("모델", 0), ("보관", 250), ("이동", 350), ("읽는 비율", 450)):
        f.text(RX + dx, PY + 46, t, 14, weight=700, color=mr.MUTED)
    y = PY + 82
    for k, lab, ms in FAM:
        f.text(RX, y, lab, 13, weight=800, color=FAM_ACC[k])
        y += 26
        for mid in ms:
            g = per_req(mid, 131072)
            S = sum(v[0] for v in g.values())
            R = sum(v[1] for v in g.values())
            mv = sum(v[1] + v[2] for v in g.values())
            marker(f, RX + 8, y - 5, 6, FAM_OF[mid], MCOL[mid])
            f.text(RX + 24, y, NAME[mid], 14, weight=700)
            f.text(RX + 250, y, cf.fsize(S), 14)
            f.text(RX + 350, y, cf.fsize(mv), 14)
            f.text(RX + 450, y, f"{R / S:.0%}", 14, weight=700)
            y += 32
        y += 6
    cf.lines(f, RX, y + 8, ["읽는 비율 = 스텝당 읽기 ÷ 보관", "(state의 다시 쓰기는 이동에만 들어감)"], 13)
    y += 70
    f.text(RX, y, "읽는 방법", 15, weight=800)
    cf.lines(f, RX, y + 28, cf.wrap("대각선 위 = 보관한 만큼 매 스텝 옮김(전부 읽음 · 용량 + 대역폭). 대각선보다 위 = 다시 쓰기까지 더해짐. "
                                    "아래로 멀수록 골라 읽는 용량형. 선이 오른쪽으로만 길면 문맥이 이동을 거의 안 늘림", 14, W - 40 - RX), 14, mr.INK, lh=22)
    mr.takeaway(f, PY + PH + 120, *TAKE["t6"])
    mr.save(f)


# ------------------------------------------------------------------ 그림 아래 한 줄 결론 (수치는 본문과 같은 계산 — print로 확인)
TAKE = {
    "t1": ("이동 차이는 가중치 크기가 가장 크게 만듦 — 요청 · 문맥이 모두 커져야 KV(full) · 색인 키(sparse) · state(hybrid) 차이가 드러남",
           "요청 256개 · 128K 이동: V3.1 2.03 TB(KV 1.34 TB) · K3.1 1.90 TB(MoE 1.47 TB · state 230 GB) · V3.2 993 GB(색인 키 270 GB) · Q3.8 348 GB · V4 187 GB"),
    "t2": ("sparse는 KV를 그대로 들고 읽기만 줄임 — hybrid는 문맥과 무관한 state가 남아 짧은 문맥에서 요청당 이동이 full보다 큼",
           "128K 이동: V3.1 5.2 GB → K3.1 1.4 · V3.2 1.1 · Q3.8 0.38 · V4 0.11 GB / 8K 이동: K3.1 0.98 GB(state 0.90 GB) > V3.1 0.33 GB"),
    "t3": ("HBM 이동은 다섯 모델 모두 '서비스 내내 + 요청 동안'이 거의 전부 — 요청 동안 몫은 full은 문맥을, hybrid는 요청 수를 따라 커짐",
           "요청 동안 이동 몫 — 256개 · 8K: Q3.8 23% · K3.1 14% · V3.1 11% · V3.2 5.2% · V4 2.6% / 256개 · 128K: V3.1 66% · V3.2 29% · Q3.8 28% · K3.1 18% · V4 16%"),
    "t4": ("골라 읽는 KV가 들어간 모델은 이동 역전만 뒤로 밀리고 보관 역전은 그대로 — 용량이 대역폭보다 먼저 막힘",
           "128K: V3.1 보관 129 · 이동 126 (함께) / V3.2 108 · 592 / Q3.8 74 · 656 / K3.1 348 · 1,137 / V4 300 · 1,372 (압축으로 둘 다 뒤로)"),
    "t5": ("골라 읽는 KV는 보관만 크고 이동은 거의 없음 — 이동 차이는 전부 읽는 데이터(full KV · 색인 키)와 다시 쓰는 state가 만듦",
           "요청 256개 · 128K: 골라 읽는 KV 보관 V3.2 1.34 TB · K3.1 928 GB · Q3.8 825 GB → 이동 21 · 15 · 13 GB / 전부 읽는 요청별 데이터 이동 V3.1 1.34 TB · V3.2 270 GB"),
    "t6": ("full만 대각선(보관 = 이동) — sparse · hybrid는 읽는 비율이 8 ~ 22%로 내려가고, hybrid는 state 다시 쓰기만큼 위로 뜸",
           "128K 읽는 비율: V3.1 100% · V4 22% · K3.1 21% · V3.2 18% · Q3.8 8% / 8K: K3.1 이동 0.98 GB > 보관 0.70 GB (state를 읽고 다시 씀)"),
}


# ------------------------------------------------------------------ 정리
SUM_ROWS = [
    ("full KV (V3.1)", "blue", "app_all", "끝에 추가 · 전부 읽음 — 토큰당 40 KB, 요청 × 문맥에 비례",
     "보관한 만큼 매 스텝 읽음", "128K에서 요청 126 ~ 129개부터 보관 · 이동 모두 요청별 데이터가 가중치를 넘음",
     "HBM 용량 · 대역폭을 함께, KV 압축 · 저정밀", "아래 계층 메모리는 읽기 대역폭이 모자라 효과 제한", "2.5"),
    ("골라 읽는 KV (V3.2 · V4 · K3.1 · Q3.8)", "blue", "app_sel", "끝에 추가 · 골라 읽음 — 요청당 고정 예산(2,048 · 512)만 읽음",
     "이동은 줄지만 보관은 그대로 — 용량이 먼저 막힘", "보관 역전이 이동 역전보다 3.3 ~ 8.9배 먼저 · V3.2 보관은 색인 키만큼 오히려 20% 늘어남",
     "전체 KV는 큰 메모리, 고를 KV만 HBM", "선택 겹침 · 블록 단위 전송이 클수록 유리", "2.1"),
    ("전부 읽는 색인 · 블록 키 (V3.2 · V4 · K3.1 · Q3.8)", "orange", "app_all", "끝에 추가 · 전부 읽음 — 토큰당 0.7 ~ 8 KB",
     "골라 읽는 모델의 요청별 이동 중 큰 몫", "요청 256개 · 128K 모델 이동: V3.2 27% · V4 12.5% · Q3.8 7.4% · K3.1 5.6%",
     "메모리 근처 선택 · 색인 연산 · top-k", "키 형식 · head 수에 따라 연산 요구가 4 ~ 124 FLOP/B로 갈림", "2.2"),
    ("다시 쓰는 state (K3.1 · Q3.8)", "green", "rmw_all", "매 스텝 전부 다시 씀 — 요청당 115 ~ 449 MB, 문맥과 무관",
     "많은 요청 · 짧은 문맥에서도 이동이 남음", "요청 256개 · 8K: Q3.8 이동 18% · K3.1 13% — 이동 비중이 보관 비중의 약 2배",
     "메모리 안 갱신으로 왕복 제거", "이득은 갱신 연산을 메모리 쪽이 감당하는지에 달림", "2.4"),
    ("MoE 가중치 (전 모델)", "gray-l", "ro_sel", "read-only · 골라 읽음 — 요청이 늘면 대부분을 읽음",
     "sparse · hybrid에서는 다시 이동의 대부분", "요청 32개 · 128K: 요청별 데이터 이동은 전체의 4 ~ 9%뿐 (V3.1은 28%)",
     "가중치 대역폭 · 요청 수에 맞춘 배치", "attention 쪽 HW 이득의 상한 = 요청별 이동 비중", "-"),
]


def fig_sum():
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


# ------------------------------------------------------------------ 본문 수치
def show():
    G = 1e9
    for B, T, lab, _ in CONDS:
        print(f"===== {lab}")
        for nm, keyf, cats in (("kind", kkey, KIND), ("life", lambda r: r["life"], LIFEK), ("char", lambda r: r["char"], CHARK)):
            print(f"  -- {nm}")
            for mid in MODELS:
                st, mv = agg(mid, B, T, keyf)
                S, V = sum(st.values()), sum(mv.values())
                fs = " ".join(f"{k}={st[k] / G:.3g}" for k, _, _ in cats if st.get(k, 0) > 1e6)
                fm = " ".join(f"{k}={mv[k] / G:.3g}({mv[k] / V:.1%})" for k, _, _ in cats if mv.get(k, 0) > 1e6)
                print(f"    {TAG[mid]:5s} store {S / G:8.1f} GB [{fs}]\n          move  {V / G:8.1f} GB [{fm}]")
    print("===== per request (B=1)")
    for T in TS:
        for mid in MODELS:
            g = per_req(mid, T)
            S = sum(v[0] for v in g.values())
            R = sum(v[1] for v in g.values())
            Wr = sum(v[2] for v in g.values())
            print(f"  T{T // 1024}K {TAG[mid]:5s} store {S / G:.3f} read {R / G:.3f} write {Wr / 1e6:.1f}MB ratio {R / S:.2f} |",
                  {k: (round(v[0] / G, 3), round((v[1] + v[2]) / G, 3)) for k, v in g.items()})
    print("===== cross / gpu")
    for mid in MODELS:
        print(f"  {TAG[mid]:5s} cross store {cross_store(mid)} move {cross_move(mid)} | minG",
              [min_gpu(mid, B, T) for B in (1, 32, 256) for T in TS], "n_gpu", M_(mid)["model"]["deploy"]["n_gpu"])
        for B, T in ((32, 131072), (256, 8192), (256, 131072)):
            d = life_move(mid, B, T)
            print(f"      B{B} T{T // 1024}K req move share {d.get('req', 0) / sum(d.values()):.3f}")


if __name__ == "__main__":
    if sys.argv[1:] == ["print"]:
        show()
    else:
        fig_kind_model()
        fig_kind_req()
        fig_life_model()
        fig_cross()
        fig_char_model()
        fig_per_req()
        fig_sum()
