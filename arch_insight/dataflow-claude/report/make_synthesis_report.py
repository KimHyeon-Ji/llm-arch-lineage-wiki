"""종합 보고서 그림 — 모델 5개(기준선 full · sparse 2 · hybrid 2)를 모델별 보고서와 같은 틀로 비교.

  python report/make_synthesis_report.py            그림 생성
  python report/make_synthesis_report.py print      본문 수치 출력

모델별 보고서는 구성 비율(합 = 100%)로 그리지만, 종합은 **바이트 그대로** 그려 모델끼리의 크기 차이가 보이게 함.
조건은 셋만: 요청 1개 · 128K / 요청 256개 · 8K / 요청 256개 · 128K. 모델 줄은 기준선 · Sparse · Hybrid로 묶어 띠를 칠함.

synth-t1-kind-model   종류별 보관 · 이동 (모델 전체, 조건 3개)
synth-t2-kind-req     요청 1개 몫의 요청별 데이터 — 종류별 (문맥 8K · 128K, 같은 눈금)
synth-t3-life-model   수명별 보관 · 이동 (모델 전체, 조건 3개)
synth-t5-char-model   성격별 보관 · 이동 (모델 전체, 조건 3개)
synth-s2a-rel         2.1 Full MLA = 100%로 놓은 요청 1개의 저장량 · 읽고 쓰는 양 (128K)
synth-s2b-ctx         2.2 문맥이 길어질 때 요청 1개가 읽고 쓰는 양 (선형 눈금 · hybrid state 바닥)
synth-s4a-share       4.1 요청이 늘 때 요청 데이터가 저장 공간 · 읽기에서 차지하는 몫 (50% = 가중치와 같음)
synth-s4b-hbm         4.2 필요한 H200 장수 (빨간 선 = 모델별 구성 장수)
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
# 종합 보고서는 attention 이름으로 부름 — 모델 버전은 구조 범례에 함께 적음
NAME = {"mla-v31": "Full MLA", "dsa-v32": "DSA", "csa-v4flash": "CSA + HCA", "kda-k3": "KDA + DSA", "qsa-qwen38": "GDN + QSA"}
MODEL = {"mla-v31": "DeepSeek-V3.1", "dsa-v32": "DeepSeek-V3.2", "csa-v4flash": "DeepSeek-V4-Flash",
         "kda-k3": "Kimi K3.1 Pro", "qsa-qwen38": "Qwen3.8-Flash-Next"}
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
LIFE_LAB = {"svc": "서비스 내내 (가중치)", "req": "요청 동안 (KV · 색인 키 · state)", "step": "스텝 안 (커널 사이 값)"}
CHAR_LAB = {"ro_all": "읽기만 · 전부 읽음", "ro_sel": "읽기만 · 골라 읽음", "app_all": "토큰마다 추가 · 전부 읽음",
            "app_sel": "토큰마다 추가 · 골라 읽음", "ring": "고정 크기 · 덮어씀", "rmw": "매 스텝 전부 다시 씀",
            "tmp": "임시 (커널 사이)"}
LIFEK = [(k, LIFE_LAB[k], mt.LIFE_COL[k]) for k in ("svc", "req", "step")]
CHARK = [(k, CHAR_LAB.get(k, tr.CHAR[k]), mt.CHAR_COL[k]) for k in tr.CHAR]
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
        t = " · ".join(f"{NAME[m]} ({MODEL[m]})" for m in ms)
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
    h1, h2 = heads or ("저장량 — HBM에 저장해 둔 양", "스텝마다 읽고 쓰는 양 — HBM 읽기 + 쓰기")
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
    cmp_fig("synth-t1-kind-model", "모델 5개 — 데이터 종류별 저장량 · 스텝마다 읽고 쓰는 양 (모델 전체 · 바이트)",
            "decode 1스텝 · 전 층 + MoE + LM head · 조건마다 다섯 모델이 같은 눈금 · MoE expert 읽기는 균등 라우팅 가정",
            KIND, CONDS, lambda m, B, T: agg(m, B, T, kkey), *TAKE["t1"])


def fig_kind_req():
    conds = [(1, 8192, "문맥 8K", "요청 1개"), (1, 131072, "문맥 128K", "요청 1개")]
    cmp_fig("synth-t2-kind-req", "모델 5개 — 요청 1개에 딸린 데이터 (KV · 색인 키 · state · 바이트)",
            "요청마다 따로 쌓이는 데이터만 셈 (가중치 제외) · 전 층 합 · 두 문맥이 같은 눈금 → 문맥이 길어질 때 무엇이 커지는지 보임",
            REQK, conds, lambda m, B, T: agg(m, B, T, gkey, only_req=True), *TAKE["t2"], shared=True,
            heads=("저장량 — 요청 1개가 저장해 둔 양", "스텝마다 읽고 쓰는 양 — 요청 1개 몫"))


def fig_life_model():
    cmp_fig("synth-t3-life-model", "모델 5개 — 수명(얼마나 오래 남나)별 저장량 · 스텝마다 읽고 쓰는 양 (모델 전체 · 바이트)",
            "스텝 안 = 한 스텝 안에서 쓰고 버리는 커널 사이 값(그린 층만 셈) — 칩 안에 남길 수 있는 것은 이것뿐 · 조건마다 같은 눈금",
            LIFEK, CONDS, lambda m, B, T: agg(m, B, T, lambda r: r["life"]), *TAKE["t3"])


def fig_char_model():
    cmp_fig("synth-t5-char-model", "모델 5개 — 읽고 쓰는 방식별 저장량 · 스텝마다 읽고 쓰는 양 (모델 전체 · 바이트)",
            "분류는 요청 1개 · 128K 기준 (요청이 많으면 MoE도 대부분을 읽음) · 조건마다 같은 눈금",
            CHARK, CONDS, lambda m, B, T: agg(m, B, T, lambda r: r["char"]), *TAKE["t5"])


# ------------------------------------------------------------------ 2장 · 4장 — 막대 · 선으로 바로 읽히게
SAVE_COL, MOVE_COL, OVER = "#2f4b8c", mr.ORANGE, "#d1495b"     # 저장 · 읽고 쓰기 · 넘침


def req_tot(mid, T):
    """요청 1개에 딸린 데이터 (저장량, 스텝마다 읽고 쓰는 양)."""
    g = per_req(mid, T)
    return sum(v[0] for v in g.values()), sum(v[1] + v[2] for v in g.values())


def dashed(f, x1, y1, x2, y2, c, sw=1.8, dash="7 6"):
    f.raw(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{sw}" '
          f'stroke-dasharray="{dash}"/>')


def ylabel(f, x, y, t):
    f.raw(f'<text x="{x}" y="{y}" font-family="{mr.FONT}" font-size="16" font-weight="700" fill="{mr.INK}" '
          f'text-anchor="middle" transform="rotate(-90 {x} {y})">{mr.esc(t)}</text>')


def halo(f, x, y, s, size, anchor, weight, color):
    """선 위에 놓이는 숫자 — 흰 테두리."""
    f.raw(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{mr.FONT}" font-size="{size}" text-anchor="{anchor}" '
          f'font-weight="{weight}" fill="{color}" stroke="#ffffff" stroke-width="5" stroke-linejoin="round" '
          f'paint-order="stroke">{mr.esc(s)}</text>')


def fig_rel():
    """2.1 — Full MLA = 100%로 놓은 요청 1개의 저장량 · 읽고 쓰는 양 (128K)."""
    T, RH, FG, TOP = 131072, 50, 10, 270
    P, PWr, MAXP = (350, 1040), 480, 1.25
    base = req_tot("mla-v31", T)
    H = TOP + 5 * RH + 2 * FG
    f = mr.Fig("synth-s2a-rel", W, H + 190)
    mr.title(f, "Full MLA를 100%로 놓고 본 요청 1개의 데이터 (문맥 128K)",
             "요청 하나에 딸린 KV · 색인 키 · state만 셈 (가중치 제외) · 막대가 100% 선보다 짧으면 Full MLA보다 적음")
    fam_legend(f, 40, 132)
    for x, h in zip(P, ("저장량 — HBM에 저장해 둔 양", "스텝마다 읽고 쓰는 양")):
        f.text(x, TOP - 62, h, 18, weight=800)
        f.text(x + PWr / MAXP, TOP - 34, "Full MLA = 100%", 12.5, "middle", 700)
        for p, t in ((0, "0"), (0.5, "50%"), (1.0, "100%")):
            f.text(x + PWr * p / MAXP, H + 28, t, 12.5, "middle", color=mr.MUTED)

    def guides(y0, y1):
        for x in P:
            f.line(x + PWr * 0.5 / MAXP, y0, x + PWr * 0.5 / MAXP, y1, "#d3d6e0", 1)
            dashed(f, x + PWr / MAXP, y0, x + PWr / MAXP, y1, mr.INK, 1.6)
    guides(TOP - 28, H + 6)

    def row(mid, y):
        guides(y, y + RH)
        v = req_tot(mid, T)
        for i in range(2):
            p = v[i] / base[i]
            w = max(PWr * p / MAXP, 3)
            f.rect(P[i], y + 10, w, RH - 20, MCOL[mid], "none", 3)
            lab = f"{p:.0%}" if p >= 0.095 else f"{p:.1%}"
            d = p - 1
            note = "기준" if mid == "mla-v31" else f"{abs(d):.0%} 많음" if d > 0 else f"{abs(d):.0%} 적음"
            xe = P[i] + w + 10
            wl = mr.text_w(lab, 16) * 1.3 + 10 + mr.text_w(note, 14) * 1.2
            if p < 1 and xe + wl > P[i] + PWr / MAXP - 4:      # 100% 선에 걸리면 막대 안쪽에 흰 글씨
                f.text(P[i] + w - 12, y + RH / 2 + 6, f"{lab} · {note}", 15, "end", 800, "#ffffff")
                continue
            f.text(xe, y + RH / 2 + 6, lab, 16, weight=800)
            f.text(xe + mr.text_w(lab, 16) * 1.15 + 10, y + RH / 2 + 6, note, 14, weight=700,
                   color=OVER if (d > 0.005 and mid != "mla-v31") else mr.MUTED)
    fam_rows(f, TOP, RH, FG, W - 40, row)
    mr.takeaway(f, H + 96, *TAKE["s2a"])
    mr.save(f)


def cross_T(a, b):
    """두 선 [(T, v)]가 처음 엇갈리는 문맥 — 선형 보간."""
    for (t0, a0), (t1, a1), (_, b0), (_, b1) in zip(a, a[1:], b, b[1:]):
        d0, d1 = a0 - b0, a1 - b1
        if d0 * d1 < 0:
            return t0 + (t1 - t0) * d0 / (d0 - d1), a0 + (a1 - a0) * d0 / (d0 - d1)
    return None


def state_rw(mid, T=8192):
    s = per_req(mid, T).get("state", [0, 0, 0])
    return s[1] + s[2]


def fig_ctx():
    """2.2 — 문맥이 길어질 때 요청 1개가 스텝마다 읽고 쓰는 양 (선형 눈금)."""
    TT = list(range(4096, 131073, 4096))
    PX, PY, PWX, PH, YMAX = 150, 190, 1180, 540, 5.5e9

    def sx(T):
        return PX + PWX * T / 131072

    def sy(v):
        return PY + PH - PH * v / YMAX
    cur = {m: [(T, req_tot(m, T)[1]) for T in TT] for m in MODELS}
    f = mr.Fig("synth-s2b-ctx", W, PY + PH + 200)
    mr.title(f, "문맥이 길어질 때 요청 1개가 스텝마다 읽고 쓰는 양",
             "요청 하나에 딸린 KV · 색인 키 · state만 셈 (가중치 제외) · 문맥에 비례하는 부분은 가파르게 오르고, 문맥과 상관없는 부분은 바닥에 깔림")
    fam_legend(f, 40, 132)
    for k in range(9):
        f.line(sx(k * 16384), PY, sx(k * 16384), PY + PH, mr.GRID, 1)
        f.text(sx(k * 16384), PY + PH + 26, "0" if k == 0 else f"{k * 16}K", 13, "middle", color=mr.MUTED)
    for g in range(6):
        f.line(PX, sy(g * 1e9), PX + PWX, sy(g * 1e9), mr.GRID, 1)
        f.text(PX - 10, sy(g * 1e9) + 5, "0" if g == 0 else f"{g} GB", 13, "end", color=mr.MUTED)
    f.rect(PX, PY, PWX, PH, "none", mr.AXIS, 0, 1.2)
    f.text(PX + PWX / 2, PY + PH + 60, "문맥 길이 (토큰)", 16, "middle", 700)
    ylabel(f, PX - 80, PY + PH / 2, "스텝마다 읽고 쓰는 양 — 요청 1개 몫")
    # KDA + DSA의 state — 문맥과 상관없는 바닥
    sv = state_rw("kda-k3")
    dashed(f, PX, sy(sv), PX + PWX, sy(sv), MCOL["kda-k3"], 2)
    for m in MODELS:
        f.raw('<polyline points="' + " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in cur[m]) +
              f'" fill="none" stroke="{MCOL[m]}" stroke-width="3.2" stroke-linejoin="round"/>')
        t, v = cur[m][-1]
        dy = {"mla-v31": 6, "kda-k3": -4, "dsa-v32": 12, "qsa-qwen38": 2, "csa-v4flash": 8}[m]
        f.text(sx(t) + 12, sy(v) + dy, NAME[m], 15, weight=800, color=MCOL[m] if m != "mla-v31" else mr.INK)
    # 8K에서의 차이
    a = dict(cur["mla-v31"])[8192]
    b = dict(cur["kda-k3"])[8192]
    f.line(sx(8192), sy(a), sx(8192), sy(b), mr.INK, 1.6)
    for v in (a, b):
        f.raw(f'<circle cx="{sx(8192):.1f}" cy="{sy(v):.1f}" r="5" fill="{mr.INK}"/>')
    dashed(f, sx(8192), sy(b) - 8, sx(8192), sy(1.62e9), mr.INK, 1.2, "3 4")
    f.text(sx(8192) - 12, sy(1.62e9) - 10, f"문맥 8K: KDA + DSA가 Full MLA의 {b / a:.0f}배", 14, "start", 800)
    # 엇갈리는 지점
    tc, vc = cross_T(cur["mla-v31"], cur["kda-k3"])
    f.raw(f'<circle cx="{sx(tc):.1f}" cy="{sy(vc):.1f}" r="8" fill="#ffffff" stroke="{mr.INK}" stroke-width="3"/>')
    f.line(sx(tc) + 6, sy(vc) + 8, sx(tc) + 30, sy(0.66e9) - 16, mr.INK, 1.2)
    f.text(sx(tc) + 20, sy(0.66e9), f"문맥 약 {tc / 1024:.0f}K에서 같아짐", 14, "start", 800)
    # state 설명
    tx, ty = sx(70000), sy(2.55e9)
    cf.lines(f, tx, ty, ["점선 = KDA + DSA의 state",
                         "문맥과 상관없이 매 스텝 전부 읽고 다시 씀",
                         f"8K에서는 읽고 쓰는 양의 {sv / b:.0%}, 128K에서도 {sv / dict(cur['kda-k3'])[131072]:.0%}"],
             14, mr.INK, lh=22)
    f.line(sx(86000), ty + 56, sx(86000), sy(sv) - 6, MCOL["kda-k3"], 1.4)
    mr.takeaway(f, PY + PH + 118, *TAKE["s2b"])
    mr.save(f)


def share_curve(mid, T=131072):
    """요청 수별 (요청 데이터 ÷ (요청 데이터 + 가중치)) — 저장량 · 읽고 쓰는 양."""
    out = []
    for B in sorted({max(1, round(2 ** (k / 4))) for k in range(45)}):
        rows = tr.items(M_(mid), B, T, "model")
        st, mv = tr.by(rows, "life", "store"), tr.by(rows, "life", "move")
        out.append((B, st["req"] / (st["req"] + st["svc"]), mv["req"] / (mv["req"] + mv["svc"])))
    return out


def fig_share():
    """4.1 — 요청이 늘 때 요청 데이터가 저장 공간 · 읽기에서 차지하는 몫 (모델별 작은 그림 5개)."""
    PX0, PWp, GAP, PY, PH = 100, 286, 38, 250, 300

    def sy(p):
        return PY + PH - PH * p
    f = mr.Fig("synth-s4a-share", W, PY + PH + 250)
    mr.title(f, "요청이 늘 때 — 요청 데이터가 저장 공간과 읽기에서 차지하는 몫 (문맥 128K)",
             "몫 = 요청마다 쌓이는 데이터 ÷ (요청 데이터 + 가중치) · 50% 점선을 넘으면 요청 데이터가 가중치보다 큼 · 가로축 = 동시 요청 수")
    x, y = 40, 134
    for c, t in ((SAVE_COL, "저장 공간 중 요청 데이터 몫"), (MOVE_COL, "스텝마다 읽고 쓰는 양 중 요청 데이터 몫")):
        f.line(x, y - 5, x + 30, y - 5, c, 4)
        f.text(x + 40, y, t, 15)
        x += 40 + mr.text_w(t, 15) + 34
    f.rect(x, y - 15, 30, 18, "#e6e9f3", "none", 3)
    f.text(x + 40, y, "두 선 사이 = 요청 데이터가 저장 공간은 차지하지만 매 스텝 읽히지는 않는 구간", 15, color=mr.MUTED)
    for i, mid in enumerate(MODELS):
        x0 = PX0 + i * (PWp + GAP)

        def sx(B, x0=x0):
            return x0 + PWp * math.log2(B) / 11
        fk = FAM_OF[mid]
        lab = dict((k, l) for k, l, _ in FAM)[fk]
        pw = mr.text_w(lab, 12) + 16
        f.rect(x0, PY - 52, pw, 22, FAM_ACC[fk], "none", 11)
        f.text(x0 + pw / 2, PY - 36, lab, 12, "middle", 800, "#ffffff")
        f.text(x0 + pw + 10, PY - 35, NAME[mid], 17, weight=800)
        for p in (0, 0.25, 0.5, 0.75, 1.0):
            if p == 0.5:
                dashed(f, x0, sy(p), x0 + PWp, sy(p), mr.INK, 1.4, "6 5")
            else:
                f.line(x0, sy(p), x0 + PWp, sy(p), mr.GRID, 1)
            if i == 0 and p in (0, 0.5, 1.0):
                f.text(x0 - 10, sy(p) + 5, f"{p:.0%}", 13, "end", color=mr.MUTED)
        for B in (1, 4, 16, 64, 256, 1024):
            f.line(sx(B), PY + PH, sx(B), PY + PH + 5, mr.AXIS, 1)
            f.text(sx(B), PY + PH + 22, str(B), 12.5, "middle", color=mr.MUTED)
        f.rect(x0, PY, PWp, PH, "none", mr.AXIS, 0, 1)
        cur = share_curve(mid)
        poly = [(sx(B), sy(s)) for B, s, _ in cur] + [(sx(B), sy(m)) for B, _, m in reversed(cur)]
        f.raw('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in poly) + '" fill="#e6e9f3"/>')
        for k, c in ((1, SAVE_COL), (2, MOVE_COL)):
            f.raw('<polyline points="' + " ".join(f"{sx(r[0]):.1f},{sy(r[k]):.1f}" for r in cur) +
                  f'" fill="none" stroke="{c}" stroke-width="3.2" stroke-linejoin="round"/>')
        cs, cm = cross_store(mid), cross_move(mid)
        for B, c in ((cs, SAVE_COL), (cm, MOVE_COL)):
            f.raw(f'<circle cx="{sx(B):.1f}" cy="{sy(0.5):.1f}" r="6" fill="{c}" stroke="#ffffff" stroke-width="2"/>')
        yb = PY + PH + 56
        if cm / cs < 1.3:
            halo(f, sx(cs), sy(0.5) - 14, f"약 {round(cs, -1):.0f}", 13, "end", 800, mr.INK)
            f.text(x0, yb, f"저장 · 읽기 모두 약 {round(cs, -1):.0f}개", 15, weight=800)
            f.text(x0, yb + 24, "저장한 KV를 매 스텝 다 읽음", 13.5, color=mr.MUTED)
        else:
            halo(f, sx(cs) - 8, sy(0.5) - 12, str(cs), 13, "end", 800, SAVE_COL)
            right = sx(cm) + 8 + mr.text_w(f"{cm:,}", 13) > x0 + PWp - 4
            halo(f, sx(cm) + (-8 if right else 8), sy(0.5) + 24, f"{cm:,}", 13, "end" if right else "start", 800, MOVE_COL)
            f.text(x0, yb, f"저장 {cs}개 → 읽기 {cm:,}개", 15, weight=800)
            f.text(x0, yb + 24, f"읽기 쪽이 {cm / cs:.1f}배 많은 요청에서 넘음", 13.5, color=mr.MUTED)
    mr.takeaway(f, PY + PH + 140, *TAKE["s4a"])
    mr.save(f)


HB_CONDS = [(32, 131072, "요청 32개 · 128K"), (256, 32768, "요청 256개 · 32K"), (256, 131072, "요청 256개 · 128K")]


def hbm_parts(mid, B, T):
    """필요한 H200 장수 (가중치 몫, 요청 데이터 몫) — 저장량 ÷ 141 GB."""
    st = tr.by(tr.items(M_(mid), B, T, "model"), "life", "store")
    return st.get("svc", 0) / 141e9, (st.get("req", 0) + st.get("step", 0)) / 141e9


def fig_hbm():
    """4.2 — H200 몇 장이 필요한가 (빨간 선 = 모델별 구성 장수)."""
    RH, FG, TOP = 50, 10, 280
    P0, PWh, GAP, MAXN = 370, 380, 70, 20
    sc = PWh / MAXN
    H = TOP + 5 * RH + 2 * FG
    f = mr.Fig("synth-s4b-hbm", W, H + 200)
    mr.title(f, "H200 몇 장이 필요한가 — 저장량 ÷ H200 1장의 HBM(141 GB)",
             "막대 끝 숫자 = 필요한 최소 장수(올림) · 빨간 선 = 모델별 구성 장수 (Full MLA · DSA 8장, KDA + DSA 16장, CSA + HCA · GDN + QSA 2장) · GPU별 배치 · 복제 · 작업 공간은 반영 안 함")
    mr.legend(f, [("가중치", mr.GRAY), ("요청마다 쌓이는 데이터 (KV · 색인 키 · state)", mr.BLUE)], 40, 134, 30)
    lx = 40 + 26 + mr.text_w("가중치", 15) + 30 + 26 + mr.text_w("요청마다 쌓이는 데이터 (KV · 색인 키 · state)", 15) + 30
    dashed(f, lx + 15, 118, lx + 15, 140, OVER, 2.6)
    f.text(lx + 30, 135, "모델별 구성 장수 — 막대가 넘으면 그 장수로는 HBM이 모자람", 15, color=mr.MUTED)
    for j, (B, T, lab) in enumerate(HB_CONDS):
        x = P0 + j * (PWh + GAP)
        f.text(x, TOP - 30, lab, 17, weight=800)
        for n in (0, 4, 8, 12, 16, 20):
            f.text(x + n * sc, H + 28, f"{n}장" if n else "0", 12.5, "middle", color=mr.MUTED)

    def guides(y0, y1, n):
        for j in range(len(HB_CONDS)):
            x = P0 + j * (PWh + GAP)
            for g in (4, 8, 12, 16, 20):
                f.line(x + g * sc, y0, x + g * sc, y1, "#d9dce5", 1)
            if n:
                dashed(f, x + n * sc, y0 + 4, x + n * sc, y1 - 4, OVER, 2.6, "6 4")
    guides(TOP - 6, H + 6, 0)

    def row(mid, y):
        n = M_(mid)["model"]["deploy"]["n_gpu"]
        guides(y, y + RH, n)
        f.text(XN + mr.text_w(NAME[mid], 15) * 1.1 + 12, y + RH / 2 + 6, f"{n}장 구성", 13, color=mr.MUTED)
        for j, (B, T, _) in enumerate(HB_CONDS):
            x = P0 + j * (PWh + GAP)
            w_, q_ = hbm_parts(mid, B, T)
            need = math.ceil(w_ + q_)
            f.rect(x, y + 12, w_ * sc, RH - 24, mr.GRAY, "none", 2)
            if q_ * sc > 0.5:
                f.rect(x + w_ * sc, y + 12, q_ * sc, RH - 24, mr.BLUE, "none", 2)
            lx_ = x + (w_ + q_) * sc + 8
            if lx_ < x + n * sc + 4 < lx_ + mr.text_w(f"{need}장", 15) * 1.3:
                lx_ = x + n * sc + 8
            f.text(lx_, y + RH / 2 + 6, f"{need}장", 15, weight=800, color=OVER if need > n else mr.INK)
    fam_rows(f, TOP, RH, FG, W - 40, row)
    mr.takeaway(f, H + 100, *TAKE["s4b"])
    mr.save(f)


# ------------------------------------------------------------------ 그림 아래 한 줄 결론 (수치는 본문과 같은 계산 — print로 확인)
TAKE = {
    "t1": ("요청이 적으면 차이는 가중치 크기뿐 — 요청 · 문맥이 모두 커져야 KV(Full MLA) · 색인 키(DSA) · state(hybrid) 차이가 드러남",
           "요청 256개 · 128K: DSA는 Full MLA보다 읽고 쓰는 양이 절반으로 줄지만 저장량은 13% 늘어남 · sparse · hybrid에서 가장 큰 몫은 여전히 MoE 가중치(66 ~ 79%)"),
    "t2": ("DSA는 KV를 그대로 저장하고 읽기만 줄임 — hybrid는 문맥과 상관없는 state가 남아 짧은 문맥에서는 Full MLA보다 많이 읽고 씀",
           "128K: DSA는 Full MLA보다 읽는 양 78% 감소 · 저장량 20% 증가 / 8K: KDA + DSA가 요청 하나당 읽고 쓰는 양이 Full MLA의 3배"),
    "t3": ("다섯 모델 모두 읽고 쓰는 양은 다음 스텝에 다시 읽힐 데이터가 거의 전부 — SRAM을 늘려 줄일 몫이 작음",
           "요청 동안 몫 — 256개 · 8K: GDN + QSA 23% · KDA + DSA 14% · Full MLA 11% · DSA 5.2% · CSA + HCA 2.6% / "
           "256개 · 128K: Full MLA 66% · DSA 29% · GDN + QSA 28% · KDA + DSA 18% · CSA + HCA 16%"),
    "t5": ("골라 읽는 KV는 저장 공간만 크게 차지하고 거의 읽히지 않음 — 읽는 양의 차이는 전부 읽는 데이터와 다시 쓰는 state가 만듦",
           "요청 256개 · 128K: 골라 읽는 KV는 저장한 것의 약 1.6%만 매 스텝 읽음 / DSA가 색인 키를 읽는 양은 Full MLA가 KV를 읽는 양의 약 1/5"),
    "s2a": ("읽는 양은 네 구조 모두 Full MLA의 2 ~ 26%로 줄지만, 저장량까지 크게 줄인 것은 압축(CSA + HCA)뿐",
            "저장량: CSA + HCA 90% 감소 · GDN + QSA 34% 감소 · KDA + DSA 14% 감소 · DSA 20% 증가 (전체 KV를 그대로 두고 색인 키를 더 저장)"),
    "s2b": ("Full MLA는 문맥에 비례해 늘고, hybrid는 state가 바닥에 깔려 거의 수평 — 짧은 문맥에서는 hybrid가 더 많이 읽고 씀",
            "문맥 8K: KDA + DSA가 Full MLA의 3배 · 약 25K에서 같아짐 · 128K: Full MLA의 26% / DSA · CSA + HCA는 색인 키 읽기만큼만 천천히 늘어남"),
    "s4a": ("골라 읽는 모델은 요청 데이터가 저장 공간을 먼저 채우고, 읽기에서는 훨씬 늦게 커짐 — 두 선이 벌어진 구간에서는 저장된 KV 대부분이 그 스텝에 읽히지 않음",
            "가중치를 넘는 요청 수 (저장 → 읽기): Full MLA 약 130개로 같음 · DSA 108 → 592 · GDN + QSA 74 → 656 · KDA + DSA 348 → 1,137 · CSA + HCA 300 → 1,372"),
    "s4b": ("요청 256개 · 128K면 다섯 모델 모두 구성 장수보다 H200이 더 필요 — 가중치만으로 구성 장수의 55 ~ 89%를 채워 요청 데이터에 남는 자리가 작음",
            "요청 256개 · 128K 필요 장수: Full MLA 15장(8장 구성) · DSA 17장(8장) · KDA + DSA 20장(16장) · GDN + QSA 9장(2장) · CSA + HCA 3장(2장)"),
}


# ------------------------------------------------------------------ 정리
SUM_ROWS = [
    ("Full MLA의 KV", "blue", "app_all", "토큰마다 추가 · 전부 읽음 — 토큰당 40 KB, 요청 수 × 문맥에 비례",
     "저장한 만큼 매 스텝 읽음", "128K에서 요청 약 130개부터 저장량 · 읽고 쓰는 양 모두 가중치를 넘음",
     "HBM 용량 · 대역폭을 함께, KV 압축 · 저정밀", "아래 계층 메모리는 읽기 속도가 모자라 효과가 제한됨", "2.5"),
    ("골라 읽는 KV (sparse · hybrid 4개)", "blue", "app_sel", "토큰마다 추가 · 골라 읽음 — 요청마다 정해진 개수(2,048 · 512)만 읽음",
     "읽는 양은 줄지만 저장량은 그대로", "저장량이 가중치를 넘는 요청 수가 읽고 쓰는 양 기준의 1/3 ~ 1/9 · DSA 저장량은 색인 키만큼 20% 늘어남",
     "전체 KV는 큰 메모리, 고를 KV만 HBM", "고르는 KV가 연속 스텝에서 겹칠수록 · 블록 단위로 옮길수록 유리", "2.1"),
    ("전부 읽는 색인 · 블록 키 (sparse · hybrid 4개)", "orange", "app_all", "토큰마다 추가 · 전부 읽음 — 토큰당 0.7 ~ 8 KB",
     "골라 읽는 모델에서 요청마다 읽는 양의 큰 몫", "요청 256개 · 128K 모델 전체 읽고 쓰는 양 중: DSA 27% · CSA + HCA 12.5% · GDN + QSA 7.4% · KDA + DSA 5.6%",
     "메모리 근처 선택 · 색인 연산 · top-k", "키 형식 · head 수에 따라 1바이트당 계산이 4 ~ 124번으로 갈림", "2.2"),
    ("다시 쓰는 state (hybrid 2개)", "green", "rmw_all", "매 스텝 전부 다시 씀 — 요청당 115 ~ 449 MB, 문맥과 무관",
     "요청이 많으면 문맥이 짧아도 읽고 쓰는 양이 남음", "요청 256개 · 8K 모델 전체 읽고 쓰는 양 중: GDN + QSA 18% · KDA + DSA 13%",
     "메모리 안에서 갱신해 왕복 제거", "이득은 갱신 연산을 메모리 쪽이 감당하는지에 달림", "2.4"),
    ("MoE 가중치 (전 모델)", "gray-l", "ro_sel", "읽기만 · 골라 읽음 — 요청이 늘면 대부분을 읽음",
     "sparse · hybrid에서는 다시 읽는 양의 대부분", "요청 32개 · 128K: 요청 데이터를 읽고 쓰는 양은 전체의 4 ~ 9%뿐 (Full MLA는 28%)",
     "가중치 읽기 속도 · 요청 수에 맞춘 배치", "attention 쪽 HW 이득의 상한 = 요청 데이터가 차지하는 비중", "-"),
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
             "줄마다 데이터 하나 · 숫자는 본문과 같은 계산 · 알약 = 기준 보고서 Ⅱ장 절")
    for x, t in zip(X, ("데이터 특성 (어느 모델 · 어떻게 읽고 쓰나)", "인사이트 (비교에서 보이는 것)", "HW로 연결")):
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


# ------------------------------------------------------------------ 기준 그림 — 종합 보고서용 쉬운 말 버전
# 모델별 보고서의 crit-c1 ~ c3는 본문이 '용량형 · 대역폭형'을 써서 그대로 두고, 같은 내용을 일반적인 말로 바꿔 따로 저장
PLAIN = [
    ("읽는 방식이 용량형 · 대역폭형을, 쓰는 방식이 쓰기 부담 · 압축 자유도를 정함",
     "읽는 방식이 '저장 공간'과 '읽기 속도' 중 무엇이 중요한지를, 쓰는 방식이 쓰기 부담을 정함"),
    ("읽는 방식 — 용량형인가 대역폭형인가", "읽는 방식 — 저장 공간이 중요한가, 읽기 속도가 중요한가"),
    ("대역폭형 — 읽기 대역폭", "읽기 속도(대역폭)가 중요"),
    ("용량형 — 큰 메모리 + 고른 것만 HBM", "저장 공간(용량)이 중요 — 큰 메모리 + 고른 것만 HBM"),
    ("기준 3. 데이터 성격", "기준 3. 읽고 쓰는 방식"),
    ("고정 크기 → 용량보다 읽기 대역폭", "크기 고정 → 저장 공간보다 읽기 속도가 문제"),
    ("동시 요청 수만큼 용량", "동시 요청 수만큼 저장 공간"),
    ("용량 — 가장 먼저 HBM을 채움", "저장 공간 — 가장 먼저 HBM을 채움"),
    ("쓰기는 작고 보관이 쌓임 → 용량", "쓰기는 작고 저장량이 쌓임 → 저장 공간"),
    ("쓰기 대역폭 — 메모리 안 갱신", "쓰기 속도 — 메모리 안에서 갱신"),
    ("매 스텝 이동량", "매 스텝 읽는 양"), ("이동이 줄어듦", "읽는 양이 줄어듦"), ("이동의 대부분", "읽는 양의 대부분"),
    ("보관이 늘어남", "저장량이 늘어남"), ("read-only", "읽기만"), ("끝에 추가", "토큰마다 추가"),
    ("요청별 · ", "요청마다 · "),
]


def plain(v):
    if isinstance(v, str):
        for a, b in PLAIN:
            v = v.replace(a, b)
        return v
    if isinstance(v, (list, tuple)):
        return type(v)(plain(x) for x in v)
    return v


for _k, _n in (("kind", "synth-crit-c1-kind"), ("life", "synth-crit-c2-life"), ("char", "synth-crit-c3-char")):
    cf.GEN["s_" + _k] = (_n,) + plain(cf.GEN[_k][1:])


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
        fig_char_model()
        fig_rel()
        fig_ctx()
        fig_share()
        fig_hbm()
        fig_sum()
        for _k in ("s_kind", "s_life", "s_char"):
            cf.gen_fig(_k)
