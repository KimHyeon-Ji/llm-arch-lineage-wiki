"""데이터 성격 보고서 그림 — 종류 · 수명 · 성격 × 모델 전체 · attention 층 1개, 요청 수 · 문맥 길이별 보관 vs 스텝당 이동.

  python report/make_traits_report.py [모델 id ...]     (기본: dsa-v32)

수치는 tools/traits.py (약식 뷰어 "데이터 성격 한눈에"와 같은 규칙). 그림 도구는 make_report.py 것을 쓴다.
그림: report/figures/<prefix>-t*.svg · .png
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

import make_report as mr  # noqa: E402
import traits as tr  # noqa: E402

BS, TS = (1, 32, 256), (8192, 32768, 131072)
WHITE, INK = "#ffffff", mr.INK
TOKEN = {"gray": mr.GRAY, "gray-l": mr.GRAY_L, "gray-ll": mr.LMH, "blue": mr.BLUE, "orange": mr.ORANGE, "green": mr.GREEN,
         "yellow": mr.YELLOW, "pink": "#e87ba4", "violet": "#6250d6"}
DARK = {mr.GRAY, mr.BLUE, mr.ORANGE, mr.GREEN, "#e87ba4", "#6250d6", "#2f4b8c", "#5b74b8"}
# 수명 = 길이 순 → 남색 한 계열의 진하기 (길수록 진함)
LIFE_COL = {"svc": "#2f4b8c", "req": "#5b74b8", "step": "#a9b8e2", "kern": "#e1e7f5"}
# 성격 = 색 계열이 쓰기 방식, 진하기가 전부 읽음(진함) / 골라 읽음(옅음)
CHAR_COL = {"ro_all": mr.GRAY, "ro_sel": mr.GRAY_L, "app_all": mr.BLUE, "app_sel": "#b7d3f6",
            "rmw": mr.GREEN, "tmp": mr.YELLOW}
PREFIX = {"dsa-v32": "dsa", "gdn-qwen36": "gdn"}


def fb(v):
    for u, s in ((1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if v >= u:
            return f"{v / u:.3g} {s}"
    return f"{v:.0f} B"


def pct(v):
    return f"{v:.0%}" if v >= 0.095 else f"{v:.1%}" if v >= 0.001 else "0.1% 미만"


def kind_cats(M, scope):
    """종류 묶음(model.yaml group) → (키, 이름, 색). 같은 색 토큰을 앞 묶음이 쓰면 더 옅은 회색."""
    cats, used = [], set()
    for r in tr.items(M, 1, 8192, scope):
        if any(c[0] == r["group"] for c in cats):
            continue
        col = TOKEN.get(r["color"], mr.GRAY_L)
        if col in used:
            col = mr.LMH
        used.add(col)
        cats.append((r["group"], r["group"], col))
    return cats


def life_cats():
    return [(k, tr.LIFE[k], LIFE_COL[k]) for k in ("svc", "req", "step")]


def char_cats(M, scope):
    seen = {r["char"] for B in BS for T in TS for r in tr.items(M, B, T, scope)}
    return [(k, tr.CHAR[k], CHAR_COL[k]) for k in tr.CHAR if k in seen]


def fig(name, M, scope, key, cats, ttl, sub, take, take_sub, chip_col=False):
    W = 1760 + (210 if chip_col else 0)
    X1, PW, TW = 236, 600, 110
    X2 = X1 + PW + TW + 30
    RH, GAP, TOP = 40, 26, 214
    H = TOP + len(BS) * (len(TS) * RH + GAP) + 120
    f = mr.Fig(name, W, H)
    mr.title(f, ttl, sub)
    mr.legend(f, [(c[1], c[2]) for c in cats], 40, 132)
    for x, t in ((X1, "보관 — HBM에 들고 있는 양"), (X2, "스텝당 이동 — HBM 읽기 + 쓰기")):
        f.text(x, TOP - 46, t, 17, weight=700)
        f.text(x, TOP - 22, "구성 비율 (합 = 100%)", 14, color=mr.MUTED)
    f.text(X1 + PW + TW, TOP - 22, "합", 14, "end", color=mr.MUTED)
    f.text(X2 + PW + TW, TOP - 22, "합", 14, "end", color=mr.MUTED)
    if chip_col:
        XC = X2 + PW + TW + 40
        f.text(XC, TOP - 46, "커널 안 (HBM 밖)", 17, weight=700)
        f.text(XC, TOP - 22, "칩 안에서만 생기는 양", 14, color=mr.MUTED)
    y = TOP
    for B in BS:
        f.text(40, y + 26, f"요청 {B}개", 18, weight=700)
        for T in TS:
            rows = tr.items(M, B, T, scope)
            for x0, field in ((X1, "store"), (X2, "move")):
                val = tr.by(rows, key, field)
                tot = sum(val.values())
                x = x0
                for ck, _, col in cats:
                    w = PW * val.get(ck, 0) / tot
                    if w > 0.25:
                        f.rect(x, y + 6, max(w - 2, 0.7), RH - 12, col, mr.BOX[1] if col == mr.LMH else "none", rx=3, sw=1)
                        lab = pct(val[ck] / tot)
                        if w >= mr.text_w(lab, 13.5) + 10:
                            f.text(x + w / 2 - 1, y + RH / 2 + 5, lab, 13.5, "middle", 700, WHITE if col in DARK else INK)
                    x += w
                f.text(x0 + PW + TW, y + RH / 2 + 6, fb(tot), 15, "end", 700)
            f.text(170, y + RH / 2 + 6, f"T {T // 1024}K", 15, color=mr.MUTED)
            if chip_col:
                c = tr.chip(M, B, T, scope)["bytes"]
                mv = sum(r["move"] for r in rows)
                f.text(XC, y + RH / 2 + 6, fb(c), 15, weight=700)
                f.text(XC + 82, y + RH / 2 + 6, f"이동의 {c / mv:.2g}배" if c >= mv else f"이동의 {c / mv:.0%}", 14, color=mr.MUTED)
            y += RH
        f.line(40, y + GAP / 2 - 2, W - 40, y + GAP / 2 - 2, mr.GRID, 1)
        y += GAP
    mr.takeaway(f, y + 36, take, take_sub)
    mr.save(f)


def make(mid):
    M = tr.load(mid)
    p, mdl = PREFIX.get(mid, mid), M["model"]["model"]
    base = f"{mdl} · decode 1스텝 · H200 이론 하한 · 저장 형식 = algo.yaml 기본값"
    T = TAKE[mid]
    for scope, sname, n0 in (("model", "모델 전체", 1), ("layer", M["model"].get("layer_label", "attention 층 1개"), 2)):
        sub = base + (" · MoE expert 읽기는 균등 라우팅 가정" if scope == "model" and "E_hit" in M["model"].get("derived", {}) else "")
        fig(f"{p}-t{n0}-kind-{scope}", M, scope, "group", kind_cats(M, scope),
            f"{mdl} — 데이터 종류별 보관 · 이동 ({sname})", sub, *T[("kind", scope)])
        fig(f"{p}-t{n0 + 2}-life-{scope}", M, scope, "life", life_cats(),
            f"{mdl} — 수명별 보관 · 이동 ({sname})", sub, *T[("life", scope)], chip_col=True)
        fig(f"{p}-t{n0 + 4}-char-{scope}", M, scope, "char", char_cats(M, scope),
            f"{mdl} — 데이터 성격별 보관 · 이동 ({sname})", sub, *T[("char", scope)])


# 그림 아래 한 줄 결론 (수치는 보고서 본문과 같은 값)
TAKE = {
    "dsa-v32": {
        ("kind", "model"): ("보관은 MoE 가중치가, 요청 · 문맥이 크면 KV cache가 채움 — 이동은 거의 모든 조건에서 MoE 가중치",
                            "요청 256개 · 128K: 보관 2.29 TB 중 KV cache 59% · 이동 997 GB 중 MoE · FFN 가중치 66% · 색인 K cache 27%"),
        ("kind", "layer"): ("attention 층 안에서 보관은 KV cache, 이동은 색인 K cache — 같은 요청별 cache인데 역할이 반대",
                            "요청 32개 · 128K: 보관 3.52 GB 중 KV cache 78% · 이동 909 MB 중 색인 K cache 61%"),
        ("life", "model"): ("HBM 이동의 대부분은 서비스 내내 남는 가중치 — 칩 안에서만 생겼다 사라지는 값은 요청 · 문맥과 함께 커짐",
                            "요청 256개 · 128K: 칩 안에서만 생기는 값 549 GB = HBM 이동 997 GB의 55%"),
        ("life", "layer"): ("층 1개로 보면 요청 동안 남는 cache가 보관의 대부분 · 스텝 안 임시값이 이동의 7 ~ 25%",
                            "요청 256개 · 128K: 칩 안에서만 생기는 값 9 GB > HBM 이동 5.5 GB — 색인 head별 점수를 칩 안에서 줄이기 때문"),
        ("char", "model"): ("보관의 대부분은 '골라 읽는' 데이터(MoE · KV) — 매 스텝 전부 읽는 요청별 데이터는 색인 K cache뿐",
                            "요청 256개 · 128K: 골라 읽는 데이터 = 보관의 88% · 이동의 68%, 색인 K cache = 보관 12% · 이동 27%"),
        ("char", "layer"): ("attention 층에서는 골라 읽는 KV가 보관을, 전부 읽는 색인 K가 이동을 차지 — 용량형과 대역폭형이 갈림",
                            "요청 256개 · 128K: KV cache 보관 83% · 이동 6% / 색인 K cache 보관 17% · 이동 80%"),
    },
}


if __name__ == "__main__":
    for mid in sys.argv[1:] or ["dsa-v32"]:
        make(mid)
