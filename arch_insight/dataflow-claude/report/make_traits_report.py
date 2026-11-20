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
PREFIX = {"dsa-v32": "dsa", "gdn-qwen36": "gdn", "mla-v31": "mla", "csa-v4flash": "v4", "kda-k3": "k3", "qsa-qwen38": "q38"}


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
    if take:
        mr.takeaway(f, y + 36, take, take_sub)
    mr.save(f)


def make(mid):
    M = tr.load(mid)
    p, mdl = PREFIX.get(mid, mid), M["model"]["model"]
    base = f"{mdl} · decode 1스텝 · H200 이론 하한 · 저장 형식 = algo.yaml 기본값"
    T = TAKE.get(mid, {})
    T = {k: T.get(k, ("", None)) for k in [(a, s) for a in ("kind", "life", "char") for s in ("model", "layer")]}
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
    "mla-v31": {
        ("kind", "model"): ("full attention은 요청 · 문맥이 커지면 KV cache가 보관과 이동을 함께 늘림 — 요청 256개 · 128K에서는 이동의 3분의 2",
                            "요청 256개 · 128K: 보관 2.02 TB 중 KV cache 67% · 이동 2.03 TB 중 KV cache 66%"),
        ("kind", "layer"): ("MLA 층은 요청 · 문맥이 커지면 KV cache가 층 보관 · 이동의 거의 전부",
                            "요청 32개 · 128K: 층 이동 2.98 GB 중 KV cache 92%"),
        ("life", "model"): ("요청 동안 남는 KV가 이동에서도 커짐 — 문맥 128K는 요청 128개부터 요청별 이동 > 가중치 이동",
                            "칩 안에서만 생기는 값(attention 점수 · 확률) = 이동의 최대 78% (요청 256개 · 128K)"),
        ("life", "layer"): ("MLA 층에서는 칩 안 값이 HBM 이동만큼 커짐 — head 128개의 점수를 칩 안에서 바로 줄이기 때문",
                            "요청 256개 · 128K: 칩 안 값 = 층 HBM 이동의 1.15배"),
        ("char", "model"): ("골라 읽는 데이터는 MoE뿐 — KV cache는 '끝에 추가 · 전부 읽음'이라 용량과 대역폭을 함께 씀",
                            "요청 32개 · 128K: KV cache 보관 20% · 이동 28%"),
        ("char", "layer"): ("층 안에서는 전부 읽는 KV가 보관 · 이동 모두 대부분 — 용량형과 대역폭형이 갈리지 않음",
                            "요청 32개 · 128K: KV cache 보관 93% · 이동 92%"),
    },
    "csa-v4flash": {
        ("kind", "model"): ("압축으로 KV가 작아져 모델 전체 이동은 거의 MoE 가중치 — 요청이 적을 때는 attention 가중치가 MoE만큼 큼",
                            "요청 1개: 이동 10.7 GB 중 attention 가중치 47% · MoE 43% / 요청 256개 · 128K: 이동 174 GB 중 MoE 85%"),
        ("kind", "layer"): ("CSA 층은 보관은 압축 KV, 이동은 요청 · 문맥이 크면 색인 K cache",
                            "요청 256개 · 128K: 층 보관 5.6 GB 중 압축 KV 88% · 이동 0.90 GB 중 색인 K cache 59%"),
        ("life", "model"): ("이동의 대부분은 서비스 내내 남는 가중치 — 요청별 cache 이동은 요청 256개 · 128K에서도 작음",
                            "요청 256개 · 128K: 요청 동안 남는 cache 이동 = 전체의 약 10%"),
        ("life", "layer"): ("CSA 층은 전체 이동이 작아 스텝 안 임시값 비중이 커짐 · 칩 안 값이 HBM 이동의 2배를 넘음",
                            "요청 256개 · 128K: 임시값 19% · 칩 안 값 = 층 HBM 이동의 2.5배"),
        ("char", "model"): ("보관 · 이동 모두 read-only로 골라 읽는 MoE가 대부분 — 압축 KV는 보관만 늘고 이동은 거의 없음",
                            "요청 32개 · 128K: 골라 읽는 KV 보관 7.7% · 이동 0.2%"),
        ("char", "layer"): ("CSA 층에서도 골라 읽는 압축 KV는 용량형, 전부 읽는 색인 K는 대역폭형",
                            "요청 32개 · 128K: 압축 KV 보관 77% · 이동 4% / 색인 K 보관 8% · 이동 31%"),
    },
    "kda-k3": {
        ("kind", "model"): ("요청이 적을 때는 BF16 attention 가중치가 이동의 절반 — 요청 · 문맥이 크면 MLA 24층의 KV와 KDA state가 커짐",
                            "요청 1개: attention 가중치 51% / 요청 256개 · 128K: KV cache 32% · state 8.5%"),
        ("kind", "layer"): ("KDA 층은 문맥 길이와 무관 — 요청이 늘수록 state 읽기 · 쓰기가 층 이동의 대부분",
                            "요청 256개: 층 이동 4.3 GB 중 state 74%"),
        ("life", "model"): ("요청 동안 남는 데이터가 두 종류 — 문맥에 비례하는 MLA KV와 요청 수에만 비례하는 KDA state",
                            "요청 256개 · 128K: 요청별 이동 = KV 32% + state 8.5%"),
        ("life", "layer"): ("KDA 층 이동은 요청 동안 남는 state가 대부분 — 칩 안 값은 작음",
                            "요청 256개: state 이동 74% · 칩 안 값 = 층 이동의 0.39배"),
        ("char", "model"): ("'매 스텝 전부 다시 씀'(state)이 처음 등장 — 보관 비중보다 이동 비중이 2배",
                            "요청 256개 · 8K: state 보관 6% · 이동 12%"),
        ("char", "layer"): ("KDA 층은 read-only 가중치와 매 스텝 다시 쓰는 state 두 가지 — 요청이 늘면 state가 이동을 차지",
                            "요청 32개: 다시 쓰는 데이터 보관 19% · 이동 32%"),
    },
    "qsa-qwen38": {
        ("kind", "model"): ("보관은 요청 · 문맥이 크면 KV cache가 가장 크지만, 이동은 MoE 가중치 다음이 GDN state",
                            "요청 256개 · 128K: 보관 1.13 TB 중 KV 73% · 이동 346 GB 중 MoE 70% · state 17% · KV 4%"),
        ("kind", "layer"): ("QSA 층은 보관은 KV cache, 이동은 블록 키 — 블록 키는 4토큰을 대표해도 매번 전부 읽힘",
                            "요청 32개 · 128K: 층 보관 9.0 GB 중 KV 96% · 이동 0.52 GB 중 블록 키 52%"),
        ("life", "model"): ("요청 동안 남는 데이터 중 보관은 KV, 이동은 state가 큼",
                            "요청 256개 · 128K: KV 보관 73% · 이동 4% / state 보관 2.6% · 이동 17%"),
        ("life", "layer"): ("QSA 층은 칩 안 값과 임시값이 작음 — 블록 점수 head가 4개뿐이라 칩 안에서 줄일 값도 작음",
                            "칩 안 값 = 층 이동의 0.06배 이하"),
        ("char", "model"): ("골라 읽는 KV는 보관의 대부분, 매 스텝 다시 쓰는 state는 이동의 6 ~ 18%",
                            "요청 32개 · 128K: KV 보관 29% · 이동 1.2% / state 보관 1% · 이동 5.6%"),
        ("char", "layer"): ("QSA 층에서 골라 읽는 KV는 용량형, 전부 읽는 블록 키는 대역폭형 — DSA와 같은 구조",
                            "요청 32개 · 128K: KV 보관 96% · 이동 26% / 블록 키 보관 3% · 이동 52%"),
    },
}


if __name__ == "__main__":
    for mid in sys.argv[1:] or ["dsa-v32"]:
        make(mid)
