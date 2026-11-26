"""'데이터를 나누는 세 가지 기준' 그림 — 모델마다 종류 · 수명 · 성격을 한 장씩, 그 모델의 데이터를 칸에 넣어 그림.

  python report/make_criteria_figs.py [모델 id ...]      (기본: 보고서 5개 모델 + 전체 성격 지도)

c1 종류 = 무엇에 비례해 커지나 (공유 가중치 · 요청별 크기 고정 · 요청별 문맥 비례)
c2 수명 = 얼마나 오래 남고 언제 다시 읽히나 → 둘 수 있는 곳
c3 성격 = 어떻게 쓰고 읽나 (쓰기 방식 × 읽기 방식) → 필요한 메모리 특성
분류는 tools/traits.py 규칙 그대로. 그림: report/figures/<prefix>-c1 ~ c3-*.svg · .png, synth-c3-char-map
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)

import make_report as mr  # noqa: E402
import make_traits_report as mt  # noqa: E402
import traits as tr  # noqa: E402

W = 1760
INK, MUTED, LINE, BOXF = mr.INK, mr.MUTED, "#dfe2ec", "#f6f7fb"
NAVY, WRITE, READ, IDLE = "#4a5573", "#eb6834", "#6b7080", "#c9ccd6"
MODELS = ["mla-v31", "dsa-v32", "csa-v4flash", "kda-k3", "qsa-qwen38"]
TAG = {"mla-v31": "V3.1", "dsa-v32": "V3.2", "csa-v4flash": "V4", "kda-k3": "K3.1", "qsa-qwen38": "Q3.8"}
REF = (32, 131072)          # 칸에 넣을 데이터를 고르는 조건 (임시값 · 칩 안 값이 다 보이는 조건)


def col(c):
    return mt.TOKEN.get(c, mr.GRAY_L)


def fsize(v):
    for u, s in ((1e12, "TB"), (1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if v >= u:
            return f"{v / u:.3g} {s}"
    return f"{v:.0f} B"


def wrap(s, size, width):
    """글자 폭 추정으로 줄바꿈 (공백 기준, 공백이 없으면 글자 단위)."""
    out, cur = [], ""
    for w in s.split(" "):
        t = (cur + " " + w).strip()
        if mr.text_w(t, size) <= width:
            cur = t
        else:
            if cur:
                out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def chip(f, x, y, fill, name, sub=None, dashed=False, size=15):
    """색 네모 + 이름 (+ 옅은 보조 글). 그린 폭을 돌려줌."""
    if dashed:
        f.raw(f'<rect x="{x:.1f}" y="{y - 13:.1f}" width="15" height="15" rx="3" fill="none" stroke="{READ}" '
              f'stroke-width="1.5" stroke-dasharray="3 2"/>')
    else:
        f.rect(x, y - 13, 15, 15, fill, IDLE if fill in (mr.LMH, mr.GRAY_L) else "none", 3, 1)
    f.text(x + 23, y, name, size, color=INK, weight=600)
    w = 23 + mr.text_w(name, size)
    if sub:
        f.text(x + w + 8, y, sub, size - 2, color=MUTED)
        w += 8 + mr.text_w(sub, size - 2)
    return w


def flow(f, x0, y0, width, chips, lh=30, draw=True):
    """칩을 폭 안에서 흘려 배치. 마지막 y를 돌려줌."""
    x, y = x0, y0
    for c in chips:
        w = 23 + mr.text_w(c[1], 15) + (8 + mr.text_w(c[2], 13) if c[2] else 0)
        if x > x0 and x + w > x0 + width:
            x, y = x0, y + lh
        if draw:
            chip(f, x, y, c[0], c[1], c[2], c[3] if len(c) > 3 else False)
        x += w + 22
    return y


def lines(f, x, y, ls, size=14, color=MUTED, weight=400, lh=21):
    for s in ls:
        f.text(x, y, s, size, color=color, weight=weight)
        y += lh
    return y


def items_of(M):
    """REF 조건의 항목 + 커지는 조건(토큰당 · 요청당 바이트)."""
    B, T = REF
    rows = {r["id"]: r for r in tr.items(M, B, T, "model")}
    a = {r["id"]: r for r in tr.items(M, 1, T // 2, "model")}
    b = {r["id"]: r for r in tr.items(M, 1, T, "model")}
    c = {r["id"]: r for r in tr.items(M, 2, T, "model")}
    for k, r in rows.items():
        r["tok"] = (b[k]["store"] - a[k]["store"]) / (T // 2)
        r["req"] = c[k]["store"] - b[k]["store"]
        r["w1"] = b[k]["store"]
    return list(rows.values())


def box(f, x, y, w, h, head, tag=None, accent=NAVY):
    f.rect(x, y, w, h, BOXF, LINE, 14, 1.2)
    f.rect(x, y, 6, h, accent, "none", 3)
    f.text(x + 24, y + 38, head, 20, weight=800)
    if tag:
        tw = mr.text_w(tag, 14) + 22
        f.rect(x + w - tw - 18, y + 20, tw, 26, "#ece8fb", "none", 13)
        f.text(x + w - tw / 2 - 18, y + 38, tag, 14, "middle", 700, "#5b47b0")


# ------------------------------------------------------------------ 기준 그림 — 세 장 같은 양식: 분류 | 데이터 | HW 대응
MCOL = {"mla-v31": "#4a5573", "dsa-v32": "#2a78d6", "csa-v4flash": "#6250d6", "kda-k3": "#1baf7a", "qsa-qwen38": "#eb6834"}
KIND_ROWS = [("w", "공유 가중치", "요청 · 문맥과 무관", mr.GRAY),
             ("fx", "요청별 · 크기 고정", "요청 수에 비례", mr.GREEN),
             ("ctx", "요청별 · 문맥에 비례", "요청 수 × 문맥 길이에 비례", mr.BLUE)]
LIFE = [("svc", "서비스 내내"), ("req", "요청 동안"), ("step", "스텝 안"), ("kern", "커널 안")]
LIFE_RULE = {"svc": "모든 요청이 공유", "req": "한 스텝 뒤에 다시 읽힘", "step": "다음 커널이 읽음", "kern": "같은 커널 안에서 끝남"}
CHAR_ROWS = [("ro_all", "read-only", "전부 읽음"), ("ro_sel", "read-only", "골라 읽음"),
             ("app_all", "끝에 추가", "전부 읽음"), ("app_sel", "끝에 추가", "골라 읽음"),
             ("ring_all", "고정 크기 · 덮어씀", "전부 읽음"), ("rmw_all", "매 스텝 전부 다시 씀", "전부 읽음"),
             ("tmp", "임시", "커널 사이")]
LH = 34

# 분류마다 대응되는 HW (제목, 보조, 기준 보고서 Ⅱ장 절) — 모델별로 다른 곳만 OVR
HW = {
    ("kind", "w"): ("가중치 대역폭 · 저정밀", "요청이 적을 때 이동의 대부분 · 요청이 많을수록 여러 요청이 나눠 읽음", None),
    ("kind", "fx"): ("요청 수만큼 용량 · 쓰기 대역폭", "문맥과 무관 → 짧은 문맥 · 많은 요청에서 비중이 가장 큼", "2.4"),
    ("kind", "ctx"): ("용량 — 큰 메모리 계층 · KV 압축", "요청 · 문맥이 함께 크면 가장 먼저 용량을 채움", "2.1"),
    ("life", "svc"): ("HBM 상주 · GPU 여러 장에 나눠 둠", "쓰기 없음 → 저정밀 · 복제가 자유로움", None),
    ("life", "req"): ("HBM 또는 그 아래 큰 메모리", "한 스텝 사이 모델 전체가 지나가 칩 안(L2 50 MiB)에 못 남음", "2.1"),
    ("life", "step"): ("L2 · 커널 합치기", "HBM 왕복을 없앨 수 있는 몫", None),
    ("life", "kern"): ("SMEM · 레지스터 — 이미 칩 안", "처리를 메모리 쪽으로 옮기면 이 값도 그쪽에서 만들고 줄여야 함", "2.3"),
    ("char", "ro_all"): ("읽기 대역폭 · 저정밀", "쓰기가 없어 압축 · 복제 · 배치가 자유로움", None),
    ("char", "ro_sel"): ("요청 수에 맞춘 배치", "요청이 적으면 일부 · 많으면 거의 전부 읽음", None),
    ("char", "app_all"): ("읽기 대역폭 — 메모리 근처 처리", "보관이 작아도 매 스텝 전부 읽음 (대역폭형)", "2.2"),
    ("char", "app_sel"): ("용량 — 큰 메모리 + 고른 것만 HBM", "보관은 크고 이동은 작음 (용량형)", "2.1"),
    ("char", "ring_all"): ("작은 고정 버퍼 — 별도 대응 불필요", "문맥과 무관 · 크기 작음", None),
    ("char", "rmw_all"): ("쓰기 대역폭 — 메모리 안 갱신", "읽은 만큼 다시 씀 · 문맥과 무관", "2.4"),
    ("char", "tmp"): ("L2 · 커널 합치기", "다음 커널이 바로 읽음", None),
}
OVR = {
    "mla-v31": {
        ("kind", "ctx"): ("용량 + 대역폭 함께 — HBM · KV 저정밀", "보관한 만큼 매 스텝 읽음 → 아래 계층 메모리는 읽기 대역폭이 모자람", "2.5"),
        ("life", "kern"): ("SMEM · 레지스터 — 이미 칩 안", "attention 점수 · 확률 — 연산 장치 안에서 끝냄", None),
        ("char", "app_all"): ("용량 + 대역폭 함께 — HBM · KV 저정밀",
                              "KV가 크고 매 스텝 전부 읽음 · attention 연산도 커서(KV 1 B당 약 420 FLOP) 메모리 근처로 옮기기 불리", "2.5"),
    },
    "dsa-v32": {
        ("life", "kern"): ("SMEM · 레지스터 — 이미 칩 안", "색인 head별 점수 · 색인 처리를 옮기면 head 합 · top-k까지 함께 옮겨야 함", "2.3"),
        ("char", "app_all"): ("읽기 대역폭 — 메모리 근처 선택", "색인 K cache: 보관은 작아도 매 스텝 전부 읽음 · 키 1 B당 약 124 FLOP", "2.2"),
    },
    "csa-v4flash": {
        ("kind", "fx"): ("작은 고정 버퍼 — 별도 대응 불필요", "압축 때문에 생긴 최근 창 · 압축기 state", None),
        ("kind", "ctx"): ("압축이 용량 문제를 먼저 줄임", "토큰당 바이트가 작아 큰 메모리 계층은 후순위", "2.1"),
        ("char", "app_all"): ("읽기 대역폭 — 메모리 근처 선택", "색인 K는 형식에 민감 (FP8 132 B / MXFP4 68 B)", "2.2"),
    },
    "kda-k3": {
        ("kind", "ctx"): ("용량 — 큰 메모리 계층", "DSA KV(24층)는 보관만 큼 · 요청당 2,048줄만 읽음", "2.1"),
        ("life", "req"): ("HBM — 또는 메모리 안 갱신", "state는 한 스텝 뒤에 다시 읽고 다시 씀 → 칩 안에 못 남음", "2.4"),
        ("char", "app_all"): ("읽기 대역폭 — 메모리 근처 선택", "색인 K: 보관은 작아도 매 스텝 전부 읽음 · 키 1 B당 약 62 FLOP (색인 head 32)", "2.2"),
        ("char", "rmw_all"): ("쓰기 대역폭 — 메모리 안 갱신", "이득은 감쇠 · rank-1 갱신을 메모리 쪽이 감당하는지에 달림", "2.4"),
    },
    "qsa-qwen38": {
        ("char", "app_all"): ("읽기 대역폭 — 메모리 근처 선택", "블록 키: 키 1 B당 약 4 FLOP — 메모리 쪽 연산 요구가 작음", "2.2"),
        ("char", "app_sel"): ("용량 — 큰 메모리 + 고른 블록만 HBM", "블록(4줄 연속) 단위라 전송이 덜 흩어짐", "2.1"),
    },
}
# 이 모델에서 중요한 분류 (보고서 '이걸로 알 수 있는 것'이 짚는 칸) — HW 칸을 보라로 강조
HL = {
    "mla-v31": {"kind": {"ctx"}, "life": {"svc", "req"}, "char": {"app_all"}},
    "dsa-v32": {"kind": {"ctx"}, "life": {"kern"}, "char": {"app_all", "app_sel"}},
    "csa-v4flash": {"kind": {"ctx", "fx"}, "life": {"svc"}, "char": {"app_all", "app_sel", "ring_all"}},
    "kda-k3": {"kind": {"fx", "ctx"}, "life": {"req"}, "char": {"rmw_all", "app_sel", "app_all"}},
    "qsa-qwen38": {"kind": {"ctx"}, "life": {"step", "kern"}, "char": {"app_sel", "app_all", "rmw_all"}},
    "_synth": {"kind": {"fx", "ctx"}, "life": {"req"}, "char": {"app_all", "app_sel", "rmw_all"}},
}
HL_FILL, HL_LINE, PILL = "#f3f1fc", "#7b61c9", "#7b61c9"


def cell_of(ch):
    """traits 성격 키 → 행 키 (덮어씀 · 다시 씀은 '전부 읽음' 행)."""
    return {"ring": "ring_all", "rmw": "rmw_all"}.get(ch, ch)


def char_col(k):
    return mt.CHAR_COL.get({"ring_all": "ring", "rmw_all": "rmw"}.get(k, k), "#ffffff")


def fh(chips, width):
    return flow(None, 0, 0, width, chips, LH, draw=False) if chips else 0


def head(f, ttl):
    f.text(40, 56, ttl, 28, weight=800)


XL, WL = 40, 330
XD, WD = 386, 600
XH = 1030
WH = W - 40 - XH


def hw_of(mid, crit, k):
    return OVR.get(mid, {}).get((crit, k), HW[(crit, k)])


def crit_fig(name, ttl, rows, dhead="이 모델의 데이터"):
    """행 = 분류 하나: [분류 · 규칙] [데이터 칩] → [HW 대응 · Ⅱ장 절]. 빈 분류도 흐리게 남겨 모델끼리 같은 줄에 맞춤."""
    def hw_lines(r):
        h1, h2, sec = r["hw"]
        tw = WH - 52 - (mr.text_w(SECS[sec], 13) + 44 if sec else 0)
        return wrap(h1, 17, tw), wrap(h2, 14, tw) if h2 else []
    def rh(r):
        if not r["chips"]:
            return 72
        a, b = hw_lines(r)
        return max(88, fh(r["chips"], WD - 56) + 54, 26 * len(a) + 21 * len(b) + 40)
    TOP = 150
    f = mr.Fig(name, W, TOP + sum(rh(r) + 12 for r in rows) + 30)
    head(f, ttl)
    for x, t in ((XL, "분류"), (XD, dhead), (XH, "HW 대응")):
        f.text(x + 4, TOP - 22, t, 17, weight=800, color=MUTED)
    who = "이 모델에서" if dhead == "이 모델의 데이터" else "모델 비교에서"
    f.text(W - 40, TOP - 22, f"보라 칸 = {who} 짚는 분류 · 오른쪽 끝 = 기준 보고서 Ⅱ장 절" if any(r.get("hl") for r in rows)
           else "오른쪽 끝 = 기준 보고서 Ⅱ장 절", 14, "end", 600, MUTED)
    y = TOP
    for r in rows:
        h = rh(r)
        on = bool(r["chips"])
        hl = on and r.get("hl")
        # 분류
        f.rect(XL, y, WL, h, "#ffffff", LINE, 12, 1.2)
        f.rect(XL, y, 10, h, r["acc"] if on else "#eceef3", LINE if r["acc"] in ("#e1e7f5", mr.LMH) else "none", 4, 1)
        f.text(XL + 30, y + h / 2 - 3, r["label"], 21, weight=800, color=INK if on else IDLE)
        f.text(XL + 30, y + h / 2 + 23, r["rule"], 15, weight=700, color=MUTED if on else IDLE)
        # 데이터
        f.rect(XD, y, WD, h, BOXF if on else "#fbfbfc", LINE if on else "#eef0f5", 12, 1.2)
        if on:
            flow(f, XD + 28, y + (h - fh(r["chips"], WD - 56)) / 2 + 6, WD - 56, r["chips"], LH)
        else:
            f.text(XD + 28, y + h / 2 + 6, "없음", 16, color=IDLE, weight=700)
        # HW
        a, b = hw_lines(r)
        f.rect(XH, y, WH, h, HL_FILL if hl else ("#ffffff" if on else "#fbfbfc"),
               HL_LINE if hl else (LINE if on else "#eef0f5"), 12, 2 if hl else 1.2)
        if on:
            arrow(f, XD + WD + 8, XH - 8, y + h / 2, HL_LINE if hl else IDLE)
        yy = y + (h - (26 * len(a) + 21 * len(b))) / 2 + 18
        for ln in a:
            f.text(XH + 26, yy, ln, 17, weight=800 if hl else 700, color=INK if on else IDLE)
            yy += 26
        for ln in b:
            f.text(XH + 26, yy - 3, ln, 14, color=MUTED if on else IDLE)
            yy += 21
        sec = r["hw"][2]
        if sec:
            tag = SECS[sec]
            pw = mr.text_w(tag, 13) + 20
            px = XH + WH - 22 - pw
            f.rect(px, y + h / 2 - 13, pw, 26, (PILL if hl else "#a99bd8") if on else "#e4e5ea", "none", 13)
            f.text(px + pw / 2, y + h / 2 + 5, tag, 13, "middle", 700, "#ffffff")
        y += h + 12
    mr.save(f)


def kind_fig(name, ttl, cols, mid=None, dhead="이 모델의 데이터"):
    hl = HL.get(mid, {}).get("kind", set())
    crit_fig(name, ttl, [dict(label=lab, rule=rule, acc=acc, chips=ch, hw=hw_of(mid, "kind", k), hl=k in hl)
                         for (k, lab, rule, acc), ch in zip(KIND_ROWS, cols)], dhead)


def life_fig(name, ttl, lanes, mid=None, dhead="이 모델의 데이터"):
    hl = HL.get(mid, {}).get("life", set())
    crit_fig(name, ttl, [dict(label=nm, rule=LIFE_RULE[k], acc=mt.LIFE_COL[k], chips=lanes.get(k), hw=hw_of(mid, "life", k),
                              hl=k in hl) for k, nm in LIFE], dhead)


def char_fig(name, ttl, cells, mid=None, dhead="이 모델의 데이터"):
    hl = HL.get(mid, {}).get("char", set())
    crit_fig(name, ttl, [dict(label=wn, rule=rn, acc=char_col(k), chips=cells.get(k), hw=hw_of(mid, "char", k), hl=k in hl)
                         for k, wn, rn in CHAR_ROWS], dhead)


def fig_kind(mid):
    M = tr.load(mid)
    rows = items_of(M)
    ws = sorted([r for r in rows if r["life"] == "svc"], key=lambda r: -r["store"])
    fx = [r for r in rows if r["life"] == "req" and r["tok"] <= 0]
    tk = [r for r in rows if r["life"] == "req" and r["tok"] > 0]
    kind_fig(f"{mt.PREFIX[mid]}-c1-kind", f"{M['model']['model']} — 데이터 종류",
             [[(col(r["color"]), r["name"], fsize(r["store"])) for r in ws],
              [(col(r["color"]), r["name"], f"요청당 {fsize(r['req'])}") for r in fx],
              [(col(r["color"]), r["name"], f"토큰당 {fsize(r['tok'])}") for r in tk]], mid)


def fig_life(mid):
    M = tr.load(mid)
    rows = items_of(M)
    lanes = {}
    for r in sorted(rows, key=lambda r: -r["store"]):
        lanes.setdefault(r["life"], []).append((col(r["color"]), r["name"], None))
    lanes["kern"] = [(None, x[0], None, True) for x in tr.chip(M, *REF, "layer")["top"][:5]]
    life_fig(f"{mt.PREFIX[mid]}-c2-life", f"{M['model']['model']} — 데이터 수명", lanes, mid)


def fig_char(mid):
    M = tr.load(mid)
    cells = {}
    for r in sorted(items_of(M), key=lambda r: -r["store"]):
        cells.setdefault(cell_of(r["char"]), []).append((col(r["color"]), r["name"], None))
    char_fig(f"{mt.PREFIX[mid]}-c3-char", f"{M['model']['model']} — 데이터 성격", cells, mid)


def fig_char_map():
    """전체 모델 성격 지도 — 행마다 어느 모델의 어떤 데이터가 있나. 가중치 행은 전 모델 공통으로 한 줄."""
    cells = {"ro_all": [(mr.GRAY, "전 모델", "attention · LM head 가중치")],
             "ro_sel": [(mr.GRAY_L, "전 모델", "MoE expert · 임베딩")],
             "tmp": [(mr.YELLOW, "전 모델", "q · 점수 · 선택 위치 등")]}
    for mid in MODELS:
        per = {}
        for r in sorted(items_of(tr.load(mid)), key=lambda r: -r["store"]):
            if r["life"] == "req":
                per.setdefault(cell_of(r["char"]), []).append(r["group"])
        for k, gs in per.items():
            cells.setdefault(k, []).append((MCOL[mid], TAG[mid], " · ".join(dict.fromkeys(gs))))
    char_fig("synth-c3-char-map", "모델 5개 — 데이터 성격", cells, "_synth", "모델별 데이터")


# ------------------------------------------------------------------ 정리 — 특성 → 인사이트 → HW
SECS = {"2.1": "2.1 계층형 KV", "2.2": "2.2 메모리 근처 선택", "2.3": "2.3 색인 연산 · top-k", "2.4": "2.4 state PIM",
        "2.5": "2.5 이종 가속기 · PD 분리", "-": "기준 보고서 범위 밖", "x": "대응 대상 아님"}
SEC_NOTE = "알약 = 기준 보고서 Ⅱ장 절"      # 종합 보고서는 make_synthesis_report.plain_secs()로 바꿔 그림
# 행 = (데이터, 색, 성격 칸, 특성, 인사이트 제목, 인사이트 보조, HW 제목, HW 보조, 절)
SUMMARY = {
    "mla-v31": [
        ("KV cache", "blue", "app_all", "끝에 추가 · 전부 읽음 — 토큰당 40 KB(61층), 요청 × 문맥에 비례",
         "보관한 만큼 매 스텝 읽음", "용량과 대역폭이 같이 커짐 · 128K는 요청 126개부터 요청별 이동 > 가중치 이동",
         "HBM 용량 · 대역폭을 함께, KV 압축 · 저정밀", "아래 계층 메모리는 읽기 대역폭이 모자라 효과 제한", "2.5"),
        ("MoE expert", "gray-l", "ro_sel", "read-only · 골라 읽음 — 요청이 늘면 사실상 전부 읽음",
         "요청 수에 따라 성격이 바뀜", "요청 적을 때 일부 · 많을 때 거의 전부 읽음 (균등 라우팅 가정)",
         "요청 수에 맞춘 가중치 배치", "고정 배치 하나로는 두 경우를 못 맞춤", "-"),
        ("attention 점수 · 확률", None, "kern", "커널 안 값 — 층 HBM 이동만큼 큼(논리 크기 합)",
         "MLA attention은 연산도 큼", "head 128개가 KV 1줄 공유 → KV 1 B당 약 420 FLOP",
         "고대역 메모리 장치 + 연산 장치 분리", "attention을 메모리 근처로 옮기는 방향은 불리", "2.5"),
    ],
    "dsa-v32": [
        ("KV cache", "blue", "app_sel", "끝에 추가 · 골라 읽음 — 요청당 2,048줄만 읽음",
         "보관만 크고 이동은 작은 용량형", "요청 256개 · 128K: 층 보관 83% · 층 이동 6%",
         "전체 KV는 큰 메모리, 고를 KV만 HBM", "연속 스텝의 선택 겹침이 클수록 유리", "2.1"),
        ("색인 K cache", "orange", "app_all", "끝에 추가 · 전부 읽음 — 토큰당 8 KB, 요청 × 문맥에 비례",
         "보관은 작아도 이동을 차지하는 대역폭형", "요청 256개 · 128K: 층 이동의 81%",
         "색인 읽기를 메모리 근처에서, 칩에는 선택 위치만", "학습된 DSA 색인 실행 HW는 미확인", "2.2"),
        ("색인 head별 점수", None, "kern", "커널 안 값 — 층 HBM 이동보다 큰 조건이 있음",
         "칩 안에서 줄여서 HBM을 아끼는 중", "HBM에 썼다 읽었다면 층 이동 약 4배 (요청 256개 · 128K)",
         "색인을 옮길 때 head 합 · top-k까지 함께", "top-k 전용 HW는 공백", "2.3"),
        ("가중치", "gray", "ro_all", "read-only — 요청이 적을 때 이동의 거의 전부",
         "요청이 적을 때는 가중치 대역폭이 시간을 정함", "요청 1개: 이동의 99% 이상이 가중치",
         "가중치 대역폭 · 저정밀", "쓰기가 없어 압축 · 복제가 자유로움", "2.5"),
    ],
    "csa-v4flash": [
        ("CSA 압축 KV", "blue", "app_sel", "끝에 추가 · 골라 읽음 — 토큰당 3 KB (V3.2 KV의 약 13분의 1)",
         "요청별 데이터가 작아 용량 압박이 늦게 옴", "128K 요청당 요청별 데이터 약 0.52 GB (V3.2 약 6.3 GB)",
         "계층형 KV의 우선순위가 낮아짐", "압축이 같은 문제를 먼저 줄임", "2.1"),
        ("색인 K cache", "orange", "app_all", "끝에 추가 · 전부 읽음 — 형식에 민감 (FP8 132 B / MXFP4 68 B)",
         "요청 · 문맥이 크면 요청별 이동의 대부분", "요청 256개 · 128K: 모델 이동 12.5% · CSA 층 이동 74%",
         "메모리 근처 선택 · 색인 연산", "키 1 B당 약 124 FLOP(FP8) — 형식이 요구를 바꿈", "2.2"),
        ("최근 창 · 압축기 state", "violet", "ring_all", "고정 크기 · 덮어씀 — 요청당 약 15 MB",
         "작고 문맥과 무관", "요청 256개에서 보관 4 GB(1.4%) · 이동 0.5%",
         "별도 대응 불필요", "state PIM의 대상 아님", "x"),
        ("MoE expert (FP4)", "gray-l", "ro_sel", "read-only · 골라 읽음 — 요청 32개 이상 이동의 79 ~ 92%",
         "모델 전체 이동은 거의 MoE 가중치", "KV가 작아져 가중치 비중이 더 커짐",
         "가중치 대역폭 · 요청 수에 맞춘 배치", "expert는 이미 FP4", "-"),
    ],
    "kda-k3": [
        ("KDA state", "green", "rmw_all", "매 스텝 전부 다시 씀 — 요청당 434 MB(69층), 문맥과 무관",
         "요청이 늘수록 층 이동의 대부분", "요청 256개: KDA 층 이동 74% · 모델 이동 비중이 보관 비중의 2 ~ 3배",
         "메모리 안에서 갱신해 왕복 제거", "이득은 감쇠 · rank-1 갱신을 메모리 쪽이 감당하는지에 달림", "2.4"),
        ("DSA KV (24층)", "blue", "app_sel", "끝에 추가 · 골라 읽음 — 토큰당 27.6 KB, 요청당 2,048줄만 읽음",
         "보관만 크고 이동은 작은 용량형", "요청 256개 · 128K: 보관 34% · 이동 0.8% (K3의 MLA는 이동 34%)",
         "전체 KV는 큰 메모리, 고를 KV만 HBM", "MLA → DSA로 기준선 대응(2.5)에서 계층형 KV로 바뀐 칸", "2.1"),
        ("색인 K cache (24층)", "orange", "app_all", "끝에 추가 · 전부 읽음 — 토큰당 3.2 KB (132 B × 24층, 형식 가정)",
         "보관은 작아도 요청별 이동에서 state 다음", "요청 1개 · 128K: 요청별 이동의 30% / 요청 256개 · 128K: 모델 이동 5.6%",
         "색인 읽기를 메모리 근처에서, 칩에는 선택 위치만", "키 1 B당 약 62 FLOP — 색인 head 32개라 V3.2의 절반", "2.2"),
        ("attention 가중치 (BF16)", "gray", "ro_all", "read-only · 전부 읽음 — 약 73 GB",
         "요청이 적을 때 이동의 절반", "요청 1개: attention 가중치 52% · MoE 44%",
         "attention 가중치 저정밀", "지금은 routed expert만 MXFP4", "2.5"),
    ],
    "qsa-qwen38": [
        ("QSA KV", "blue", "app_sel", "끝에 추가 · 골라 읽음 — 토큰당 24.6 KB, 요청당 2,048토큰만 읽음",
         "보관의 대부분 · 이동은 작음 (용량형)", "요청 256개 · 128K: 보관 73% · 이동 4%",
         "전체 KV는 큰 메모리, 고를 블록만 HBM", "블록(4줄 연속) 단위라 전송이 덜 흩어짐", "2.1"),
        ("블록 키", "orange", "app_all", "끝에 추가 · 전부 읽음 — 토큰당 768 B",
         "긴 문맥에서 QSA 층 이동의 대부분 (대역폭형)", "요청 32개 · 128K: 층 이동 52% · 키 1 B당 약 4 FLOP",
         "블록 선택을 메모리 근처에서", "메모리 쪽 연산 요구가 작은 편", "2.2"),
        ("GDN state", "green", "rmw_all", "매 스텝 전부 다시 씀 — 요청당 113 MB(36층)",
         "요청이 늘면 MoE 다음으로 큰 이동", "요청 256개: 모델 이동 17 ~ 18%",
         "메모리 안 갱신", "이득은 갱신 연산을 메모리 쪽이 감당하는지에 달림", "2.4"),
        ("세 성격이 한 모델에", None, "mix", "골라 읽음(KV) · 전부 읽음(블록 키) · 다시 씀(state)",
         "대응 후보가 셋", "한 장치로 묶을지 층 종류별로 나눌지는 비용 비교가 필요",
         "층 종류별 이종 구성 검토", "MoE는 BF16 — 저정밀 여지도 큼", "2.5"),
    ],
}


def tb(f, x, y, w, t1, t2, s1=17, s2=14):
    """제목(굵게) + 보조(옅게) 줄바꿈해서 그림. 마지막 y."""
    for ln in wrap(t1, s1, w):
        f.text(x, y, ln, s1, weight=700)
        y += s1 + 9
    for ln in wrap(t2, s2, w):
        f.text(x, y, ln, s2, color=MUTED)
        y += s2 + 8
    return y


def arrow(f, x1, x2, y, c=NAVY):
    f.line(x1, y, x2 - 8, y, c, 2.4)
    f.raw(f'<path d="M{x2 - 10:.1f},{y - 7:.1f} L{x2:.1f},{y:.1f} L{x2 - 10:.1f},{y + 7:.1f} Z" fill="{c}"/>')


def fig_summary(mid):
    M = tr.load(mid)
    mdl = M["model"]["model"]
    rows = SUMMARY[mid]
    X = [40, 640, 1190]
    WS = [530, 480, W - 40 - 1190]
    PAD = 24
    def h_of(r):
        a = 64 + len(wrap(r[3], 14, WS[0] - 2 * PAD)) * 22
        b = 30 + len(wrap(r[4], 17, WS[1] - 2 * PAD)) * 26 + len(wrap(r[5], 14, WS[1] - 2 * PAD)) * 22
        c = 64 + len(wrap(r[6], 17, WS[2] - 2 * PAD)) * 26 + len(wrap(r[7], 14, WS[2] - 2 * PAD)) * 22
        return max(a, b, c) + 12
    TOP = 170
    H = TOP + sum(h_of(r) + 18 for r in rows) + 60
    f = mr.Fig(f"{mt.PREFIX[mid]}-sum", W, H)
    mr.title(f, f"{mdl} — 정리: 데이터 특성 → 인사이트 → HW",
             "줄마다 데이터 하나의 성격이 무엇을 정하고, 어떤 HW 대응으로 이어지는지 · 숫자는 본문과 같은 계산 · 괄호 절 번호 = 기준 보고서 Ⅱ장")
    for x, t in zip(X, ("데이터 특성 (종류 · 성격 · 커지는 조건)", "인사이트 (그래프에서 보이는 것)", "HW로 연결")):
        f.text(x, TOP - 22, t, 16, weight=700, color=MUTED)
    y = TOP
    for r in rows:
        h = h_of(r)
        item, c, ck, trait, i1, i2, h1, h2, sec = r
        f.rect(X[0], y, WS[0], h, BOXF, LINE, 12, 1.2)
        f.rect(X[0], y, 7, h, char_col(ck) if ck not in ("kern", "mix") else (IDLE if ck == "kern" else NAVY), "none", 3)
        if c:
            chip(f, X[0] + PAD, y + 36, col(c), item, size=18)
        else:
            chip(f, X[0] + PAD, y + 36, None, item, dashed=True, size=18)
        lines(f, X[0] + PAD, y + 66, wrap(trait, 14, WS[0] - 2 * PAD), 14, MUTED, lh=22)
        f.rect(X[1], y, WS[1], h, "#ffffff", LINE, 12, 1.2)
        tb(f, X[1] + PAD, y + 36, WS[1] - 2 * PAD, i1, i2)
        f.rect(X[2], y, WS[2], h, "#f3f1fc", "#d9d2f5", 12, 1.2)
        tag = SECS[sec]
        tw = mr.text_w(tag, 13) + 20
        f.rect(X[2] + PAD, y + 18, tw, 26, "#7b61c9" if sec not in ("-", "x") else IDLE, "none", 13)
        f.text(X[2] + PAD + tw / 2, y + 36, tag, 13, "middle", 700, "#ffffff")
        tb(f, X[2] + PAD, y + 72, WS[2] - 2 * PAD, h1, h2)
        arrow(f, X[0] + WS[0] + 10, X[1] - 10, y + h / 2)
        arrow(f, X[1] + WS[1] + 10, X[2] - 10, y + h / 2)
        y += h + 18
    mr.save(f)


# ------------------------------------------------------------------ 기준 설명 그림 (모델 무관) — 분류 · 뜻 → HW 특성
# 행 = (막대 색, 분류, 뜻, 예, HW 제목, HW 보조, Ⅱ장 절) · ("#", 제목) = 묶음 머리
GEN = {
    "kind": ("crit-c1-kind", "기준 1. 데이터 종류 — 무엇에 비례해 커지나",
             "커지는 조건이 많을수록 HBM을 먼저 채움", "→ HW에서 문제가 되는 것", [
                 (mr.GRAY, "공유 가중치", "요청 · 문맥과 무관 — 모든 요청이 같이 씀", "가중치",
                  "고정 크기 → 용량보다 읽기 대역폭",
                  "요청이 적으면 매 스텝 이동의 대부분 · 요청이 많으면 한 번 읽어 여러 요청이 나눠 씀", None),
                 (mr.GREEN, "요청별 · 크기 고정", "요청 수에만 비례 — 문맥이 길어도 그대로", "linear attention state",
                  "동시 요청 수만큼 용량", "짧은 문맥 · 많은 요청에서 비중이 큼", None),
                 (mr.BLUE, "요청별 · 문맥 비례", "요청 수 × 문맥 길이에 비례 — 토큰마다 쌓임", "KV cache",
                  "용량 — 가장 먼저 HBM을 채움", "요청 · 문맥이 함께 크면 가중치보다 커짐 → 큰 메모리 계층 · 압축", "2.1"),
             ]),
    "life": ("crit-c2-life", "기준 2. 수명 — 얼마나 오래 남나",
             "다시 읽힐 때까지 오래 걸릴수록 칩에서 먼 큰 메모리 · 금방 버릴수록 칩 안", "→ 둘 곳", [
                 ("#2f4b8c", "서비스 내내", "안 바뀌고 모든 요청이 같이 씀", "가중치",
                  "HBM에 상주", "커서 칩 안에 못 둠 · 바뀌지 않아 GPU 여러 장에 나눠 두거나 복제 가능", None),
                 ("#5b74b8", "요청 동안", "요청이 끝날 때까지 남고 매 스텝 다시 읽힘", "KV cache · state",
                  "HBM 또는 그 아래 큰 메모리", "다시 읽기 전에 모델 전체가 지나가 칩 안에 못 남음 · 요청 수만큼 쌓임", "2.1"),
                 ("#a9b8e2", "스텝 안", "한 커널이 쓰고 다음 커널이 바로 읽고 버림", "q · 점수 · 선택 위치",
                  "칩 안 L2에 남길 수 있음", "커널 합치기 · L2 상주로 HBM 왕복을 없앰 — SRAM이 효과를 내는 곳", None),
                 ("#e1e7f5", "커널 안", "한 커널 안에서 만들고 바로 줄여 끝남", "attention 점수 · 확률",
                  "이미 칩 안 (레지스터 · SMEM)", "HBM 부담 없음 · 처리를 다른 장치로 옮기면 이 값도 그쪽에서 끝내야 함", None),
             ]),
    "char": ("crit-c3-char", "기준 3. 데이터 성격 — 어떻게 읽고 쓰나",
             "읽는 방식이 용량형 · 대역폭형을, 쓰는 방식이 쓰기 부담 · 압축 자유도를 정함", "→ 필요한 메모리 특성", [
                 ("#", "읽는 방식 — 용량형인가 대역폭형인가"),
                 (NAVY, "전부 읽음", "매 스텝 들고 있는 만큼 다 읽음", "dense 가중치 · full attention KV · 색인 K",
                  "대역폭형 — 읽기 대역폭", "크기 = 매 스텝 이동량 · 메모리 근처에서 처리해 결과만 보내면 이동이 줄어듦", "2.2"),
                 (IDLE, "골라 읽음", "들고 있는 것 중 일부만 읽음", "sparse attention KV · MoE expert",
                  "용량형 — 큰 메모리 + 고른 것만 HBM", "대부분 쉬고 있음 → 크고 느린 메모리에 두고 고른 것만 빠르게 가져옴", "2.1"),
                 ("#", "쓰는 방식 — 쓰기 부담과 압축 자유도"),
                 (mr.GRAY, "read-only", "한 번 올리면 안 바뀜", "가중치",
                  "압축 · 저정밀 · 복제가 자유로움", "쓰기가 없어 미리 변환해 두고 여러 장치에 나눠 둬도 됨", None),
                 (mr.BLUE, "끝에 추가", "토큰마다 한 줄 추가 · 지난 줄은 그대로", "KV cache",
                  "쓰기는 작고 보관이 쌓임 → 용량", "스텝당 쓰기는 한 줄 · 문맥만큼 보관이 늘어남", None),
                 ("#6250d6", "고정 크기 · 덮어씀", "최근 N개만 남기고 덮어씀", "sliding window KV",
                  "작은 고정 버퍼", "문맥이 길어도 안 커짐 → 별도 대응 불필요", None),
                 (mr.GREEN, "매 스텝 전부 다시 씀", "읽은 만큼 다시 씀", "linear attention state",
                  "쓰기 대역폭 — 메모리 안 갱신", "읽기 = 쓰기 → 메모리 안에서 갱신하면 칩까지 왕복이 사라짐", "2.4"),
                 (mr.YELLOW, "임시", "다음 커널이 읽고 버림", "q · 점수",
                  "칩 안에서 끝내기", "수명 '스텝 안'과 같은 데이터 · 커널 합치기 · L2 상주", None),
             ]),
}
GL, GLW, GLAB = 40, 780, 280      # 왼쪽 상자 · 분류 이름 칸 폭
GR = 890
GRW = W - 40 - GR


def gen_fig(key):
    """기준 하나 = 분류마다 '이런 데이터면 → 이런 HW 특성' 한 줄. 모델 데이터는 넣지 않음."""
    name, ttl, rule, rhead, rows = GEN[key][:5]
    lhead = GEN[key][5] if len(GEN[key]) > 5 else "분류 · 어떤 데이터인가"
    dw = GLW - GLAB - 40

    def parts(r):
        _, lab, what, ex, h1, h2, sec = r
        tw = GRW - 56 - (mr.text_w(SECS[sec], 13) + 44 if sec else 0)
        return wrap(what, 17, dw), wrap("예: " + ex, 15, dw), wrap(h1, 20, tw), wrap(h2, 15, tw)

    def rh(r):
        if r[0] == "#":
            return 40
        a, b, c, d = parts(r)
        return max(92, 26 * len(a) + 22 * len(b) + 40, 30 * len(c) + 22 * len(d) + 40)

    TOP = 180
    f = mr.Fig(name, W, TOP + sum(rh(r) + 12 for r in rows) + 24)
    f.text(40, 56, ttl, 28, weight=800)
    f.text(40, 100, rule, 20, weight=800, color="#5b47b0")
    f.text(GL + 4, TOP - 24, lhead, 17, weight=800, color=MUTED)
    f.text(GR + 4, TOP - 24, rhead, 17, weight=800, color=MUTED)
    if any(r[0] != "#" and r[6] for r in rows):
        f.text(W - 40, TOP - 24, SEC_NOTE, 14, "end", 600, MUTED)
    y = TOP
    for r in rows:
        h = rh(r)
        if r[0] == "#":
            f.text(GL + 4, y + 27, r[1], 17, weight=800, color=NAVY)
            f.line(GL + 16 + mr.text_w(r[1], 17), y + 21, W - 40, y + 21, LINE, 1.2)
            y += h + 12
            continue
        acc, lab, *_ , sec = r
        a, b, c, d = parts(r)
        # 왼쪽 — 분류 이름 | 뜻 · 예
        f.rect(GL, y, GLW, h, "#ffffff", LINE, 12, 1.2)
        f.rect(GL, y, 10, h, acc, LINE if acc in ("#e1e7f5", IDLE) else "none", 4, 1)
        f.text(GL + 32, y + h / 2 + 8, lab, 22, weight=800)
        yy = y + (h - 26 * len(a) - 22 * len(b)) / 2 + 19
        for ln in a:
            f.text(GL + GLAB, yy, ln, 17, weight=600)
            yy += 26
        for ln in b:
            f.text(GL + GLAB, yy, ln, 15, color=MUTED)
            yy += 22
        # 오른쪽 — HW 특성
        arrow(f, GL + GLW + 12, GR - 10, y + h / 2, HL_LINE)
        f.rect(GR, y, GRW, h, HL_FILL, "#ddd5f5", 12, 1.2)
        yy = y + (h - 30 * len(c) - 22 * len(d)) / 2 + 21
        for ln in c:
            f.text(GR + 28, yy, ln, 20, weight=800)
            yy += 30
        for ln in d:
            f.text(GR + 28, yy - 2, ln, 15, color=MUTED)
            yy += 22
        if sec:
            tag = SECS[sec]
            pw = mr.text_w(tag, 13) + 20
            px = GR + GRW - 22 - pw
            f.rect(px, y + h / 2 - 13, pw, 26, PILL, "none", 13)
            f.text(px + pw / 2, y + h / 2 + 5, tag, 13, "middle", 700, "#ffffff")
        y += h + 12
    mr.save(f)


if __name__ == "__main__":
    ids = sys.argv[1:] or MODELS + ["map", "gen"]
    for mid in ids:
        if mid == "map":
            fig_char_map()
            continue
        if mid == "gen":
            for k in GEN:
                gen_fig(k)
            continue
        fig_kind(mid)
        fig_life(mid)
        fig_char(mid)
        if mid in SUMMARY:
            fig_summary(mid)
