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
TAG = {"mla-v31": "V3.1", "dsa-v32": "V3.2", "csa-v4flash": "V4", "kda-k3": "K3", "qsa-qwen38": "Q3.8"}
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


# ------------------------------------------------------------------ 기준 그림 — 분류만 눈에 띄게 (설명은 보고서 본문)
MCOL = {"mla-v31": "#4a5573", "dsa-v32": "#2a78d6", "csa-v4flash": "#6250d6", "kda-k3": "#1baf7a", "qsa-qwen38": "#eb6834"}
KIND_COLS = [("공유 가중치", "요청 · 문맥과 무관", mr.GRAY),
             ("요청별 · 크기 고정", "요청 수에 비례", mr.GREEN),
             ("요청별 · 문맥에 비례", "요청 수 × 문맥 길이에 비례", mr.BLUE)]
LIFE = [("svc", "서비스 내내"), ("req", "요청 동안"), ("step", "스텝 안"), ("kern", "커널 안")]
CROWS = [("ro", "read-only", mr.GRAY), ("app", "끝에 추가", mr.BLUE), ("ring", "고정 크기 · 덮어씀", "#6250d6"),
         ("rmw", "매 스텝 전부 다시 씀", mr.GREEN), ("tmp", "임시 (커널 사이)", mr.YELLOW)]
TINT = {"ro_all": "#eeede9", "ro_sel": "#f6f6f3", "app_all": "#dde9fa", "app_sel": "#eef4fd",
        "ring_all": "#e8e5f9", "rmw_all": "#dcf2ea", "tmp": "#fcf1d6"}
LH = 34


def cell_of(ch):
    """traits 성격 키 → 격자 칸 키 (덮어씀 · 다시 씀은 '전부 읽음' 칸)."""
    return {"ring": "ring_all", "rmw": "rmw_all"}.get(ch, ch)


def char_col(k):
    return mt.CHAR_COL.get({"ring_all": "ring", "rmw_all": "rmw"}.get(k, k), "#ffffff")


def fh(chips, width):
    return flow(None, 0, 0, width, chips, LH, draw=False) if chips else 0


def head(f, ttl):
    f.text(40, 56, ttl, 28, weight=800)


def kind_fig(name, ttl, cols):
    GAP = 28
    CW = (W - 80 - 2 * GAP) / 3
    BH = 140 + max(fh(c, CW - 56) for c in cols) + 34
    TOP = 100
    f = mr.Fig(name, W, TOP + BH + 40)
    head(f, ttl)
    for i, (t, tag, acc) in enumerate(KIND_COLS):
        x = 40 + i * (CW + GAP)
        f.rect(x, TOP, CW, BH, BOXF, LINE, 16, 1.2)
        f.rect(x, TOP, CW, 12, acc, "none", 6)
        f.text(x + 28, TOP + 62, t, 26, weight=800)
        f.text(x + 28, TOP + 96, tag, 16, weight=700, color=acc if acc != mr.GRAY else MUTED)
        if cols[i]:
            flow(f, x + 28, TOP + 146, CW - 56, cols[i], LH)
        else:
            f.text(x + 28, TOP + 146, "없음", 16, color=IDLE, weight=700)
    mr.save(f)


def life_fig(name, ttl, lanes):
    LW, GAPX = 300, 20
    X2 = 40 + LW + GAPX
    DW = W - 40 - X2 - 56
    RH = {k: max(78, fh(lanes.get(k), DW) + 50) for k, _ in LIFE}
    TOP = 104
    H = TOP + 2 * 50 + sum(RH[k] + 12 for k, _ in LIFE) + 30
    f = mr.Fig(name, W, H)
    head(f, ttl)
    y = TOP
    for gi, (keys, g) in enumerate(((("svc", "req"), "HBM에 남음 — 한 스텝 뒤에 다시 읽힘"),
                                    (("step", "kern"), "칩 안에서 끝남 — 같은 스텝 · 같은 커널 안에서 다시 읽힘"))):
        f.text(40, y + 32, g, 19, weight=800, color=NAVY if gi == 0 else "#1b8a62")
        y += 50
        for k, nm in LIFE:
            if k not in keys:
                continue
            h = RH[k]
            dark = k in ("svc", "req")
            f.rect(40, y, LW, h, mt.LIFE_COL[k], LINE if k == "kern" else "none", 12, 1)
            f.text(40 + 26, y + h / 2 + 9, nm, 24, weight=800, color="#ffffff" if dark else INK)
            f.rect(X2, y, W - 40 - X2, h, "#ffffff", LINE, 12, 1.2)
            ch = lanes.get(k)
            if ch:
                flow(f, X2 + 28, y + (h - fh(ch, DW)) / 2 + 6, DW, ch, LH)
            else:
                f.text(X2 + 28, y + h / 2 + 6, "없음", 16, color=IDLE, weight=700)
            y += h + 12
    mr.save(f)


def char_fig(name, ttl, cells):
    LW, G = 330, 16
    CW = (W - 80 - LW - 2 * G) / 2
    xs = [40 + LW + G, 40 + LW + G + CW + G]
    rows = [r for r in CROWS if any(cells.get(k) for k in ([r[0]] if r[0] == "tmp" else [f"{r[0]}_all", f"{r[0]}_sel"]))]
    def rh(r):
        if r[0] == "tmp":
            return max(78, fh(cells.get("tmp"), W - 40 - xs[0] - 56) + 50)
        return max(78, max(fh(cells.get(f"{r[0]}_{c}"), CW - 56) for c in ("all", "sel")) + 50)
    TOP = 160
    f = mr.Fig(name, W, TOP + sum(rh(r) + 12 for r in rows) + 30)
    head(f, ttl)
    for x, t in zip(xs, ("전부 읽음", "골라 읽음")):
        f.text(x + 4, TOP - 26, t, 24, weight=800)
    f.text(40, TOP - 26, "쓰기 방식 ↓   읽기 방식 →", 15, weight=700, color=MUTED)
    y = TOP
    for wk, wn, acc in rows:
        h = rh((wk,))
        f.rect(40, y, LW, h, "#ffffff", LINE, 12, 1.2)
        f.rect(40, y, 10, h, acc, "none", 4)
        f.text(40 + 30, y + h / 2 + 8, wn, 21, weight=800)
        keys = ["tmp"] if wk == "tmp" else [f"{wk}_all", f"{wk}_sel"]
        widths = [W - 40 - xs[0]] if wk == "tmp" else [CW, CW]
        for k, x, w in zip(keys, xs, widths):
            ch = cells.get(k)
            if k in ("ring_sel", "rmw_sel"):
                continue
            f.rect(x, y, w, h, TINT[k] if ch else "#fbfbfc", LINE if ch else "#eef0f5", 12, 1.2)
            if ch:
                flow(f, x + 28, y + (h - fh(ch, w - 56)) / 2 + 6, w - 56, ch, LH)
        y += h + 12
    mr.save(f)


def fig_kind(mid):
    M = tr.load(mid)
    rows = items_of(M)
    ws = sorted([r for r in rows if r["life"] == "svc"], key=lambda r: -r["store"])
    fx = [r for r in rows if r["life"] == "req" and r["tok"] <= 0]
    tk = [r for r in rows if r["life"] == "req" and r["tok"] > 0]
    kind_fig(f"{mt.PREFIX[mid]}-c1-kind", f"{M['model']['model']} — 데이터 종류",
             [[(col(r["color"]), r["name"], fsize(r["store"])) for r in ws],
              [(col(r["color"]), r["name"], f"요청당 {fsize(r['req'])}") for r in fx],
              [(col(r["color"]), r["name"], f"토큰당 {fsize(r['tok'])}") for r in tk]])


def fig_life(mid):
    M = tr.load(mid)
    rows = items_of(M)
    lanes = {}
    for r in sorted(rows, key=lambda r: -r["store"]):
        lanes.setdefault(r["life"], []).append((col(r["color"]), r["name"], None))
    lanes["kern"] = [(None, x[0], None, True) for x in tr.chip(M, *REF, "layer")["top"][:5]]
    life_fig(f"{mt.PREFIX[mid]}-c2-life", f"{M['model']['model']} — 데이터 수명", lanes)


def fig_char(mid):
    M = tr.load(mid)
    cells = {}
    for r in sorted(items_of(M), key=lambda r: -r["store"]):
        cells.setdefault(cell_of(r["char"]), []).append((col(r["color"]), r["name"], None))
    char_fig(f"{mt.PREFIX[mid]}-c3-char", f"{M['model']['model']} — 데이터 성격", cells)


def fig_char_map():
    """전체 모델 성격 지도 — 칸마다 어느 모델의 어떤 데이터가 있나. 가중치 칸은 전 모델 공통으로 한 줄."""
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
    char_fig("synth-c3-char-map", "모델 5개 — 데이터 성격", cells)


# ------------------------------------------------------------------ 정리 — 특성 → 인사이트 → HW
SECS = {"2.1": "2.1 계층형 KV", "2.2": "2.2 메모리 근처 선택", "2.3": "2.3 색인 연산 · top-k", "2.4": "2.4 state PIM",
        "2.5": "2.5 이종 가속기 · PD 분리", "-": "기준 보고서 범위 밖", "x": "대응 대상 아님"}
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
         "요청이 늘수록 층 이동의 대부분", "요청 256개: 층 이동 74% · 이동 비중이 보관 비중의 약 2배",
         "메모리 안에서 갱신해 왕복 제거", "이득은 감쇠 · rank-1 갱신을 메모리 쪽이 감당하는지에 달림", "2.4"),
        ("MLA KV (24층)", "blue", "app_all", "끝에 추가 · 전부 읽음 — 토큰당 27.6 KB",
         "긴 문맥에서는 기준선과 같은 부담", "요청 256개 · 128K: 보관 36% · 이동 34%",
         "HBM 용량 · 대역폭, KV 저정밀", "기준선(V3.1)과 같은 대응", "2.5"),
        ("attention 가중치 (BF16)", "gray", "ro_all", "read-only · 전부 읽음 — 약 72 GB",
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


def arrow(f, x1, x2, y):
    f.line(x1, y, x2 - 8, y, NAVY, 2.4)
    f.raw(f'<path d="M{x2 - 10:.1f},{y - 7:.1f} L{x2:.1f},{y:.1f} L{x2 - 10:.1f},{y + 7:.1f} Z" fill="{NAVY}"/>')


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


if __name__ == "__main__":
    ids = sys.argv[1:] or MODELS + ["map"]
    for mid in ids:
        if mid == "map":
            fig_char_map()
            continue
        fig_kind(mid)
        fig_life(mid)
        fig_char(mid)
        if mid in SUMMARY:
            fig_summary(mid)
