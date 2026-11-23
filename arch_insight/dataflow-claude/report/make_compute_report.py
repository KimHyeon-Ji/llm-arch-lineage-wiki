"""연산 종합 보고서(synthesis-compute.md) 그림 — 모델 5개의 decode 1스텝을 연산 쪽에서 비교.

  <venv python> report/make_compute_report.py            그림 생성
  <venv python> report/make_compute_report.py print      본문 수치 출력

메모리 종합(synthesis-traits.md)과 같은 틀: 모델 5개 · 기준선 / Sparse / Hybrid 띠 · 조건 셋.
수치는 tools/ops.py (algo · map · model · ops.yaml 식). 그림은 report/figures/comp-*.

comp-c1-unit · c2-axis · c3-ai   연산을 나누는 기준 (모델 무관) — 분류 · 뜻 → HW
comp-t1-cat-model    연산 묶음별 FLOP · 이론 시간 (모델 전체, 조건 3개)
comp-t2-unit         연산기 · 정밀도별 FLOP 구성 + 연산이 시간을 정하는 몫
comp-t3-growth       문맥 길이 vs (요청별 데이터 연산 FLOP ÷ 가중치 GEMM FLOP) — FLOP 역전점
comp-t4-ai           산술 강도 — 요청별 데이터 연산 | 공유 가중치 GEMM (요청 1 · 32 · 256개)
comp-t5-irregular    top-k · gather — 후보 → 선택, 덩어리 크기, 스텝당 덩어리 수
comp-t6-chain        직렬 경로 — 층 종류별 커널 의존 깊이
comp-sum             정리 — 연산 특성 → 인사이트 → HW
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
import make_synthesis_report as ms  # noqa: E402
import ops  # noqa: E402

W = cf.W
MODELS, TAG, MCOL, NAME = cf.MODELS, cf.TAG, cf.MCOL, ms.NAME
PINK, VIOLET = "#e87ba4", "#7b61c9"
CAT_COL = {"proj": mr.GRAY, "moe": mr.GRAY_L, "head": mr.LMH, "attn": mr.BLUE, "idx": mr.ORANGE,
           "topk": PINK, "state": mr.GREEN, "misc": mr.YELLOW}
CATS = [(k, lab, CAT_COL[k]) for k, lab in ops.CATS]
UCOL = {("gemm", "fp8"): "#2f4b8c", ("gemm", "bf16"): "#7d93cf", ("vec", "fp32"): mr.GREEN}
DARK = mt.DARK | {"#2f4b8c", "#7d93cf"}
CONDS = ms.CONDS
T128 = 131072
HW = ops.load("dsa-v32")["hw"]
RIDGE = {"FP32 CUDA core": HW["peak"]["vec"] / HW["hbm_bw"], "BF16": HW["peak"]["bf16"] / HW["hbm_bw"],
         "FP8": HW["peak"]["fp8"] / HW["hbm_bw"]}


def ff(v):
    for u, s in ((1e15, "PFLOP"), (1e12, "TFLOP"), (1e9, "GFLOP"), (1e6, "MFLOP")):
        if v >= u:
            return f"{v / u:.3g} {s}"
    return f"{v:.0f} FLOP"


def ft(v):
    return f"{v:.3g} s" if v >= 1 else f"{v * 1e3:.3g} ms" if v >= 1e-3 else f"{v * 1e6:.3g} µs"


def fn(v):
    """개수 — 만 · 억 단위."""
    if v >= 1e8:
        return f"{v / 1e8:.3g}억"
    if v >= 1e6:
        return f"{round(v / 1e4):,}만"
    if v >= 1e4:
        return f"{v / 1e4:.3g}만"
    return f"{v:,.0f}"


def pc(v):
    return f"{v:.0%}" if v >= 0.095 else f"{v:.1%}" if v >= 0.001 else "0.1% 미만"


# ------------------------------------------------------------------ 기준 그림 (모델 무관) — make_criteria_figs.gen_fig 양식
cf.GEN.update({
    "ops1": ("comp-c1-unit", "기준 1. 연산기 — 무엇으로 계산하나",
             "FLOP은 거의 전부 행렬곱 — FLOP이 작거나 없는 연산이 따로 유닛과 시간을 요구함", "→ HW에서 필요한 것", [
                 (mr.GRAY, "행렬곱 (GEMM)", "행렬 × 행렬 — Tensor core", "가중치 투영 · MoE · attention QK · PV · 색인 점수",
                  "Tensor core 피크 · 정밀도", "FLOP의 98% 이상 · 어느 정밀도(FP8 · BF16) 유닛이 받느냐가 연산 시간을 정함", None),
                 (mr.GREEN, "GEMV · 원소별", "행렬 × 벡터 · 감쇠 · 정규화 — CUDA core", "linear state 갱신 · softmax · 정규화",
                  "넓은 벡터 유닛 · 메모리 안 처리", "FLOP은 1% 남짓 · FP32 state를 매 스텝 전부 읽고 다시 씀", "2.4"),
                 (PINK, "비교 · 선택 (top-k)", "점수 비교 · 상위 k개 고르기", "색인 점수 13만 개 → 2,048개",
                  "선택 전용 유닛", "FLOP 0 — FLOP 그림에 안 보이지만 층마다 직렬 단계 하나", "2.3"),
                 (mr.BLUE, "흩어진 읽기 (gather)", "고른 줄만 골라 모아 읽기", "sparse attention의 KV 2,048줄",
                  "작은 덩어리 읽기 효율 · 주소 계산", "덩어리가 작고 많을수록 대역폭을 못 채움 — 블록 단위가 유리", "2.2"),
             ], "분류 · 어떤 연산인가"),
    "ops2": ("comp-c2-axis", "기준 2. 커지는 축 — 토큰 1개의 FLOP이 무엇에 비례하나",
             "문맥에 비례하는 연산이 긴 문맥의 FLOP을 정함 — 요청 수는 모든 연산을 똑같이 늘림", "→ HW에서 문제가 되는 것", [
                 (mr.GRAY, "고정 — 가중치 연산", "문맥과 무관 · 토큰마다 같은 FLOP", "투영 · MoE · LM head",
                  "활성 파라미터 수가 정함", "모델 크기 · 고르는 expert 수가 그대로 FLOP", None),
                 (mr.BLUE, "문맥 비례", "토큰마다 지난 문맥 전체와 계산", "full attention 본체 · 색인 · 블록 점수",
                  "긴 문맥 FLOP의 대부분 → 색인 연산기 규모", "압축 · 블록 키 · 색인 head 수는 기울기만 줄임", "2.3"),
                 (mr.GREEN, "고정 예산 — 요청별 연산", "문맥이 길어도 그대로 — 고른 k줄 · state 크기", "sparse attention 본체 · state 갱신",
                  "문맥과 무관한 요청별 부담", "요청 수에만 비례 · 짧은 문맥에서 상대 비중이 큼", None),
             ], "분류 · 어떤 연산인가"),
    "ops3": ("comp-c3-ai", "기준 3. 산술 강도 — HBM 1 B를 읽을 때 몇 번 계산하나",
             "강도가 H200 경계(BF16 약 206 · FP8 약 412 FLOP/B)보다 높으면 연산이, 낮으면 대역폭이 시간을 정함", "→ 필요한 HW 특성", [
                 (mr.GRAY, "요청 수로 커짐", "공유 가중치를 여러 요청이 나눠 씀", "투영 · LM head (MoE는 expert로 나뉘어 느리게 커짐)",
                  "요청이 적으면 대역폭형 → 가중치 대역폭 · 배치", "강도 ≈ 요청 수(BF16) · 요청 수 × 2(FP8)", None),
                 (mr.BLUE, "구조로 고정 · 수백", "KV 1줄을 head 수십 ~ 128개가 함께 씀", "MLA · DSA · CSA attention 본체",
                  "연산형 — Tensor core가 시간을 정함", "요청을 모아도 강도가 그대로 (요청마다 자기 KV)", "2.5"),
                 (mr.ORANGE, "구조로 고정 · 수 ~ 수십", "키 1줄을 색인 head 몇 개가 씀", "색인 · 블록 점수 · GQA attention",
                  "대역폭형 — 메모리 근처에 둘 연산기 크기", "head 수 · 키 형식이 강도를 정함", "2.2"),
                 (mr.GREEN, "1 이하", "값 하나를 한두 번 계산하고 다시 씀", "linear state 갱신",
                  "대역폭만 — 메모리 안 갱신", "연산기는 작아도 됨 · 읽은 만큼 다시 씀", "2.4"),
             ], "분류 · 어떤 연산인가"),
})


# ------------------------------------------------------------------ 공통 — 막대 (값 형식만 다름)
X1, PW, TW = ms.X1, ms.PW, ms.TW
X2 = ms.X2


def bar(f, x0, y, RH, vals, cats, scale, fmt, total=True):
    x = x0
    for k, _, c in cats:
        v = vals.get(k, 0)
        w = v * scale
        if w > 0.4:
            f.rect(x, y + 6, max(w - 2, 0.8), RH - 12, c, mr.BOX[1] if c == mr.LMH else "none", 3, 1)
            lab = fmt(v)
            if w >= mr.text_w(lab, 12.5) + 12:
                f.text(x + w / 2 - 1, y + RH / 2 + 5, lab, 12.5, "middle", 700, "#ffffff" if c in DARK else mr.INK)
        x += w
    if total:
        f.text(x + 8, y + RH / 2 + 6, fmt(sum(vals.values())), 14, weight=800)


def cmp_fig(name, ttl, sub, cats, valf, fmts, heads, take):
    RH, FG, CH, CG, TOP = 36, 8, 40, 26, 250
    data = {(c[0], c[1], m): valf(m, c[0], c[1]) for c in CONDS for m in MODELS}
    mx = [{c[:2]: max(sum(data[(c[0], c[1], m)][i].values()) for m in MODELS) for c in CONDS} for i in (0, 1)]
    block = CH + 5 * RH + 2 * FG + CG
    f = mr.Fig(name, W, TOP + len(CONDS) * block + 130)
    mr.title(f, ttl, sub)
    mr.legend(f, [(c[1], c[2]) for c in cats], 40, 130, 18)
    ms.fam_legend(f, 40, 172)
    for x, t in ((X1, heads[0]), (X2, heads[1])):
        f.text(x, TOP - 22, t, 17, weight=700)
    y = TOP
    for B, T, lab, note in CONDS:
        f.text(40, y + 24, lab, 18, weight=800)
        f.text(40 + mr.text_w(lab, 18) + 14, y + 24, note, 14, color=mr.MUTED)
        for i, x in ((0, X1), (1, X2)):
            f.text(x + PW + TW, y + 24, f"막대 끝 = {fmts[i](mx[i][(B, T)])}", 12.5, "end", color=mr.MUTED)
        y += CH

        def row(mid, yy, B=B, T=T):
            a, b = data[(B, T, mid)]
            bar(f, X1, yy, RH, a, cats, PW / mx[0][(B, T)], fmts[0])
            bar(f, X2, yy, RH, b, cats, PW / mx[1][(B, T)], fmts[1])
        y = ms.fam_rows(f, y, RH, FG, W - 40, row) + CG
    mr.takeaway(f, y + 36, *take)
    mr.save(f)


# ------------------------------------------------------------------ t1 연산 묶음별 FLOP · 이론 시간
def fig_cat():
    cmp_fig("comp-t1-cat-model", "모델 5개 — 연산 묶음별 FLOP · 이론 시간 (모델 전체 · decode 1스텝)",
            "이론 시간 = 커널마다 max(HBM 바이트 ÷ 4.8 TB/s, FLOP ÷ 그 연산기 피크)의 합 · H200 1장으로 환산(나눠 담으면 장수만큼 줄어듦) · 조건마다 같은 눈금",
            CATS, lambda m, B, T: (ops.by_cat(ops.rows(m, B, T), "flops"), ops.by_cat(ops.rows(m, B, T), "t")),
            (ff, ft), ("FLOP — 계산량", "이론 시간 — 메모리 · 연산 중 큰 쪽의 합"), TAKE["t1"])


# ------------------------------------------------------------------ t2 연산기 · 정밀도 + 연산이 시간을 정하는 몫
def fig_unit():
    RH, FG, TOP = 76, 10, 250
    PX, PWa = 400, 560
    QX, QW = 1080, 190
    f = mr.Fig("comp-t2-unit", W, TOP + 5 * RH + 2 * FG + 170)
    mr.title(f, "연산기 · 정밀도별 FLOP 구성과 '연산이 시간을 정하는' 몫 (모델 전체)",
             "왼쪽 = FLOP 100%를 연산기 · 정밀도로 나눔 (요청 수와 무관 — 문맥만 바꿈) · 오른쪽 = 이론 시간 중 FLOP ÷ 피크가 바이트 ÷ 대역폭보다 긴 커널의 몫")
    mr.legend(f, [(lab, UCOL[u]) for u, lab in ops.UNITS] + [("연산이 시간을 정함", VIOLET)], 40, 130, 26)
    ms.fam_legend(f, 40, 172)
    f.text(PX, TOP - 22, "FLOP 구성 — 문맥 8K · 128K", 17, weight=700)
    f.text(QX, TOP - 22, "이론 시간 중 연산형 몫", 17, weight=700)
    for i, (B, T, lab, _) in enumerate(CONDS):
        f.text(QX + i * QW + 4, TOP - 2, lab, 13, weight=700, color=mr.MUTED)

    def row(mid, y):
        for j, T in enumerate((8192, T128)):
            u = ops.by_unit(ops.rows(mid, 1, T))
            tot = sum(u.values())
            yy = y + 8 + j * 30
            f.text(PX - 10, yy + 18, f"{T // 1024}K", 12.5, "end", color=mr.MUTED)
            x = PX
            for k, _ in ops.UNITS:
                w = PWa * u.get(k, 0) / tot
                if w > 0.4:
                    f.rect(x, yy, max(w - 2, 0.8), 24, UCOL[k], "none", 3)
                    lab = pc(u.get(k, 0) / tot)
                    if w >= mr.text_w(lab, 12) + 10:
                        f.text(x + w / 2, yy + 17, lab, 12, "middle", 700, "#ffffff")
                x += w
        for i, (B, T, _, _) in enumerate(CONDS):
            bt = ops.bound_time(ops.rows(mid, B, T))
            s = bt["cmp"] / (bt["cmp"] + bt["mem"])
            x0, yy = QX + i * QW, y + RH / 2 - 12
            f.rect(x0, yy, QW - 70, 24, "#eceef4", "none", 4)
            if s > 0.004:
                f.rect(x0, yy, max((QW - 70) * s, 2), 24, VIOLET, "none", 4)
            f.text(x0 + QW - 64, yy + 18, pc(s) if s >= 0.001 else "≈ 0%", 14, weight=800)
    ms.fam_rows(f, TOP + 10, RH, FG, W - 40, row)
    mr.takeaway(f, TOP + 10 + 5 * RH + 2 * FG + 70, *TAKE["t2"])
    mr.save(f)


# ------------------------------------------------------------------ t3 문맥 길이 vs FLOP 비 (역전점)
def ratio(mid, T):
    p = ops.per_token(mid, T)
    return sum(p.get(c, 0) for c in ops.REQ) / sum(p.get(c, 0) for c in ops.WEIGHT)


def fig_growth():
    PX, PY, PWX, PH = 150, 160, 900, 640
    xl, xh, yl, yh = 12, 21, -2.0, 2.0          # 2^12(4K) ~ 2^21(2M) · 10^-2 ~ 10^2

    def sx(T):
        return PX + PWX * (math.log2(T) - xl) / (xh - xl)

    def sy(r):
        return PY + PH - PH * (math.log10(max(r, 10 ** yl)) - yl) / (yh - yl)
    f = mr.Fig("comp-t3-growth", W, PY + PH + 210)
    mr.title(f, "문맥이 길어지면 FLOP은 어디서 늘어나나 — 요청별 데이터 연산 ÷ 가중치 GEMM (토큰 1개 · 모델 전체)",
             "요청별 데이터 연산 = attention 본체 + 색인 · 블록 점수 + state 갱신 · 가중치 GEMM = 투영 + MoE + LM head · 두 축 모두 로그 · 점선 = 지원 길이 밖")
    for e in range(xl, xh + 1):
        f.line(sx(2 ** e), PY, sx(2 ** e), PY + PH, mr.GRID, 1)
        f.text(sx(2 ** e), PY + PH + 26, f"{2 ** e // 1024}K" if e < 20 else f"{2 ** e // 2 ** 20}M", 13, "middle", color=mr.MUTED)
    for e in range(int(yl), int(yh) + 1):
        for m in (1, 3):
            v = m * 10 ** e
            if 10 ** yl <= v <= 10 ** yh:
                f.line(PX, sy(v), PX + PWX, sy(v), mr.GRID, 1)
                f.text(PX - 10, sy(v) + 5, f"{v:g}", 13, "end", color=mr.MUTED)
    f.rect(PX, PY, PWX, PH, "none", mr.AXIS, 0, 1.2)
    f.raw(f'<line x1="{PX}" y1="{sy(1):.1f}" x2="{PX + PWX}" y2="{sy(1):.1f}" stroke="{mr.INK}" stroke-width="2" stroke-dasharray="8 6"/>')
    f.text(PX + 40, sy(1) + 26, "역전 (= 1)", 14, weight=800)
    f.text(PX + PWX / 2, PY + PH + 60, "문맥 길이 (토큰)", 16, "middle", 700)
    f.raw(f'<text x="{PX - 76}" y="{PY + PH / 2}" font-family="{mr.FONT}" font-size="16" font-weight="700" fill="{mr.INK}" '
          f'text-anchor="middle" transform="rotate(-90 {PX - 76} {PY + PH / 2})">요청별 데이터 연산 FLOP ÷ 가중치 GEMM FLOP</text>')
    Ts = [2 ** (xl + i * (xh - xl) / 36) for i in range(37)]
    lab_dy = {"mla-v31": -14, "dsa-v32": -14, "csa-v4flash": 22, "kda-k3": -14, "qsa-qwen38": 22}
    f.raw(f'<clipPath id="plot"><rect x="{PX}" y="{PY}" width="{PWX}" height="{PH}"/></clipPath><g clip-path="url(#plot)">')
    ends = {}
    for mid in MODELS:
        tmax = ops.load(mid)["algo"]["dims"]["T_max"]["value"]
        pin = [(sx(T), sy(ratio(mid, T))) for T in Ts if T <= tmax * 1.001] + [(sx(tmax), sy(ratio(mid, tmax)))]
        pout = [(sx(tmax), sy(ratio(mid, tmax)))] + [(sx(T), sy(ratio(mid, T))) for T in Ts if T > tmax]
        f.raw('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pin) + f'" fill="none" stroke="{MCOL[mid]}" stroke-width="3"/>')
        if len(pout) > 1:
            f.raw('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pout) + f'" fill="none" stroke="{MCOL[mid]}" '
                  f'stroke-width="2.2" stroke-dasharray="6 5"/>')
        x, y = pin[-1]
        ms.marker(f, x, y, 6, ms.FAM_OF[mid], MCOL[mid])
        ends[mid] = (x, y, pout[-1] if len(pout) > 1 else pin[-1])
    f.raw("</g>")
    for mid in MODELS:
        c = ops.cross_ctx(mid)
        if c and c <= 2 ** xh:
            f.raw(f'<circle cx="{sx(c):.1f}" cy="{sy(1):.1f}" r="8" fill="#ffffff" stroke="{MCOL[mid]}" stroke-width="3"/>')
            f.text(sx(c), sy(1) + lab_dy[mid] + (8 if lab_dy[mid] > 0 else 0), fk(c), 13, "middle", 800, MCOL[mid])
        x, y, (xe, ye) = ends[mid]
        if ye > PY + 10:
            f.text(xe + 8, ye + 5, TAG[mid], 14, "start", 800, MCOL[mid])
        else:
            f.text(x + 10, y + 22, TAG[mid], 14, "start", 800, MCOL[mid])
    # 오른쪽 표
    RX = PX + PWX + 60
    f.text(RX, PY + 10, "토큰 1개 · 문맥 128K", 17, weight=800)
    for t, dx in (("모델", 0), ("가중치 GEMM", 190), ("요청별 연산", 320), ("역전 문맥", 460)):
        f.text(RX + dx, PY + 46, t, 14, weight=700, color=mr.MUTED)
    y = PY + 82
    for k, lab, ms_ in ms.FAM:
        f.text(RX, y, lab, 13, weight=800, color=ms.FAM_ACC[k])
        y += 26
        for mid in ms_:
            p = ops.per_token(mid, T128)
            wv = sum(p.get(c, 0) for c in ops.WEIGHT)
            rv = sum(p.get(c, 0) for c in ops.REQ)
            c = ops.cross_ctx(mid)
            tmax = ops.load(mid)["algo"]["dims"]["T_max"]["value"]
            ms.marker(f, RX + 8, y - 5, 6, ms.FAM_OF[mid], MCOL[mid])
            f.text(RX + 24, y, TAG[mid], 14, weight=700)
            f.text(RX + 190, y, ff(wv), 14)
            f.text(RX + 320, y, ff(rv), 14, weight=700)
            f.text(RX + 460, y, (fk(c) + (" (지원 밖)" if c > tmax else "")) if c else "—", 14, weight=700)
            y += 32
        y += 6
    cf.lines(f, RX, y + 10, cf.wrap("요청별 연산 = attention 본체 · 색인 · state. 둘 다 요청 수에 비례해 비율은 요청 수와 무관 · "
                                    "선의 기울기 = 문맥에 비례하는 연산(full attention · 색인)의 몫", 13.5, W - 40 - RX), 13.5, mr.MUTED, lh=21)
    mr.takeaway(f, PY + PH + 120, *TAKE["t3"])
    mr.save(f)


def fk(T):
    """문맥 길이 — 그림 가로축과 같은 2진 단위 (1K = 1,024)."""
    if T >= 2 ** 20:
        return f"{T / 2 ** 20:.2g}M"
    k = T / 1024
    return f"{k:.2g}K" if k < 10 else f"{k:.0f}K"


# ------------------------------------------------------------------ t4 산술 강도
AI_LO, AI_HI = -0.5, 3.2   # log10 FLOP/B
PA, PB, PWI = 400, 1090, 630


def ai_x(x0, v):
    return x0 + PWI * (math.log10(min(max(v, 10 ** AI_LO), 10 ** AI_HI)) - AI_LO) / (AI_HI - AI_LO)


def ai_axis(f, x0, top, bot):
    for v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        f.line(ai_x(x0, v), top, ai_x(x0, v), bot, mr.GRID, 1)
        f.text(ai_x(x0, v), bot + 22, f"{v:g}", 12.5, "middle", color=mr.MUTED)
    for i, (lab, v) in enumerate(RIDGE.items()):
        x = ai_x(x0, v)
        dy = 22 if lab == "FP8" else 0
        f.raw(f'<line x1="{x:.1f}" y1="{top - 6 - dy}" x2="{x:.1f}" y2="{bot}" stroke="{VIOLET}" stroke-width="1.6" stroke-dasharray="5 4"/>')
        f.text(x, top - 12 - dy, f"{lab} {v:.0f}", 12, "middle", 700, VIOLET)
    f.text(x0 + PWI / 2, bot + 46, "산술 강도 — FLOP ÷ HBM 바이트 (로그)", 14, "middle", 700, mr.MUTED)


def dot(f, x, y, shape, fill, r=7, hollow=False):
    st = f'fill="{"#ffffff" if hollow else fill}" stroke="{fill if hollow else "#ffffff"}" stroke-width="{2.4 if hollow else 1.6}"'
    if shape == "c":
        f.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {st}/>')
    elif shape == "s":
        f.raw(f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" rx="2" {st}/>')
    else:
        d = r * 1.3
        f.raw(f'<path d="M{x:.1f},{y - d:.1f} L{x + d:.1f},{y:.1f} L{x:.1f},{y + d:.1f} L{x - d:.1f},{y:.1f} Z" {st}/>')


def fig_ai():
    RH, FG, TOP = 74, 10, 270
    bot = TOP + 5 * RH + 2 * FG
    f = mr.Fig("comp-t4-ai", W, bot + 230)
    mr.title(f, "산술 강도 — 요청별 데이터 연산은 구조가, 공유 가중치 GEMM은 요청 수가 정함 (문맥 128K)",
             "보라 점선 = H200 경계(피크 ÷ 4.8 TB/s) — 오른쪽이면 연산이, 왼쪽이면 대역폭이 시간을 정함 · 층 종류가 여럿이면 FLOP · 바이트 합으로 계산")
    items = [("c", mr.BLUE, "attention 본체"), ("s", mr.ORANGE, "색인 · 블록 점수"), ("d", mr.GREEN, "state 갱신"),
             ("c", mr.GRAY, "투영 · 기타 (요청 1 → 32 → 256개)"), ("s", "#b9b7ae", "MoE · FFN (요청 1 → 32 → 256개)")]
    x = 40
    for sh, c, lab in items:
        dot(f, x + 8, 124, sh, c, 7, hollow=(c == "#b9b7ae"))
        f.text(x + 24, 129, lab, 14.5, color=mr.MUTED)
        x += 40 + mr.text_w(lab, 14.5)
    ms.fam_legend(f, 40, 172)
    f.text(PA, TOP - 58, "요청별 데이터 연산 — 요청 수와 무관", 17, weight=800)
    f.text(PB, TOP - 58, "공유 가중치 GEMM — 요청 수만큼 커짐", 17, weight=800)
    ai_axis(f, PA, TOP - 6, bot)
    ai_axis(f, PB, TOP - 6, bot)

    def row(mid, y):
        cy = y + RH / 2
        for sh, c, cats, dy in (("c", mr.BLUE, ["attn"], -12), ("s", mr.ORANGE, ["idx"], 22), ("d", mr.GREEN, ["state"], 22)):
            v = ops.intensity(mid, 1, T128, cats)
            if v:
                x = ai_x(PA, v)
                dot(f, x, cy, sh, c)
                f.text(x, cy + dy, f"{v:.3g}", 12.5, "middle", 800, c if c != mr.GREEN else "#13825b")
        for sh, c, cats, yy in (("c", mr.GRAY, ["proj", "head"], cy - 13), ("s", "#b9b7ae", ["moe"], cy + 13)):
            pts = [ai_x(PB, ops.intensity(mid, B, T128, cats)) for B in (1, 32, 256)]
            f.line(pts[0], yy, pts[-1], yy, c, 2.2)
            for i, x in enumerate(pts):
                dot(f, x, yy, sh, c, 5 if i < 2 else 7, hollow=(sh == "s"))
            v = ops.intensity(mid, 256, T128, cats)
            f.text(pts[-1] + 12, yy + 5, f"{v:.3g}", 12.5, "start", 800, mr.INK)
    ms.fam_rows(f, TOP, RH, FG, W - 40, row)
    mr.takeaway(f, bot + 110, *TAKE["t4"])
    mr.save(f)


# ------------------------------------------------------------------ t5 top-k · gather
def lbar(f, x0, w, y, v, lo, hi, c, h=20):
    ww = w * (math.log10(max(v, 10 ** lo)) - lo) / (hi - lo)
    f.rect(x0, y, ww, h, c, "none", 3)
    return x0 + ww


def fig_irr():
    RH, FG, TOP = 78, 10, 250
    AX, AW = 400, 420          # top-k
    BX, BW = 960, 300          # 덩어리 크기
    CX, CW = 1380, 200         # 스텝당 덩어리 수
    f = mr.Fig("comp-t5-irregular", W, TOP + 5 * RH + 2 * FG + 160)
    mr.title(f, "FLOP 없는 불규칙 연산 — top-k 후보 → 선택, gather 덩어리 크기 · 개수 (문맥 128K)",
             "막대 = 로그 눈금 · 층 1개 · 요청 1개 기준(왼쪽 두 칸) · 오른쪽 = 모델 전체 · 요청 256개 · 스텝 1번 · hybrid는 sparse 층만")
    mr.legend(f, [("top-k 후보 (점수 개수)", "#f6c9d9"), ("고르는 개수", PINK), ("gather 덩어리 크기", mr.BLUE),
                  ("스텝당 gather 덩어리 수", "#2f4b8c")], 40, 130, 26)
    ms.fam_legend(f, 40, 172)
    for x, t in ((AX, "top-k — 후보 → 선택 (층 1개 · 요청 1개)"), (BX, "gather 덩어리 — 크기 × 개수"),
                 (CX, "스텝당 덩어리 수 (요청 256개)")):
        f.text(x, TOP - 22, t, 16, weight=700)

    def row(mid, y):
        irr = ops.irregular(mid)
        if not irr:
            f.text(AX, y + RH / 2 + 6, "top-k · gather 없음 — KV 전부를 순서대로 읽음 (연속 스트림)", 14, color=mr.MUTED)
            return
        g = irr[0]
        cy = y + 12
        e1 = lbar(f, AX, AW, cy, g["cand"], 2, 6, "#f6c9d9")
        lbar(f, AX, AW, cy, g["pick"], 2, 6, PINK)
        f.text(e1 + 8, cy + 15, f"{g['cand']:,.0f} → {g['pick']:,.0f} ({g['pick'] / g['cand']:.1%})", 13.5, weight=800)
        unit = "블록 (4토큰)" if g["chunk"] != g["row"] else "줄"
        f.text(AX, cy + 44, f"{g['layer']} {g['n']:.0f}층 · 점수 단위 = {'블록' if g['chunk'] != g['row'] else '엔트리' if 'CSA' in g['layer'] else '토큰'}",
               12.5, color=mr.MUTED)
        e2 = lbar(f, BX, BW, cy, g["chunk"], 2, 4.2, mr.BLUE)
        f.text(e2 + 8, cy + 15, f"{mt.fb(g['chunk'])}", 13.5, weight=800)
        f.text(BX, cy + 44, f"× {g['pick']:,.0f}{unit} = {mt.fb(g['chunk'] * g['pick'])}", 12.5, color=mr.MUTED)
        n = g["n"] * 256 * g["pick"]
        e3 = lbar(f, CX, CW, cy, n, 5, 8, "#2f4b8c")
        f.text(e3 + 8, cy + 15, f"{fn(n)}개", 13.5, weight=800)
        f.text(CX, cy + 44, f"{mt.fb(n * g['chunk'])} · top-k 후보 {fn(g['n'] * 256 * g['cand'])}개", 12.5, color=mr.MUTED)
    ms.fam_rows(f, TOP, RH, FG, W - 40, row)
    for x0, w, lo, hi, ticks in ((AX, AW, 2, 6, [(100, "100"), (1e3, "1천"), (1e4, "1만"), (1e5, "10만"), (1e6, "100만")]),
                                 (BX, BW, 2, 4.2, [(100, "100 B"), (1e3, "1 KB"), (1e4, "10 KB")]),
                                 (CX, CW, 5, 8, [(1e5, "10만"), (1e6, "100만"), (1e7, "1천만"), (1e8, "1억")])):
        yb = TOP + 5 * RH + 2 * FG + 8
        for v, lab in ticks:
            x = x0 + w * (math.log10(v) - lo) / (hi - lo)
            f.line(x, yb, x, yb + 6, mr.AXIS, 1)
            f.text(x, yb + 24, lab, 12, "middle", color=mr.MUTED)
    mr.takeaway(f, TOP + 5 * RH + 2 * FG + 90, *TAKE["t5"])
    mr.save(f)


# ------------------------------------------------------------------ t6 직렬 경로
def fig_chain():
    LX, LW, BWb, BH, TOP = 420, 124, 114, 26, 230
    RX = LX + 8 * LW + 24
    rows = []
    for k, lab, ms_ in ms.FAM:
        for mid in ms_:
            cs = ops.chains(mid)
            for i, c in enumerate(cs):
                st = {}
                for _, l in c["chain"]:
                    st[l] = st.get(l, 0) + 1
                rows.append((k, mid, i, len(cs), c, max(st.values())))
    def rh(r):
        return 26 + r[5] * (BH + 6)
    H = TOP + sum(rh(r) + 8 for r in rows) + 2 * 10 + 130
    f = mr.Fig("comp-t6-chain", W, H)
    mr.title(f, "직렬 경로 — 층 1개가 차례로 거치는 커널 (의존 깊이)",
             "가로 = 깊이(앞 커널이 HBM에 쓴 값을 읽으면 다음 깊이) · 같은 칸에 쌓인 커널은 서로 기다리지 않음 · MoE · FFN 커널은 모델마다 같아 뺌 · 회색 글 = 같은 구조 모델의 배치로 대신 셈")
    mr.legend(f, [(lab, CAT_COL[c]) for c, lab in ops.CATS if c not in ("moe", "head")], 40, 130, 22)
    ms.fam_legend(f, 40, 172)
    for l in range(1, 9):
        f.text(LX + (l - 1) * LW + BWb / 2, TOP - 16, f"깊이 {l}", 13, "middle", 700, mr.MUTED)
    f.text(RX, TOP - 16, "깊이 × 층 수 = 직렬 단계", 13, weight=700, color=mr.MUTED)
    y = TOP
    tot = {}
    for fam in [k for k, _, _ in ms.FAM]:
        fr = [r for r in rows if r[0] == fam]
        h = sum(rh(r) + 8 for r in fr) - 8
        f.rect(ms.XF, y, W - 40 - ms.XF, h, ms.FAM_BG[fam], "none", 8)
        f.rect(ms.XF, y, 6, h, ms.FAM_ACC[fam], "none", 3)
        lab = next(l for k, l, _ in ms.FAM if k == fam)
        pw = mr.text_w(lab, 14) + 22
        f.rect(ms.XF + 16, y + h / 2 - 13, pw, 26, ms.FAM_ACC[fam], "none", 13)
        f.text(ms.XF + 16 + pw / 2, y + h / 2 + 5, lab, 14, "middle", 800, "#ffffff")
        for _, mid, i, nc, c, mxs in fr:
            hh = rh((0, 0, 0, 0, 0, mxs))
            if i == 0:
                f.text(ms.XN, y + 26, NAME[mid], 15, weight=800)
            own = c["src"] == mid
            f.text(ms.XN, y + (50 if i == 0 else 26), f"{c['name']} {c['n']:.0f}층" + ("" if own else f" ({TAG.get(c['src'], 'Qwen3.6')} 배치)"),
                   13, weight=700, color=mr.INK if own else mr.MUTED)
            used = {}
            lv = max(l for _, l in c["chain"])
            f.line(LX + 10, y + 13 + BH / 2, LX + (lv - 1) * LW + BWb - 10, y + 13 + BH / 2, "#c9ccd6", 2)
            for kk, l in c["chain"]:
                j = used.get(l, 0)
                used[l] = j + 1
                ki = c["kinfo"][kk["id"]]
                col = CAT_COL[ki["cat"]]
                bx, by = LX + (l - 1) * LW, y + 13 + j * (BH + 6)
                f.rect(bx, by, BWb, BH, col, "none", 6)
                f.text(bx + BWb / 2, by + 18, ki["short"], 12.5, "middle", 700, "#ffffff" if col in DARK else mr.INK)
            s = lv * c["n"]
            tot[mid] = tot.get(mid, 0) + s
            f.text(RX, y + 13 + BH / 2 + 5, f"{lv} × {c['n']:.0f} = {s:,.0f}", 14, weight=700, color=mr.INK if own else mr.MUTED)
            if i == nc - 1:
                f.text(W - 50, y + hh - 14 - (0 if nc == 1 else 0), f"모델 {tot[mid]:,.0f}", 15, "end", 800)
            y += hh + 8
        y += 10
    mr.takeaway(f, y + 40, *TAKE["t6"])
    mr.save(f)
    return tot


# ------------------------------------------------------------------ 정리
SUM_ROWS = [   # (항목, 색 토큰, 왼쪽 띠 색, 특성, 인사이트, 인사이트 보조, HW, HW 보조, 절)
    ("attention 본체 (V3.1 · V3.2 · V4 · K3.1)", mr.BLUE, "KV 1줄을 head 64 ~ 128개가 공유 — 1 B당 180 ~ 420 FLOP, 요청 수와 무관",
     "연산형 — full은 긴 문맥 FLOP의 97%", "V3.1 요청 256개 · 128K: 이론 시간의 81%를 연산이 정함 · sparse는 고른 512 ~ 2,048줄만이라 시간 몫 5% 미만",
     "연산형 attention과 대역폭형 나머지를 나눠 배치", "sparse에서는 몫이 작아 attention 전용 연산기의 이득은 full에 한정", "2.5"),
    ("색인 · 블록 점수 (V3.2 · V4 · K3.1 · Q3.8)", mr.ORANGE, "문맥 비례 FLOP — 토큰마다 지난 모든 키와 내적, 1 B당 4 ~ 121 FLOP",
     "sparse는 attention FLOP을 색인으로 옮김", "V3.2 128K 토큰당 FLOP의 55%가 색인 · 색인 + attention이 문맥 약 4만부터 가중치 GEMM을 넘음",
     "메모리 근처 색인 연산 — 저정밀 행렬 연산기", "필요한 연산 밀도가 모델마다 약 30배 차이 (Q3.8 4 · V3.2 121 FLOP/B)", "2.2"),
    ("top-k · gather (V3.2 · V4 · K3.1 · Q3.8)", PINK, "FLOP 0 — 후보 3.3 ~ 13만 개에서 512 ~ 2,048개 · 덩어리 584 B ~ 8 KB",
     "FLOP 그림에는 없고 층마다 직렬 단계를 더함", "요청 256개 · 128K 스텝당 gather 덩어리: V3.2 3,200만 개(656 B) vs Q3.8 157만 개(8 KB 블록)",
     "선택 전용 유닛 · 블록 단위 gather", "덩어리가 클수록 HBM · 아래 계층 전송 효율이 좋음 (Q3.8 블록 방식)", "2.3"),
    ("state 갱신 (K3.1 · Q3.8)", mr.GREEN, "GEMV · 원소별 FP32 — 1 B당 약 0.9 FLOP, FLOP 몫 0.3 ~ 1.4%",
     "FLOP은 작은데 시간은 MoE 다음", "요청 256개 · 8K 이론 시간 몫: Q3.8 18% · K3.1 13% — 연산은 거의 없고 읽기 = 쓰기",
     "state 갱신 PIM — 연산기는 작아도 됨", "GEMV와 원소별 연산만 있으면 됨 · 이득은 왕복 제거", "2.4"),
    ("가중치 GEMM (전 모델)", mr.GRAY, "공유 가중치 — 강도가 요청 수에 비례, MoE는 expert로 나뉘어 요청 256개에서도 6 ~ 26 FLOP/B",
     "sparse · hybrid 이론 시간의 대부분", "요청 32개 · 128K: MoE 시간 몫 85 ~ 88% (V3.1 54%) — 요청별 연산이 줄어 다시 가중치 읽기가 정함",
     "가중치 대역폭 · 요청 수에 맞춘 배치", "attention 쪽 연산 HW 이득의 상한 = 요청별 연산 시간 몫", "-"),
]


def fig_sum():
    X, WS, PAD = [40, 640, 1190], [530, 480, W - 40 - 1190], 24

    def h_of(r):
        a = 64 + len(cf.wrap(r[3], 14, WS[0] - 2 * PAD)) * 22
        b = 30 + len(cf.wrap(r[4], 17, WS[1] - 2 * PAD)) * 26 + len(cf.wrap(r[5], 14, WS[1] - 2 * PAD)) * 22
        c = 64 + len(cf.wrap(r[6], 17, WS[2] - 2 * PAD)) * 26 + len(cf.wrap(r[7], 14, WS[2] - 2 * PAD)) * 22
        return max(a, b, c) + 12
    TOP = 170
    f = mr.Fig("comp-sum", W, TOP + sum(h_of(r) + 18 for r in SUM_ROWS) + 60)
    mr.title(f, "종합 — 연산 특성 → 인사이트 → HW",
             "줄마다 연산 묶음 하나 · 숫자는 본문과 같은 계산 · 알약 = 기준 보고서 Ⅱ장 절")
    for x, t in zip(X, ("연산 특성 (어느 모델 · 어떤 연산)", "인사이트 (비교에서 보이는 것)", "HW로 연결")):
        f.text(x, TOP - 22, t, 16, weight=700, color=mr.MUTED)
    y = TOP
    for r in SUM_ROWS:
        h = h_of(r)
        item, c, trait, i1, i2, h1, h2, sec = r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]
        f.rect(X[0], y, WS[0], h, cf.BOXF, cf.LINE, 12, 1.2)
        f.rect(X[0], y, 7, h, c, "none", 3)
        cf.chip(f, X[0] + PAD, y + 36, c, item, size=18)
        cf.lines(f, X[0] + PAD, y + 66, cf.wrap(trait, 14, WS[0] - 2 * PAD), 14, mr.MUTED, lh=22)
        f.rect(X[1], y, WS[1], h, "#ffffff", cf.LINE, 12, 1.2)
        cf.tb(f, X[1] + PAD, y + 36, WS[1] - 2 * PAD, i1, i2)
        f.rect(X[2], y, WS[2], h, "#f3f1fc", "#d9d2f5", 12, 1.2)
        tag = cf.SECS[sec]
        tw = mr.text_w(tag, 13) + 20
        f.rect(X[2] + PAD, y + 18, tw, 26, VIOLET if sec not in ("-", "x") else cf.IDLE, "none", 13)
        f.text(X[2] + PAD + tw / 2, y + 36, tag, 13, "middle", 700, "#ffffff")
        cf.tb(f, X[2] + PAD, y + 72, WS[2] - 2 * PAD, h1, h2)
        cf.arrow(f, X[0] + WS[0] + 10, X[1] - 10, y + h / 2)
        cf.arrow(f, X[1] + WS[1] + 10, X[2] - 10, y + h / 2)
        y += h + 18
    mr.save(f)


# ------------------------------------------------------------------ 그림 아래 한 줄 결론 (print로 확인한 값)
TAKE = {
    "t1": ("FLOP과 시간은 다른 곳에 감 — FLOP은 문맥 쪽(full attention · 색인)으로, 시간은 여전히 가중치 읽기(MoE)가 정함",
           "요청 256개 · 128K: V3.2 FLOP 55%가 색인이지만 시간은 MoE 65% · 색인 28% / V3.1은 attention이 FLOP 97% · 시간 80%로 둘이 일치하는 유일한 경우"),
    "t2": ("FLOP은 98% 이상이 Tensor core — 그런데 연산이 시간을 정하는 것은 full(V3.1)의 attention뿐",
           "연산형 시간 몫: V3.1 22% → 44%(32개) → 81%(256개 · 128K) / sparse · hybrid는 어느 조건에서도 8% 이하 · K3.1 · Q3.8은 거의 전부 BF16"),
    "t3": ("full은 문맥 약 4천부터 attention FLOP이 가중치 GEMM을 넘음 — sparse는 역전을 4만(V3.2) · 23만(V4)으로 늦추고, 그 뒤 늘어나는 FLOP은 색인",
           "128K 토큰당 FLOP: V3.1 2.3 TFLOP(97%가 attention) → V3.2 242 GFLOP(55%가 색인) / hybrid는 1M 근처(K3.1) · 지원 길이 밖(Q3.8)"),
    "t4": ("요청을 모아도 attention 본체 · 색인 · state의 강도는 그대로 — 구조가 정한 값이 연산형 · 대역폭형을 가름",
           "attention 본체 180 ~ 420(연산형, Q3.8만 12) · 색인 4 ~ 121 · state 약 0.9 / 투영은 요청 256개에 250 ~ 390까지 오르지만 MoE는 6 ~ 26에 머묾"),
    "t5": ("top-k · gather는 FLOP이 0이지만 양이 큼 — 덩어리 크기가 같은 바이트를 몇 번에 나눠 읽는지 정함",
           "요청 256개 · 128K 스텝당 gather 덩어리: V3.2 3,200만 개(656 B) · K3.1 1,260만 개(1,152 B) · V4 275만 개 · Q3.8 157만 개(8 KB 블록)"),
    "t6": ("sparse 층은 색인 점수 → top-k로 층마다 직렬 단계 2개가 늘어남 — 층 수를 곱하면 모델 전체 직렬 단계가 정해짐",
           "모델 전체: V3.1 366 → V3.2 488 · K3.1 537(KDA 345 + DSA 192) · Q3.8 240 · V4 214 — 이론 하한에 없는 커널 실행 · 동기화 지연이 이 수만큼 쌓임"),
}


# ------------------------------------------------------------------ 본문 수치
def show():
    for B, T, lab, _ in CONDS + [(32, T128, "요청 32개 · 128K", "")]:
        print(f"===== {lab}")
        for mid in MODELS:
            rs = ops.rows(mid, B, T)
            F, Tt = sum(r["flops"] for r in rs), sum(r["t"] for r in rs)
            fc, tc, bt, bu = ops.by_cat(rs, "flops"), ops.by_cat(rs, "t"), ops.bound_time(rs), ops.by_unit(rs)
            print(f"  {TAG[mid]:5s} FLOP {ff(F):>13s} time {ft(Tt):>9s} cmp-bound {bt['cmp'] / Tt:.1%}")
            print("     flop", {k: pc(v / F) for k, v in fc.items() if v})
            print("     time", {k: pc(v / Tt) for k, v in tc.items() if v})
            print("     unit", {f"{k[0]}/{k[1]}": pc(v / F) for k, v in bu.items()})
    print("===== intensity (128K)")
    for mid in MODELS:
        print(f"  {TAG[mid]:5s}", {c: [None if ops.intensity(mid, B, T128, [c]) is None else round(ops.intensity(mid, B, T128, [c]), 1)
                                     for B in (1, 32, 256)] for c in ("attn", "idx", "state", "proj", "moe", "head")})
    print("===== per token 128K / cross")
    for mid in MODELS:
        p = ops.per_token(mid, T128)
        print(f"  {TAG[mid]:5s} cross {ops.cross_ctx(mid)} ratio128K {ratio(mid, T128):.2f}", {k: ff(v) for k, v in p.items() if v})
    print("===== irregular / chains")
    for mid in MODELS:
        print(f"  {TAG[mid]:5s}", ops.irregular(mid))
        for c in ops.chains(mid):
            print("       ", c["name"], c["n"], c["src"], "depth", max(l for _, l in c["chain"]))
        kt = ops.kernel_times(mid, 1, T128)
        print("        kernels <1us at B1 128K:", [k for k, v in kt.items() if v < 1], {k: round(v, 2) for k, v in kt.items()})


if __name__ == "__main__":
    if sys.argv[1:] == ["print"]:
        show()
    else:
        for k in ("ops1", "ops2", "ops3"):
            cf.gen_fig(k)
        fig_cat()
        fig_unit()
        fig_growth()
        fig_ai()
        fig_irr()
        fig_chain()
        fig_sum()
