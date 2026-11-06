"""Build six 16:9 figures for 02-deck-draft.md (PNG + editable SVG).

Run: python arch_insight/figures/make_figures.py
Requires matplotlib and numpy. No model execution or GPU is required.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("attention_cost_model", ROOT / "analysis" / "cost_model.py")
model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(model)
SUMMARY = model.generate()
model.validate(SUMMARY)
P = SUMMARY["parameters"]
R = SUMMARY["at_128k"]
MIB = model.MIB

INK = "#23314A"
MUTED = "#64748B"
GRID = "#E5EAF1"
GREY = "#8B95A5"
BLUE = "#3268DA"
CYAN = "#009DBD"
ORANGE = "#D88922"
GREEN = "#15956C"
DARK_GREEN = "#086B4B"
PALE_BLUE = "#EDF3FF"
PALE_GREEN = "#EAF6F0"
PALE_ORANGE = "#FFF4E5"

plt.rcParams.update({
    "font.family": "Malgun Gothic", "font.size": 13,
    "axes.unicode_minus": False, "svg.fonttype": "none",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": INK,
    "text.color": INK, "figure.facecolor": "white",
})
TEXT_EXTENTS = []


def canvas(title: str, subtitle: str, footer: str):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
    fig.text(.05, .94, title, fontsize=23, weight="bold", va="top")
    fig.text(.05, .865, subtitle, fontsize=11.5, color=MUTED, va="top")
    fig.text(.05, .047, footer, fontsize=10.1, color=MUTED, va="bottom")
    return fig


def drawing(fig):
    ax = fig.add_axes([.04, .15, .92, .66])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return ax


def box(ax, x, y, w, h, face="white", edge=GRID, radius=.014, zorder=1):
    patch = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          linewidth=1.1, facecolor=face, edgecolor=edge,
                          transform=ax.transAxes, zorder=zorder)
    ax.add_patch(patch)
    return patch


def txt(ax, x, y, text, size=14, color=INK, weight="normal", ha="left", va="center"):
    return ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, transform=ax.transAxes, linespacing=1.5)


def tiles(ax, x, y, count=8, selected=None, color=BLUE, width=.037, height=.09, gap=.012):
    selected = set(range(count)) if selected is None else set(selected)
    for i in range(count):
        c = color if i in selected else "#E8EDF3"
        box(ax, x+i*(width+gap), y, width, height, c, c, radius=.005)


def clean(ax):
    for edge in ("top", "right", "left"):
        ax.spines[edge].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=.8)


def save(fig, name):
    HERE.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    # Detect figure-boundary clipping; manual review also checks label overlap.
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < -2 or bounds.y0 < -2 or bounds.x1 > fig.bbox.width+2 or bounds.y1 > fig.bbox.height+2:
            # Invisible/out-of-range tick labels can still be instantiated by mpl.
            if text.axes is not None and not text.axes.get_visible():
                continue
            clipped.append(text.get_text())
    TEXT_EXTENTS.append((name, clipped))
    fig.savefig(HERE / f"{name}.png", dpi=150)
    fig.savefig(HERE / f"{name}.svg")
    plt.close(fig)


def fig01():
    fig = canvas("MoE가 전문가를 고르듯, Sparse는 읽을 과거를 고른다",
                 "동작 개념도  |  칸 수·선택 비율은 설명용이며 실제 모델의 수치가 아님",
                 "공통점: 선택해도 보관할 데이터는 남는다.  차이: 가중치는 요청 간 공유, KV는 기본적으로 요청마다 별도.")
    ax = drawing(fig)
    for left, color, head, selector, outcome, bottom in [
        (.01, ORANGE, "FFN → MoE", "라우터", "전문가 일부 실행", "전문가 가중치: 모델이 함께 보유"),
        (.52, BLUE, "Full → Sparse attention", "색인·선택기", "과거 KV 일부 조회", "과거 KV: 요청마다 길이에 따라 증가"),
    ]:
        box(ax, left, .19, .47, .77, "#FAFBFD")
        txt(ax, left+.235, .87, head, 18, color, "bold", "center")
        box(ax, left+.135, .65, .20, .11, face="white", edge=color)
        txt(ax, left+.235, .705, selector, 14, color, "bold", "center")
        ax.annotate("", xy=(left+.235, .49), xytext=(left+.235, .63),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        tiles(ax, left+.042, .37, selected=[1, 4, 6], color=color)
        txt(ax, left+.235, .275, outcome, 15, weight="bold", ha="center")
        txt(ax, left+.235, .10, bottom, 12, MUTED, ha="center")
    save(fig, "01-moe-analogy")


def fig02():
    fig = canvas("원본 선택·이력 압축·요약 상태는 줄이는 대상이 다르다",
                 "구조 지도  |  DeepSeek-V3/V3.2/V4 · Qwen3.8-Flash-Next 공개 구조 기반",
                 "GQA/MLA는 KV 표현 방식.  HCA는 압축 이력을 전부 읽는다.  Hybrid의 요약 상태는 매 토큰 갱신한다.")
    ax = drawing(fig)
    cards = [
        ("전체 조회", "Full", GREY, "저장된 KV 모두 읽기"),
        ("선택 조회", "DSA · QSA", BLUE, "큰 KV는 일부만 읽기"),
        ("이력 압축", "CSA · HCA", ORANGE, "여러 토큰을 묶어 저장"),
        ("고정 상태 + 조회", "Hybrid", GREEN, "요약층과 조회층을 혼합"),
    ]
    for i, (head, label, c, desc) in enumerate(cards):
        x = .008+i*.25
        box(ax, x, .49, .235, .48, "#FAFBFD")
        txt(ax, x+.1175, .89, head, 15, c, "bold", "center")
        txt(ax, x+.1175, .81, label, 12, MUTED, ha="center")
        if i < 2:
            tiles(ax, x+.027, .65, 6, None if i == 0 else [0, 4], c, .023, .075, .009)
        elif i == 2:
            tiles(ax, x+.024, .70, 6, color="#D5DCE6", width=.023, height=.04, gap=.009)
            ax.annotate("", xy=(x+.116, .595), xytext=(x+.116, .68), arrowprops=dict(arrowstyle="->", color=c, lw=1.7))
            tiles(ax, x+.08, .565, 2, color=c, width=.027, height=.052, gap=.018)
        else:
            for j in range(4):
                box(ax, x+.055+j*.033, .65, .025, .075,
                    GREEN if j < 3 else BLUE, "none", radius=.003)
        txt(ax, x+.1175, .535, desc, 11.5, ha="center")
    txt(ax, .02, .345, "Qwen3.8-Flash-Next", 15, weight="bold")
    txt(ax, .98, .345, "36층 고정 상태 + 12층 QSA", 16, GREEN, "bold", "right")
    for j in range(P["layers"]):
        c = BLUE if (j+1) % 4 == 0 else GREEN
        box(ax, .02+j*.02, .19, .016, .09, c, "none", radius=.002)
    txt(ax, .02, .095, "길이에 따라 원본 KV를 쌓는 층", 13, MUTED)
    txt(ax, .98, .095, "12 / 48 = 25%", 21, BLUE, "bold", "right")
    save(fig, "02-methods-and-hybrid")


def fig03():
    ratio = R["Full MLA"]["read_bytes"] / R["DSA"]["read_bytes"]
    fig = canvas(f"DSA: 읽기는 {ratio:.1f}배 적게, 보관량은 22% 더 많게",
                 "Decode · 128K · B=1 · 층당 · 동일 MLA KV 표현 · KV·색인 BF16 구조 비교(계산값)",
                 "쿼리 1개 = 새 토큰 1개의 모든 헤드.  KV·색인 payload 기준이며, 실제 HBM 전송량·속도 배수가 아님.")
    for pos, key, head in [([.12, .31, .34, .43], "read_bytes", "매 스텝 읽기 (MiB)"),
                            ([.59, .31, .34, .43], "stored_modelled_bytes", "요청당 보유 (MiB)")]:
        ax = fig.add_axes(pos)
        vals = [R[m][key]/MIB for m in ("Full MLA", "DSA")]
        ax.barh([1, 0], vals, height=.38, color=[GREY, BLUE])
        ax.set(yticks=[1, 0], yticklabels=["Full MLA", "DSA"], xlim=(0, 210), ylim=(-.6, 1.6), xticks=[0, 50, 100, 150, 200])
        ax.set_title(head, loc="left", fontsize=17, pad=20, weight="bold")
        clean(ax)
        for y, value in zip([1, 0], vals):
            label = f"{value:,.2f}".rstrip("0").rstrip(".")
            ax.text(value+4, y, label, fontsize=16, va="center", weight="bold")
    fig.text(.5, .18, "선택하지 않은 KV도 다음 쿼리를 위해 계속 보관", ha="center", fontsize=17, color=BLUE, weight="bold")
    save(fig, "03-read-vs-storage")


def fig04():
    fig = canvas("찾기·가져오기·상태 갱신이 서로 다른 비용을 만든다",
                 "Decode · 128K · B=1 · 대표 층당 조회/상태 I/O payload(계산값) · 각 모델의 차원은 서로 다름",
                 "KV·색인 BF16, GDN 행렬 state FP32·conv BF16.  추가 쓰기·압축 생성·임시 버퍼 제외.  물리 HBM 트래픽 아님.")
    ax = fig.add_axes([.13, .25, .79, .49])
    names = ["DSA", "CSA", "QSA", "GDN"]
    ys = np.arange(len(names))[::-1]
    left = np.zeros(4)
    pieces = [
        ("index_read_bytes", BLUE, "색인 읽기"),
        ("kv_read_bytes", ORANGE, "조회 KV 읽기"),
        ("window_read_bytes", ORANGE, None),
        ("state_read_bytes", GREEN, "state 읽기"),
        ("state_write_bytes", DARK_GREEN, "state 쓰기"),
    ]
    for key, color, label in pieces:
        vals = np.array([R[m][key]/MIB for m in names])
        ax.barh(ys, vals, left=left, height=.46, color=color, label=label)
        left += vals
    for y, total in zip(ys, left):
        ax.text(total+.6, y, f"{total:.2f}", fontsize=16, va="center", weight="bold")
    ax.text(16, 3, "색인 32", fontsize=14, color="white", ha="center", va="center", weight="bold")
    ax.text(4, 2, "8", fontsize=14, color="white", ha="center", va="center", weight="bold")
    ax.text(4, 1, "8", fontsize=14, color="white", ha="center", va="center", weight="bold")
    ax.set(yticks=ys, yticklabels=names, xlim=(0, 39), ylim=(-.6, 3.6), xticks=[0, 10, 20, 30])
    ax.set_xlabel("조회·상태 갱신량 (MiB / 스텝 / 층)", labelpad=10)
    clean(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(-.01, 1.08), ncol=4, frameon=False, fontsize=11.5, handlelength=1.1, columnspacing=1.5)
    fig.text(.5, .145, "DSA는 읽기의 93%가 색인  |  GDN은 3.06 MiB 읽기 + 3.06 MiB 쓰기", fontsize=15, color=INK, ha="center")
    save(fig, "04-operation-bytes")


def fig05():
    fig = canvas("선택량이 고정돼도, 색인 검색 비용은 계속 증가한다",
                 "Decode · B=1 · KV·색인 BF16 · 각 방식과 동일 KV 표현의 Full 대비 논리 읽기 절감 배수",
                 "실선: 공개 config 길이 이내  /  점선: 그 밖의 비용식 외삽.  속도·정확도·모델 전체 성능을 뜻하지 않음.")
    ax = fig.add_axes([.10, .24, .76, .55])
    samples = sorted(set(int(x) for x in np.geomspace(8*1024, 1024*1024, 180)) | {128*1024, P["dsa_limit"], P["qsa_limit"]})
    for method, full, color, limit in [("DSA", "Full MLA", BLUE, P["dsa_limit"]), ("QSA", "Full GQA", CYAN, P["qsa_limit"])]:
        xs = np.array(samples)
        vals = np.array([model.layer_cost(full, int(s), P)["read_bytes"] / model.layer_cost(method, int(s), P)["read_bytes"] for s in xs])
        supported = xs <= limit
        extrapolated = xs >= limit
        ax.plot(xs[supported], vals[supported], color=color, lw=3.2)
        ax.plot(xs[extrapolated], vals[extrapolated], color=color, lw=2.6, ls=(0, (4, 3)))
        i128 = samples.index(128*1024)
        ax.scatter([128*1024], [vals[i128]], s=50, color=color, zorder=4)
        ax.annotate(f"128K: {vals[i128]:.1f}×", (128*1024, vals[i128]), xytext=(-68, 18), textcoords="offset points", color=color, fontsize=12, weight="bold")
        ceiling = P["mla_entry_bytes"]/P["dsa_index_bytes"] if method == "DSA" else P["gqa_entry_bytes"]/(P["qsa_index_bytes"]/P["qsa_ratio"])
        ax.text(xs[-1]*1.09, vals[-1], f"{method}\n상한 {ceiling:g}×", fontsize=14, color=color, weight="bold", va="center")
    ax.axhline(1, color=GREY, lw=1)
    ax.set_xscale("log", base=2)
    ax.set(xlim=(8*1024, 2*1024*1024), ylim=(0, 35), yticks=[1, 8, 16, 24, 32])
    ax.xaxis.set_major_locator(FixedLocator([8*1024, 32*1024, 128*1024, 1024*1024]))
    ax.set_xticklabels(["8K", "32K", "128K", "1M"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("문맥 길이 (토큰)", labelpad=12)
    ax.set_ylabel("읽기 절감 배수 (Full = 1×)", labelpad=12)
    clean(ax)
    ax.grid(axis="y", color=GRID, lw=.8)
    fig.text(.5, .13, "색인 자체를 줄이거나, 여러 층에서 선택 결과를 재사용할 여지가 남는다", fontsize=15, color=INK, ha="center")
    save(fig, "05-indexing-ceiling")


def fig06():
    fig = canvas("크게 보관할 KV와 자주 고칠 state를 구분해 설계한다",
                 "128K · B=1 · 대표 층의 데이터 묶음별 · KV·색인 BF16 / GDN state FP32 · 양 축 로그 눈금",
                 "X축은 고유 논리 읽기량÷보유량이며 BW 사용률이 아님.  가중치는 요청 간 공유하는 읽기 전용 데이터로 별도 취급.")
    ax = fig.add_axes([.10, .25, .54, .53])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(.007, 2.2), ylim=(.06, 320), xticks=[.01, .1, 1], yticks=[.125, 1, 8, 32, 128])
    ax.set_xticklabels(["1%", "10%", "100%"])
    ax.set_yticklabels(["0.125", "1", "8", "32", "128"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("스텝당 읽는 비율", labelpad=10)
    ax.set_ylabel("데이터 보유량 (MiB)", labelpad=10)
    clean(ax)
    ax.grid(axis="y", color=GRID, lw=.8)
    points = [
        (P["dsa_k"]/(128*1024), R["DSA"]["stored_main_bytes"]/MIB, BLUE, "o", "DSA 본 KV\n144 MiB · 1.56%", (10, -8), "left"),
        (1, R["DSA"]["stored_index_bytes"]/MIB, BLUE, "o", "DSA 색인\n32 MiB · 100%", (-14, 12), "right"),
        (1, SUMMARY["state_mib_per_layer"], GREEN, "s", "GDN state\n3.06 MiB · 읽고 갱신", (-14, -5), "right"),
        (1, R["CSA"]["stored_window_bytes"]/MIB, ORANGE, "D", "최근 128토큰 창\n0.125 MiB · 교체", (-14, -6), "right"),
    ]
    for x, y, c, marker, text, offset, ha in points:
        ax.scatter([x], [y], s=100, color=c, marker=marker, zorder=4, edgecolors="white", linewidths=1.3)
        ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", color=c, fontsize=11.3, weight="bold", ha=ha, va="center")
    side = fig.add_axes([.70, .20, .26, .61])
    side.set(xlim=(0, 1), ylim=(0, 1))
    side.axis("off")
    for y, c, face, title, body in [
        (.69, BLUE, PALE_BLUE, "① KV 용량 계층", "큰 KV를 보관\n접근 지연·batch 부하 확인"),
        (.375, ORANGE, PALE_ORANGE, "② scan·gather 경로", "주소·동시 read·prefetch\n실측 지연이 큰 단계부터"),
        (.06, GREEN, PALE_GREEN, "③ state 온칩 보관", "자주 읽고 고치는 상태\n요청 수 × 크기가 SRAM 안에"),
    ]:
        box(side, 0, y, 1, .26, face, "none")
        txt(side, .07, y+.19, title, 14, c, "bold")
        txt(side, .07, y+.085, body, 11.2, MUTED)
    fig.text(.10, .13, "● 한 번 추가 후 반복 읽기     ■ 매 토큰 갱신     ◆ 오래된 항목 교체", fontsize=11.5, color=MUTED)
    save(fig, "06-memory-map")


if __name__ == "__main__":
    for make in (fig01, fig02, fig03, fig04, fig05, fig06):
        make()
    for name, clipped in TEXT_EXTENTS:
        print(f"{name}: " + (f"check clipped text {clipped}" if clipped else "text within canvas"))
    if any(clipped for _, clipped in TEXT_EXTENTS):
        raise SystemExit("Some text extends beyond the canvas; review before publishing.")
    print("Wrote 6 PNGs (1920x1080) and 6 SVGs.")
