"""Current presentation figures: overview story + direct, family-grouped charts.

Run: python arch_insight/figures/make_revised_figures.py
Reference overview/diagram assets are reused from figures-claude without editing it.
Main KV is BF16; index formats follow the stated method scenarios (see notes).
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
DATA = ROOT / "data"
REFERENCE = ROOT / "figures-claude"
RAW = REPO / "reports/llm-memory-requirements/data/raw"
S = 1_000_000
GB, MB = 1e9, 1e6
FULL, SELECT, COMPRESS, STATE = "#9C9A92", "#2A78D6", "#EB6834", "#1BAF7A"
LIGHT_SELECT, LIGHT_COMPRESS = "#B7D3F6", "#F8C6B1"
INK, MUTED, GRID = "#2B2F3A", "#6B7080", "#E6E8F1"
SCENARIO = "main KV BF16; DSA index FP8+FP32 scale; CSA FP4 payload; QSA index BF16 assumed; GDN matrix FP32"

plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 13,
                     "svg.fonttype": "none", "axes.unicode_minus": False,
                     "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK})


def cfg(filename):
    c = json.loads((RAW/filename).read_text(encoding="utf-8"))
    return c.get("text_config", c)


D = cfg("cfg_deepseek-ai_DeepSeek-V3.2.json")
V = cfg("cfg_deepseek-ai_DeepSeek-V4-Pro.json")
Q = cfg("cfg_Qwen_Qwen3.8-Flash-Next.json")
MLA = 2*(D["kv_lora_rank"]+D["qk_rope_head_dim"])
DSA_INDEX = D["index_head_dim"]+4  # one FP32 scale for the 128 FP8 elements
GQA = 2*Q["num_key_value_heads"]*Q["head_dim"]*2
V4 = V["head_dim"]*2
CSA_INDEX = V["index_head_dim"]//2  # quantization scale/padding are excluded, explicitly
QSA_INDEX = Q["indexer_head_dim"]*2
GDN = (Q["linear_num_value_heads"]*Q["linear_key_head_dim"]*Q["linear_value_head_dim"]*4
       +(2*Q["linear_num_key_heads"]*Q["linear_key_head_dim"]+Q["linear_num_value_heads"]*Q["linear_value_head_dim"])
       *(Q["linear_conv_kernel_dim"]-1)*2)
METHODS = ["MLA", "DSA", "CSA", "HCA", "GQA", "QSA", "GDN"]


def cost(kind, s):
    index = main = window = state = read_main = read_index = write_state = 0
    if kind in ("MLA", "DSA"):
        main = s*MLA
        read_main = main if kind == "MLA" else min(s, D["index_topk"])*MLA
        if kind == "DSA":
            index = read_index = s*DSA_INDEX
    elif kind in ("CSA", "HCA"):
        n = s//(4 if kind == "CSA" else 128)
        main, window = n*V4, min(s, V["sliding_window"])*V4
        read_main = (min(n, V["index_topk"]) if kind == "CSA" else n)*V4+window
        if kind == "CSA":
            index = read_index = n*CSA_INDEX
    elif kind in ("GQA", "QSA"):
        main = s*GQA
        read_main = main
        if kind == "QSA":
            n, tail = divmod(s, Q["indexer_compress_ratio"])
            index = read_index = n*QSA_INDEX
            read_main = (min(n, Q["indexer_budget"]//4)*4+tail)*GQA
    elif kind == "GDN":
        state = read_main = write_state = GDN
    else:
        raise ValueError(kind)
    limit = D["max_position_embeddings"] if kind in ("MLA", "DSA") else V["max_position_embeddings"] if kind in ("CSA", "HCA") else Q["max_position_embeddings"]
    return dict(method=kind, tokens=s, index_read_bytes=read_index,
                main_read_bytes=read_main, total_read_bytes=read_index+read_main,
                index_store_bytes=index, main_store_bytes=main+window+state,
                total_store_bytes=index+main+window+state, state_write_bytes=write_state,
                within_config=s<=limit,
                omitted="compression state + CSA quant scales" if kind in ("CSA", "HCA") else "allocation/padding + scratch + append traffic",
                scenario=SCENARIO)


def export_data():
    DATA.mkdir(parents=True, exist_ok=True)
    rows = [cost(k, s) for s in (2048, 8192, 131072, S, 10*S) for k in METHODS]
    with (DATA/"revised-deck-costs.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lineage = list(csv.DictReader((REPO/"reports/llm-memory-requirements/data/lineage-kv1m.csv").open(encoding="utf-8-sig")))
    halves = defaultdict(Counter)
    for row in lineage:
        h = row["date"][:4]+" H"+str(1 if int(row["date"][5:7])<=6 else 2)
        halves[h][row["category"]] += 1
    summary = {"at_1m": {k:cost(k, S) for k in METHODS}, "half_year_counts": dict(halves),
               "gdn_state_bytes_per_layer": GDN, "qwen_state_layers": Q["layer_types"].count("linear_attention"),
               "scenario": SCENARIO}
    (DATA/"revised-deck-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    assert sum(sum(c.values()) for c in halves.values()) == 55
    assert sum(halves["2026 H2"].values()) == 15 and halves["2026 H2"]["full"] == 1
    assert cost("DSA", S)["total_read_bytes"] == 134359296
    assert cost("CSA", S)["total_read_bytes"] == 17179648
    assert cost("HCA", S)["total_read_bytes"] == 8130560
    assert cost("QSA", S)["total_read_bytes"] == 68194304
    return summary


def fmt(value):
    if value >= GB:
        return f"{value/GB:.2f} GB"
    if value >= 100*MB:
        return f"{value/MB:.0f} MB"
    if value >= 10*MB:
        return f"{value/MB:.1f} MB"
    return f"{value/MB:.1f} MB"


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():
            continue
        b = t.get_window_extent(renderer)
        if b.x0 < -3 or b.y0 < -3 or b.x1 > fig.bbox.width+3 or b.y1 > fig.bbox.height+3:
            clipped.append(t.get_text())
    if clipped:
        raise ValueError(f"{name}: text outside canvas: {clipped}")
    fig.savefig(HERE/f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(HERE/f"{name}.svg", facecolor="white")
    plt.close(fig)
    print("ok", name)


def read_store():
    fig = plt.figure(figsize=(16, 9))
    fig.text(.025, .96, "층당 · 요청 1개 · 1M=100만 토큰 · Decode 1스텝 · KV BF16 · 색인 정밀도는 방식별 설정", color=MUTED, fontsize=12.5)
    fig.text(.74, .96, "옅은 색: 색인   /   진한 색: KV·state", color=MUTED, fontsize=12.5)
    labels = ["전부 읽기 · MLA", "골라 읽기 · DSA", "4줄 → 1줄 · CSA", "128줄 → 1줄 · HCA", "전부 읽기 · GQA", "골라 읽기 · QSA", "Linear state"]
    subs = ["DeepSeek-V3", "DeepSeek-V3.2", "DeepSeek-V4", "DeepSeek-V4", "Qwen · Full 층", "Qwen · QSA 층", "Qwen · GDN 층"]
    ys = [7.0, 6.0, 5.0, 4.0, 2.55, 1.55, .55]
    colors = [FULL, SELECT, COMPRESS, COMPRESS, FULL, SELECT, STATE]
    lights = [FULL, LIGHT_SELECT, LIGHT_COMPRESS, LIGHT_COMPRESS, FULL, LIGHT_SELECT, STATE]
    leftlabels = fig.add_axes([.025, .13, .245, .72]); leftlabels.axis("off")
    leftlabels.set(ylim=(-.2, 8), xlim=(0, 1))
    leftlabels.text(0, 7.8, "DeepSeek", fontsize=12, color=MUTED, weight="bold")
    leftlabels.text(0, 3.35, "Qwen", fontsize=12, color=MUTED, weight="bold")
    for y, lab, sub in zip(ys, labels, subs):
        leftlabels.text(0, y+.08, lab, fontsize=15, weight="bold", va="center")
        leftlabels.text(0, y-.25, sub, fontsize=10.5, color=MUTED)
    for xpos, heading, part in [(.28, "디코드마다 읽는 양", "read"), (.64, "계속 보관할 크기", "store")]:
        ax = fig.add_axes([xpos, .13, .295, .72])
        ax.set(ylim=(-.2, 8), xlim=(0, 2.8), xticks=[0, 1, 2], yticks=[])
        ax.set_xticklabels(["0 GB", "1 GB", "2 GB"], fontsize=11)
        ax.set_title(heading, loc="left", fontsize=18, weight="bold", pad=18)
        ax.set_axisbelow(True); ax.grid(axis="x", color=GRID)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.tick_params(length=0)
        for method, y, col, light in zip(METHODS, ys, colors, lights):
            r = cost(method, S)
            idx, main = r[f"index_{part}_bytes"], r[f"main_{part}_bytes"]
            if idx: ax.barh(y, idx/GB, height=.44, color=light)
            # Tiny values remain to scale; labels carry the value, no minimum-width inflation.
            ax.barh(y, main/GB, left=idx/GB, height=.44, color=col)
            total = idx+main
            baseline = cost("MLA" if method in ("MLA", "DSA", "CSA", "HCA") else "GQA", S)[f"total_{part}_bytes"]
            label = fmt(total)
            if method == "GDN":
                label += " + 같은 양 쓰기" if part == "read" else " (고정)"
            elif method not in ("MLA", "GQA"):
                fraction = total/baseline
                label += f"  (1/{1/fraction:.0f})" if fraction<.95 else f"  (×{fraction:.2f})"
            ax.text(total/GB+.035, y, label, fontsize=12.3, weight="bold" if part=="read" else "normal", va="center")
    fig.text(.025, .035, "1M은 방식 비교용 계산 길이. 실제 지원 길이·양자화 scale·압축 대기 등은 부록에 구분. 읽기 절감은 속도 배수가 아님.", fontsize=11, color=MUTED)
    save(fig, "revised-05-read-vs-store")


def gain():
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(left=.08, right=.79, top=.85, bottom=.19)
    fig.text(.03, .95, "층당 · Decode 1스텝 · 각 계열의 Full 대비 읽기 절감 배수 · 속도 배수가 아님", fontsize=12, color=MUTED)
    xs = np.unique(np.r_[np.geomspace(1024, S, 250).astype(int), 8192, 131072, D["max_position_embeddings"], Q["max_position_embeddings"]])
    for method, base, color, limit in [("DSA", "MLA", SELECT, D["max_position_embeddings"]),
                                      ("QSA", "GQA", "#1594B5", Q["max_position_embeddings"]),
                                      ("CSA", "MLA", COMPRESS, V["max_position_embeddings"])]:
        vals = np.array([cost(base, int(s))["total_read_bytes"]/cost(method, int(s))["total_read_bytes"] for s in xs])
        ax.plot(xs[xs<=limit], vals[xs<=limit], color=color, lw=3)
        if max(xs)>limit:
            mask=xs>=limit; ax.plot(xs[mask], vals[mask], color=color, lw=2.5, ls=(0,(4,3)))
        r = cost(base, S)["total_read_bytes"]/cost(method, S)["total_read_bytes"]
        ax.text(S*1.08, r, f"{method}  약 {r:.0f}×", color=color, fontsize=15, weight="bold", va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlim=(1024, 1.15*S), ylim=(.6, 100), xticks=[1024,8192,131072,S], yticks=[1,3,10,30,100])
    ax.set_xticklabels(["1K", "8K", "128K", "1M"]); ax.set_yticklabels(["1×", "3×", "10×", "30×", "100×"])
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(color=GRID); ax.set_axisbelow(True)
    ax.axhline(1, color=INK, lw=1.5)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0, pad=9)
    ax.set_xlabel("문맥 길이 (토큰)", labelpad=12)
    ax.set_ylabel("Full 대비 읽기 절감", labelpad=12)
    ax.annotate("DSA: 약 9배에서 포화\n색인을 매번 전부 훑기 때문", xy=(4e5,8.3), xytext=(2e4,4),
                fontsize=12, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED),
                bbox=dict(facecolor="white", edgecolor="none", pad=3))
    fig.text(.03, .045, "실선: config 길이 이내 · 점선: 비용식 외삽  |  KV BF16, DSA 색인 FP8+scale, CSA FP4 payload, QSA BF16 가정", fontsize=10.5, color=MUTED)
    save(fig, "revised-06-read-saving")


def behavior():
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_axes([.025,.08,.95,.78]); ax.axis("off"); ax.set(xlim=(0,1),ylim=(0,1))
    fig.text(.025,.95,"요청 1개 · 1M · B=1 · 모델 전체 데이터 묶음별 · 보유량 대비 논리 읽기·쓰기",fontsize=12,color=MUTED)
    cols=[(.0,"데이터"),(.18,"보유량"),(.32,"매 스텝 읽기"),(.57,"매 스텝 쓰기"),(.82,"검토할 메모리 특성")]
    for x,label in cols: ax.text(x,.98,label,fontsize=13,color=MUTED,weight="bold")
    rows=[
        ("가중치","모든 요청이 공유","모델 고정",None,"활성 부분","expert·batch에 따라 다름",0,"0","추론 중 값이 바뀌지 않음","재사용·공급 대역폭",FULL),
        ("본 KV","DSA · 61층",f"{61*MLA*S/GB:.1f} GB",2048/S,"0.2%","고른 2,048줄만",1/S,"한 줄 추가","과거 KV는 그대로","큰 용량·랜덤 접근",SELECT),
        ("색인 키","DSA · 61층",f"{61*DSA_INDEX*S/GB:.1f} GB",1,"100%","매번 전부 훑음",1/S,"한 줄 추가","새 색인 키만 추가","순차 읽기 대역폭",SELECT),
        ("요약 state","GDN · 36층",f"{36*GDN/MB:.0f} MB",1,"100%","매번 읽고 사용",1,"100%","매번 갱신","온칩 처리·쓰기",STATE),
    ]
    for i,(name,sub,size,rd,rdlabel,rdnote,wr,wrlabel,wrnote,mem,color) in enumerate(rows):
        y=.80-i*.22
        if i%2==0: ax.add_patch(plt.Rectangle((-.008,y-.08),1.015,.18,color="#F5F6FA",zorder=-1,clip_on=False))
        ax.text(0,y+.035,name,fontsize=18,weight="bold",color=color)
        ax.text(0,y-.025,sub,fontsize=10,color=MUTED)
        ax.text(.18,y+.01,size,fontsize=18,weight="bold")
        for x,frac,lab,note in[(.32,rd,rdlabel,rdnote),(.57,wr,wrlabel,wrnote)]:
            if frac is None:
                ax.text(x,y+.01,lab,fontsize=15,weight="bold")
            else:
                ax.barh(y+.02,.145,left=x,height=.045,color="#E7EAF1")
                ax.barh(y+.02,.145*frac,left=x,height=.045,color=color)
                ax.text(x+.155,y+.02,lab,fontsize=12.5,weight="bold",va="center")
            ax.text(x,y-.04,note,fontsize=10,color=MUTED)
        ax.text(.82,y+.01,mem,fontsize=12.5,weight="bold")
    fig.text(.025,.02,"고정 상태 전체가 SRAM에 들어간다는 뜻은 아님. 현재 층×활성 요청의 작업 집합과 실제 HBM 이동을 확인한다.",fontsize=11,color=MUTED)
    save(fig,"revised-07-memory-behavior")


def copy_reference():
    mapping={"01-trend":"revised-01-trend", "02-moe-analogy":"revised-02-moe-analogy",
             "03-kv-axes":"revised-03-kv-axes", "04-layer-strips":"revised-04-layers"}
    for src,dst in mapping.items():
        for ext in ("png","svg"):
            shutil.copyfile(REFERENCE/f"{src}.{ext}",HERE/f"{dst}.{ext}")
        print("reused",dst)


if __name__=="__main__":
    result=export_data()
    copy_reference()
    read_store(); gain(); behavior()
    for kind in METHODS:
        r=result["at_1m"][kind]
        print(kind, "read",fmt(r["total_read_bytes"]),"store",fmt(r["total_store_bytes"]))
