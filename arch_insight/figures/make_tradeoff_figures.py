"""Original analytical charts for architecture decisions (no borrowed figures).

Run: python arch_insight/figures/make_tradeoff_figures.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location("tradeoff_model",ROOT/"analysis/architecture_tradeoffs.py")
model=importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(model)
R=model.generate(); model.validate(R)
P=R["parameters"]
BLUE="#2868C8"
CYAN="#138B9C"
ORANGE="#D67E29"
GREEN="#18966F"
DARKGREEN="#0A664B"
INK="#25334B"
MUTED="#6B7788"
GRID="#E5EAF1"
plt.rcParams.update({"font.family":"Malgun Gothic","font.size":13,
                     "svg.fonttype":"none","axes.unicode_minus":False,
                     "text.color":INK,"axes.labelcolor":MUTED,
                     "xtick.color":MUTED,"ytick.color":INK})


def canvas(subtitle,footer):
    fig=plt.figure(figsize=(14,7.875),dpi=160)
    fig.text(.045,.95,subtitle,fontsize=12,color=MUTED,va="top")
    fig.text(.045,.035,footer,fontsize=10.5,color=MUTED,va="bottom")
    return fig


def clean(ax,axis="y"):
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis=axis,color=GRID)
    ax.tick_params(length=0,pad=9)


def save(fig,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    clipped=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text(): continue
        b=t.get_window_extent(renderer)
        if b.x0 < -3 or b.y0 < -3 or b.x1>fig.bbox.width+3 or b.y1>fig.bbox.height+3:
            clipped.append(t.get_text())
    if clipped: raise ValueError((name,clipped))
    fig.savefig(HERE/f"{name}.png",dpi=160,facecolor="white")
    fig.savefig(HERE/f"{name}.svg",facecolor="white")
    plt.close(fig)
    print("ok",name)


def selector():
    fig=canvas("층당 · Decode 1쿼리 · ‘색인 읽기 ÷ (색인 + 선택 KV 읽기)’ · 계산값",
               "DSA FP8 색인+scale / QSA BF16 색인 가정 / CSA FP4 payload. 바이트 비중이며 시간·병목 판정이 아님.")
    ax=fig.add_axes([.09,.26,.72,.57])
    xs=np.geomspace(8192,131072,180).astype(int)
    for name,color,label_y in [("DSA",BLUE,89),("QSA",CYAN,71),("CSA",ORANGE,60)]:
        values=[]
        for s in xs:
            idx,kv=model.sparse_reads(name,int(s),P)
            values.append(idx/(idx+kv)*100)
        ax.plot(xs,values,color=color,lw=3)
        ax.plot([xs[-1],xs[-1]*1.08],[values[-1],label_y],color=color,lw=1)
        ax.text(xs[-1]*1.10,label_y,f"{name} {values[-1]:.0f}%",color=color,fontsize=14,weight="bold",va="center")
        cross=R["selector_crossover_tokens"][name]
        ax.scatter([cross],[50],color=color,s=65,zorder=5)
    ax.axhline(50,color=MUTED,lw=1.5,ls=(0,(4,3)))
    ax.text(8700,53,"50%: 색인과 선택 KV의 읽기량이 같음",fontsize=11.5,color=MUTED)
    ax.set_xscale("log",base=2)
    ax.set(xlim=(8192,190000),ylim=(0,100),xticks=[8192,32768,131072],yticks=[0,25,50,75,100])
    ax.set_xticklabels(["8K","32K","128K"])
    ax.set_yticklabels(["0%","25%","50%","75%","100%"])
    ax.set_xlabel("문맥 길이",labelpad=12); ax.set_ylabel("색인이 차지하는 읽기 비중",labelpad=12)
    clean(ax)
    fig.text(.09,.13,"색인이 절반을 넘는 길이",fontsize=12,color=MUTED)
    for x,name,color in [(.36,"DSA",BLUE),(.56,"QSA",CYAN),(.76,"CSA",ORANGE)]:
        cross=R["selector_crossover_tokens"][name]
        fig.text(x,.13,f"{name} 약 {cross/10000:.1f}만 토큰",fontsize=14,color=color,weight="bold",ha="center")
    save(fig,"decision-02-selector-crossover")


def tier():
    fig=canvas("설계 시나리오 · DSA 61층 · 128K · 요청당 50 token/s · 선택 KV 전부를 용량 계층에서 공급",
               "모델 복제본 1개의 합산 BW 필요조건. 1GB=10^9 B. 증폭·100GB/s 예산은 가정이며 접근 지연·다른 트래픽은 별도.")
    ax=fig.add_axes([.09,.25,.82,.58])
    xs=np.arange(3)
    for offset,amp,color,label in [(-.18,1,BLUE,"읽기 증폭 1배"),(.18,4,ORANGE,"읽기 증폭 4배")]:
        vals=[next(v["required_aggregate_bandwidth_GBps"] for v in R["tier_scenarios"] if v["batch"]==b and v["read_amplification"]==amp) for b in (1,8,32)]
        bars=ax.bar(xs+offset,vals,width=.31,color=color,label=label)
        for bar,value in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,value+16,f"{value:.0f}",ha="center",fontsize=14,weight="bold",color=color)
    ax.axhline(100,color=MUTED,ls=(0,(4,3)),lw=1.5)
    ax.text(2.48,112,"가정: 외부 계층 100 GB/s",ha="right",fontsize=10.5,color=MUTED,
            bbox=dict(facecolor="white",edgecolor="none",pad=2))
    labels=[f"{b}요청\n보관 KV {R['dsa_main_kv_storage_GB_at_128k']*b:.1f} GB" for b in (1,8,32)]
    ax.set(xticks=xs,xticklabels=labels,ylim=(0,1050),yticks=[0,100,250,500,1000],xlim=(-.6,2.6))
    ax.set_ylabel("필요한 외부 계층 합산 대역폭 (GB/s)",labelpad=12)
    clean(ax)
    ax.legend(frameon=False,loc="upper left",ncol=2,fontsize=12)
    fig.text(.5,.11,"8요청: 58 → 230 GB/s  |  조회율이 낮아도 batch·생성 속도·과다 읽기가 BW를 키운다",ha="center",fontsize=14,weight="bold")
    save(fig,"decision-03-tier-bandwidth")


def hybrid():
    fig=canvas("Qwen3.8-Flash-Next · 12 QSA + 36 GDN · B=1 · Decode 1스텝 · 조회·상태 I/O 계산",
               "QSA: KV·색인 BF16 가정. GDN: 상태 한 번 읽기+쓰기. 실제 HBM 트래픽·가중치·FFN·통신은 포함하지 않음.")
    ax=fig.add_axes([.10,.24,.80,.59])
    rows=R["hybrid_io"]
    xs=np.arange(3)
    bottom=np.zeros(3)
    for key,color,label in [("qsa_read_bytes",BLUE,"QSA 조회 읽기"),("state_read_bytes",GREEN,"GDN state 읽기"),("state_write_bytes",DARKGREEN,"GDN state 쓰기")]:
        vals=np.array([r[key]/1e6 for r in rows])
        ax.bar(xs,vals,bottom=bottom,width=.52,color=color,label=label)
        bottom+=vals
    for i,r in enumerate(rows):
        ax.text(i,bottom[i]+18,f"state {r['state_io_share']*100:.0f}%",ha="center",fontsize=16,color=GREEN,weight="bold")
        ax.text(i,r["qsa_read_bytes"]/1e6/2,f"{r['qsa_read_bytes']/1e6:.0f} MB",ha="center",va="center",fontsize=13,color="white",weight="bold")
    ax.set(xticks=xs,xticklabels=["32K","128K","256K"],ylim=(0,570),yticks=[0,100,200,300,400,500])
    ax.set_ylabel("조회·상태 갱신량 (MB/스텝/모델)",labelpad=12)
    ax.set_xlabel("문맥 길이",labelpad=12)
    clean(ax)
    ax.legend(frameon=False,loc="upper left",ncol=3,fontsize=11.5)
    fig.text(.5,.11,"36개 GDN 층의 state 읽기+쓰기는 매 스텝 약 231 MB로 일정",ha="center",fontsize=15,weight="bold",color=GREEN)
    save(fig,"decision-04-hybrid-state-io")


def sram():
    fig=canvas("설계 시나리오 · GDN state · 전용 state SRAM 32 MiB 가정 · 1 MiB=2^20 B",
               "현재 층이 SRAM에 맞아도, 다음 토큰까지 다른 층의 state를 모두 유지할 수 있다는 뜻은 아니다. 다른 버퍼는 별도.")
    for position,scope,title,limit,ticks in [([.09,.27,.36,.52],"current_layer","현재 층 작업 집합",38,[0,16,32]),
                                            ([.57,.27,.36,.52],"all_gdn_layers","36층을 다음 토큰까지 유지",1050,[0,250,500,1000])]:
        ax=fig.add_axes(position)
        values=[next(r["state_MiB"] for r in R["sram_working_sets"] if r["batch"]==b and r["scope"]==scope) for b in (1,8)]
        bars=ax.bar([0,1],values,width=.45,color=GREEN if scope=="current_layer" else ORANGE)
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+limit*.035,f"{value:.1f} MiB",ha="center",fontsize=14,weight="bold")
        ax.axhline(32,color=MUTED,lw=1.5,ls=(0,(4,3)))
        ax.text(-.42,32+limit*.035,"예산 32 MiB",fontsize=10.5,color=MUTED,
                bbox=dict(facecolor="white",edgecolor="none",pad=2))
        ax.set(xticks=[0,1],xticklabels=["1요청","8요청"],ylim=(0,limit),yticks=ticks,xlim=(-.6,1.6))
        ax.set_title(title,fontsize=16,weight="bold",loc="left",pad=20)
        ax.set_ylabel("필요 state 용량 (MiB)",labelpad=8)
        clean(ax)
    fig.text(.5,.13,"한 층의 임시 상주와, 여러 층을 거친 뒤 다음 토큰에서의 재사용은 다른 용량 문제",ha="center",fontsize=14,weight="bold")
    save(fig,"decision-05-state-working-set")


if __name__=="__main__":
    for f in (selector,tier,hybrid,sram): f()
    print("Four independently computed decision charts; PNG and SVG exported.")
