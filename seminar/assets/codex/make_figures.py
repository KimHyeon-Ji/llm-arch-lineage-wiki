"""Codex 세미나 그림. python make_figures.py 로 SVG/PNG를 재생성한다.

도형의 가로/세로 방향은 실제 텐서 축을 뜻한다. 셀 개수는 축의 *예시*이며,
수치가 붙은 막대만 해당 수치에 비례한다. 파일은 이 폴더에만 생성한다.
"""
from html import escape
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
BROWSER = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
P = {"q": "#6b50cc", "k": "#c44898", "v": "#4b9b62", "i": "#3682c8",
     "c": "#d5962b", "ink": "#31394c", "muted": "#68728a", "line": "#d8dce8",
     "off": "#edf0f6", "bg": "#ffffff"}
F = "'Malgun Gothic','Noto Sans KR',sans-serif"


class Fig:
    def __init__(self, name, title, h=590):
        self.name, self.h = name, h
        self.s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 {h}" width="1440" height="{h}">',
                  '<rect width="100%" height="100%" fill="white"/>']
        self.text(36, 47, title, 28, P["q"], 700)

    def raw(self, x): self.s.append(x)

    def rect(self, x, y, w, h, fill="white", stroke=None, r=8, sw=1.5):
        self.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke or fill}" stroke-width="{sw}"/>')

    def text(self, x, y, value, size=18, color=None, weight=400, anchor="start"):
        self.raw(f'<text x="{x}" y="{y}" font-family="{F}" font-size="{size}" font-weight="{weight}" fill="{color or P["ink"]}" text-anchor="{anchor}">{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color=None, sw=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or P["muted"]}" stroke-width="{sw}"{d}/>')

    def arrow(self, x1, y1, x2, y2, color=None):
        color = color or P["muted"]
        self.line(x1, y1, x2, y2, color, 2.5)
        import math
        a = math.atan2(y2-y1, x2-x1)
        p1 = (x2-10*math.cos(a-.55), y2-10*math.sin(a-.55))
        p2 = (x2-10*math.cos(a+.55), y2-10*math.sin(a+.55))
        self.raw(f'<polygon points="{x2},{y2} {p1[0]},{p1[1]} {p2[0]},{p2[1]}" fill="{color}"/>')

    def label(self, x, y, w, h, title, sub, color="q"):
        self.rect(x, y, w, h, "#f8f9fd", P[color], 7, 2)
        self.text(x+w/2, y+h/2+3, title, 22, P[color], 700, "middle")
        self.text(x+w/2, y+h+24, sub, 17, P["muted"], 400, "middle")

    def matrix(self, x, y, rows, cols, cw, ch, color, hot=None):
        hot = set(hot or [])
        for r in range(rows):
            for c in range(cols):
                fill = P[color] if (r,c) in hot else ("#e8e3f8" if color=="q" else "#f7e7f1" if color=="k" else "#e5f1e8" if color=="v" else "#e4eef8")
                self.rect(x+c*(cw+2), y+r*(ch+2), cw, ch, fill, "white", 2, 1)
        return cols*(cw+2)-2, rows*(ch+2)-2

    def cells(self, x, y, n, selected=(), color="q", cw=27, h=30, gap=4):
        for j in range(n):
            self.rect(x+j*(cw+gap), y, cw, h, P[color] if j in selected else P["off"], "white", 3, 1)

    def note(self, lines):
        y=self.h-68
        self.rect(36,y-25,1368,48,"#f7f8fc",P["line"],8,1)
        self.text(56,y+6,lines,17,P["muted"])

    def save(self):
        path=HERE/(self.name+".svg")
        path.write_text("".join(self.s)+"</svg>",encoding="utf-8")
        if BROWSER.exists():
            png=HERE/(self.name+".png")
            subprocess.run([str(BROWSER),"--headless=new","--disable-gpu","--hide-scrollbars",
                            "--force-device-scale-factor=1",f"--window-size=1440,{self.h}",
                            f"--screenshot={png}",path.as_uri()],capture_output=True,timeout=60,check=False)
        print(path.name)


def decode():
    f=Fig("01-decode-attention","한 토큰을 생성할 때: 행렬의 어느 축을 읽나",660)
    f.text(48,100,"현재 토큰의 query",19,P["q"],700)
    f.matrix(54,136,1,4,34,42,"q",{(0,0),(0,2)})
    f.text(122,224,"q : 1 × dₕ",18,P["muted"],anchor="middle")
    f.text(222,171,"×",28)
    f.matrix(276,112,4,12,32,42,"k",{(0,0),(1,2),(2,7),(3,10)})
    f.text(510,314,"Kᵀ : dₕ × S",18,P["muted"],anchor="middle")
    f.text(727,171,"=",28)
    f.matrix(777,136,1,12,32,42,"q",{(0,0),(0,2),(0,7),(0,10)})
    f.text(999,224,"score : 1 × S",18,P["muted"],anchor="middle")
    f.text(840,267,"mask + softmax ↓  (모양은 1 × S 그대로)",17,P["muted"])
    f.text(52,385,"같은 S개 위치의 value를 가중합",19,P["v"],700)
    f.matrix(54,425,1,12,32,42,"q",{(0,0),(0,2),(0,7),(0,10)})
    f.text(255,495,"p : 1 × S",18,P["muted"],anchor="middle")
    f.text(482,455,"×",28)
    f.matrix(532,389,12,4,34,12,"v",{(0,0),(2,1),(7,2),(10,3)})
    f.text(726,455,"=",28)
    f.matrix(775,425,1,4,34,42,"v",{(0,0),(0,2)})
    f.text(852,510,"out : 1 × dₕ",18,P["muted"],anchor="middle")
    f.text(1010,414,"K와 V의 12칸은",19)
    f.text(1010,447,"과거 토큰 S개를 뜻하는 예시",18,P["muted"])
    f.text(1010,481,"실제 크기는 S에 따라 증가",18,P["muted"])
    f.note("그림은 head 하나의 decode 연산이다. prefill의 Q는 1행이 아니라 여러 행이다.")
    return f


def cache():
    f=Fig("02-kv-cache-write-read","KV cache: 이번 토큰은 한 줄 쓰고, 과거 S줄을 읽는다",650)
    for b,(name,n) in enumerate([("요청 A",5),("요청 B",8),("요청 C",3)]):
        x=60+b*440
        f.text(x,109,name,22,P["ink"],700)
        f.text(x,139,f"길이 S={n} (예시)",17,P["muted"])
        for label,col,dx in [("K","k",0),("V","v",135)]:
            f.text(x+dx+45,185,label,20,P[col],700,"middle")
            for j in range(n):
                f.rect(x+dx,202+j*31,90,25,P[col] if j==n-1 else ("#f6e7f0" if col=="k" else "#e7f1e8"),P[col],3,1)
        f.text(x+300,202+(n-1)*31+19,"← 새 행 append",16,P["muted"])
        f.text(x+300,235,"← 기존 행 read",16,P["muted"])
        f.line(x+95,206,x+275,206,P["line"],2)
    f.text(60,508,"배치의 KV 메모리 ∝ Σᵣ Sᵣ × 2 × L × nₖᵥ × dₕ × bytes",23,P["q"],700)
    f.note("가중치는 요청 사이에 공유해도, 요청별 토큰 기록은 별도로 쌓인다. 길이가 다른 요청은 행 수도 다르다.")
    return f


def levers():
    f=Fig("03-three-levers","어디를 줄이는가: cache의 너비 C와 읽는 행 수 R")
    for j in range(4):
        y=125+j*93
        name,c,r,col=[("기본",12,16,"q"),("GQA / MLA",4,16,"k"),("SWA / sparse",12,5,"i"),("둘 다",4,5,"v")][j]
        f.text(52,y+20,name,21,P["ink"],700)
        f.text(270,y+20,"저장 폭 C",17,P["muted"])
        f.cells(390,y-5,c,range(c),color=col,cw=19,h=25,gap=2)
        f.text(730,y+20,"읽는 위치 R",17,P["muted"])
        f.cells(870,y-5,r,range(r),color=col,cw=19,h=25,gap=2)
        f.text(1260,y+20,f"{c} × {r}",19,P[col],700)
    f.note("그림의 12칸·16칸은 개념용 축 길이. 실제 용량은 L·S·C, 본 attention의 읽기는 R·C에 비례한다.")
    return f


def heads():
    f=Fig("04-mha-mqa-gqa","Q head 8개는 그대로, 저장하는 KV head 수만 달라진다",620)
    configs=[("MHA",8,60),("GQA",2,525),("MQA",1,990)]
    for name,n,x in configs:
        f.text(x+188,102,name,24,P["q"],700,"middle")
        f.text(x+188,130,f"Q 8개 · KV {n}개",17,P["muted"],anchor="middle")
        for i in range(8):
            xx=x+i*45
            f.rect(xx,158,34,35,"#e9e5fa",P["q"],4,1.4)
            f.text(xx+17,182,str(i+1),15,P["q"],700,"middle")
            target=i if n==8 else i//4 if n==2 else 0
            kx=x+(target*45 if n==8 else (75+target*170 if n==2 else 165))
            f.line(xx+17,194,kx+17,278,"#a6acbd",1.5)
        for i in range(n):
            kx=x+(i*45 if n==8 else (75+i*170 if n==2 else 165))
            f.rect(kx,282,34,34,"#f8e7f1",P["k"],4,1.4)
            f.rect(kx,322,34,34,"#e5f1e8",P["v"],4,1.4)
            for row in range(5):
                f.rect(kx,372+row*19,34,15,"#f5e7f0",P["k"],2,1)
                f.rect(kx,478+row*19,34,15,"#e6f1e9",P["v"],2,1)
        f.text(x+188,594,f"토큰당 K+V 저장량 = {2*n} × dₕ",17,P["muted"],anchor="middle")
    return f


def mla_cache():
    f=Fig("05-mla-cache","MLA는 head별 K·V 대신 공통 latent와 위치 key를 저장한다",650)
    f.text(50,113,"비교를 위한 가상 MHA: 128 KV heads × 128차원 × K/V",19,P["muted"])
    f.rect(50,145,1260,64,"#e6f1e9",P["v"],5)
    f.rect(50,145,630,64,"#f6e5ef",P["k"],5)
    f.text(366,185,"K = 16,384",21,P["k"],700,"middle")
    f.text(996,185,"V = 16,384",21,P["v"],700,"middle")
    f.text(1340,184,"32,768",20,P["muted"],700)
    f.text(50,290,"DeepSeek-V3 MLA cache: c_KV 512 + k_R 64",20,P["muted"])
    f.rect(50,326,22.15,64,P["c"],P["c"],2)
    f.text(93,367,"576 values / token / layer — 같은 눈금, 약 57배 작음",20,P["ink"],700)
    f.text(50,448,"MLA 막대만 확대 (내부 구성 512 : 64 = 8 : 1)",18,P["muted"])
    f.rect(50,467,213,45,"#f8efd9",P["c"],4)
    f.rect(263,467,27,45,"#f6e5ef",P["k"],4)
    f.text(156,496,"c_KV 512",17,P["c"],700,"middle")
    f.text(310,497,"k_R 64",17,P["k"])
    f.text(50,552,"c_KV에서 head별 content K·V를 계산하지만, 그 전체를 cache에 펼쳐 저장하지 않는다.",18)
    f.note("상단 두 막대는 값 수에 비례. MHA는 동일 128×128 KV head의 가상 비교이며 dtype·quantization을 반영한 bytes가 아니다.")
    return f


def absorb():
    f=Fig("06-mla-absorption","MLA absorption: 과거 S개를 펼치지 않고 query 1개를 옮긴다")
    f.text(48,105,"느린 표현: cache의 모든 c_s를 head별 K로 복원",20,P["k"],700)
    f.label(50,143,210,84,"C", "S × 512","c")
    f.arrow(276,184,344,184)
    f.label(356,143,260,84,"K = C Wᵁᴷ", "S × H × 128","k")
    f.arrow(630,184,688,184)
    f.label(700,143,170,84,"Q Kᵀ", "H × S","q")
    f.text(955,187,"S행을 매 토큰마다 변환",20,P["muted"])
    f.line(48,284,1387,284,P["line"],2)
    f.text(48,330,"실제 요령: query를 latent 좌표로 먼저 변환",20,P["q"],700)
    f.label(50,366,210,84,"Q", "H × 128","q")
    f.arrow(276,407,344,407)
    f.label(356,366,260,84,"Q′ = Q(Wᵁᴷ)ᵀ", "H × 512","q")
    f.arrow(630,407,688,407)
    f.label(700,366,170,84,"Q′ Cᵀ", "H × S","c")
    f.text(955,410,"query 1행만 변환",20,P["muted"])
    f.note("content score의 결합법칙을 그린 것. RoPE 위치 성분 k_R은 별도로 계산하고, V·출력 투영도 결합한다.")
    return f


def swa():
    f=Fig("07-swa","SWA는 각 query가 보는 key 열을 최근 W개로 자른다",650)
    f.text(80,102,"Full causal attention",21,P["q"],700)
    f.text(765,102,"Sliding window (W=4 예시)",21,P["q"],700)
    for base,win in [(90,False),(775,True)]:
        for r in range(10):
            for c in range(10):
                on=c<=r and (not win or c>=r-3)
                f.rect(base+c*40,135+r*37,35,32,P["q"] if on else P["off"],"white",2,1)
        f.text(base+205,535,"key position 0 → 9",17,P["muted"],anchor="middle")
        f.text(base+442,337,"query position ↓",17,P["muted"])
    f.note("흰 칸은 mask. SWA가 window 밖 KV를 버릴 수 있는지는 모델의 다른 branch·layer와 위치 정책에 달려 있다.")
    return f


def dsa():
    f=Fig("08-dsa","DSA: 별도 index key는 전체 스캔, MLA latent는 top-k만 읽기",700)
    n=24; picked={2,7,13,18,22}
    f.text(48,111,"① indexer key cache",20,P["i"],700)
    f.text(48,141,"S × 128 · 별도 저장",17,P["muted"])
    f.cells(404,105,n,range(n),color="i",cw=31,h=38,gap=4)
    f.text(405,174,"S개 전부 점수화 → top-k 위치 선택 (실제 k=2,048)",17,P["muted"])
    f.text(48,267,"② 선택한 위치",20,P["q"],700)
    f.cells(404,229,n,picked,color="q",cw=31,h=38,gap=4)
    for j in picked: f.arrow(420+j*35,271,420+j*35,341,P["muted"])
    f.text(48,392,"③ MLA latent cache",20,P["c"],700)
    f.text(48,422,"S × (512 + 64)",17,P["muted"])
    f.cells(404,351,n,picked,color="c",cw=31,h=38,gap=4)
    f.text(405,426,"선택한 k개만 gather하여 본 attention",17,P["muted"])
    f.rect(48,490,1342,78,"#f4f8fd",P["i"],8)
    f.text(70,521,"읽기 범위",19,P["ink"],700)
    f.text(260,521,"indexer: S개 × 128차원",18,P["i"],700)
    f.text(756,521,"main: k개 × 576차원",18,P["c"],700)
    f.text(70,551,"얕고 넓게 찾은 뒤, 깊고 좁게 계산한다. indexer의 O(S) 비용은 남는다.",17,P["muted"])
    f.note("위 24칸 중 5칸 선택은 흐름 예시다. 실제 top-k 비율은 문맥 길이에 따라 달라진다.")
    return f


def csa_hca():
    f=Fig("09-csa-hca","CSA와 HCA: 같은 원본 위치를 서로 다른 비율로 압축",700)
    for top,name,ratio,col in [(112,"CSA",4,"c"),(338,"HCA",128,"q")]:
        f.text(46,top-18,name,23,P[col],700)
        f.text(150,top-18,f"{ratio} token → 1 compressed KV",18,P["muted"])
        for b in range(6):
            x=56+b*132
            f.rect(x,top,116,40,P["off"],P["line"],4)
            f.text(x+58,top+26,f"원본 {b*ratio}…{(b+1)*ratio-1}",14,P["muted"],anchor="middle")
            f.arrow(x+58,top+43,x+58,top+77)
            if name=="CSA" and b>0:
                f.line(x-16,top+30,x+43,top+77,P["c"],1.4,"5 4")
            fill=P[col] if (name=="CSA" and b in (1,4)) else ("#f8efd9" if name=="CSA" else "#e9e5fa")
            f.rect(x+30,top+82,56,34,fill,P[col],4)
        f.text(885,top+92,"S/4 entry를 점수화 → top-k" if name=="CSA" else "S/128 entry를 전부 읽음",19,P[col],700)
        f.text(885,top+124,"선택·gather 비용 있음" if name=="CSA" else "압축 손실은 더 큼",17,P["muted"])
    f.rect(46,553,1348,45,"#f7f8fc",P["line"],7)
    f.text(68,582,"둘 다 최근 raw token 128개의 local window를 병행한다. 압축 entry가 raw token과 동일하다는 뜻은 아니다.",17,P["muted"])
    f.note("CSA 점선 = 직전 4토큰도 반영. 학습된 압축이며 단순 평균이 아니다. 블록 개수는 축을 설명하는 예시다.")
    return f


def qsa():
    f=Fig("10-qsa","QSA: 검색용 key만 4토큰씩 묶고, 본 attention은 원본 KV로",660)
    f.text(55,105,"각 token의 index key는 4개씩 묶음 · 본 attention용 원본 GQA KV는 별도 유지",19,P["v"],700)
    for b in range(6):
        x=55+b*170
        f.rect(x-5,129,144,49,"white",P["line"],6)
        for i in range(4):
            f.rect(x+i*36,135,30,36,"#e5eff9",P["i"],3)
        f.arrow(x+68,185,x+68,237)
        f.rect(x+36,244,64,34,P["i"] if b in (1,4) else "#e5eff9",P["i"],4)
    f.text(1090,160,"4개 연속 위치 = 1 block",18,P["muted"])
    f.text(55,328,"index key 4개 평균 → 1 block key · 점수 상위 512 block",19,P["i"],700)
    f.text(55,369,"선택 block의 원본 4개 KV를 모두 읽음 → 최대 2,048 token",19,P["q"],700)
    f.text(55,400,"맨 끝에 덜 찬 block은 압축하지 않고 별도로 읽는다.",17,P["muted"])
    f.text(55,434,"48 layers: Gated DeltaNet 36 / QSA 12",19,P["muted"])
    for i in range(48):
        f.rect(55+i*27,456,23,34,P["q"] if i%4==3 else "#d7dce8","white",3)
    f.text(55,530,"회색 = 길이에 따라 늘어나는 token KV cache 없음",17,P["muted"])
    f.text(760,530,"보라색 = QSA layer의 KV cache 있음",17,P["q"])
    f.note("block key의 평균은 고정 연산이지만, 평균할 index key 자체는 학습된 projection의 출력이다.")
    return f


if __name__ == "__main__":
    for make in (decode,cache,levers,heads,mla_cache,absorb,swa,dsa,csa_hca,qsa):
        make().save()
