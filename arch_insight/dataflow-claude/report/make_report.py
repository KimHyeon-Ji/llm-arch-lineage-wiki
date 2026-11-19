"""보고서(report/dsa-v32.md · report/gdn-qwen36.md)용 그림과 수치.

  <venv python> report/make_report.py           → report/figures/*.svg · *.png, 수치는 화면에 출력
  <venv python> report/make_report.py snap      → 약식 화면 스냅샷만 다시

- 수치는 viewer와 같은 yaml(buildlib)과 model_totals.py에서 계산한다. 손으로 고치지 않는다.
- 양식은 Patent/HBM_Dataflow/figures-claude/make_figures.py와 같다 (SVG 직접 작성 → headless Chrome PNG 2배).
- 스냅샷: viewer/<id>-lite.html을 그대로 열고 머리말 · 흐름 한눈에 · 꼬리말만 숨겨 '커널별 이동'을 찍는다.
"""
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(HERE, "figures")
sys.path.insert(0, HERE)

import model_totals as mt  # noqa: E402
import buildlib as bl  # noqa: E402

CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
BROWSER = next((c for c in CHROME if os.path.exists(c)), None)

FONT = "'Noto Sans KR','Malgun Gothic',sans-serif"
INK, MUTED, AXIS, GRID = "#2b2f3a", "#6b7080", "#9aa0b0", "#e6e8f1"
BOX = ("#f6f7fb", "#dfe2ec")
# 약식 화면과 같은 데이터 종류 색
GRAY, GRAY_L, LMH = "#9c9a92", "#dddcd6", "#efeeea"
BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


# ---------------------------------------------------------------- SVG 도구 (make_figures.py와 같음)
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text_w(s, size):
    return sum(size if ord(c) > 0x2000 else size * 0.56 for c in s)


class Fig:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.o = name, w, h, []

    def raw(self, s):
        self.o.append(s)

    def rect(self, x, y, w, h, fill, stroke="none", rx=4, sw=1.5):
        self.raw(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, s, size=20, anchor="start", weight=400, color=INK):
        self.raw(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" text-anchor="{anchor}" '
                 f'font-weight="{weight}" fill="{color}">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, color=AXIS, sw=1.5):
        self.raw(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                 f'stroke-width="{sw}" stroke-linecap="round"/>')

    def svg(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}"><rect width="100%" height="100%" fill="#ffffff"/>' + "".join(self.o) + "</svg>")


def title(f, t, sub):
    f.text(40, 44, t, 26, weight=700)
    f.text(40, 78, sub, 17, color=MUTED)


def takeaway(f, y, t, sub=None):
    f.text(40, y, "→ " + t, 20, weight=700)
    if sub:
        f.text(40, y + 32, sub, 15, color=MUTED)


def legend(f, items, x, y, gap=30):
    for nm, fill in items:
        f.rect(x, y - 14, 18, 18, fill, "#c9ccd6" if fill == LMH else "none", 3, 1)
        f.text(x + 26, y + 1, nm, 15, color=MUTED)
        x += 26 + text_w(nm, 15) + gap


def save(f):
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, f.name + ".svg"), "w", encoding="utf-8") as fh:
        fh.write(f.svg())
    if BROWSER:
        with tempfile.TemporaryDirectory() as td:
            html = os.path.join(td, "fig.html")
            with open(html, "w", encoding="utf-8") as fh:
                fh.write(f'<html><body style="margin:0;background:#fff">{f.svg()}</body></html>')
            subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            "--force-device-scale-factor=2", f"--window-size={f.w},{f.h}",
                            f"--screenshot={os.path.join(FIG, f.name + '.png')}", "file:///" + html.replace("\\", "/")],
                           check=False, capture_output=True, timeout=120)
    print("ok", f.name)


def fmt_gb(v):
    return f"{v / 1e9:,.0f} GB" if v >= 100e9 else f"{v / 1e9:.1f} GB"


def pct(v):
    return f"{v:.0%}" if v >= 0.095 else f"{v:.1%}"


# ---------------------------------------------------------------- 그림: 모델 전체 중 attention 비중
BS, TS = (1, 32, 256), (8192, 32768, 131072)


def fig_share(name, model_title, sub, cats, parts_fn, att_keys, att_label, take, take_sub):
    rows = [(B, T, parts_fn(B, T)) for B in BS for T in TS]
    W, X0, BW = 1600, 250, 1060
    RH, GAP, TOP = 46, 30, 196
    H = TOP + len(BS) * (len(TS) * RH + GAP) + 130
    f = Fig(name, W, H)
    title(f, f"{model_title} — 모델 전체 decode 1스텝 HBM 이동량 중 {att_label} 몫", sub)
    legend(f, [(c[1], c[2]) for c in cats], 40, 128)
    f.text(X0, TOP - 22, "구성 비율 (한 스텝 읽기 + 쓰기 = 100%)", 15, color=MUTED)
    f.text(X0 + BW + 30, TOP - 22, f"{att_label} 합", 15, color=MUTED)
    f.text(W - 40, TOP - 22, "한 스텝 합", 15, "end", color=MUTED)
    y = TOP
    for gi, B in enumerate(BS):
        f.text(40, y + 24, f"요청 {B}개", 18, weight=700)
        for T in TS:
            p = next(r[2] for r in rows if r[0] == B and r[1] == T)
            tot = sum(p.values())
            f.text(170, y + RH / 2 + 6, f"T {T // 1024}K", 16, color=MUTED)
            x = X0
            for key, _, col in cats:
                w = BW * p[key] / tot
                if w > 0.3:
                    f.rect(x, y + 6, max(w - 2, 0.6), RH - 12, col, rx=3)
                    lab = pct(p[key] / tot)
                    if w >= text_w(lab, 14) + 12:
                        dark = col in (BLUE, ORANGE, GREEN, GRAY)
                        f.text(x + w / 2 - 1, y + RH / 2 + 5, lab, 14, "middle", 700, "#ffffff" if dark else INK)
                x += w
            att = sum(p[k] for k in att_keys) / tot
            f.text(X0 + BW + 30, y + RH / 2 + 6, pct(att), 18, weight=700)
            f.text(W - 40, y + RH / 2 + 6, fmt_gb(tot), 16, "end", color=MUTED)
            y += RH
        if gi < len(BS) - 1:
            f.line(40, y + GAP / 2, W - 40, y + GAP / 2, GRID, 1)
        y += GAP
    takeaway(f, H - 76, take, take_sub)
    return f, rows


def dsa_share():
    cats = [("attn_w", "attention 가중치", GRAY), ("idxk", "색인 K cache", ORANGE), ("kv", "KV cache", BLUE),
            ("other", "중간 결과 · 토큰 벡터", YELLOW), ("ffn", "MoE · dense FFN 가중치", GRAY_L), ("lm_head", "LM head", LMH)]
    return fig_share(
        "dsa-01-model-share", "DeepSeek-V3.2",
        "61층 (dense 3 + MoE 58) · FP8 · KV cache 656 B/줄 · H200 기준 이론 하한 · expert 수는 균등 라우팅 가정(상한) · GPU 수와 무관한 합",
        cats, lambda B, T: mt.dsa_total(B, T)[0], ("attn_w", "idxk", "kv", "other"), "attention",
        "요청이 적으면 attention 가중치, 요청 · 문맥이 커지면 색인 K cache가 attention 몫을 정한다 — KV cache 읽기는 2,048줄 고정이라 작다",
        "요청 32개는 건드리는 expert가 163개로 늘어 MoE 가중치가 대부분 · 요청 256개 × 128K에서 색인 K cache만 27%")


def gdn_share():
    cats = [("attn_w", "attention 가중치 (linear 48 + full 16)", GRAY), ("state", "recurrent · conv state", GREEN),
            ("full_kv", "full 층 KV cache", BLUE), ("other", "토큰 벡터", YELLOW), ("ffn", "FFN 가중치", GRAY_L), ("lm_head", "LM head", LMH)]

    def parts(B, T):
        p = mt.gdn_total(B, T)[0]
        return {"attn_w": p["lin_w"] + p["full_w"], "state": p["state"], "full_kv": p["full_kv"],
                "other": p["lin_oth"], "ffn": p["ffn"], "lm_head": p["lm_head"]}
    return fig_share(
        "gdn-01-model-share", "Qwen3.6-27B",
        "64층 = Gated DeltaNet 48 + full attention 16 · BF16 · state FP32 · H200 기준 이론 하한 · GPU 수와 무관한 합",
        cats, parts, ("attn_w", "state", "full_kv", "other"), "hybrid attention",
        "hybrid 모델의 요청별 이동은 state가 아니라 남은 full 16층의 KV cache가 대부분 — 문맥 약 4.7K 토큰부터 full 층 KV가 state보다 크다",
        "state는 48층 × 6.4 MB(읽기 + 쓰기)로 문맥과 무관 · 요청이 적을 때만 attention 가중치가 attention 몫의 대부분")



def local_font(html):
    """잘라내기용: 웹 글꼴 Noto Sans KR(유니코드 범위별 조각이 늦게 들어와 줄바꿈이 바뀜)을 빼고
    이 PC에 설치된 Noto Sans KR로 그린다 — 높이 재는 실행과 찍는 실행의 레이아웃을 같게."""
    return html.replace("family=Noto+Sans+KR:wght@400;500;700;800&", "")


TRIM_HTML = """<html><body style="margin:0"><pre id="o"></pre><script>
const im = new Image();
im.onload = () => {
  const W = im.naturalWidth, H = im.naturalHeight, c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d"); g.drawImage(im, 0, 0);
  const px = g.getImageData(0, 0, W, H).data, bg = [px[(H - 1) * W * 4], px[(H - 1) * W * 4 + 1], px[(H - 1) * W * 4 + 2]];
  let last = H - 1;
  scan: for (; last > 0; last--) {
    for (let x = 0; x < W; x += 2) {
      const i = (last * W + x) * 4;
      if (Math.abs(px[i] - bg[0]) + Math.abs(px[i + 1] - bg[1]) + Math.abs(px[i + 2] - bg[2]) > 12) break scan;
    }
  }
  const h = Math.min(H, last + 24), o = document.createElement("canvas");
  o.width = W; o.height = h; o.getContext("2d").drawImage(c, 0, 0);
  document.getElementById("o").textContent = h + " " + o.toDataURL("image/png");
};
im.src = "__SRC__";
</script></body></html>"""


def trim_png(path):
    """찍은 PNG의 아래쪽 빈 배경을 잘라낸다 (Chrome canvas로 — PIL 없이)."""
    import base64
    with tempfile.TemporaryDirectory() as td:
        page = os.path.join(td, "trim.html")
        with open(page, "w", encoding="utf-8") as fh:
            fh.write(TRIM_HTML.replace("__SRC__", "file:///" + path.replace("\\", "/")))
        dom = subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--allow-file-access-from-files",
                              "--virtual-time-budget=20000", "--dump-dom", "file:///" + page.replace("\\", "/")],
                             capture_output=True, timeout=180).stdout.decode("utf-8", "replace")
    m = re.search(r'<pre id="o">(\d+) data:image/png;base64,([A-Za-z0-9+/=]+)</pre>', dom)
    if not m:
        print("  trim 실패 — 원본 유지", os.path.basename(path))
        return
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(m.group(2)))
    print("  trim →", m.group(1), "px", os.path.basename(path))

# ---------------------------------------------------------------- 약식 화면 스냅샷
HIDE = ('<style>header.top, section[aria-label="흐름 한눈에"], #foot {display:none !important}'
        ' body{padding-top:8px !important}</style>')
MEASURE = ("<script>addEventListener('load',()=>setTimeout(()=>{const r=document.querySelector('.wrap')"
           ".getBoundingClientRect();document.title='H'+Math.ceil(r.bottom+window.scrollY+12)},800));</script>")


def snap(mid, width=1280):
    if not BROWSER:
        print("브라우저 없음 — 스냅샷 생략")
        return
    with open(os.path.join(ROOT, "viewer", f"{mid}-lite.html"), encoding="utf-8") as fh:
        html = local_font(fh.read()).replace("</head>", HIDE + "\n</head>", 1).replace("</body>", MEASURE + "\n</body>", 1)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "snap.html")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(html)
        url = "file:///" + p.replace("\\", "/")
        dom = subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--force-device-scale-factor=2", f"--window-size={width},1200",
                              "--virtual-time-budget=9000", "--dump-dom", url],
                             capture_output=True, timeout=120).stdout.decode("utf-8", "replace")
        m = re.search(r"<title>H(\d+)</title>", dom)
        h = int(m.group(1)) if m else 2400
        out = os.path.join(FIG, f"{mid.split('-')[0]}-02-kernel-snapshot.png")
        subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--window-size={width},{h}", "--virtual-time-budget=9000", f"--screenshot={out}", url],
                       capture_output=True, timeout=120)
    print("ok snapshot", mid, width, "x", h)


# ---------------------------------------------------------------- 약식 화면 커널별 조각 (그림 바로 아래 설명용)
CROP_CSS = ('<style>header.top, section.ctlbar, section[aria-label="흐름 한눈에"], #foot,'
            ' section[aria-label="커널별 이동"] > .blk {display:none !important}'
            ' body{padding-top:8px !important; padding-bottom:0 !important}</style>')
CROP_JS = """<script>addEventListener('load',()=>setTimeout(()=>{
const K="__K__", OPEN=__OPEN__, BI=__BI__, grid=document.getElementById('kgrid');
if(BI>=0) document.querySelector('#bseg button[data-i="'+BI+'"]').click();   // 요청 수 버튼
const card=grid.querySelector('.kcard[data-k="'+K+'"]');
if(OPEN){ card.click(); const all=document.querySelector('.op-all[data-k="'+K+'"]'); if(all) all.click(); }   // 텐서 모양 + 칩 안 계산 단계 모두 펼침
const r0=parseInt(card.style.gridRow), keep=new Set([r0,r0+1]);
if(OPEN) keep.add(r0+2);
let ph=null; grid.querySelectorAll('.phrow').forEach(p=>{if(parseInt(p.style.gridRow)<r0) ph=p;});
if(ph) keep.add(parseInt(ph.style.gridRow));
[...grid.children].forEach(el=>{ if(el.classList.contains('lane-bg')) return;
  const r=parseInt(el.style.gridRow); if(!keep.has(r)) el.style.display='none'; });
setInterval(()=>{   // 웹 글꼴 조각이 늦게 들어오면 줄바꿈이 바뀌므로 끝까지 다시 잰다 (마지막 값 사용)
const b=document.querySelector('.wrap').getBoundingClientRect();
document.title='H'+Math.ceil(b.bottom+window.scrollY+10);},250);
},900));</script>"""


CROP_B = 32                                                    # 보고서 커널별 조각의 요청 수 (문맥은 화면 기본 128K)


def snap_kernel(mid, kid, open_=False, width=1280, b=CROP_B):
    if not BROWSER:
        return
    with open(os.path.join(ROOT, "viewer", f"{mid}-lite.html"), encoding="utf-8") as fh:
        bi = [1, 2, 4, 8, 16, 32, 64, 128, 256].index(b)   # template-lite.html B_OPTS
        js = CROP_JS.replace("__K__", kid).replace("__OPEN__", "true" if open_ else "false").replace("__BI__", str(bi))
        html = local_font(fh.read()).replace("</head>", CROP_CSS + "\n</head>", 1).replace("</body>", js + "\n</body>", 1)
    name = f"{mid.split('-')[0]}-k-{kid.lower()}{'-open' if open_ else ''}.png"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "crop.html")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(html)
        url = "file:///" + p.replace("\\", "/")
        dom = subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--force-device-scale-factor=2", f"--window-size={width},1600",
                              "--virtual-time-budget=9000", "--dump-dom", url],
                             capture_output=True, timeout=120).stdout.decode("utf-8", "replace")
        m = re.search(r"<title>H(\d+)</title>", dom)
        h = int(m.group(1)) if m else 600
        subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--window-size={width},{h}", "--virtual-time-budget=9000",
                        f"--screenshot={os.path.join(FIG, name)}", url], capture_output=True, timeout=120)
    trim_png(os.path.join(FIG, name))
    print("ok crop", name, h)


CROP_OPEN = {"dsa-v32": ("K6", "K8"), "gdn-qwen36": ("K3",)}   # 핵심 커널은 텐서 모양 펼친 조각도

# '흐름 한눈에' 패널 — 요청 수 · 문맥 길이 조건별
FLOW_CSS = ('<style>header.top, section.ctlbar, section[aria-label="커널별 이동"], #foot {display:none !important}'
            ' body{padding-top:8px !important; padding-bottom:0 !important}</style>')
FLOW_JS = """<script>addEventListener('load',()=>setTimeout(()=>{
document.querySelector('#tseg button[data-i="__TI__"]').click();
document.querySelector('#bseg button[data-i="__BI__"]').click();
setInterval(()=>{   // 웹 글꼴 조각이 늦게 들어오면 줄바꿈이 바뀌므로 끝까지 다시 잰다 (마지막 값 사용)
const b=document.querySelector('.wrap').getBoundingClientRect();
document.title='H'+Math.ceil(b.bottom+window.scrollY+10);},250);
},900));</script>"""
T_OPTS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]   # template-lite.html
B_OPTS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
FLOW_CASES = {"dsa-v32": [(1, 131072), (32, 131072), (256, 131072), (32, 8192), (32, 32768)],
              "gdn-qwen36": [(1, 131072), (32, 131072), (256, 131072)]}


def snap_flow(mid, B, T, width=1280):
    if not BROWSER:
        return
    with open(os.path.join(ROOT, "viewer", f"{mid}-lite.html"), encoding="utf-8") as fh:
        js = FLOW_JS.replace("__TI__", str(T_OPTS.index(T))).replace("__BI__", str(B_OPTS.index(B)))
        html = local_font(fh.read()).replace("</head>", FLOW_CSS + "\n</head>", 1).replace("</body>", js + "\n</body>", 1)
    name = f"{mid.split('-')[0]}-flow-b{B}-t{T // 1024}k.png"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "flow.html")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(html)
        url = "file:///" + p.replace("\\", "/")
        dom = subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--force-device-scale-factor=2", f"--window-size={width},1200",
                              "--virtual-time-budget=9000", "--dump-dom", url],
                             capture_output=True, timeout=120).stdout.decode("utf-8", "replace")
        m = re.search(r"<title>H(\d+)</title>", dom)
        h = int(m.group(1)) if m else 640
        subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--window-size={width},{h}", "--virtual-time-budget=9000",
                        f"--screenshot={os.path.join(FIG, name)}", url], capture_output=True, timeout=120)
    trim_png(os.path.join(FIG, name))
    print("ok flow", name, h)


def flows():
    for mid, cases in FLOW_CASES.items():
        for B, T in cases:
            snap_flow(mid, B, T)


def crops():
    for mid in ("dsa-v32", "gdn-qwen36"):
        _, mp = mt.load(mid)
        for k in mp["kernels"]:
            snap_kernel(mid, k["id"])
        for kid in CROP_OPEN[mid]:
            snap_kernel(mid, kid, open_=True)


# ---------------------------------------------------------------- 그림: 커널별 병목 위치 (요청 1 → 32 → 256)
KIND_COL = {"weight": GRAY, "kv": BLUE, "idxk": ORANGE, "inter": YELLOW, "sel": "#e87ba4", "act": "#6250d6",
            "state": GREEN, "conv": "#e87ba4"}
PB = (1, 32, 256)


def profile(mid, T=131072):
    """커널마다 요청 1 · 32 · 256개의 읽기 · 쓰기 · FLOP · 시간 비 · 1바이트당 연산 · 주 데이터."""
    algo, mp = mt.load(mid)
    tens, _ = bl.merged(algo, mp)
    out = []
    per_b = {B: bl.kernel_numbers(algo, mp, bl.dims_of(algo, mp, T=T, B=B)) for B in PB}
    for i, k in enumerate(mp["kernels"]):
        rows = {}
        for B in PB:
            n = per_b[B][i]
            mv = n["read"] + n["write"]
            rows[B] = dict(rd=n["read"], wr=n["write"], fl=n["flops"], ratio=n["t_cmp"] / n["t_mem"] if n["t_mem"] else 0,
                           ai=n["flops"] / mv if mv else 0, rk=n["read_by_kind"])
        rk1 = rows[1]["rk"]
        main = max(rk1, key=rk1.get)
        own = [kd for kd in rk1 if kd in ("kv", "idxk", "state", "conv") and rk1[kd] >= 0.3 * rows[1]["rd"]]
        if own:                                              # 요청별 상주 데이터가 30% 이상이면 그 색
            main = max(own, key=rk1.get)
        out.append(dict(id=k["id"], label=k["label"], main=main, b=rows))
    return out


def fig_bound(mid, name, model_title, sub, take, take_sub):
    prof = profile(mid)
    W, LX, X0, X1 = 1680, 40, 380, 1090
    DEC_LO, DEC_HI = -3, 1                                   # 0.001 ~ 10
    px = lambda r: X0 + (math.log10(r) - DEC_LO) / (DEC_HI - DEC_LO) * (X1 - X0)
    RH, TOP = 50, 232
    H = TOP + len(prof) * RH + 200
    f = Fig(name, W, H)
    title(f, f"{model_title} — 커널별 병목 위치: 요청이 늘면 어디로 움직이나", sub)
    # 범례
    x = 40
    for lab, kind in (("● 요청 1개", "f"), ("• 요청 32개", "m"), ("○ 요청 256개", "h")):
        cy = 124
        if kind == "f":
            f.raw(f'<circle cx="{x + 8}" cy="{cy}" r="8" fill="{INK}"/>')
        elif kind == "m":
            f.raw(f'<circle cx="{x + 8}" cy="{cy}" r="5" fill="{INK}"/>')
        else:
            f.raw(f'<circle cx="{x + 8}" cy="{cy}" r="7" fill="#ffffff" stroke="{INK}" stroke-width="2.5"/>')
        f.text(x + 24, cy + 6, lab[2:], 15, color=MUTED)
        x += 24 + text_w(lab[2:], 15) + 30
    kinds_used = []
    for p in prof:
        if any(p["b"][B]["ratio"] > 0 for B in PB) and p["main"] not in kinds_used:
            kinds_used.append(p["main"])
    algo, _ = mt.load(mid)
    x += 20
    for kd in kinds_used:
        nm = "주 데이터: " + algo["kinds"][kd]["label"] if kd == kinds_used[0] else algo["kinds"][kd]["label"]
        f.rect(x, 124 - 9, 16, 16, KIND_COL.get(kd, GRAY), rx=3)
        f.text(x + 24, 124 + 6, nm, 15, color=MUTED)
        x += 24 + text_w(nm, 15) + 26
    # 영역
    xb = px(1)
    f.rect(xb, TOP - 40, X1 - xb, len(prof) * RH + 40, "#fbf3ec", rx=0)
    f.text(xb - 12, TOP - 16, "← 메모리 대역폭이 병목", 15, "end", 700, MUTED)
    f.text(xb + 12, TOP - 16, "연산이 병목 →", 15, "start", 700, "#a33a17")
    f.text(1130, TOP - 16, "층 1개 이동량 (요청 1 → 256)", 15, color=MUTED)
    f.text(1430, TOP - 16, "1바이트당 연산 (1 → 256)", 15, color=MUTED)
    for d in range(DEC_LO, DEC_HI + 1):
        xx = px(10 ** d)
        f.line(xx, TOP - 4, xx, TOP + len(prof) * RH, GRID if d else "#c98a5e", 1 if d else 2)
    y = TOP
    for p in prof:
        cy = y + RH / 2
        f.line(LX, y + RH, W - 40, y + RH, GRID, 1)
        f.text(LX, cy + 6, p["id"], 16, weight=700)
        f.text(LX + 52, cy + 6, p["label"], 16)
        col = KIND_COL.get(p["main"], GRAY)
        r = [p["b"][B]["ratio"] for B in PB]
        if all(v <= 0 for v in r):
            f.text(X0 + 6, cy + 6, "연산 거의 없음 — 바이트 · 실행 지연이 시간 결정", 14, color=MUTED)
        else:
            xs = [px(max(v, 10 ** DEC_LO)) for v in r]
            if abs(xs[2] - xs[0]) > 14:
                f.line(xs[0], cy, xs[2] - math.copysign(10, xs[2] - xs[0]), cy, col, 4)
            f.raw(f'<circle cx="{xs[1]:.1f}" cy="{cy}" r="5" fill="{col}"/>')
            f.raw(f'<circle cx="{xs[2]:.1f}" cy="{cy}" r="8" fill="#ffffff" stroke="{col}" stroke-width="3"/>')
            f.raw(f'<circle cx="{xs[0]:.1f}" cy="{cy}" r="8" fill="{col}"/>')
        b1, b2 = p["b"][1], p["b"][256]
        f.text(1130, cy + 6, f"{bl.fmt_b(b1['rd'] + b1['wr'])} → {bl.fmt_b(b2['rd'] + b2['wr'])}", 15)
        ai = lambda v: "0" if v < 0.05 else (f"{v:.1f}" if v < 10 else f"{v:,.0f}")
        f.text(1430, cy + 6, f"{ai(b1['ai'])} → {ai(b2['ai'])} FLOP/B", 15)
        y += RH
    # 축
    for d in range(DEC_LO, DEC_HI + 1):
        f.text(px(10 ** d), y + 24, f"{10 ** d:g}", 14, "middle", color=MUTED)
    f.text((X0 + X1) / 2, y + 50, "연산 시간 ÷ 메모리 시간 (H200 이론값, 로그 눈금)", 15, "middle", color=MUTED)
    takeaway(f, H - 76, take, take_sub)
    return f


def dsa_bound():
    return fig_bound(
        "dsa-v32", "dsa-03-kernel-bound", "DSA attention 층 1개",
        "문맥 128K · H200 이론값 (HBM 4.8 TB/s · FP8 1,979 · BF16 989 TFLOPS) · 1보다 작으면 메모리 대역폭이 병목",
        "요청이 늘면 가중치 GEMM · Sparse MLA는 연산 쪽으로 가지만, 색인 점수는 제자리 — 요청 수로 못 푸는 대역폭 문제",
        "색인 점수는 요청마다 자기 색인 K cache를 읽어 1바이트당 연산이 고정 · 실효 성능(작은 GEMM 효율 · 실효 대역폭)에 따라 경계 위치는 달라짐")


def gdn_bound():
    return fig_bound(
        "gdn-qwen36", "gdn-03-kernel-bound", "Gated DeltaNet linear 층 1개",
        "문맥 무관 · H200 이론값 (HBM 4.8 TB/s · BF16 989 · FP32 vector 67 TFLOPS) · 1보다 작으면 메모리 대역폭이 병목",
        "가중치 GEMM은 요청 약 200개에서 연산 쪽으로 넘어가지만, state 갱신은 제자리 — 1바이트당 0.9 FLOP라 늘 대역폭 문제",
        "state는 요청마다 따로라 요청이 늘어도 1바이트당 연산이 그대로 · GEMM 경계는 실효 성능에 따라 달라짐")


def print_profile(mid):
    print("\n== profile", mid)
    for p in profile(mid):
        print(f"  {p['id']:4} main={p['main']:<6} " + "  ".join(
            f"B{B}: R {bl.fmt_b(v['rd'])} W {bl.fmt_b(v['wr'])} F {v['fl'] / 1e9:.3g}G ratio {v['ratio']:.3f} AI {v['ai']:.1f}"
            for B, v in p["b"].items()))


# ---------------------------------------------------------------- 본문용 수치
def kernel_facts(mid, B, T):
    algo, mp = mt.load(mid)
    d = bl.dims_of(algo, mp, T=T, B=B)
    tens, _ = bl.merged(algo, mp)
    ks = bl.kernel_numbers(algo, mp, d)
    rd, wr = sum(k["read"] for k in ks), sum(k["write"] for k in ks)
    print(f"\n== {mid} B={B} T={T // 1024}K  read {bl.fmt_b(rd)}  write {bl.fmt_b(wr)}  lower bound {sum(k['t'] for k in ks) * 1e6:.1f} us"
          + (f"  split {d['S_acc']} direct {d['U_dir']}" if "S_acc" in d else ""))
    for k in ks:
        print(f"  {k['id']:4} {k['label']:<16} R {bl.fmt_b(k['read']):>9} ({k['read'] / rd:5.1%})  W {bl.fmt_b(k['write']):>9}  t {k['t'] * 1e6:6.2f} us")
    rk, wk = {}, {}
    for k in mp["kernels"]:
        for e in k["hbm_read"]:
            kd = tens[e["t"]]["kind"]; rk[kd] = rk.get(kd, 0) + bl.ev(tens[e.get("via", e["t"])]["bytes"], d)
        for e in k.get("hbm_write", []):
            kd = tens[e["t"]]["kind"]; wk[kd] = wk.get(kd, 0) + bl.ev(tens[e.get("via", e["t"])]["bytes"], d)
    pers = {kd for kd, v in algo["kinds"].items() if v.get("persistent")}
    hr = sum(v for kd, v in rk.items() if kd not in pers)
    hw = sum(v for kd, v in wk.items() if kd not in pers)
    print("  read by kind :", {kd: bl.fmt_b(v) for kd, v in rk.items()})
    print("  write by kind:", {kd: bl.fmt_b(v) for kd, v in wk.items()})
    print(f"  handoff(L2 후보) read {bl.fmt_b(hr)} ({hr / rd:.1%})  write {bl.fmt_b(hw)} ({hw / wr:.1%})")


def facts():
    for B, T in ((1, 131072), (32, 131072), (1, 8192), (256, 131072)):
        kernel_facts("dsa-v32", B, T)
    for B, T in ((1, 131072), (32, 131072)):
        kernel_facts("gdn-qwen36", B, T)
    # 교차점
    print("\nDSA 색인 K cache = 선택 KV 읽기 되는 T:", 656 * 2048 / 132)
    p1 = mt.gdn_total(1, 1)[0]
    st = p1["state"]
    print("GDN full 층 KV(16층) = state 왕복(48층) 되는 T:", st / (16 * 4096), " state/req", bl.fmt_b(st))
    for B in (1, 8, 32):
        print("DSA experts", B, round(mt.ds_experts(B), 1))


def main():
    only = sys.argv[1:]
    if not only or "share" in only:
        save(dsa_share()[0])
        save(gdn_share()[0])
    if not only or "bound" in only:
        save(dsa_bound())
        save(gdn_bound())
        print_profile("dsa-v32")
        print_profile("gdn-qwen36")
    if not only or "crop" in only:
        crops()
    if not only or "flow" in only:
        flows()
    if "snap" in only:                                       # 전체 스냅샷 (보고서에서는 커널별 조각을 씀)
        snap("dsa-v32")
        snap("gdn-qwen36")
    if not only or "facts" in only:
        facts()


if __name__ == "__main__":
    main()
