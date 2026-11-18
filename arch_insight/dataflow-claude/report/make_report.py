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
        html = fh.read().replace("</head>", HIDE + "\n</head>", 1).replace("</body>", MEASURE + "\n</body>", 1)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "snap.html")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(html)
        url = "file:///" + p.replace("\\", "/")
        dom = subprocess.run([BROWSER, "--headless=new", "--disable-gpu", f"--window-size={width},1200",
                              "--virtual-time-budget=6000", "--dump-dom", url],
                             capture_output=True, timeout=120).stdout.decode("utf-8", "replace")
        m = re.search(r"<title>H(\d+)</title>", dom)
        h = int(m.group(1)) if m else 2400
        out = os.path.join(FIG, f"{mid.split('-')[0]}-02-kernel-snapshot.png")
        subprocess.run([BROWSER, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--window-size={width},{h}", "--virtual-time-budget=6000", f"--screenshot={out}", url],
                       capture_output=True, timeout=120)
    print("ok snapshot", mid, width, "x", h)


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
    if not only or "snap" in only:
        snap("dsa-v32")
        snap("gdn-qwen36")
    if not only or "facts" in only:
        facts()


if __name__ == "__main__":
    main()
