"""algo.yaml + map-gpu.yaml → 검증 · 수치 · 뷰어 데이터.

  python build.py

1. 검증 : ref_decode.py를 실제 차원(T = 131,072)의 meta device로 실행해
           yaml의 텐서 shape 식과 GEMM FLOP 식이 실행 결과와 같은지 확인한다.
2. 수치 : 커널별 HBM 읽기 · 쓰기 바이트, FLOP, 이론 하한 시간 → numbers.md
3. 데이터: dataflow.json, 그리고 ../../viewer/template.html에 넣어 ../../viewer/dsa-v32.html
"""
import json
import os
import sys

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

from dflow import Rec, ev  # noqa: E402
from ref_decode import V32_CFG, decode_step, new_cache, weights_meta  # noqa: E402


def load():
    with open(os.path.join(HERE, "algo.yaml"), encoding="utf-8") as f:
        algo = yaml.safe_load(f)
    with open(os.path.join(HERE, "map-gpu.yaml"), encoding="utf-8") as f:
        mp = yaml.safe_load(f)
    hw = mp["hw"]  # PyYAML은 1979e12 같은 표기를 문자열로 읽는다
    hw["hbm_bw"] = float(hw["hbm_bw"])
    hw["peak"] = {k: float(v) for k, v in hw["peak"].items()}
    return algo, mp


def dims_of(algo, **over):
    d = {k: v["value"] for k, v in algo["dims"].items()}
    for name, f in algo["formats"].items():
        d[name] = f["options"][f["default"]]["value"]
    d.update(over)
    return d


# ── 1. 검증 ────────────────────────────────────────────────────────────────

def verify(algo):
    dims = dims_of(algo)
    T, B = dims["T"], dims["B"]
    W = weights_meta(V32_CFG)
    x = torch.empty(B, V32_CFG["d"], device="meta", dtype=torch.bfloat16)
    cache = new_cache(V32_CFG, B, T, x)
    rec = Rec()
    decode_step(W, V32_CFG, cache, x, T - 1, rec=rec)

    traced = dict(rec.tensors)
    for n, w in W.items():
        traced[n] = {"shape": list(w.shape)}
    traced["x"] = {"shape": list(x.shape)}
    traced["cache.kv"] = {"shape": list(cache["kv"].shape)}
    traced["cache.ik"] = {"shape": list(cache["ik"].shape)}

    out, bad = [], 0
    for tid, t in algo["tensors"].items():
        name = t.get("trace", tid)
        exp = [ev(e, dims) for e in t["shape"]]
        got = traced.get(name, {}).get("shape")
        ok = got == exp
        bad += not ok
        out.append({"what": f"shape {tid}", "expect": exp, "traced": got, "ok": ok})
    assert rec.order == [s["id"] for s in algo["steps"]], (rec.order, "단계 순서가 algo.yaml과 다르다")
    for s in algo["steps"]:
        got = rec.gemm_flops(s["id"])
        if "flops" in s:
            exp = ev(s["flops"], dims)
            ok = got == exp
        else:
            exp, ok = 0, got == 0
        bad += not ok
        out.append({"what": f"gemm_flops {s['id']}", "expect": exp, "traced": got, "ok": ok,
                    "ops": sorted({o["op"] for o in rec.steps[s["id"]]["ops"]})})
    return out, bad


# ── 2. 수치 ────────────────────────────────────────────────────────────────

def kernel_numbers(algo, mp, dims):
    tens, steps = algo["tensors"], {s["id"]: s for s in algo["steps"]}
    hw = mp["hw"]
    res = []
    for k in mp["kernels"]:
        rd = sum(ev(tens[e.get("via", e["t"])]["bytes"], dims) for e in k["hbm_read"])
        wr = sum(ev(tens[e.get("via", e["t"])]["bytes"], dims) for e in k["hbm_write"])
        fl, tc = 0, 0.0
        for sid in k["steps"]:
            s = steps[sid]
            if "flops" in s:
                f = ev(s["flops"], dims)
                fl += f
                tc += f / hw["peak"][s.get("prec", "bf16")]
        tm = (rd + wr) / hw["hbm_bw"]
        by_kind = {}
        for e in k["hbm_read"]:
            t = tens[e["t"]]
            by_kind[t["kind"]] = by_kind.get(t["kind"], 0) + ev(tens[e.get("via", e["t"])]["bytes"], dims)
        res.append({"id": k["id"], "label": k["label"], "read": rd, "write": wr, "flops": fl,
                    "t_mem": tm, "t_cmp": tc, "t": max(tm, tc), "read_by_kind": by_kind})
    return res


def fmt_b(b):
    for u, s in ((1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if b >= u:
            return f"{b / u:.3g} {s}"
    return f"{b:.0f} B"


def fmt_t(t):
    return f"{t * 1e6:.2f} µs"


def numbers_md(algo, mp, checks, bad):
    L = ["# DSA (DeepSeek-V3.2) — 층 1개 decode 1스텝 수치", "",
         "build.py가 algo.yaml · map-gpu.yaml에서 계산한다. 손으로 고치지 않는다.", "",
         "기준: attention 블록만 (MoE · FFN 제외), 요청 1개, H200 이론 하한.", ""]
    for fmt in ("fp8", "bf16"):
        dims = dims_of(algo, kv_row=algo["formats"]["kv_row"]["options"][fmt]["value"])
        ks = kernel_numbers(algo, mp, dims)
        L += [f"## 128K · KV {fmt.upper()} ({dims['kv_row']} B/줄)", "",
              "| 커널 | HBM 읽기 | HBM 쓰기 | FLOP | 메모리 시간 | 연산 시간 | 하한 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for k in ks:
            L.append(f"| {k['id']} {k['label']} | {fmt_b(k['read'])} | {fmt_b(k['write'])} | "
                     f"{k['flops'] / 1e9:.3g} G | {fmt_t(k['t_mem'])} | {fmt_t(k['t_cmp'])} | {fmt_t(k['t'])} |")
        tot = {x: sum(k[x] for k in ks) for x in ("read", "write", "flops", "t")}
        L += [f"| **합계** | {fmt_b(tot['read'])} | {fmt_b(tot['write'])} | {tot['flops'] / 1e9:.3g} G | | | {fmt_t(tot['t'])} |", ""]
        kinds = {}
        for k in ks:
            for kk, v in k["read_by_kind"].items():
                kinds[kk] = kinds.get(kk, 0) + v
        L += ["읽기 구성: " + " · ".join(f"{algo['kinds'][kk]['label']} {fmt_b(v)}" for kk, v in kinds.items()), ""]

    # 보고서 대조
    dims = dims_of(algo, kv_row=1152)
    t = algo["tensors"]
    steps = {s["id"]: s for s in algo["steps"]}
    idx_rd = ev(t["idx_cache"]["bytes"], dims)
    fl_i = ev(steps["i5"]["flops"], dims)
    fl_a = ev(steps["a2"]["flops"], dims) + ev(steps["a4"]["flops"], dims)
    L += ["## 보고서 수치와 대조 (128K · KV BF16)", "",
          "| 항목 | 계산값 | 보고서 |", "|---|---:|---|",
          f"| 색인 키 읽기 | {fmt_b(idx_rd)} | 17 MB (§1.5) |",
          f"| 색인 점수 FLOP | {fl_i / 1e9:.3g} GFLOP | 2.15 GFLOP (§1.5) |",
          f"| 색인 ÷ 본 attention FLOP | {fl_i / fl_a:.2f}배 | 약 3.8배 (§1.5) |",
          f"| 색인 산술 강도 | {fl_i / idx_rd:.0f} FLOP/B | 124 FLOP/B (§2.2) |",
          f"| 선택 KV 읽기 | {fmt_b(ev(t['kv_sel']['bytes'], dims))} | 1.2 KB × 2,048 ≈ 2.4 MB (§1.5 · §2.2) |",
          f"| 선택 위치 | {fmt_b(ev(t['idx']['bytes'], dims))} | 8 KB (§2.2) |", ""]

    # 가중치 vs 요청별 읽기가 같아지는 B
    L += ["## 가중치 읽기 = 요청별 읽기가 되는 동시 요청 수 (KV FP8)", "",
          "가중치는 요청이 몇 개든 한 번 읽는다. 요청별 데이터(색인 K · KV · 중간값)는 B배로 는다.", "",
          "| T | 가중치 | 요청 1개당 | 같아지는 B |", "|---|---:|---:|---:|"]
    for T in (32768, 131072, 1048576):
        dims = dims_of(algo, T=T, B=1)
        ks = kernel_numbers(algo, mp, dims)
        w = sum(k["read_by_kind"].get("weight", 0) for k in ks)
        per = sum(k["read"] + k["write"] for k in ks) - w
        L.append(f"| {T // 1024}K | {fmt_b(w)} | {fmt_b(per)} | {w / per:.1f} |")
    L += ["", "## 검증 (meta 실행, T = 131,072)", "",
          f"shape · GEMM FLOP 검사 {len(checks)}개 중 실패 {bad}개.", ""]
    return "\n".join(L) + "\n"


def main():
    algo, mp = load()
    checks, bad = verify(algo)
    for c in checks:
        if not c["ok"]:
            print("FAIL", c)
    print(f"검증: {len(checks)}개 중 실패 {bad}개")
    with open(os.path.join(HERE, "numbers.md"), "w", encoding="utf-8") as f:
        f.write(numbers_md(algo, mp, checks, bad))
    data = {"algo": algo, "map": mp, "checks": {"n": len(checks), "fail": bad}}
    js = json.dumps(data, ensure_ascii=False)
    with open(os.path.join(HERE, "dataflow.json"), "w", encoding="utf-8") as f:
        f.write(js)
    tpl = os.path.join(ROOT, "viewer", "template.html")
    if os.path.exists(tpl):
        with open(tpl, encoding="utf-8") as f:
            html = f.read()
        body = html.replace("/*__DATA__*/null", js)
        # 1) 공유용 완전한 문서 — 브라우저 · GitHub Pages · 파일 첨부로 바로 열림
        head_end = body.index("</style>") + len("</style>")
        full = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                + body[:head_end] + "\n</head>\n<body>\n" + body[head_end:] + "\n</body>\n</html>\n")
        with open(os.path.join(ROOT, "viewer", "dsa-v32.html"), "w", encoding="utf-8") as f:
            f.write(full)
        # 2) claude.ai artifact용 본문 (게시할 때 문서 뼈대가 씌워짐)
        os.makedirs(os.path.join(ROOT, "viewer", "artifact"), exist_ok=True)
        with open(os.path.join(ROOT, "viewer", "artifact", "dsa-v32.html"), "w", encoding="utf-8") as f:
            f.write(body)
        print("viewer/dsa-v32.html (공유용) · viewer/artifact/dsa-v32.html (artifact용) 생성")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
