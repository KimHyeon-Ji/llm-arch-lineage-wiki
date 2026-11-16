"""모델별 build.py가 같이 쓰는 부분: yaml 읽기 · 차원 · 커널 수치 · numbers 표 · 뷰어 출력.

모델마다 다른 것(참조 구현 meta 실행 · 보고서 대조)은 각 models/<id>/build.py에 둔다.
"""
import json
import os

import yaml

from dflow import ev

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(here):
    with open(os.path.join(here, "algo.yaml"), encoding="utf-8") as f:
        algo = yaml.safe_load(f)
    with open(os.path.join(here, "map-gpu.yaml"), encoding="utf-8") as f:
        mp = yaml.safe_load(f)
    hw = mp["hw"]  # PyYAML은 1979e12 같은 표기를 문자열로 읽는다
    hw["hbm_bw"] = float(hw["hbm_bw"])
    hw["peak"] = {k: float(v) for k, v in hw["peak"].items()}
    return algo, mp


def ui_format(algo):
    for name, f in algo["formats"].items():
        if f.get("ui"):
            return name
    return None


def flashmla_sched(B, topk, n_sm, sc):
    """FlashMLA sparse decode split-KV 스케줄러(get_decoding_sched_meta.cu)를 그대로 따라 한다.
    반환: (split된 요청들의 split 수 합, split 없는 요청 수)."""
    if topk <= sc["min_topk"]:
        return 0, B
    parts = max(n_sm // sc["parts_div"], 1)
    nb = (topk - 1) // sc["block"] + 1
    ov = sc["overhead"]
    payload = -(-B * (nb + ov) // parts) + ov
    sp, req, blk, ns = [0] * B, 0, 0, 0
    for _ in range(parts):
        remain = payload
        while req < B:
            if remain >= nb - blk + ov:
                sp[req] = ns + 1
                remain -= nb - blk + ov
                req, blk, ns = req + 1, 0, 0
            else:
                if remain - ov > 0:
                    blk += remain - ov
                    ns += 1
                    remain = 0
                break
    assert req == B
    return sum(x for x in sp if x > 1), sum(1 for x in sp if x <= 1)


def dims_of(algo, mp=None, fmt=None, **over):
    d = {k: v["value"] for k, v in algo["dims"].items()}
    ui = ui_format(algo)
    for name, f in algo["formats"].items():
        key = fmt if (name == ui and fmt) else f["default"]
        d[name] = f["options"][key]["value"]
    d.update(over)
    if mp is not None and "sched" in mp:
        d["S_acc"], d["U_dir"] = flashmla_sched(d["B"], d["k"], mp["hw"]["n_sm"], mp["sched"])
    return d


def merged(algo, mp):
    """알고리즘 텐서 · 단계 + 배치 전용 텐서 · 단계."""
    tens = dict(algo["tensors"], **mp.get("extra_tensors", {}))
    steps = {s["id"]: s for s in algo["steps"]}
    for s in mp.get("extra_steps", []):
        steps[s["id"]] = s
    return tens, steps


def kernel_numbers(algo, mp, dims):
    tens, steps = merged(algo, mp)
    hw = mp["hw"]
    res = []
    for k in mp["kernels"]:
        rd = sum(ev(tens[e.get("via", e["t"])]["bytes"], dims) for e in k["hbm_read"])
        wr = sum(ev(tens[e.get("via", e["t"])]["bytes"], dims) for e in k.get("hbm_write", []))
        fl, tc = 0, 0.0
        for sid in k["steps"]:
            s = steps[sid]
            if "flops" in s:
                f = ev(s["flops"], dims)
                fl += f
                tc += f / (hw["peak"][s.get("prec", "bf16")] if s["unit"] == "gemm" else hw["peak"]["vec"])
        tm = (rd + wr) / hw["hbm_bw"]
        by_kind = {}
        for e in k["hbm_read"]:
            kd = tens[e["t"]]["kind"]
            by_kind[kd] = by_kind.get(kd, 0) + ev(tens[e.get("via", e["t"])]["bytes"], dims)
        res.append({"id": k["id"], "label": k["label"], "read": rd, "write": wr, "flops": fl,
                    "t_mem": tm, "t_cmp": tc, "t": max(tm, tc), "read_by_kind": by_kind})
    return res


def weight_bytes(ks):
    return sum(k["read_by_kind"].get("weight", 0) for k in ks)


def cross_b(algo, mp, T, fmt=None):
    """요청별 이동(읽기 + 쓰기 − 가중치)이 가중치 이상이 되는 가장 작은 B."""
    for B in range(1, 4097):
        ks = kernel_numbers(algo, mp, dims_of(algo, mp, fmt, T=T, B=B))
        w = weight_bytes(ks)
        if sum(k["read"] + k["write"] for k in ks) - w >= w:
            return B
    return None


def fmt_b(b):
    for u, s in ((1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if b >= u:
            return f"{b / u:.3g} {s}"
    return f"{b:.0f} B"


def fmt_t(t):
    return f"{t * 1e6:.2f} µs"


def kernel_tables(algo, mp, T=131072):
    """ui 형식 옵션마다 커널 표 (128K · 요청 1개)."""
    L = []
    ui = ui_format(algo)
    for opt, o in algo["formats"][ui]["options"].items():
        dims = dims_of(algo, mp, opt, T=T)
        ks = kernel_numbers(algo, mp, dims)
        extra = f" · split {dims['S_acc']}개" if "S_acc" in dims else ""
        L += [f"## {T // 1024}K · {algo['formats'][ui]['title']} {o['label']}{extra}", "",
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
    return L


def cross_table(algo, mp, Ts=(32768, 131072, 1048576)):
    L = ["## 요청별 이동이 가중치보다 커지는 동시 요청 수", "",
         "가중치는 요청이 몇 개든 한 번 읽는다. 요청별 데이터는 B와 함께 늘지만 비례하지 않을 수 있어",
         "(예: split 부분 출력), B = 1, 2, 3 …을 차례로 넣어 처음 넘는 B를 찾는다.", "",
         "| T | 가중치 | 요청 1개일 때 요청별 이동 | 요청별 ≥ 가중치가 되는 B |", "|---|---:|---:|---:|"]
    for T in Ts:
        ks = kernel_numbers(algo, mp, dims_of(algo, mp, T=T, B=1))
        w = weight_bytes(ks)
        per = sum(k["read"] + k["write"] for k in ks) - w
        L.append(f"| {T // 1024}K | {fmt_b(w)} | {fmt_b(per)} | {cross_b(algo, mp, T)} |")
    return L + [""]


def write_outputs(here, algo, mp, checks, bad, numbers_md):
    with open(os.path.join(here, "numbers.md"), "w", encoding="utf-8") as f:
        f.write(numbers_md)
    data = {"algo": algo, "map": mp, "checks": {"n": len(checks), "fail": bad}}
    js = json.dumps(data, ensure_ascii=False)
    with open(os.path.join(here, "dataflow.json"), "w", encoding="utf-8") as f:
        f.write(js)
    page = algo["page"]
    with open(os.path.join(ROOT, "viewer", "template.html"), encoding="utf-8") as f:
        html = f.read()
    body = html.replace("/*__DATA__*/null", js).replace("__TITLE__", page["doc_title"])
    # 1) 공유용 완전한 문서 — 브라우저 · GitHub Pages · 파일 첨부로 바로 열림
    head_end = body.index("</style>") + len("</style>")
    full = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + body[:head_end] + "\n</head>\n<body>\n" + body[head_end:] + "\n</body>\n</html>\n")
    name = page["file"]
    with open(os.path.join(ROOT, "viewer", f"{name}.html"), "w", encoding="utf-8") as f:
        f.write(full)
    # 2) claude.ai artifact용 본문 (게시할 때 문서 뼈대가 씌워짐)
    os.makedirs(os.path.join(ROOT, "viewer", "artifact"), exist_ok=True)
    with open(os.path.join(ROOT, "viewer", "artifact", f"{name}.html"), "w", encoding="utf-8") as f:
        f.write(body)
    print(f"viewer/{name}.html (공유용) · viewer/artifact/{name}.html (artifact용) 생성")


def check_flops_expected(step, dims):
    """GEMM FLOP 검증 기댓값: check_flops가 있으면 그것, GEMM 단계는 flops, 나머지는 0."""
    if "check_flops" in step:
        return ev(step["check_flops"], dims)
    if step["unit"] == "gemm" or step.get("gemm_traced"):
        return ev(step["flops"], dims)
    return 0
