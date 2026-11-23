"""연산 성격 — 모델 전체 decode 1스텝의 연산을 묶음 · 연산기 · 커지는 축 · 산술 강도 · 직렬 경로로 나눈다.

연산 종합 보고서(report/synthesis-compute.md · report/make_compute_report.py)가 쓴다.

읽는 파일: models/<id>/algo.yaml · map-gpu.yaml · model.yaml · ops.yaml
- 그린 층: map-gpu.yaml 커널 그대로(buildlib.kernel_numbers) × ops.yaml layers. 커널 → 묶음은 ops.yaml kernels
- 그린 층 밖(다른 층 종류 · MoE · FFN · LM head): ops.yaml ops 식
- 이론 시간 = 커널(연산 줄)마다 max(HBM 바이트 ÷ 대역폭, FLOP ÷ 그 연산기 피크)의 합 — 실행 지연 · 동기화 미포함
- V3.2 split-KV 부분 출력은 S_acc = 0으로 뺌 (synthesis-traits.md와 같은 처리)
- 직렬 깊이: 커널 j가 앞 커널 i가 HBM에 쓴 텐서를 읽으면 i → j. 깊이 = 가장 긴 의존 사슬의 커널 수
"""
import os

import yaml

import buildlib as bl
import traits as tr
from dflow import ev

CATS = [   # 연산 묶음 — 그림 순서
    ("proj",  "가중치 GEMM · 투영 · 기타"),
    ("moe",   "가중치 GEMM · MoE · FFN"),
    ("head",  "가중치 GEMM · LM head"),
    ("attn",  "attention 본체 (QK · PV)"),
    ("idx",   "색인 · 블록 점수"),
    ("topk",  "top-k 선택"),
    ("state", "state · conv 갱신"),
    ("misc",  "쓰기 · 정규화 · 합치기"),
]
CAT = dict(CATS)
WEIGHT = ("proj", "moe", "head")          # 공유 가중치를 곱하는 연산 — 요청 수로 강도가 커짐
REQ = ("attn", "idx", "topk", "state")    # 요청별 데이터(KV · 색인 키 · state)를 읽는 연산 — 강도는 구조가 정함
UNITS = [  # 연산기 · 정밀도 — FLOP 구성
    (("gemm", "fp8"),  "Tensor core FP8"),
    (("gemm", "bf16"), "Tensor core BF16"),
    (("vec", "fp32"),  "CUDA core FP32 (GEMV · 원소별)"),
]
_M = {}


def load(mid):
    if mid not in _M:
        M = tr.load(mid)
        with open(os.path.join(bl.ROOT, "models", mid, "ops.yaml"), encoding="utf-8") as f:
            M["ops"] = yaml.safe_load(f)
        M["hw"] = M["map"]["hw"]
        M["layer"] = M["ops"].get("layer") or M["model"].get("layer_label", "").replace(" 층 1개", "")
        _M[mid] = M
    return _M[mid]


def dims(M, B, T):
    d = tr.dims(M, B, T)
    if "S_acc" in d:
        d["S_acc"], d["U_dir"] = 0, B
    return d


def ukey(unit, prec):
    return ("gemm", prec or "bf16") if unit == "gemm" else ("vec", "fp32")


def peak(hw, unit, prec):
    return hw["peak"][prec or "bf16"] if unit == "gemm" else hw["peak"]["vec"]


def rows(mid, B, T):
    """연산 줄 — 모델 전체 합. 줄마다 flops · bytes · t_mem · t_cmp · t · units{(unit, prec): FLOP}."""
    M = load(mid)
    d, hw = dims(M, B, T), M["hw"]
    L = ev(M["ops"]["layers"], d)
    _, steps = bl.merged(M["algo"], M["map"])
    kmap = {k["id"]: k for k in M["map"]["kernels"]}
    out = []
    for k in bl.kernel_numbers(M["algo"], M["map"], d):
        km = M["ops"]["kernels"][k["id"]]
        if km.get("skip"):
            continue
        units = {}
        for sid in kmap[k["id"]]["steps"]:
            s = steps[sid]
            if s.get("flops"):
                u = ukey(s["unit"], s.get("prec"))
                units[u] = units.get(u, 0) + ev(s["flops"], d) * L
        out.append({"id": k["id"], "label": km["short"], "layer": M["layer"], "cat": km["cat"], "drawn": True, "n": L,
                    "flops": k["flops"] * L, "bytes": (k["read"] + k["write"]) * L,
                    "t_mem": k["t_mem"] * L, "t_cmp": k["t_cmp"] * L, "t": k["t"] * L, "units": units})
    for o in M["ops"]["ops"]:
        n = ev(o["layers"], d)
        fl, by = ev(o["flops"], d) * n, ev(o["bytes"], d) * n
        tm, tc = by / hw["hbm_bw"], fl / peak(hw, o["unit"], o.get("prec"))
        out.append({"id": o["id"], "label": o["label"], "layer": o.get("layer", "공통"), "cat": o["cat"], "drawn": False,
                    "n": n, "flops": fl, "bytes": by, "t_mem": tm, "t_cmp": tc, "t": max(tm, tc),
                    "units": {ukey(o["unit"], o.get("prec")): fl} if fl else {}})
    return out


def by_cat(rs, field):
    out = {}
    for r in rs:
        out[r["cat"]] = out.get(r["cat"], 0) + r[field]
    return out


def by_unit(rs):
    out = {}
    for r in rs:
        for u, v in r["units"].items():
            out[u] = out.get(u, 0) + v
    return out


def bound_time(rs):
    """이론 시간을 '연산이 정함 · 메모리가 정함'으로 — 줄마다 큰 쪽."""
    c = sum(r["t"] for r in rs if r["t_cmp"] > r["t_mem"])
    return {"cmp": c, "mem": sum(r["t"] for r in rs) - c}


def intensity(mid, B, T, cats, layer=None):
    """묶음들의 FLOP ÷ HBM 바이트 (층 종류를 고르면 그 층만)."""
    rs = [r for r in rows(mid, B, T) if r["cat"] in cats and (layer is None or r["layer"] == layer)]
    by = sum(r["bytes"] for r in rs)
    return sum(r["flops"] for r in rs) / by if by else None


def per_token(mid, T):
    """요청 1개 · 토큰 1개의 묶음별 FLOP (모델 전체)."""
    return by_cat(rows(mid, 1, T), "flops")


def cross_ctx(mid, lo=1024, hi=1 << 24):
    """요청별 데이터 연산 FLOP ≥ 가중치 GEMM FLOP이 되는 문맥 길이 (이분 탐색). 넘지 않으면 None."""
    def ok(T):
        p = per_token(mid, T)
        return sum(p.get(c, 0) for c in REQ) >= sum(p.get(c, 0) for c in WEIGHT)
    if not ok(hi):
        return None
    if ok(lo):
        return lo
    while hi - lo > 64:
        m = (lo + hi) // 2
        lo, hi = (lo, m) if ok(m) else (m, hi)
    return hi


# ------------------------------------------------------------------ 불규칙 연산 — top-k · gather
def irregular(mid, T=131072):
    """sparse 층 종류마다: 층 수, 요청 1개 · 층 1개의 top-k 후보 수 · 고르는 수, gather 덩어리 크기 · 수."""
    M = load(mid)
    d = dims(M, 1, T)
    out = []
    st = {s["id"]: s for s in M["algo"]["steps"]}
    sel = next((s for s in st.values() if s["unit"] == "sel"), None)
    if sel:
        tens = M["algo"]["tensors"]
        pick = ev(tens["idx"]["shape"][1], d)
        chunk = ev(tens["kv_sel"]["bytes"], d) / pick
        out.append({"layer": M["layer"], "n": ev(M["ops"]["layers"], d), "cand": ev(sel["elems"], d),
                    "pick": pick, "chunk": chunk, "row": ev(tens["kv_sel"]["bytes"], d) / ev(tens["kv_sel"]["shape"][1], d)})
    for o in M["ops"]["ops"]:
        if o["cat"] == "topk":
            att = next(x for x in M["ops"]["ops"] if x.get("layer") == o.get("layer") and "gather" in x)
            g = att["gather"]
            out.append({"layer": o["layer"], "n": ev(o["layers"], d), "cand": ev(o["elems"], d),
                        "pick": ev(g["rows"], d), "chunk": ev(g["row"], d), "row": ev(g["row"], d)})
    return out


# ------------------------------------------------------------------ 직렬 경로 — 커널 의존 깊이
def chain(mp, skip=()):
    """커널마다 깊이(1부터). 반환: [(커널, 깊이)] — 커널 순서."""
    ks = [k for k in mp["kernels"] if k["id"] not in skip]
    wr, lev = [], {}
    for k in ks:
        rd = {x for e in k["hbm_read"] for x in (e["t"], e.get("via", e["t"]))}
        dep = [lev[i["id"]] for i, w in wr if rd & w]
        lev[k["id"]] = 1 + max(dep, default=0)
        wr.append((k, {x for e in k.get("hbm_write", []) for x in (e["t"], e.get("via", e["t"]))}))
    return [(k, lev[k["id"]]) for k in ks]


def chains(mid):
    """층 종류마다 {name, n(층 수), chain[(커널, 깊이)], src(배치 출처 모델), kinfo(커널 → 묶음 · 짧은 이름)}.
    그린 층 + ops.yaml chains (그린 층 밖 층 종류 — 같은 구조 모델의 배치로 대신 셈)."""
    M = load(mid)
    d = dims(M, 1, 131072)
    ki = M["ops"]["kernels"]
    skip = [k for k, v in ki.items() if v.get("skip")]
    out = [{"name": M["layer"], "n": ev(M["ops"]["layers"], d), "chain": chain(M["map"], skip), "src": mid, "kinfo": ki}]
    for name, c in (M["ops"].get("chains") or {}).items():
        if c["from"] == mid:
            mp, kinfo = M["map"], ki
        else:
            mp = bl.load(os.path.join(bl.ROOT, "models", c["from"]))[1]
            kinfo = c.get("kernels") or load(c["from"])["ops"]["kernels"]
        out.append({"name": name, "n": ev(str(c["layers"]), d), "chain": chain(mp, c.get("skip", ())), "src": c["from"],
                    "kinfo": kinfo})
    return out


def kernel_times(mid, B, T):
    """그린 층 1개의 커널별 이론 하한 (µs)."""
    M = load(mid)
    return {k["id"]: k["t"] * 1e6 for k in bl.kernel_numbers(M["algo"], M["map"], dims(M, B, T))}
