"""algo.yaml + map-gpu.yaml → 검증 · 수치 · 뷰어 데이터 (DeepSeek-V3.2 DSA).

  python build.py

1. 검증 : ref_decode.py를 실제 차원(T = 131,072)의 meta device로 실행해
           yaml의 텐서 shape 식과 GEMM FLOP 식이 실행 결과와 같은지 확인한다.
2. 수치 : 커널별 HBM 읽기 · 쓰기 바이트, FLOP, 이론 하한 시간 → numbers.md
3. 데이터: dataflow.json, ../../viewer/dsa-v32-lite.html (공유용) · ../../viewer/artifact/dsa-v32-lite.html
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, HERE)

from dflow import Rec, ev  # noqa: E402
import buildlib as bl  # noqa: E402
from ref_decode import V32_CFG, decode_step, new_cache, weights_meta  # noqa: E402


def verify(algo):
    dims = bl.dims_of(algo)
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
        exp = [ev(e, dims) for e in t["shape"]]
        got = traced.get(t.get("trace", tid), {}).get("shape")
        bad += got != exp
        out.append({"what": f"shape {tid}", "expect": exp, "traced": got, "ok": got == exp})
    assert rec.order == [s["id"] for s in algo["steps"]], (rec.order, "단계 순서가 algo.yaml과 다르다")
    for s in algo["steps"]:
        got, exp = rec.gemm_flops(s["id"]), bl.check_flops_expected(s, dims)
        bad += got != exp
        out.append({"what": f"gemm_flops {s['id']}", "expect": exp, "traced": got, "ok": got == exp})
    return out, bad


def numbers_md(algo, mp, checks, bad):
    L = ["# DSA (DeepSeek-V3.2) — 층 1개 decode 1스텝 수치", "",
         "build.py가 algo.yaml · map-gpu.yaml에서 계산한다. 손으로 고치지 않는다.", "",
         "기준: attention 블록만 (MoE · FFN 제외), 요청 1개, H200 이론 하한.", ""]
    L += bl.kernel_tables(algo, mp)
    dims = bl.dims_of(algo, mp, "bf16")
    t, steps = algo["tensors"], {s["id"]: s for s in algo["steps"]}
    idx_rd = ev(t["idx_cache"]["bytes"], dims)
    fl_i = ev(steps["i5"]["flops"], dims)
    fl_a = ev(steps["a2"]["flops"], dims) + ev(steps["a4"]["flops"], dims)
    L += ["## 보고서 수치와 대조 (128K · KV BF16)", "",
          "| 항목 | 계산값 | 보고서 |", "|---|---:|---|",
          f"| 색인 키 읽기 | {bl.fmt_b(idx_rd)} | 17 MB (§1.5) |",
          f"| 색인 점수 FLOP | {fl_i / 1e9:.3g} GFLOP | 2.15 GFLOP (§1.5) |",
          f"| 색인 ÷ 본 attention FLOP | {fl_i / fl_a:.2f}배 | 약 3.8배 (§1.5) |",
          f"| 색인 산술 강도 | {fl_i / idx_rd:.0f} FLOP/B | 124 FLOP/B (§2.2) |",
          f"| 선택 KV 읽기 | {bl.fmt_b(ev(t['kv_sel']['bytes'], dims))} | 1.2 KB × 2,048 ≈ 2.4 MB (§1.5 · §2.2) |",
          f"| 선택 위치 | {bl.fmt_b(ev(t['idx']['bytes'], dims))} | 8 KB (§2.2) |", ""]
    L += bl.cross_table(algo, mp)
    L += ["## 검증 (meta 실행, T = 131,072)", "", f"shape · GEMM FLOP 검사 {len(checks)}개 중 실패 {bad}개.", ""]
    return "\n".join(L) + "\n"


def main():
    algo, mp = bl.load(HERE)
    checks, bad = verify(algo)
    for c in checks:
        if not c["ok"]:
            print("FAIL", c)
    print(f"검증: {len(checks)}개 중 실패 {bad}개")
    bl.write_outputs(HERE, algo, mp, checks, bad, numbers_md(algo, mp, checks, bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
