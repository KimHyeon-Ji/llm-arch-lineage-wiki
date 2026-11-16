"""algo.yaml + map-gpu.yaml → 검증 · 수치 · 뷰어 데이터 (Qwen3.6-27B Gated DeltaNet).

  python build.py

1. 검증 : ref_decode.py를 실제 차원의 meta device로 실행해 텐서 shape · GEMM FLOP 식을 확인한다.
2. 수치 : 커널별 HBM 읽기 · 쓰기 · FLOP · 이론 하한 → numbers.md
3. 데이터: dataflow.json, ../../viewer/gdn-qwen36.html (공유용) · ../../viewer/artifact/gdn-qwen36.html
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, HERE)

from dflow import Rec, ev  # noqa: E402
import buildlib as bl  # noqa: E402
from ref_decode import Q36_CFG, decode_step, new_cache, weights_meta  # noqa: E402


def verify(algo):
    dims = bl.dims_of(algo)
    B = dims["B"]
    W = weights_meta(Q36_CFG)
    x = torch.empty(B, Q36_CFG["d"], device="meta", dtype=torch.bfloat16)
    cache = new_cache(Q36_CFG, B, x)
    rec = Rec()
    decode_step(W, Q36_CFG, cache, x, rec=rec)

    traced = dict(rec.tensors)
    for n, w in W.items():
        traced[n] = {"shape": list(w.shape)}
    traced["x"] = {"shape": list(x.shape)}
    traced["cache.conv"] = {"shape": list(cache["conv"].shape)}
    traced["cache.ssm"] = {"shape": list(cache["ssm"].shape)}

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
        out.append({"what": f"gemm_flops {s['id']}", "expect": exp, "traced": got, "ok": got == exp,
                    "ops": sorted({o["op"] for o in rec.steps[s["id"]]["ops"]})})
    return out, bad


def numbers_md(algo, mp, checks, bad):
    L = ["# Gated DeltaNet (Qwen3.6-27B linear 층) — 층 1개 decode 1스텝 수치", "",
         "build.py가 algo.yaml · map-gpu.yaml에서 계산한다. 손으로 고치지 않는다.", "",
         "기준: linear attention 블록만 (FFN · full 층 제외), 요청 1개, H200 이론 하한. linear 층은 T와 무관하다.", ""]
    L += bl.kernel_tables(algo, mp)
    d = bl.dims_of(algo, mp)
    t = algo["tensors"]
    st = ev(t["ssm_state"]["bytes"], d)
    cv = ev(t["conv_state"]["bytes"], d)
    kv_full = d["T"] * d["n_kv_f"] * d["d_hf"] * 2 * 2
    L += ["## 보고서 수치와 대조 · full attention 층과 비교 (요청 1개)", "",
          "| 항목 | 계산값 | 비고 |", "|---|---:|---|",
          f"| state + conv state (층당) | {bl.fmt_b(st + cv)} | 보고서 약 3.2 MB/층 (§1.6 · 부록 B, Qwen3.8 FP32) |",
          f"| 매 스텝 state 읽기 + 쓰기 | {bl.fmt_b(2 * st)} | T와 무관 |",
          f"| 같은 위치 full 층의 KV 읽기 (128K) | {bl.fmt_b(kv_full)} | KV head {d['n_kv_f']} × {d['d_hf']} × K·V × BF16 = {d['n_kv_f'] * d['d_hf'] * 4} B/토큰 |",
          f"| state 왕복 ÷ full 층 KV 읽기 | {2 * st / kv_full:.1%} | 128K 기준 |", ""]
    L += bl.cross_table(algo, mp)
    L += ["## 검증 (meta 실행)", "", f"shape · GEMM FLOP 검사 {len(checks)}개 중 실패 {bad}개.", ""]
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
