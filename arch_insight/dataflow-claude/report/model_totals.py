"""모델 전체 decode 1스텝의 HBM 이동량 — attention 블록은 viewer와 같은 yaml(buildlib)에서, 나머지는 config에서 계산.

기준 (모두 이론 하한: 텐서를 한 번씩만 읽고 쓴다고 본 값)
- 모델 전체가 한 스텝에 HBM에서 읽고 쓰는 바이트의 합. GPU 여러 장에 나눠도 합은 같다고 보고,
  가중치 복제(DP attention 등) · GPU 간 통신 · MTP 층 · 임베딩 행 읽기 · 잔차 · 정규화 값은 뺐다.
- DeepSeek-V3.2 (config: reports/llm-memory-requirements/data/raw/cfg_deepseek-ai_DeepSeek-V3.2.json)
  61층 = dense FFN 3층 + MoE 58층, expert 256개 중 8개 + 공유 1개, FP8 가중치, LM head BF16.
  요청 B개가 건드리는 expert 수는 균등 라우팅 가정: 256 × (1 − (1 − 8/256)^B) → 실제(쏠림)보다 많게 잡힌 상한.
- Qwen3.6-27B (config: cfg_Qwen_Qwen3.6-27B.json) 64층 = linear 48 + full attention 16 (4층마다),
  FFN 17,408 dense, BF16. full 층: q head 24 (+ 출력 gate) · KV head 4 · head 256, KV cache BF16.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import buildlib as bl  # noqa: E402


def load(mid):
    return bl.load(os.path.join(ROOT, "models", mid))


def block_by_kind(algo, mp, B, T):
    """attention 블록 1개(층 1개)의 HBM 읽기 + 쓰기를 데이터 종류별로."""
    d = bl.dims_of(algo, mp, T=T, B=B)
    tens, _ = bl.merged(algo, mp)
    out = {}
    for k in mp["kernels"]:
        for e in k["hbm_read"] + k.get("hbm_write", []):
            kd = tens[e["t"]]["kind"]
            out[kd] = out.get(kd, 0) + bl.ev(tens[e.get("via", e["t"])]["bytes"], d)
    return out


# ── DeepSeek-V3.2 ──
DS = dict(layers=61, dense=3, d=7168, ffn=18432, moe=2048, E=256, topk=8, shared=1, vocab=129280)


def ds_experts(B):
    return DS["E"] * (1 - (1 - DS["topk"] / DS["E"]) ** B)


def dsa_total(B, T):
    algo, mp = load("dsa-v32")
    blk = block_by_kind(algo, mp, B, T)
    L = DS["layers"]
    expert = 3 * DS["d"] * DS["moe"]                          # FP8 1 B/값
    moe_layer = (DS["shared"] + ds_experts(B)) * expert + 2 * DS["d"] * DS["E"]   # + router (BF16)
    parts = {
        "attn_w":  L * blk.get("weight", 0),
        "idxk":    L * blk.get("idxk", 0),
        "kv":      L * blk.get("kv", 0),
        "other":   L * sum(v for kd, v in blk.items() if kd not in ("weight", "idxk", "kv")),
        "ffn":     DS["dense"] * 3 * DS["d"] * DS["ffn"] + (L - DS["dense"]) * moe_layer,
        "lm_head": 2 * DS["d"] * DS["vocab"],
    }
    return parts, blk


# ── Qwen3.6-27B ──
QW = dict(layers=64, lin=48, full=16, d=5120, ffn=17408, vocab=248320, nq=24, nkv=4, hd=256)


def gdn_total(B, T):
    algo, mp = load("gdn-qwen36")
    blk = block_by_kind(algo, mp, B, T)
    full_w = 2 * QW["d"] * (QW["nq"] * QW["hd"] * 2 + 2 * QW["nkv"] * QW["hd"]) + 2 * QW["nq"] * QW["hd"] * QW["d"]
    kv_tok = 2 * QW["nkv"] * QW["hd"] * 2                     # K · V × BF16 = 4,096 B/토큰/층
    parts = {
        "lin_w":   QW["lin"] * blk.get("weight", 0),
        "state":   QW["lin"] * (blk.get("state", 0) + blk.get("conv", 0)),
        "lin_oth": QW["lin"] * sum(v for kd, v in blk.items() if kd not in ("weight", "state", "conv")),
        "full_w":  QW["full"] * full_w,
        "full_kv": QW["full"] * B * (T + 1) * kv_tok,         # 읽기 T줄 + 새 1줄 쓰기
        "ffn":     QW["layers"] * 3 * QW["d"] * QW["ffn"] * 2,
        "lm_head": 2 * QW["d"] * QW["vocab"],
    }
    return parts, blk, full_w


if __name__ == "__main__":
    for B in (1, 8, 32, 128, 256):
        for T in (8192, 32768, 131072):
            p, _ = dsa_total(B, T)
            tot = sum(p.values())
            att = p["attn_w"] + p["idxk"] + p["kv"] + p["other"]
            print(f"DSA B={B:<4} T={T // 1024:>4}K  total {tot / 1e9:8.1f} GB  attn {att / tot:6.1%}  "
                  + "  ".join(f"{k} {v / tot:6.1%}" for k, v in p.items()) + f"  experts {ds_experts(B):.0f}")
    print()
    for B in (1, 8, 32, 128, 256):
        for T in (8192, 32768, 131072):
            p, _, fw = gdn_total(B, T)
            tot = sum(p.values())
            print(f"GDN B={B:<4} T={T // 1024:>4}K  total {tot / 1e9:8.1f} GB  "
                  + "  ".join(f"{k} {v / tot:6.1%}" for k, v in p.items()))
    print("full layer weights", fw / 1e6, "MB")
