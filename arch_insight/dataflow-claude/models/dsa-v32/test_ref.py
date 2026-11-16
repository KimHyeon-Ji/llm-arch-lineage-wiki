"""참조 구현(ref_decode.py)이 HF DeepseekV32Attention과 같은 출력을 내는지 확인한다.

작은 차원 · 랜덤 가중치로 T개 토큰을 decode 루프로 하나씩 넣고,
각 위치의 출력을 HF의 전체 시퀀스 forward(causal) 같은 위치 출력과 비교한다.

  python test_ref.py          (transformers ≥ 5.x, deepseek_v32 포함 환경)

검사
  1. fp8=False : 각 위치의 top-k 선택 집합이 HF indexer와 같고, 출력도 같다.
                 (집합이 다른 위치는 경계 점수 차이가 부동소수점 오차 수준인 동점뿐이어야 한다)
  2. fp8=True  : Hadamard + FP8 색인 · 656 B KV 형식을 넣었을 때 bf16 기준 대비
                 top-k 겹침률과 출력 오차 (배포 정밀도의 영향, 정보용)
"""
import torch
from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32Attention, DeepseekV32RotaryEmbedding,
)

from ref_decode import decode_step, new_cache, weights_from_hf


class Grab:
    """참조 구현의 텐서 값을 붙잡아 두는 기록기."""

    def __init__(self, names):
        self.names, self.v = names, {}

    def step(self, sid):
        import contextlib
        return contextlib.nullcontext()

    def t(self, name, x):
        if name in self.names:
            self.v[name] = x.clone()
        return x


def build(seed=0):
    torch.manual_seed(seed)
    cfg = DeepseekV32Config(
        vocab_size=128, hidden_size=256, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=64, kv_lora_rank=128, qk_rope_head_dim=16, qk_nope_head_dim=32, v_head_dim=32,
        index_n_heads=4, index_head_dim=128, index_topk=16, num_hidden_layers=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    cfg._attn_implementation = "eager"
    attn = DeepseekV32Attention(cfg, 0).float().eval()
    with torch.no_grad():
        for n, p in attn.named_parameters():
            if "norm" in n and n.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif "norm" in n:
                p.copy_(0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
    rot = DeepseekV32RotaryEmbedding(cfg)
    rcfg = {
        "d": cfg.hidden_size, "c_q": cfg.q_lora_rank, "n_h": cfg.num_attention_heads,
        "d_nope": cfg.qk_nope_head_dim, "d_rope": cfg.qk_rope_head_dim, "d_v": cfg.v_head_dim,
        "c_kv": cfg.kv_lora_rank, "n_I": cfg.index_n_heads, "d_I": cfg.index_head_dim,
        "k": cfg.index_topk, "theta": 10000.0, "scale": attn.scaling,
    }
    return cfg, attn, rot, rcfg


@torch.no_grad()
def run_ref(attn, rcfg, x, fp8):
    T = x.shape[1]
    W = weights_from_hf(attn)
    cache = new_cache(rcfg, 1, T, x, fp8=fp8)
    ys, idxs, ss = [], [], []
    for t in range(T):
        g = Grab({"idx", "s"})
        ys.append(decode_step(W, rcfg, cache, x[:, t], t, rec=g, fp8=fp8))
        idxs.append(set(g.v["idx"][0].tolist()))
        ss.append(g.v["s"][0])
    return torch.stack(ys, 1), idxs, ss


@torch.no_grad()
def main(T=128):
    cfg, attn, rot, rcfg = build()
    x = torch.randn(1, T, cfg.hidden_size)
    pos = torch.arange(T)[None]
    cos, sin = rot(x, pos)
    hf, _ = attn(x, (cos, sin), attention_mask=None, past_key_values=None, position_ids=pos)
    q_resid = attn.q_a_layernorm(attn.q_a_proj(x))
    hf_idx = attn.indexer(x, q_resid, (cos, sin), None, pos)[0]          # [T, k]

    ref, idxs, ss = run_ref(attn, rcfg, x, fp8=False)
    err = ((ref - hf).norm(dim=-1) / hf.norm(dim=-1))[0]
    k = rcfg["k"]
    same, ties, worst = 0, [], 0.0
    for t in range(T):
        h = {i for i in hf_idx[t].tolist() if i <= t}
        if h == idxs[t]:
            same += 1
            worst = max(worst, err[t].item())
        else:
            srt = ss[t].sort(descending=True).values
            gap = ((srt[k - 1] - srt[k]) / srt[0].abs()).item()
            ties.append((t, gap))
    print(f"[1] fp8=False  T={T}: 선택 집합 일치 {same}/{T} 위치, 그 위치들의 최대 상대 오차 {worst:.1e}")
    for t, gap in ties:
        print(f"    위치 {t}: 집합 다름 — 경계(k번째 vs k+1번째) 점수 상대 차이 {gap:.1e} → 동점")
    assert worst < 1e-4 and all(g < 1e-5 for _, g in ties), "참조 구현이 HF와 다르다"

    ref8, idxs8, _ = run_ref(attn, rcfg, x, fp8=True)
    ov = [len(a & b) / len(b) for a, b in zip(idxs8[k:], idxs[k:])]
    err8 = ((ref8 - ref).norm(dim=-1) / ref.norm(dim=-1))[0, k:]
    print(f"[2] fp8=True   top-k 겹침률 평균 {sum(ov)/len(ov):.1%} (최소 {min(ov):.1%}), "
          f"출력 상대 오차 중앙값 {err8.median():.1e} — 배포 정밀도 영향, 정보용")
    print("    통과: 참조 구현(흡수형 MLA + gather)은 HF(풀어 쓴 MLA + 마스크)와 같은 출력을 낸다.")


if __name__ == "__main__":
    main()
