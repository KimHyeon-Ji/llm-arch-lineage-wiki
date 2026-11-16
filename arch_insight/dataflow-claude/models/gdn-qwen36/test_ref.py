"""참조 구현(ref_decode.py)이 HF Qwen3_5GatedDeltaNet과 같은 출력을 내는지 확인한다.

작은 차원 · 랜덤 가중치로 T개 토큰을 decode 루프(1스텝 recurrent, conv state 3칸)로 하나씩 넣고,
HF의 전체 시퀀스 forward(chunk gated delta rule, conv1d padding) 같은 위치 출력과 비교한다.

  python test_ref.py          (transformers ≥ 5.x, qwen3_5 포함 환경)

검사
  1. state FP32 : 모든 위치의 출력이 HF와 같다.
  2. state BF16 : state를 BF16으로 저장하면(이동량 절반) 출력이 얼마나 달라지는지 — 정보용.
"""
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from ref_decode import decode_step, new_cache, weights_from_hf


def build(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=96, num_hidden_layers=1, layer_types=["linear_attention"],
        linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, hidden_act="silu", rms_norm_eps=1e-6,
    )
    m = Qwen3_5GatedDeltaNet(cfg, 0).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n == "A_log":
                p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 4)))
            elif n == "dt_bias":
                p.copy_(torch.randn_like(p) * 0.5)
            elif n.startswith("norm"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
    rcfg = {"d": 96, "n_k": 2, "n_v": 6, "d_k": 16, "d_v": 16, "d_conv": 4, "eps": 1e-6}
    return m, rcfg


@torch.no_grad()
def run_ref(m, rcfg, x, state_dtype):
    W = weights_from_hf(m)
    cache = new_cache(rcfg, x.shape[0], x[:, 0], state_dtype=state_dtype)
    return torch.stack([decode_step(W, rcfg, cache, x[:, t]) for t in range(x.shape[1])], 1)


@torch.no_grad()
def main(T=80):
    m, rcfg = build()
    x = torch.randn(1, T, 96)
    hf = m(x)                                              # cache 없음 → chunk 방식 전체 시퀀스
    ref = run_ref(m, rcfg, x, torch.float32)
    err = ((ref - hf).norm(dim=-1) / hf.norm(dim=-1))[0]
    print(f"[1] state FP32  T={T}: 모든 위치 최대 상대 오차 {err.max():.1e}")
    assert err.max() < 1e-4, "참조 구현이 HF와 다르다"
    ref16 = run_ref(m, rcfg, x, torch.bfloat16)
    e16 = ((ref16 - ref).norm(dim=-1) / ref.norm(dim=-1))[0]
    print(f"[2] state BF16  출력 상대 오차 중앙값 {e16.median():.1e} · 최대 {e16.max():.1e} — 정보용")
    print("    통과: 참조 구현(1스텝 recurrent · conv state 3칸)은 HF(chunk 방식)와 같은 출력을 낸다.")


if __name__ == "__main__":
    main()
