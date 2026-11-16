"""Gated DeltaNet (Qwen3.5 / 3.6 linear attention 층) — 층 1개 · decode 1스텝 참조 구현.

배포 커널 흐름을 따른다:
  in_proj GEMM → causal_conv1d_update (conv state K-1 = 3칸) → fused recurrent gated delta rule
  (l2norm · gate · state 감쇠 · 기억 읽기 · rank-1 갱신 · 출력) → gated RMSNorm → out_proj

HF `transformers.models.qwen3_5.Qwen3_5GatedDeltaNet`과 출력이 같다 (test_ref.py).
HF는 conv state를 4칸으로 두지만 실제 커널(causal_conv1d_update)은 K-1 = 3칸이면 충분하다.

단계 id(l1 ~ l10)는 algo.yaml · map-gpu.yaml의 steps와 1:1로 대응한다. 가중치는 [입력, 출력] 방향 (x @ W).
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from dflow import NullRec  # noqa: E402


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def conv_dim(cfg):
    return 2 * cfg["n_k"] * cfg["d_k"] + cfg["n_v"] * cfg["d_v"]


def new_cache(cfg, B, like, state_dtype=torch.float32):
    return {
        "conv": torch.zeros(B, conv_dim(cfg), cfg["d_conv"] - 1, device=like.device, dtype=like.dtype),
        "ssm": torch.zeros(B, cfg["n_v"], cfg["d_k"], cfg["d_v"], device=like.device, dtype=state_dtype),
    }


def decode_step(W, cfg, cache, x, rec=None):
    """x: [B, d] (input_layernorm 이후). 반환 y: [B, d] (잔차 더하기 전). cache는 제자리에서 갱신된다."""
    rec = rec or NullRec()
    B = x.shape[0]
    nk, nv, dk, dv = cfg["n_k"], cfg["n_v"], cfg["d_k"], cfg["d_v"]

    with rec.step("l1"):                                   # 입력 투영: q·k·v · gate z · β · 감쇠 a
        qkv = rec.t("qkv", x @ W["W_qkv"])                 # [B, C]
        z = rec.t("z", x @ W["W_z"])                       # [B, n_v·d_v]
        b = rec.t("b", x @ W["W_b"])                       # [B, n_v]
        a = rec.t("a", x @ W["W_a"])                       # [B, n_v]
    with rec.step("l2"):                                   # 짧은 합성곱 (4탭 depthwise) + SiLU
        win = rec.t("conv_win", torch.cat([cache["conv"], qkv.unsqueeze(-1)], -1))   # [B, C, 4]
        qkv_c = rec.t("qkv_c", F.silu((win * W["w_conv"]).sum(-1)))                 # [B, C]
    with rec.step("l3"):                                   # conv state를 한 칸 밀어 갱신
        cache["conv"].copy_(win[..., 1:])
    with rec.step("l4"):                                   # q·k L2 정규화, head 확장, β · g 계산
        q, k, v = qkv_c.split([nk * dk, nk * dk, nv * dv], -1)
        r = nv // nk
        q = rec.t("q", (l2norm(q.reshape(B, nk, dk).float()) * dk ** -0.5).repeat_interleave(r, 1))   # [B, n_v, d_k]
        k = rec.t("k", l2norm(k.reshape(B, nk, dk).float()).repeat_interleave(r, 1))                  # [B, n_v, d_k]
        v = rec.t("v", v.reshape(B, nv, dv).float())                                                  # [B, n_v, d_v]
        beta = rec.t("beta", torch.sigmoid(b.float()))                                                # [B, n_v]
        g = rec.t("g", -W["A_log"].float().exp() * F.softplus(a.float() + W["dt_bias"].float()))     # [B, n_v]
    with rec.step("l5"):                                   # state 감쇠: S ← S · exp(g)
        S = rec.t("S_dec", cache["ssm"].float() * g.exp()[..., None, None])    # [B, n_v, d_k, d_v]
    with rec.step("l6"):                                   # 기억 읽기: kv_mem = Sᵀ k
        kv_mem = rec.t("kv_mem", torch.einsum("bhkv,bhk->bhv", S, k))           # [B, n_v, d_v]
    with rec.step("l7"):                                   # delta rule 갱신: S ← S + k ⊗ β(v − kv_mem)
        delta = (v - kv_mem) * beta[..., None]
        S = rec.t("S_new", S + k.unsqueeze(-1) * delta.unsqueeze(-2))
        cache["ssm"].copy_(S.to(cache["ssm"].dtype))
    with rec.step("l8"):                                   # 출력 읽기: o = Sᵀ q
        o = rec.t("o", torch.einsum("bhkv,bhk->bhv", S, q))                     # [B, n_v, d_v]
    with rec.step("l9"):                                   # gated RMSNorm: RMSNorm(o) · w · SiLU(z)
        o32 = o.float()
        on = W["w_norm"] * (o32 * torch.rsqrt(o32.pow(2).mean(-1, keepdim=True) + cfg["eps"])).to(x.dtype)
        on = rec.t("o_n", (on * F.silu(z.reshape(B, nv, dv).float())).to(x.dtype))   # [B, n_v, d_v]
    with rec.step("l10"):                                  # 출력 투영
        y = rec.t("y", on.reshape(B, nv * dv) @ W["W_out"])                      # [B, d]
    return y


def weights_from_hf(m):
    """HF Qwen3_5GatedDeltaNet → 참조 구현 가중치 dict."""
    return {
        "W_qkv": m.in_proj_qkv.weight.T, "W_z": m.in_proj_z.weight.T,
        "W_b": m.in_proj_b.weight.T, "W_a": m.in_proj_a.weight.T,
        "w_conv": m.conv1d.weight.squeeze(1), "A_log": m.A_log, "dt_bias": m.dt_bias,
        "w_norm": m.norm.weight, "W_out": m.out_proj.weight.T,
    }


def weights_meta(cfg):
    d, nv, dv, C = cfg["d"], cfg["n_v"], cfg["d_v"], conv_dim(cfg)
    m = lambda *s: torch.empty(*s, device="meta", dtype=torch.bfloat16)  # noqa: E731
    return {
        "W_qkv": m(d, C), "W_z": m(d, nv * dv), "W_b": m(d, nv), "W_a": m(d, nv),
        "w_conv": m(C, cfg["d_conv"]), "A_log": m(nv), "dt_bias": m(nv),
        "w_norm": m(dv), "W_out": m(nv * dv, d),
    }


# Qwen3.6-27B (config.json text_config) — linear 층 차원
Q36_CFG = {"d": 5120, "n_k": 16, "n_v": 48, "d_k": 128, "d_v": 128, "d_conv": 4, "eps": 1e-6}
