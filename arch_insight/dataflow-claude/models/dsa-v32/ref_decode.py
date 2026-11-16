"""DeepSeek-V3.2 DSA — 층 1개 · decode 1스텝 참조 구현 (배포 커널 동작 기준).

HF `transformers.models.deepseek_v32` 와 출력이 같다 (test_ref.py). 다른 점은 계산 순서다.

  1. 흡수형 MLA : KV cache에는 c_kv(512) + k_rope(64) = 576값만 둔다.
                  W_UK는 쿼리 쪽, W_UV는 출력 쪽에 곱한다 (FlashMLA 방식).
                  HF는 KV를 head 128개로 풀어서 캐시한다.
  2. 선택 gather : top-k 이후 마스크를 씌우지 않고, 선택한 2,048줄만 모아서 attention 한다
                  (FlashMLA sparse decode 방식). HF는 전체 T에 -inf 마스크를 더한다.
  3. fp8=True   : 공식 inference 코드처럼 색인 q · k에 Hadamard 회전 + FP8(e4m3) 양자화를
                  적용한다. KV는 656 B 행 형식(c_kv FP8 + scale 4개 + k_rope BF16)을 값으로 흉내 낸다.

단계 id는 algo.yaml · map-gpu.yaml의 steps와 1:1로 대응한다.
가중치는 [입력, 출력] 방향으로 둔다 (x @ W).
"""
import math
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from dflow import NullRec  # noqa: E402

FP8_MAX = 448.0


# ── 기본 연산 ──────────────────────────────────────────────────────────────

def rms_norm(x, g, eps=1e-6):
    x32 = x.float()
    return g * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rope_cos_sin(pos, dim, theta, like):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=like.device, dtype=torch.float32) / dim))
    f = pos * inv
    emb = torch.cat([f, f])
    return emb.cos().to(like.dtype), emb.sin().to(like.dtype)


def rope_interleave(x, cos, sin):
    """MLA용 RoPE. 짝 (x0,x1),(x2,x3)… 회전 후 [짝수|홀수] 순으로 둔다 (HF와 같은 배치)."""
    h = cos.shape[-1] // 2
    c, s = cos[:h], sin[:h]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def rope_half(x, cos, sin):
    """색인용 RoPE (앞뒤 절반 회전, non-interleaved)."""
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos + rot * sin


def hadamard_matrix(n, like):
    h = torch.ones(1, 1, device=like.device, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h * n ** -0.5).to(like.dtype)


def act_quant(x, block=128):
    """블록별 FP8 양자화 (scale은 2의 거듭제곱, ue8m0). 반환: (fp8 값, scale[..., n_block])."""
    shp = x.shape
    xb = x.reshape(*shp[:-1], shp[-1] // block, block).float()
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / FP8_MAX)))
    q = (xb / scale).to(torch.float8_e4m3fn)
    return q.reshape(shp), scale.squeeze(-1)


def dequant(q, scale, block=128):
    shp = q.shape
    return (q.float().reshape(*shp[:-1], shp[-1] // block, block) * scale.unsqueeze(-1)).reshape(shp)


def kv_656(kv, c_kv):
    """656 B 행 형식의 값 흉내: c_kv 512 → FP8 (128마다 scale), k_rope 64 → BF16."""
    q, s = act_quant(kv[..., :c_kv])
    rope = kv[..., c_kv:].to(torch.bfloat16).to(kv.dtype)
    return torch.cat([dequant(q, s).to(kv.dtype), rope], -1)


# ── 캐시 ─────────────────────────────────────────────────────────────────

def new_cache(cfg, B, T_max, like, fp8=False):
    c = {"kv": torch.zeros(B, T_max, cfg["c_kv"] + cfg["d_rope"], device=like.device, dtype=like.dtype)}
    if fp8:
        c["ik"] = torch.zeros(B, T_max, cfg["d_I"], device=like.device, dtype=torch.float8_e4m3fn)
        c["ik_s"] = torch.zeros(B, T_max, cfg["d_I"] // 128, device=like.device, dtype=torch.float32)
    else:
        c["ik"] = torch.zeros(B, T_max, cfg["d_I"], device=like.device, dtype=like.dtype)
    return c


# ── decode 1스텝 ────────────────────────────────────────────────────────────

def decode_step(W, cfg, cache, x, pos, rec=None, fp8=False):
    """x: [B, d] (input_layernorm 이후), pos: 새 토큰 위치. 반환 y: [B, d] (잔차 더하기 전)."""
    rec = rec or NullRec()
    B = x.shape[0]
    T = pos + 1
    n_h, d_nope, d_rope, d_v = cfg["n_h"], cfg["d_nope"], cfg["d_rope"], cfg["d_v"]
    c_kv, n_I, d_I, k = cfg["c_kv"], cfg["n_I"], cfg["d_I"], cfg["k"]
    cos, sin = rope_cos_sin(pos, d_rope, cfg["theta"], x)
    bidx = torch.arange(B, device=x.device)[:, None]

    # ── MLA 입력 투영 ──
    with rec.step("p1"):                                   # x → q_a, kv_a (같은 x를 쓰는 GEMM 2개)
        q_a = rec.t("q_a", x @ W["W_qa"])                  # [B, c_q]
        kv_a = rec.t("kv_a", x @ W["W_kva"])               # [B, c_kv + d_rope]
    with rec.step("p2"):                                   # KV latent 정규화 + RoPE
        c_lat = rms_norm(kv_a[:, :c_kv], W["g_kva"])
        k_r = rope_interleave(kv_a[:, c_kv:], cos, sin)
        kv_new = rec.t("kv_new", torch.cat([c_lat, k_r], -1))   # [B, 576]
    with rec.step("p3"):                                   # KV cache에 1줄 쓰기
        cache["kv"][:, pos] = kv_656(kv_new, c_kv) if fp8 else kv_new
    with rec.step("p4"):                                   # q latent 정규화 + 업투영
        qr = rec.t("qr", rms_norm(q_a, W["g_qa"]))         # [B, c_q]  (색인도 이 값을 쓴다)
        q = rec.t("q", (qr @ W["W_qb"]).view(B, n_h, d_nope + d_rope))
    with rec.step("p5"):                                   # q RoPE + W_UK 흡수
        q_nope, q_rope = q.split([d_nope, d_rope], -1)
        q_rope = rope_interleave(q_rope, cos, sin)
        q_lat = torch.einsum("bhd,hdc->bhc", q_nope, W["W_UK"])     # [B, n_h, c_kv]
        q_full = rec.t("q_full", torch.cat([q_lat, q_rope], -1))   # [B, n_h, 576]

    # ── 색인 (lightning indexer) ──
    with rec.step("i1"):                                   # 색인 q: qr → [n_I, d_I]
        qI = (qr @ W["W_Iq"]).view(B, n_I, d_I)
        qI = torch.cat([rope_half(qI[..., :d_rope], cos, sin), qI[..., d_rope:]], -1)
        if fp8:
            qI_q, qI_s = act_quant(qI @ hadamard_matrix(d_I, qI))   # qI_s: [B, n_I, 1]
            qI = rec.t("qI", qI_q)
        else:
            qI = rec.t("qI", qI)
    with rec.step("i2"):                                   # 색인 k: x → [d_I]
        kI = F.layer_norm(x @ W["W_Ik"], (d_I,), W["ln_Ik_w"], W["ln_Ik_b"], 1e-6)
        kI = torch.cat([rope_half(kI[:, :d_rope], cos, sin), kI[:, d_rope:]], -1)
        if fp8:
            kI_q, kI_s = act_quant(kI @ hadamard_matrix(d_I, kI))
            kI = rec.t("kI", kI_q)
        else:
            kI = rec.t("kI", kI)
    with rec.step("i3"):                                   # 색인 K cache에 1줄 쓰기
        cache["ik"][:, pos] = kI
        if fp8:
            cache["ik_s"][:, pos] = kI_s
    with rec.step("i4"):                                   # head 가중치 w
        w = (x @ W["W_Iw"]) * (n_I ** -0.5) * (d_I ** -0.5)
        if fp8:
            w = w * qI_s.squeeze(-1)
        w = rec.t("wI", w)                                 # [B, n_I]
    with rec.step("i5"):                                   # 색인 점수: 모든 T줄과 내적
        ik = cache["ik"][:, :T]
        logits = rec.t("logits", torch.einsum("bhd,btd->bht", qI.float(), ik.float()))  # [B, n_I, T]
    with rec.step("i6"):                                   # ReLU → head 가중합
        s = torch.einsum("bh,bht->bt", w.float(), F.relu(logits))
        if fp8:
            s = s * cache["ik_s"][:, :T, 0]
        s = rec.t("s", s)                                  # [B, T]

    # ── 선택 ──
    with rec.step("s1"):                                   # top-k
        idx = rec.t("idx", s.topk(min(k, T), -1).indices)  # [B, k]

    # ── sparse MLA ──
    with rec.step("a1"):                                   # 선택 행 gather
        kv_sel = rec.t("kv_sel", cache["kv"][bidx, idx])   # [B, k, 576]
    with rec.step("a2"):                                   # QK: head 128개가 같은 행을 공유
        sc = rec.t("sc", torch.einsum("bhc,bkc->bhk", q_full, kv_sel) * cfg["scale"])  # [B, n_h, k]
    with rec.step("a3"):
        p = rec.t("p", torch.softmax(sc.float(), -1).to(x.dtype))
    with rec.step("a4"):                                   # PV: V = c_kv 부분 (512)
        o_lat = rec.t("o_lat", torch.einsum("bhk,bkc->bhc", p, kv_sel[..., :c_kv]))  # [B, n_h, c_kv]

    # ── 출력 ──
    with rec.step("o1"):                                   # W_UV 풀기
        o = rec.t("o", torch.einsum("bhc,hcv->bhv", o_lat, W["W_UV"]))  # [B, n_h, d_v]
    with rec.step("o2"):
        y = rec.t("y", o.reshape(B, n_h * d_v) @ W["W_o"])  # [B, d]
    return y


# ── 가중치 ─────────────────────────────────────────────────────────────────

def weights_from_hf(attn):
    """HF DeepseekV32Attention 모듈 → 참조 구현 가중치 dict."""
    n_h = attn.num_heads
    d_nope, d_v, c_kv = attn.qk_nope_head_dim, attn.v_head_dim, attn.kv_lora_rank
    kvb = attn.kv_b_proj.weight.view(n_h, d_nope + d_v, c_kv)
    ix = attn.indexer
    return {
        "W_qa": attn.q_a_proj.weight.T, "g_qa": attn.q_a_layernorm.weight,
        "W_qb": attn.q_b_proj.weight.T,
        "W_kva": attn.kv_a_proj_with_mqa.weight.T, "g_kva": attn.kv_a_layernorm.weight,
        "W_UK": kvb[:, :d_nope, :], "W_UV": kvb[:, d_nope:, :].transpose(1, 2),
        "W_o": attn.o_proj.weight.T,
        "W_Iq": ix.wq_b.weight.T, "W_Ik": ix.wk.weight.T,
        "ln_Ik_w": ix.k_norm.weight, "ln_Ik_b": ix.k_norm.bias,
        "W_Iw": ix.weights_proj.weight.T,
    }


def weights_meta(cfg):
    """실제 차원의 가중치를 meta device로 만든다 (메모리 0)."""
    d, c_q, n_h = cfg["d"], cfg["c_q"], cfg["n_h"]
    dn, dr, dv, ckv = cfg["d_nope"], cfg["d_rope"], cfg["d_v"], cfg["c_kv"]
    nI, dI = cfg["n_I"], cfg["d_I"]
    m = lambda *s: torch.empty(*s, device="meta", dtype=torch.bfloat16)  # noqa: E731
    return {
        "W_qa": m(d, c_q), "g_qa": m(c_q), "W_qb": m(c_q, n_h * (dn + dr)),
        "W_kva": m(d, ckv + dr), "g_kva": m(ckv),
        "W_UK": m(n_h, dn, ckv), "W_UV": m(n_h, ckv, dv), "W_o": m(n_h * dv, d),
        "W_Iq": m(c_q, nI * dI), "W_Ik": m(d, dI), "ln_Ik_w": m(dI), "ln_Ik_b": m(dI),
        "W_Iw": m(d, nI),
    }


# shape · FLOP 검사 전용 차원. 실제 V3.2의 YaRN(factor 40, mscale)은 생략했다 — 실제 가중치로 돌리면 출력이 달라진다.
V32_CFG = {
    "d": 7168, "c_q": 1536, "n_h": 128, "d_nope": 128, "d_rope": 64, "d_v": 128, "c_kv": 512,
    "n_I": 64, "d_I": 128, "k": 2048, "theta": 10000.0,
    "scale": (128 + 64) ** -0.5,
}
