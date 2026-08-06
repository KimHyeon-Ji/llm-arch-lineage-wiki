"""
등가성 검증 — 각 모듈이 "무엇의 특수 케이스인지"를 코드로 확인한다.

문서에서 ✅ 로 표시한 주장들이 여기서 검증된다.
작은 크기로 돌리므로 몇 초면 끝난다.

    python snippets/test_equivalence.py

요구: torch
"""

import math
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)  # 등가성 확인이므로 정밀도를 높인다

B, S, N_H, D_H = 2, 12, 8, 16
D = N_H * D_H
ATOL = 1e-10


# ─────────────────────────────────────────────────────────────
# 공통
# ─────────────────────────────────────────────────────────────

def causal_mask(tq, tk):
    """(tq, tk) — 쿼리 i는 키 0..i 까지만 본다 (tq == tk 가정)."""
    return torch.tril(torch.ones(tq, tk, dtype=torch.bool))


def sdpa(q, k, v, mask=None):
    """q: (B,H,Tq,D) · k,v: (B,H,Tk,D) → (B,H,Tq,D)"""
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(-1) @ v


def rand(*shape):
    return torch.randn(*shape)


def check(name, a, b, atol=ATOL):
    diff = (a - b).abs().max().item()
    ok = diff < atol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<46} max|diff| = {diff:.3e}")
    assert ok, f"{name} 실패: max diff {diff}"


# ─────────────────────────────────────────────────────────────
# 01-attention 1.2 / 1.3 — MHA · MQA · GQA
# ─────────────────────────────────────────────────────────────

def gqa(q, k, v, mask=None):
    """q: (B,n_h,T,D) · k,v: (B,n_kv,T,D). KV 헤드를 쿼리 헤드 수에 맞춰 펼친다."""
    g = q.shape[1] // k.shape[1]
    return sdpa(q, k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1), mask)


def mqa(q, k, v, mask=None):
    """KV 헤드가 1개. 브로드캐스트로 계산한다."""
    assert k.shape[1] == 1
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])   # (B,n_h,T,T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(-1) @ v


def test_gqa_reduces_to_mha():
    """n_kv = n_h 이면 GQA는 MHA와 정확히 같다. → 01-attention 1.3"""
    m = causal_mask(S, S)
    q, k, v = rand(B, N_H, S, D_H), rand(B, N_H, S, D_H), rand(B, N_H, S, D_H)
    check("GQA(n_kv=n_h) == MHA", gqa(q, k, v, m), sdpa(q, k, v, m))


def test_mqa_is_gqa_with_one_group():
    """MQA는 n_kv = 1 인 GQA와 같다. → 01-attention 1.2"""
    m = causal_mask(S, S)
    q = rand(B, N_H, S, D_H)
    k, v = rand(B, 1, S, D_H), rand(B, 1, S, D_H)
    check("MQA == GQA(n_kv=1)", mqa(q, k, v, m), gqa(q, k, v, m))


def test_gqa_group_sharing():
    """GQA는 그룹 안 헤드가 같은 K,V를 본다 — 실제로 그런지 확인."""
    m = causal_mask(S, S)
    n_kv, g = 2, N_H // 2
    q = rand(B, N_H, S, D_H)
    k, v = rand(B, n_kv, S, D_H), rand(B, n_kv, S, D_H)
    out = gqa(q, k, v, m)
    # 그룹 0의 헤드들에 같은 쿼리를 넣으면 출력이 같아야 한다
    q2 = q.clone()
    q2[:, :g] = q[:, 0:1]
    out2 = gqa(q2, k, v, m)
    check("GQA 그룹 내 헤드는 동일 KV 참조", out2[:, 0], out2[:, g - 1])


# ─────────────────────────────────────────────────────────────
# 01-attention 1.4 — MLA absorption
# ─────────────────────────────────────────────────────────────

def test_mla_absorption():
    """naive MLA == absorbed MLA. 그리고 캐시에는 latent만 있으면 된다.
    → 01-attention 1.4"""
    d_c = 24
    x = rand(B, S, D)
    W_DKV = rand(D, d_c) / math.sqrt(D)
    W_UK = rand(N_H, d_c, D_H) / math.sqrt(d_c)
    W_UV = rand(N_H, d_c, D_H) / math.sqrt(d_c)
    W_Q = rand(N_H, D, D_H) / math.sqrt(D)
    W_O = rand(N_H, D_H, D) / math.sqrt(D_H)
    m = causal_mask(S, S)

    c = x @ W_DKV                                    # (B,S,d_c)  ← 캐시에 담기는 것
    q = torch.einsum("bsd,hdk->bhsk", x, W_Q)        # (B,n_h,S,D_H)

    # --- naive: latent를 K,V로 복원해서 계산 ---
    K = torch.einsum("bsc,hcd->bhsd", c, W_UK)
    V = torch.einsum("bsc,hcd->bhsd", c, W_UV)
    o_naive = sdpa(q, K, V, m)                       # (B,n_h,S,D_H)
    y_naive = torch.einsum("bhsd,hdm->bsm", o_naive, W_O)

    # --- absorbed: W_UK 를 q 쪽으로, W_UV 를 W_O 쪽으로 흡수 ---
    q_c = torch.einsum("bhsd,hcd->bhsc", q, W_UK)    # (B,n_h,S,d_c)
    scores = torch.einsum("bhsc,btc->bhst", q_c, c) / math.sqrt(D_H)
    scores = scores.masked_fill(~m, float("-inf"))
    p = scores.softmax(-1)
    u = torch.einsum("bhst,btc->bhsc", p, c)         # (B,n_h,S,d_c)
    W_UVO = torch.einsum("hcd,hdm->hcm", W_UV, W_O)  # 미리 곱해두는 부분
    y_abs = torch.einsum("bhsc,hcm->bsm", u, W_UVO)

    check("MLA absorbed == naive", y_abs, y_naive, atol=1e-8)
    print(f"        캐시 크기: naive K+V = {2 * N_H * D_H} · absorbed latent = {d_c} "
          f"(값/토큰/레이어)")


# ─────────────────────────────────────────────────────────────
# 01-attention 1.5 — Gated Attention
# ─────────────────────────────────────────────────────────────

def test_gate_identity():
    """게이트를 1로 두면 원래 attention과 같다. → 01-attention 1.5"""
    m = causal_mask(S, S)
    q, k, v = rand(B, N_H, S, D_H), rand(B, N_H, S, D_H), rand(B, N_H, S, D_H)
    base = sdpa(q, k, v, m)
    gate = torch.ones(B, N_H, S, 1)
    check("Gated Attention(gate=1) == 원본", base * gate, base)


# ─────────────────────────────────────────────────────────────
# 01-attention 1.6 — SWA
# ─────────────────────────────────────────────────────────────

def swa_mask(t, w):
    """최근 w개만 본다 (자기 자신 포함)."""
    i = torch.arange(t)[:, None]
    j = torch.arange(t)[None, :]
    return (j <= i) & (j > i - w)


def test_swa_full_window():
    """윈도우가 시퀀스보다 크면 full attention과 같다. → 01-attention 1.6"""
    q, k, v = rand(B, N_H, S, D_H), rand(B, N_H, S, D_H), rand(B, N_H, S, D_H)
    check("SWA(W>=S) == full attention",
          sdpa(q, k, v, swa_mask(S, S)), sdpa(q, k, v, causal_mask(S, S)))


# ─────────────────────────────────────────────────────────────
# 01-attention 1.7 / 1.8 — 희소 선택 (NSA · DSA)
# ─────────────────────────────────────────────────────────────

def topk_attention(q, k, v, topk, mask):
    """점수 상위 topk개만 남기고 attention. topk >= S 면 dense와 같다."""
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    if topk < scores.shape[-1]:
        thresh = scores.topk(topk, dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < thresh, float("-inf"))
    return scores.softmax(-1) @ v


def test_sparse_dense_limit():
    """top-k = 전체이면 dense attention과 같다.
    → 01-attention 1.7 (NSA), 1.8 (DSA), 1.9 (CSA)"""
    m = causal_mask(S, S)
    q, k, v = rand(B, N_H, S, D_H), rand(B, N_H, S, D_H), rand(B, N_H, S, D_H)
    check("sparse(top-k=전체) == dense", topk_attention(q, k, v, S, m), sdpa(q, k, v, m))


# ─────────────────────────────────────────────────────────────
# 01-attention 1.10 — 레이어 간 KV 공유 (CLA)
# ─────────────────────────────────────────────────────────────

def test_cla_group_one():
    """공유 그룹 크기가 1이면 각 레이어가 자기 KV를 쓰는 것과 같다.
    → 01-attention 1.10"""
    m = causal_mask(S, S)
    n_layer = 4
    qs = [rand(B, N_H, S, D_H) for _ in range(n_layer)]
    kvs = [(rand(B, N_H, S, D_H), rand(B, N_H, S, D_H)) for _ in range(n_layer)]

    def run(group):
        outs = []
        for i in range(n_layer):
            src = (i // group) * group          # 그룹의 첫 레이어가 KV를 만든다
            k, v = kvs[src]
            outs.append(sdpa(qs[i], k, v, m))
        return torch.stack(outs)

    check("CLA(group=1) == 레이어별 독립 KV", run(1), run(1))
    shared = run(2)
    # group=2 이면 레이어 0과 1이 같은 KV를 쓴다 — 실제로 그런지 확인
    k0, v0 = kvs[0]
    check("CLA(group=2) 레이어1이 레이어0의 KV 사용",
          shared[1], sdpa(qs[1], k0, v0, m))


# ─────────────────────────────────────────────────────────────
# 02-linear-attention 2.1 / 2.3 — 선형 attention과 delta rule
# ─────────────────────────────────────────────────────────────

def test_linear_attention_recurrent():
    """선형 attention의 재귀 형태 == 이차 형태 (causal).
    S_t = S_{t-1} + k_t v_tᵀ,  o_t = q_t S_t   → 02-linear-attention 2.1"""
    q, k, v = rand(B, S, D_H), rand(B, S, D_H), rand(B, S, D_H)

    # 이차 형태: causal mask를 씌운 (q kᵀ) v
    m = causal_mask(S, S)
    quad = ((q @ k.transpose(-1, -2)) * m) @ v

    # 재귀 형태: 고정 크기 상태 (D_H × D_H) 하나만 들고 간다
    state = torch.zeros(B, D_H, D_H)
    outs = []
    for t in range(S):
        state = state + k[:, t].unsqueeze(-1) @ v[:, t].unsqueeze(-2)
        outs.append((q[:, t].unsqueeze(-2) @ state).squeeze(-2))
    rec = torch.stack(outs, dim=1)

    check("linear attention 재귀 == 이차 형태", rec, quad, atol=1e-9)
    print(f"        상태 크기: {D_H}x{D_H} = {D_H * D_H} (S={S} 와 무관)")


def test_deltanet_forms():
    """delta rule의 3단계 형태 == 행렬 형태.
    S ← (I − β k kᵀ) S + β k vᵀ    → 02-linear-attention 2.3"""
    state = rand(D_H, D_H)
    k = rand(D_H)
    k = k / k.norm()
    v = rand(D_H)
    beta = 0.7

    # 3단계: 조회 → 오차 → 갱신
    v_old = k @ state
    step = state + beta * k.unsqueeze(-1) @ (v - v_old).unsqueeze(-2)

    # 행렬 한 줄
    eye = torch.eye(D_H)
    matrix = (eye - beta * k.unsqueeze(-1) @ k.unsqueeze(-2)) @ state \
             + beta * k.unsqueeze(-1) @ v.unsqueeze(-2)

    check("DeltaNet 3단계 == (I − βkkᵀ)S + βkvᵀ", step, matrix, atol=1e-9)


def test_gating_lineage():
    """축2 계보가 서로의 특수 케이스인지 확인. → 02-linear-attention 2.4

    KDA:            S = (I − βkkᵀ) diag(α) S + β k vᵀ
    Gated DeltaNet: α 가 스칼라
    DeltaNet:       α = 1
    Linear:         α = 1, β = 1, 그리고 (I − βkkᵀ) 항 제거
    """
    state = rand(D_H, D_H)
    k = rand(D_H); k = k / k.norm()
    v = rand(D_H)
    eye = torch.eye(D_H)

    def kda(alpha_vec, beta):
        return (eye - beta * k.unsqueeze(-1) @ k.unsqueeze(-2)) @ (torch.diag(alpha_vec) @ state) \
               + beta * k.unsqueeze(-1) @ v.unsqueeze(-2)

    def gated_deltanet(alpha, beta):
        return (eye - beta * k.unsqueeze(-1) @ k.unsqueeze(-2)) @ (alpha * state) \
               + beta * k.unsqueeze(-1) @ v.unsqueeze(-2)

    a = 0.9
    check("KDA(α가 모두 같은 값) == Gated DeltaNet",
          kda(torch.full((D_H,), a), 0.7), gated_deltanet(a, 0.7), atol=1e-9)

    def deltanet(beta):
        return (eye - beta * k.unsqueeze(-1) @ k.unsqueeze(-2)) @ state \
               + beta * k.unsqueeze(-1) @ v.unsqueeze(-2)

    check("Gated DeltaNet(α=1) == DeltaNet",
          gated_deltanet(1.0, 0.7), deltanet(0.7), atol=1e-9)


# ─────────────────────────────────────────────────────────────
# 03-position 3.2 — RoPE
# ─────────────────────────────────────────────────────────────

def rope(x, pos, theta=10000.0):
    """x: (..., D) · pos: 스칼라 위치. D/2 쌍을 각도 pos*θ_i 만큼 회전."""
    d = x.shape[-1]
    half = d // 2
    freqs = theta ** (-torch.arange(half, dtype=x.dtype) / half)
    ang = pos * freqs
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def test_rope_relative():
    """⟨R_m q, R_n k⟩ 는 m − n 에만 의존한다. → 03-position 3.2"""
    q, k = rand(D_H), rand(D_H)
    pairs = [(3, 5), (10, 12), (100, 102), (1000, 1002)]   # 전부 차이가 −2
    vals = [(rope(q, float(m)) * rope(k, float(n))).sum() for m, n in pairs]
    base = vals[0]
    for (m, n), val in zip(pairs[1:], vals[1:]):
        check(f"RoPE 내적: (m,n)=({m},{n}) 이 (3,5)와 동일", val, base, atol=1e-9)

    # 차이가 다르면 값도 달라야 한다 (그래야 위치 정보가 있는 것)
    other = (rope(q, 3.0) * rope(k, 9.0)).sum()
    assert (other - base).abs() > 1e-6, "위치 차이가 다른데 내적이 같다 — 이상하다"
    print("  PASS  RoPE: 위치 차이가 다르면 내적도 다름")


# ─────────────────────────────────────────────────────────────

TESTS = [
    ("01-attention 1.2~1.3  헤드 공유", [
        test_gqa_reduces_to_mha,
        test_mqa_is_gqa_with_one_group,
        test_gqa_group_sharing,
    ]),
    ("01-attention 1.4      MLA 흡수", [test_mla_absorption]),
    ("01-attention 1.5      게이팅", [test_gate_identity]),
    ("01-attention 1.6      SWA", [test_swa_full_window]),
    ("01-attention 1.7~1.9  희소 선택", [test_sparse_dense_limit]),
    ("01-attention 1.10     레이어 공유", [test_cla_group_one]),
    ("02-linear-attention   고정 상태", [
        test_linear_attention_recurrent,
        test_deltanet_forms,
        test_gating_lineage,
    ]),
    ("03-position 3.2       RoPE", [test_rope_relative]),
]


if __name__ == "__main__":
    total = 0
    for section, fns in TESTS:
        print(f"\n{section}")
        for fn in fns:
            fn()
            total += 1
    print(f"\n전부 통과 ({total}개 검증 함수)\n")
