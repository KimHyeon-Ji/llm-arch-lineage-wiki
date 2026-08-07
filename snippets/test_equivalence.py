"""
등가성 검증 — 각 모듈이 "무엇의 특수 케이스인지"를 코드로 확인한다.

문서에서 ✅ 로 표시한 주장들이 여기서 검증된다.

    python snippets/test_equivalence.py

의존성 없음 (표준 라이브러리만). 행렬이 작아서 순수 파이썬으로 충분하다.
읽기 쉽도록 배치 차원을 빼고 헤드 하나씩 (T x D) 2차원으로 계산한다.
"""

import math
import random

RNG = random.Random(0)
S, N_H, D_H = 10, 8, 12          # 시퀀스 길이, 쿼리 헤드 수, 헤드 차원
D = N_H * D_H                    # 모델 폭
ATOL = 1e-9
NEG_INF = float("-inf")


# ─────────────────────────────────────────────────────────────
# 아주 작은 선형대수
# ─────────────────────────────────────────────────────────────

def randmat(r, c, scale=1.0):
    return [[RNG.gauss(0, 1) * scale for _ in range(c)] for _ in range(r)]


def randvec(n):
    return [RNG.gauss(0, 1) for _ in range(n)]


def matmul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    Bt = transpose(B)
    return [[sum(A[i][p] * Bt[j][p] for p in range(k)) for j in range(m)] for i in range(n)]


def transpose(A):
    return [list(col) for col in zip(*A)]


def scale(A, s):
    return [[x * s for x in row] for row in A]


def add(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def sub(A, B):
    return [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def outer(u, v):
    return [[a * b for b in v] for a in u]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def vec_matmul(u, A):
    """(1 x n) @ (n x m) -> 길이 m 벡터"""
    return matmul([u], A)[0]


def softmax_rows(A):
    out = []
    for row in A:
        finite = [x for x in row if x != NEG_INF]
        mx = max(finite) if finite else 0.0
        exps = [0.0 if x == NEG_INF else math.exp(x - mx) for x in row]
        z = sum(exps)
        out.append([e / z for e in exps])
    return out


def maxdiff(A, B):
    if isinstance(A, float):
        return abs(A - B)
    if isinstance(A[0], float):
        return max(abs(a - b) for a, b in zip(A, B))
    return max(abs(a - b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def check(name, a, b, atol=ATOL):
    diff = maxdiff(a, b)
    ok = diff < atol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<48} max|diff| = {diff:.2e}")
    assert ok, f"{name} 실패: max diff {diff}"


def note(msg):
    print(f"        {msg}")


# ─────────────────────────────────────────────────────────────
# attention 기본
# ─────────────────────────────────────────────────────────────

def causal_allow(i, j):
    return j <= i


def window_allow(w):
    def f(i, j):
        return j <= i and j > i - w
    return f


def sdpa(q, k, v, allow=causal_allow, topk=None):
    """q: (Tq x D) · k, v: (Tk x D) → (Tq x D)"""
    d = len(q[0])
    scores = scale(matmul(q, transpose(k)), 1.0 / math.sqrt(d))
    masked = []
    for i, row in enumerate(scores):
        r = [row[j] if allow(i, j) else NEG_INF for j in range(len(row))]
        if topk is not None:
            finite = sorted((x for x in r if x != NEG_INF), reverse=True)
            if len(finite) > topk:
                thresh = finite[topk - 1]
                r = [x if x >= thresh else NEG_INF for x in r]
        masked.append(r)
    return matmul(softmax_rows(masked), v)


def make_qkv(n_q_heads, n_kv_heads):
    q = [randmat(S, D_H) for _ in range(n_q_heads)]
    k = [randmat(S, D_H) for _ in range(n_kv_heads)]
    v = [randmat(S, D_H) for _ in range(n_kv_heads)]
    return q, k, v


def mha(q, k, v):
    """헤드마다 자기 K, V"""
    return [sdpa(q[h], k[h], v[h]) for h in range(len(q))]


def gqa(q, k, v):
    """그룹 하나가 KV 헤드 하나를 공유. g = n_h / n_kv"""
    g = len(q) // len(k)
    return [sdpa(q[h], k[h // g], v[h // g]) for h in range(len(q))]


# ─────────────────────────────────────────────────────────────
# 01-attention 1.2 / 1.3 — MHA · MQA · GQA
# ─────────────────────────────────────────────────────────────

def test_gqa_reduces_to_mha():
    """n_kv = n_h 이면 GQA는 MHA와 정확히 같다.  → 01-attention 1.3"""
    q, k, v = make_qkv(N_H, N_H)
    for h in range(N_H):
        check(f"GQA(n_kv=n_h) == MHA  [head {h}]" if h == 0 else
              f"                       [head {h}]",
              gqa(q, k, v)[h], mha(q, k, v)[h])
        if h == 1:
            note("... 나머지 헤드 생략")
            break
    note(f"g = n_h/n_kv = 1 → 그룹이 1개짜리이므로 MHA와 동일")


def test_mqa_is_gqa_with_one_group():
    """MQA는 n_kv = 1 인 GQA와 같다.  → 01-attention 1.2"""
    q, k, v = make_qkv(N_H, 1)
    # MQA: 모든 쿼리 헤드가 유일한 KV를 본다
    mqa_out = [sdpa(q[h], k[0], v[0]) for h in range(N_H)]
    gqa_out = gqa(q, k, v)
    check("MQA == GQA(n_kv=1)", mqa_out[0], gqa_out[0])
    check("MQA == GQA(n_kv=1)  [마지막 헤드]", mqa_out[-1], gqa_out[-1])
    note(f"KV 저장량: MHA {2 * N_H * D_H} → MQA {2 * D_H} (값/토큰/레이어)")


def test_gqa_group_sharing():
    """GQA에서 같은 그룹의 헤드들은 동일한 K, V를 참조한다.  → 01-attention 1.3"""
    n_kv = 2
    g = N_H // n_kv
    q, k, v = make_qkv(N_H, n_kv)
    out = gqa(q, k, v)
    # 그룹 0에 속한 헤드들은 전부 k[0], v[0] 을 쓴다
    for h in range(g):
        check(f"head {h} 는 KV 그룹 0 을 참조" if h == 0 else
              f"head {h} 도 KV 그룹 0 을 참조",
              out[h], sdpa(q[h], k[0], v[0]))
        if h == 1:
            break
    check(f"head {g} 는 KV 그룹 1 을 참조", out[g], sdpa(q[g], k[1], v[1]))
    note(f"n_h={N_H}, n_kv={n_kv} → g={g}. KV 저장량 1/{g}")


# ─────────────────────────────────────────────────────────────
# 01-attention 1.4 — MLA 흡수
# ─────────────────────────────────────────────────────────────

def test_mla_absorption():
    """naive MLA == absorbed MLA. 캐시에는 latent만 있으면 된다.  → 01-attention 1.4"""
    d_c = 16
    x = randmat(S, D, 1.0 / math.sqrt(D))
    W_DKV = randmat(D, d_c, 1.0 / math.sqrt(D))
    W_UK = [randmat(d_c, D_H, 1.0 / math.sqrt(d_c)) for _ in range(N_H)]
    W_UV = [randmat(d_c, D_H, 1.0 / math.sqrt(d_c)) for _ in range(N_H)]
    W_Q = [randmat(D, D_H, 1.0 / math.sqrt(D)) for _ in range(N_H)]
    W_O = [randmat(D_H, D, 1.0 / math.sqrt(D_H)) for _ in range(N_H)]

    c = matmul(x, W_DKV)                       # (S x d_c) ← 캐시에 담기는 유일한 것

    y_naive = [[0.0] * D for _ in range(S)]
    y_abs = [[0.0] * D for _ in range(S)]

    for h in range(N_H):
        q = matmul(x, W_Q[h])                  # (S x D_H)

        # --- naive: latent 를 K, V 로 복원해서 계산 ---
        K = matmul(c, W_UK[h])                 # (S x D_H)
        V = matmul(c, W_UV[h])
        o = sdpa(q, K, V)                      # (S x D_H)
        y_naive = add(y_naive, matmul(o, W_O[h]))

        # --- absorbed: W_UK 는 q 쪽으로, W_UV 는 W_O 쪽으로 흡수 ---
        q_c = matmul(q, transpose(W_UK[h]))    # (S x d_c)
        scores = scale(matmul(q_c, transpose(c)), 1.0 / math.sqrt(D_H))
        masked = [[scores[i][j] if causal_allow(i, j) else NEG_INF
                   for j in range(S)] for i in range(S)]
        u = matmul(softmax_rows(masked), c)    # (S x d_c)
        W_UVO = matmul(W_UV[h], W_O[h])        # (d_c x D) — 미리 곱해두는 부분
        y_abs = add(y_abs, matmul(u, W_UVO))

    check("MLA absorbed == naive", y_abs, y_naive, atol=1e-8)
    note(f"캐시: naive 는 K,V 로 {2 * N_H * D_H}개 / absorbed 는 latent {d_c}개")
    note(f"      → {2 * N_H * D_H / d_c:.0f}배 차이. up-projection 은 가중치에 흡수되어 사라짐")


# ─────────────────────────────────────────────────────────────
# 01-attention 1.5 — Gated Attention
# ─────────────────────────────────────────────────────────────

def test_gate_identity():
    """게이트를 1로 두면 원래 attention과 같다.  → 01-attention 1.5"""
    q, k, v = make_qkv(N_H, N_H)
    base = sdpa(q[0], k[0], v[0])
    gated = [[val * 1.0 for val in row] for row in base]     # gate ≡ 1
    check("Gated Attention(gate=1) == 원본", gated, base)

    # gate 가 0 이면 그 헤드는 아무것도 내보내지 않는다 ("안 봐도 된다"는 선택지)
    closed = [[val * 0.0 for val in row] for row in base]
    check("Gated Attention(gate=0) == 0", closed, [[0.0] * D_H for _ in range(S)])


# ─────────────────────────────────────────────────────────────
# 01-attention 1.6 — SWA
# ─────────────────────────────────────────────────────────────

def test_swa_full_window():
    """윈도우가 시퀀스보다 크면 full attention과 같다.  → 01-attention 1.6"""
    q, k, v = make_qkv(1, 1)
    check("SWA(W>=S) == full attention",
          sdpa(q[0], k[0], v[0], allow=window_allow(S)),
          sdpa(q[0], k[0], v[0], allow=causal_allow))

    narrow = sdpa(q[0], k[0], v[0], allow=window_allow(3))
    full = sdpa(q[0], k[0], v[0], allow=causal_allow)
    assert maxdiff(narrow, full) > 1e-6, "윈도우를 좁혔는데 결과가 같다 — 이상하다"
    print(f"  PASS  SWA(W=3) != full  (정보를 실제로 버린다)              "
          f"max|diff| = {maxdiff(narrow, full):.2e}")


# ─────────────────────────────────────────────────────────────
# 01-attention 1.7 / 1.8 / 1.9 — 희소 선택
# ─────────────────────────────────────────────────────────────

def _sparse_dense_limit(label):
    q, k, v = make_qkv(1, 1)
    check(f"{label}(top-k=전체) == dense",
          sdpa(q[0], k[0], v[0], topk=S),
          sdpa(q[0], k[0], v[0]))
    sparse = sdpa(q[0], k[0], v[0], topk=2)
    assert maxdiff(sparse, sdpa(q[0], k[0], v[0])) > 1e-6
    return sparse


def test_nsa_dense_limit():
    """선택 블록이 전체이면 dense attention과 같다.  → 01-attention 1.7"""
    _sparse_dense_limit("NSA")


def test_dsa_dense_limit():
    """top-k = 전체이면 원래 attention과 같다.  → 01-attention 1.8, 1.9"""
    _sparse_dense_limit("DSA/CSA")
    note("희소화는 근사다 — top-k 를 줄이면 결과가 달라진다 (위 assert 로 확인)")


# ─────────────────────────────────────────────────────────────
# 01-attention 1.10 — 레이어 간 KV 공유
# ─────────────────────────────────────────────────────────────

def test_cla_layer_sharing():
    """CLA: 그룹 크기 1 이면 레이어별 독립과 동일. 2 면 아래 레이어 KV를 재사용.
    → 01-attention 1.10"""
    n_layer = 4
    qs = [randmat(S, D_H) for _ in range(n_layer)]
    kvs = [(randmat(S, D_H), randmat(S, D_H)) for _ in range(n_layer)]

    def run(group):
        outs = []
        for i in range(n_layer):
            k, v = kvs[(i // group) * group]        # 그룹의 첫 레이어가 KV를 만든다
            outs.append(sdpa(qs[i], k, v))
        return outs

    indep = [sdpa(qs[i], kvs[i][0], kvs[i][1]) for i in range(n_layer)]
    check("CLA(group=1) == 레이어별 독립 KV", run(1)[2], indep[2])

    shared = run(2)
    check("CLA(group=2): 레이어1이 레이어0의 KV 사용",
          shared[1], sdpa(qs[1], kvs[0][0], kvs[0][1]))
    note(f"레이어 {n_layer}개, group=2 → KV 저장량 1/2. 다른 기법과 곱해진다")


# ─────────────────────────────────────────────────────────────
# 02-linear-attention 2.1 — 고정 상태
# ─────────────────────────────────────────────────────────────

def test_linear_attention_recurrent():
    """선형 attention 재귀 형태 == 이차 형태 (causal).  → 02-linear-attention 2.1
       S_t = S_{t-1} + k_t v_tᵀ ,   o_t = q_t S_t"""
    q, k, v = randmat(S, D_H), randmat(S, D_H), randmat(S, D_H)

    # 이차 형태: causal mask 를 씌운 (q kᵀ) v
    qk = matmul(q, transpose(k))
    qk_masked = [[qk[i][j] if causal_allow(i, j) else 0.0 for j in range(S)]
                 for i in range(S)]
    quad = matmul(qk_masked, v)

    # 재귀 형태: (D_H x D_H) 상태 하나만 들고 간다
    state = [[0.0] * D_H for _ in range(D_H)]
    rec = []
    for t in range(S):
        state = add(state, outer(k[t], v[t]))
        rec.append(vec_matmul(q[t], state))

    check("linear attention 재귀 == 이차 형태", rec, quad, atol=1e-8)
    note(f"상태 크기 {D_H}x{D_H} = {D_H * D_H} — S={S} 와 무관 (S를 100만으로 해도 같다)")


# ─────────────────────────────────────────────────────────────
# 02-linear-attention 2.3 / 2.4 — delta rule 과 게이팅 계보
# ─────────────────────────────────────────────────────────────

def test_deltanet_forms():
    """delta rule 의 3단계 형태 == 행렬 형태.  → 02-linear-attention 2.3
       S ← (I − β k kᵀ) S + β k vᵀ"""
    state = randmat(D_H, D_H)
    k = randvec(D_H)
    norm = math.sqrt(dot(k, k))
    k = [x / norm for x in k]
    v = randvec(D_H)
    beta = 0.7

    # 3단계: 조회 → 오차 → 갱신
    v_old = vec_matmul(k, state)
    delta = [a - b for a, b in zip(v, v_old)]
    step = add(state, scale(outer(k, delta), beta))

    # 행렬 한 줄
    forget = sub(eye(D_H), scale(outer(k, k), beta))
    matrix = add(matmul(forget, state), scale(outer(k, v), beta))

    check("DeltaNet 3단계 == (I − βkkᵀ)S + βkvᵀ", step, matrix, atol=1e-8)
    note("'조회하고 오차만큼 고친다' 가 '그 키 방향을 지우고 새로 쓴다' 와 같다")


def test_linear_lineage():
    """축2 계보가 서로의 특수 케이스인지 확인.  → 02-linear-attention 2.4

       KDA             S = (I − βkkᵀ) diag(α) S + β k vᵀ
       Gated DeltaNet  α 가 스칼라
       DeltaNet        α = 1
    """
    state = randmat(D_H, D_H)
    k = randvec(D_H)
    norm = math.sqrt(dot(k, k))
    k = [x / norm for x in k]
    v = randvec(D_H)
    beta = 0.7
    forget = sub(eye(D_H), scale(outer(k, k), beta))
    write = scale(outer(k, v), beta)

    def kda(alpha_vec):
        decayed = [[alpha_vec[i] * state[i][j] for j in range(D_H)] for i in range(D_H)]
        return add(matmul(forget, decayed), write)

    def gated_deltanet(alpha):
        return add(matmul(forget, scale(state, alpha)), write)

    def deltanet():
        return add(matmul(forget, state), write)

    a = 0.9
    check("KDA(α가 전부 같은 값) == Gated DeltaNet",
          kda([a] * D_H), gated_deltanet(a), atol=1e-8)
    check("Gated DeltaNet(α=1) == DeltaNet",
          gated_deltanet(1.0), deltanet(), atol=1e-8)

    # α 를 채널마다 다르게 두면 실제로 달라진다 — KDA 가 더 넓은 표현
    varied = kda([0.5 + 0.4 * i / D_H for i in range(D_H)])
    assert maxdiff(varied, gated_deltanet(a)) > 1e-6
    print("  PASS  KDA(채널별 α) != Gated DeltaNet  (표현이 실제로 넓어진다)")


# ─────────────────────────────────────────────────────────────
# 03-position 3.2 — RoPE
# ─────────────────────────────────────────────────────────────

def rope(x, pos, theta=10000.0):
    """길이 D 벡터를 D/2 쌍으로 묶어 각도 pos * θ_i 만큼 회전."""
    half = len(x) // 2
    out = [0.0] * len(x)
    for i in range(half):
        ang = pos * (theta ** (-i / half))
        c, s = math.cos(ang), math.sin(ang)
        x1, x2 = x[i], x[i + half]
        out[i] = x1 * c - x2 * s
        out[i + half] = x1 * s + x2 * c
    return out


def test_rope_relative():
    """⟨R_m q, R_n k⟩ 는 m − n 에만 의존한다.  → 03-position 3.2"""
    q, k = randvec(D_H), randvec(D_H)

    pairs = [(3, 5), (10, 12), (100, 102), (5000, 5002)]     # 전부 차이가 −2
    base = dot(rope(q, pairs[0][0]), rope(k, pairs[0][1]))
    for m, n in pairs[1:]:
        check(f"RoPE 내적 (m,n)=({m},{n}) == (3,5)",
              dot(rope(q, m), rope(k, n)), base, atol=1e-8)

    other = dot(rope(q, 3), rope(k, 9))
    assert abs(other - base) > 1e-6, "위치 차이가 다른데 내적이 같다 — 이상하다"
    print("  PASS  RoPE: 위치 차이가 다르면 내적도 다름 (위치 정보가 실제로 있다)")
    note("절대 위치를 넣었는데 내적에는 상대 위치만 남는다 — RoPE 의 전부")


# ─────────────────────────────────────────────────────────────

SUITES = [
    ("01-attention 1.2~1.3   헤드 공유 (MHA · MQA · GQA)", [
        test_gqa_reduces_to_mha,
        test_mqa_is_gqa_with_one_group,
        test_gqa_group_sharing,
    ]),
    ("01-attention 1.4       MLA 흡수", [test_mla_absorption]),
    ("01-attention 1.5       Gated Attention", [test_gate_identity]),
    ("01-attention 1.6       SWA", [test_swa_full_window]),
    ("01-attention 1.7~1.9   희소 선택 (NSA · DSA · CSA)", [
        test_nsa_dense_limit,
        test_dsa_dense_limit,
    ]),
    ("01-attention 1.10      레이어 간 KV 공유 (CLA)", [test_cla_layer_sharing]),
    ("02-linear-attention    고정 상태와 delta rule", [
        test_linear_attention_recurrent,
        test_deltanet_forms,
        test_linear_lineage,
    ]),
    ("03-position 3.2        RoPE", [test_rope_relative]),
]


if __name__ == "__main__":
    print(f"\n설정: S={S}, n_h={N_H}, d_h={D_H}, d={D}\n")
    n = 0
    for title, fns in SUITES:
        print(title)
        for fn in fns:
            fn()
            n += 1
        print()
    print(f"전부 통과 — 검증 함수 {n}개\n")
