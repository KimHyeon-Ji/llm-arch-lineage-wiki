# snippets — 코드로 확인하기

문서에서 ✅ 로 표시한 주장들을 실제로 돌려서 확인하는 곳이다.

```
python snippets/test_equivalence.py
```

**의존성이 없다.** 표준 라이브러리만 쓴다. 행렬이 작아서(12×12 정도)
순수 파이썬으로 충분하고, 어디서든 바로 돌아간다.

---

## 왜 이걸 두는가

"GQA는 MHA의 일반화다" 같은 문장은 읽으면 그런가 보다 하고 넘어가게 된다.
`n_kv = n_h` 로 두고 돌렸을 때 출력이 **정확히 0.0 만큼 다르다**는 걸 보면
계보가 다르게 남는다.

그래서 이 파일들은 성능 측정용이 아니다. **관계를 확인하는 용도**다.

---

## 무엇을 확인하나

| 검증 | 확인하는 것 | 문서 |
|---|---|---|
| `test_gqa_reduces_to_mha` | `n_kv = n_h` → MHA와 동일 | `01` 1.3 |
| `test_mqa_is_gqa_with_one_group` | MQA == `n_kv=1` 인 GQA | `01` 1.2 |
| `test_gqa_group_sharing` | 그룹 내 헤드가 같은 KV 참조 | `01` 1.3 |
| **`test_mla_absorption`** | **흡수형 == naive, latent만 캐시해도 결과 불변** | `01` 1.4 |
| `test_gate_identity` | gate=1 → 원본, gate=0 → 0 | `01` 1.5 |
| `test_swa_full_window` | `W ≥ S` → full attention과 동일 | `01` 1.6 |
| `test_nsa_dense_limit` | top-k=전체 → dense와 동일 | `01` 1.7 |
| `test_dsa_dense_limit` | 동일 (DSA·CSA) | `01` 1.8~1.9 |
| `test_cla_layer_sharing` | group=1 → 독립, group=2 → 아래 레이어 KV 재사용 | `01` 1.10 |
| `test_linear_attention_recurrent` | 재귀 형태 == 이차 형태, 상태 크기가 `S`와 무관 | `02` 2.1 |
| `test_deltanet_forms` | 3단계 delta rule == `(I − βkkᵀ)S + βkvᵀ` | `02` 2.3 |
| `test_linear_lineage` | KDA ⊃ Gated DeltaNet ⊃ DeltaNet | `02` 2.4 |
| `test_rope_relative` | `⟨R_m q, R_n k⟩` 가 `m − n` 에만 의존 | `04` 4.2 |

---

## 볼 만한 것 두 개

**`test_mla_absorption`** — 이 위키에서 가장 확인할 가치가 있는 부분이다.
`01-attention` 1.4에서 up-projection을 매 토큰마다 명시적으로 복원하지 않아도 된다고
했는데, 코드로 보면 결합법칙이 명확하다.

```
naive:     q · (c·W_UK)ᵀ        → K를 복원해야 한다
absorbed:  (q·W_UKᵀ) · cᵀ       → 괄호만 옮겼는데 복원이 사라진다
```

부동소수점 반올림 범위 안에서 같은 결과가 나오며, **전체 K를 materialize하지 않고
latent만 캐시**할 수 있다.

**`test_linear_lineage`** — 축2 계보가 실제로 포함 관계인지 확인한다.

```
KDA(α가 전부 같은 값)  == Gated DeltaNet    ✓
Gated DeltaNet(α=1)   == DeltaNet          ✓
KDA(채널별 α)          != Gated DeltaNet    ✓  ← 표현이 실제로 넓어진다
```

마지막 줄이 중요하다. 같기만 하면 KDA를 만들 이유가 없었을 테니까.

---

## 한계

- **정확성만 확인한다.** 속도나 메모리는 측정하지 않는다.
- 배치 차원을 뺐다. 읽기 쉽도록 헤드 하나씩 `(T × D)` 2차원으로 계산한다.
- 실제 커널의 최적화(융합, 타일링, gather 처리)는 재현하지 않는다.
  **논문 수식과 실제 커널의 차이**는 각 문서의 ⚠️ 박스를 참조.
