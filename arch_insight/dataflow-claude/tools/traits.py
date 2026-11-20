"""데이터 성격 — 모델 전체 / attention 층 1개의 HBM 데이터를 종류 · 수명 · 성격으로 나눈다.

약식 뷰어 "데이터 성격 한눈에"(viewer/template-lite.html)와 같은 규칙. 보고서 그림(report/make_traits_report.py)이 쓴다.

읽는 파일: models/<id>/algo.yaml · map-gpu.yaml · model.yaml

판정 규칙 (값에서 판정, 손으로 적지 않음)
- 수명: 커널 사이 나머지 = 스텝 안 · 스텝당 쓰기 0 = 서비스 내내(가중치) · 그 밖 = 요청 동안(cache · state)
        on_chip 중 HBM 이동이 아닌 값 = 커널 안 (HBM 밖 — 따로 셈)
- 성격: 쓰기 0 = read-only · 보관만큼 쓰면 전부 다시 씀 · model.yaml mode: ring = 고정 크기 · 덮어씀(ring buffer)
        · 그 밖 = 끝에 추가 · 커널 사이 = 임시
        read-only · 끝에 추가는 다시 '전부 읽음 / 골라 읽음'으로 나눔:
        요청 1개 · 128K에서 스텝당 읽기 < 보관 × 0.9 이면 골라 읽음 (MoE expert · sparse KV · 임베딩) — 기준 조건의 라벨
- 쓰기는 여러 스텝 평균 (압축 엔트리 · 블록 키는 m토큰마다 1줄 → 스텝당 1/m줄)
- 커널 안 값은 커널별 논리 크기의 합 — 동시에 칩에 올라가는 양(점유량)이 아님
"""
import os

import yaml

import buildlib as bl
from dflow import ev

REF_B, REF_T, SEL_FRAC = 1, 131072, 0.9

LIFE = {   # 수명 — 길이 순
    "svc":  "서비스 내내",
    "req":  "요청 동안",
    "step": "스텝 안",
    "kern": "커널 안",
}
CHAR = {   # 성격
    "ro_all":  "read-only · 전부 읽음",
    "ro_sel":  "read-only · 골라 읽음",
    "app_all": "끝에 추가 · 전부 읽음",
    "app_sel": "끝에 추가 · 골라 읽음",
    "ring":    "고정 크기 · 덮어씀",
    "rmw":     "매 스텝 전부 다시 씀",
    "tmp":     "임시 (커널 사이)",
}


def load(mid):
    here = os.path.join(bl.ROOT, "models", mid)
    algo, mp = bl.load(here)
    with open(os.path.join(here, "model.yaml"), encoding="utf-8") as f:
        mdl = yaml.safe_load(f)
    tens, _ = bl.merged(algo, mp)
    M = {"id": mid, "algo": algo, "map": mp, "model": mdl, "tens": tens}
    M["match"] = _matchers(M)
    return M


def _matchers(M):
    m, named = {}, []
    for it in M["model"]["data"]:
        b = it.get("block")
        if not b or b.get("rest"):
            continue
        m[it["id"]] = (lambda kd: lambda t: M["tens"][t]["kind"] == kd)(b["kind"]) if "kind" in b \
            else (lambda ts: lambda t: t in ts)(set(b["t"]))
        named.append(m[it["id"]])
    for it in M["model"]["data"]:
        b = it.get("block", {})
        if b.get("rest"):
            ex = set(b.get("exclude", []))
            m[it["id"]] = lambda t, ex=ex: t not in ex and not any(f(t) for f in named)
    return m


def dims(M, B, T):
    d = bl.dims_of(M["algo"], M["map"], B=B, T=T)
    d.update({k: v["value"] for k, v in M["model"].get("dims", {}).items()})
    for k, v in M["model"].get("derived", {}).items():
        d[k] = ev(v["expr"], d)
    return d


def _raw(M, it, B, T, scope):
    """항목 하나의 (보관, 스텝당 읽기, 스텝당 쓰기, 색, 임시 여부). scope = model | layer"""
    d = dims(M, B, T)
    b = it.get("block")
    L = 1 if scope == "layer" else ev(it.get("layers", 1), d)
    if not b:
        return ev(it["store"], d), ev(it.get("read", 0), d), ev(it.get("write", 0), d), it.get("color", "gray"), False
    f = M["match"][it["id"]]
    rd = wr = 0
    for k in M["map"]["kernels"]:
        rd += sum(ev(M["tens"][e.get("via", e["t"])]["bytes"], d) for e in k["hbm_read"] if f(e["t"]))
        wr += sum(ev(M["tens"][e.get("via", e["t"])]["bytes"], d) for e in k.get("hbm_write", []) if f(e["t"]))
    own = [t for t in M["algo"]["tensors"] if f(t)]
    tmp = bool(b.get("rest"))
    store = 0 if tmp else L * sum(ev(M["tens"][t]["bytes"], d) for t in own)
    color = it.get("color") or M["algo"]["kinds"][M["tens"][own[0]]["kind"]]["color"]
    return store, rd * L, wr * L, color, tmp


def items(M, B, T, scope="model"):
    out = []
    for it in M["model"]["data"]:
        if scope == "layer" and not it.get("block"):
            continue
        store, rd, wr, color, tmp = _raw(M, it, B, T, scope)
        if tmp:
            life, char = "step", "tmp"
        elif it.get("mode") == "ring":
            life, char = "req", "ring"
        elif wr == 0 or wr < 0.99 * store:
            s0, r0, _, _, _ = _raw(M, it, REF_B, REF_T, scope)
            sel = r0 < SEL_FRAC * s0
            life = "svc" if wr == 0 else "req"
            char = ("ro" if wr == 0 else "app") + ("_sel" if sel else "_all")
        else:
            life, char = "req", "rmw"
        out.append({"id": it["id"], "name": it.get("short", it["label"]), "group": it.get("group", it["label"]),
                    "store": store, "read": rd, "write": wr,
                    "move": rd + wr, "color": color, "life": life, "char": char})
    return out


def chip(M, B, T, scope="model"):
    """칩 안에서만 생겼다 사라지는 값 (on_chip 중 HBM 이동 · 타일 단계가 아닌 것) — 커널별 논리 크기 합, 큰 값 이름.
    타일로 나눠 계산하므로 동시 점유량이 아니다."""
    d = dims(M, B, T)
    per = {}
    for k in M["map"]["kernels"]:
        hb = {x for e in k["hbm_read"] + k.get("hbm_write", []) for x in (e["t"], e.get("via", e["t"]))}
        for e in k.get("on_chip", []):
            e = {"t": e} if isinstance(e, str) else e
            if not e.get("stage") and e["t"] not in hb:
                per[e["t"]] = per.get(e["t"], 0) + ev(M["tens"][e["t"]]["bytes"], d)
    rest = next((x for x in M["model"]["data"] if x.get("block", {}).get("rest")), None)
    L = 1 if scope == "layer" or not rest else ev(rest.get("layers", 1), d)
    top = sorted(per.items(), key=lambda x: -x[1])
    return {"bytes": L * sum(per.values()), "top": [(M["tens"][t]["label"], L * v) for t, v in top]}


def by(rows, key, field):
    """rows를 key(life · char · id)별로 field(store · move · read · write) 합."""
    out = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + r[field]
    return out
