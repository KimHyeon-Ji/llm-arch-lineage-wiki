"""Derive architecture decision conditions from public model configurations.

This is a byte-level analytical model, not a GPU/latency measurement.
Run: python arch_insight/analysis/architecture_tradeoffs.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "reports/llm-memory-requirements/data/raw"
DATA = ROOT / "data"
MIB = 2**20


def read(name):
    path = RAW / name
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("text_config", value)


def parameters():
    ds = read("cfg_deepseek-ai_DeepSeek-V3.2.json")
    qw = read("cfg_Qwen_Qwen3.8-Flash-Next.json")
    v4 = read("cfg_deepseek-ai_DeepSeek-V4-Pro.json")
    matrix = qw["linear_num_value_heads"] * qw["linear_key_head_dim"] * qw["linear_value_head_dim"] * 4
    conv = (2*qw["linear_num_key_heads"]*qw["linear_key_head_dim"]
            + qw["linear_num_value_heads"]*qw["linear_value_head_dim"]) * (qw["linear_conv_kernel_dim"]-1) * 2
    return dict(mla_bytes=2*(ds["kv_lora_rank"]+ds["qk_rope_head_dim"]),
                dsa_index_bytes=ds["index_head_dim"]+4, dsa_k=ds["index_topk"],
                dsa_layers=ds["num_hidden_layers"], dsa_limit=ds["max_position_embeddings"],
                qsa_kv_bytes=2*qw["num_key_value_heads"]*qw["head_dim"]*2,
                qsa_index_bytes=qw["indexer_head_dim"]*2, qsa_block=qw["indexer_compress_ratio"],
                qsa_k=qw["indexer_budget"], qsa_limit=qw["max_position_embeddings"],
                qsa_layers=qw["layer_types"].count("full_attention"),
                gdn_layers=qw["layer_types"].count("linear_attention"),
                gdn_state_bytes=matrix+conv,
                csa_kv_bytes=v4["head_dim"]*2, csa_index_bytes=v4["index_head_dim"]//2,
                csa_k=v4["index_topk"], csa_block=4, window=v4["sliding_window"])


def sparse_reads(method, tokens, p):
    if method == "DSA":
        return tokens*p["dsa_index_bytes"], min(tokens,p["dsa_k"])*p["mla_bytes"]
    if method == "QSA":
        blocks, tail = divmod(tokens,p["qsa_block"])
        chosen = min(blocks,p["qsa_k"]//p["qsa_block"])*p["qsa_block"]+tail
        return blocks*p["qsa_index_bytes"], chosen*p["qsa_kv_bytes"]
    if method == "CSA":
        blocks = tokens//p["csa_block"]
        return (blocks*p["csa_index_bytes"],
                (min(blocks,p["csa_k"])+min(tokens,p["window"]))*p["csa_kv_bytes"])
    raise ValueError(method)


def export_csv(name, rows):
    with (DATA/name).open("w",encoding="utf-8-sig",newline="") as f:
        out = csv.DictWriter(f,fieldnames=list(rows[0]))
        out.writeheader(); out.writerows(rows)


def generate():
    DATA.mkdir(parents=True,exist_ok=True)
    p = parameters()
    thresholds = {
        "DSA":p["dsa_k"]*p["mla_bytes"]/p["dsa_index_bytes"],
        "QSA":p["qsa_k"]*p["qsa_kv_bytes"]/(p["qsa_index_bytes"]/p["qsa_block"]),
        "CSA":(p["csa_k"]+p["window"])*p["csa_kv_bytes"]/(p["csa_index_bytes"]/p["csa_block"]),
    }
    selector = []
    contexts = sorted(set([round(8192*(16**(i/80))) for i in range(81)]
                          + [32768,65536,73728,131072]))
    for method in ("DSA","QSA","CSA"):
        for tokens in contexts:
            index,kv = sparse_reads(method,tokens,p)
            selector.append(dict(method=method,tokens=tokens,index_read_bytes=index,
                                 selected_kv_read_bytes=kv,index_byte_share=index/(index+kv)))
    # All selected main KV is outside the fast tier; no hot-cache hits are assumed.
    # Target token rate and read amplification are design scenarios, not measurements.
    tier = []
    target_rate = 50
    selected_per_step = p["dsa_layers"]*p["dsa_k"]*p["mla_bytes"]
    for batch in (1,8,32):
        for amplification in (1,4):
            tier.append(dict(batch=batch,target_tokens_per_second_per_request=target_rate,
                             read_amplification=amplification,
                             payload_bytes_per_step_per_request=selected_per_step,
                             required_aggregate_bandwidth_GBps=selected_per_step*batch*target_rate*amplification/1e9))
    hybrid = []
    for tokens in (32768,131072,262144):
        index,kv = sparse_reads("QSA",tokens,p)
        qsa_bytes = p["qsa_layers"]*(index+kv)
        state_read = state_write = p["gdn_layers"]*p["gdn_state_bytes"]
        total = qsa_bytes+state_read+state_write
        hybrid.append(dict(tokens=tokens,qsa_read_bytes=qsa_bytes,
                           state_read_bytes=state_read,state_write_bytes=state_write,
                           state_io_share=(state_read+state_write)/total,total_modelled_bytes=total))
    sram = []
    budget_mib = 32
    for batch in (1,8):
        for layers,label in ((1,"current_layer"),(p["gdn_layers"],"all_gdn_layers")):
            sram.append(dict(batch=batch,scope=label,layers=layers,
                             state_MiB=batch*layers*p["gdn_state_bytes"]/MIB,
                             dedicated_state_sram_budget_MiB=budget_mib))
    provenance = []
    for filename in ("cfg_deepseek-ai_DeepSeek-V3.2.json","cfg_deepseek-ai_DeepSeek-V4-Pro.json","cfg_Qwen_Qwen3.8-Flash-Next.json"):
        path = RAW/filename
        provenance.append(dict(path=str(path.relative_to(ROOT.parent)).replace("\\","/"),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    summary = dict(parameters=p,selector_crossover_tokens=thresholds,
                   main_context_tokens=131072,
                   dsa_main_kv_storage_GB_at_128k=p["dsa_layers"]*131072*p["mla_bytes"]/1e9,
                   dsa_index_storage_GB_at_128k=p["dsa_layers"]*131072*p["dsa_index_bytes"]/1e9,
                   tier_scenarios=tier,hybrid_io=hybrid,sram_working_sets=sram,
                   provenance=provenance,
                   max_current_layer_requests_for_32MiB=math.floor(budget_mib*MIB/p["gdn_state_bytes"]))
    export_csv("tradeoff-selector.csv",selector)
    export_csv("tradeoff-tier-bandwidth.csv",tier)
    export_csv("tradeoff-hybrid-io.csv",hybrid)
    export_csv("tradeoff-state-sram.csv",sram)
    (DATA/"architecture-tradeoffs.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return summary


def validate(r):
    assert r["selector_crossover_tokens"]["QSA"] == 65536
    assert r["selector_crossover_tokens"]["CSA"] == 73728
    assert math.isclose(r["selector_crossover_tokens"]["DSA"],17873.454545454544)
    b8 = [x for x in r["tier_scenarios"] if x["batch"]==8]
    assert math.isclose(b8[0]["required_aggregate_bandwidth_GBps"],57.5668224)
    assert math.isclose(b8[1]["required_aggregate_bandwidth_GBps"],230.2672896)
    assert math.isclose(r["hybrid_io"][1]["state_io_share"],0.6046332046332046)
    assert r["max_current_layer_requests_for_32MiB"]==10
    assert r["parameters"]["gdn_layers"]+r["parameters"]["qsa_layers"]==48


if __name__=="__main__":
    result=generate(); validate(result)
    print("Derived selector crossover, tier BW requirement, hybrid I/O and state SRAM working sets.")
