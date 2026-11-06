"""Configuration-based attention/state payload analysis; not a runtime benchmark.

Run from any directory: python arch_insight/analysis/cost_model.py
Main scenario: main KV and index keys are BF16; recurrent matrix is FP32.
Cached values are counted once per logical read pass, with head reuse assumed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "reports" / "llm-memory-requirements" / "data" / "raw"
DATA = ROOT / "data"
K = 1024
MIB = 2**20
GIB = 2**30


def config(name: str) -> dict:
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def write_csv(name: str, rows: list[dict]) -> None:
    with (DATA / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parameters() -> dict:
    ds = config("cfg_deepseek-ai_DeepSeek-V3.2.json")
    v4 = config("cfg_deepseek-ai_DeepSeek-V4-Pro.json")
    qw = config("cfg_Qwen_Qwen3.8-Flash-Next.json")["text_config"]
    matrix = qw["linear_num_value_heads"] * qw["linear_key_head_dim"] * qw["linear_value_head_dim"] * 4
    conv = (2 * qw["linear_num_key_heads"] * qw["linear_key_head_dim"]
            + qw["linear_num_value_heads"] * qw["linear_value_head_dim"]) * (qw["linear_conv_kernel_dim"] - 1) * 2
    return {
        "mla_entry_bytes": (ds["kv_lora_rank"] + ds["qk_rope_head_dim"]) * 2,
        "dsa_index_bytes": ds["index_head_dim"] * 2,
        "dsa_k": ds["index_topk"],
        "dsa_limit": ds["max_position_embeddings"],
        "v4_entry_bytes": v4["head_dim"] * 2,
        "v4_index_bytes": v4["index_head_dim"] * 2,
        "v4_index_heads": v4["index_n_heads"],
        "v4_index_dim": v4["index_head_dim"],
        "csa_ratio": 4,
        "hca_ratio": 128,
        "csa_k": v4["index_topk"],
        "window": v4["sliding_window"],
        "v4_limit": v4["max_position_embeddings"],
        "gqa_entry_bytes": 2 * qw["num_key_value_heads"] * qw["head_dim"] * 2,
        "qsa_index_bytes": qw["indexer_head_dim"] * 2,
        "qsa_ratio": qw["indexer_compress_ratio"],
        "qsa_k": qw["indexer_budget"],
        "qsa_limit": qw["max_position_embeddings"],
        "gdn_matrix_bytes": matrix,
        "gdn_conv_bytes": conv,
        "gdn_layers": qw["layer_types"].count("linear_attention"),
        "qsa_layers": qw["layer_types"].count("full_attention"),
        "layers": qw["num_hidden_layers"],
        "dsa_layers": ds["num_hidden_layers"],
        "dsa_index_heads": ds["index_n_heads"],
        "dsa_index_dim": ds["index_head_dim"],
        "qsa_index_heads": qw["indexer_n_heads"],
        "qsa_index_dim": qw["indexer_head_dim"],
    }


def layer_cost(method: str, tokens: int, p: dict, batch: int = 1, dsa_index_bytes: int | None = None) -> dict:
    # Window and compression states are kept separate from the growing cache.
    # For CSA/HCA the pre-compression buffers are deliberately reported as unknown,
    # rather than equated to zero; their formulas depend on the serving kernel.
    r = dict(method=method, context_tokens=tokens, batch=batch,
             stored_main_bytes=0, stored_index_bytes=0, stored_window_bytes=0,
             stored_state_bytes=0, index_read_bytes=0, kv_read_bytes=0,
             window_read_bytes=0, state_read_bytes=0, state_write_bytes=0,
             append_write_bytes=0, compression_buffer_bytes=0,
             index_dot_flops=0, selected_items=0)
    if method in ("Full MLA", "DSA"):
        r["stored_main_bytes"] = tokens * p["mla_entry_bytes"]
        r["append_write_bytes"] = p["mla_entry_bytes"]
        r["kv_read_bytes"] = r["stored_main_bytes"]
        if method == "DSA":
            ib = p["dsa_index_bytes"] if dsa_index_bytes is None else dsa_index_bytes
            r["stored_index_bytes"] = r["index_read_bytes"] = tokens * ib
            r["selected_items"] = min(tokens, p["dsa_k"])
            r["kv_read_bytes"] = r["selected_items"] * p["mla_entry_bytes"]
            r["append_write_bytes"] += ib
            r["index_dot_flops"] = 2 * tokens * p["dsa_index_heads"] * p["dsa_index_dim"]
    elif method in ("CSA", "HCA"):
        ratio = p["csa_ratio"] if method == "CSA" else p["hca_ratio"]
        entries = tokens // ratio
        r["stored_main_bytes"] = entries * p["v4_entry_bytes"]
        r["stored_window_bytes"] = r["window_read_bytes"] = min(tokens, p["window"]) * p["v4_entry_bytes"]
        if method == "CSA":
            r["stored_index_bytes"] = r["index_read_bytes"] = entries * p["v4_index_bytes"]
            r["selected_items"] = min(entries, p["csa_k"])
            r["index_dot_flops"] = 2 * entries * p["v4_index_heads"] * p["v4_index_dim"]
        else:
            r["selected_items"] = entries
        r["kv_read_bytes"] = r["selected_items"] * p["v4_entry_bytes"]
        # Average output bytes per incoming token, not the peak on a block boundary.
        r["append_write_bytes"] = p["v4_entry_bytes"] + (p["v4_entry_bytes"] + (p["v4_index_bytes"] if method == "CSA" else 0)) / ratio
        r["compression_buffer_bytes"] = "not_modelled"
    elif method in ("Full GQA", "QSA"):
        r["stored_main_bytes"] = tokens * p["gqa_entry_bytes"]
        r["append_write_bytes"] = p["gqa_entry_bytes"]
        r["kv_read_bytes"] = r["stored_main_bytes"]
        if method == "QSA":
            blocks, tail = divmod(tokens, p["qsa_ratio"])
            r["stored_index_bytes"] = r["index_read_bytes"] = blocks * p["qsa_index_bytes"]
            r["selected_items"] = min(blocks, p["qsa_k"] // p["qsa_ratio"])
            r["kv_read_bytes"] = (r["selected_items"] * p["qsa_ratio"] + tail) * p["gqa_entry_bytes"]
            r["append_write_bytes"] += p["qsa_index_bytes"] / p["qsa_ratio"]
            r["index_dot_flops"] = 2 * blocks * p["qsa_index_heads"] * p["qsa_index_dim"]
    elif method == "GDN":
        r["stored_state_bytes"] = p["gdn_matrix_bytes"] + p["gdn_conv_bytes"]
        # Normalized tensor I/O model: read and write the retained state once.
        # A fused resident implementation need not transfer all of this to HBM.
        r["state_read_bytes"] = r["state_write_bytes"] = r["stored_state_bytes"]
    else:
        raise ValueError(method)
    for key in list(r):
        if key.endswith("_bytes") and isinstance(r[key], (int, float)):
            r[key] *= batch
    r["index_dot_flops"] *= batch
    r["stored_modelled_bytes"] = sum(r[k] for k in ("stored_main_bytes", "stored_index_bytes", "stored_window_bytes", "stored_state_bytes"))
    r["read_bytes"] = sum(r[k] for k in ("index_read_bytes", "kv_read_bytes", "window_read_bytes", "state_read_bytes"))
    r["write_bytes"] = r["state_write_bytes"] + r["append_write_bytes"]
    r["io_bytes"] = r["read_bytes"] + r["write_bytes"]
    limit = p["v4_limit"] if method in ("CSA", "HCA") else p["qsa_limit"] if method in ("Full GQA", "QSA", "GDN") else p["dsa_limit"]
    r["within_config_limit"] = tokens <= limit
    return r


def generate() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    p = parameters()
    methods = ["Full MLA", "DSA", "CSA", "HCA", "Full GQA", "QSA", "GDN"]
    rows = [layer_cost(m, s, p, b) for b in (1, 8) for s in (32*K, 128*K, 1024*K) for m in methods]
    write_csv("costs.csv", rows)
    curve_s = sorted(set([int(round(2**(x/8))) for x in range(104, 161)] + [32*K, 128*K, p["dsa_limit"], p["qsa_limit"], 1024*K]))
    curves = []
    for s in curve_s:
        for method, full in (("DSA", "Full MLA"), ("QSA", "Full GQA")):
            sparse = layer_cost(method, s, p)
            dense = layer_cost(full, s, p)
            curves.append(dict(method=method, context_tokens=s,
                               read_reduction=dense["read_bytes"] / sparse["read_bytes"],
                               within_config_limit=sparse["within_config_limit"]))
    write_csv("read-curves.csv", curves)
    provenance = []
    for model, filename, url in [
        ("DSA", "cfg_deepseek-ai_DeepSeek-V3.2.json", "https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json"),
        ("CSA/HCA", "cfg_deepseek-ai_DeepSeek-V4-Pro.json", "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json"),
        ("QSA/GDN", "cfg_Qwen_Qwen3.8-Flash-Next.json", "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json"),
    ]:
        provenance.append(dict(method=model, local_snapshot=str((RAW / filename).relative_to(ROOT.parent)).replace("\\", "/"),
                               sha256=hashlib.sha256((RAW / filename).read_bytes()).hexdigest(), source_url=url,
                               scenario="KV/index BF16; recurrent matrix FP32; conv BF16; MTP excluded"))
    write_csv("methods.csv", provenance)
    s = 128*K
    main = {m: layer_cost(m, s, p) for m in methods}
    hybrid = p["qsa_layers"] * main["QSA"]["stored_modelled_bytes"] + p["gdn_layers"] * main["GDN"]["stored_modelled_bytes"]
    all_qsa = p["layers"] * main["QSA"]["stored_modelled_bytes"]
    sensitivity = []
    for ib, label in ((p["dsa_index_bytes"], "BF16 index"), (132, "FP8 index + FP32 scale")):
        row = layer_cost("DSA", s, p, dsa_index_bytes=ib)
        sensitivity.append(dict(scenario=label, index_bytes_per_token=ib,
                                read_mib=row["read_bytes"] / MIB,
                                stored_mib=row["stored_modelled_bytes"] / MIB,
                                read_reduction=main["Full MLA"]["read_bytes"] / row["read_bytes"],
                                asymptotic_read_reduction=p["mla_entry_bytes"] / ib))
    write_csv("precision-sensitivity.csv", sensitivity)
    summary = {"parameters": p, "at_128k": main,
               "hybrid_storage_gib": hybrid/GIB, "hypothetical_48_qsa_gib": all_qsa/GIB,
               "hybrid_storage_reduction": all_qsa/hybrid,
               "state_mib_per_layer": main["GDN"]["stored_modelled_bytes"]/MIB,
               "precision_sensitivity": sensitivity}
    (DATA / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def validate(summary: dict) -> None:
    r = summary["at_128k"]
    assert r["Full MLA"]["read_bytes"] == 144*MIB
    assert r["DSA"]["read_bytes"] == (32+2.25)*MIB
    assert r["DSA"]["stored_modelled_bytes"] == 176*MIB
    assert r["QSA"]["read_bytes"] == 12*MIB
    assert r["QSA"]["stored_modelled_bytes"] == 264*MIB
    assert r["CSA"]["read_bytes"] == 9.125*MIB
    assert r["HCA"]["read_bytes"] == 1.125*MIB
    assert summary["parameters"]["gdn_layers"] == 36
    assert summary["parameters"]["qsa_layers"] == 12
    assert math.isclose(summary["hybrid_storage_gib"], 3.2012786865234375)
    for method in r:
        b8 = layer_cost(method, 128*K, summary["parameters"], 8)
        assert b8["read_bytes"] == 8*r[method]["read_bytes"]
        assert b8["stored_modelled_bytes"] == 8*r[method]["stored_modelled_bytes"]


if __name__ == "__main__":
    result = generate()
    validate(result)
    print("Wrote analytical data; independent byte totals and batch scaling checked.")
    for method, row in result["at_128k"].items():
        print(f"{method:9} read={row['read_bytes']/MIB:8.3f} MiB  stored={row['stored_modelled_bytes']/MIB:8.3f} MiB")
