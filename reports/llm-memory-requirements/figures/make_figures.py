"""LLM 메모리 요구량 보고서용 계산 · 그림 생성기. 실행: python make_figures.py [그림 이름 ...]
→ 같은 폴더에 *.svg, *.png / ../data/memory-results.csv
그림 양식은 deepstudy/speculative-decoding/figures/make_figures.py와 같다 (SVG를 직접 그리고 headless 브라우저로 PNG를 찍는다).

수치의 근거 (자료 기준일 2026-10-27, 원본은 ../data/raw/)
- 층 구성 · head 수 · 차원 · state dtype : 각 모델의 공식 HF config.json
  (Llama 3.1 70B와 Gemma 3 27B는 접근 제한 때문에 unsloth 미러의 config를 사용)
- 체크포인트 크기                         : model.safetensors.index.json의 total_size (요약: ../data/raw/checkpoint-sizes.json).
  DeepSeek-V3 / V3.2는 index 값이 BF16 기준으로 잘못 적혀 있어 HF API의 usedStorage를 사용
- V4.1-Flash Engram 크기                  : safetensors header (layers.1/14.engram.*, FP8 + E8M0 scale)
- V4.1-Flash global KV 890 B/token · FP4 · SWA FP8 · Engram host prefetch : arXiv:2609.19969
- V3.2 FP8 KV 656 B/token/layer           : FlashMLA "FP8 with scale" 형식 (512 FP8 + 16 B scale + 128 B RoPE BF16)
- 활성 파라미터                            : 각 모델 카드

모든 용량은 10진 단위다 (1 GB = 10^9 B). 체크포인트와 GPU 표기가 모두 10진이기 때문이다. 그림 A1만 KiB를 쓴다.
"""
import csv
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FONT = "'Noto Sans KR','Malgun Gothic',sans-serif"
MONO = "Consolas,'Courier New',monospace"

INK = "#343a4d"
MUTED = "#7b8196"
PURPLE = "#6b4fc8"
GRID = "#e6e8f1"
AXIS = "#8b90a3"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#e0a030", "#6b4fc8"]   # 차트 계열 — 이 순서로만 쓴다
BOX = ("#f6f5fd", "#d9d2f3")                                       # 설명 상자

GB = 1e9
K = 1024
T32, T128 = 32 * K, 128 * K


# ---------------------------------------------------------------- 모델 데이터
# kv: 대화 길이에 따라 커지는 KV의 토큰당 바이트 (공식 배포 dtype), kv_bf16: 같은 구조를 BF16으로 쓴 값
# win: (층 수, window, 토큰당 바이트) — window 안에서만 보존되는 KV (상한이 있음)
# fixed: 요청당 고정 state 바이트 (recurrent · SSM state + conv state)
# w_gpu: GPU에 상주하는 가중치 바이트 (공식 배포 포맷), w_bf16: 같은 부분을 BF16으로 바꾼 값

def gdn_state(layers, v_heads, k_heads, dk, dv, ssm_bytes=4, conv_k=4, conv_bytes=2):
    """Gated DeltaNet / KDA 계열: 행렬 state (v_heads × dk × dv) + short conv state."""
    mat = layers * v_heads * dk * dv * ssm_bytes
    conv = layers * (2 * k_heads * dk + v_heads * dv) * (conv_k - 1) * conv_bytes
    return mat + conv


def mamba2_state(layers, heads, head_dim, d_state, d_inner, n_groups, ssm_bytes=4, conv_k=4, conv_bytes=2):
    mat = layers * heads * head_dim * d_state * ssm_bytes
    conv = layers * (d_inner + 2 * n_groups * d_state) * (conv_k - 1) * conv_bytes
    return mat + conv


MODELS = [
    dict(key="llama70", name="Llama 3.1 70B", fam="GQA", date="2024-07", sub="dense · GQA",
         arch="dense · GQA 80층 (KV head 8 × 128)",
         total_p="70.6B", active_p="70.6B", fmt="BF16", kv_fmt="BF16",
         w_gpu=141.11e9, w_bf16=141.11e9, w_note="",
         kv=80 * 8 * 128 * 2 * 2, kv_bf16=80 * 8 * 128 * 2 * 2, fixed=0, ctx=128 * K, ctx_note="128K"),
    dict(key="gemma27", name="Gemma 3 27B", fam="GQA", date="2025-03", sub="dense · local 52층 + global 10층",
         arch="dense · local 52층(window 1024) + global 10층",
         total_p="27.4B (vision 포함)", active_p="27.4B", fmt="BF16", kv_fmt="BF16",
         w_gpu=54.86e9, w_bf16=54.86e9, w_note="",
         kv=10 * 16 * 128 * 2 * 2, kv_bf16=10 * 16 * 128 * 2 * 2,
         win=(52, 1024, 16 * 128 * 2 * 2), fixed=0, ctx=128 * K, ctx_note="128K"),
    dict(key="dsv3", name="DeepSeek-V3", fam="MLA", date="2024-12", sub="MoE · MLA",
         arch="MoE · MLA 61층 (latent 512 + RoPE 64)",
         total_p="671B + MTP 14B", active_p="37B", fmt="FP8", kv_fmt="BF16",
         w_gpu=688.73e9, w_bf16=684.49e9 * 2, w_note="",
         kv=61 * 576 * 2, kv_bf16=61 * 576 * 2, fixed=0, ctx=128 * K, ctx_note="128K"),
    dict(key="dsv32", name="DeepSeek-V3.2", fam="MLA", date="2025-12", sub="MoE · MLA + sparse 선택 · FP8 KV",
         arch="MoE · MLA 61층 + sparse 선택(top-2048)용 indexer",
         total_p="671B + MTP 14B", active_p="37B", fmt="FP8", kv_fmt="FP8 (+scale)",
         w_gpu=689.48e9, w_bf16=685.36e9 * 2, w_note="",
         kv=61 * (656 + 132), kv_bf16=61 * (576 + 128) * 2, fixed=0, ctx=128 * K, ctx_note="128K"),
    dict(key="dsv41f", name="DeepSeek-V4.1-Flash", fam="MLA", date="2026-09", sub="MoE · 압축 KV · FP4 KV · Engram은 host",
         arch="MoE · 압축 global KV + 층 간 공유 + SWA 128, Engram",
         total_p="552B 본체 + Engram 196B", active_p="8B prefill / 16B decode", fmt="FP8 + FP4 expert",
         kv_fmt="FP4 (+E4M3 scale)",
         w_gpu=510.29e9 - 202.79e9, w_bf16=(763.21e9 - 196.6e9) * 2,
         w_note="Engram 203 GB는 host 메모리에서 prefetch",
         kv=890, kv_bf16=3200, win=(40, 128, 576), fixed=0, ctx=1024 * K, ctx_note="1M"),
    dict(key="q38_27", name="Qwen3.8-27B", fam="Hybrid", date="2026-08", sub="dense FFN · DeltaNet 48층 + attention 16층",
         arch="dense FFN · Gated DeltaNet 48층 + full attention 16층",
         total_p="27.8B (vision · MTP 포함)", active_p="27.8B", fmt="FP8 (BF16도 공개)", kv_fmt="BF16",
         w_gpu=30.9e9, w_bf16=55.56e9, w_note="",
         kv=16 * 4 * 256 * 2 * 2, kv_bf16=16 * 4 * 256 * 2 * 2,
         fixed=gdn_state(48, 48, 16, 128, 128), ctx=256 * K, ctx_note="262K (YaRN 1M)"),
    dict(key="q38_24t", name="Qwen3.8-2.4T-A95B", fam="Hybrid", date="2026-08", sub="MoE · DeltaNet 69층 + attention 23층",
         arch="MoE · Gated DeltaNet 69층 + full attention 23층",
         total_p="2.45T", active_p="95B", fmt="FP8 (BF16도 공개)", kv_fmt="BF16",
         w_gpu=2496.07e9, w_bf16=4892.37e9, w_note="",
         kv=23 * 4 * 256 * 2 * 2, kv_bf16=23 * 4 * 256 * 2 * 2,
         fixed=gdn_state(69, 128, 16, 128, 128), ctx=256 * K, ctx_note="262K (1M 확장)"),
    dict(key="k3", name="Kimi K3", fam="Hybrid", date="2026-06", sub="MoE · KDA 69층 + MLA 24층 · MXFP4",
         arch="MoE · KDA 69층 + Gated MLA 24층",
         total_p="2.8T", active_p="104B", fmt="MXFP4 (QAT)", kv_fmt="BF16",
         w_gpu=1560.86e9, w_bf16=2779.93e9 * 2, w_note="",
         kv=24 * 576 * 2, kv_bf16=24 * 576 * 2,
         fixed=gdn_state(69, 96, 96, 128, 128), ctx=1024 * K, ctx_note="1M"),
    dict(key="nemo3u", name="Nemotron 3 Ultra", fam="Hybrid", date="2026-06", sub="MoE · Mamba 48층 + attention 12층 · NVFP4",
         arch="MoE · Mamba-2 48층 + attention 12층",
         total_p="550B", active_p="55B", fmt="NVFP4 (BF16도 공개)", kv_fmt="BF16",
         w_gpu=352.28e9, w_bf16=1121.05e9, w_note="",
         kv=12 * 2 * 128 * 2 * 2, kv_bf16=12 * 2 * 128 * 2 * 2,
         fixed=mamba2_state(48, 256, 64, 128, 16384, 8), ctx=256 * K, ctx_note="262K (공식 최대 1M)"),
]

# 부록용 — 본문 그림에는 넣지 않는다
APPENDIX = [
    dict(key="q3next", name="Qwen3-Next 80B-A3B", fam="Hybrid", date="2025-09", sub="MoE · DeltaNet 36층 + attention 12층",
         arch="MoE · Gated DeltaNet 36층 + full attention 12층", total_p="81.3B", active_p="3B",
         fmt="FP8 (BF16도 공개)", kv_fmt="BF16", w_gpu=82.0e9, w_bf16=162.65e9, w_note="",
         kv=12 * 2 * 256 * 2 * 2, kv_bf16=12 * 2 * 256 * 2 * 2,
         fixed=gdn_state(36, 32, 16, 128, 128), ctx=256 * K, ctx_note="262K"),
    dict(key="klinear", name="Kimi Linear 48B-A3B", fam="Hybrid", date="2025-10", sub="MoE · KDA 20층 + MLA 7층",
         arch="MoE · KDA 20층 + MLA 7층", total_p="49.1B", active_p="3B",
         fmt="BF16", kv_fmt="BF16", w_gpu=98.25e9, w_bf16=98.25e9, w_note="",
         kv=7 * 576 * 2, kv_bf16=7 * 576 * 2,
         fixed=gdn_state(20, 32, 32, 128, 128), ctx=1024 * K, ctx_note="1M"),
    dict(key="q38fn", name="Qwen3.8-Flash-Next", fam="Hybrid", date="2026-08", sub="MoE · DeltaNet 36층 + sparse attention 12층",
         arch="MoE · Gated DeltaNet 36층 + sparse attention 12층, n-gram 51B",
         total_p="125B + n-gram 51B + MTP 4B", active_p="6B", fmt="FP8 (BF16도 공개)", kv_fmt="BF16",
         w_gpu=185.5e9, w_bf16=360.0e9, w_note="n-gram 표 포함",
         kv=12 * 2 * 256 * 2 * 2 + 12 * 128 * 2 // 4, kv_bf16=12 * 2 * 256 * 2 * 2 + 12 * 128 * 2 // 4,
         fixed=gdn_state(36, 48, 16, 128, 128), ctx=256 * K, ctx_note="262K (YaRN 1M)"),
]

NODES = [("H100 SXM 80GB × 8", 640e9), ("B200 180GB × 8", 1440e9)]

def state_bytes(m, t):
    """요청 하나가 길이 t 토큰을 보유할 때의 대화 상태 바이트."""
    return kv_part(m, t) + m["fixed"]


def kv_part(m, t):
    s = m["kv"] * t
    if m.get("win"):
        layers, w, per = m["win"]
        s += layers * min(t, w) * per
    return s


# ---------------------------------------------------------------- 시리즈별 계보 (6절) — 같은 기준으로 비교한다
# 모든 KV · indexer key는 BF16, recurrent state는 config의 state dtype(없으면 FP32로 가정)으로 센다.
# MTP 층의 KV는 넣지 않는다. 공개일은 공식 HF 저장소 생성일이다 (자료 기준일 2026-11-02).
T1M = 1024 * K
CKPT = 4096                                   # prefix 재사용 지점 — 이 토큰 수마다 state 사본 하나
CAT_LABEL = {"full": "Full attention만", "sparse": "Sparse attention (+ full)",
             "swa": "Sliding window + full", "state": "State + full/sparse"}
CAT_COLOR = {"full": SERIES[0], "sparse": SERIES[1], "swa": SERIES[2], "state": SERIES[4]}
CAT_LIGHT = {"full": "#b9d3f2", "sparse": "#f6c4ae", "swa": "#a9e0cb", "state": "#d3c8f1"}


def gqa(layers, kv_heads, k_dim, v_dim=None):
    return layers * kv_heads * (k_dim + (v_dim or k_dim)) * 2


def mla(layers, rope=64):
    return layers * (512 + rope) * 2


def idx(layers, ratio=1):
    return layers * 128 * 2 // ratio


def _cfg(name):
    import json
    with open(os.path.join(DATA, "raw", f"cfg_{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def v4(name):
    """DeepSeek-V4 계열: CSA(1/4 압축 + indexer) · HCA(1/128 압축) · SWA 128 (모든 층). entry 512차원, K = V."""
    c = _cfg(name)
    r = c["compress_ratios"][:c["num_hidden_layers"]]          # 끝의 항목은 MTP 층
    csa, hca, swa = r.count(4), r.count(128), r.count(0)
    kv = csa * (1024 // 4 + 256 // 4) + hca * 1024 // 128
    return dict(kv=kv, win=[(len(r), 128, 1024)], csa=csa, hca=hca, swa=swa)


_V4F, _V4P = v4("deepseek-ai_DeepSeek-V4-Flash"), v4("deepseek-ai_DeepSeek-V4-Pro")


def L(series, co, name, short, date, cat, arch, kv, win=(), fixed=0, ctx="", note=""):
    return dict(series=series, co=co, name=name, short=short, date=date, cat=cat, arch=arch, kv=kv, win=list(win),
                fixed=fixed, ctx=ctx, note=note)


LINEAGE = [
    # GLM — Z.ai
    L("GLM", "Z.ai", "GLM-4.5", "4.5", "2025-07-20", "full", "GQA 92층 (KV 8 × 128)", gqa(92, 8, 128), ctx="128K"),
    L("GLM", "Z.ai", "GLM-4.6", "4.6", "2025-09-29", "full", "GQA 92층 (KV 8 × 128)", gqa(92, 8, 128), ctx="200K"),
    L("GLM", "Z.ai", "GLM-4.7", "4.7", "2025-12-22", "full", "GQA 92층 (KV 8 × 128)", gqa(92, 8, 128), ctx="200K"),
    L("GLM", "Z.ai", "GLM-4.7-Flash", "4.7-Flash", "2026-01-19", "full", "MLA 47층", mla(47), ctx="200K"),
    L("GLM", "Z.ai", "GLM-5", "5", "2026-02-11", "sparse", "MLA + DSA 78층 (indexer 78층)", mla(78) + idx(78), ctx="200K"),
    L("GLM", "Z.ai", "GLM-5.1", "5.1", "2026-04-03", "sparse", "MLA + DSA 78층 (indexer 78층)", mla(78) + idx(78), ctx="200K"),
    L("GLM", "Z.ai", "GLM-5.2", "5.2", "2026-06-16", "sparse", "MLA + DSA 78층 (indexer 21층, 나머지 57층은 선택 공유)",
      mla(78) + idx(21), ctx="1M"),
    L("GLM", "Z.ai", "GLM-5.3", "5.3", "2026-08-25", "sparse", "MLA + DSA 78층 (indexer 21층, 나머지 57층은 선택 공유)",
      mla(78) + idx(21), ctx="1M"),
    L("GLM", "Z.ai", "GLM-5.3-Flash", "5.3-Flash", "2026-08-25", "state", "KDA 34층 + MLA·DSA 11층 (RoPE 없음, indexer 1/4 압축)",
      mla(11, 0) + idx(11, 4), fixed=gdn_state(34, 64, 64, 128, 128), ctx="1M"),
    # DeepSeek
    L("DeepSeek", "DeepSeek", "DeepSeek-V3", "V3", "2024-12-25", "full", "MLA 61층", mla(61), ctx="128K"),
    L("DeepSeek", "DeepSeek", "DeepSeek-V3.1", "V3.1", "2025-08-21", "full", "MLA 61층", mla(61), ctx="128K"),
    L("DeepSeek", "DeepSeek", "DeepSeek-V3.2-Exp", "V3.2-Exp", "2025-09-29", "sparse", "MLA + DSA 61층 (indexer 61층)",
      mla(61) + idx(61), ctx="128K"),
    L("DeepSeek", "DeepSeek", "DeepSeek-V3.2", "V3.2", "2025-12-01", "sparse", "MLA + DSA 61층 (indexer 61층)",
      mla(61) + idx(61), ctx="128K"),
    L("DeepSeek", "DeepSeek", "DeepSeek-V4-Flash", "V4-Flash", "2026-04-22", "sparse",
      f"CSA {_V4F['csa']}층 + HCA {_V4F['hca']}층 + SWA만 {_V4F['swa']}층 (모든 층 SWA 128)", _V4F["kv"], _V4F["win"], ctx="1M"),
    L("DeepSeek", "DeepSeek", "DeepSeek-V4-Pro", "V4-Pro", "2026-04-22", "sparse",
      f"CSA {_V4P['csa']}층 + HCA {_V4P['hca']}층 + SWA만 {_V4P['swa']}층 (모든 층 SWA 128)", _V4P["kv"], _V4P["win"], ctx="1M"),
    L("DeepSeek", "DeepSeek", "DeepSeek-V4.1-Flash", "V4.1-Flash", "2026-09-10", "sparse",
      "압축 global KV + 층 간 공유 + SWA 128 (40층)", 3200, [(40, 128, 1024)], ctx="1M",
      note="global KV는 기술 보고서의 BF16 환산값"),
    # Hy — Tencent
    L("Hy", "Tencent", "Hunyuan-A13B", "A13B", "2025-06-25", "full", "GQA 32층 (KV 8 × 128)", gqa(32, 8, 128), ctx="32K"),
    L("Hy", "Tencent", "Hy3", "Hy3", "2026-07-02", "full", "GQA 80층 (KV 8 × 128)", gqa(80, 8, 128), ctx="256K"),
    L("Hy", "Tencent", "Hy4-preview", "Hy4", "2026-08-27", "sparse", "MLA + DSA 78층 (indexer 21층, 나머지 57층은 선택 공유)",
      mla(78) + idx(21), ctx="1M"),
    # MiniMax
    L("MiniMax", "MiniMax", "MiniMax-Text-01", "Text-01", "2025-01-12", "state", "lightning attention 70층 + GQA 10층 (KV 8 × 128)",
      gqa(10, 8, 128), fixed=70 * 64 * 128 * 128 * 4, ctx="1M", note="lightning state는 FP32로 가정"),
    L("MiniMax", "MiniMax", "MiniMax-M1", "M1", "2025-06-13", "state", "lightning attention 70층 + GQA 10층 (KV 8 × 128)",
      gqa(10, 8, 128), fixed=70 * 64 * 128 * 128 * 4, ctx="1M", note="lightning state는 FP32로 가정"),
    L("MiniMax", "MiniMax", "MiniMax-M2", "M2", "2025-10-22", "full", "GQA 62층 (KV 8 × 128)", gqa(62, 8, 128), ctx="196K"),
    L("MiniMax", "MiniMax", "MiniMax-M2.1", "M2.1", "2025-12-20", "full", "GQA 62층 (KV 8 × 128)", gqa(62, 8, 128), ctx="196K"),
    L("MiniMax", "MiniMax", "MiniMax-M2.5", "M2.5", "2026-02-12", "full", "GQA 62층 (KV 8 × 128)", gqa(62, 8, 128), ctx="196K"),
    L("MiniMax", "MiniMax", "MiniMax-M2.7", "M2.7", "2026-04-09", "full", "GQA 62층 (KV 8 × 128)", gqa(62, 8, 128), ctx="200K"),
    L("MiniMax", "MiniMax", "MiniMax-M3", "M3", "2026-06-02", "sparse", "GQA 60층 (KV 4 × 128) · MSA 57층 + full 3층",
      gqa(60, 4, 128) + idx(57), ctx="1M"),
    L("MiniMax", "MiniMax", "MiniMax-M3.1-Flash", "M3.1-Flash", "2026-10-17", "sparse", "GQA 60층 (KV 4 × 128) · MSA 60층",
      gqa(60, 4, 128) + idx(60), ctx="1M"),
    # Qwen — Alibaba
    L("Qwen", "Alibaba", "Qwen3-235B-A22B", "3-235B", "2025-04-27", "full", "GQA 94층 (KV 4 × 128)", gqa(94, 4, 128), ctx="40K"),
    L("Qwen", "Alibaba", "Qwen3-Next 80B-A3B", "3-Next", "2025-09-09", "state", "Gated DeltaNet 36층 + GQA 12층 (KV 2 × 256)",
      gqa(12, 2, 256), fixed=gdn_state(36, 32, 16, 128, 128), ctx="262K"),
    L("Qwen", "Alibaba", "Qwen3.5-397B-A17B", "3.5-397B", "2026-02-16", "state", "Gated DeltaNet 45층 + GQA 15층 (KV 2 × 256)",
      gqa(15, 2, 256), fixed=gdn_state(45, 64, 16, 128, 128), ctx="262K"),
    L("Qwen", "Alibaba", "Qwen3.5-27B", "3.5-27B", "2026-02-24", "state", "Gated DeltaNet 48층 + GQA 16층 (KV 4 × 256)",
      gqa(16, 4, 256), fixed=gdn_state(48, 48, 16, 128, 128), ctx="262K"),
    L("Qwen", "Alibaba", "Qwen3.6-35B-A3B", "3.6-35B", "2026-04-15", "state", "Gated DeltaNet 30층 + GQA 10층 (KV 2 × 256)",
      gqa(10, 2, 256), fixed=gdn_state(30, 32, 16, 128, 128), ctx="262K"),
    L("Qwen", "Alibaba", "Qwen3.6-27B", "3.6-27B", "2026-04-21", "state", "Gated DeltaNet 48층 + GQA 16층 (KV 4 × 256)",
      gqa(16, 4, 256), fixed=gdn_state(48, 48, 16, 128, 128), ctx="262K"),
    L("Qwen", "Alibaba", "Qwen3.8-27B", "3.8-27B", "2026-08-05", "state", "Gated DeltaNet 48층 + GQA 16층 (KV 4 × 256)",
      gqa(16, 4, 256), fixed=gdn_state(48, 48, 16, 128, 128), ctx="262K"),
    L("Qwen", "Alibaba", "Qwen3.8-2.4T-A95B", "3.8-2.4T", "2026-08-08", "state", "Gated DeltaNet 69층 + GQA 23층 (KV 4 × 256)",
      gqa(23, 4, 256), fixed=gdn_state(69, 128, 16, 128, 128), ctx="262K"),
    L("Qwen", "Alibaba", "Qwen3.8-Flash-Next", "3.8-Flash-Next", "2026-08-24", "state",
      "Gated DeltaNet 36층 + sparse GQA 12층 (indexer 1/4 압축)", gqa(12, 2, 256) + idx(12, 4),
      fixed=gdn_state(36, 48, 16, 128, 128), ctx="262K"),
    # Nemotron — NVIDIA
    L("Nemotron", "NVIDIA", "Nemotron-H-56B", "H-56B", "2025-04-08", "state", "Mamba-2 54층 + GQA 10층 (KV 8 × 128)",
      gqa(10, 8, 128), fixed=mamba2_state(54, 256, 64, 256, 16384, 8), ctx="8K", note="SSM state는 FP32로 가정"),
    L("Nemotron", "NVIDIA", "Nemotron Nano 9B v2", "Nano-9B-v2", "2025-08-12", "state", "Mamba-2 27층 + GQA 4층 (KV 8 × 128)",
      gqa(4, 8, 128), fixed=mamba2_state(27, 128, 80, 128, 10240, 8), ctx="128K", note="SSM state는 FP32로 가정"),
    L("Nemotron", "NVIDIA", "Nemotron 3 Nano 30B-A3B", "3-Nano", "2025-12-04", "state", "Mamba-2 23층 + GQA 6층 (KV 2 × 128)",
      gqa(6, 2, 128), fixed=mamba2_state(23, 64, 64, 128, 4096, 8), ctx="262K"),
    L("Nemotron", "NVIDIA", "Nemotron 3 Super 120B-A12B", "3-Super", "2026-03-10", "state", "Mamba-2 40층 + GQA 8층 (KV 2 × 128)",
      gqa(8, 2, 128), fixed=mamba2_state(40, 128, 64, 128, 8192, 8), ctx="262K"),
    L("Nemotron", "NVIDIA", "Nemotron 3 Ultra 550B-A55B", "3-Ultra", "2026-06-03", "state", "Mamba-2 48층 + GQA 12층 (KV 2 × 128)",
      gqa(12, 2, 128), fixed=mamba2_state(48, 256, 64, 128, 16384, 8), ctx="262K"),
    L("Nemotron", "NVIDIA", "Nemotron 3.5 Lightning 30B-A3B", "3.5-Lightning", "2026-08-01", "state",
      "Mamba-2 23층 + GQA 6층 (KV 2 × 128)", gqa(6, 2, 128), fixed=mamba2_state(23, 64, 64, 128, 4096, 8), ctx="262K"),
    L("Nemotron", "NVIDIA", "Nemotron 3.5 Bolt 74B-A9B", "3.5-Bolt", "2026-10-09", "state",
      "Mamba-2 40층 + GQA 8층 (KV 2 × 128)", gqa(8, 2, 128), fixed=mamba2_state(40, 128, 64, 128, 8192, 8), ctx="262K"),
    # Kimi — Moonshot AI
    L("Kimi", "Moonshot AI", "Kimi K2", "K2", "2025-07-11", "full", "MLA 61층", mla(61), ctx="128K"),
    L("Kimi", "Moonshot AI", "Kimi Linear 48B-A3B", "Linear", "2025-10-30", "state", "KDA 20층 + MLA 7층",
      mla(7), fixed=gdn_state(20, 32, 32, 128, 128), ctx="1M", note="KDA state는 FP32로 가정"),
    L("Kimi", "Moonshot AI", "Kimi K2.5", "K2.5", "2026-01-01", "full", "MLA 61층", mla(61), ctx="256K"),
    L("Kimi", "Moonshot AI", "Kimi K2.6", "K2.6", "2026-04-14", "full", "MLA 61층", mla(61), ctx="256K"),
    L("Kimi", "Moonshot AI", "Kimi K3", "K3", "2026-06-13", "state", "KDA 69층 + MLA 24층",
      mla(24), fixed=gdn_state(69, 96, 96, 128, 128), ctx="1M", note="KDA state는 FP32로 가정"),
    # Muse — Meta
    L("Muse", "Meta", "Muse Glimmer 30B", "Glimmer-30B", "2026-08-09", "swa", "SWA 39층 (window 2048) + full 13층 (KV 2 × 128)",
      gqa(13, 2, 128), [(39, 2048, gqa(1, 2, 128))], ctx="128K"),
    L("Muse", "Meta", "Muse Spark Open 816B-A42B", "Spark-Open", "2026-09-15", "swa",
      "SWA 46층 (window 2048) + full 16층 (KV 16 × 64)", gqa(16, 16, 64), [(46, 2048, gqa(1, 16, 64))], ctx="1M"),
    # MiMo — Xiaomi
    L("MiMo", "Xiaomi", "MiMo-V2-Flash", "V2-Flash", "2025-12-16", "swa", "SWA 39층 (window 128, KV 8) + full 9층 (KV 4)",
      gqa(9, 4, 192, 128), [(39, 128, gqa(1, 8, 192, 128))], ctx="256K"),
    L("MiMo", "Xiaomi", "MiMo-V2.5", "V2.5", "2026-04-27", "swa", "SWA 39층 (window 128, KV 8) + full 9층 (KV 4)",
      gqa(9, 4, 192, 128), [(39, 128, gqa(1, 8, 192, 128))], ctx="1M"),
    L("MiMo", "Xiaomi", "MiMo-V2.5-Pro", "V2.5-Pro", "2026-04-27", "swa", "SWA 60층 (window 128, KV 8) + full 10층 (KV 8)",
      gqa(10, 8, 192, 128), [(60, 128, gqa(1, 8, 192, 128))], ctx="1M"),
    L("MiMo", "Xiaomi", "MiMo-V2.6-Flash", "V2.6-Flash", "2026-09-21", "swa", "SWA 39층 (window 128, KV 8) + full 9층 (KV 4)",
      gqa(9, 4, 192, 128), [(39, 128, gqa(1, 8, 192, 128))], ctx="1M"),
    L("MiMo", "Xiaomi", "MiMo-V2.6-Pro", "V2.6-Pro", "2026-09-21", "swa", "SWA 60층 (window 128, KV 8) + full 10층 (KV 8)",
      gqa(10, 8, 192, 128), [(60, 128, gqa(1, 8, 192, 128))], ctx="1M"),
]
LINEAGE.sort(key=lambda m: (m["date"], m["name"]))
SERIES_ORDER = ["GLM", "DeepSeek", "Hy", "MiniMax", "Qwen", "Nemotron", "Kimi", "Muse", "MiMo"]

# 가중치 — (공식 체크포인트 중 가장 작은 것 GB, 그 포맷, 같은 가중치를 BF16으로 GB). 원본: ../data/raw/checkpoint-sizes-lineage.json
# 크기 = safetensors dtype별 원소 수 × 바이트. FP4 · INT4처럼 묶어 저장하는 dtype이 있으면 index의 total_size(없으면 usedStorage).
# BF16 = BF16 저장소의 크기, 없으면 파라미터 수 × 2. V4.1-Flash는 host에 두는 Engram 203 GB를 뺀 GPU 상주분이다.
WEIGHTS = {
    "GLM-4.5": (360.8, "FP8", 716.7), "GLM-4.6": (360.8, "FP8", 713.6), "GLM-4.7": (361.6, "FP8", 716.7),
    "GLM-4.7-Flash": (62.4, "BF16", 62.4), "GLM-5": (756.0, "FP8", 1507.7), "GLM-5.1": (756.0, "FP8", 1507.7),
    "GLM-5.2": (755.4, "FP8", 1506.7), "GLM-5.3": (755.4, "FP8", 1506.6), "GLM-5.3-Flash": (328.3, "FP8", 642.6),
    "DeepSeek-V3": (688.4, "FP8", 1369.0), "DeepSeek-V3.1": (688.6, "FP8", 1369.0),
    "DeepSeek-V3.2-Exp": (689.5, "FP8", 1370.8), "DeepSeek-V3.2": (689.3, "FP8", 1370.8),
    "DeepSeek-V4-Flash": (159.6, "FP4 expert + FP8", 581.8), "DeepSeek-V4-Pro": (864.7, "FP4 expert + FP8", 3197.6),
    "DeepSeek-V4.1-Flash": (307.5, "FP4 expert + FP8", 1133.2),
    "Hunyuan-A13B": (80.9, "FP8", 160.8), "Hy3": (299.9, "FP8", 597.6), "Hy4-preview": (813.8, "FP8", 1560.0),
    "MiniMax-Text-01": (914.7, "BF16", 914.7), "MiniMax-M1": (912.2, "BF16", 912.2),
    "MiniMax-M2": (230.1, "FP8", 457.4), "MiniMax-M2.1": (230.1, "FP8", 457.4), "MiniMax-M2.5": (230.1, "FP8", 457.4),
    "MiniMax-M2.7": (230.1, "FP8", 457.4), "MiniMax-M3": (854.2, "BF16", 854.2),
    "MiniMax-M3.1-Flash": (250.1, "8-bit", 441.0),
    "Qwen3-235B-A22B": (239.0, "FP8", 470.2), "Qwen3-Next 80B-A3B": (82.0, "FP8", 162.6),
    "Qwen3.5-397B-A17B": (406.1, "FP8", 806.8), "Qwen3.5-27B": (30.9, "FP8", 55.6), "Qwen3.6-35B-A3B": (37.5, "FP8", 71.9),
    "Qwen3.6-27B": (30.9, "FP8", 55.6), "Qwen3.8-27B": (30.9, "FP8", 55.6), "Qwen3.8-2.4T-A95B": (2496.1, "FP8", 4892.4),
    "Qwen3.8-Flash-Next": (185.5, "FP8", 360.0),
    "Nemotron-H-56B": (112.6, "BF16", 112.6), "Nemotron Nano 9B v2": (7.8, "NVFP4", 17.8),
    "Nemotron 3 Nano 30B-A3B": (19.3, "NVFP4", 63.2), "Nemotron 3 Super 120B-A12B": (80.3, "NVFP4", 247.2),
    "Nemotron 3 Ultra 550B-A55B": (352.3, "NVFP4", 1121.0), "Nemotron 3.5 Lightning 30B-A3B": (21.6, "NVFP4", 65.8),
    "Nemotron 3.5 Bolt 74B-A9B": (149.8, "BF16", 149.8),
    "Kimi K2": (1028.9, "FP8", 2052.8), "Kimi Linear 48B-A3B": (98.2, "BF16", 98.2), "Kimi K2.5": (595.1, "INT4", 2053.8),
    "Kimi K2.6": (595.1, "INT4", 2053.8), "Kimi K3": (1560.9, "MXFP4", 5559.9),
    "Muse Glimmer 30B": (59.6, "BF16", 59.6), "Muse Spark Open 816B-A42B": (533.0, "NVFP4", 1644.3),
    "MiMo-V2-Flash": (313.0, "FP8", 619.6), "MiMo-V2.5": (315.0, "FP8", 621.6), "MiMo-V2.5-Pro": (1033.1, "FP8", 2046.4),
    "MiMo-V2.6-Flash": (172.9, "FP4 혼합", 621.6), "MiMo-V2.6-Pro": (566.0, "FP4 혼합", 2048.4),
}
for _m in LINEAGE:
    _w, _m["wfmt"], _b = WEIGHTS[_m["name"]]
    _m["w"], _m["wbf16"] = _w * GB, _b * GB


def kv1m(m, t=T1M):
    """1M 토큰을 보유한 요청 하나 — state 사본 1개. (KV, window, state) 바이트."""
    win = sum(layers * min(t, w) * per for layers, w, per in m["win"])
    return m["kv"] * t, win, m["fixed"]


def kv1m_ckpt(m, t=T1M):
    """같은 요청을 4096 토큰마다 재개할 수 있게 저장 — state 사본 t/4096개, window KV도 지점마다 하나."""
    n = t // CKPT
    win = sum(layers * min(t, n * w) * per for layers, w, per in m["win"])
    return m["kv"] * t, win, n * m["fixed"]


# ---------------------------------------------------------------- SVG 도우미 (deepstudy와 같은 양식)
def esc(s):
    s = str(s)
    lead = len(s) - len(s.lstrip(" "))
    s = " " * lead + s[lead:]   # SVG는 앞쪽 공백을 지운다 — 들여쓰기를 살린다
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Fig:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.o = name, w, h, []

    def raw(self, s):
        self.o.append(s)

    def rect(self, x, y, w, h, fill, stroke="none", rx=6, sw=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=18, anchor="middle", weight=400, color=INK, mono=False, halo=False):
        f = MONO if mono else FONT
        h = ' stroke="#ffffff" stroke-width="8" stroke-linejoin="round" paint-order="stroke"' if halo else ""
        self.raw(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{f}" font-size="{size}" text-anchor="{anchor}" '
                 f'font-weight="{weight}" fill="{color}"{h}>{esc(s)}</text>')

    def lines(self, x, y, rows, size=16, lh=28, anchor="start", color=INK, weight=400):
        for i, r in enumerate(rows):
            self.text(x, y + i * lh, r, size, anchor, weight, color)

    def title(self, x, y, s, size=24, anchor="start"):
        self.text(x, y, s, size, anchor, 700, PURPLE)

    def line(self, x1, y1, x2, y2, color=AXIS, sw=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                 f'stroke-width="{sw}" stroke-linecap="round"{d}/>')

    def poly(self, pts, color, sw=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.raw(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" '
                 f'stroke-linejoin="round" stroke-linecap="round"{d}/>')

    def dot(self, x, y, r, fill):
        self.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')

    def ring(self, x, y, r, stroke, fill="#ffffff", sw=2.2):
        self.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def svg(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}"><rect width="100%" height="100%" fill="#ffffff"/>' + "".join(self.o) + "</svg>")


def logx(x0, w, lo, hi):
    a, b = math.log10(lo), math.log10(hi)
    return lambda v: x0 + (math.log10(v) - a) / (b - a) * w


def fmt_gb(v):
    """GB 단위 값 → 사람이 읽는 문자열."""
    if v >= 1000:
        return f"{v / 1000:.2f} TB"
    if v >= 100:
        return f"{v:.0f} GB"
    if v >= 1:
        return f"{v:.1f} GB"
    return f"{v * 1000:.0f} MB"


def box(f, x, y, w, h):
    f.rect(x, y, w, h, BOX[0], BOX[1], 10, 1.5)


def legend_items(f, x, y, items):
    """items: (종류, 색, 이름). 종류 = dot · ring · bar · line."""
    for kind, col, name in items:
        if kind == "dot":
            f.dot(x + 8, y - 5, 6.5, col)
        elif kind == "ring":
            f.ring(x + 8, y - 5, 6, col)
        elif kind == "bar":
            f.rect(x, y - 13, 16, 16, col, "none", 3)
        else:
            f.line(x - 2, y - 5, x + 20, y - 5, col, 3)
        f.text(x + 26, y, name, 15, "start", color=MUTED)
        x += 60 + text_w(name, 15)


def text_w(s, size):
    return sum(size * (0.98 if ord(c) > 0x1100 else 0.56) for c in s)


# ---------------------------------------------------------------- 공개일 순 그림의 공통 요소
def _months(d):
    y, m, dd = map(int, d.split("-"))
    return (y - 2024) * 12 + (m - 1) + (dd - 1) / 31


def _half(d):
    y, m = int(d[:4]), int(d[5:7])
    return f"{y} H{1 if m <= 6 else 2}"


HALVES = ["2024 H2", "2025 H1", "2025 H2", "2026 H1", "2026 H2"]
WEIGHT_FILL, WEIGHT_LIGHT = "#4a5573", "#cdd2df"                    # 가중치 (요청과 무관한 본체)
GPU_COLOR = "#c2366b"                                                # GPU 용량 기준선
GPU_LINES = [("H100 1장", 80e9), ("B200 1장", 180e9), ("H100 × 8", 640e9), ("B200 × 8", 1440e9)]


def median(vals):
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _cat_legend(f, x, y):
    for c in ("full", "sparse", "swa", "state"):
        f.dot(x + 8, y - 5, 7, CAT_COLOR[c])
        f.text(x + 24, y, CAT_LABEL[c], 15, "start", color=MUTED)
        x += 60 + text_w(CAT_LABEL[c], 15)
    return x


def _log_y(top, bot, lo, hi):
    return lambda v: bot - (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (bot - top)


def _date_columns(f, ms, x0, x1, top, bot, Y, ticks, value):
    """모델을 공개일 순으로 한 열씩 놓는다. 반기 경계 · 반기 중앙값 점선 · 회전한 모델 이름. (열 중심 x 목록, 열 폭)."""
    for v, lab in ticks:
        f.line(x0, Y(v), x1, Y(v), GRID, 1)
        f.text(x0 - 10, Y(v) + 5, lab, 13, "end", color=MUTED, mono=True)
    step = (x1 - x0) / len(ms)
    for h in HALVES:
        ii = [i for i, m in enumerate(ms) if _half(m["date"]) == h]
        if not ii:
            continue
        a, b = x0 + step * ii[0], x0 + step * (ii[-1] + 1)
        f.line(a, top - 46, a, bot, "#c9cddb", 1.2, "5 4")
        if len(ii) < 3:
            continue
        f.text((a + b) / 2, top - 28, h, 14, weight=700, color=MUTED)
        med = median(value(ms[i]) for i in ii)
        f.line(a + 4, Y(med), b - 4, Y(med), INK, 2.5, "8 5")
        f.text((a + b) / 2, top - 8, f"중앙값 {fmt_gb(med)}", 13, weight=700)
    for i, m in enumerate(ms):
        x = x0 + step * (i + 0.5)
        f.raw(f'<text transform="translate({x + 4:.1f},{bot + 12}) rotate(-60)" font-family="{FONT}" font-size="12.5" '
              f'text-anchor="end" fill="{INK}">{esc(m["name"])}</text>')
    f.line(x0, bot, x1, bot, AXIS, 1.5)
    return [x0 + step * (i + 0.5) for i in range(len(ms))], step


def _gpu_lines(f, x0, x1, Y, lines):
    for label, cap in lines:
        y = Y(cap / GB)
        f.line(x0, y, x1, y, GPU_COLOR, 2)
        f.text(x1 + 10, y - 3, label, 14, "start", 700, GPU_COLOR)
        f.text(x1 + 10, y + 15, f"{cap / 1e12:.2f} TB" if cap >= 1e12 else f"{cap / GB:.0f} GB", 13, "start",
               color=GPU_COLOR, mono=True)


# ---------------------------------------------------------------- 1. 시리즈별 attention 경로
PATHS = {   # 시리즈가 지나온 길 — (분류, 말). 그림 오른쪽 열
    "GLM": [("full", "full"), ("sparse", "sparse"), ("state", "+ state")],
    "DeepSeek": [("full", "full"), ("sparse", "sparse"), ("sparse", "압축")],
    "Hy": [("full", "full"), ("sparse", "sparse")],
    "MiniMax": [("state", "state"), ("full", "full"), ("sparse", "sparse")],
    "Qwen": [("full", "full"), ("state", "state")],
    "Nemotron": [("state", "처음부터 state")],
    "Kimi": [("full", "full"), ("state", "state")],
    "Muse": [("swa", "sliding window")],
    "MiMo": [("swa", "sliding window")],
}


def fig_paths():
    x0, x1, top = 220, 1250, 150
    colw = (x1 - x0) / len(HALVES)
    pill_h, gap = 26, 6

    def flow(ms):
        """칸 안에 알약을 왼쪽부터 채운다. ((모델, x, 폭, 줄) 목록, 줄 수)."""
        out, x, line = [], 0, 0
        for m in ms:
            w = text_w(m["short"], 13) + 30
            if x and x + w > colw - 16:
                x, line = 0, line + 1
            out.append((m, x, w, line))
            x += w + gap
        return out, line + 1

    rows = []
    for s in SERIES_ORDER:
        cells = [flow([m for m in LINEAGE if m["series"] == s and _half(m["date"]) == h]) for h in HALVES]
        rows.append((s, cells, max(n for _, n in cells) * (pill_h + gap) + 26))
    bot = top + sum(r[2] for r in rows)
    H = int(bot + 320)
    f = Fig("01-attention-paths", 1560, H)
    f.title(40, 44, "9개 시리즈 55개 모델의 attention 구성 — 반기별")
    _cat_legend(f, 40, 84)

    for i, h in enumerate(HALVES):
        f.text(x0 + colw * (i + 0.5), top - 18, h, 15, weight=700, color=MUTED, mono=True)
        f.line(x0 + colw * i, top - 6, x0 + colw * i, bot, GRID, 1)
    f.line(x1, top - 6, x1, bot, GRID, 1)
    f.text(x1 + 30, top - 18, "분류 변화", 15, "start", 700, MUTED)
    y = top
    for s, cells, rh in rows:
        f.line(40, y, 1520, y, GRID, 1)
        f.text(40, y + rh / 2 + 1, s, 18, "start", 700)
        f.text(40, y + rh / 2 + 21, next(m["co"] for m in LINEAGE if m["series"] == s), 13, "start", color=MUTED)
        for i, (items, _) in enumerate(cells):
            cx = x0 + colw * i + 10
            for m, px, w, line in items:
                py = y + 13 + line * (pill_h + gap)
                f.rect(cx + px, py, w, pill_h, CAT_LIGHT[m["cat"]], "none", 13)
                f.dot(cx + px + 13, py + pill_h / 2, 5, CAT_COLOR[m["cat"]])
                f.text(cx + px + 23, py + 18, m["short"], 13, "start", 600)
        px = x1 + 30
        for j, (c, word) in enumerate(PATHS[s]):
            if j:
                f.text(px + 2, y + rh / 2 + 7, "→", 16, "start", color=MUTED)
                px += 26
            f.text(px, y + rh / 2 + 7, word, 16, "start", 700, CAT_COLOR[c])
            px += text_w(word, 16) + 4
        y += rh
    f.line(40, bot, 1520, bot, AXIS, 1.5)

    # ---- 반기별 비율
    y = bot + 64
    f.text(40, y, "반기별 비율", 18, "start", 700)
    f.text(40, y + 24, "공개 모델 수 · 크기 변형도 하나씩", 14, "start", color=MUTED)
    bx0, bw = 220, 860
    for i, h in enumerate(HALVES[1:]):
        yy = y + 48 + i * 40
        ms = [m for m in LINEAGE if _half(m["date"]) == h]
        f.text(bx0 - 16, yy + 20, h, 15, "end", 600, mono=True)
        x = bx0
        for c in ("full", "sparse", "swa", "state"):
            k = sum(m["cat"] == c for m in ms)
            if k:
                w = bw * k / len(ms)
                f.rect(x, yy, w - 2, 28, CAT_COLOR[c], "none", 3)
                f.text(x + w / 2, yy + 20, f"{k}" if w < 90 else f"{k}개 · {k / len(ms):.0%}", 14, weight=700,
                       color="#ffffff")
                x += w
        f.text(bx0 + bw + 12, yy + 20, f"{len(ms)}개", 14, "start", color=MUTED)
    box(f, 1150, y - 26, 370, 214)
    f.lines(1174, y + 10, ["• 2025 H2 — 14개 중 full 7개",
                           "• 2026 H1 — 20개 중 full 5개",
                           "• 2026 H2 — 15개 중 full 1개 (Hy3)",
                           "• 칸 안의 색 = 그 모델의 분류",
                           "• 오른쪽 열 = 시리즈의 분류가",
                           "  바뀐 순서"], 15, 30)
    f.text(40, H - 18, "분류 순서: 층 하나라도 recurrent state를 쓰면 State → 선택 · 압축 attention이 있으면 Sparse → window 층이 있으면 Sliding window → 나머지는 Full.",
           14, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 2. 가중치 — 공개일 순
def fig_weights():
    ms = LINEAGE
    f = Fig("02-weights-timeline", 1560, 920)
    f.title(40, 44, "55개 모델의 가중치 크기 — 공개일 순")
    legend_items(f, 40, 84, [("bar", WEIGHT_FILL, "공식 체크포인트 중 가장 작은 것 (FP8 · FP4 변형 포함)"),
                             ("bar", WEIGHT_LIGHT, "같은 가중치를 BF16으로 쓰면 늘어나는 양")])
    x0, x1, top, bot = 110, 1400, 170, 620
    Y = _log_y(top, bot, 5, 7000)
    ticks = [(10, "10 GB"), (30, "30"), (100, "100 GB"), (300, "300"), (1000, "1 TB"), (3000, "3 TB")]
    xs, step = _date_columns(f, ms, x0, x1, top, bot, Y, ticks, lambda m: m["w"] / GB)
    bw = step * 0.66
    for m, x in zip(ms, xs):
        w, b = m["w"] / GB, m["wbf16"] / GB
        if b > w * 1.03:
            f.rect(x - bw / 2, Y(b), bw, Y(w) - Y(b), WEIGHT_LIGHT, "none", 2)
        f.rect(x - bw / 2, Y(w), bw, bot - Y(w), WEIGHT_FILL, "none", 2)
    _gpu_lines(f, x0, x1, Y, GPU_LINES)
    over = [sum(m["w"] > cap for m in ms) for _, cap in GPU_LINES]
    f.text(40, 878, f"공식 체크포인트로 H100 × 8(640 GB)을 넘는 모델 {over[2]}개, B200 × 8(1.44 TB)을 넘는 모델 {over[3]}개 (55개 중).  "
           "가중치만 센 값이다 — KV · 엔진 여유분은 따로 든다.", 14, "start", color=MUTED)
    f.text(40, 900, "점선 = 반기 중앙값 (진한 막대 기준).  V4.1-Flash는 host에 두는 Engram 203 GB를 뺐다.  크기는 HF safetensors 메타데이터 (2026-11-03 조회).",
           14, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 3. 1M 토큰 KV · state — 공개일 순
def fig_lineage_kv():
    ms = LINEAGE
    f = Fig("03-kv1m-timeline", 1560, 920)
    f.title(40, 44, "55개 모델의 1M 토큰 요청 하나의 KV · state 크기 — 공개일 순")
    x = _cat_legend(f, 40, 84)
    f.rect(x + 10, 71, 16, 16, "#c9cddb", "none", 3)
    f.text(x + 34, 84, "4096 토큰마다 재개 지점을 저장하면 늘어나는 양", 15, "start", color=MUTED)
    x0, x1, top, bot = 110, 1400, 170, 620
    Y = _log_y(top, bot, 1, 1000)
    ticks = [(1, "1 GB"), (3, "3"), (10, "10 GB"), (30, "30"), (100, "100 GB"), (300, "300"), (1000, "1 TB")]
    xs, step = _date_columns(f, ms, x0, x1, top, bot, Y, ticks, lambda m: sum(kv1m(m)) / GB)
    bw = step * 0.66
    for m, x in zip(ms, xs):
        s1, s2 = sum(kv1m(m)) / GB, sum(kv1m_ckpt(m)) / GB
        if s2 > s1 * 1.03:
            f.rect(x - bw / 2, Y(s2), bw, Y(s1) - Y(s2), CAT_LIGHT[m["cat"]], "none", 2)
        f.rect(x - bw / 2, Y(s1), bw, bot - Y(s1), CAT_COLOR[m["cat"]], "none", 2)
    _gpu_lines(f, x0, x1, Y, GPU_LINES[:3])
    over = [sum(sum(kv1m(m)) > cap for m in ms) for _, cap in GPU_LINES[:2]]
    f.text(40, 878, f"요청 하나의 1M KV만으로 H100 1장(80 GB)을 넘는 모델 {over[0]}개, B200 1장(180 GB)을 넘는 모델 {over[1]}개 (55개 중).  "
           "점선 = 반기 중앙값 (진한 막대 기준).", 14, "start", color=MUTED)
    f.text(40, 900, "조건: 1,048,576 토큰 · KV와 indexer key는 BF16, recurrent state는 config dtype(대부분 FP32) · MTP 층 제외 · "
           "기본 최대 길이가 1M보다 짧은 모델도 같은 길이로 계산.", 14, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 4. 가중치 + 요청 하나의 KV · state
KV128_FILL, KV1M_FILL = SERIES[0], "#a9c8f0"                        # 128K까지 / 128K → 1M


def fig_stack():
    ms = LINEAGE
    pitch, head = 22, 34
    x0, x1, hi = 330, 1290, 2800
    X = lambda v: x0 + v / hi * (x1 - x0)  # noqa: E731
    top = 190
    rows, y, prev = [], top, None
    for m in ms:
        h = _half(m["date"])
        if h != prev:
            rows.append(("head", h, y))
            y += head
            prev = h
        rows.append(("model", m, y + pitch / 2))
        y += pitch
    bot = y + 8
    H = int(bot + 110)
    f = Fig("04-weights-plus-state", 1560, H)
    f.title(40, 44, "55개 모델의 가중치 + 요청 하나의 KV · state — 128K와 1M, 공개일 순")
    legend_items(f, 40, 84, [("bar", WEIGHT_FILL, "가중치 (공식 체크포인트 중 가장 작은 것)"),
                             ("bar", KV128_FILL, "KV · state — 128K 토큰"),
                             ("bar", KV1M_FILL, "128K → 1M 토큰으로 늘어나는 양")])
    f.text(40, 112, "선형 축 · 막대 길이가 그대로 메모리 크기다.  오른쪽 숫자 = 1M 요청 하나까지 합친 크기와 필요한 GPU 수(용량만 기준)", 14,
           "start", color=MUTED)

    for v in range(0, hi + 1, 500):
        f.line(X(v), top - 6, X(v), bot, GRID, 1)
        f.text(X(v), bot + 24, "0" if v == 0 else (f"{v / 1000:g} TB" if v >= 1000 else f"{v} GB"), 13, color=MUTED,
               mono=True)
    for i, (label, cap) in enumerate(GPU_LINES):
        x = X(cap / GB)
        f.line(x, top - 30 + (i % 2) * 14, x, bot, GPU_COLOR, 1.6)
        f.text(x + 4, top - 34 + (i % 2) * 14, f"{label} {fmt_gb(cap / GB).replace('.0 GB', ' GB')}", 12.5, "start",
               700, GPU_COLOR, halo=True)

    for kind, obj, yy in rows:
        if kind == "head":
            f.text(40, yy + 23, obj, 14, "start", 700, MUTED, mono=True)
            f.line(40, yy + 30, x1 + 230, yy + 30, GRID, 1)
            continue
        m = obj
        w, k128, k1m = m["w"] / GB, sum(kv1m(m, T128)) / GB, sum(kv1m(m)) / GB
        f.dot(48, yy, 5, CAT_COLOR[m["cat"]])
        f.text(60, yy + 5, m["name"], 13, "start", 600)
        bh = 14
        f.rect(X(0), yy - bh / 2, max(X(w) - X(0), 1), bh, WEIGHT_FILL, "none", 2)
        f.rect(X(w), yy - bh / 2, max(X(w + k128) - X(w), 1), bh, KV128_FILL, "none", 0)
        f.rect(X(w + k128), yy - bh / 2, max(X(w + k1m) - X(w + k128), 1), bh, KV1M_FILL, "none", 2)
        tot = w + k1m
        n_h, n_b = math.ceil(tot / 80), math.ceil(tot / 180)
        f.text(X(tot) + 8, yy + 5, f"{fmt_gb(tot)} · H100 {n_h}장 / B200 {n_b}장", 12.5, "start", color=INK, halo=True)
    f.line(x0, bot, x1, bot, AXIS, 1.5)
    over = sum(m["w"] + sum(kv1m(m)) > 640e9 for m in ms)
    kv_big = sum(sum(kv1m(m)) > m["w"] for m in ms)
    f.text(40, H - 40, f"1M 요청 하나까지 합쳐 H100 × 8(640 GB)을 넘는 모델 {over}개.  1M 요청 하나의 KV · state가 가중치보다 큰 모델 {kv_big}개.  "
           "왼쪽 점 색 = attention 분류 (그림 01과 같다).", 14, "start", color=MUTED)
    f.text(40, H - 18, "조건: KV는 BF16, state는 config dtype · 요청 하나 · 공유 없음 · 계산값.  GPU 수는 용량 ÷ 장당 용량의 올림이며 병렬화 복제 · 엔진 여유분은 넣지 않았다.",
           14, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- A1. 토큰당 KV — 구조와 정밀도
def fig_per_token():
    ms = sorted(MODELS + APPENDIX, key=lambda m: -m["kv_bf16"])
    pitch, y0 = 48, 104
    cys = [y0 + pitch / 2 + i * pitch for i in range(len(ms))]
    bot = y0 + pitch * len(ms)
    H = int(bot + 120)
    f = Fig("A1-kv-per-token", 1560, H)
    f.title(40, 44, "대표 모델의 토큰당 KV — 공식 dtype과 BF16")

    x0, w = 400, 660
    X = logx(x0, w, 0.5, 500)
    for t in (1, 3, 10, 30, 100, 300):
        f.line(X(t), y0, X(t), bot, GRID, 1)
        f.text(X(t), bot + 26, f"{t} KiB", 14, color=MUTED, mono=True)
    f.line(x0, bot, x0 + w, bot, AXIS, 1.5)
    for m, cy in zip(ms, cys):
        f.text(40, cy - 1, m["name"], 16, "start", 600)
        f.text(40, cy + 17, f"KV {m['kv_fmt']}", 13, "start", color=MUTED)
        b, o = m["kv_bf16"] / K, m["kv"] / K
        if abs(b - o) / b > 0.02:
            f.line(X(o), cy, X(b), cy, "#c9cddb", 3)
            f.ring(X(b), cy, 6.5, MUTED)
        f.dot(X(o), cy, 7.5, INK)
        lab = f"{o:,.1f}" if o < 10 else f"{o:,.0f}"
        f.text(X(o) - 14, cy + 5, lab, 14, "end", 700, mono=True)

    bx, by, bw = 1110, y0, 410
    box(f, bx, by, bw, 400)
    f.text(bx + 24, by + 40, "구조와 정밀도를 나눠 본다", 18, "start", 700)
    f.lines(bx + 24, by + 84, ["• 검은 점 — 공식 KV dtype",
                               "• 빈 점 — 같은 구조를 BF16으로",
                               "• 두 점의 거리가 정밀도 효과다",
                               "  V3.2 (FP8) 1.8배 · V4.1-Flash (FP4) 3.6배",
                               "",
                               "• 나머지 차이는 모두 구조에서 온다",
                               "  KV head 수 · latent 차원 ·",
                               "  남은 attention 층 수 · 층 간 공유",
                               "",
                               "• Llama 3.1 70B → V4.1-Flash",
                               "  구조 1/100 × 정밀도 1/3.6 ≈ 1/370"], 15, 28)
    legend_items(f, x0, bot + 64, [("dot", INK, "공식 KV dtype"), ("ring", MUTED, "같은 구조를 BF16으로")])
    f.text(40, H - 18, "길이에 따라 커지는 부분만 (window 층 · 고정 state 제외).  1 KiB = 1,024 B — Raschka KV 노트와 맞춰 비교하기 위해 이 그림만 KiB를 쓴다.",
           15, "start", color=MUTED)
    return f


# ---------------------------------------------------------------- 표 데이터
def write_csv():
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "memory-results.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "group", "date", "weights_gpu_GB", "weights_bf16_GB", "kv_B_per_token", "kv_bf16_B_per_token",
                    "fixed_state_MB", "state_32K_GB", "state_128K_GB", "N_equal_32K", "N_equal_128K", "ctx"])
        for m in MODELS + APPENDIX:
            w.writerow([m["name"], "main" if m in MODELS else "appendix", m["date"], round(m["w_gpu"] / GB, 1),
                        round(m["w_bf16"] / GB, 1), m["kv"], m["kv_bf16"], round(m["fixed"] / 1e6, 1),
                        round(state_bytes(m, T32) / GB, 3), round(state_bytes(m, T128) / GB, 3),
                        round(m["w_gpu"] / state_bytes(m, T32), 1), round(m["w_gpu"] / state_bytes(m, T128), 1),
                        m["ctx_note"]])
    path2 = os.path.join(DATA, "lineage-kv1m.csv")
    with open(path2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["series", "company", "model", "date", "category", "attention", "kv_bf16_B_per_token", "fixed_state_MB",
                    "kv1m_state1_GB", "kv1m_ckpt4096_GB", "kv128k_state1_GB", "weights_GB", "weights_format",
                    "weights_bf16_GB", "N_equal_1M", "ctx"])
        for m in LINEAGE:
            w.writerow([m["series"], m["co"], m["name"], m["date"], m["cat"], m["arch"], m["kv"], round(m["fixed"] / 1e6, 1),
                        round(sum(kv1m(m)) / GB, 1), round(sum(kv1m_ckpt(m)) / GB, 1),
                        round(sum(kv1m(m, T128)) / GB, 2), round(m["w"] / GB, 1), m["wfmt"], round(m["wbf16"] / GB, 1),
                        round(m["w"] / sum(kv1m(m)), 1), m["ctx"]])
    return path


def print_summary():
    print(f"{'model':22s} {'W_gpu':>8s} {'W_bf16':>8s} {'kv B/t':>8s} {'fixed MB':>9s} {'S32K':>8s} {'S128K':>8s} "
          f"{'N32':>8s} {'N128':>8s} {'S@max':>8s}")
    for m in MODELS + APPENDIX:
        s32, s128 = state_bytes(m, T32), state_bytes(m, T128)
        print(f"{m['name']:22s} {m['w_gpu'] / GB:8.1f} {m['w_bf16'] / GB:8.1f} {m['kv']:8d} {m['fixed'] / 1e6:9.1f} "
              f"{s32 / GB:8.3f} {s128 / GB:8.3f} {m['w_gpu'] / s32:8.1f} {m['w_gpu'] / s128:8.1f} "
              f"{state_bytes(m, m['ctx']) / GB:8.2f}")
    print("\nnode fit (weights only, nominal capacity) and 128K conversations in the rest (logical upper bound)")
    for label, cap in NODES:
        for m in MODELS:
            left = cap - m["w_gpu"]
            n = left / state_bytes(m, T128) if left > 0 else 0
            print(f"  {label:20s} {m['name']:22s} {'fits' if left > 0 else 'OVER':5s} {n:10.0f}")


FIGS = [fig_paths, fig_weights, fig_lineage_kv, fig_stack, fig_per_token]
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


def main():
    print_summary()
    print(write_csv())
    browser = next((c for c in CHROME if os.path.exists(c)), None)
    only = sys.argv[1:]
    for make in FIGS:
        f = make()
        if only and not any(o in f.name for o in only):
            continue
        with open(os.path.join(HERE, f.name + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(f.svg())
        if browser:
            html = os.path.join(HERE, "_tmp.html")
            with open(html, "w", encoding="utf-8") as fh:
                fh.write(f'<html><body style="margin:0;background:#fff">{f.svg()}</body></html>')
            subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            "--force-device-scale-factor=2", f"--window-size={f.w},{f.h}",
                            f"--screenshot={os.path.join(HERE, f.name + '.png')}", "file:///" + html.replace("\\", "/")],
                           check=False, capture_output=True, timeout=120)
            os.remove(html)
        print("ok", f.name)


if __name__ == "__main__":
    main()
