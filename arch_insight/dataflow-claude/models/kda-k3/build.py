"""algo.yaml + map-gpu.yaml → 수치 · 뷰어 (Kimi K3.1 Pro KDA 층). 참조 구현 검증 없음.

  python build.py      → numbers.md · dataflow.json · ../../viewer/kda-k3-lite.html
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import buildlib as bl  # noqa: E402

if __name__ == "__main__":
    sys.exit(bl.build_noref(HERE))
