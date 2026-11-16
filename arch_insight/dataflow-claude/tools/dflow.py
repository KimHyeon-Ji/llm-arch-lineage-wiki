"""dataflow-claude 공용 도구.

- Rec      : 참조 구현 안에서 단계(step)와 이름 붙은 텐서를 기록한다.
- _OpLog   : 단계 안에서 실행된 aten 연산과 GEMM FLOP을 기록한다 (TorchDispatchMode).
- ev       : yaml의 shape · 바이트 · FLOP 식을 차원 값으로 계산한다.

meta device에서 돌리면 실제 차원(예: T = 131,072)으로도 메모리 0으로 shape를 얻는다.
"""
import contextlib

import torch
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten


def _gemm_flops(func, args):
    if func is aten.mm.default:
        a, b = args[0], args[1]
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]
    if func is aten.addmm.default:
        a, b = args[1], args[2]
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]
    if func is aten.bmm.default:
        a, b = args[0], args[1]
        return 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
    if func is aten.baddbmm.default:
        a, b = args[1], args[2]
        return 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
    return 0


class _OpLog(TorchDispatchMode):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        outs = out if isinstance(out, (tuple, list)) else [out]
        self.sink.append({
            "op": func.overloadpacket.__name__,
            "gemm_flops": _gemm_flops(func, args),
            "out": [list(o.shape) for o in outs if isinstance(o, torch.Tensor)],
        })
        return out


class Rec:
    """참조 구현이 단계와 텐서를 남기는 기록기."""

    def __init__(self, log_ops=True):
        self.log_ops = log_ops
        self.order, self.steps, self.tensors = [], {}, {}
        self.cur = None

    @contextlib.contextmanager
    def step(self, sid):
        ops = []
        self.steps[sid] = {"ops": ops}
        self.order.append(sid)
        self.cur = sid
        if self.log_ops:
            with _OpLog(ops):
                yield
        else:
            yield
        self.cur = None

    def t(self, name, x):
        self.tensors[name] = {
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "step": self.cur,
        }
        return x

    def gemm_flops(self, sid):
        return sum(o["gemm_flops"] for o in self.steps[sid]["ops"])


class NullRec:
    """기록하지 않는 기록기 (테스트용)."""

    def step(self, sid):
        return contextlib.nullcontext()

    def t(self, name, x):
        return x


def ev(expr, dims):
    """'2*B*n_I*d_I*T' 같은 식을 dims로 계산한다. 숫자면 그대로 돌려준다."""
    if isinstance(expr, (int, float)):
        return expr
    return eval(str(expr), {"__builtins__": {}}, dict(dims))
