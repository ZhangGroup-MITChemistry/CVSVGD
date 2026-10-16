from dataclasses import dataclass
from typing import Protocol

import torch

from CVSVGD.cv_state_manager import CVState


@dataclass
class KernelResult:
    k: torch.Tensor
    grad_k: torch.Tensor
    k_trace: torch.Tensor|None

class KernelBase(Protocol):
    def __call__(self, data:CVState, compute_trace: bool, bw:float|None) -> KernelResult:
        raise NotImplementedError

class IMQKernel(KernelBase):
    def __init__(self, c:float=1.0, b:float=-0.5):
        self.c:float = c
        self.b:float = b

    def __call__(self, data:CVState, compute_trace: bool = False, bw:float|None=None) -> KernelResult:
        r2,dr2,d2r2 = self.get_diffs(data,compute_trace)
        DIM = r2.shape[0]
        if bw is None:
            h = torch.median(r2.reshape(DIM, -1), dim=1).values
        else:
            h = bw
        h = torch.as_tensor(h, device=r2.device, dtype=r2.dtype).reshape(-1, 1, 1)
        u = self.c**2 + (r2 / h).sum(dim=0)
        k = u**self.b
        dk_du = self.b * u ** (self.b - 1.0)
        grad_k = (dr2 / h) * dk_du[None, :, :]
        if compute_trace and d2r2 is not None:
            quad = self.b * (self.b - 1.0) * u ** (self.b - 2.0) * ((dr2 / h) ** 2).sum(0)
            lin = dk_du * (d2r2 / h).sum(0)
            k_trace = -(quad + lin)
        else:
            k_trace = None
        return KernelResult(k=k,grad_k=grad_k,k_trace=k_trace)

    def get_diffs(self, data:CVState, compute_trace:bool) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor|None]:
        x = data.particles
        delta = x[:, None, :] - x[:, :, None]
        p = data.period[:,None,None]
        p_factor = (2*torch.pi/p)
        r2_t = torch.where(data.periodic[:,None,None], 2.0 - 2.0*torch.cos(p_factor*delta), delta**2)
        dr2_t = torch.where(data.periodic[:,None,None], 2.0 * p_factor*torch.sin(p_factor*delta), 2*delta)
        if compute_trace:
            d2r2_t = torch.where(data.periodic[:,None,None], 2.0 * p_factor**2 * torch.cos(p_factor*delta), torch.full_like(delta, 2.0))
        else:
            d2r2_t = None
        return r2_t,dr2_t,d2r2_t
