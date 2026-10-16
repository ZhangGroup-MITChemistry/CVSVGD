import torch

from CVSVGD.base_types.kernels import KernelBase, KernelResult
from CVSVGD.cv_state_manager import CVState


def compute_phi(grad_logp: torch.Tensor, cvstate:CVState, kernel: KernelBase) -> torch.Tensor:
    kern_res:KernelResult = kernel(cvstate, compute_trace = False, bw = None)
    score_term = grad_logp @ kern_res.k
    phi = (score_term + kern_res.grad_k.sum(2)) / cvstate.n_particles
    return phi

def compute_ksd(grad_logp: torch.Tensor, cvstate:CVState, kernel: KernelBase, bw = 1.0) -> float:
    kern_res:KernelResult = kernel(cvstate, compute_trace=True, bw=bw)
    score_term = grad_logp @ kern_res.k
    ksd = stein_discrepancy(score_term, kern_res).item()
    return  ksd

def stein_discrepancy(s, kern_res):
    N = kern_res.shape[0]
    t1 = (s.T @ s) * kern_res.k
    t2 = torch.einsum("di,dij->ij", s, kern_res.k_grad)
    t3 = -torch.einsum("dj,dij->ij", s, kern_res.k_grad)
    up = t1 + t2 + t3 + kern_res.k_trace
    return up.sum() / (N * N)
