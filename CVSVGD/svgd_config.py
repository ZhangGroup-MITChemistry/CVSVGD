import sys
from collections.abc import Iterable
from typing import TextIO

import torch

from CVSVGD.base_types.kernels import IMQKernel, KernelBase


class SVGDConfig:
    def __init__(
        self,
        state_directory: str,
        checkpoint_file: str,
        n_particles: int = 200,
        n_iterations: int = 200,
        svgd_step_size: float = 0.1,
        svgd_temp: float = 5000.0,
        kernel_func: KernelBase|None = None,
        compute_ksd: bool = False,
        bw_ksd: float = 1.0,
        log_file: str|None = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype: torch.dtype = torch.float32,
        initial_distribution: Iterable[float]|None = None
    ):
        self.n_particles: int = n_particles
        self.n_iterations: int = n_iterations
        self.svgd_step_size: float = svgd_step_size
        self.svgd_temp: float = svgd_temp
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.state_directory: str = state_directory
        self.checkpoint_file: str = checkpoint_file
        self.initial_distribution: Iterable[float]|None = initial_distribution
        if kernel_func is None:
            kernel_func = IMQKernel()
        self.kernel_func:KernelBase = kernel_func
        self.compute_ksd:bool = compute_ksd
        self.bw_ksd:float = bw_ksd
        self.log_file:str|None = log_file
        self.svgd_log_type:str = 'table' if log_file is None else 'csv'
