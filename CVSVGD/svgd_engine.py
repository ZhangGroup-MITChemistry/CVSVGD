import atexit
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

import parsl
import torch
from parsl.config import Config as ParslConfig
from torch.optim.optimizer import Optimizer as TorchOptimizer

from CVSVGD.base_types.cv_def import CVDef
from CVSVGD.base_types.sampler import Sampler, SamplerProperties
from CVSVGD.cv_state_manager import CVState
from CVSVGD.logging import StateSaver, SVGDRunLogger
from CVSVGD.mean_force_manager import MeanForceManager, MeanForceResult
from CVSVGD.stein_functions import compute_ksd, compute_phi
from CVSVGD.svgd_config import SVGDConfig
from CVSVGD.utilities.worker_env import ensure_worker_importable

GAS_CONSTANT = (8.31446261815324/1000) # kilojoules / (mole * kelvin)

@dataclass
class SVGDState:
    particles: torch.Tensor
    samples: torch.Tensor|None
    mean_forces: torch.Tensor|None

class SVGDEngine:
    def __init__(
        self,
        svgd_config:SVGDConfig,
        cv_defs: Sequence[CVDef],
        sampler: type[Sampler],
        sampler_props: SamplerProperties,
        parsl_config: ParslConfig,
        optimizer_type: type[TorchOptimizer]=torch.optim.Adam,
        optimizer_kwargs: dict|None=None,
        restart: bool = False
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_type is torch.optim.Adam:
                optimizer_kwargs = {'betas': (0.6, 0.999), 'eps': 0.1}

        ensure_worker_importable(sampler,type(sampler_props))

        self.svgd_conf: SVGDConfig = svgd_config
        self.state_saver: StateSaver = StateSaver(
            svgd_config.checkpoint_file,
            svgd_config.state_directory,
        )
        self.parsl_conf: ParslConfig = parsl_config
        self.cv_state:CVState = CVState(svgd_config.n_particles,cv_defs, svgd_config.device, svgd_config.dtype)
        self.mean_force_manager:MeanForceManager = MeanForceManager(sampler_props,sampler)

        ''' Internal Variables '''
        self._svgd_iterator: int = 0
        self._parsl_loaded: bool = False
        self._continuing: bool = False
        self._engine_initialized: bool = False
        self._closed: bool = False
        self._sampler_type: type[Sampler] = sampler
        self.restart: bool = restart

        self._dfk: parsl.DataFlowKernel | None
        self._optim_fac: type[TorchOptimizer] = optimizer_type
        self._optim_kwarg: dict = optimizer_kwargs
        self.optimizer: TorchOptimizer
        self.beta_svgd: float
        self.logger:SVGDRunLogger

    def initialize_run(self, load_parsl:bool = True):
        opt_state = None
        _log_write_mode = "w"
        ckpt = self.state_saver.load_checkpoint(self.svgd_conf.device)
        if ckpt is not None and not self.restart:
            self.cv_state.particles.copy_(ckpt["particles"].to(self.svgd_conf.dtype))
            self._svgd_iterator = ckpt["iteration"] + 1
            torch.set_rng_state(ckpt["rng"].cpu())
            opt_state = ckpt["optimizer"]

        log_file = None
        if self.svgd_conf.log_file is None:
            log_file = sys.stdout
        else:
            log_file = self.svgd_conf.log_file
        self.logger = SVGDRunLogger(write_mode=_log_write_mode, log_type=self.svgd_conf.svgd_log_type, log_file=log_file)

        if "lr" not in self._optim_kwarg and "lr" not in getattr(self._optim_fac, "keywords", {}) or {}:
            self._optim_kwarg.setdefault("lr", self.svgd_conf.svgd_step_size)
        self.optimizer = self._optim_fac([self.cv_state.particles], **self._optim_kwarg)

        self.beta_svgd = 1 / (self.svgd_conf.svgd_temp * GAS_CONSTANT)

        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
            for st in self.optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device=self.svgd_conf.device)

        if not self._parsl_loaded and load_parsl:
            self._dfk = parsl.load(self.parsl_conf)
            self._parsl_loaded = True
            atexit.register(self.shutdown)

        self._engine_initialized = True
        if self._svgd_iterator > 0:
            return
        if self.svgd_conf.initial_distribution is not None:
            if not isinstance(self.svgd_conf.initial_distribution, torch.Tensor):
                init_dist = torch.tensor(self.svgd_conf.initial_distribution,
                            dtype=self.svgd_conf.dtype,
                            device=self.svgd_conf.device)
                try:
                    init_dist = init_dist.reshape(self.cv_state.particles.shape)
                except RuntimeError:
                    raise RuntimeError("[CVSVGD] Given initial distribution cannot be reshaped to CV-particle dimensions.")
                self.cv_state.particles.copy_(init_dist)
        else:
            init_dist = torch.rand(
                (self.cv_state.cv_dim, self.svgd_conf.n_particles),
                dtype=self.svgd_conf.dtype,
                device=self.svgd_conf.device,
            )
            init_dist = torch.stack(
                [
                    cv.prior_bounds[0] + init_dist[d] * (cv.prior_bounds[1] - cv.prior_bounds[0])
                    for d, cv in enumerate(self.cv_state.cv_defs)
                ]
            )
            self.cv_state.particles.copy_(init_dist)
            self.cv_state.particles.copy_(self.cv_state.wrap())
        self.mean_force_manager.initialize_samplers(self.cv_state)

    def run(self):
        if not self._engine_initialized:
            self.initialize_run()
        time_start = time.time()

        if self._continuing and (
            self._svgd_iterator == self.svgd_conf.n_iterations - 1
        ):
            return
        for _ in range(self._svgd_iterator, self.svgd_conf.n_iterations):
            self.optimizer.zero_grad()
            t_s = time.time()
            mean_force_res: MeanForceResult = self.mean_force_manager.calc_mean_forces(self.cv_state, self._svgd_iterator)
            grad_logp = -self.beta_svgd*mean_force_res.mean_forces
            t_e = time.time()
            sim_time = t_e - t_s

            t_s = time.time()
            phi = compute_phi(grad_logp, self.cv_state, self.svgd_conf.kernel_func)
            ksd = None
            if self.svgd_conf.compute_ksd:
                ksd = compute_ksd(grad_logp, self.cv_state, self.svgd_conf.kernel_func, self.svgd_conf.bw_ksd)

            t_e = time.time()
            phi_time = t_e - t_s
            if torch.any(torch.isnan(phi)):
                self.shutdown()
                raise RuntimeError(
                    f"[CVSVGD] NaN detected in phi vector at iteration {self._svgd_iterator}"
                )
            self.cv_state.particles.grad = -phi.detach()

            self.state_saver.save_checkpoint(
                self.optimizer,
                self.cv_state.particles,
                mean_force_res.samples,
                mean_force_res.mean_forces,
                self._svgd_iterator,
            )
            if self._svgd_iterator < self.svgd_conf.n_iterations - 1:
                self.optimizer.step()

            with torch.no_grad():
                time_end = time.time()
                time_elapsed = time_end - time_start
                phi_norm = torch.linalg.norm(phi)
                self.logger(phi_norm, sim_time, phi_time, time_elapsed, self._svgd_iterator, ksd=ksd)
                time_start = time.time()
            self._svgd_iterator += 1

    '''
        Repeats the sampling at a given iteration, or the last run iteration if none is specified.
        The state at the given iteration will be overwritten with the new data.
        A new sim_properties object must be given to specify new simulation parameters;
        for example, increased sampling time.
    '''
    def sample_at_state(self, sim_properties: SamplerProperties, iteration:int|None = None):
        my_mean_force_man = MeanForceManager(sim_properties, self._sampler_type)
        backup_state = self.cv_state.particles.clone()
        if not self._engine_initialized:
            self.initialize_run()
        n_done = self._svgd_iterator
        if n_done == 0:
            raise RuntimeError("[CVSVGD] No iterations have been run yet.")
        if iteration is None:
            iteration = n_done - 1
        elif iteration < 0:
            iteration += n_done
        if not 0 <= iteration < n_done:
            raise IndexError(f"[CVSVGD] iteration must be in [0, {n_done - 1}] (or negative from the end).")

        st = self.state_saver.load_state(iteration)
        self.cv_state.particles.copy_(st["particles"])

        '''
        If the requested iteration is equal to the last iteration run, then there should still be checkpoints saved
        so intializing again isn't necessary, but, for generality with any implemented sampler, we reinitialize at
        the new state regardless. A bit wasteful compute-wise but for now I think this is the most general option.
        '''
        my_mean_force_man.initialize_samplers(self.cv_state)
        mean_force_res:MeanForceResult = my_mean_force_man.calc_mean_forces(self.cv_state, iteration)
        self.state_saver.save_state(
            self.cv_state.particles,
            mean_force_res.samples,
            mean_force_res.mean_forces,
            iteration,
        )
        self.cv_state.particles.copy_(backup_state)

    def get_state(self, iteration: int | None = None) -> SVGDState:
        if not self._engine_initialized:
            self.initialize_run(load_parsl=False)
        samples = None
        mean_forces = None
        n_done = self._svgd_iterator
        if n_done == 0:
            raise RuntimeError("[CVSVGD] No iterations have been run yet.")
        if iteration is None:
            iteration = n_done - 1
        elif iteration < 0:
            iteration += n_done
        if not 0 <= iteration < n_done:
            raise IndexError(f"[CVSVGD] iteration must be in [0, {n_done - 1}] (or negative from the end).")

        st = self.state_saver.load_state(iteration)
        particles, samples, mean_forces = st["particles"], st["samples"], st["mean_forces"]
        return SVGDState(particles=particles, mean_forces=mean_forces, samples=samples)

    def shutdown(self):
        if self._closed:
            return
        self._closed = True

        if self._parsl_loaded:
            try:
                if self._dfk is not None:
                    self._dfk.cleanup()
            except Exception as e:
                print(f"[CVSVGD] Parsl cleanup failed: {e}")
            finally:
                try:
                    parsl.clear()
                except Exception:
                    pass

                self._parsl_loaded = False
                self._dfk = None
        atexit.unregister(self.shutdown)

    def __enter__(self):
        if not self._engine_initialized:
            self.initialize_run()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
