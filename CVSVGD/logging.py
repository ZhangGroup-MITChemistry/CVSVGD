import atexit
import os
import sys
from typing import IO

import torch


class SVGDRunLogger:
    def __init__(self, write_mode:str, log_type:str, log_file:str|IO[str] = sys.stdout):
        self.log_write_mode: str = write_mode
        self.log_filename:str = ""
        self._logstream: IO[str]
        self.log_to_stdout: bool = False
        if isinstance(log_file, str):
            self.log_filename = log_file
        else:
            self._logstream = log_file
            self.log_to_stdout = True
        self.log_file_opened: bool = False
        self.log_type:str = log_type
        self.time_per_iter_avg:float = 0.0
        atexit.register(self.close)

    def close(self) -> None:
        if self.log_to_stdout:
            self._logstream.flush()
            self.log_file_opened = False

        elif self.log_file_opened and self._logstream and not self._logstream.closed:
            self._logstream.flush()
            self._logstream.close()
            self.log_file_opened = False

    def __call__(self, phi_norm:float, sim_time:float, phi_time:float,
                time_elapse:float, iteration:int, ksd:float|None = None) -> None:
        if not self.log_file_opened:
            if not self.log_to_stdout:
                self._logstream = open(file=self.log_filename, mode=self.log_write_mode)
            if 'a' not in self.log_write_mode and self.log_type == 'csv':
                print_str = (
                    "\"Step,\""
                    "\"Norm Phi,\""
                    + ("\"KSD,\"" if ksd is not None else "") +
                    "\"Seconds Per Iteration,\""
                    "\"Mean Force Compute Time,\""
                    "\"Phi Compute Time\n\""
                )
                self._logstream.write(print_str)
                self._logstream.flush()
            self.log_file_opened = True

        sim_time = sim_time / 60
        phi_time = phi_time * 1000
        self.time_per_iter_avg = self.time_per_iter_avg + ((time_elapse - self.time_per_iter_avg)/(iteration+1))
        if self.log_type == 'table':
            print_str = (
                f"Step: {iteration:6d}| Norm(Phi): {phi_norm:.2f}"
                + (f" | KSD: {ksd:.2f} | " if ksd is not None else "")+
                f" | Min/Iter: {self.time_per_iter_avg / 60:.2f}"
                f" | Total Mean Force Time: {sim_time:.2f}min | Phi Calc Time: {phi_time:.2f}ms\n"
            )
        else:
            print_str = (
                f"{iteration},{phi_norm:.2f},"
                + (f"{ksd:.2f}," if ksd is not None else "") +
                f"{self.time_per_iter_avg:.2f},"
                f"{sim_time:.2f},{phi_time:2f}\n"
            )
        self._logstream.write(print_str)
        self._logstream.flush()

class StateSaver:
    def __init__(self, checkpoint_file: str, state_log_dir: str):
        os.makedirs(state_log_dir, exist_ok=True)
        self.checkpoint_path: str = checkpoint_file
        self.state_log_dir: str = state_log_dir

    def state_path(self, iteration: int) -> str:
        return os.path.join(self.state_log_dir, f"state_iter_{iteration}.pt")

    @staticmethod
    def _save_temp(obj, path: str):
        tmp = f"{path}.tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save_checkpoint(self, optim, particles, sim_samples, mean_forces, iteration: int):
        self._save_temp({
            "optimizer": optim.state_dict(),
            "particles": particles.detach().cpu(),
            "iteration": iteration,
            "rng": torch.get_rng_state(),
        }, self.checkpoint_path)
        self.save_state(particles, sim_samples, mean_forces, iteration)

    def save_state(self, particles, sim_samples, mean_forces, iteration: int):
        self._save_temp({
            "particles": particles.detach().cpu(),
            "samples": sim_samples.detach().cpu(),
            "mean_forces": mean_forces.detach().cpu(),
            "iteration": iteration,
        }, self.state_path(iteration))

    def load_checkpoint(self, device) -> dict | None:
        if not os.path.exists(self.checkpoint_path):
            return None
        return torch.load(self.checkpoint_path, map_location=device, weights_only=True)

    def load_state(self, iteration: int, device="cpu") -> dict:
        return torch.load(self.state_path(iteration), map_location=device, weights_only=True)
