from dataclasses import dataclass

import torch
from parsl.app.app import python_app

from CVSVGD.base_types.sampler import (
    Sampler,
    SamplerProperties,
    SimulationJob,
    SimulationResult,
)
from CVSVGD.cv_state_manager import CVState


@dataclass
class MeanForceResult:
    mean_forces: torch.Tensor
    samples: torch.Tensor

class MeanForceManager:
    def __init__(self, sampler_properties: SamplerProperties, sampler: type[Sampler]) -> None:
        self.sampler:type[Sampler] = sampler
        self.sampler_prop:SamplerProperties = sampler_properties
        self.init_samples = python_app(sampler.initialize)
        self.sim_samples = python_app(sampler.simulate)

    def initialize_samplers(self, cvstate: CVState) -> None:
        if not self.sampler_prop.initialize:
            return
        futures = []
        for i in range(cvstate.n_particles):
            sim_job = SimulationJob(particle_id=i, cv_names=tuple(cvstate.names), center=tuple(cvstate.particles[:,i].cpu().tolist()), iteration=0)
            futures.append(self.init_samples(sim_job, self.sampler_prop))
        _ = [f.result() for f in futures]

    def calc_mean_forces(self, cvstate: CVState, svgd_iteration: int) -> MeanForceResult:
        futures = []
        wrapped_particles = cvstate.wrap()
        for i in range(cvstate.n_particles):
            sim_job = SimulationJob(particle_id=i, cv_names=tuple(cvstate.names), center=tuple(wrapped_particles[:,i].cpu().tolist()), iteration=svgd_iteration)
            futures.append(self.sim_samples(sim_job, self.sampler_prop))
        results:list[SimulationResult] = [f.result() for f in futures]
        sample_stack, mean_forces = self.map_to_torch(results, cvstate, cvstate.particles.dtype, cvstate.particles.device)
        return MeanForceResult(mean_forces=mean_forces,samples=sample_stack)

    def map_to_torch(self, simres: list[SimulationResult], cvstate: CVState, dtype: torch.dtype, device: torch.device):
        sim_samples = []
        mean_forces = []
        sim_sorter = []
        mf_sorter = []
        for cv_d in simres:
            sim_res = torch.stack(
                [
                    torch.tensor(cv_d.samples[cvname], dtype=dtype, device=device)
                    for cvname in cvstate.names
                ]
            )
            sim_samples.append(sim_res)
            sim_sorter.append(cv_d.particle_id)
            mf_res = torch.stack(
                [
                    torch.tensor(cv_d.mean_force[cvname], dtype=dtype, device=device)
                    for cvname in cvstate.names
                ]
            )
            mean_forces.append(mf_res)
            mf_sorter.append(cv_d.particle_id)

        '''Sorting is probably unnecessary, but its done as a precaution JIC'''
        sim_sorter = torch.tensor(sim_sorter).argsort()
        mf_sorter = torch.tensor(mf_sorter).argsort()
        sim_samples = torch.stack(sim_samples, dim=1)[:,sim_sorter]
        mean_forces = torch.stack(mean_forces, dim=1)[:,mf_sorter]
        return sim_samples, mean_forces
