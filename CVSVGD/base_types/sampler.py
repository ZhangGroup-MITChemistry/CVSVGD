from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable


@dataclass(frozen=True, slots=True)
class SimulationJob:
    particle_id: int
    cv_names: tuple[str,...]
    center: tuple[float, ...]
    iteration: int

@dataclass(frozen=True, slots=True)
class SimulationResult:
    particle_id: int
    center: tuple[float, ...]
    iteration: int
    samples: dict[str, list[float]]
    mean_force: dict[str, float]

@runtime_checkable
class SamplerProperties(Protocol):
    initialize: bool = True
    equilibrate: bool = False


P_contra = TypeVar("P_contra", bound=SamplerProperties, contravariant=True)

@runtime_checkable
class Sampler(Protocol[P_contra]):
    @staticmethod
    def initialize(job: SimulationJob, samp_props:P_contra) -> None: raise NotImplementedError
    @staticmethod
    def simulate(job: SimulationJob, samp_props:P_contra) -> SimulationResult: raise NotImplementedError
