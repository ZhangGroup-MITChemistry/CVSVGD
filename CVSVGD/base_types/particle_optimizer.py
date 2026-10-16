from collections import defaultdict
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ParticleOptimizer(Protocol):
    state: defaultdict[Any, Any]
    def __init__(self, params: Any, *args: Any, **kwargs: Any) -> None: raise NotImplementedError
    def zero_grad(self) -> None: raise NotImplementedError
    def step(self) -> None: raise NotImplementedError
    def state_dict(self) -> dict: raise NotImplementedError
    def load_state_dict(self, state_dict: dict) -> None: raise NotImplementedError
