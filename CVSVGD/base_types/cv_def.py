import math
from dataclasses import dataclass


@dataclass
class CVDef:
    name: str
    length_scale: float|None = None
    periodic: bool = False
    period: float | None = None
    restraint_type: str = 'harmonic'
    hard_bounds: tuple[float,float] = (float('-inf'), float('inf'))
    prior_bounds: tuple[float,float]|None = None
    _post_init_was_run:bool = False
    def __post_init__(self):
        if self.prior_bounds is None:
            self.prior_bounds = self.hard_bounds
        if not all(map(math.isfinite, self.prior_bounds)):
            raise ValueError(f"[CVSVGD] {self.name}: finite prior_bounds or hard_bounds required for initialization.")

        if self.periodic:
            if self.period is None:
                if not (math.isfinite(self.hard_bounds[0]) and math.isfinite(self.hard_bounds[1])):
                    raise ValueError(f"[CVSVGD] {self.name}: periodic CV needs a period or finite hard_bounds.")
                self.period = self.hard_bounds[1] - self.hard_bounds[0]
            if self.period <= 0:
                raise ValueError(f"[CVSVGD] {self.name}: period must be positive.")
        elif self.period is not None:
            raise ValueError(f"[CVSVGD] {self.name}: period given but periodic=False.")
        self._post_init_was_run = True
