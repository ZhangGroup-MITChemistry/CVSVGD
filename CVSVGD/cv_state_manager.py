from collections.abc import Sequence

import torch

from CVSVGD.base_types.cv_def import CVDef


class CVState:
    def __init__(self, n_particles:int, cv_defs:Sequence[CVDef], device:torch.device, dtype:torch.dtype) -> None:
        self.validate_cvdefs(cv_defs)
        self.cv_dim:int = len(cv_defs)
        self.n_particles:int = n_particles
        self.particles:torch.Tensor = torch.empty([self.cv_dim, self.n_particles], device=device, dtype=dtype)
        self.periodic:torch.Tensor = torch.tensor([cv.periodic for cv in cv_defs],device=device, dtype=torch.bool)
        self.period:torch.Tensor = torch.tensor([1.0 if cv.period is None else cv.period for cv in cv_defs],device=device, dtype=dtype)
        self.length_scales:torch.Tensor = torch.tensor([1.0 if cv.length_scale is None else cv.length_scale for cv in cv_defs],device=device, dtype=dtype)
        self.names:list[str] = [cv.name for cv in cv_defs]
        self.restraints:list[str] = [cv.restraint_type for cv in cv_defs]
        self.cv_defs:Sequence[CVDef] = cv_defs

    def validate_cvdefs(self, cv_defs:Sequence[CVDef]) -> None:
        for cvdef in cv_defs:
            if not isinstance(cvdef, CVDef):
                raise TypeError("[CVSVGD] Passed cvdef is not type CVDef. Object should be inherited from CVSVGD.sampler_protocols.cv_def.CVDef")
            if not cvdef._post_init_was_run:
                raise RuntimeError("[CVSVGD] Custom CVDef class did not run __post_init__. Ensure super().__post_init__() is called in custom implementation.")
            if cvdef.prior_bounds is None and (cvdef.hard_bounds[1] == float('inf') or cvdef.hard_bounds[0] == float('-inf')):
                raise ValueError(f"[CVSVGD] Prior bounds for CV {cvdef.name} not specified, and hard bounds contain INF. Finite prior bounds are necessary for initialization.")
            elif cvdef.prior_bounds is None:
                cvdef.prior_bounds = (cvdef.hard_bounds[0],cvdef.hard_bounds[1])

            if cvdef.prior_bounds is not None and (cvdef.prior_bounds[1] == float('inf') or cvdef.prior_bounds[0] == float('-inf')):
                raise ValueError(f"[CVSVGD] Prior bounds for CV {cvdef.name} contain INF. Finite prior bounds are necessary for initialization.")

            if cvdef.periodic and cvdef.period is None:
                cvdef.period = abs(cvdef.hard_bounds[1] - cvdef.hard_bounds[0])

            if cvdef.periodic and not (cvdef.period and cvdef.period > 0):
                raise ValueError(f"[CVSVGD] {cvdef.name}: periodic CV needs a positive period")

    def wrap(self) -> torch.Tensor:
        p = self.period[:, None]
        return torch.where(self.periodic[:, None], self.particles - p * torch.floor(self.particles / p + 0.5), self.particles)
