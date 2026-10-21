# CVSVGD
 
Stein Variational Gradient Descent (SVGD) in collective-variable (CV) space, with mean-force sampling run in parallel via [Parsl](https://parsl-project.org/).
 
## Requirements
 
- Python >= 3.10
- `torch`, `parsl` (installed automatically)
- Optional: `openmm` for the OpenMM sampler (`pip install -e ".[openmm]"`)
## Install
 
To install directly from github, go to the preferred install directory (e.g. `$HOME/.local/src`):
```bash
mkdir -p "$HOME/.local/src" && cd "$HOME/.local/src"
```

Then run the one-liner:
```bash
git clone https://github.com/ZhangGroup-MITChemistry/CVSVGD.git cvsvgd && cd cvsvgd && pip install -e .
```

A pypi install will be available soon.
 
## Usage

Basic usage for OpenMM 
```python
from CVSVGD.svgd_config import SVGDConfig
from CVSVGD.svgd_engine import SVGDEngine
from CVSVGD.samplers.OMMSampler import OMMCVDef, OMMSampler, OMMSamplerProperties
```
 
To use with OpenMM, define your CVs (`OMMCVDef`), simulation configuration with `OMMSamplerProperites`, and a Parsl `Config`, then pass them to `SVGDEngine`. Full working scripts are in `examples/`:
 
| Example | Description |
|---|---|
| `examples/double_well_toy_model/` | Analytic double-well toy model |
| `examples/alanine_dipeptide/` | OpenMM, alanine dipeptide |
| `examples/alanine_tripeptide/` | OpenMM, alanine tripeptide |
 
Can be extended to hypothetically use any sampler, with a minimal example given in `examples/double_well_toy_model/`. A more detailed guide on how to do this will be included in the future.
 
## Citation
Riveros, I.; Bin, Z. Collective-Variable Stein Variational Gradient Descent for Free Energy Surface Exploration, 2026. https://arxiv.org/abs/2610.27429v1
