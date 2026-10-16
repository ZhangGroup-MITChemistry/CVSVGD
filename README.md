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

## License
MIT License

Copyright (c) 2026 ZhangGroup-MITChemistry

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 
## Citation
