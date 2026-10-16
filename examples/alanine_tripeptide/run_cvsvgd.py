import openmm as mm
import torch
from openmm import app, unit
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.executors.high_throughput.errors import ManagerLost
from parsl.launchers import SingleNodeLauncher
from parsl.providers import SlurmProvider

import matplotlib.pyplot as plt

from CVSVGD.samplers.OMMSampler import OMMCVDef, OMMSampler, OMMSamplerProperties
from CVSVGD.svgd_config import SVGDConfig
from CVSVGD.svgd_engine import SVGDEngine
from CVSVGD.analysis.fep_mbf import FreeEnergyProfileMBF

output_dir = "temp_10000_simtime_1ps_N_200_new"

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_PREEMPTIONS = 20

def retry_handler(exception, task_record):
    if isinstance(exception, ManagerLost):
        return 0.0 if task_record["fail_count"] < MAX_PREEMPTIONS else 100.0
    return 1.0

parsl_config = Config(
    retries=4,
    retry_handler=retry_handler,
    run_dir=f"{output_dir}/parsl_runinfo",
    executors=[
        HighThroughputExecutor(
            label="partition_4_jobs",
            available_accelerators=1,
            block_error_handler=False, #for dealing with lost preemptable nodes.
            provider=SlurmProvider(
                partition="mit_preemptable",
                nodes_per_block=1,
                init_blocks=4,
                min_blocks=4,
                max_blocks=4,
                gres="gpu:1",
                mem_per_node=8,
                cores_per_node=1,
                exclusive=False,
                launcher=SingleNodeLauncher(),
                walltime="6:00:00",
            ),
        ),
        HighThroughputExecutor(
            label="partition_2_jobs",
            available_accelerators=1,
            provider=SlurmProvider(
                partition="mit_normal_gpu",
                nodes_per_block=1,
                init_blocks=2,
                min_blocks=2,
                max_blocks=2,
                gres="gpu:1",
                mem_per_node=8,
                cores_per_node=1,
                exclusive=False,
                launcher=SingleNodeLauncher(),
                walltime="6:00:00",
            ),
        ),
    ],
)

bias_torsion_psi1 = mm.CustomTorsionForce("theta")
bias_torsion_psi1.addTorsion(4, 6, 8, 14)

bias_torsion_phi1 = mm.CustomTorsionForce("theta")
bias_torsion_phi1.addTorsion(6, 8, 14, 16)

bias_torsion_psi2 = mm.CustomTorsionForce("theta")
bias_torsion_psi2.addTorsion(14, 16, 18, 24)

bias_torsion_phi2 = mm.CustomTorsionForce("theta")
bias_torsion_phi2.addTorsion(16, 18, 24, 26)

cv_defs: list[OMMCVDef] = [
    OMMCVDef(
        name="psi1",
        omm_force=bias_torsion_psi1,
        hard_bounds=(-torch.pi, torch.pi),
        periodic=True,
    ),
    OMMCVDef(
        name="phi1",
        omm_force=bias_torsion_phi1,
        hard_bounds=(-torch.pi, torch.pi),
        periodic=True,
    ),
    OMMCVDef(
        name="psi2",
        omm_force=bias_torsion_psi2,
        hard_bounds=(-torch.pi, torch.pi),
        periodic=True,
    ),
    OMMCVDef(
        name="phi2",
        omm_force=bias_torsion_phi2,
        hard_bounds=(-torch.pi, torch.pi),
        periodic=True,
    ),
]

setup_dir = "./setup_files/"
system_file = f"{setup_dir}/amber_trialanine.xml"
pdb_file = f"{setup_dir}/solvated_trialanine.pdb"
pdb = app.PDBFile(pdb_file)
with open(system_file, "r") as f:
    system = mm.XmlSerializer.deserialize(f.read())
t = 300 * unit.kelvin
friccoef = 5 / unit.picosecond
timestep = 1 * unit.femtosecond
integrator = mm.LangevinMiddleIntegrator(t, friccoef, timestep)
platform_name = "CUDA"
platform_properties = {"Precision": "mixed"}

sim_config = OMMSamplerProperties(
    cv_defs=cv_defs,
    force_constant=2500,
    integrator=integrator,
    system=system,
    topology=pdb.getTopology(),
    initial_positions=pdb.getPositions(),
    platform_name=platform_name,
    platform_properties=platform_properties,
    output_directory=output_dir,
    sim_time=1 * unit.picosecond,
    sim_timestep=timestep,
    sim_report_interval=0.01 * unit.picosecond,
    sim_initialization_time=6.0 * unit.picosecond, #5 picoseconds used in article, but 6 is just a bit more stable so fewer retries are needed
    sim_initialization_iterations=10,
    sim_equilibration_time=0.15 * unit.picosecond,
    sim_equilibration_iterations=30,
    equilibrate=True,
)

svgd_config = SVGDConfig(
    state_directory=f"{output_dir}/svgd_states",
    checkpoint_file=f"{output_dir}/svgd_checkpoint.chk",
    n_particles=200,
    n_iterations=150,
    svgd_step_size=0.6,
    svgd_temp=10000.0,
)

svgd_engine = SVGDEngine(svgd_config, cv_defs, OMMSampler, sim_config, parsl_config, restart=False)
svgd_engine.run()

sim_config = OMMSamplerProperties(
    cv_defs=cv_defs,
    force_constant=2500,
    integrator=integrator,
    system=system,
    topology=pdb.getTopology(),
    initial_positions=pdb.getPositions(),
    platform_name=platform_name,
    platform_properties=platform_properties,
    output_directory=output_dir,
    '''Increase sampling time to 2ns for mean forces'''
    sim_time=2 * unit.nanosecond,
    sim_timestep=timestep,
    sim_report_interval=0.01 * unit.picosecond,
    sim_initialization_time=6.0 * unit.picosecond,
    sim_initialization_iterations=10,
    sim_equilibration_time=0.1 * unit.picosecond,
    sim_equilibration_iterations=20,
    '''Turn off equilibration since MD systems are already at correct position'''
    equilibrate=False,
)

svgd_engine.sample_at_state(sim_config, iteration=-1)
print("Run Done")

svgd_state = svgd_engine.get_state(iteration=-1)
mbf = FreeEnergyProfileMBF(svgd_state.particles.to(device), cv_defs)
mbf.fit(svgd_state.mean_forces.to(device),torch.tensor([1.0]*len(cv_defs),dtype=dtype,device=device), n_iter=300, lr=5e-3)

grid_density = 50
'''Depending on the memory available, batch_size may have to be reduced'''
PMF, bounds = mbf.marginalize_PMF(grid_density=grid_density,marginal=(0,1),batch_size=1_000_000)
plt.imshow(torch.flipud(PMF.reshape([grid_density]*2).T).cpu(), extent = (bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]),cmap="Spectral_r")
plt.xlabel(cv_defs[0].name)
plt.ylabel(cv_defs[1].name)
plt.colorbar()
plt.savefig("./test_marginal_2D_pmf.pdf")
plt.close()

PMF, bounds = mbf.marginalize_PMF(grid_density=grid_density,marginal=(0,),batch_size=1_000_000)
plt.plot(torch.linspace(bounds[0][0],bounds[0][1],grid_density).cpu(),PMF.cpu())
plt.xlabel(cv_defs[0].name)
plt.ylabel("Free Energy")
plt.savefig("./test_marginal_1D_pmf.pdf")
plt.close()
