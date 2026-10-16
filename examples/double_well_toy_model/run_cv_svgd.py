import openmm as mm
import torch
from parsl.config import Config
import matplotlib.pyplot as plt

from CVSVGD.base_types.cv_def import CVDef
from CVSVGD.base_types.sampler import (
    Sampler,
    SamplerProperties,
    SimulationJob,
    SimulationResult,
)
from CVSVGD.svgd_config import SVGDConfig
from CVSVGD.svgd_engine import SVGDEngine
from CVSVGD.analysis.fep_mbf import FreeEnergyProfileMBF

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAS_CONSTANT = (1.0 * mm.unit.BOLTZMANN_CONSTANT_kB * mm.unit.AVOGADRO_CONSTANT_NA).value_in_unit(mm.unit.kilojoule_per_mole / mm.unit.kelvin)

centers = torch.ones([3], dtype=dtype, device=device)
centers = torch.stack([centers, -1 * centers]).T
g_weights = torch.tensor([0.5, 0.5], dtype=dtype, device=device)
k_vals = torch.tensor([2, 2], dtype=dtype, device=device)

class DWMeanForcesProp(SamplerProperties):
    def __init__(self, centers:torch.Tensor, k_vals:torch.Tensor, g_weights:torch.Tensor) -> None:
        self.initialize:bool = False
        self.centers = centers.cpu().tolist()
        self.k = k_vals.cpu().tolist()
        self.w = g_weights.cpu().tolist()
        self.n_dim, self.n_centers = centers.shape

    def logP_job(self, values:torch.Tensor, k:torch.Tensor, dtype:torch.dtype, device:torch.device, dims:list[int]|None=None):
        centers = torch.tensor(self.centers,dtype=dtype,device=device)
        w = torch.tensor(self.w,dtype=dtype,device=device)
        delta = values[:,None] - centers[dims,:]
        sq_dist = (delta**2).sum(dim=0)
        n_integrated = self.n_dim - (len(dims) if dims is not None else 0)
        log_w = (
            -(1/2) * k * sq_dist
            + (n_integrated/2) * torch.log(2*torch.pi/k)
            + (self.n_dim/2)  * torch.log(k/2*torch.pi)
            + torch.log(w)
        )
        return log_w,delta

    def mean_force_job(self, values, dtype:torch.dtype, device:torch.device, dims:list[int]|None=None):
        k = torch.tensor(self.k,dtype=dtype,device=device)
        log_w,delta = self.logP_job(values,k,dtype,device,dims)
        resp = torch.softmax(log_w, dim=-1)
        force = (k * delta * resp[None, :]).sum(dim=-1)
        return force

class DWMeanForces(Sampler[DWMeanForcesProp]):
    @staticmethod
    def initialize(job: SimulationJob, samp_props: DWMeanForcesProp) -> None:
        return None
    @staticmethod
    def simulate(job: SimulationJob, samp_props: DWMeanForcesProp) -> SimulationResult:
        import torch
        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        value = torch.tensor(job.center,dtype=dtype,device=device)
        mean_force = samp_props.mean_force_job(value,dtype,device,[0,1])
        mean_force = {name:mf for name,mf in zip(job.cv_names,mean_force.cpu().tolist())}
        samples = {name:[v] for name,v in zip(job.cv_names,job.center)}
        return SimulationResult(particle_id=job.particle_id,center=job.center,mean_force=mean_force,samples=samples,iteration=job.iteration)


svgd_config = SVGDConfig(
    state_directory="low_barrier/svgd_states",
    checkpoint_file="low_barrier/svgd_checkpoint.chk",
    n_particles=1000,
    n_iterations=150,
    svgd_step_size=0.6,
    svgd_temp=(1/GAS_CONSTANT),
)

sim_prop = DWMeanForcesProp(centers,k_vals,g_weights)
parsl_config = Config()

cv_defs: list[CVDef] = [
    CVDef(name="x", prior_bounds=(-5, 5)),
    CVDef(name="y", prior_bounds=(-5, 5)),
]

'''Low Barrier Model'''
svgd_engine = SVGDEngine(svgd_config,cv_defs,DWMeanForces,sim_prop,parsl_config,restart=False)
svgd_engine.run()
lb_svgd_state = svgd_engine.get_state()

mbf = FreeEnergyProfileMBF(lb_svgd_state.particles.to(device), cv_defs)
mbf.fit(lb_svgd_state.mean_forces.to(device),torch.tensor([1.0,1.0],dtype=dtype,device=device), n_iter=300, lr=1e-1, reg=1e-8)

grid_density = 100
eval_grid = torch.stack(torch.meshgrid([torch.linspace(-5,5,grid_density,device=device)] * 2, indexing='ij'),dim=0).reshape([2,-1])
PMF = mbf.evaluate(eval_grid).cpu()
PMF = PMF - PMF.min()

plt.imshow(torch.flipud(PMF.reshape([grid_density]*2).T), extent = (-5,5,-5,5),cmap="Spectral_r")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()
plt.savefig("./low_barrier_pmf.pdf")
plt.close()

xy = lb_svgd_state.particles.cpu()
plt.scatter(xy[0],xy[1])
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("./low_barrier_scatter.pdf")
plt.close()
svgd_engine.shutdown()

'''High Energy Barrier Model'''

svgd_config = SVGDConfig(
    state_directory="high_barrier/svgd_states",
    checkpoint_file="high_barrier/svgd_checkpoint.chk",
    n_particles=1000,
    n_iterations=150,
    svgd_step_size=0.6,
    svgd_temp=5000.0,
)
k_vals = torch.tensor([50, 25], dtype=dtype, device=device)
sim_prop = DWMeanForcesProp(centers,k_vals,g_weights)
svgd_engine = SVGDEngine(svgd_config,cv_defs,DWMeanForces,sim_prop,parsl_config,restart=False)
svgd_engine.run()

hb_svgd_state = svgd_engine.get_state()
mbf = FreeEnergyProfileMBF(hb_svgd_state.particles.to(device), cv_defs)
mbf.fit(hb_svgd_state.mean_forces.to(device),torch.tensor([1.0,1.0],dtype=dtype,device=device), n_iter=300, lr=1e-1, reg=1e-8)

grid_density = 100
eval_grid = torch.stack(torch.meshgrid([torch.linspace(-5,5,grid_density,device=device)] * 2, indexing='ij'),dim=0).reshape([2,-1])
PMF = mbf.evaluate(eval_grid).cpu()
PMF = PMF - PMF.min()

plt.imshow(torch.flipud(PMF.reshape([grid_density]*2).T), extent = (-5,5,-5,5),cmap="Spectral_r",vmax=200)
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()
plt.savefig("./high_barrier_pmf.pdf")
plt.close()

xy = hb_svgd_state.particles.cpu()
plt.scatter(xy[0],xy[1])
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("./high_barrier_scatter.pdf")
plt.close()
