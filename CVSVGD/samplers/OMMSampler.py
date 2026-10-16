import signal
from dataclasses import dataclass
from typing import TextIO

import openmm as mm
from openmm import app, unit
import math

from CVSVGD.base_types.cv_def import CVDef
from CVSVGD.base_types.sampler import (
    Sampler,
    SamplerProperties,
    SimulationJob,
    SimulationResult,
)

GAS_CONSTANT = (
    1.0 * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
).in_units_of(unit.kilojoule_per_mole / unit.kelvin)

@dataclass(kw_only=True)
class OMMCVDef(CVDef):
    omm_force: mm.Force
    def __post_init__(self):
        super().__post_init__()
        if self.restraint_type != "harmonic":
            raise NotImplementedError("OMMSampler only supports harmonic restraints.")

class OMMSamplerProperties(SamplerProperties):
    def __init__(
        self,
        cv_defs: list[OMMCVDef],
        force_constant: float,

        integrator: mm.Integrator,
        system: mm.System,
        platform_name: str,
        topology: app.Topology,
        initial_positions: unit.Quantity,
        output_directory: str,
        platform_properties: dict|None = None,

        sim_temperature: unit.Quantity = 300 * unit.kelvin,
        sim_time: unit.Quantity = 50 * unit.picosecond,
        sim_timestep: unit.Quantity = 2 * unit.femtosecond,
        sim_report_interval: unit.Quantity = 0.1 * unit.picosecond,
        equilibrate = True,
        sim_equilibration_iterations: int = 10,
        sim_equilibration_time: unit.Quantity = 1 * unit.picosecond,

        initialize = True,
        sim_initialization_iterations: int = 10,
        sim_initialization_time: unit.Quantity = 1 * unit.picosecond,
    ) -> None:
        if platform_properties is None:
            platform_properties = {"Precision": "mixed"}

        self.cv_data:list[dict] = []
        for cvd in cv_defs:
            cv_dict = {}
            cv_dict['name'] = cvd.name
            cv_dict['force_xml'] = mm.XmlSerializer.serialize(cvd.omm_force)
            cv_dict['length_scale'] = cvd.length_scale if cvd.length_scale is not None else 1.0
            cv_dict['period'] = cvd.period if cvd.periodic else math.inf
            if cvd.restraint_type != 'harmonic':
                raise NotImplementedError("Non-harmonic restriants not implemented for OMMSampler.")
            else:
                if cvd.periodic:
                    cv_dict['restr_str'] = "((({cv}-{cv0})/{cv_lenscl}) - ({period}/{cv_lenscl})*floor(((({cv}-{cv0})/{cv_lenscl})/({period}/{cv_lenscl}))+0.5))^2"
                else:
                    cv_dict['restr_str'] = "(({cv}-{cv0})/{cv_lenscl})^2"
                cv_dict['restr_type'] = cvd.restraint_type
            self.cv_data.append(cv_dict)

        self.output_directory = output_directory
        self.force_constant:float = force_constant
        self.temperature: unit.Quantity = sim_temperature
        self.topology: app.Topology = topology
        self.system_xml: str = mm.XmlSerializer.serialize(system)
        self.integrator_xml: str = mm.XmlSerializer.serialize(integrator)
        self.platform_name: str = platform_name
        self.platform_properties: dict = platform_properties
        self.initial_positions: unit.Quantity = initial_positions

        self.timestep: unit.Quantity = sim_timestep
        self.time: unit.Quantity = sim_time
        self.report_interval: int = int(
            sim_report_interval.value_in_unit(unit.femtosecond)
            / self.timestep.value_in_unit(unit.femtosecond)
        )

        self.steps: int = self.time.value_in_unit(
            unit.femtosecond
        ) / self.timestep.value_in_unit(unit.femtosecond)
        self.n_simulation_samples: int = int(self.steps / self.report_interval)

        self.initialization_steps: int = int(
            sim_initialization_time.value_in_unit(unit.femtosecond)
            / self.timestep.value_in_unit(unit.femtosecond))
        self.initialization_iterations: int = sim_initialization_iterations

        self.equilibration_steps: int = int(
            sim_equilibration_time.value_in_unit(unit.femtosecond)
            / self.timestep.value_in_unit(unit.femtosecond))
        self.equilibration_iterations: int = sim_equilibration_iterations

        if self.report_interval > self.steps:
            raise ValueError(
                f"report_interval ({self.report_interval} steps) exceeds total steps "
                f"({self.steps}); no data would be reported."
            )

        self.sim_RT = (self.temperature * GAS_CONSTANT).value_in_unit(
            unit.kilojoule_per_mole
        )
        self.initialize = initialize
        self.equilibrate = equilibrate

    def get_restraint_force(self,centers:tuple[float,...]):
        energy_expr = "0.5*k_CV*("
        for cvd in self.cv_data:
            if cvd['period'] < float('inf'):
                energy_expr += (
                    "(" + cvd['restr_str'].format(cv=cvd['name'], cv0=f"{cvd['name']}_0", cv_lenscl=f"{cvd['length_scale']}", period=f"{cvd['period']}") + ")"
                )
            else:
                energy_expr += (
                    "(" + cvd['restr_str'].format(cv=cvd['name'], cv0=f"{cvd['name']}_0", cv_lenscl=f"{cvd['length_scale']}") + ")"
                )
            energy_expr += "+"
        energy_expr = energy_expr.rstrip("+") + ")"
        cv_force = mm.CustomCVForce(energy_expr)
        for i, cvd in enumerate(self.cv_data):
            myforce = mm.XmlSerializer.deserialize(cvd['force_xml'])
            cv_force.addCollectiveVariable(cvd['name'], myforce)
            cv_force.addGlobalParameter(f"{cvd['name']}_0", centers[i])
        cv_force.addGlobalParameter("k_CV", self.force_constant)
        return cv_force

    def update_restraint_force_centers(self,centers:tuple[float,...], context: mm.Context):
        for i, cvd in enumerate(self.cv_data):
            context.setParameter(f"{cvd['name']}_0", centers[i])
    def calc_mean_force(self,centers:tuple[float,...], samples:dict[str,list[float]]) -> dict[str,float]:
        mean_force:dict[str,float] = {}
        for i, cvd in enumerate(self.cv_data):
            if cvd['restr_type'] == 'harmonic':
                lenscl = cvd['length_scale']
                diffs = [cv - centers[i] for cv in samples[cvd['name']]]
                if cvd['period'] < float('inf'):
                    p = cvd['period']
                    diffs = [d - p * math.floor(d / p + 0.5) for d in diffs]
                mean_force[cvd['name']] = - self.force_constant * sum(diffs) / (lenscl**2 * len(diffs))
            else:
                raise NotImplementedError
        return mean_force

    def wrap_cv_values(self,cv_vals:tuple[float,...]) -> tuple[float,...]:
        cv_vals_wrapped:list[float] = []
        for i,cvd in enumerate(self.cv_data):
            if cvd['period'] < float('inf'):
                p = cvd['period']
                new_val = cv_vals[i] - p * math.floor(cv_vals[i] / p + 0.5)
                cv_vals_wrapped.append(new_val)
            else:
                cv_vals_wrapped.append(cv_vals[i])
        return tuple(cv_vals_wrapped)


class OMMSampler(Sampler[OMMSamplerProperties]):
    @staticmethod
    def initialize(job:SimulationJob, samp_props:OMMSamplerProperties):
        import math
        import os

        import openmm as mm
        from openmm import app, unit

        pos_output_dir = f"{samp_props.output_directory}/sim_state"
        os.makedirs(pos_output_dir, exist_ok=True)
        target = samp_props.wrap_cv_values(job.center)

        cv_force = samp_props.get_restraint_force(target)
        system = mm.XmlSerializer.deserialize(samp_props.system_xml)
        integrator = mm.XmlSerializer.deserialize(samp_props.integrator_xml)
        system.addForce(cv_force)
        platform = mm.Platform.getPlatformByName(samp_props.platform_name)

        simulation = app.Simulation(
            samp_props.topology, system, integrator, platform, samp_props.platform_properties
        )

        init_state_file = f"{pos_output_dir}/particle_{job.particle_id}_state.xml"
        init_pos = samp_props.initial_positions
        simulation.context.setPositions(init_pos)
        init_cvs = cv_force.getCollectiveVariableValues(simulation.context)
        samp_props.update_restraint_force_centers(init_cvs, simulation.context)
        simulation.context.setParameter("k_CV", samp_props.force_constant)
        simulation.minimizeEnergy(
            tolerance=1e-6 * unit.kilojoule_per_mole / unit.nanometers, maxIterations=0
        )
        simulation.context.setVelocitiesToTemperature(
            samp_props.temperature.value_in_unit(unit.kelvin)
        )
        delta_cv = tuple(target[i] - init_cvs[i] for i in range(len(init_cvs)))
        delta_cv = samp_props.wrap_cv_values(delta_cv)
        initialization_steps_per_iteration = max(1,int(samp_props.initialization_steps/samp_props.initialization_iterations))
        for i in range(1, samp_props.initialization_iterations + 1):
            center_i = tuple(init_cvs[j] + (i / samp_props.initialization_iterations) * delta_cv[j] for j in range(len(init_cvs)))
            samp_props.update_restraint_force_centers(center_i, simulation.context)
            simulation.step(initialization_steps_per_iteration)
        samp_props.update_restraint_force_centers(target, simulation.context)
        simulation.saveState(init_state_file)

    @staticmethod
    def simulate(job: SimulationJob, samp_props:OMMSamplerProperties) -> SimulationResult:
        import os
        import json
        import math
        from pathlib import Path

        import openmm as mm
        from openmm import app, unit

        dcd_output_dir = f"{samp_props.output_directory}/traj"
        log_output_dir = f"{samp_props.output_directory}/logs"
        chk_output_dir = f"{samp_props.output_directory}/chks"
        cvs_output_dir = f"{samp_props.output_directory}/cvs"
        pos_output_dir = f"{samp_props.output_directory}/sim_state"

        os.makedirs(dcd_output_dir, exist_ok=True)
        os.makedirs(log_output_dir, exist_ok=True)
        os.makedirs(chk_output_dir, exist_ok=True)
        os.makedirs(cvs_output_dir, exist_ok=True)
        os.makedirs(pos_output_dir, exist_ok=True)

        target = samp_props.wrap_cv_values(job.center)
        cv_force = samp_props.get_restraint_force(target)
        system = mm.XmlSerializer.deserialize(samp_props.system_xml)
        integrator = mm.XmlSerializer.deserialize(samp_props.integrator_xml)
        system.addForce(cv_force)
        platform = mm.Platform.getPlatformByName(samp_props.platform_name)

        simulation = app.Simulation(
            samp_props.topology, system, integrator, platform, samp_props.platform_properties
        )
        dcd_file = f"{dcd_output_dir}/particle_{job.particle_id}_output.dcd"
        cv_file = f"{cvs_output_dir}/particle_{job.particle_id}_cv.csv"
        state_file = f"{log_output_dir}/particle_{job.particle_id}_state.csv"
        state_chckpoint_file = (
            f"{chk_output_dir}/particle_{job.particle_id}_iteration_{job.iteration}_state_checkpoint.xml"
        )
        meta_file = f"{chk_output_dir}/particle_{job.particle_id}_iteration_{job.iteration}_state_metadata.json"
        init_state_file = f"{pos_output_dir}/particle_{job.particle_id}_state.xml"

        done = 0
        resuming = False

        if Path(state_chckpoint_file).is_file() and Path(meta_file).is_file():
            meta = json.loads(Path(meta_file).read_text())
            resuming = (meta["iteration"] == job.iteration and
                all(math.isclose(a, b, abs_tol=1e-8) for a, b in zip(meta["center"], target)))
        if not resuming:
            """Clear previous files (except init_pos) to ensure no overwrite issues"""
            Path(dcd_file).unlink(missing_ok=True)
            Path(cv_file).unlink(missing_ok=True)
            Path(state_file).unlink(missing_ok=True)
            Path(state_chckpoint_file).unlink(missing_ok=True)
            Path(meta_file).write_text(json.dumps({"iteration": job.iteration, "center": list(target)}))

            if Path(init_state_file).is_file():
                simulation.loadState(init_state_file)
            else:
                simulation.context.setPositions(samp_props.initial_positions)
                init_cvs = cv_force.getCollectiveVariableValues(simulation.context)
                samp_props.update_restraint_force_centers(init_cvs, simulation.context)
                simulation.context.setParameter("k_CV", samp_props.force_constant)
                simulation.minimizeEnergy(
                    tolerance=1 * unit.kilojoule_per_mole / unit.nanometers, maxIterations=5000
                )

            simulation.context.setVelocitiesToTemperature(
                samp_props.temperature.value_in_unit(unit.kelvin)
            )
            if samp_props.equilibrate:
                init_cvs = cv_force.getCollectiveVariableValues(simulation.context)
                samp_props.update_restraint_force_centers(init_cvs, simulation.context)
                delta_cv = tuple(target[i] - init_cvs[i] for i in range(len(init_cvs)))
                delta_cv = samp_props.wrap_cv_values(delta_cv)
                simulation.context.setParameter("k_CV", samp_props.force_constant)
                equilibration_steps_per_iteration = max(1,int(samp_props.equilibration_steps/samp_props.equilibration_iterations))
                for i in range(1, samp_props.equilibration_iterations + 1):
                    center_i = tuple(init_cvs[j] + (i / samp_props.equilibration_iterations) * delta_cv[j] for j in range(len(init_cvs)))
                    samp_props.update_restraint_force_centers(center_i, simulation.context)
                    simulation.step(equilibration_steps_per_iteration)
                samp_props.update_restraint_force_centers(target, simulation.context)
                simulation.currentStep = 0
                done = 0
        else:
            simulation.loadState(state_chckpoint_file)
            samp_props.update_restraint_force_centers(target, simulation.context)
            done = simulation.currentStep


        dcd_reporter = app.DCDReporter(
            dcd_file, samp_props.report_interval, enforcePeriodicBox=False, append=resuming
        )
        state_reporter = app.StateDataReporter(
            state_file,
            samp_props.report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            speed=True,
            append=resuming,
        )
        state_checkpoint_reporter = app.CheckpointReporter(
            state_chckpoint_file, samp_props.report_interval, writeState=True
        )
        cv_reporter = OMMCVReporter(cv_file, samp_props.report_interval, cv_force, append=resuming, resume_step=done if resuming else None)

        simulation.reporters.append(dcd_reporter)
        simulation.reporters.append(state_reporter)
        simulation.reporters.append(cv_reporter)
        simulation.reporters.append(state_checkpoint_reporter)
        try:
            simulation.step(samp_props.steps - done)
        except Exception:
            Path(state_chckpoint_file).unlink(missing_ok=True)
            Path(meta_file).unlink(missing_ok=True)
            raise
        finally:
            cv_reporter.close()
        tmp = init_state_file + ".tmp"
        simulation.saveState(tmp)
        os.replace(tmp, init_state_file)
        Path(state_chckpoint_file).unlink() #Erase checkpoint and metadata since job is done
        Path(meta_file).unlink(missing_ok=True)
        samples = cv_reporter.cv_dict
        mean_force = samp_props.calc_mean_force(target,samples)
        return SimulationResult(particle_id=job.particle_id,center=job.center,iteration=job.iteration,samples=samples,mean_force=mean_force)

class OMMCVReporter:
    def __init__(
        self,
        file: str,
        reportInterval: int,
        cv_force: mm.CustomCVForce,
        append: bool = False,
        resume_step: int | None = None,
    ):
        import os
        self._reportInterval: int = reportInterval
        self._cv_force: mm.CustomCVForce = cv_force
        n_cv: int = cv_force.getNumCollectiveVariables()
        self._names: list[str] = [cv_force.getCollectiveVariableName(i) for i in range(n_cv)]
        self._rows: dict[int, tuple[float, ...]] = {}

        header = "#\"Step\"" + "".join(f",\"{n}\"" for n in self._names) + "\n"
        if append and os.path.isfile(file):
            with open(file, "r") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    tok = line.strip().split(",")
                    if len(tok) != n_cv + 1:
                        continue
                    try:
                        step = int(tok[0])
                        vals = tuple(float(v) for v in tok[1:])
                    except ValueError:
                        continue
                    if resume_step is not None and step > resume_step:
                        continue
                    self._rows[step] = vals
            # Rewrite the file so it matches exactly what we kept.
            tmp = file + ".tmp"
            with open(tmp, "w") as f:
                f.write(header)
                for step in sorted(self._rows):
                    f.write(self._format_row(step, self._rows[step]))
            os.replace(tmp, file)
            self._out: TextIO = open(file, "a")
        else:
            self._out = open(file, "w")
            self._out.write(header)
            self._out.flush()

    @staticmethod
    def _format_row(step: int, vals) -> str:
        return "%d" % step + "".join(",%.10g" % v for v in vals) + "\n"

    @property
    def cv_dict(self) -> dict[str, list[float]]:
        steps = sorted(self._rows)
        return {n: [self._rows[s][i] for s in steps] for i, n in enumerate(self._names)}

    @property
    def n_samples(self) -> int:
        return len(self._rows)

    def close(self):
        if not self._out.closed:
            self._out.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return {"steps": steps, "periodic": None, "include": ["positions"]}

    def report(self, simulation, state):
        step = simulation.currentStep
        vals = tuple(self._cv_force.getCollectiveVariableValues(simulation.context))
        self._rows[step] = vals
        self._out.write(self._format_row(step, vals))
        self._out.flush()

#class OMMCVReporter:
#    def __init__(
#        self,
#        file: str,
#        reportInterval: int,
#        cv_force: mm.CustomCVForce,
#        append: bool = False,
#    ):
#        self._reportInterval: int = reportInterval
#        self._cv_force: mm.CustomCVForce = cv_force
#        n_cv: int = cv_force.getNumCollectiveVariables()
#        self._cv_dict: dict[str,list[float]] = {
#            cv_force.getCollectiveVariableName(i): [] for i in range(n_cv)
#        }
#        if append:
#            f = open(file, "r")
#            for line in f:
#                if line.startswith("#"):
#                    continue
#                tok = line.strip().split(",")
#                cvs = tok[1:]
#                for k,val in zip(self._cv_dict.keys(),cvs):
#                    self._cv_dict[k].append(float(val))
#            f.close()
#
#        self._out: TextIO = open(file, "a" if append else "w")
#        if not append:
#            self._out.write("#\"Step\"")
#            for cv_name in self._cv_dict.keys():
#                self._out.write(f",\"{cv_name}\"")
#            self._out.write("\n")
#            self._out.flush()
#
#        signal.signal(signal.SIGTERM, self.terminate_handler)
#        signal.signal(signal.SIGINT, self.terminate_handler)
#
#    def __del__(self):
#        self._out.close()
#
#    def describeNextReport(self, simulation):
#        steps = self._reportInterval - simulation.currentStep % self._reportInterval
#        return {"steps": steps, "periodic": None, "include": ["positions"]}
#
#    def report(self, simulation, state):
#        step = simulation.currentStep
#        cvs = self._cv_force.getCollectiveVariableValues(simulation.context)
#        self._out.write("%d" % (step))
#        for i, cv in enumerate(cvs):
#            self._out.write(",%g" % (cv))
#            self._cv_dict[self._cv_force.getCollectiveVariableName(i)].append(cv)
#        self._out.write("\n")
#        self._out.flush()
#
#    def terminate_handler(self, signal, frame):
#        self._out.flush()
