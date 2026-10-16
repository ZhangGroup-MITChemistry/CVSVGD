import os
import sys
import types
import warnings
from pathlib import Path

_REGISTERED: set[str] = set()


class CVSVGDWorkerEnvWarning(UserWarning):
    """Workers may not be able to import CVSVGD or user sampler code."""


def parsl_is_loaded() -> bool:
    if "parsl" not in sys.modules:
        return False
    import parsl
    try:
        parsl.dfk()
    except Exception:
        return False
    return True

def _import_root(mod: types.ModuleType) -> Path | None:
    file = getattr(mod, "__file__", None)
    if file is None:
        return None
    path = Path(file).resolve()
    depth = mod.__name__.count(".")
    if path.name == "__init__.py":
        depth += 1
    root = path.parent
    for _ in range(depth):
        root = root.parent
    return root

def ensure_worker_importable(*objs, stacklevel: int = 2) -> list[str]:
    """
    Make the modules defining `objs` (classes, functions or modules) importable
    in this process and in any worker process spawned later.

    Must run before Parsl launches workers, i.e. before parsl.load(), to cover
    the initial blocks.
    """
    roots: list[str] = []
    for obj in objs:
        mod = obj if isinstance(obj, types.ModuleType) else sys.modules.get(obj.__module__)
        if mod is None:
            continue
        if mod.__name__ == "__main__":
            main_file = getattr(mod, "__file__", None)
            root = Path(main_file).resolve().parent if main_file else None
        else:
            root = _import_root(mod)
        if root is not None:
            roots.append(str(root))

    pythonpath = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    new_for_workers = [r for r in roots if r not in pythonpath]

    for r in roots:
        if r not in sys.path:
            sys.path.insert(0, r)
        if r not in pythonpath:
            pythonpath.insert(0, r)
        _REGISTERED.add(r)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)

    if new_for_workers and parsl_is_loaded():
        warnings.warn(
            "Parsl was loaded before CVSVGD could configure the worker environment. "
            "Custom samplers (and other code outside an installed package) may fail "
            "to import on workers that were already started. To fix this, allow CVSVGD to "
            "handle parsl.load() or add "
            f"the following to your provider's worker_init:\n    {worker_init_snippet()}",
            CVSVGDWorkerEnvWarning,
            stacklevel=stacklevel,
        )
    return roots

def worker_init_snippet() -> str:
    paths = os.pathsep.join(sorted(_REGISTERED))
    return f'export PYTHONPATH="{paths}${{PYTHONPATH:+:$PYTHONPATH}}"'
