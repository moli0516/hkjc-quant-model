import importlib
import inspect
import os
import pkgutil

EXECUTION_ORDER = {
    "RatingClassGenerator": 10,
    "HorseProfileGenerator": 20,
    "SectionalSpeedGenerator": 30,
    "PaceStrategyGenerator": 40,
    "HorseRollingGenerator": 50,
    "HumanSireGenerator": 60,
    "SynergyFitnessGenerator": 70,
    "TrackDistanceGenerator": 80,
    "ContextRelativeGenerator": 90,
    "OddsMarketGenerator": 100,
    "InjuryRestGenerator": 110,
    # 交叉特徵 Generator 必須排在最後面！
    "InteractionGenerator": 999,
}


def load_all_generators(key_cols: list[str] = None):
    key_cols = key_cols or ["race_id", "horse_id"]
    generator_instances = []

    pkg_dir = os.path.dirname(__file__)
    pkg_name = __name__

    for _, module_name, is_pkg in pkgutil.iter_modules([pkg_dir]):
        if module_name.startswith("_") or is_pkg:
            continue

        full_module_name = f"{pkg_name}.{module_name}"
        module = importlib.import_module(full_module_name)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__ == full_module_name
                and name.endswith("Generator")
            ):
                generator_instances.append(obj(key_cols=key_cols))

    generator_instances.sort(
        key=lambda gen: EXECUTION_ORDER.get(gen.__class__.__name__, 500)
    )

    return generator_instances