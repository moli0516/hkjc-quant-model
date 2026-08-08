import importlib
import pkgutil
import sys
from typing import List

from database.models.base import Base
from database.models.race import Race
from database.models.result import RaceResult
from database.models.sectional import RaceSectional
from database.models.horse import Horse
from database.models.trail import RaceTrial
from database.models.trackwork import RaceTrackwork

__all__: List[str] = [
    "Base",
    "Race",
    "RaceResult",
    "RaceSectional",
    "Horse",
    "RaceTrial",
    "RaceTrackwork",
]


def reload_models() -> None:
    """動態熱加載 database/models 目錄下的所有模型模組"""
    pkg_name = __name__
    pkg_dir = __path__

    for _, module_name, _ in pkgutil.iter_modules(pkg_dir):
        full_module_name = f"{pkg_name}.{module_name}"
        if full_module_name in sys.modules:
            importlib.reload(sys.modules[full_module_name])
