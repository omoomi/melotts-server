from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(slots=True)
class ExternalModelModules:
    utils: ModuleType


def load_utils_module(model_dir: Path) -> ExternalModelModules:
    model_dir = model_dir.resolve()
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    spec = importlib.util.spec_from_file_location("melotts_rknn_utils", model_dir / "utils.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load utils.py from {model_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return ExternalModelModules(utils=module)

