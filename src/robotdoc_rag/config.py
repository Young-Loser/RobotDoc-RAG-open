from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import get_project_root


DEFAULT_CONFIG_PATH = get_project_root() / "configs" / "default.json"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
