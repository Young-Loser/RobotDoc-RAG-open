from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable


def run_script(script_path: Path, project_root: Path) -> int:
    cmd = [sys.executable, str(script_path)]
    completed = subprocess.run(cmd, cwd=project_root)
    return completed.returncode


def run_pipeline(script_paths: Iterable[Path], project_root: Path, stop_on_error: bool = True) -> int:
    last_code = 0
    for script_path in script_paths:
        print(f"[pipeline] running: {script_path.relative_to(project_root)}")
        code = run_script(script_path, project_root)
        last_code = code
        if code != 0:
            print(f"[pipeline] failed with exit code {code}: {script_path.relative_to(project_root)}")
            if stop_on_error:
                return code
    return last_code
