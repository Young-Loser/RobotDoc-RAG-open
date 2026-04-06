from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from robotdoc_rag.config import load_config
from robotdoc_rag.paths import ProjectPaths, get_project_root
from robotdoc_rag.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run configured robotdoc-rag pipelines.")
    parser.add_argument(
        "pipeline",
        nargs="?",
        default="data_rebuild",
        help="Pipeline name defined in configs/default.json",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a JSON config file.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available pipelines and exit.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running later steps even if a previous step fails.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = get_project_root()
    project_paths = ProjectPaths.from_root(project_root)
    project_paths.ensure_core_dirs()

    pipelines = config.get("pipelines", {})

    if args.list:
        print("Available pipelines:")
        for name, steps in pipelines.items():
            print(f"- {name}")
            for step in steps:
                print(f"  - {step}")
        return 0

    if args.pipeline not in pipelines:
        parser.error(f"Unknown pipeline: {args.pipeline}")

    script_paths = [project_root / step for step in pipelines[args.pipeline]]
    return run_pipeline(script_paths, project_root, stop_on_error=not args.keep_going)


if __name__ == "__main__":
    raise SystemExit(main())
