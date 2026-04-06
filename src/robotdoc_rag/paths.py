from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    raw_pdfs_dir: Path
    pages_dir: Path
    ocr_dir: Path
    eval_dir: Path
    train_dir: Path
    outputs_dir: Path
    docs_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        return cls(
            root=root,
            data_dir=root / "data",
            raw_pdfs_dir=root / "data" / "raw_pdfs",
            pages_dir=root / "data" / "pages",
            ocr_dir=root / "data" / "ocr",
            eval_dir=root / "data" / "eval",
            train_dir=root / "data" / "train",
            outputs_dir=root / "outputs",
            docs_dir=root / "docs",
        )

    def ensure_core_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.raw_pdfs_dir,
            self.pages_dir,
            self.ocr_dir,
            self.eval_dir,
            self.train_dir,
            self.outputs_dir,
            self.docs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
