from __future__ import annotations

import argparse
from pathlib import Path

import fitz
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "data" / "raw_pdfs"
PAGE_DIR = ROOT / "data" / "pages"
MANIFEST = ROOT / "data" / "manifest.csv"
DOC_MANIFEST = ROOT / "data" / "documents.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render PDF files into page images.")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR, help="Directory containing input PDF files.")
    parser.add_argument("--page-dir", type=Path, default=PAGE_DIR, help="Directory for rendered page images.")
    parser.add_argument("--manifest-file", type=Path, default=MANIFEST, help="Output page manifest CSV.")
    parser.add_argument("--doc-manifest-file", type=Path, default=DOC_MANIFEST, help="Output document manifest CSV.")
    parser.add_argument("--dpi", type=int, default=160, help="Rendering DPI.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N PDFs after sorting.")
    parser.add_argument("--doc-id", default=None, help="Render a single PDF whose stem matches this doc_id.")
    return parser


def render_one_pdf(pdf_path: Path, out_root: Path, dpi: int = 160):
    doc = fitz.open(pdf_path)
    rows = []
    doc_id = pdf_path.stem
    doc_out = out_root / doc_id
    doc_out.mkdir(parents=True, exist_ok=True)

    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img_name = f"page_{page_idx:04d}.png"
        img_path = doc_out / img_name
        pix.save(str(img_path))

        rows.append(
            {
                "page_id": f"{doc_id}:{page_idx:04d}",
                "doc_id": doc_id,
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path),
                "page_idx": page_idx,
                "image_name": img_name,
                "image_path": str(img_path),
                "image_rel_path": str(img_path.relative_to(ROOT)),
                "dpi": dpi,
            }
        )

    return {
        "doc_id": doc_id,
        "pdf_name": pdf_path.name,
        "pdf_path": str(pdf_path),
        "page_dir": str(doc_out),
        "page_count": len(doc),
        "dpi": dpi,
        "file_size_bytes": pdf_path.stat().st_size,
    }, rows


def main() -> None:
    args = build_parser().parse_args()

    args.page_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_file.parent.mkdir(parents=True, exist_ok=True)
    args.doc_manifest_file.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(args.pdf_dir.glob("*.pdf"))
    if args.doc_id:
        pdf_files = [p for p in pdf_files if p.stem == args.doc_id]
    if args.limit is not None:
        pdf_files = pdf_files[: args.limit]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {args.pdf_dir}")

    all_rows = []
    doc_rows = []
    skipped = []

    for pdf_path in tqdm(pdf_files, desc="Rendering PDFs"):
        try:
            if pdf_path.stat().st_size == 0:
                print(f"[SKIP] Empty file: {pdf_path.name}")
                skipped.append((pdf_path.name, "empty_file"))
                continue

            doc_row, rows = render_one_pdf(pdf_path, args.page_dir, dpi=args.dpi)
            all_rows.extend(rows)
            doc_rows.append(doc_row)

        except Exception as exc:
            print(f"[SKIP] Failed to render {pdf_path.name}: {exc}")
            skipped.append((pdf_path.name, str(exc)))

    df_pages = pd.DataFrame(all_rows).sort_values(["doc_id", "page_idx"]).reset_index(drop=True)
    df_docs = pd.DataFrame(doc_rows).sort_values(["doc_id"]).reset_index(drop=True)

    df_pages.to_csv(args.manifest_file, index=False, encoding="utf-8")
    df_docs.to_csv(args.doc_manifest_file, index=False, encoding="utf-8")

    print(f"Saved page manifest to: {args.manifest_file}")
    print(f"Saved document manifest to: {args.doc_manifest_file}")
    print(f"Total PDFs rendered: {len(df_docs)}")
    print(f"Total pages: {len(df_pages)}")

    if skipped:
        print("\nSkipped files:")
        for name, reason in skipped:
            print(f" - {name}: {reason}")


if __name__ == "__main__":
    main()
