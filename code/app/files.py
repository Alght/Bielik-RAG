from pathlib import Path


def _discover_files():
    app_dir = Path(__file__).resolve().parent
    repo_root = app_dir.parent

    candidates = [
        Path("/original_documents"),
        Path("/text_documents"),
        repo_root / "original_documents",
        repo_root / "text_documents",
        app_dir / "original_documents",
        app_dir / "text_documents",
    ]

    original_dir = next(
        (
            p
            for p in candidates
            if p.exists() and p.is_dir() and "original" in p.name.lower()
        ),
        None,
    )
    text_dir = next(
        (
            p
            for p in candidates
            if p.exists() and p.is_dir() and "text" in p.name.lower()
        ),
        None,
    )

    if original_dir is None or text_dir is None:
        raise FileNotFoundError(
            "Required corpus directories not found. Expected original_documents and text_documents folders."
        )

    target_pdf = original_dir / "KodeksKarny.pdf"
    target_txt = text_dir / "KodeksKarny.txt"

    if not target_pdf.exists():
        raise FileNotFoundError(f"Required Kodeks Karny PDF not found: {target_pdf}")
    if not target_txt.exists():
        raise FileNotFoundError(f"Required Kodeks Karny TXT not found: {target_txt}")

    return {
        "kodeks_karny": {
            "pdf_file": str(target_pdf),
            "txt_file": str(target_txt),
        }
    }


files_dir = _discover_files()
