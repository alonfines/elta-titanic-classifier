"""Fetch the Titanic `train.csv` from Kaggle and save it to the local data/ directory.

Per the assignment brief, only `train.csv` is used — `test.csv` and `gender_submission.csv`
are never downloaded. Train/validation splitting happens later, in code, from this one file.

Requires a Kaggle API token. Get one at https://www.kaggle.com/settings -> API -> "Create New
Token", which downloads a `kaggle.json`. Save it to `~/.kaggle/kaggle.json` (and `chmod 600` it),
or set the KAGGLE_USERNAME / KAGGLE_KEY environment variables instead. You must also have accepted
the competition rules at https://www.kaggle.com/competitions/titanic/rules — Kaggle returns a 403
for the download API otherwise, even with valid credentials.

Usage:
    python fetch_data.py [--data-dir data] [--force]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import zipfile
from pathlib import Path

COMPETITION = "titanic"
FILENAME = "train.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", default="data", help="Directory to save train.csv into (default: data/)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if train.csv already exists"
    )
    return parser.parse_args()


def _authenticated_api():
    """Import the kaggle package and return an authenticated API client, or exit with a clear message."""
    try:
        # kaggle's __init__ eagerly attempts authentication as a side effect of import and prints
        # its own (already fairly clear) help text if it fails; swallow that first pass so we can
        # retry explicitly below and control the error message ourselves instead of double-printing.
        with contextlib.redirect_stdout(io.StringIO()):
            import kaggle
    except ImportError as exc:
        raise SystemExit(
            "The `kaggle` package is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    api = kaggle.api
    if not getattr(api, "_authenticated", False):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                api.authenticate()
        except SystemExit as exc:
            raise SystemExit(
                "Kaggle API credentials not found.\n\n"
                "1. Log into https://www.kaggle.com/settings -> API Tokens -> 'Generate New Token' to "
                "download kaggle.json.\n"
                "2. Save it to ~/.kaggle/kaggle.json and run: chmod 600 ~/.kaggle/kaggle.json\n"
                "   (or set the KAGGLE_USERNAME / KAGGLE_KEY environment variables instead).\n"
            ) from exc
    return api


def fetch_train_csv(data_dir: Path, force: bool = False) -> Path:
    target = data_dir / FILENAME
    if target.exists() and not force:
        print(f"{target} already exists, skipping download (use --force to re-download).")
        return target

    api = _authenticated_api()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {FILENAME} from Kaggle competition '{COMPETITION}'...")
    try:
        api.competition_download_file(COMPETITION, FILENAME, path=str(data_dir), force=force, quiet=False)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 403:
            raise SystemExit(
                f"Kaggle returned 403 Forbidden.\nYou likely haven't accepted the competition rules yet. "
                f"Visit https://www.kaggle.com/competitions/{COMPETITION}/rules, click "
                "'I Understand and Accept', then re-run this script.\n"
                f"\nOriginal error: {exc}"
            ) from exc
        raise SystemExit(f"Failed to download {FILENAME} from Kaggle: {exc}") from exc

    # Some Kaggle API versions/paths deliver a .zip even for a single-file download; unwrap it.
    zip_path = data_dir / f"{FILENAME}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
        zip_path.unlink()

    if not target.exists():
        raise SystemExit(
            f"Download finished but {target} was not found. Check {data_dir}/ for what was actually saved."
        )
    return target


def main() -> None:
    args = parse_args()
    target = fetch_train_csv(Path(args.data_dir), force=args.force)

    import pandas as pd

    df = pd.read_csv(target)
    print(f"Saved {target} ({len(df)} rows, {len(df.columns)} columns).")


if __name__ == "__main__":
    main()
