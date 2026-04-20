from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path
import shutil


def _extract_and_cleanup_zips(outdir: Path) -> None:
    for zip_path in outdir.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(outdir)
        zip_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the Kaggle flight price dataset")
    parser.add_argument(
        "--dataset",
        default="shubhambathwal/flight-price-prediction",
        help="Kaggle dataset in the form owner/dataset",
    )
    parser.add_argument(
        "--outdir",
        default="data",
        help="Directory to download/extract dataset into",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    kaggle_cli = shutil.which("kaggle")
    if kaggle_cli is None:
        candidate = Path(sys.executable).with_name("kaggle.exe")
        if candidate.exists():
            kaggle_cli = str(candidate)
        else:
            candidate = Path(sys.executable).with_name("kaggle")
            if candidate.exists():
                kaggle_cli = str(candidate)

    if kaggle_cli is None:
        print(
            "Could not find the Kaggle CLI executable. Ensure the `kaggle` package is installed in this environment.",
            file=sys.stderr,
        )
        return 1

    cmd = [
        kaggle_cli,
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(outdir),
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "Python was not found when attempting to run the Kaggle CLI.",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as e:
        print(
            "Kaggle download failed. If you haven't authenticated yet, configure Kaggle API credentials:\n"
            "- Create a Kaggle token at https://www.kaggle.com/settings\n"
            "- Place kaggle.json at %USERPROFILE%\\.kaggle\\kaggle.json\n"
            "- Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n",
            file=sys.stderr,
        )
        return e.returncode

    _extract_and_cleanup_zips(outdir)
    print("Done. Files saved in:", outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
