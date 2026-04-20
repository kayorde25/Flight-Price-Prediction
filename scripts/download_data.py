## scripts/download_data.py`

```python
from pathlib import Path
import zipfile
import subprocess
import sys

DATASET = "shubhambathwal/flight-price-prediction"
OUTDIR = Path("data")
ZIP_PATH = OUTDIR / "flight-price-prediction.zip"

OUTDIR.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable,
    "-m",
    "kaggle",
    "datasets",
    "download",
    "-d",
    DATASET,
    "-p",
    str(OUTDIR),
]

subprocess.run(cmd, check=True)

if ZIP_PATH.exists():
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(OUTDIR)
    ZIP_PATH.unlink()

print("Done. Files saved in:", OUTDIR.resolve())
