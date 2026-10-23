"""Download the harmonised CGM files used by the pipeline (about 20 MB) from the Glucose-ML GitHub repository.

    python scripts/get_data.py

Optional, for meals / Fitbit / full clinical panel: download the original CGMacros release from PhysioNet
(https://physionet.org/content/cgmacros/1.0.0/, 627 MB) and unzip it into data/raw/physionet_cgmacros/.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "glucose_ml"
REPO = "https://github.com/Augmented-Health-Lab/Glucose-ML-Project.git"
FOLDERS = ["3_Glucose-ML-collection/CGMacros_Dexcom", "3_Glucose-ML-collection/ShanghaiT2DM"]


def main():
    if all((DEST / Path(f).name).exists() for f in FOLDERS):
        print(f"Data already present in {DEST}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        run = lambda *a: subprocess.run(a, cwd=tmp, check=True)  # noqa: E731
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, "gml")
        subprocess.run(["git", "sparse-checkout", "set", *FOLDERS], cwd=Path(tmp) / "gml", check=True)
        for f in FOLDERS:
            src = Path(tmp) / "gml" / f
            dst = DEST / src.name
            shutil.copytree(src, dst, dirs_exist_ok=True)
            for z in dst.glob("*.zip"):
                z.unlink()
    print(f"Done. Files in {DEST}")


if __name__ == "__main__":
    sys.exit(main())
