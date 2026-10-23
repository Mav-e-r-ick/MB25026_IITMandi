"""Central configuration. Every research definition lives here so it can be changed in one place
and is quoted verbatim in the evaluation report."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS_DIR = ROOT / "models"

# Harmonised copies published by the Glucose-ML project (MIT code; data keep original licences).
GML_CGMACROS = DATA_RAW / "glucose_ml" / "CGMacros_Dexcom"
GML_SHANGHAI = DATA_RAW / "glucose_ml" / "ShanghaiT2DM"
# Optional original PhysioNet release (meals, Fitbit, bio.csv). Unzip CGMacros_dateshifted365.zip here.
PHYSIONET_CGMACROS = DATA_RAW / "physionet_cgmacros"


@dataclass(frozen=True)
class TaskConfig:
    """Prediction task definition (research definition, NOT a clinical treatment target)."""

    step_min: int = 5                 # grid resolution for CGMacros (Dexcom G6)
    history_min: int = 180            # input window
    horizon_min: int = 120            # forecast horizon
    threshold_mgdl: float = 180.0     # excursion threshold (configurable)
    min_future_coverage: float = 0.8  # fraction of future readings required for a valid label
    min_history_coverage: float = 0.8 # fraction of history readings required to make a prediction
    max_staleness_min: int = 15       # last reading must be no older than this to predict
    forecast_points_min: tuple[int, ...] = field(default=(15, 30, 60, 90, 120))

    @property
    def n_hist(self) -> int:
        return self.history_min // self.step_min

    @property
    def n_hor(self) -> int:
        return self.horizon_min // self.step_min


TASK = TaskConfig()
SEED = 2026
