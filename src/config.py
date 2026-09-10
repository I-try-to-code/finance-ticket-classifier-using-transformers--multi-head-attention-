"""Configuration and environment reproducibility constants for BANKING77 pipeline."""

from pathlib import Path
import platform
import sys
from typing import Any, Dict


# Dataset identification
DATASET_NAME: str = "banking77"
EXPECTED_NUM_INTENTS: int = 77

# Split parameters
RANDOM_SEED: int = 42
VAL_SIZE: float = 0.20
STRATIFY_COLUMN: str = "label"

# Project directory layout
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

REPORT_PATH: Path = REPORTS_DIR / "banking77_audit_report.md"


def get_environment_metadata() -> Dict[str, Any]:
    """Collect runtime environment and package versions for reproducibility audit."""
    metadata: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "os": platform.platform(),
        "dataset_name": DATASET_NAME,
        "random_seed": RANDOM_SEED,
        "validation_ratio": VAL_SIZE,
    }

    packages = ["datasets", "pandas", "numpy", "sklearn", "pytest"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            metadata[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            metadata[f"{pkg}_version"] = "not_installed"

    return metadata
