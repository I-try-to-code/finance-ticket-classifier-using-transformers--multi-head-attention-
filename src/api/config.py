"""Configuration and environment handling for Banking77 FastAPI inference service."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_banking77"


@dataclass
class ApiSettings:
    """Runtime configuration for the FastAPI service."""

    model_dir: Path = Path(os.getenv("BANKING77_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    device: Optional[str] = os.getenv("BANKING77_DEVICE", None)
    max_length: int = int(os.getenv("BANKING77_MAX_LENGTH", "128"))
    api_title: str = "Banking77 Intent Classification API"
    api_version: str = "1.0.0"
    api_description: str = (
        "Production-style FastAPI inference service for 77-class banking intent classification "
        "powered by fine-tuned DistilBERT (V2)."
    )
    host: str = os.getenv("BANKING77_HOST", "0.0.0.0")
    port: int = int(os.getenv("BANKING77_PORT", "8000"))


settings = ApiSettings()
