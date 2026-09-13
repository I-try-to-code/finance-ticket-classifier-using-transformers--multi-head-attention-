"""FastAPI inference service package for Banking77 Intent Classification."""

from src.api.app import app, create_app
from src.api.config import settings

__all__ = ["app", "create_app", "settings"]
