"""CLI and ASGI entry point for Banking77 FastAPI inference service."""

import argparse
from pathlib import Path
import sys
import uvicorn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import app
from src.api.config import settings


def main():
    parser = argparse.ArgumentParser(description="Start Banking77 FastAPI Inference Server")
    parser.add_argument("--host", type=str, default=settings.host, help="Host interface to bind")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"\nStarting Banking77 Inference Service on http://{args.host}:{args.port}")
    print(f"Swagger Documentation available at http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
