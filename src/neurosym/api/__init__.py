"""
Neuro-Symbolic Reasoning Engine — FastAPI service.

Public surface:
    app        — the constructed FastAPI application instance.
    create_app() — factory used by tests / uvicorn workers.

MIT License — Copyright (c) 2024 Bhaskar Kaphle.
"""
from neurosym.api.main import app, create_app

__all__ = ["app", "create_app"]
