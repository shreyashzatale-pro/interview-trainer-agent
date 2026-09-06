#!/usr/bin/env python3
"""
Entry point for the Interview Trainer Agent.
Run: python run.py
Or:  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import uvicorn
from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower(),
    )
