"""Entry point: python -m oracle"""
import uvicorn

from .server import CFG, app

if __name__ == "__main__":
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="warning")
