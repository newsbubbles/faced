"""Launch the faced face web UI (path-independent entry point).

    python scripts/serve.py [--model gemma-3-1b] [--port 8000]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faced.server import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    run(args.model, host=args.host, port=args.port)
