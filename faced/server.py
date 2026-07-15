"""FastAPI server: streams per-token emotion meters + face params over SSE.

    GET /                -> the face web UI
    GET /api/meta        -> model key, axis list, face params, per-axis flags
    GET /api/stream?prompt=...  -> text/event-stream of
        {"i", "t", "meters": {...}, "face": {...}} per token, then event: done

Single-user local demo: one model + one EmotionReader (EMA state) are shared.
"""
from __future__ import annotations

import json

from .backends import load, REPO_ROOT
from .readout import EmotionReader
from .faceparams import FaceMapper
from .generate import stream

WEB = REPO_ROOT / "web"


def build_app(model_key: str | None = None):
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse, FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="faced")
    b = load(model_key)
    reader = EmotionReader(b.key)
    mapper = FaceMapper()

    @app.get("/")
    def index():
        return FileResponse(str(WEB / "index.html"))

    @app.get("/api/meta")
    def meta():
        return {
            "model": b.key,
            "emotions": reader.emotions,
            "params": mapper.params,
            "axes": {e: {"accepted": reader.calib.accepted(e),
                         "bipolar": reader.calib.is_bipolar(e),
                         "neg_label": reader.meta[e].get("neg_label")}
                     for e in reader.emotions},
        }

    @app.get("/api/stream")
    def stream_ep(prompt: str, max_tokens: int = 120, temperature: float = 0.0):
        def gen():
            for ev in stream(b, prompt, reader, max_tokens=max_tokens,
                             temperature=temperature):
                payload = {"i": ev["i"], "t": ev["t"], "meters": ev["meters"],
                           "face": mapper.to_params(ev["meters"])}
                yield f"data: {json.dumps(payload)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
    return app


def run(model_key: str | None = None, host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run(build_app(model_key), host=host, port=port)
