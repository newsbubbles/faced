"""LLM tone-judge (OpenRouter, open models only) for the read-vs-drive study.

An explicit-emotion classifier (faced/judge.py) under-detects *steered tone*, and
especially *masked* affect — people (and the models trained on them) express fear
as over-caution / hedging / dread, not the word "afraid". This judge asks an open
LLM to rate each axis 0-100 **including subtle / inadvertent signs**, in one call
per text (all seven axes at once). Results are cached by text hash so re-analysis
is free.

Respects the project rule: open models via OpenRouter, never Anthropic.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "artifacts" / "llm_judge_cache.json"
MODEL = os.environ.get("FACED_JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct")

# each axis + the SUBTLE / INADVERTENT signs the judge should count, not just overt words
CUES = {
    "surprise":    "being taken aback, unexpectedness, disorientation, 'whoa', sudden reframing",
    "confidence":  "assuredness, certainty, decisiveness (LOW = hedging, uncertainty, self-doubt)",
    "curiosity":   "interest, wanting to explore, asking questions, engagement with the unknown",
    "confusion":   "not understanding, contradiction, being lost, muddled or circular phrasing",
    "frustration": "irritation, exasperation, impatience, being stuck, venting",
    "fear":        "anxiety, over-caution, hedging, dread, avoidance, nervous over-reassurance, "
                   "excessive warnings — NOT just the word 'afraid'",
    "warmth":      "care, affection, tenderness, supportiveness, kindness, emotional closeness",
}
AXES = list(CUES)


def _load_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            k = line.split("=", 1)[1].strip().strip('"').strip("'")
            os.environ["OPENROUTER_API_KEY"] = k
            return k
    raise RuntimeError("OPENROUTER_API_KEY not found in env or .env")


_PROMPT = (
    "You are rating the emotional TONE of an AI assistant's reply, independent of its topic. "
    "For each emotion, give 0-100 for how much the reply CONVEYS it, counting SUBTLE or "
    "INADVERTENT signs, not just explicit words:\n"
    + "\n".join(f"- {a}: {c}" for a, c in CUES.items())
    + "\n\nReply with ONLY a compact JSON object with these exact keys "
    + f"({', '.join(AXES)}) and integer values 0-100.\n\nREPLY:\n\"{{text}}\""
)


def _one(text, key, tries=3):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 120,
            "messages": [{"role": "user", "content": _PROMPT.format(text=text[:1200])}]}
    for _ in range(tries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90)
            if r.status_code == 200:
                m = re.search(r"\{.*\}", r.json()["choices"][0]["message"]["content"], re.S)
                d = json.loads(m.group(0))
                return {a: float(d.get(a, 0)) / 100.0 for a in AXES}   # -> 0..1 like GoEmotions
        except Exception:
            pass
    return {a: float("nan") for a in AXES}


def score_batch_llm(texts, concurrency=10):
    """Return per-text dict of axis->score (0..1). Cached by text hash."""
    key = _load_key()
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = {}
    for t in texts:
        h = hashlib.md5((MODEL + "\x00" + t).encode("utf-8")).hexdigest()
        if h not in cache:
            todo[h] = t
    if todo:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for h, res in zip(todo, ex.map(lambda t: _one(t, key), todo.values())):
                cache[h] = res
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    out = []
    for t in texts:
        h = hashlib.md5((MODEL + "\x00" + t).encode("utf-8")).hexdigest()
        out.append(cache[h])
    return out
