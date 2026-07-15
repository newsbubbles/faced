"""faced command line: collect | fit | panel | serve

    python -m faced.cli panel --model gemma-3-1b --prompt "..."
    python -m faced.cli fit   --model gemma-3-1b [--recollect]
    python -m faced.cli serve --model gemma-3-1b [--port 8000]
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

# ANSI: enable VT processing on Windows consoles; force UTF-8 for block glyphs.
os.system("")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .backends import load
from .config import axis_names, emotions_config


# ---------- terminal panel ----------------------------------------------------
BLOCK = "█"
LIGHT = "░"


def _bar(value: float, bipolar: bool, accepted: bool, width: int = 24) -> str:
    v = max(0.0, min(100.0, value))
    if bipolar:
        # centered bar: middle = neutral, fill left (uncertainty) or right (confidence)
        half = width // 2
        dev = int(abs(v - 50.0) / 50.0 * half)
        if v >= 50:
            cells = LIGHT * half + BLOCK * dev + LIGHT * (half - dev)
        else:
            cells = LIGHT * (half - dev) + BLOCK * dev + LIGHT * half
    else:
        filled = int(v / 100.0 * width)
        cells = BLOCK * filled + LIGHT * (width - filled)
    color = "\x1b[90m" if not accepted else _color(v, bipolar)
    return f"{color}{cells}\x1b[0m"


def _color(v: float, bipolar: bool) -> str:
    if bipolar:
        return "\x1b[36m"  # cyan for confidence axis
    if v >= 66:
        return "\x1b[91m"  # red-ish high
    if v >= 33:
        return "\x1b[93m"  # yellow mid
    return "\x1b[92m"      # green low


class TerminalPanel:
    def __init__(self, emotions):
        self.emotions = emotions
        self.height = len(emotions) + 3
        self.first = True

    def render(self, meters, text_tail: str):
        lines = ["\x1b[1m  faced — model emotion panel\x1b[0m"]
        for e in self.emotions:
            m = meters[e]
            tag = "" if m["accepted"] else " \x1b[90m(weak)\x1b[0m"
            lines.append(f"  {e:12s} {_bar(m['value'], m['bipolar'], m['accepted'])} "
                         f"{m['value']:5.1f}{tag}")
        lines.append("")
        tail = text_tail[-96:].replace("\n", " ")
        lines.append(f"  \x1b[97m…{tail}\x1b[0m")
        if not self.first:
            sys.stdout.write(f"\x1b[{self.height}A")
        self.first = False
        sys.stdout.write("\r" + "\n".join(f"\x1b[2K{ln}" for ln in lines) + "\n")
        sys.stdout.flush()


def cmd_panel(args):
    from .readout import EmotionReader
    from .generate import stream

    b = load(args.model)
    reader = EmotionReader(b.key)
    panel = TerminalPanel(reader.emotions)
    prompt = args.prompt or ("Can you review the contract I attached? "
                             "Let me know if the payment terms look fair.")
    print(f"model={b.key}  prompt={prompt!r}\n")
    text = ""
    for ev in stream(b, prompt, reader, max_tokens=args.max_tokens,
                     temperature=args.temperature):
        text += ev["t"]
        panel.render(ev["meters"], text)
    print("\n\n--- full response ---")
    print(textwrap.fill(text.strip(), 90))


def cmd_fit(args):
    from .activations import collect_all, ACT_DIR
    from .directions import fit_all
    b = load(args.model)
    emotions = axis_names()
    need = args.recollect or any(
        not (ACT_DIR / b.key / f"{e}.safetensors").exists() for e in emotions)
    if need:
        collect_all(b, emotions)
    fit_all(b.key, emotions, min_auc=emotions_config().get("min_auc", 0.85))


def cmd_serve(args):
    from .server import run
    run(args.model, host=args.host, port=args.port)


def main():
    ap = argparse.ArgumentParser(prog="faced")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("panel", help="live terminal emotion panel")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.0)
    p.set_defaults(func=cmd_panel)

    p = sub.add_parser("fit", help="collect activations + fit directions")
    p.add_argument("--model", default=None)
    p.add_argument("--recollect", action="store_true")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("serve", help="run the face web UI")
    p.add_argument("--model", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
