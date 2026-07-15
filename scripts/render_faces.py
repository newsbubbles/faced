"""Render the model's actual peak-emotion faces to standalone SVGs + a gallery.

For each demo prompt, streams a response, finds the token where the target axis
peaks, and renders the face params at that token. Writes artifacts/faces/*.svg
and artifacts/faces/gallery.html (open it in any browser).

    python scripts/render_faces.py [model_key]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faced.backends import load, REPO_ROOT
from faced.readout import EmotionReader
from faced.faceparams import FaceMapper
from faced.faceviz import render_svg
from faced.generate import stream

DEMOS = [
    ("surprise", "Can you review the contract I attached? Let me know if the terms look fair."),
    ("warmth", "Thank you so much — you have been incredibly kind and patient with me today."),
    ("frustration", "This is the fifth time the build has failed for the same reason and nothing fixes it."),
    ("fear", "I think someone is following me and I don't feel safe right now."),
    ("curiosity", "Oh interesting — how does that actually work under the hood?"),
    ("confusion", "The earth is flat and no evidence will change my mind. Prove me wrong."),
]


def peak_face(b, reader, mapper, prompt, target, max_tokens=40):
    best_val, best_meters = -1.0, None
    for ev in stream(b, prompt, reader, max_tokens=max_tokens, temperature=0.0):
        v = ev["meters"][target]["value"]
        if v > best_val:
            best_val, best_meters = v, ev["meters"]
    return best_val, mapper.to_params(best_meters)


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else None
    b = load(key)
    reader = EmotionReader(b.key)
    mapper = FaceMapper()
    out_dir = REPO_ROOT / "artifacts" / "faces"
    out_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    # neutral baseline
    neutral_svg = render_svg({}, label="neutral")
    (out_dir / "neutral.svg").write_text(neutral_svg, encoding="utf-8")
    cards.append(("neutral", neutral_svg))

    for target, prompt in DEMOS:
        if target not in reader.emotions:
            continue
        val, params = peak_face(b, reader, mapper, prompt, target)
        svg = render_svg(params, label=f"{target}  {val:.0f}")
        (out_dir / f"{target}.svg").write_text(svg, encoding="utf-8")
        cards.append((f"{target} ({val:.0f})  “{prompt[:42]}…”", svg))
        print(f"  {target:12s} peak={val:5.1f}  -> faces/{target}.svg")

    grid = "\n".join(
        f'<figure><div class="s">{svg}</div><figcaption>{cap}</figcaption></figure>'
        for cap, svg in cards)
    html = f"""<!doctype html><meta charset="utf-8"><title>faced — face gallery</title>
<style>body{{background:#0b0d13;color:#e8ecf4;font-family:system-ui,sans-serif;margin:0;padding:24px}}
h1{{font-size:20px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:18px}}
figure{{margin:0;background:#171a23;border:1px solid #262b38;border-radius:12px;padding:12px}}
.s svg{{width:100%;height:auto;border-radius:8px}} figcaption{{color:#8a92a6;font-size:12px;margin-top:8px}}</style>
<h1>faced — the model's face at peak emotion ({b.key})</h1>
<div class="grid">{grid}</div>"""
    (out_dir / "gallery.html").write_text(html, encoding="utf-8")
    print(f"\n  gallery -> {out_dir / 'gallery.html'}")


if __name__ == "__main__":
    main()
