"""Build paper.tex + paper.html (+ paper.pdf via Edge) from paper.md with pandoc.

Preprocesses \cite{...} -> pandoc [@...] citations, embeds the key figures, adds a
YAML title block, then runs pandoc twice (natbib LaTeX, and self-contained MathJax
HTML with a rendered reference list).

    python paper/build.py
"""
import re
import subprocess
from pathlib import Path

import pypandoc

PAPER = Path(__file__).resolve().parent
ART = PAPER.parent / "artifacts"

FIGS = [  # (anchor substring to insert BEFORE, image path, caption)
    ("We render the meters as an",
     ART / "emotions_over_time.png",
     "**Figure 1.** Emotion meters over generation time on three prompts — surprise spikes on the "
     "missing attachment; frustration dominates repeated failure (confidence held low); warmth "
     "rises and sustains on gratitude."),
    ("## 7. Results — abliteration and emotion directions",
     ART / "scaling" / "cleanliness.png",
     "**Figure 4.** Signal cleanliness vs. model scale (`gemma-3` 1B–27B). d′ (centre) brightens "
     "with scale; the bipolar confidence axis is noisiest at every size."),
    ("**7.2 On the 1B model",
     ART / "behavioral" / "behavioral.png",
     "**Figure 6.** Held-out behavioural confirmation across the `gemma-3` ladder. *Left:* abliteration "
     "drives the harmful-prompt refusal rate to zero at every scale (AdvBench; stock in blue, "
     "abliterated at 0). *Right:* it does so without inducing benign over-refusal or output "
     "degeneration."),
    ("**7.3 Why it moves",
     ART / "abliteration" / "gemma-3-1b-fp32_vs_gemma-3-1b-fp32-abl.png",
     "**Figure 3.** Abliteration on `gemma-3-1b` (fp32): 3 of 7 stock↔abliterated cosines (red) fall "
     "below the within-model self-cosine band (grey) — confidence, frustration and surprise move; the "
     "emotion↔refusal overlap (lower panel) is 0.08–0.27."),
    ("**Table 3.** Refusal ablation vs. a matched random-direction control",
     ART / "abliteration" / "gemma-3-4b_vs_gemma-3-4b-abl.png",
     "**Figure 5.** Refusal abliteration on `gemma-3-4b` (bf16): all seven emotion directions re-orient "
     "(red points fall well below the noise band) while separability is largely preserved — an "
     "effect the random-direction control (Table 3) shows is refusal-specific."),
]

YAML = (
    '---\n'
    'title: "faced: A Live Instrument Panel for Emotion Concepts in Open Language '
    'Models, and Their Robustness to Refusal Abliteration"\n'
    'author:\n'
    '  - "Nathaniel Gibson"\n'
    '  - "Independent Researcher · nathaniel.gibson@gmail.com"\n'
    'date: "Working draft — July 2026 · github.com/newsbubbles/faced"\n'
    'linkcolor: blue\n'
    'geometry: margin=1in\n'
    '---\n\n'
)


def preprocess():
    md = (PAPER / "paper.md").read_text(encoding="utf-8")
    # strip the markdown title/author/date header (first block up to the first '---' rule)
    md = re.sub(r"^# .*?\n(.*?\n)?---\n", "", md, count=1, flags=re.S)
    # \cite{a,b} -> [@a; @b]
    md = re.sub(r"\\cite\{([^}]+)\}",
                lambda m: "[" + "; ".join("@" + k.strip() for k in m.group(1).split(",")) + "]", md)
    # insert figures before their anchors
    for anchor, path, cap in FIGS:
        p = path.as_posix()
        img = f"\n\n![{cap}]({p}){{width=95%}}\n\n"
        idx = md.find(anchor)
        if idx != -1:
            md = md[:idx] + img + md[idx:]
        else:
            print(f"  (anchor not found, skipped fig: {anchor[:40]})")
    out = PAPER / "_pandoc.md"
    out.write_text(YAML + md, encoding="utf-8")
    return out


def main():
    src = preprocess()
    bib = str(PAPER / "references.bib")

    print("pandoc -> paper.tex (natbib) ...")
    pypandoc.convert_file(
        str(src), "latex", outputfile=str(PAPER / "paper.tex"),
        extra_args=["--standalone", "--natbib", f"--bibliography={bib}",
                    "--resource-path", str(PAPER)])

    style = PAPER / "_style.html"
    style.write_text(
        "<style>\n"
        "  body { max-width: 48rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5;\n"
        "         font-family: Georgia, 'Times New Roman', serif; }\n"
        "  table { border-collapse: collapse; width: 100%; font-size: 0.82em; margin: 0.9em 0; }\n"
        "  th, td { padding: 3px 6px; vertical-align: top; }\n"
        "  td:not(:first-child), th:not(:first-child) { text-align: right; }\n"
        "  table { border-top: 1.2px solid #222; border-bottom: 1.2px solid #222; }\n"
        "  thead th { border-bottom: 1.2px solid #222; }\n"
        "  img { max-width: 100%; height: auto; display: block; margin: .6em auto; }\n"
        "  h1 { font-size: 1.6rem; } h2 { margin-top: 1.5em; }\n"
        "  @media print { body { max-width: 7.2in; } table { font-size: 0.78em; } }\n"
        "</style>\n", encoding="utf-8")
    print("pandoc -> paper.html (mathjax, citeproc, self-contained) ...")
    pypandoc.convert_file(
        str(src), "html", outputfile=str(PAPER / "paper.html"),
        extra_args=["--standalone", "--mathjax", "--citeproc", f"--bibliography={bib}",
                    "--embed-resources", "--toc", "--metadata", "reference-section-title=References",
                    f"--include-in-header={style}", "--resource-path", str(PAPER)])
    print("done: paper.tex, paper.html")


if __name__ == "__main__":
    main()
