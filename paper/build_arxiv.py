"""Assemble an arXiv-ready LaTeX bundle from the generated paper.tex.

arXiv compiles your source on its own servers (default engine: pdfLaTeX), so the
pandoc-generated `paper.tex` needs three fixes it can't ship with:
  * \includegraphics absolute Windows paths  -> bare filenames (one flat dir)
  * \bibliography absolute path              -> references
  * body Unicode (→ ↔ ′ ≈ ≥ − Δ α · §)       -> \DeclareUnicodeCharacter so it
                                                compiles under pdfLaTeX, not only
                                                Xe/LuaLaTeX
It also turns the two-entry \author (which `article` renders as two co-authors)
into a single author with the affiliation on a second line.

    python paper/build.py        # (re)generate paper.tex first
    python paper/build_arxiv.py  # then bundle -> paper/arxiv/

Upload the whole paper/arxiv/ folder to arXiv (or to Overleaf to compile-check
first; set the compiler to pdfLaTeX to mirror arXiv).
"""
import re
import shutil
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ART = PAPER.parent / "artifacts"
OUT = PAPER / "arxiv"

# pdfLaTeX-safe replacements for every non-ASCII glyph in the source.
UNICODE = {
    0x2192: r"\ensuremath{\rightarrow}",       # →
    0x2194: r"\ensuremath{\leftrightarrow}",   # ↔
    0x2032: r"\ensuremath{{}^\prime}",         # ′  (as in d′)
    0x2248: r"\ensuremath{\approx}",           # ≈
    0x2265: r"\ensuremath{\geq}",              # ≥
    0x2212: r"\ensuremath{-}",                 # − (minus)
    0x0394: r"\ensuremath{\Delta}",            # Δ
    0x03B1: r"\ensuremath{\alpha}",            # α
    0x00B7: r"\textperiodcentered{}",          # ·
    0x00A7: r"\S{}",                           # §
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tex = (PAPER / "paper.tex").read_text(encoding="utf-8")

    # 1) figure paths -> bare basenames; remember which files to copy
    figs = []

    def repl(m):
        p = Path(m.group(1))
        figs.append(p.as_posix())
        return "{" + p.name + "}"

    tex = re.sub(r"\{((?:[A-Za-z]:)?[^{}]*?/artifacts/[^{}]*?\.png)\}", repl, tex)

    # 2) bibliography -> bare name (references.bib in the same dir)
    tex = re.sub(r"\\bibliography\{[^}]*\}", r"\\bibliography{references}", tex)

    # 3) single author with affiliation on its own line (not two co-authors)
    tex = tex.replace(r"Nathaniel Gibson \and Independent Researcher",
                      r"Nathaniel Gibson\\ Independent Researcher")

    # 4) inject pdfLaTeX unicode declarations right before \begin{document}
    decl = ("\\ifPDFTeX\n"
            + "\n".join(f"\\DeclareUnicodeCharacter{{{cp:04X}}}{{{rep}}}"
                        for cp, rep in UNICODE.items())
            + "\n\\fi\n")
    tex = tex.replace("\\begin{document}", decl + "\\begin{document}", 1)

    (OUT / "paper.tex").write_text(tex, encoding="utf-8")

    # copy bib + the referenced figures (flattened to basenames)
    shutil.copy(PAPER / "references.bib", OUT / "references.bib")
    for posix in figs:
        rel = posix.split("/artifacts/", 1)[1]
        shutil.copy(ART / rel, OUT / Path(rel).name)

    names = sorted(x.name for x in OUT.iterdir())
    print(f"arXiv bundle -> {OUT}")
    print(f"  {len(figs)} figures + references.bib + paper.tex")
    print("  files:", names)


if __name__ == "__main__":
    main()
