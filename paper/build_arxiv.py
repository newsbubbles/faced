"""Assemble a SELF-CONTAINED, arXiv-ready LaTeX bundle from paper.md.

arXiv compiles your source itself (default engine pdfLaTeX) and **does not run
BibTeX** — so a bundle that ships only `.bib` gets an empty bibliography. Rather
than depend on a hand-generated `.bbl`, we regenerate the LaTeX with the
bibliography **inlined** via pandoc `--citeproc`. The result needs no `.bib`, no
`.bbl`, no BibTeX pass.

We then apply the fixes the arXiv guides call for:
  * `\\pdfoutput=1` as the first line  -> forces pdfLaTeX (our \\DeclareUnicode…
    and .png figures require it, not dvi-latex)
  * \\includegraphics paths            -> bare, underscore-free filenames, flat dir
  * two-entry \\author                 -> one author, affiliation on line 2
  * \\DeclareUnicodeCharacter for the 10 non-ASCII glyphs (pdfLaTeX-safe)
  * a 4-passes \\typeout after \\end{document} so refs resolve

    python paper/build_arxiv.py        # (regenerates its own _pandoc.md)

Bundle -> paper/arxiv/ : paper.tex + the 5 figures. Upload that folder to arXiv
(compile-check on Overleaf first, Compiler = pdfLaTeX).
"""
import re
import shutil
from pathlib import Path

import pypandoc

import build  # sibling module: reuse its preprocess() (embeds figures, \cite -> [@..])

PAPER = Path(__file__).resolve().parent
ART = PAPER.parent / "artifacts"
OUT = PAPER / "arxiv"

UNICODE = {
    0x2192: r"\ensuremath{\rightarrow}",       # →
    0x2194: r"\ensuremath{\leftrightarrow}",   # ↔
    0x2032: r"\ensuremath{{}^\prime}",         # ′
    0x2248: r"\ensuremath{\approx}",           # ≈
    0x2265: r"\ensuremath{\geq}",              # ≥
    0x2212: r"\ensuremath{-}",                 # −
    0x0394: r"\ensuremath{\Delta}",            # Δ
    0x03B1: r"\ensuremath{\alpha}",            # α
    0x00B7: r"\textperiodcentered{}",          # ·
    0x00A7: r"\S{}",                           # §
}


def main():
    shutil.rmtree(OUT, ignore_errors=True)       # start clean (no stale files ship)
    OUT.mkdir(parents=True, exist_ok=True)
    src = build.preprocess()                     # writes/refreshes paper/_pandoc.md
    bib = str(PAPER / "references.bib")

    # LaTeX with the bibliography INLINED (citeproc) -> no BibTeX needed on arXiv
    tex = pypandoc.convert_file(
        str(src), "latex",
        extra_args=["--standalone", "--citeproc", f"--bibliography={bib}",
                    "--metadata", "reference-section-title=References",
                    "--resource-path", str(PAPER)])

    # 1) figure paths -> bare, underscore-free basenames; remember what to copy
    figs = {}  # sanitized_name -> path relative to artifacts/

    def repl(m):
        p = Path(m.group(1))
        san = p.name.replace("_", "-")
        figs[san] = p.as_posix().split("/artifacts/", 1)[1]
        return "{" + san + "}"

    tex = re.sub(r"\{((?:[A-Za-z]:)?[^{}]*?/artifacts/[^{}]*?\.png)\}", repl, tex)

    # 2) single author with affiliation on its own line
    tex = tex.replace(r"Nathaniel Gibson \and Independent Researcher",
                      r"Nathaniel Gibson\\ Independent Researcher")

    # 3) force pdfLaTeX on arXiv (first line)
    tex = "\\pdfoutput=1\n" + tex

    # 4) pdfLaTeX unicode declarations before \begin{document}
    decl = ("\\ifPDFTeX\n"
            + "\n".join(f"\\DeclareUnicodeCharacter{{{cp:04X}}}{{{rep}}}"
                        for cp, rep in UNICODE.items())
            + "\n\\fi\n")
    tex = tex.replace("\\begin{document}", decl + "\\begin{document}", 1)

    # 5) make arXiv run enough passes for cross-references
    tex = tex.replace(
        "\\end{document}",
        "\\end{document}\n\\typeout{get arXiv to do 4 passes: "
        "Label(s) may have changed. Rerun}\n", 1)

    (OUT / "paper.tex").write_text(tex, encoding="utf-8")
    for san, rel in figs.items():
        shutil.copy(ART / rel, OUT / san)

    names = sorted(x.name for x in OUT.iterdir())
    print(f"arXiv bundle -> {OUT}")
    print(f"  self-contained (bibliography inlined; no .bib/.bbl needed)")
    print(f"  {len(figs)} figures + paper.tex")
    print("  files:", names)


if __name__ == "__main__":
    main()
