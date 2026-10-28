"""Build the EURASIP JIVP submission (springer/main.tex, Springer Nature sn-jnl template v3.1) from the LNCS master.

Springer asks for one .tex file: the tables and numbers are inlined; the front matter uses \\fnm/\\sur/\\affil. For
JIVP (SpringerOpen research article): double spacing with line numbers (referee, lineno), APS reference style, a list
of abbreviations and the Declarations section with the journal's headings. Run: python scripts/make_springer.py
"""
import os
import re
import shutil

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(HERE, "springer")
TEMPLATE = os.path.join(OUT, "template")          # the unpacked Springer Nature package (sn-jnl.cls, bst/)


def inline_inputs(text, base_dir):
    """Replace every \\input{path} (not on a commented line) by the contents of that file, recursively."""
    def repl(m):
        line_start = text.rfind("\n", 0, m.start()) + 1
        if text[line_start:m.start()].lstrip().startswith("%"):
            return m.group(0)
        path = m.group(1) + ("" if os.path.splitext(m.group(1))[1] else ".tex")
        sub = open(os.path.join(base_dir, path), encoding="utf-8").read()
        return inline_inputs(sub, base_dir).rstrip("\n")
    return re.sub(r"\\input\{([^}]*)\}", repl, text)


def between(text, start, end):
    i = text.index(start) + len(start)
    return text[i:text.index(end, i)]


PREAMBLE = r"""\documentclass[pdflatex,sn-aps,referee,lineno]{sn-jnl}% APS reference style; double spacing, line numbers

\usepackage{graphicx}%
\usepackage{multirow}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{xcolor}%
\usepackage{booktabs}%
\usepackage{adjustbox}%
\usepackage{tikz}%
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds}%

\graphicspath{{figures/}}

%NUMBERS%

\newcommand{\R}{\mathbb{R}}
\newcommand{\vect}[1]{\mathbf{#1}}

\raggedbottom

\begin{document}

\title[Learning-Free 3D Car Shapes from a Roadside Video]{Learning-Free 3D Car Shapes from a Single Roadside Video}

\author*[1]{\fnm{Dmitriy} \sur{Zhuravlev}}\email{dzhuravlev@ukr.net}

\affil*[1]{\orgname{Taras Shevchenko National University of Kyiv}, \orgaddress{\city{Kyiv}, \country{Ukraine}}}

\abstract{%ABSTRACT%}

\keywords{%KEYWORDS%}

\maketitle
"""

DECLARATIONS = r"""
\backmatter

\section*{Abbreviations}
CAD: computer-aided design; GPS: global positioning system; ICP: iterative closest point; INS: inertial navigation
system; IoU: intersection over union; IPM: inverse perspective mapping; LiDAR: light detection and ranging; RANSAC:
random sample consensus; SUV: sport utility vehicle; V2X: vehicle-to-everything.

\section*{Declarations}

\bmhead{Availability of data and materials}
The datasets analysed during the current study are available from their providers after registration: V2X-Seq
\cite{v2xseq} and DAIR-V2X \cite{dairv2x}. The code behind the method, the tables and the figures is available in the
GitHub repository RoadsideCarShape \cite{code}.

\bmhead{Competing interests}
The author declares no competing interests.

\bmhead{Funding}
This research received no funding.

\bmhead{Authors' contributions}
DZ conceived and designed the study, implemented the method, performed the experiments and wrote the manuscript. The
author read and approved the final manuscript.

\bmhead{Acknowledgements}
Not applicable.
"""


def main():
    src = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
    numbers = "".join(open(os.path.join(HERE, "tables", f), encoding="utf-8").read()
                      for f in ("numbers.tex", "numbers_large.tex") if os.path.exists(os.path.join(HERE, "tables", f)))
    abstract = between(src, r"\begin{abstract}", r"\keywords{").strip()
    keywords = between(src, r"\keywords{", r"\end{abstract}").strip().rstrip("}")
    keywords = re.sub(r"\s+", " ", re.sub(r"\s*\\and\s*", ", ", keywords))
    body = src[src.index(r"\section{Introduction}"):src.index(r"\bibliographystyle")]
    body = inline_inputs(body, HERE)
    # sn-jnl redefines tabular: a scaled table breaks, so tables are set in a smaller font instead (figures keep it)
    body = body.replace("\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", "\\footnotesize\n\\begin{tabular}")
    body = body.replace("\\end{tabular}}", "\\end{tabular}")
    out = (PREAMBLE.replace("%NUMBERS%", numbers.strip()).replace("%ABSTRACT%", abstract)
           .replace("%KEYWORDS%", keywords)
           + "\n" + body.rstrip() + "\n" + DECLARATIONS + "\n\\bibliography{references}\n\n\\end{document}\n")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(out)
    shutil.copy(os.path.join(HERE, "references.bib"), os.path.join(OUT, "references.bib"))
    shutil.copy(os.path.join(TEMPLATE, "sn-jnl.cls"), OUT)
    shutil.copy(os.path.join(TEMPLATE, "bst", "sn-aps.bst"), OUT)
    fig = os.path.join(OUT, "figures")
    if not os.path.exists(fig):
        os.symlink(os.path.join("..", "figures"), fig)
    print("springer/main.tex:", len(out.splitlines()), "lines")


if __name__ == "__main__":
    main()
