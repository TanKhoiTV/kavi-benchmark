// main.typ — KAVI Benchmark Report (FLEURS + VIVOS, ASR + NMT)
// Compile: typst compile main.typ report.pdf
// Watch:   typst watch main.typ report.pdf

// ---------------------------------------------------------------------------
// DATA lives in data.json — each section file loads it directly via
// `#let d = json("../data.json")` so bindings are in-scope (Typst #let
// bindings do not cross #include boundaries; #include creates a new scope).
// Regenerate data.json from docs/benchmark/raw-results/*.csv when data changes.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Page & text setup
// ---------------------------------------------------------------------------
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  numbering: "1",
)
#set text(font: "Segoe UI", size: 11pt, lang: "vi")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1.")
#set figure(numbering: "1")
#set list(indent: 1.5em)
#set table.cell(inset: (x: 8pt, y: 6pt))
#show table: it => align(center, it)

// ---------------------------------------------------------------------------
// Title page
// ---------------------------------------------------------------------------
#align(center)[
  #v(3cm)
  #text(size: 22pt, weight: "bold")[Benchmark Report]
  #v(0.4cm)
  #text(size: 14pt)[ASR & NMT — FLEURS + VIVOS]
  #v(1cm)
  #text(size: 11pt, fill: gray)[KAVI Prototype — 2026]
  #v(0.3cm)
  #text(size: 10pt, fill: gray)[Internal Draft]
  #v(2cm)
]

#align(center)[
  #table(
    columns: 2,
    align: (right, left),
    stroke: none,
    [Authors:], [KAVI Team],
    [Date:],     [#datetime.today().display()],
    [Status:],   [Draft],
  )
]

#pagebreak()

// ---------------------------------------------------------------------------
// Table of contents
// ---------------------------------------------------------------------------
#heading(level: 1, numbering: none)[Mục lục]
#outline(title: none)
#pagebreak()

// ---------------------------------------------------------------------------
// Body
// ---------------------------------------------------------------------------
#include "sections/01-overview.typ"
#include "sections/02-models.typ"
#include "sections/03-datasets.typ"
#include "sections/04-setup.typ"
#include "sections/05-results-asr.typ"
#include "sections/06-results-mt.typ"
#include "sections/07-analysis.typ"
#include "sections/08-conclusion.typ"
#include "sections/09-appendix.typ"
