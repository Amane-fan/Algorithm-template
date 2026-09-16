// All visual rules live here. The builder copies this next to templates.typ.
#let book(config: (:), cover: none, body) = {
  let meta = config.book
  let opt = config.layout
  let accent = rgb("#" + opt.accent)
  let ink = rgb("#202B30")
  let muted = rgb("#68767A")
  let rule = rgb("#D4DFDF")

  set document(title: meta.title, author: meta.author)
  set text(font: opt.font, size: opt.body-size * 1pt, fill: ink, lang: "zh")
  set par(leading: 0.6em, justify: false)
  set page(
    paper: opt.paper,
    binding: left,
    margin: (top: opt.margin-top * 1mm, bottom: opt.margin-bottom * 1mm,
             inside: opt.margin-inside * 1mm, outside: opt.margin-outside * 1mm),
    header: context {
      let topics = query(heading.where(level: 1)).filter(h => h.location().position().page <= here().position().page)
      let name = if topics.len() > 0 { topics.last().body } else { [] }
      set text(size: 7pt, fill: muted)
      grid(columns: (1fr, auto), name, meta.title)
      v(3pt)
      line(length: 100%, stroke: 0.45pt + rule)
    },
    footer: context {
      set text(size: 8pt, fill: accent)
      align(center, counter(page).display("1"))
    },
  )
  set heading(numbering: "1.1", outlined: true)
  show heading.where(level: 1): it => {
    if opt.section-new-page { pagebreak(weak: true) }
    block(above: 18pt, below: 10pt, sticky: true)[
      #grid(columns: (auto, 1fr), gutter: 9pt, align: horizon,
        context text(size: 24pt, weight: "bold", fill: accent)[#counter(heading).display("01")],
        text(size: 19pt, weight: "bold", it.body),
      )
      #v(4pt)
      #line(length: 100%, stroke: 0.8pt + accent)
    ]
  }
  show heading.where(level: 2): it => block(above: 12pt, below: 5pt, sticky: true)[
    #text(size: 11pt, weight: "bold")[
      #if it.numbering != none { context text(fill: accent, counter(heading).display(it.numbering)); h(7pt) }
      #it.body
    ]
  ]
  show heading.where(level: 3): set text(size: 10pt, weight: "bold")
  show heading: set block(sticky: true)
  show link: set text(fill: accent)
  show quote.where(block: true): block.with(inset: (left: 10pt), stroke: (left: 1.5pt + rule))
  set table(stroke: 0.4pt + rule, inset: 5pt)
  set raw(theme: ".templatebook-code.tmTheme", tab-size: 4)
  show table.cell.where(y: 0): set text(weight: "bold", fill: accent)
  show image: set image(width: 100%)
  show raw.where(block: false): set text(font: opt.code-font, size: 0.9em)
  show raw.where(block: true): set text(font: opt.code-font, size: opt.code-size * 1pt, ligatures: false)
  show raw.where(block: true): set par(leading: opt.code-leading * 1pt)
  show raw.where(block: true): it => block(
    width: 100%, breakable: it.lines.len() > opt.keep-short-code-lines, above: 5pt, below: 9pt,
    inset: (x: 8pt, y: 6pt),
    stroke: 0.6pt + rgb("#A8B7BA"),
  )[
    // Keep syntax highlighting and preserve whitespace; allow long tokens to wrap.
    #show raw.line: line => {
      show regex("[^\\s]"): ch => ch + sym.zws
      line
    }
    #it
  ]

  // Cover has no running header or page number.
  page(header: none, footer: none)[
    #v(20mm)
    #text(size: 8pt, tracking: 2pt, fill: accent, weight: "bold")[ALGORITHM LIBRARY]
    #v(10mm)
    #text(size: 32pt, weight: "bold", meta.title)
    #v(5mm)
    #text(size: 12pt, fill: muted, meta.subtitle)
    #v(12mm)
    #line(length: 28mm, stroke: 2pt + accent)
    #v(12mm)
    #if cover != none {
      cover
    } else {
      block(width: 100%, height: 72mm, inset: (left: 10mm, top: 15mm), stroke: (left: 0.7pt + rule))[
        #text(size: 42pt, fill: accent, weight: "light")[{ … }]
        #v(8mm)
        #text(size: 10pt, fill: muted)[#config.topic-count topics  /  #config.module-count modules]
      ]
    }
    #v(1fr)
    #line(length: 100%, stroke: 0.5pt + rule)
    #v(5mm)
    #grid(columns: (1fr, auto), text(size: 9pt, meta.author), text(size: 8pt, fill: muted)[TEMPLATES])
    #v(8mm)
  ]

  counter(page).update(1)
  [
    #set page(header: none, footer: context align(center, text(size: 8pt, fill: muted, counter(page).display("i"))))
    #text(size: 22pt, weight: "bold")[目录]
    #v(3mm)
    #text(size: 8pt, fill: muted)[CONTENTS]
    #v(7mm)
    #show outline.entry: set text(size: 8.5pt)
    #show outline.entry: set par(leading: 0.65em)
    #show outline.entry.where(level: 1): set text(weight: "bold", fill: accent)
    #show outline: it => context {
      let entries = query(it.target).filter(h => h.outlined)
      let halfway = calc.ceil(entries.len() / 2)
      let boundaries = entries.enumerate().filter(pair => pair.at(0) > 0 and pair.at(1).level == 1)
      let cut = if boundaries.len() > 0 {
        boundaries.sorted(key: pair => calc.abs(pair.at(0) - halfway)).first().at(0)
      } else { halfway }
      for (i, item) in entries.enumerate() {
        if opt.toc-columns == 2 and i == cut { colbreak() }
        outline.entry(item.level, item)
      }
    }
    #columns(opt.toc-columns, gutter: 10mm)[
      #outline(title: none, indent: 9pt)
    ]
    #pagebreak()
  ]
  counter(page).update(1)
  body
}
