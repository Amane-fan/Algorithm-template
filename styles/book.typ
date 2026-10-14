// All visual rules live here. The builder copies this next to templates.typ.
#let original-cover(meta, opt, cover: none) = {
  let accent = rgb("#293E80")
  let ink = rgb("#20272D")
  let muted = rgb("#748087")
  let title-parts = meta.title.split(" の ")
  let label(body) = text(size: 7.5pt, tracking: 1.5pt, fill: accent, body)
  page(
    paper: opt.paper,
    margin: (x: 24mm, y: 22mm),
    header: none, footer: none, numbering: none,
  )[
    #set text(font: opt.font, fill: ink, lang: "zh")
    #set par(leading: 0pt)
    #block(width: 100%, height: 100%)[
      #place(top + center, label[ALGORITHMS / DATA STRUCTURES])
      #if title-parts.len() == 2 {
        place(top + center, dy: 27mm,
          text(size: 15pt, fill: muted, title-parts.first() + " の"))
      }
      #place(top + center, dy: 40mm,
        text(size: if title-parts.len() == 2 { 48pt } else { 36pt },
          weight: "bold", tracking: -0.8pt, fill: accent,
          if title-parts.len() == 2 { title-parts.last() } else { meta.title }))
      #place(top + center, dy: 74mm, line(length: 10mm, stroke: 0.8pt + accent))
      #if cover != none {
        place(top + center, dy: 91mm, cover)
      }
      #place(top + center, dy: 225mm, text(size: 10pt, fill: muted, meta.subtitle))
      #place(bottom + center, label[#meta.author])
    ]
  ]
}

// Select a candidate with `typst compile --input cover-design=a ...`.
// C (light constellation) is the selected default; other designs remain available.
#let cover-art(design) = {
  let shapes = ""
  if design == "a" {
    shapes += "<circle cx='335' cy='310' r='203' fill='#c95337'/>"
    shapes += "<path d='M105 440 L335 65 L565 440 Z' fill='#eee6d7'/>"
    for i in range(22) {
      let y = 150 + i * 14
      shapes += "<path d='M90 " + str(y) + " Q335 " + str(y - 100) + " 580 " + str(y) + "' fill='none' stroke='#283e3a' stroke-width='1.1'/>"
    }
    shapes += "<circle cx='335' cy='310' r='203' fill='none' stroke='#c95337' stroke-width='1.4'/>"
    shapes += "<path d='M335 65V550M70 440H600' fill='none' stroke='#283e3a' stroke-width='.7'/>"
    shapes += "<circle cx='335' cy='440' r='5' fill='#283e3a'/>"
  } else if design == "b" {
    for i in range(64) {
      let a = i * 2.1
      shapes += "<path d='M" + str(75+a) + " 570 C" + str(-120+a*2) + " 270 " + str(680-a*2) + " 310 " + str(595-a) + " 30' fill='none' stroke='#4e796f' stroke-opacity='.76' stroke-width='1.1'/>"
    }
    shapes += "<circle cx='430' cy='200' r='43' fill='#c79e63'/>"
  } else if design == "c" {
    for radius in (125, 200, 275) {
      shapes += "<circle cx='335' cy='300' r='" + str(radius) + "' fill='none' stroke='#7894aa' stroke-opacity='.55' stroke-width='.8'/>"
    }
    let points = range(19).map(i => {
      let angle = i * 137.508deg
      let radius = 60 + calc.sqrt(i) * 49
      (335 + calc.cos(angle) * radius, 300 + calc.sin(angle) * radius)
    })
    for (i, p) in points.enumerate() {
      for j in range(i + 1, points.len()) {
        let q = points.at(j)
        let distance = calc.sqrt(calc.pow(p.at(0)-q.at(0), 2) + calc.pow(p.at(1)-q.at(1), 2))
        if distance < 190 {
          shapes += "<path d='M" + str(p.at(0)) + " " + str(p.at(1)) + "L" + str(q.at(0)) + " " + str(q.at(1)) + "' stroke='#496f94' stroke-opacity='.78' stroke-width='1'/>"
        }
      }
      shapes += "<circle cx='" + str(p.at(0)) + "' cy='" + str(p.at(1)) + "' r='" + str(if calc.rem(i, 4) == 0 { 4.5 } else { 2.5 }) + "' fill='#ad7c35'/>"
    }
    shapes += "<path d='M335 0V600M35 300H635' stroke='#7894aa' stroke-opacity='.45' stroke-width='.7'/>"
  } else {
    for i in range(15) {
      let pos = 35 + i * 42
      shapes += "<path d='M" + str(pos) + " 15V590M35 " + str(pos) + "H623' stroke='#294ed3' stroke-opacity='.18' stroke-width='1'/>"
    }
    for i in range(12) {
      let size = 510 - i * 36
      shapes += "<rect x='" + str(85+i*36) + "' y='" + str(50+i*36) + "' width='" + str(size) + "' height='" + str(size) + "' fill='none' stroke='#294ed3' stroke-width='2.8'/>"
    }
    shapes += "<rect x='481' y='446' width='114' height='114' fill='#e37445'/>"
  }
  image(bytes("<svg xmlns='http://www.w3.org/2000/svg' width='670' height='600' viewBox='0 0 670 600'>" + shapes + "</svg>"), width: 172mm, height: 154mm)
}

#let book-cover(meta, opt, cover: none, design: sys.inputs.at("cover-design", default: "c")) = {
  if design == "original" { original-cover(meta, opt, cover: cover) } else {
    assert(("a", "b", "c", "d").contains(design), message: "cover-design must be original, a, b, c, or d")
    let paper = if design == "c" { rgb("#f6f8fa") } else if design == "a" { rgb("#f5f0e6") } else if design == "b" { rgb("#f2f3eb") } else { rgb("#f7f6f2") }
    let ink = if design == "c" { rgb("#294766") } else if design == "a" { rgb("#283e3a") } else if design == "b" { rgb("#244c43") } else { rgb("#294ed3") }
    let muted = if design == "c" { rgb("#64798c") } else { rgb("#6b756f") }
    let parts = meta.title.split(" の ")
    let owner = if parts.len() == 2 { parts.first() + " の" } else { meta.author }
    let title = if parts.len() == 2 { parts.last() } else { meta.title }
    let label(body) = text(size: 7pt, tracking: 1.6pt, fill: muted, body)
    page(paper: opt.paper, fill: paper, margin: 0mm, header: none, footer: none, numbering: none)[
      #set text(font: opt.font, fill: ink, lang: "zh")
      #set par(leading: 0pt)
      #block(width: 100%, height: 100%)[
        #if design == "a" {
          place(top + left, dx: 23mm, dy: 21mm, label[AMANE / ALGORITHM LIBRARY])
          place(top + left, dx: 23mm, dy: 40mm, text(size: 16pt, font: "Noto Serif SC", owner))
          place(top + left, dx: 21mm, dy: 52mm, text(font: "Georgia", size: 49pt, title))
          place(top + left, dx: 23mm, dy: 79mm, text(size: 9pt, fill: muted, meta.subtitle))
          place(top + center, dy: 101mm, cover-art(design))
          place(top + left, dx: 23mm, dy: 267mm, line(length: 164mm, stroke: 0.6pt + ink))
          place(top + left, dx: 23mm, dy: 273mm, label[算法与数据结构 / #meta.author])
          place(top + right, dx: -23mm, dy: 273mm, label[COLLECTED TEMPLATES])
        } else if design == "b" {
          place(top + center, dy: 22mm, label[ALGORITHMS & DATA STRUCTURES])
          place(top + center, dy: 48mm, text(font: "Noto Serif SC", size: 15pt, owner))
          place(top + center, dy: 61mm, text(font: "Georgia", size: 46pt, title))
          place(top + center, dy: 94mm, cover-art(design))
          place(top + center, dy: 259mm, text(size: 9pt, fill: muted)[算法与数据结构 · 模板手册])
          place(top + center, dy: 277mm, label[#meta.author])
        } else if design == "c" {
          place(top + center, dy: 21mm, label[THE ALGORITHM ATLAS])
          place(top + center, dy: 43mm, text(size: 15pt, owner))
          place(top + center, dy: 55mm, text(font: "Georgia", size: 45pt, title))
          place(top + center, dy: 86mm, text(size: 8pt, fill: muted, tracking: 1pt, meta.subtitle))
          place(top + center, dy: 103mm, cover-art(design))
          place(top + center, dy: 268mm, label[算法与数据结构 / #meta.author])
        } else {
          place(top + left, dx: 20mm, dy: 19mm, label[AMANE / ALGORITHMS & DATA STRUCTURES])
          place(top + left, dx: 20mm, dy: 33mm, line(length: 170mm, stroke: 1pt + ink))
          place(top + left, dx: 20mm, dy: 44mm, text(size: 16pt, weight: "bold", owner))
          place(top + left, dx: 18.5mm, dy: 58mm, text(font: "Arial", size: 53pt, weight: "bold", tracking: -1.8pt, title))
          place(top + left, dx: 20mm, dy: 87mm, text(size: 9pt, fill: muted)[算法与数据结构 / 模板手册])
          place(top + center, dy: 107mm, cover-art(design))
          place(top + left, dx: 20mm, dy: 274mm, text(size: 8pt, weight: "bold", meta.author))
          place(top + right, dx: -20mm, dy: 274mm, label[STRUCTURE. LOGIC. CODE.])
        }
      ]
    ]
  }
}

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

  book-cover(meta, opt, cover: cover)

  // Physical page 2 is intentionally empty, including headers and numbering.
  page(paper: opt.paper, header: none, footer: none, numbering: none)[]

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
