#let ink = rgb("#18222F")
#let muted = rgb("#697586")
#let accent = rgb("#D1644B")
#let accent-soft = rgb("#F3DDD6")
#let paper = rgb("#FCFAF7")
#let code-bg = rgb("#F4F2EE")
#let code-rule = rgb("#DDD8D0")

#set document(
  title: [Amane の Templates],
  author: "Amane",
  keywords: ("algorithm", "templates", "C++20"),
)

#set page(
  paper: "a4",
  binding: left,
  margin: (
    top: 17mm,
    bottom: 17mm,
    inside: 16mm,
    outside: 13mm,
  ),
  fill: paper,
  header: context {
    let p = counter(page).get().first()
    if p > 1 {
      set text(font: "Maple Mono NF", size: 7.8pt, fill: muted)
      grid(
        columns: (1fr, auto),
        align: (left, right),
        [AMANE / ALGORITHM TEMPLATES],
        [C++20],
      )
      v(2.5pt)
      line(length: 100%, stroke: 0.45pt + code-rule)
    }
  },
  footer: context {
    let p = counter(page).get().first()
    if p > 1 {
      set text(font: "Maple Mono NF", size: 8pt, fill: muted)
      line(length: 100%, stroke: 0.45pt + code-rule)
      v(3pt)
      grid(
        columns: (1fr, auto, 1fr),
        align: (left, center, right),
        [PRINT EDITION],
        counter(page).display("1"),
        [2026.07],
      )
    }
  },
)

#set text(
  font: "Maple Mono NF",
  size: 9.2pt,
  fill: ink,
  lang: "zh",
  region: "CN",
)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(
  font: "Maple Mono NF",
  size: 20pt,
  weight: "bold",
  fill: ink,
)
#show heading.where(level: 2): set text(
  font: "Maple Mono NF",
  size: 12.5pt,
  weight: "bold",
  fill: ink,
)
#show heading.where(level: 1): set block(above: 18pt, below: 10pt)
#show heading.where(level: 2): set block(above: 13pt, below: 4pt)

#show raw: set text(
  font: "Maple Mono NF",
  size: 6.45pt,
  fill: rgb("#25313F"),
)
#show raw.where(block: true): it => block(
  width: 100%,
  breakable: true,
  fill: code-bg,
  stroke: 0.45pt + code-rule,
  radius: 2.5pt,
  inset: (x: 6pt, y: 5pt),
  above: 3pt,
  below: 8pt,
)[#it]

#let note(body) = block(
  width: 100%,
  fill: rgb("#F8EDE8"),
  stroke: (left: 2pt + accent),
  inset: (left: 7pt, right: 7pt, top: 5pt, bottom: 5pt),
  above: 3pt,
  below: 6pt,
)[
  #set text(size: 8pt, fill: rgb("#59443E"))
  #body
]

#let cpp-template(title, path, remark: none) = {
  heading(level: 2)[#title]
  if remark != none { note(remark) }
  raw(read(path), lang: "cpp", block: true)
}

// 从 templates.md 中按二级标题提取代码块。只有无对应 cpp 的章节才调用。
#let markdown = read("templates.md")
#let md-code(title, index: none) = {
  let marker = "## " + title
  let section = markdown.split(marker).at(1).split(regex("\r?\n#{1,2} ")).first()
  let chunks = section.split("```")
  let snippets = ()
  for (i, chunk) in chunks.enumerate() {
    if calc.odd(i) {
      snippets.push(chunk)
    }
  }
  let selected = if index == none { snippets } else { (snippets.at(index),) }
  for snippet in selected {
    let lines = snippet.split("\n")
    let language = lines.first().trim().replace("c++", "cpp")
    let body = lines.slice(1).join("\n").trim()
    raw(body, lang: language, block: true)
  }
}

#let md-template(title, source-title: none, index: none, remark: none) = {
  heading(level: 2)[#if source-title == none { title } else { source-title }]
  if remark != none { note(remark) }
  md-code(title, index: index)
}

#let category(title, kicker) = {
  pagebreak(weak: true)
  block(
    breakable: false,
    width: 100%,
    above: 8pt,
    below: 3pt,
  )[
    #heading(level: 1)[#title]
    #block(
      width: 100%,
      fill: accent,
      inset: (x: 7pt, y: 3pt),
      below: 7pt,
    )[
      #set text(size: 7pt, weight: "bold", fill: white)
      #upper(kicker)
    ]
  ]
}

#let toc-column(from, to) = context {
  let entries = query(selector(heading).after(here()))
  let stop = calc.min(to, entries.len())
  for entry in entries.slice(from, stop) {
    let location = entry.location()
    let numbers = counter(heading).at(location)
    let number = numbering("1.1", ..numbers)
    let page-number = counter(page).at(location).first()
    block(below: 3pt)[
      #set text(
        size: if entry.level == 1 { 9.2pt } else { 8.55pt },
        weight: if entry.level == 1 { "bold" } else { "regular" },
        fill: ink,
      )
      #grid(
        columns: (auto, 1fr, auto),
        column-gutter: 4pt,
        link(location)[
          #if entry.level == 2 { h(8pt) }
          #number #h(3.5pt) #entry.body
        ],
        align(horizon)[
          #line(
            length: 100%,
            stroke: (
              paint: code-rule,
              thickness: 0.5pt,
              dash: "dotted",
            ),
          )
        ],
        link(location)[#page-number],
      )
    ]
  }
}

// 封面
#align(center + horizon)[
  #text(
    font: "Maple Mono NF",
    size: 34pt,
    weight: "bold",
    fill: ink,
  )[Amane の Templates]
]

// 第二页开始：目录。目录条目自动显示每份模板的起始页。
#pagebreak()
#heading(level: 1, numbering: none, outlined: false)[目录]
#line(length: 100%, stroke: 1pt + accent)
#v(7pt)
#grid(
  columns: (1fr, 1fr),
  column-gutter: 10mm,
  toc-column(0, 32),
  toc-column(32, 100),
)
#pagebreak()

#category("数据结构", "DATA STRUCTURES")

#cpp-template("并查集", "DS/DSU.cpp")
#cpp-template("树状数组", "DS/Fenwick.cpp")
#cpp-template("线段树", "DS/SegmentTree.cpp")
#cpp-template(
  "懒标记线段树",
  "DS/LazySegmentTree.cpp",
  remark: [维护幺半群信息时需正确设置幺元；`Info::apply` 与 `Tag::apply` 应保持标记语义一致。],
)
#cpp-template("李超线段树", "DS/LiChaoTree.cpp")
#cpp-template("RMQ / Sparse Table", "DS/RMQ.cpp")
#cpp-template("线性基", "DS/LinearBasis.cpp")
#cpp-template(
  "离散化",
  "Tools/sparse.cpp",
  remark: [`offset` 决定参与排序去重的起始下标；1 索引数组应传入 `1`。],
)
#md-template("笛卡尔树")
#md-template("分块", remark: [示例维护区间加与区间和。])
#md-template("莫队", remark: [示例为离线查询区间不同数字数量。])
#md-template("珂朵莉树")
#md-template("组合哈希")

#category("数学", "MATHEMATICS")

#cpp-template("组合数（自动扩容）", "Math/Comb.cpp")
#cpp-template("快速幂", "Math/power.cpp")
#cpp-template("欧拉筛", "Math/sieve.cpp")
#md-template("埃式筛")
#md-template("线性求逆元", remark: [模数 `P` 必须为质数。])
#cpp-template("扩展欧几里得", "Math/exgcd.cpp")
#md-template("中国剩余定理")
#cpp-template("卢卡斯定理", "Math/Lucas.cpp")
#cpp-template("高斯消元", "Math/gauss.cpp")
#cpp-template("矩阵快速幂", "Math/Matrix.cpp")

#heading(level: 2)[曼哈顿距离与切比雪夫距离]
#note[
  映射 `(x, y) -> (x + y, x - y)` 后，原坐标系的曼哈顿距离变为新坐标系的切比雪夫距离。
  反向映射可写为 `((x + y) / 2, (x - y) / 2)`。
]

#md-template("数论分块")
#cpp-template("计算几何", "Math/Geometry.cpp")
#md-template("素数测试与因式分解")
#cpp-template("多项式 / NTT", "Math/Polynomial.cpp")
#cpp-template("快速 GCD", "Math/FastGCD.cpp")
#cpp-template("向下取整与向上取整", "Math/divide.cpp")

#category("字符串", "STRINGS")

#cpp-template("KMP", "String/KMP.cpp", remark: [下标从 `0` 开始。])
#cpp-template("字符串哈希（随机底数）", "String/StringHash.cpp")
#md-template(
  "字符串哈希（随机底数模数）",
  source-title: "字符串哈希（随机底数与模数）",
)
#cpp-template("字典树", "String/Trie.cpp")
#md-template(
  "字典树",
  source-title: "01 字典树",
  index: 1,
  remark: [补充整数最大异或查询变体。],
)
#cpp-template("Z 函数", "String/z_algorithm.cpp", remark: [下标从 `0` 开始。])
#cpp-template("Manacher", "String/Manacher.cpp", remark: [下标从 `0` 开始。])

#category("图与树", "GRAPHS & TREES")

#cpp-template("Dijkstra", "Graph/Dijkstra.cpp")
#md-template("Floyd")
#cpp-template("SPFA", "Graph/SPFA.cpp")
#md-template(
  "差分约束",
  remark: [约束 `a - b <= c` 转化为一条 `b -> a`、边权为 `c` 的有向边。],
)
#cpp-template("LCA（倍增）", "Graph/LCA(倍增).cpp")
#cpp-template("LCA（DFS 序）", "Graph/LCA(DFS序).cpp")
#cpp-template("树链剖分", "Graph/HLD.cpp")
#md-template("克鲁斯卡尔重构树")
#cpp-template("树哈希", "Graph/TreeHash.cpp")
#md-template("树的重心")
#cpp-template("Hierholzer（无向图）", "Graph/Hierholzer(无向图).cpp")
#cpp-template("Hierholzer（有向图）", "Graph/Hierholzer(有向图).cpp")
#cpp-template("强连通分量", "Graph/SCC.cpp")

#category("常用工具", "UTILITIES")

#cpp-template("chmin / chmax", "Tools/change.cpp")
#cpp-template("int128 输入输出与运算", "Tools/int128.cpp")
#md-template("编译脚本")
#md-template("对拍（linux）", source-title: "对拍（Linux）")
#md-template("各种随机数的生成", source-title: "随机数据生成")
