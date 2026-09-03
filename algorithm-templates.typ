#let ink = rgb("#20252D")
#let muted = rgb("#737C86")
#let accent = rgb("#3E7C78")
#let accent-soft = rgb("#DDEAE7")
#let paper = rgb("#FBFAF6")
#let code-bg = rgb("#F2F3F0")
#let rule = rgb("#D7DAD5")
#let min-code-lines-at-start = 11
#let section-name = state("section-name", [])
#let section-number = state("section-number", [])

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
    outside: 14mm,
  ),
  fill: paper,
  header-ascent: 8mm,
  footer-descent: 8mm,
  header: context {
    let page-number = counter(page).get().first()
    if page-number > 2 {
      set text(size: 7.2pt, fill: muted)
      grid(
        columns: (1fr, auto),
        align: (left, right),
        section-name.get(),
        [Amane の Templates],
      )
      v(2.5pt)
      line(length: 100%, stroke: 0.45pt + rule)
    }
  },
  footer: context {
    let page-number = counter(page).get().first()
    if page-number > 1 {
      let label = if page-number < 10 {
        "0" + str(page-number)
      } else {
        str(page-number)
      }
      align(right)[
        #line(length: 13mm, stroke: 0.55pt + accent)
        #h(4pt)
        #text(size: 7.2pt, weight: "semibold", fill: accent)[#label]
      ]
    }
  },
)

#set text(
  font: "Maple Mono NF",
  size: 8.6pt,
  fill: ink,
  lang: "zh",
  region: "CN",
)
#set par(leading: 0.5em)
#set heading(outlined: true)

#show heading.where(level: 1): it => block(
  width: 100%,
  breakable: false,
  above: 2pt,
  below: 13pt,
)[
  #grid(
    columns: (18mm, 1fr),
    column-gutter: 7pt,
    align: (right + horizon, left + horizon),
    context text(
      size: 27pt,
      weight: "bold",
      fill: accent-soft,
    )[#section-number.get()],
    [
      #text(size: 6.6pt, weight: "semibold", fill: accent)[SECTION]
      #v(1.5pt)
      #text(size: 19pt, weight: "bold", fill: ink)[#it.body]
    ],
  )
  #v(5pt)
  #line(length: 100%, stroke: 0.7pt + accent)
]

#show heading.where(level: 2): it => block(
  width: 100%,
  sticky: true,
  breakable: false,
  above: 12pt,
  below: 5pt,
)[
  #grid(
    columns: (4pt, 1fr),
    column-gutter: 7pt,
    rect(width: 4pt, height: 13pt, radius: 2pt, fill: accent),
    text(size: 11.2pt, weight: "semibold", fill: ink)[#it.body],
  )
]

#show outline.entry.where(level: 1): set text(
  size: 9pt,
  weight: "bold",
  fill: ink,
)
#show outline.entry.where(level: 1): it => block(
  above: 8pt,
  below: 7pt,
)[#it]
#show outline.entry.where(level: 2): set text(
  size: 8.3pt,
  fill: muted,
)
#show outline.entry.where(level: 2): it => block(
  above: 7pt,
  below: 7pt,
)[#it]

#show raw: set text(
  font: "Maple Mono NF",
  size: 6.35pt,
  fill: rgb("#27323B"),
)
#show raw.line: code-line => if (
  code-line.number < min-code-lines-at-start
  and code-line.number < code-line.count
) {
  block(
    width: 100%,
    sticky: true,
    above: 0pt,
    below: 0pt,
  )[
    #if code-line.text == "" {
      text(" ")
    } else {
      code-line.body
    }
  ]
} else {
  code-line.body
}
#show raw.where(block: true): it => block(
  width: 100%,
  breakable: true,
  fill: code-bg,
  stroke: 0.45pt + rule,
  radius: 3pt,
  inset: (left: 5pt, right: 5pt, top: 5pt, bottom: 5pt),
  above: 0pt,
  below: 10pt,
  clip: true,
)[#it]

#let template(title, path) = {
  heading(level: 2)[#title]
  raw(read(path), lang: "cpp", block: true, tab-size: 2)
}

#let category(number, title) = {
  section-name.update(title)
  section-number.update(number)
  pagebreak(weak: true)
  heading(level: 1)[#title]
}

#block(width: 100%, height: 100%)[
  #place(top + right, dx: -3mm, dy: 12mm)[
    #circle(radius: 27mm, fill: accent-soft)
  ]
  #place(top + right, dx: -3mm, dy: 12mm)[
    #circle(
      radius: 19mm,
      fill: paper,
      stroke: 0.65pt + accent,
    )
  ]
  #place(bottom + left, dx: 6mm, dy: -15mm)[
    #grid(
      columns: (4mm, 1fr),
      column-gutter: 9mm,
      rect(
        width: 3mm,
        height: 76mm,
        radius: 1.5mm,
        fill: accent,
      ),
      block(width: 132mm)[
        #grid(
          columns: (1fr,),
          row-gutter: 2mm,
          text(size: 45pt, weight: "bold", fill: ink)[Amane],
          text(size: 15pt, weight: "medium", fill: accent)[の],
          text(size: 23pt, weight: "medium", fill: muted)[Templates],
        )
        #v(9mm)
        #line(length: 100%, stroke: 0.55pt + rule)
      ],
    )
  ]
]

#pagebreak()
#block(width: 100%, below: 8pt)[
  #text(size: 6.6pt, weight: "semibold", fill: accent)[CONTENTS]
  #v(1pt)
  #text(size: 23pt, weight: "bold", fill: ink)[目录]
  #v(4pt)
  #line(length: 100%, stroke: 0.7pt + accent)
]
#outline(title: none, depth: 2, indent: 14pt)

#category("01", "数据结构")
#template("并查集", "DataStructure/DSU.cpp")
#template("树状数组", "DataStructure/Fenwick.cpp")
#template("ST表", "DataStructure/RMQ.cpp")
#template("线段树", "DataStructure/SegmentTree.cpp")
#template("懒标记线段树", "DataStructure/LazySegmentTree.cpp")
#template("李超树", "DataStructure/LiChaoTree.cpp")
#template("莫队", "DataStructure/Mo_Algorithm.cpp")
#template("线性基", "DataStructure/LinearBasis.cpp")
#template("取模类(Short Version)", "DataStructure/MInt(Short Version).cpp")

#category("02", "图论")
#template("Dijkstra", "Graph/Dijkstra.cpp")
#template("SPFA", "Graph/SPFA.cpp")
#template("LCA(倍增)", "Graph/LCA(倍增).cpp")
#template("LCA(DFS序)", "Graph/LCA(DFS序).cpp")
#template("重链剖分", "Graph/HLD.cpp")
#template("树上启发式合并", "Graph/dsu_on_tree.cpp")
#template("树的重心", "Graph/树的重心.cpp")
#template("树哈希", "Graph/TreeHash.cpp")
#template("强连通分量", "Graph/SCC.cpp")
#template("边双", "Graph/EBCC.cpp")
#template("欧拉回路(无向图)", "Graph/Hierholzer(无向图).cpp")
#template("欧拉回路(有向图)", "Graph/Hierholzer(有向图).cpp")
#template("匈牙利算法", "Graph/匈牙利算法.cpp")

#category("03", "数学")
#template("快速幂", "Math/power.cpp")
#template("扩展欧几里得", "Math/exgcd.cpp")
#template("FastGCD", "Math/FastGCD.cpp")
#template("上下取整", "Math/divide.cpp")
#template("线性筛", "Math/sieve.cpp")
#template("组合数", "Math/Comb.cpp")
#template("卢卡斯", "Math/Lucas.cpp")
#template("矩阵类", "Math/Matrix.cpp")
#template("高斯消元", "Math/gauss.cpp")
#template("多项式", "Math/Polynomial.cpp")
#template("几何", "Math/Geometry.cpp")

== 卡特兰数

令 $C_n$ 表示第 $n$ 个卡特兰数：

$ C_n = 1 / (n + 1) binom(2n, n) = binom(2n, n) - binom(2n, n - 1) $

初值与常用递推式：

$ C_0 = 1, quad C_n = sum_(i = 0)^(n - 1) C_i C_(n - 1 - i) $

$ C_n = (4n - 2) / (n + 1) C_(n - 1) quad (n >= 1) $

== 第二类斯特林数

令 $S_(n, k)$ 表示将 $n$ 个不同元素划分为 $k$ 个非空、无标号集合的方案数。

$ S_(0, 0) = 1, quad S_(n, 0) = 0 quad (n > 0), quad S_(0, k) = 0 quad (k > 0) $

递推式：

$ S_(n, k) = S_(n - 1, k - 1) + k S_(n - 1, k) $

容斥形式：

$ S_(n, k) = 1 / k! sum_(i = 0)^k (-1)^(k - i) binom(k, i) i^n $

#category("04", "字符串")
#template("KMP", "String/KMP.cpp")
#template("Z函数", "String/z_algorithm.cpp")
#template("马拉车", "String/Manacher.cpp")
#template("字符串哈希", "String/StringHash.cpp")
#template("字典树", "String/Trie.cpp")
#template("最小表示法", "String/minRotation.cpp")

#category("05", "其他")
#template("change", "others/change.cpp")
#template("自定义哈希", "others/hash.cpp")
#template("随机数", "others/random.cpp")
