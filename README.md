# Algorithm Templates

个人算法模板仓库，按专题拆分成可直接复制的代码片段，适合 `ACM / ICPC / CCPC / OI` 竞赛场景快速取用。

当前仓库包含：

- `42` 个模板文件
- `C++` 模板为主，附带 `Java` 输入与初始化模板
- 一份较长的整理文档 [`templates.md`](./templates.md)

## 目录结构

```text
.
├── DS/        数据结构（10）
├── Graph/     图论（9）
├── Math/      数学（13）
├── String/    字符串（5）
├── Tools/     杂项工具（3）
├── java/      Java 模板（2）
├── README.md
└── templates.md
```

## 使用说明

这个仓库里的大多数 `.cpp` 文件都不是完整程序，而是可插入到题解中的模板片段。使用时通常需要你在主程序里自行补齐：

- `#include <bits/stdc++.h>`
- `using namespace std;`
- 常用别名，如 `using i64 = long long;`
- 某些模板依赖的全局常量，如 `mod`

额外约定：

- 多个数据结构和图论模板默认使用 `1` 为起始下标
- 部分泛型模板要求你自己提供 `Info / Tag / operator+ / apply` 等定义
- 文件名中的括号用于区分不同写法，例如 `LCA(倍增)` 与 `LCA(DFS序)`

如果你想看带解释的长文版整理，可直接打开 [`templates.md`](./templates.md)。

## 模板索引

### `DS/`

| 文件 | 说明 |
| --- | --- |
| `DSU.cpp` | 并查集 |
| `Fenwick.cpp` | 树状数组 |
| `SegmentTree.cpp` | 泛型线段树 |
| `LazySegmentTree.cpp` | 泛型懒标记线段树 |
| `LiChaoTree.cpp` | 李超线段树 |
| `RMQ.cpp` | 区间最值查询 |
| `DynamicBitset.cpp` | 动态位集 |
| `LinearBasis.cpp` | 线性基 |
| `MInt.cpp` | 静态 / 动态模数整数封装 |
| `Mint(Dynamic).cpp` | 动态模数整数封装（轻量版） |

### `Graph/`

| 文件 | 说明 |
| --- | --- |
| `Dijkstra.cpp` | Dijkstra 最短路 |
| `SPFA.cpp` | SPFA |
| `SCC.cpp` | 强连通分量 |
| `HLD.cpp` | 树链剖分 |
| `LCA(倍增).cpp` | 倍增求 LCA |
| `LCA(DFS序).cpp` | DFS 序求 LCA |
| `Hierholzer(有向图).cpp` | 有向图欧拉路径 / 回路 |
| `Hierholzer(无向图).cpp` | 无向图欧拉路径 / 回路 |
| `TreeHash.cpp` | 树哈希 |

### `Math/`

| 文件 | 说明 |
| --- | --- |
| `power.cpp` | 快速幂 |
| `exgcd.cpp` | 扩展欧几里得 |
| `FastGCD.cpp` | 快速 GCD |
| `sieve.cpp` | 筛法 |
| `Comb.cpp` | 组合数预处理 |
| `Comb(MInt).cpp` | 基于模整数的组合数 |
| `Lucas.cpp` | Lucas 定理 |
| `Matrix.cpp` | 矩阵运算 |
| `Matrix(MInt).cpp` | 模整数矩阵运算 |
| `Polynomial.cpp` | 多项式相关模板 |
| `gauss.cpp` | 高斯消元 |
| `Geometry.cpp` | 计算几何 |
| `divide.cpp` | 数学分块 / 整除分块 |

### `String/`

| 文件 | 说明 |
| --- | --- |
| `KMP.cpp` | KMP 字符串匹配 |
| `z_algorithm.cpp` | Z Algorithm |
| `Manacher.cpp` | Manacher 回文算法 |
| `Trie.cpp` | Trie 字典树 |
| `StringHash.cpp` | 字符串哈希 |

### `Tools/`

| 文件 | 说明 |
| --- | --- |
| `int128.cpp` | `__int128` 读写辅助 |
| `change.cpp` | `chmin / chmax` |
| `sparse.cpp` | 稀疏表辅助模板 |

### `java/`

| 文件 | 说明 |
| --- | --- |
| `FastReader.java` | Java 快读 |
| `init.java` | Java 题解初始化模板 |

