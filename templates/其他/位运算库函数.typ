= GNU C++17 内建函数

#table(
  columns: (1fr, 1fr, 1.6fr),
  [`unsigned int` 参数], [`unsigned long long` 参数], [作用],
  [`__builtin_popcount(x)`], [`__builtin_popcountll(x)`], [二进制中 1 的个数],
  [`__builtin_parity(x)`], [`__builtin_parityll(x)`], [1 的个数模 2],
  [`__builtin_clz(x)`], [`__builtin_clzll(x)`], [前导 0 的个数],
  [`__builtin_ctz(x)`], [`__builtin_ctzll(x)`], [末尾 0 的个数],
  [`__builtin_ffs(x)`], [`__builtin_ffsll(x)`], [最低位 1 的位置，从 1 开始；输入 0 返回 0],
)

- 均返回 `int`；`ffs` / `ffsll` 的参数分别为 `int` / `long long`。
- *`clz(0)`、`ctz(0)` 及其 `ll` 版本行为未定义，必须先判零。*
- 64 位数使用 `ll` 版本，无后缀版本可能截断高位；`l` 版本的位宽依平台而定。

= C++20 标准库 <bit>

需要 `#include <bit>`，以下函数均使用 `std::` 前缀，参数 `x` 必须为无符号整数（如 `unsigned`、`unsigned long long`）。

#table(
  columns: (1fr, 2fr),
  [函数], [作用与注意事项],
  [`popcount(x)`], [二进制中 1 的个数],
  [`countl_zero(x)` / `countr_zero(x)`], [前导 / 末尾 0 的个数；输入 0 返回类型位宽],
  [`countl_one(x)` / `countr_one(x)`], [前导 / 末尾 1 的个数；全为 1 时返回类型位宽],
  [`has_single_bit(x)`], [是否为 2 的非负整数次幂；输入 0 返回 `false`],
  [`bit_width(x)`], [表示 x 所需的二进制位数；输入 0 返回 0],
  [`bit_floor(x)`], [不超过 x 的最大 2 的幂；输入 0 返回 0],
  [`bit_ceil(x)`], [不小于 x 的最小 2 的幂；输入 0 返回 1；*结果必须能用输入类型表示*],
  [`rotl(x, s)` / `rotr(x, s)`], [按类型完整位宽循环左移 / 右移；移位量对位宽取模，负数表示反向],
)
