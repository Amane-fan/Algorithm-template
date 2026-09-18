= GNU C++17 内建函数

以下按竞赛环境中 `u32 = unsigned` 为 32 位、`u64 = unsigned long long` 为 64 位说明。
位下标从最低位开始计数，最低位下标为 0。

#table(
  columns: (1fr, 1fr, 1.6fr),
  [`unsigned int` 参数], [`unsigned long long` 参数], [含义],
  [`__builtin_popcount(x)`], [`__builtin_popcountll(x)`], [二进制中 1 的个数],
  [`__builtin_parity(x)`], [`__builtin_parityll(x)`], [1 的个数模 2],
  [`__builtin_clz(x)`], [`__builtin_clzll(x)`], [从最高位开始连续 0 的个数],
  [`__builtin_ctz(x)`], [`__builtin_ctzll(x)`], [从最低位开始连续 0 的个数],
)

- 这些函数都返回 `int`。`popcount(0)`、`parity(0)` 均为 0。
- *`clz(0)`、`ctz(0)` 及其 `ll` 版本行为未定义，必须先判零。*
- 后缀 `l` 对应 `unsigned long`，其位宽依平台而定；64 位数直接用 `ll` 版本。
- 传入 `u64` 不会让无后缀函数自动变成 64 位版本，超出 `unsigned int` 的高位会被截断。

```cpp
u64 x = 40;                         // 二进制 101000
int cnt = __builtin_popcountll(x);  // 2
int parity = __builtin_parityll(x); // 0
int lz = __builtin_clzll(x);        // 58
int tz = __builtin_ctzll(x);        // 3，也是最低位 1 的下标
int hi = 63 - __builtin_clzll(x);   // 5，也是 floor(log2(x))

// 允许 x == 0 的写法：不存在有效位时，下标记为 -1。
int low = x ? __builtin_ctzll(x) : -1;
int high = x ? 63 - __builtin_clzll(x) : -1;
int width = x ? 64 - __builtin_clzll(x) : 0;
```

`__builtin_ffs(int)` / `__builtin_ffsll(long long)` 返回最低位 1 的位置，*从 1 开始计数*；输入 0 时返回 0。它们接收有符号参数。

```cpp
int pos = __builtin_ffsll(40LL); // 4，相当于非零时 ctzll(x) + 1
int zero = __builtin_ffsll(0LL); // 0
```

= C++20 标准库 <bit>

需要 `#include <bit>` 并启用 C++20。下列函数接收无符号整数；使用 `40U`、`40ULL` 或显式转换，不能直接传入有符号的 `40`。

```cpp
u64 x = 40;
int cnt = std::popcount(x);     // 2，1 的个数
int lz = std::countl_zero(x);   // 58，前导 0 的个数
int tz = std::countr_zero(x);   // 3，末尾 0 的个数
int lo = std::countl_one(x);    // 0，前导 1 的个数
int to = std::countr_one(7U);   // 3，末尾 1 的个数

bool one = std::has_single_bit(x); // false，是否为 2 的整数次幂
int width = std::bit_width(x);     // 6，表示 x 所需的二进制位数
u64 down = std::bit_floor(x);      // 32，不超过 x 的最大 2 的幂
u64 up = std::bit_ceil(x);         // 64，不小于 x 的最小 2 的幂

u64 left = std::rotl(x, 2);    // 160，按完整 64 位循环左移
u64 right = std::rotr(x, 3);   // 5，按完整 64 位循环右移
```

- `countl_zero(0ULL)`、`countr_zero(0ULL)` 返回 64；对 `0U` 返回 32。
- `countl_one(~0ULL)`、`countr_one(~0ULL)` 返回 64。
- `bit_width(0U) = 0`，`bit_floor(0U) = 0`，`bit_ceil(0U) = 1`，`has_single_bit(0U) = false`。
- *`bit_ceil` 的结果必须能用输入类型表示。* 对 `u64`，输入不能超过 `1ULL << 63`；对 `u32`，不能超过 `1U << 31`。
- `rotl` / `rotr` 会把移位量对类型位宽取模；负数表示向相反方向旋转。循环移位会把移出的位补回另一端。

= 常用组合与边界

下面仍按 64 位无符号数说明，GNU C++17 即可使用。

```cpp
u64 x = 40, y = 24;
u64 lowbit = x & -x;                    // 8；x == 0 时结果为 0
u64 rest = x & (x - 1);                 // 32，清除最低位的 1
bool one = x != 0 && (x & (x - 1)) == 0; // 是否为 2 的幂
int dist = __builtin_popcountll(x ^ y); // 2，两个数不同的二进制位数

int k = 5; // 要求 0 <= k < 64
bool bit = (x >> k) & 1ULL; // 检查第 k 位
u64 a = x | (1ULL << k);   // 将第 k 位置 1
u64 b = x & ~(1ULL << k);  // 将第 k 位置 0
u64 c = x ^ (1ULL << k);   // 翻转第 k 位

int n = 64; // 要求 0 <= n <= 64
u64 mask = n == 64 ? ~0ULL : (1ULL << n) - 1; // 低 n 位全为 1
```

- `1 << k` 的左操作数是 `int`，处理 64 位掩码应写 `1ULL << k`。
- 普通 `<<` / `>>` 的移位量不能为负，也不能达到左操作数提升后的位宽；`1ULL << 64` 不合法。
- GNU 的 `__builtin_popcountll` 只处理 `unsigned long long`。统计 `u128` 时需拆成两个 64 位数：

```cpp
u128 x = (u128(1) << 100) | 7;
int cnt = __builtin_popcountll(u64(x))
        + __builtin_popcountll(u64(x >> 64)); // 4
```

参考：#link("https://gcc.gnu.org/onlinedocs/gcc-15.1.0/gcc/Bit-Operation-Builtins.html")[GCC 内建函数文档]、#link("https://eel.is/c++draft/bit")[C++ 标准草案 <bit>]。
