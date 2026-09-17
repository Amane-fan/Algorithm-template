`std::bitset` 常用成员函数

- `count()`: 返回 `true` 的数量。
- `size()`: 返回 `bitset` 的大小。
- `test(pos)`: 它和 `vector` 中的 `at()` 的作用类似，与 `[]` 运算符的区别在于会进行越界检查。
- `any()`: 若存在某一位是 `true`，则返回 `true`，否则返回 `false`。
- `none()`: 若所有位都是 `false`，则返回 `true`，否则返回 `false`。
- `all()`: 若所有位都是 `true`，则返回 `true`，否则返回 `false`。

- `set()`: 将整个 `bitset` 设置成 `true`。
- `set(pos, val = true)`: 将某一位设置成 `true` 或 `false`。

- `reset()`: 将整个 `bitset` 设置成 `false`。
- `reset(pos)`: 将某一位设置成 `false`，相当于 `set(pos, false)`。

- `flip()`: 翻转每一位，即 $0 arrow.l.r 1$。相当于异或一个全部为 `1` 的 `bitset`。
- `flip(pos)`: 翻转某一位。

- `to_string()`: 返回转换后的字符串表示。
- `to_ulong()`: 返回转换后的 `unsigned long` 表示。

  `long` 在 Windows NT 及 32 位 POSIX 系统下通常与 `int` 大小相同，
  在 64 位 POSIX 系统下通常与 `long long` 大小相同。

- `to_ullong()`: 从 *C++11* 起支持，返回转换后的 `unsigned long long` 表示。

- `_Find_first()`: 返回 `bitset` 中第一个为 `true` 的位置的下标。
  若不存在 `true`，则返回 `bitset` 的大小。

- `_Find_next(pos)`: 返回 `pos` 后面，即下标严格大于 `pos` 的位置中，
  第一个为 `true` 的位置的下标。
  若 `pos` 后面不存在 `true`，则返回 `bitset` 的大小。

- 计算区间 `[l, r]` 中 `1` 的个数: `(b >> l).count() - (b >> (r + 1)).count() `