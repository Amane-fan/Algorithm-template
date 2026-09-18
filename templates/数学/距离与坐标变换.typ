= 三种距离

对于二维平面上的两点，记坐标差为 $a = x_1 - x_2$、$b = y_1 - y_2$。

- 曼哈顿距离：$d_1 = abs(a) + abs(b)$。
- 欧几里得距离：$d_2 = sqrt(a^2 + b^2)$。
- 切比雪夫距离：$d_(infinity) = max(abs(a), abs(b))$。

*曼哈顿距离和欧几里得距离不能仅凭一个距离值唯一互相换算。*
例如位移 $(2, 0)$ 和 $(1, 1)$ 的曼哈顿距离都是 2，但欧几里得距离分别为 2 和 $sqrt(2)$。
已知两点坐标时，可以分别代入公式计算。

三者满足：
$ d_(infinity) <= d_2 <= d_1 <= sqrt(2) d_2. $

= 曼哈顿距离与切比雪夫距离转换

对每个点进行变换：
$ u = x + y, quad v = x - y. $

逆变换为：
$ x = (u + v) / 2, quad y = (u - v) / 2. $

设变换后两点的坐标差为 $Delta u$、$Delta v$，则：

$
  abs(a) + abs(b) = max(abs(a + b), abs(a - b))
                 = max(abs(Delta u), abs(Delta v)).
$

即 *原坐标的曼哈顿距离 = 新坐标的切比雪夫距离*。
反过来还有：

$ max(abs(a), abs(b)) = (abs(Delta u) + abs(Delta v)) / 2. $

即 *原坐标的切比雪夫距离 = 新坐标的曼哈顿距离的一半*，注意这里有系数 2。

原坐标为整数时，变换后的 $u, v$ 一定同奇偶；逆变换时也需要满足这一条件，才能得到整数点。

= 对欧几里得距离的影响

同一个变换满足：
$ (Delta u)^2 + (Delta v)^2 = 2(a^2 + b^2). $

所以 *新坐标的欧几里得距离 = 原坐标的欧几里得距离乘以 $sqrt(2)$*。
若使用归一化的正交变换：

$ u = (x + y) / sqrt(2), quad v = (x - y) / sqrt(2), $

则欧几里得距离保持不变，但新坐标通常不再是整数。

= GNU C++17 用法

使用现有别名 `i64 = long long`、`i128 = __int128`。转换前先提升类型，避免加减法在 `i64` 中溢出。

```cpp
array<i128, 2> transform(i64 x, i64 y) {
    return {i128(x) + y, i128(x) - y};
}

array<i128, 2> restore(i128 u, i128 v) {
    assert((u - v) % 2 == 0); // 要求对应整数点。
    return {(u + v) / 2, (u - v) / 2};
}
```

下面示例的两点为 $(1, 2)$、$(4, 6)$：

```cpp
i64 x1 = 1, y1 = 2, x2 = 4, y2 = 6;
auto abs128 = [](i128 x) { return x < 0 ? -x : x; };
i128 dx = i128(x1) - x2, dy = i128(y1) - y2;

i128 manhattan = abs128(dx) + abs128(dy);       // 7
i128 chebyshev = max(abs128(dx), abs128(dy));   // 4
i128 dist2 = dx * dx + dy * dy;                 // 25，欧氏距离平方
long double dist = hypotl((long double)dx,
                          (long double)dy);    // 5

auto [u1, v1] = transform(x1, y1);
auto [u2, v2] = transform(x2, y2);
i128 ans = max(abs128(u1 - u2), abs128(v1 - v2)); // 7，原曼哈顿距离
auto [x, y] = restore(u1, v1);                    // 还原为 (1, 2)
```

只比较欧氏距离大小时，优先比较距离平方，避免开方误差。
仍需保证平方和不超过 `i128` 范围；上述写法在原坐标绝对值不超过 $10^18$ 时安全。
`restore` 按接收 `transform` 结果及其范围内坐标使用，任意接近 `i128` 极值的输入仍可能使加减法溢出。

= 常见应用

- *曼哈顿距离限制变成矩形查询*：到 $(x_0, y_0)$ 的曼哈顿距离不超过 $r$，等价于变换后落在闭矩形 $[u_0-r, u_0+r] times [v_0-r, v_0+r]$ 内。这里要求 $r >= 0$。
- *最远曼哈顿点对*：至少有一个点时，答案为 $max(max(u)-min(u), max(v)-min(v))$，扫描一遍即可；只有一个点时答案为 0。
- *整数格点计数*：变换后只考虑 $u, v$ 同奇偶的点，不能直接把矩形内所有整数点都算进去。
