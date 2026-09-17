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

== 组合数恒等式

=== Pascal 恒等式

$
binom(n, k)
=
binom(n-1, k)
+
binom(n-1, k-1)
$

=== 一行组合数之和
$
sum_(k=0)^n binom(n, k) = 2^n
$

$
sum_(k=0)^n binom(n, k) x^k
=
(1+x)^n
$

=== 交错和

当 $n > 0$ 时：

$
sum_(k=0)^n (-1)^k binom(n, k)
=
0
$

=== Hockey-stick 恒等式

$
sum_(i=k)^n binom(i, k)
=
binom(n+1, k+1)
$

$
sum_(i=0)^n binom(r+i, r)
=
binom(r+n+1, r+1)
$

=== Vandermonde 恒等式

$
sum_k
binom(n, k)
binom(m, r-k)
=
binom(n+m, r)
$

$
sum_(k=0)^r
binom(n, k)
binom(n, r-k)
=
binom(2n, r)
$

=== 平方和恒等式

$
sum_(k=0)^n binom(n, k)^2
=
binom(2n, n)
$

=== 带 $k$ 的组合数

$
k binom(n, k)
=
n binom(n-1, k-1)
$

$
sum_(k=0)^n
k binom(n, k)
=
n 2^(n-1)
$

=== 带 $k(k-1)$ 的组合数

$
k(k-1) binom(n, k)
=
n(n-1) binom(n-2, k-2)
$

$
sum_(k=0)^n
k(k-1) binom(n, k)
=
n(n-1) 2^(n-2)
$

$
k^2 = k(k-1) + k
$

$
sum_(k=0)^n
k^2 binom(n, k)
=
n(n+1) 2^(n-2)
$

=== 吸收恒等式

$
binom(n, k) binom(k, r)
=
binom(n, r) binom(n-r, k-r)
$

$
binom(n, k) binom(n-k, r)
=
binom(n, r) binom(n-r, k)
$

=== 相邻组合数之比

$
binom(n, k+1) / binom(n, k)
=
(n-k)/(k+1)
$