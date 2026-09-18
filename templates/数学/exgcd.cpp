// 返回 {g, x, y}，满足 a * x + b * y = g。
// 按 a, b >= 0 使用时，g = gcd(a, b)；负数输入不保证 g >= 0。
// T 使用有符号整数类型，如 i64、i128；时间复杂度 O(log(max(a, b)))。
// exgcd(0, 0) 返回 {0, 1, 0}，此时不能用 g 做取模或除法。
template <class T>
array<T, 3> exgcd(const T &a, const T &b) {
    if (b == T(0)) {
        return {a, T(1), T(0)};
    }
    auto [g, x, y] = exgcd(b, a % b);
    return {g, y, x - a / b * y};
}

/*
// 1. 求 gcd 和一组贝祖系数。
i64 a = 30, b = 18;
auto [g, x, y] = exgcd(a, b); // g = 6，且 30 * x + 18 * y = 6。

// 2. 求 a 在模 mod 下的逆元：要求 mod > 1，且 gcd(a, mod) == 1。
// 先将 a 规范到 [0, mod)，mod 不要求是质数。
{
    i64 a = 3, mod = 11;
    a %= mod;
    if (a < 0) {
        a += mod;
    }
    auto [g, x, y] = exgcd(a, mod);
    if (g == 1) {
        i64 inv = x % mod;
        if (inv < 0) {
            inv += mod;
        }
        cout << inv << '\n'; // 4；g != 1 时不存在逆元。
    }
}

// 3. 解 a * X + b * Y = c：要求 a, b 不同时为 0。
// 当且仅当 c % g == 0 时有整数解。
i64 c = 12;
if (c % g == 0) {
    i128 X = i128(x) * (c / g);
    i128 Y = i128(y) * (c / g);
    // 通解：X + (b / g) * t，Y - (a / g) * t，其中 t 为任意整数。
}
// a == b == 0 时单独判断：c == 0 则任意整数对都是解，否则无解。
*/
