// 解同余方程组 x ≡ a[i][0] (mod a[i][1])，模数不要求两两互质。
// a 中每项为 {余数, 模数}，不保留下标 0；余数可为负，模数必须为正。
// 返回 {最小非负解 r, 所有模数的最小公倍数 m}，通解为 x = r + k * m。
// 无解返回 {0, 0}；空方程组返回 {0, 1}。
// 要求合并过程中的最小公倍数不超过 i64 上限，中间乘法使用 i128。
// k 个方程的时间复杂度为 O(k log M)，M 为模数的最大值。
array<i64, 2> exCRT(const vector<array<i64, 2>> &a) {
    auto exgcd = [](auto &&self, i128 x, i128 y) -> array<i128, 3> {
        if (y == 0) {
            return {x, 1, 0};
        }
        auto [g, u, v] = self(self, y, x % y);
        return {g, v, u - x / y * v};
    };

    i64 r = 0, m = 1;
    for (auto [b, mod] : a) {
        assert(mod > 0);
        b %= mod;
        if (b < 0) {
            b += mod;
        }

        auto [g, x, y] = exgcd(exgcd, m, mod);
        i128 d = i128(b) - r;
        if (d % g != 0) {
            return {0, 0};
        }

        // 令新解为 r + m * t，求 (m / g) * t ≡ d / g (mod mod / g)。
        i128 q = mod / g;
        i128 t = d / g * x % q;
        if (t < 0) {
            t += q;
        }
        i128 next = i128(m) * q;
        assert(next <= numeric_limits<i64>::max());
        r = i64((i128(r) + i128(m) * t) % next);
        m = i64(next);
    }
    return {r, m};
}

/*
vector<array<i64, 2>> a = {{2, 6}, {5, 9}};
auto [r, m] = exCRT(a); // r = 14, m = 18，即 x = 14 + 18k。
if (m == 0) {
    cout << "No solution\n";
} else {
    cout << r << '\n'; // 最小非负解。
    // 若要求最小正解，输出 (r == 0 ? m : r)。
}
*/
