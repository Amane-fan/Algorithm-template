// GNU C++17；需 i64 = long long，i128 = __int128。
// factorize(n)：1 <= n < 2^63，升序返回质因子（含重复）。
// n = 1 返回空数组；例：360 -> {2, 2, 2, 3, 3, 5}。
// isprime(n)：确定性判素，O(log n)；n < 2 返回 false。
// 分解：启发式期望 O(n^(1/4))，忽略对数因子；空间 O(log n)。
i64 mul(i64 a, i64 b, i64 m) {
    return i128(a) * b % m;
}

i64 power(i64 a, i64 b, i64 m) {
    i64 res = 1 % m;
    for (; b; b >>= 1, a = mul(a, a, m)) {
        if (b & 1) {
            res = mul(res, a, m);
        }
    }
    return res;
}

bool isprime(i64 n) {
    if (n < 2 || n % 2 == 0) {
        return n == 2;
    }
    // 这组底数覆盖全部 64 位整数，不要随意删减。
    static constexpr i64 A[] = {
        2, 325, 9375, 28178, 450775, 9780504, 1795265022
    };
    int s = __builtin_ctzll(n - 1);
    i64 d = (n - 1) >> s; // n - 1 = d * 2^s，d 为奇数。
    for (auto a : A) {
        if (a % n == 0) {
            continue;
        }
        i64 x = power(a % n, d, n);
        if (x == 1 || x == n - 1) {
            continue;
        }
        for (int i = 1; i < s && x != n - 1; i++) {
            x = mul(x, x, n);
        }
        if (x != n - 1) {
            return false;
        }
    }
    return true;
}

// 仅对合数调用，返回任意非平凡因子（不一定是质数）。
i64 rho(i64 n) {
    if (n % 2 == 0) {
        return 2;
    }
    static mt19937_64 rng(
        chrono::steady_clock::now().time_since_epoch().count());
    while (true) {
        i64 c = rng() % (n - 1) + 1;
        i64 y = rng() % (n - 1) + 1;
        auto g = [&](i64 x) -> i64 {
            return (i128(x) * x + c) % n; // 加法也需防溢出。
        };
        i64 x = 0, z = 0, d = 1;
        // Brent 倍增：固定 x，分段推进 y，每 128 步批量求 gcd。
        for (i64 len = 1; d == 1; len *= 2) {
            x = y;
            for (i64 k = 0; k < len && d == 1; k += 128) {
                z = y;
                i64 v = 1;
                for (i64 i = 0; i < min(128LL, len - k); i++) {
                    y = g(y);
                    v = mul(v, abs(x - y), n);
                }
                d = gcd(v, n);
            }
        }
        if (d == n) { // 批量乘积丢失信息，回到本块起点逐步检查。
            do {
                z = g(z);
                d = gcd(abs(x - z), n);
            } while (d == 1);
        }
        if (d < n) {
            return d;
        }
        // 仍失败则重新随机起点和常数 c。
    }
}

vector<i64> factorize(i64 x) {
    assert(x >= 1);
    vector<i64> p;
    auto dfs = [&](auto &&self, i64 n) -> void {
        if (n <= 10000) { // 小数直接试除。
            for (int i = 2; i * i <= n; i++) {
                for (; n % i == 0; n /= i) {
                    p.push_back(i);
                }
            }
            if (n > 1) {
                p.push_back(n);
            }
            return;
        }
        if (isprime(n)) {
            p.push_back(n);
            return;
        }
        i64 d = rho(n);
        self(self, d);
        self(self, n / d);
    };
    dfs(dfs, x);
    sort(p.begin(), p.end());
    return p;
}
