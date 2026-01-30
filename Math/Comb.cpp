constexpr int mod = 1e9 + 7;
i64 power(i64 a, i64 b) {
    i64 res = 1;
    while (b) {
        if (b & 1) res = res * a % mod;
        b >>= 1;
        a = a * a % mod;
    }
    return res;
}


struct Comb {
    int n;
    vector<i64> _fac, _infac, _inv;
    Comb(): n{0}, _fac{1}, _infac{1}, _inv{0} {}
    Comb(int m): Comb() {
        init(m);
    }
    void init(int m) {
        if (m <= n) return;
        _fac.resize(m + 1);
        _infac.resize(m + 1);
        _inv.resize(m + 1);
        for (int i = n + 1; i <= m; i++) {
            _fac[i] = _fac[i - 1] * i % mod;
        }
        _infac[m] = power(_fac[m], mod - 2, mod);
        for (int i = m; i > n; i--) {
            _infac[i - 1] = _infac[i] * i % mod;
            _inv[i] = _infac[i] * _fac[i - 1] % mod;
        }
        n = m;
    }
    i64 fac(int m) {
        if (m > n) init(m * 2);
        return _fac[m];
    }
    i64 infac(int m) {
        if (m > n) init(m * 2);
        return _infac[m];
    }
    i64 inv(int m) {
        if (m > n) init(m * 2);
        return _inv[m];
    }
    i64 binom(int a, int b) {
        if (a < b || b < 0) return 0ll;
        return fac(a) * infac(a - b) % mod * infac(b) % mod;
    }
} comb;