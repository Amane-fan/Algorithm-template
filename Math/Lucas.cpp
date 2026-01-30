struct Lucas {
    int p;
    vector<int> fact, invfact;

    static long long mod_pow(long long a, long long e, int mod) {
        long long r = 1 % mod;
        while (e > 0) {
            if (e & 1) r = (long long)((u128)r * a % mod);
            a = (long long)((u128)a * a % mod);
            e >>= 1;
        }
        return r;
    }

    static long long mod_inv(long long a, int mod) {
        return mod_pow((a % mod + mod) % mod, mod - 2, mod);
    }

    Lucas(int prime) {
        p = prime;
        fact.resize(p);
        invfact.resize(p);

        fact[0] = 1;
        for (int i = 1; i < p; ++i)
            fact[i] = (long long)fact[i - 1] * i % p;

        invfact[p - 1] = (int)mod_inv(fact[p - 1], p);
        for (int i = p - 2; i >= 0; --i)
            invfact[i] = (long long)invfact[i + 1] * (i + 1) % p;
    }

    int C_small(int n, int k) {
        if (k < 0 || k > n) return 0;
        return (long long)fact[n] * invfact[k] % p * invfact[n - k] % p;
    }

    int C(u64 n, u64 k) {
        if (k > n) return 0;
        int res = 1;
        while (n > 0 || k > 0) {
            int ni = (int)(n % p);
            int ki = (int)(k % p);
            int cur = C_small(ni, ki);
            if (cur == 0) return 0;
            res = (long long)res * cur % p;
            n /= p;
            k /= p;
        }
        return res;
    }
};