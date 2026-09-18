// O(n) 预处理时间和空间，O(1) 查询。
// query(a, b) 要求 a, b >= 0 且 min(a, b) <= n。
// FastGCD gcd(1000000);      // 预处理到 10^6，n 可以为 0。
// int ans = gcd.query(12, 18); // 返回 6，两个参数可以交换。
// gcd.init(2000000);         // 重新预处理到 2 * 10^6。
// 也可先 FastGCD gcd; 再 gcd.init(n)。query(0, b) 返回 b。
struct FastGCD {
    int n, m;
    vector<array<int, 3>> f;
    vector<int> g;

    FastGCD(int N = 0) {
        init(N);
    }

    void init(int N) {
        assert(N >= 0);
        n = N;
        m = 0;
        while (1LL * (m + 1) * (m + 1) <= n) {
            m++;
        }
        f.assign(size_t(n) + 1, {1, 1, 1});

        vector<int> lp(size_t(n) + 1), prime;
        for (int i = 2; i <= n; i++) {
            if (lp[i] == 0) {
                lp[i] = i;
                prime.push_back(i);
            }
            for (int p : prime) {
                if (p > lp[i] || 1LL * i * p > n) {
                    break;
                }
                lp[i * p] = p;
            }
        }

        // 三个因子有序，且每个合数因子都不超过 sqrt(n)。
        for (int i = 2; i <= n; i++) {
            f[i] = f[i / lp[i]];
            f[i][0] *= lp[i];
            if (f[i][0] > f[i][1]) {
                swap(f[i][0], f[i][1]);
            }
            if (f[i][1] > f[i][2]) {
                swap(f[i][1], f[i][2]);
            }
        }

        g.assign(size_t(m + 1) * (m + 1), 0);
        for (int i = 0; i <= m; i++) {
            g[size_t(i) * (m + 1)] = g[i] = i;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= i; j++) {
                int d = g[size_t(j) * (m + 1) + i % j];
                g[size_t(i) * (m + 1) + j] = d;
                g[size_t(j) * (m + 1) + i] = d;
            }
        }
    }

    int query(int a, int b) const {
        assert(a >= 0 && b >= 0);
        if (a > b) {
            swap(a, b);
        }
        assert(a <= n);
        if (a == 0) {
            return b;
        }

        int ans = 1;
        for (int p : f[a]) {
            if (p == 1) {
                continue;
            }
            int d;
            if (p <= m) {
                d = g[size_t(p) * (m + 1) + b % p];
            } else {
                d = b % p == 0 ? p : 1;
            }
            ans *= d;
            b /= d;
        }
        return ans;
    }
};
