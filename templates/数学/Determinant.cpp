// 方阵下标 [1, n]，第 0 行/列不使用；按值传入，不修改原矩阵。
// 浮点版求实数行列式；模质数版常用于矩阵树定理、计数问题。
// 约定 0 阶行列式为 1，此时传入 vector(1, vector<T>(1))。
using F = long double;

// 列主元高斯消元，O(n^3) 时间，O(n^2) 空间。
// eps 用于判定主元为 0，需按数据尺度调整；不适合求大整数精确值。
F det(vector<vector<F>> a, F eps = 1e-12L) {
    int n = int(a.size()) - 1;
    F ans = 1;
    for (int i = 1; i <= n; i++) {
        int p = i;
        for (int j = i + 1; j <= n; j++) {
            if (fabs(a[j][i]) > fabs(a[p][i])) {
                p = j;
            }
        }
        if (fabs(a[p][i]) <= eps) {
            return 0;
        }
        if (p != i) {
            swap(a[p], a[i]);
            ans = -ans; // 交换两行，行列式变号。
        }
        ans *= a[i][i];
        for (int j = i + 1; j <= n; j++) {
            F t = a[j][i] / a[i][i];
            a[j][i] = 0;
            for (int k = i + 1; k <= n; k++) {
                a[j][k] -= t * a[i][k];
            }
        }
    }
    return ans;
}

// 模质数行列式，返回 [0, mod)；要求 mod 为质数且 2 <= mod <= INT_MAX。
// 依赖 using i64 = long long；乘法用 i64，输入可为负数。
// 时间 O(n^3 + n log mod)，空间 O(n^2)；合数模数不能使用此版本。
i64 detMod(vector<vector<i64>> a, int mod) {
    int n = int(a.size()) - 1;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            a[i][j] = (a[i][j] % mod + mod) % mod;
        }
    }
    // 模快速幂，用费马小定理求非零主元的逆元。
    auto power = [&](i64 x, int k) {
        i64 res = 1;
        for (; k; k >>= 1, x = x * x % mod) {
            if (k & 1) {
                res = res * x % mod;
            }
        }
        return res;
    };
    i64 ans = 1;
    for (int i = 1; i <= n; i++) {
        int p = i;
        while (p <= n && a[p][i] == 0) {
            p++;
        }
        if (p > n) {
            return 0;
        }
        if (p != i) {
            swap(a[p], a[i]);
            ans = (mod - ans) % mod;
        }
        ans = ans * a[i][i] % mod;
        i64 inv = power(a[i][i], mod - 2);
        for (int j = i + 1; j <= n; j++) {
            i64 t = a[j][i] * inv % mod;
            a[j][i] = 0;
            for (int k = i + 1; k <= n; k++) {
                a[j][k] = (a[j][k] - t * a[i][k] % mod + mod) % mod;
            }
        }
    }
    return ans;
}

// vector<vector<F>> a = {{0, 0, 0}, {0, 1, 2}, {0, 3, 4}};
// cout << det(a); // -2
// vector<vector<i64>> b = {{0, 0, 0}, {0, 1, 2}, {0, 3, 4}};
// cout << detMod(b, 998244353); // 998244351
