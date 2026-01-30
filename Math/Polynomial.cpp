constexpr i64 mod = 998244353;
constexpr i64 G = 3;

i64 power(i64 a, i64 b = mod - 2) {
    i64 res = 1;
    while (b) {
        if (b & 1) {
            res = res * a % mod;
        }
        a = a * a % mod;
        b >>= 1;
    }
    return res;
}

template<i64 mod, i64 G>
struct PolyNTT {
    // NTT 整数模意义下的多项式乘法
    static void ntt(vector<i64>& a, bool invert) {
        int n = a.size();
        vector<int> rev(n);
        for (int i = 0; i < n; i++) {
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) ? (n >> 1) : 0);
            if (i < rev[i]) swap(a[i], a[rev[i]]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            i64 wlen = power(G, (mod - 1) / len);
            if (invert) wlen = power(wlen);
            for (int i = 0; i < n; i += len) {
                i64 w = 1;
                for (int j = 0; j < len / 2; j++) {
                    i64 u = a[i + j];
                    i64 v = a[i + j + len / 2] * w % mod;
                    a[i + j] = (u + v) % mod;
                    a[i + j + len / 2] = (u - v + mod) % mod;
                    w = w * wlen % mod;
                }
            }
        }
        if (invert) {
            i64 inv_n = power(n);
            for (i64 &x : a) x = x * inv_n % mod;
        }
    }
 
    // FFT 通用多项式乘法 (实现为 NTT)
    static void fft(vector<i64>& a, bool invert) {
        ntt(a, invert);
    }

    // FWT 子集卷积 / 集合运算
    static void fwt(vector<i64>& a, bool invert) {
        int n = a.size();
        for (int len = 1; len < n; len <<= 1) {
            for (int i = 0; i < n; i++) {
                if (i & len) continue;
                i64 x = a[i], y = a[i + len];
                if (invert) {
                    // IFWT for XOR
                    a[i] = (x + y) % mod * power(2) % mod;
                    a[i + len] = (x - y + mod) % mod * power(2) % mod;
                } else {
                    // FWT for XOR
                    a[i] = (x + y) % mod;
                    a[i + len] = (x - y + mod) % mod;
                }
            }
        }
    }
 
    // 多项式加法
    static vector<i64> add(vector<i64> a, vector<i64> b) {
        int n = max(a.size(), b.size());
        a.resize(n);
        b.resize(n);
        for (int i = 0; i < n; i++) {
            a[i] = (a[i] + b[i]) % mod;
        }
        return a;
    }
 
    // 多项式减法
    static vector<i64> sub(vector<i64> a, vector<i64> b) {
        int n = max(a.size(), b.size());
        a.resize(n);
        b.resize(n);
        for (int i = 0; i < n; i++) {
            a[i] = (a[i] - b[i] + mod) % mod;
        }
        return a;
    }
 
    // 多项式乘法
    static vector<i64> multiply(vector<i64> a, vector<i64> b) {
        if (a.empty() || b.empty()) return {};
        int res_deg = (int)a.size() + (int)b.size() - 2;
        int sz = 1;
        while (sz < (int)a.size() + (int)b.size()) sz <<= 1;
        a.resize(sz);
        b.resize(sz);
        ntt(a, false);
        ntt(b, false);
        for (int i = 0; i < sz; i++) a[i] = a[i] * b[i] % mod;
        ntt(a, true);
        a.resize(res_deg + 1);
        return a;
    }
 
    // 多项式除法（返回商）
    static vector<i64> divide(const vector<i64>& a, const vector<i64>& b) {
        int n = a.size(), m = b.size();
        if (n < m) return {0};
        vector<i64> ra = a;
        vector<i64> rb = b;
        reverse(ra.begin(), ra.end());
        reverse(rb.begin(), rb.end());
        vector<i64> rb_inv;
        poly_inv(rb, rb_inv, n - m + 1);
        vector<i64> q = multiply(ra, rb_inv);
        q.resize(n - m + 1);
        reverse(q.begin(), q.end());
        return q;
    }
 
    // 多项式模除（返回余数）
    static vector<i64> Mod(const vector<i64>& a, const vector<i64>& b) {
        if (a.size() < b.size()) return a;
        vector<i64> q = divide(a, b);
        vector<i64> r = sub(a, multiply(q, b));
        r.resize(min((int)r.size(), (int)b.size() - 1));
        return r;
    }
 
    // 多项式取逆，递归构造
    static void poly_inv(const vector<i64>& a, vector<i64>& b, int deg) {
        if (deg == 1) {
            b.assign(1, power(a[0]));
            return;
        }
        poly_inv(a, b, (deg + 1) / 2);
        int sz = 1;
        while (sz < 2 * deg) sz <<= 1;
        vector<i64> a_slice(a.begin(), a.begin() + min((int)a.size(), deg));
        a_slice.resize(sz);
        b.resize(sz);
        ntt(a_slice, false);
        ntt(b, false);
        for (int i = 0; i < sz; i++) {
            b[i] = (2 * b[i] - a_slice[i] * b[i] % mod * b[i] % mod + mod) % mod;
        }
        ntt(b, true);
        b.resize(deg);
    }
 
    // 多项式求导
    static vector<i64> derivative(const vector<i64>& a) {
        int n = a.size();
        if (n <= 1) return {};
        vector<i64> res(n - 1);
        for (int i = 1; i < n; i++) res[i - 1] = a[i] * i % mod;
        return res;
    }
 
    // 多项式积分
    static vector<i64> integral(const vector<i64>& a) {
        int n = a.size();
        vector<i64> res(n + 1);
        vector<i64> inv(n + 2);
        inv[1] = 1;
        for (int i = 2; i <= n + 1; i++) inv[i] = mod - (mod / i) * inv[mod % i] % mod;
        for (int i = 0; i < n; i++) res[i + 1] = a[i] * inv[i + 1] % mod;
        return res;
    }
 
    // 多项式对数
    static void ln(const vector<i64>& a, vector<i64>& b, int deg) {
        vector<i64> a_der = derivative(a);
        vector<i64> a_inv;
        poly_inv(a, a_inv, deg);
        vector<i64> t = multiply(a_der, a_inv);
        t.resize(deg - 1);
        b = integral(t);
        b.resize(deg);
    }
 
    // 多项式指数
    static void exp(const vector<i64>& a, vector<i64>& b, int deg) {
        if (deg == 1) {
            b.assign(1, 1);
            return;
        }
        exp(a, b, (deg + 1) / 2);
        b.resize(deg);
        vector<i64> ln_b;
        ln(b, ln_b, deg);
        vector<i64> a_slice(a.begin(), a.begin() + min((int)a.size(), deg));
        a_slice.resize(deg, 0);
        for (int i = 0; i < deg; i++) {
            ln_b[i] = (a_slice[i] - ln_b[i] + mod) % mod;
        }
        ln_b[0] = (ln_b[0] + 1) % mod;
        b = multiply(b, ln_b);
        b.resize(deg);
    }
 
    // 多点求值
    static void build_eval_tree(vector<vector<i64>>& tree, const vector<i64>& xs, int u, int l, int r) {
        if (r - l == 1) {
            tree[u] = {(mod - xs[l]) % mod, 1};
        } else {
            int m = (l + r) / 2;
            build_eval_tree(tree, xs, 2 * u, l, m);
            build_eval_tree(tree, xs, 2 * u + 1, m, r);
            tree[u] = multiply(tree[2 * u], tree[2 * u + 1]);
        }
    }
 
    static void fast_eval_rec(const vector<i64>& f, const vector<vector<i64>>& tree, vector<i64>& res, int u, int l, int r) {
        if (f.size() < 256) { // 小范围暴力求值优化
            for (int i = l; i < r; ++i) {
                i64 x = res[i];
                i64 y = 0, p = 1;
                for (i64 coeff : f) {
                    y = (y + coeff * p) % mod;
                    p = p * x % mod;
                }
                res[i] = y;
            }
            return;
        }
        if (r - l == 1) {
            res[l] = f.size() ? f[0] : 0;
            return;
        }
        int m = (l + r) / 2;
        vector<i64> rem_l = Mod(f, tree[2 * u]);
        vector<i64> rem_r = Mod(f, tree[2 * u + 1]);
        fast_eval_rec(rem_l, tree, res, 2 * u, l, m);
        fast_eval_rec(rem_r, tree, res, 2 * u + 1, m, r);
    }
 
    static vector<i64> fast_eval(const vector<i64>& f, const vector<i64>& xs) {
        int n = xs.size();
        if (n == 0) return {};
        vector<vector<i64>> tree(4 * n);
        build_eval_tree(tree, xs, 1, 0, n);
        vector<i64> res = xs;
        fast_eval_rec(f, tree, res, 1, 0, n);
        return res;
    }
 
    // 快速插值
    static vector<i64> interpolate(const vector<i64>& xs, const vector<i64>& ys) {
        int n = xs.size();
        if (n == 0) return {};
        vector<vector<i64>> tree(4 * n);
        build_eval_tree(tree, xs, 1, 0, n);
        vector<i64> all_poly = tree[1];
        vector<i64> der = derivative(all_poly);
        vector<i64> val = fast_eval(der, xs);
        vector<i64> weights(n);
        for (int i = 0; i < n; i++) {
            weights[i] = ys[i] * power(val[i]) % mod;
        }
 
        function<vector<i64>(int, int, int)> solve = [&](int u, int l, int r) -> vector<i64> {
            if (r - l == 1) {
                return {weights[l]};
            }
            int m = (l + r) / 2;
            vector<i64> left = solve(2 * u, l, m);
            vector<i64> right = solve(2 * u + 1, m, r);
            return add(multiply(left, tree[2 * u + 1]), multiply(right, tree[2 * u]));
        };
 
        return solve(1, 0, n);
    }
};

using Poly = PolyNTT<mod, G>;
using poly = vector<i64>;