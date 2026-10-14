#include <bits/stdc++.h>
using namespace std;
using i64 = long long;

#include "../templates/数据结构/MInt(Short Version).cpp"
#include "../templates/数学/Polynomial.cpp"

// 独立的 O(n^2) 参考实现；从仓库根目录运行：
// g++ -std=c++17 -O2 tests/polynomial_test.cpp -o polynomial_test
using Vec = vector<i64>;
mt19937_64 rng(20261014);
int checks = 0;

i64 norm(i64 x) {
    x %= mod;
    return x < 0 ? x + mod : x;
}
i64 modpow(i64 a, i64 k) {
    i64 b = 1;
    for (; k; k >>= 1, a = a * a % mod) {
        if (k & 1) b = b * a % mod;
    }
    return b;
}
void check(bool ok, const string &label) {
    checks++;
    if (!ok) throw runtime_error(label);
}
void equal(const Poly &a, const Vec &b, const string &label) {
    check(a.size() == b.size(), label + ": size");
    for (int i = 0; i < int(b.size()); i++) {
        check(a[i].x == b[i], label + ": coefficient " + to_string(i));
    }
}
Poly poly(const Vec &a) {
    return Poly(a.begin(), a.end());
}
Vec values(const Poly &a) {
    Vec b;
    for (auto x : a) b.push_back(x.x);
    return b;
}
Vec randomVec(int n) {
    Vec a(n);
    for (auto &x : a) x = rng() % mod;
    return a;
}
Vec cut(Vec a, int n) {
    a.resize(n);
    return a;
}
Vec multiply(const Vec &a, const Vec &b, int n) {
    Vec c(n);
    for (int i = 0; i < min(n, int(a.size())); i++) {
        for (int j = 0; j < int(b.size()) && i + j < n; j++) {
            c[i + j] = (c[i + j] + a[i] * b[j]) % mod;
        }
    }
    return c;
}
Vec inverse(const Vec &a, int n) {
    Vec b(n);
    if (!n) return b;
    b[0] = modpow(a[0], mod - 2);
    for (int i = 1; i < n; i++) {
        i64 sum = 0;
        for (int j = 1; j <= i && j < int(a.size()); j++) {
            sum = (sum + a[j] * b[i - j]) % mod;
        }
        b[i] = norm(-sum * b[0]);
    }
    return b;
}
Vec logarithm(const Vec &a, int n) {
    if (!n) return {};
    Vec d(max(0, int(a.size()) - 1));
    for (int i = 1; i < int(a.size()); i++) d[i - 1] = a[i] * i % mod;
    Vec c = multiply(d, inverse(a, n), n - 1), b(n);
    for (int i = 1; i < n; i++) b[i] = c[i - 1] * modpow(i, mod - 2) % mod;
    return b;
}
Vec exponential(const Vec &a, int n) {
    Vec b(n);
    if (!n) return b;
    b[0] = 1;
    for (int i = 1; i < n; i++) {
        i64 sum = 0;
        for (int j = 1; j <= i && j < int(a.size()); j++) {
            sum = (sum + j * a[j] % mod * b[i - j]) % mod;
        }
        b[i] = sum * modpow(i, mod - 2) % mod;
    }
    return b;
}
Vec integerPower(Vec a, i64 k, int n) {
    Vec b(n);
    if (!n) return b;
    b[0] = 1;
    for (; k; k >>= 1, a = multiply(a, a, n)) {
        if (k & 1) b = multiply(b, a, n);
    }
    return b;
}
void trim(Vec &a) {
    while (!a.empty() && !a.back()) a.pop_back();
}
pair<Vec, Vec> divide(Vec a, Vec b) {
    trim(a);
    trim(b);
    Vec q(max(0, int(a.size()) - int(b.size()) + 1));
    i64 inv = modpow(b.back(), mod - 2);
    while (a.size() >= b.size()) {
        int k = a.size() - b.size();
        i64 c = a.back() * inv % mod;
        q[k] = c;
        for (int j = 0; j < int(b.size()); j++) a[k + j] = norm(a[k + j] - c * b[j]);
        trim(a);
    }
    trim(q);
    return {q, a};
}

int main() {
    for (int n : {0, 1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 127, 129, 257}) {
        for (int trial = 0; trial < 8; trial++) {
            Vec a = randomVec(n), b = randomVec(rng() % 270);
            Poly A = poly(a), B = poly(b);
            int m = a.empty() || b.empty() ? 0 : int(a.size() + b.size()) - 1;
            equal(A * B, multiply(a, b, m), "convolution");
            Vec sum(max(a.size(), b.size())), diff(sum.size());
            for (int i = 0; i < int(sum.size()); i++) {
                i64 x = i < int(a.size()) ? a[i] : 0;
                i64 y = i < int(b.size()) ? b[i] : 0;
                sum[i] = norm(x + y);
                diff[i] = norm(x - y);
            }
            equal(A + B, sum, "addition");
            equal(A - B, diff, "subtraction");
            equal(A.cut(n + 5), cut(a, n + 5), "padding");
            equal(A.cut(n / 2), cut(a, n / 2), "truncation");
            equal(A.integr().deriv(), a, "calculus");
            equal(Z(3) * A, values(A * Z(3)), "scalar");

            Vec nonzero = a;
            if (nonzero.empty()) nonzero.push_back(1);
            nonzero[0] = rng() % (mod - 1) + 1;
            equal(poly(nonzero).inv(n), inverse(nonzero, n), "inverse");
            nonzero[0] = 1;
            equal(poly(nonzero).log(n), logarithm(nonzero, n), "log");
            equal(poly(nonzero).log(n).exp(n), cut(nonzero, n), "exp(log)");
            nonzero[0] = 0;
            equal(poly(nonzero).exp(n), exponential(nonzero, n), "exp");
            equal(poly(nonzero).exp(n).log(n), cut(nonzero, n), "log(exp)");

            i64 k = rng() % 9;
            equal(A.pow(k, n), integerPower(a, k, n), "power");
            if (n) a[0] = 0;
            if (n > 1 && trial % 2) a[1] = 0;
            equal(poly(a).pow(k, n), integerPower(a, k, n), "power leading zeros");

            Vec square = multiply(a, a, n);
            auto root = poly(square).sqrt(n);
            check(root.has_value(), "square root exists");
            check(int(root->size()) == n, "square root length");
            equal((*root * *root).cut(n), square, "square root identity");
            int t = 0;
            while (t < n && square[t] == 0) t++;
            if (t < n) {
                int shift = t / 2;
                Vec expected = cut(a, n);
                if (expected[shift] > mod / 2) {
                    for (auto &x : expected) x = norm(-x);
                }
                for (int i = n - shift; i < n; i++) expected[i] = 0;
                equal(*root, expected, "canonical square root");
            }

            if (b.empty()) b.push_back(1);
            b[0] = 1;
            a.push_back(0);
            b.push_back(0);
            auto [q, r] = poly(a).divmod(poly(b));
            auto [eq, er] = divide(a, b);
            equal(q, eq, "division quotient");
            equal(r, er, "division remainder");
            equal(poly(a) / poly(b), eq, "operator /");
            equal(poly(a) % poly(b), er, "operator %");
        }
    }
    for (int n : {1, 2, 4, 8, 16, 32, 64, 128, 512}) {
        Poly a = poly(randomVec(n)), b = a;
        Poly::ntt(b, false);
        if (n <= 64) {
            Z step = power(Z(3), (mod - 1) / n);
            for (int i = 0; i < n; i++) {
                Z sum = 0;
                for (int j = 0; j < n; j++) sum += a[j] * power(step, i * j);
                check(b[i] == sum, "direct DFT");
            }
        }
        Poly::ntt(b, true);
        equal(b, values(a), "NTT roundtrip");
    }
    for (int i = 0; i < 2000; i++) {
        Z x = rng() % mod;
        auto root = Poly::modSqrt(x * x);
        check(root && *root * *root == x * x, "modular square root");
        check(root->x <= mod / 2, "smaller modular root");
        if (x != 0) check(!Poly::modSqrt(Z(3) * x * x), "nonresidue");
    }
    check(Poly{}.sqrt(0).has_value(), "empty precision succeeds");
    equal(*Poly{}.sqrt(10), Vec(10), "zero square root");
    check(!Poly{3}.sqrt(1), "nonresidue constant");
    check(!Poly{0, 1}.sqrt(2), "odd valuation");
    equal(*Poly{0, 1}.sqrt(1), Vec(1), "ignore terms outside precision");
    equal(*Poly{0, 0, 4}.sqrt(4), Vec{0, 2, 0, 0}, "shifted root");
    equal(Poly{}.pow(0, 5), Vec{1, 0, 0, 0, 0}, "zero to zero");
    for (i64 k : {i64(mod - 1), i64(mod), i64(mod + 1), LLONG_MAX}) {
        for (const Vec &a : {Vec{2, 3, 4}, Vec{0, 0, 2}, Vec{1, 1}}) {
            equal(poly(a).pow(k, 35), integerPower(a, k, 35), "huge power");
        }
    }
    // 中等规模走完整 NTT 路径，检查牛顿迭代与低次系数。
    const int n = 4096;
    Poly a = poly(randomVec(n));
    a[0] = 1;
    equal((a * a.inv(n)).cut(n), values(Poly{1}.cut(n)), "large inverse");
    equal(a.log(n).exp(n), values(a), "large exp(log)");
    auto root = (a * a).cut(n).sqrt(n);
    check(root.has_value(), "large square root exists");
    equal(*root, values(a), "large square root");
    equal(a.pow(3, n), values(((a * a).cut(n) * a).cut(n)), "large power");
    cout << "PASS: " << checks << " checks; naive references, boundary cases and NTT identities.\n";
}
