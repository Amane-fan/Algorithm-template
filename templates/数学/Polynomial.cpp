// GNU C++17；先粘贴短版 MInt，要求 mod = 998244353、Z = MInt<mod>。
// a[i] 为 x^i 的系数；Poly(n) 建 n 项，Poly{c} 建常数 c。
// cut / inv / log / exp / pow / sqrt 的 n 均为项数，结果恰好 n 项。
// n = 0 返回空；缺少的系数视为 0；trim() 删除末尾零，零多项式为空。
// NTT 长度至多 2^23；乘法要求 a.size() + b.size() - 1 <= 2^23。
// 各次内部乘法也受此限制；幂级数操作取 n <= 2^22 是安全上界。
// 求导、积分 O(n)；乘法、求逆、log、exp、pow、sqrt、除法 O(n log n)。
// pow 另有 O(log k)，sqrt 另有 O(log^2 mod)；空间 O(n)。
static_assert(mod == 998244353);

struct Poly : vector<Z> {
    using vector<Z>::vector;

    // 取前 n 项，不足补 0；等价于模 x^n，并保证结果长度为 n。
    Poly cut(int n) const {
        assert(n >= 0);
        Poly a(begin(), begin() + min(n, int(size())));
        a.resize(n);
        return a;
    }
    // 原地删除高次项末尾的 0；之后 size()-1 为次数，空数组表示零。
    void trim() {
        while (!empty() && back() == 0) {
            pop_back();
        }
    }
    // 逐项相加，结果长度取两者最大值，不删除末尾零。
    friend Poly operator+(Poly a, const Poly &b) {
        a.resize(max(a.size(), b.size()));
        for (int i = 0; i < int(b.size()); i++) {
            a[i] += b[i];
        }
        return a;
    }
    // 逐项相减，结果长度取两者最大值，不删除末尾零。
    friend Poly operator-(Poly a, const Poly &b) {
        a.resize(max(a.size(), b.size()));
        for (int i = 0; i < int(b.size()); i++) {
            a[i] -= b[i];
        }
        return a;
    }
    // 多项式乘常数：每个系数乘 c，长度不变。
    friend Poly operator*(Poly a, Z c) {
        for (auto &x : a) {
            x *= c;
        }
        return a;
    }
    // 常数乘多项式，与 a * c 等价。
    friend Poly operator*(Z c, Poly a) { return a * c; }

    // 原地 NTT，invert=false 为正变换，true 为逆变换；长度须为 2 的幂。
    // 逆变换用正变换后反转下标 1..n-1，再乘 n 的逆元实现。
    static void ntt(Poly &a, bool invert) {
        int n = a.size();
        assert(n > 0 && (n & (n - 1)) == 0 && n <= (1 << 23));
        for (int i = 1, j = 0; i < n; i++) {
            int k = n >> 1;
            for (; j & k; k >>= 1) {
                j ^= k;
            }
            j ^= k;
            if (i < j) {
                std::swap(a[i], a[j]);
            }
        }
        for (int len = 2; len <= n; len *= 2) {
            Z step = power(Z(3), (mod - 1) / len);
            for (int i = 0; i < n; i += len) {
                Z w = 1;
                for (int j = 0; j < len / 2; j++, w *= step) {
                    Z u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                }
            }
        }
        if (invert) {
            reverse(a.begin() + 1, a.end());
            Z invn = Z(n).inv();
            for (auto &x : a) {
                x *= invn;
            }
        }
    }
    // 完整卷积：c[k] = sum(a[i]*b[k-i])，非空时长度为 |a|+|b|-1。
    // 任一为空则返回空；保留前 n 项可写 (a*b).cut(n)。
    friend Poly operator*(Poly a, Poly b) {
        if (a.empty() || b.empty()) {
            return {};
        }
        int n = int(a.size() + b.size()) - 1;
        assert(n <= (1 << 23));
        if (min(a.size(), b.size()) <= 32) { // 小规模直接卷积。
            Poly c(n);
            for (int i = 0; i < int(a.size()); i++) {
                for (int j = 0; j < int(b.size()); j++) {
                    c[i + j] += a[i] * b[j];
                }
            }
            return c;
        }
        int m = 1;
        while (m < n) {
            m *= 2;
        }
        a.resize(m);
        b.resize(m);
        ntt(a, false);
        ntt(b, false);
        for (int i = 0; i < m; i++) {
            a[i] *= b[i];
        }
        ntt(a, true);
        a.resize(n);
        return a;
    }

    // 求导：b[i-1] = i*a[i]；非空时长度减 1，常数与零的导数为空。
    Poly deriv() const {
        int n = size();
        Poly b(max(0, n - 1));
        for (int i = 1; i < n; i++) {
            b[i - 1] = (*this)[i] * i;
        }
        return b;
    }
    // 积分：b[i+1] = a[i]/(i+1)，b[0]=0；长度加 1，要求 size()<mod。
    Poly integr() const {
        int n = size();
        assert(n < mod);
        static vector<Z> inv{0, 1};
        for (int i = inv.size(); i <= n; i++) {
            inv.push_back(-Z(mod / i) * inv[mod % i]);
        }
        Poly b(n + 1);
        for (int i = 0; i < n; i++) {
            b[i + 1] = (*this)[i] * inv[i + 1];
        }
        return b;
    }
    // 幂级数求逆：返回 B，使 A*B = 1 (mod x^n)；n>0 时要求 a[0]!=0。
    Poly inv(int n) const {
        assert(n >= 0);
        if (n == 0) {
            return {};
        }
        assert(!empty() && (*this)[0] != 0);
        Poly b{(*this)[0].inv()};
        for (int m = 1; m < n; m *= 2) {
            int k = min(2 * m, n);
            Poly c = (cut(k) * b).cut(k) * Z(-1);
            c[0] += 2;
            b = (b * c).cut(k); // B <- B * (2 - A * B)。
        }
        return b;
    }
    // 返回 log A 的前 n 项；n>0 时要求 a[0]=1，结果常数项为 0。
    // 公式：log A = integral(A'/A)。
    Poly log(int n) const {
        assert(n >= 0);
        if (n == 0) {
            return {};
        }
        assert(!empty() && (*this)[0] == 1);
        return (cut(n).deriv() * inv(n)).cut(n - 1).integr();
    }
    // 返回 exp A 的前 n 项；n>0 时要求 a[0]=0，结果常数项为 1。
    // 空数组视为零多项式，可直接调用。
    Poly exp(int n) const {
        assert(n >= 0);
        if (n == 0) {
            return {};
        }
        assert(empty() || (*this)[0] == 0);
        Poly b{1};
        for (int m = 1; m < n; m *= 2) {
            int k = min(2 * m, n);
            Poly c = cut(k) - b.log(k);
            c[0] += 1;
            b = (b * c).cut(k); // B <- B * (1 + A - log B)。
        }
        return b;
    }
    // 返回 A^k 的前 n 项，k>=0；允许前导零和任意常数项。
    // 约定 A^0=1，包括 0^0；指数 k 使用 i64，不要提前对 mod 取模。
    Poly pow(i64 k, int n) const {
        assert(k >= 0 && n >= 0);
        if (n == 0) {
            return {};
        }
        if (k == 0) {
            return Poly{1}.cut(n);
        }
        int t = 0;
        while (t < min(n, int(size())) && (*this)[t] == 0) {
            t++;
        }
        if (t == min(n, int(size())) || (t && k > (n - 1) / t)) {
            return Poly(n);
        }
        // A = x^t * c * B，B[0] = 1；先判断位移，避免 t*k 溢出。
        int shift = t * k, m = n - shift;
        Z c = (*this)[t];
        Poly a(begin() + t, begin() + min(int(size()), t + m));
        Poly b = ((a * c.inv()).log(m) * Z(k)).exp(m) * power(c, k);
        b.insert(b.begin(), shift, Z(0));
        return b;
    }

    // 求 r^2=a (mod mod)，无解返回 nullopt；有解取两个根中较小者。
    // Tonelli-Shanks 算法，供幂级数开平方求首个非零系数的根。
    static optional<Z> modSqrt(Z a) {
        if (a == 0) {
            return Z(0);
        }
        if (power(a, (mod - 1) / 2) != 1) {
            return nullopt;
        }
        // mod - 1 = 119 * 2^23，3 是二次非剩余。
        int s = 23;
        Z c = power(Z(3), 119), x = power(a, 60), t = power(a, 119);
        while (t != 1) {
            int i = 0;
            Z y = t;
            while (y != 1) {
                y *= y;
                i++;
            }
            Z b = power(c, 1LL << (s - i - 1));
            x *= b;
            c = b * b;
            t *= c;
            s = i;
        }
        return x.x <= (-x).x ? x : -x;
    }
    // 返回 n 项的 B，使 B^2=A (mod x^n)，无解返回 nullopt。
    // 自动处理前导零；首个非零系数取较小模根，自由的高位补 0。
    optional<Poly> sqrt(int n) const {
        assert(n >= 0);
        if (n == 0) {
            return Poly{};
        }
        int t = 0;
        while (t < min(n, int(size())) && (*this)[t] == 0) {
            t++;
        }
        if (t == min(n, int(size()))) {
            return Poly(n);
        }
        if (t & 1) {
            return nullopt;
        }
        auto r = modSqrt((*this)[t]);
        if (!r) {
            return nullopt;
        }
        Poly a(begin() + t, begin() + min(n, int(size()))), b{*r};
        int m = n - t;
        for (int len = 1; len < m; len *= 2) {
            int k = min(2 * len, m);
            b = (b + (a.cut(k) * b.inv(k)).cut(k)) * Z((mod + 1) / 2);
        }
        b.insert(b.begin(), t / 2, Z(0));
        b.resize(n);
        return b;
    }

    // 多项式带余除法：返回 {Q,R}，使 A=Q*B+R 且 deg R<deg B。
    // B 必须非零；允许输入有末尾零，返回的商与余数均 trim。
    pair<Poly, Poly> divmod(Poly b) const {
        Poly a = *this;
        a.trim();
        b.trim();
        assert(!b.empty());
        if (a.size() < b.size()) {
            return {{}, a};
        }
        int n = int(a.size() - b.size()) + 1;
        Poly ra(a.rbegin(), a.rend()), rb(b.rbegin(), b.rend());
        Poly q = (ra.cut(n) * rb.inv(n)).cut(n);
        reverse(q.begin(), q.end());
        Poly r = (a - b * q).cut(int(b.size()) - 1);
        q.trim();
        r.trim();
        return {q, r};
    }
    // 返回带余除法的商，b 必须非零；不是幂级数除法 a*b.inv(n)。
    friend Poly operator/(const Poly &a, const Poly &b) {
        return a.divmod(b).first;
    }
    // 返回带余除法的余数；若同时需要商和余数，只调用一次 divmod。
    friend Poly operator%(const Poly &a, const Poly &b) {
        return a.divmod(b).second;
    }
};

/*
Poly a = {1, 2, 3}, b = {1, 1};
auto c = a * b;             // {1, 3, 5, 3}
auto d = a.inv(4);          // {1, -2, 1, 4}，负数以模意义表示。
auto e = a.log(4).exp(4);   // a.cut(4)
auto f = a.pow(5, 10);      // a^5 的前 10 项。
auto [q, r] = a.divmod(b);  // q = {-1, 3}，r = {2}。
auto s = a.sqrt(4);         // optional<Poly>，先判断再取值。
if (s) { auto root = *s; }  // root * root 的前 4 项等于 a.cut(4)。
*/
