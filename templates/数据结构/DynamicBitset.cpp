// GNU C++17，依赖 vector、algorithm、cassert、ostream；using u64 = unsigned long long;
// 长度 n >= 0，下标 [0, n)，[] 只读，写入用 set(i, v)。
// 位运算要求长度相同，移位 k >= 0；resize 保留原有位，新增位填 v。
// 单点操作 O(1)，整体操作 O(ceil(n / 64))。
// DynamicBitset f(n); f.set(0); f |= f << k; // 01 背包（和 < n）
// 枚举 1：for (int i = f.find_first(); i < f.size(); i = f.find_next(i))
struct DynamicBitset {
    int n;
    vector<u64> a;

    explicit DynamicBitset(int n = 0, bool v = false)
        : n(n), a(n / 64 + (n % 64 != 0), v ? ~u64(0) : 0) {
        trim();
    }
    // 保证末块超出 n 的位为 0。
    void trim() {
        if (n % 64) {
            a.back() &= (u64(1) << (n % 64)) - 1;
        }
    }
    int size() const { return n; }
    void resize(int m, bool v = false) {
        a.resize(m / 64 + (m % 64 != 0), v ? ~u64(0) : 0);
        if (v && m > n && n % 64) {
            a[n / 64] |= ~u64(0) << (n % 64);
        }
        n = m;
        trim();
    }
    bool operator[](int i) const {
        return a[i / 64] >> (i % 64) & 1;
    }
    DynamicBitset &set(int i, bool v = true) {
        u64 mask = u64(1) << (i % 64);
        if (v) {
            a[i / 64] |= mask;
        } else {
            a[i / 64] &= ~mask;
        }
        return *this;
    }
    DynamicBitset &reset(int i) { return set(i, false); }
    DynamicBitset &flip(int i) {
        a[i / 64] ^= u64(1) << (i % 64);
        return *this;
    }
    DynamicBitset &set() {
        fill(a.begin(), a.end(), ~u64(0));
        trim();
        return *this;
    }
    DynamicBitset &reset() {
        fill(a.begin(), a.end(), 0);
        return *this;
    }
    DynamicBitset &flip() {
        for (auto &x : a) {
            x = ~x;
        }
        trim();
        return *this;
    }
    int count() const {
        int ans = 0;
        for (auto x : a) {
            ans += __builtin_popcountll(x);
        }
        return ans;
    }
    bool any() const {
        for (auto x : a) {
            if (x) {
                return true;
            }
        }
        return false;
    }
    bool none() const { return !any(); }
    bool all() const { return count() == n; }

    // 严格大于 i 的下一个 1，不存在返回 n；find_next(-1) 找第一个 1。
    int find_next(int i) const {
        if (i >= n - 1) {
            return n;
        }
        i++;
        int j = i / 64;
        u64 x = a[j] & (~u64(0) << (i % 64));
        while (true) {
            if (x) {
                return j * 64 + __builtin_ctzll(x);
            }
            if (++j == int(a.size())) {
                return n;
            }
            x = a[j];
        }
    }
    int find_first() const { return find_next(-1); }

    DynamicBitset &operator&=(const DynamicBitset &b) {
        assert(n == b.n);
        for (int i = 0; i < int(a.size()); i++) {
            a[i] &= b.a[i];
        }
        return *this;
    }
    DynamicBitset &operator|=(const DynamicBitset &b) {
        assert(n == b.n);
        for (int i = 0; i < int(a.size()); i++) {
            a[i] |= b.a[i];
        }
        return *this;
    }
    DynamicBitset &operator^=(const DynamicBitset &b) {
        assert(n == b.n);
        for (int i = 0; i < int(a.size()); i++) {
            a[i] ^= b.a[i];
        }
        return *this;
    }
    DynamicBitset &operator<<=(int k) {
        if (k >= n) {
            return reset();
        }
        int d = k / 64, s = k % 64;
        // 从高到低更新，避免覆盖尚未读取的块。
        for (int i = int(a.size()) - 1; i >= 0; i--) {
            u64 x = i >= d ? a[i - d] << s : 0;
            if (s && i > d) {
                x |= a[i - d - 1] >> (64 - s);
            }
            a[i] = x;
        }
        trim();
        return *this;
    }
    DynamicBitset &operator>>=(int k) {
        if (k >= n) {
            return reset();
        }
        int d = k / 64, s = k % 64, m = a.size();
        for (int i = 0; i < m; i++) {
            u64 x = i + d < m ? a[i + d] >> s : 0;
            if (s && i + d + 1 < m) {
                x |= a[i + d + 1] << (64 - s);
            }
            a[i] = x;
        }
        return *this;
    }
    friend DynamicBitset operator&(DynamicBitset a, const DynamicBitset &b) {
        a &= b;
        return a;
    }
    friend DynamicBitset operator|(DynamicBitset a, const DynamicBitset &b) {
        a |= b;
        return a;
    }
    friend DynamicBitset operator^(DynamicBitset a, const DynamicBitset &b) {
        a ^= b;
        return a;
    }
    friend DynamicBitset operator<<(DynamicBitset a, int k) {
        a <<= k;
        return a;
    }
    friend DynamicBitset operator>>(DynamicBitset a, int k) {
        a >>= k;
        return a;
    }
    friend DynamicBitset operator~(DynamicBitset a) {
        a.flip();
        return a;
    }
    friend bool operator==(const DynamicBitset &a, const DynamicBitset &b) {
        return a.n == b.n && a.a == b.a;
    }
    friend bool operator!=(const DynamicBitset &a, const DynamicBitset &b) {
        return !(a == b);
    }
    // cout << a：从高位到低位输出，保留前导 0，不附加换行。
    friend ostream &operator<<(ostream &os, const DynamicBitset &a) {
        for (int i = a.n - 1; i >= 0; i--) {
            os.put(a[i] ? '1' : '0');
        }
        return os;
    }
};
