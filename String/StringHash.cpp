constexpr u64 mod = (1ull << 61) - 1;
mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
uniform_int_distribution<u64> dist(mod / 2, mod - 1);
const u64 base = dist(rnd);

struct StringHash {
    vector<u64> h;
    vector<u64> p;
    StringHash() {}
    StringHash(const string &s) {
        init(s);
    }
    static u64 add(u64 a, u64 b) {
        a += b;
        if (a >= mod) a -= mod;
        return a;
    }
    static u64 mul(u64 a, u64 b) {
        u128 c = u128(a) * b;
        return add(c >> 61, c & mod);
    }
    void init(const string &s) {
        int n = s.size() - 1;
        p.resize(n + 1);
        h.resize(n + 1);
        p[0] = 1;
        for (int i = 1; i <= n; i++) {
            p[i] = mul(p[i - 1], base);
            h[i] = mul(h[i - 1], base);
            h[i] = add(h[i], s[i]);
        }
    }
    u64 get(int l, int r) {
        return add(h[r], mod - mul(h[l - 1], p[r - l + 1]));
    } 
};