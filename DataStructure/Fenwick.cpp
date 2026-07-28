template <class T>
struct Fenwick {
    int n;
    vector<T> a;
    Fenwick() {}
    Fenwick(int N): n(N), a(N + 1) {}
    void add(int x, const T &v) {
        for (int i = x; i <= n; i += (i & -i)) {
            a[i] = a[i] + v;
        }
    }
    void set(int x, const T &v) {
        add(x, v - sum(x, x));
    }
    T sum(int x) {
        T ans{};
        for (int i = x; i > 0; i -= (i & -i)) {
            ans = ans + a[i];
        }
        return ans;
    }
    T sum(int l, int r) {
        if (l > r) {
            return T{0};
        }
        return sum(r) - sum(l - 1);
    }
    int lower_bound(T v) {
        int x = 0;
        for (int i = 1 << __lg(n); i > 0; i >>= 1) {
            if (x + i <= n && a[x + i] < v) {
                x += i;
                v = v - a[x];
            }
        }
        return x + 1;
    }
    int upper_bound(T v) {
        int x = 0;
        for (int i = 1 << __lg(n); i > 0; i >>= 1) {
            if (x + i <= n && a[x + i] <= v) {
                x += i;
                v = v - a[x];
            }
        }
        return x + 1;
    }
};