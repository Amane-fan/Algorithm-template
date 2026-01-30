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
    void modify(int x, const T &v) {
        add(x, v - rangeQuery(x, x));
    }
    T query(int x) {
        T ans {};
        for (int i = x; i > 0; i -= (i & -i)) {
            ans = ans + a[i];
        }
        return ans;
    }
    T rangeQuery(int l, int r) {
        return query(r) - query(l - 1);
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