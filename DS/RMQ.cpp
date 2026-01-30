template<class T, class F>
struct RMQ {
    int n;
    vector<T> a;
    array<vector<T>, 20> f;
    F fun;
    RMQ() {}
    RMQ(const vector<T> &a, F &&fun_): fun(fun_) {
        this->a = a;
        this->n = int(a.size()) - 1;
        f.fill(vector<T>(n + 1));
        for (int i = 1; i <= n; i++) {
            f[0][i] = a[i];
        }
        for (int j = 1; j <= __lg(n); j++) {
            for (int i = 1; i + (1 << j) - 1 <= n; i++) {
                f[j][i] = fun(f[j - 1][i], f[j - 1][i + (1 << (j - 1))]);
            }
        }
    }
    T query(int l, int r) {
        int k = __lg(r - l + 1);
        return fun(f[k][l], f[k][r - (1 << k) + 1]);
    }
};