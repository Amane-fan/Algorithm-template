// 静态矩阵：预处理时间、空间 O(nm log n log m)，查询 O(1)，不支持单点修改。
// a 为 (n + 1) * (m + 1) 的矩阵，有效下标从 1 开始，n, m >= 1。
// auto op = [](int x, int y) { return min(x, y); }; // 求最大值时改为 max。
// RMQ2D<int, decltype(op)> st(a, op);
// int ans = st.query(x1, y1, x2, y2); // 闭区间 [x1, x2] * [y1, y2]。
// st.init(a); // 矩阵变化后重新预处理，沿用原合并函数。
// op 需满足结合律、交换律和幂等性，例如 min/max/gcd，不能用于求和。
template <class T, class F>
struct RMQ2D {
    int n, m;
    vector<int> lg;
    vector<vector<vector<vector<T>>>> f;
    F fun;

    RMQ2D(const vector<vector<T>> &a, F fun_) : fun(fun_) {
        init(a);
    }

    void init(const vector<vector<T>> &a) {
        assert(a.size() >= 2 && a[1].size() >= 2);
        n = int(a.size()) - 1;
        m = int(a[1].size()) - 1;
        for (int i = 1; i <= n; i++) {
            assert(int(a[i].size()) == m + 1);
        }
        lg.assign(max(n, m) + 1, 0);
        for (int i = 2; i < int(lg.size()); i++) {
            lg[i] = lg[i / 2] + 1;
        }

        f.assign(lg[n] + 1, vector<vector<vector<T>>>(lg[m] + 1));
        // f[p][q][i][j] 表示以 (i, j) 为左上角、大小为 2^p * 2^q 的矩形。
        for (int p = 0; p <= lg[n]; p++) {
            for (int q = 0; q <= lg[m]; q++) {
                int rows = n - (1 << p) + 1;
                int cols = m - (1 << q) + 1;
                f[p][q].assign(rows + 1, vector<T>(cols + 1));
                for (int i = 1; i <= rows; i++) {
                    for (int j = 1; j <= cols; j++) {
                        if (p == 0 && q == 0) {
                            f[p][q][i][j] = a[i][j];
                        } else if (p == 0) {
                            int d = 1 << (q - 1);
                            f[p][q][i][j] = fun(f[p][q - 1][i][j],
                                               f[p][q - 1][i][j + d]);
                        } else {
                            int d = 1 << (p - 1);
                            f[p][q][i][j] = fun(f[p - 1][q][i][j],
                                               f[p - 1][q][i + d][j]);
                        }
                    }
                }
            }
        }
    }

    T query(int x1, int y1, int x2, int y2) const {
        assert(1 <= x1 && x1 <= x2 && x2 <= n);
        assert(1 <= y1 && y1 <= y2 && y2 <= m);
        int p = lg[x2 - x1 + 1], q = lg[y2 - y1 + 1];
        int x = x2 - (1 << p) + 1, y = y2 - (1 << q) + 1;
        return fun(fun(f[p][q][x1][y1], f[p][q][x1][y]),
                   fun(f[p][q][x][y1], f[p][q][x][y]));
    }
};
