// 单点加 / 单点赋值 / 矩形和，适合坐标范围较小的二维网格。
// 下标 [1, n] × [1, m]，矩形为闭区间；初始全 0。
// 单次操作 O(log n * log m)，空间 O(nm)；大坐标稀疏点集不宜直接使用。
// T{} 为加法零元，支持 +、-；整数求和通常使用 i64。
template <class T>
struct Fenwick2D {
    int n, m;
    vector<vector<T>> a;

    Fenwick2D(int n = 0, int m = 0) { init(n, m); }
    void init(int N, int M) {
        n = N;
        m = M;
        a.assign(n + 1, vector<T>(m + 1));
    }
    // (x, y) 加 v，要求 1 <= x <= n，1 <= y <= m。
    void add(int x, int y, const T &v) {
        for (int i = x; i <= n; i += (i & -i)) {
            for (int j = y; j <= m; j += (j & -j)) {
                a[i][j] = a[i][j] + v;
            }
        }
    }
    void set(int x, int y, const T &v) {
        add(x, y, v - sum(x, y, x, y));
    }
    // [1, x] × [1, y] 的和，允许 x = 0 或 y = 0，返回 0。
    T sum(int x, int y) const {
        T ans{};
        for (int i = x; i > 0; i -= (i & -i)) {
            for (int j = y; j > 0; j -= (j & -j)) {
                ans = ans + a[i][j];
            }
        }
        return ans;
    }
    // [x1, x2] × [y1, y2] 的和，空矩形返回 0。
    T sum(int x1, int y1, int x2, int y2) const {
        if (x1 > x2 || y1 > y2) {
            return T{};
        }
        return sum(x2, y2) - sum(x1 - 1, y2)
             - sum(x2, y1 - 1) + sum(x1 - 1, y1 - 1);
    }
};

// Fenwick2D<i64> tr(n, m);
// tr.add(2, 3, 5);          // (2, 3) 加 5。
// tr.set(2, 3, 7);          // (2, 3) 赋值为 7。
// cout << tr.sum(2, 3);     // 左上角 [1, 2] × [1, 3] 的和，为 7。
// cout << tr.sum(2, 2, 3, 4); // 行 [2, 3]、列 [2, 4] 的和，为 7。
