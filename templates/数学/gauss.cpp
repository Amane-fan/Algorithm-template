// 解 n 个方程、m 个未知数的实数线性方程组，允许 n != m。
// 增广矩阵 a[n+1][m+2]：a[i][1..m] 为系数，a[i][m+1] 为常数。
// 第 0 行/列不使用，但第 0 行也要分配 m+2 个元素以确定未知数个数。
// 按值传入，不修改原矩阵；返回 -1 无解、0 无穷多解、1 唯一解。
// 有解时 ans[1..m] 为一组解（自由变量取 0）；无解时 ans 清空。
// 时间 O(n*m*min(n,m) + m^2)，空间 O(nm + m)。
using F = long double;

// 列主元消元 + 回代；eps 为绝对判零阈值，需按数据尺度调整。
int gauss(vector<vector<F>> a, vector<F> &ans, F eps = 1e-12L) {
    int n = int(a.size()) - 1, m = int(a[0].size()) - 2;
    vector<int> pos(m + 1); // pos[j]：第 j 个未知数对应的主元行，0 表示自由变量。
    int row = 1;
    for (int col = 1; col <= m && row <= n; col++) {
        int p = row;
        for (int i = row + 1; i <= n; i++) {
            if (fabs(a[i][col]) > fabs(a[p][col])) {
                p = i;
            }
        }
        if (fabs(a[p][col]) <= eps) {
            continue;
        }
        swap(a[p], a[row]);
        pos[col] = row;
        for (int i = row + 1; i <= n; i++) {
            F t = a[i][col] / a[row][col];
            a[i][col] = 0;
            for (int j = col + 1; j <= m + 1; j++) {
                a[i][j] -= t * a[row][j];
            }
        }
        row++;
    }
    for (int i = row; i <= n; i++) {
        if (fabs(a[i][m + 1]) > eps) { // 0 = 非零常数。
            ans.clear();
            return -1;
        }
    }
    ans.assign(m + 1, 0);
    for (int j = m; j >= 1; j--) {
        if (pos[j] == 0) {
            continue;
        }
        int i = pos[j];
        F v = a[i][m + 1];
        for (int k = j + 1; k <= m; k++) {
            v -= a[i][k] * ans[k];
        }
        ans[j] = v / a[i][j];
    }
    return row - 1 == m ? 1 : 0;
}

// vector<vector<F>> a(n + 1, vector<F>(m + 2));
// 读入每行 a[i][1..m+1] 后：vector<F> ans; int res = gauss(a, ans);
// res == -1 时无解；否则可使用 ans[1..m]，res == 0 表示还有其他解。
// 示例：2x+y=5，x-y=1，得到 ans[1]=2，ans[2]=1。
