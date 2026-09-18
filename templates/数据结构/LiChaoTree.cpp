// LiChaoTree<i64> 求最小值，LiChaoTree<i64, false> 求最大值。
// 横坐标为整数，定义域和线段范围均为闭区间。
// add(k, b) 插入直线；add(l, r, k, b) 插入仅在 [l, r] 有效的线段。
// query(x) 无有效直线时返回 T 的最大值（求最小值）或最低值（求最大值）。
// k * x + b 在 T 中计算，可能溢出时使用 i128。
template <class T = i64, bool isMin = true>
struct LiChaoTree {
    struct Line {
        T k = 0, b = 0;

        T get(int x) const {
            return k * T(x) + b;
        }
    };

    struct Node {
        Line line;
        int ls = 0, rs = 0;
        bool has = false;
    };

    int L, R, root;
    vector<Node> tr;

    LiChaoTree(int l = 0, int r = 0) {
        init(l, r);
    }

    void init(int l, int r) {
        assert(l <= r);
        L = l;
        R = r;
        root = 0;
        tr.assign(1, Node());
    }

    static bool better(T a, T b) {
        return isMin ? a < b : a > b;
    }

    static T emptyValue() {
        return isMin ? numeric_limits<T>::max() : numeric_limits<T>::lowest();
    }

    int newNode() {
        tr.emplace_back();
        return int(tr.size()) - 1;
    }

    int insert(int p, int l, int r, Line v) {
        if (p == 0) {
            p = newNode();
        }
        if (!tr[p].has) {
            tr[p].line = v;
            tr[p].has = true;
            return p;
        }
        int mid = int(l + (i64(r) - l) / 2);
        if (better(v.get(mid), tr[p].line.get(mid))) {
            swap(v, tr[p].line);
        }
        if (l == r) {
            return p;
        }
        if (better(v.get(l), tr[p].line.get(l))) {
            int ls = insert(tr[p].ls, l, mid, v);
            tr[p].ls = ls;
        } else if (better(v.get(r), tr[p].line.get(r))) {
            int rs = insert(tr[p].rs, mid + 1, r, v);
            tr[p].rs = rs;
        }
        return p;
    }

    int insertSegment(int p, int l, int r, int x, int y, Line v) {
        if (y < l || r < x) {
            return p;
        }
        if (x <= l && r <= y) {
            return insert(p, l, r, v);
        }
        if (p == 0) {
            p = newNode();
        }
        int mid = int(l + (i64(r) - l) / 2);
        int ls = insertSegment(tr[p].ls, l, mid, x, y, v);
        tr[p].ls = ls;
        int rs = insertSegment(tr[p].rs, mid + 1, r, x, y, v);
        tr[p].rs = rs;
        return p;
    }

    void add(T k, T b) {
        root = insert(root, L, R, {k, b});
    }

    void add(int l, int r, T k, T b) {
        if (l <= r) {
            root = insertSegment(root, L, R, l, r, {k, b});
        }
    }

    T query(int p, int l, int r, int x) const {
        if (p == 0) {
            return emptyValue();
        }
        T ans = tr[p].has ? tr[p].line.get(x) : emptyValue();
        if (l == r) {
            return ans;
        }
        int mid = int(l + (i64(r) - l) / 2);
        T res = x <= mid ? query(tr[p].ls, l, mid, x)
                        : query(tr[p].rs, mid + 1, r, x);
        return better(res, ans) ? res : ans;
    }

    T query(int x) const {
        assert(L <= x && x <= R);
        return query(root, L, R, x);
    }
};
