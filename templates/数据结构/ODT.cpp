// 珂朵莉树：用 set 保存若干连续等值段，按段暴力操作。
// 适用：等值段较少，或有频繁的大区间赋值、数据随机的区间操作题。
// 若每次遍历后都赋值并删除这些段，可按段的增删次数作均摊分析。
// 区间加、独立查询不会减少段数；一般情形无对数保证，构造数据可卡。
// S 为当前段数，K 为区间内段数：assign/add/sum 为 O(log S + K)，
// kth 为 O(log S + K log K)；最坏 K = n，空间 O(n)。
// GNU C++17，依赖常用 STL 和 using i64 = long long;
// 下标 [1, n]，操作为闭区间，要求 1 <= l <= r <= n，n < INT_MAX。
// 值及区间和须在 i64 范围内；查询也会分裂区间，不自动合并相邻等值段。
struct ODT {
    struct Node {
        int l, r;
        mutable i64 v; // set 只按 l 排序，允许直接修改 v。
        bool operator<(const Node &b) const { return l < b.l; }
    };
    int n;
    set<Node> s;

    ODT(int n = 0, i64 v = 0) { init(n, v); }
    ODT(const vector<i64> &a) { init(a); }
    void init(int N, i64 v = 0) {
        n = N;
        s.clear();
        if (n) {
            s.insert({1, n, v});
        }
    }
    // a[0] 不使用，初始化时合并相邻等值元素。
    void init(const vector<i64> &a) {
        n = int(a.size()) - 1;
        s.clear();
        for (int l = 1, r; l <= n; l = r + 1) {
            r = l;
            while (r < n && a[r + 1] == a[l]) {
                r++;
            }
            s.insert(s.end(), {l, r, a[l]});
        }
    }
    // 返回以 x 为左端点的段；split(n + 1) 返回 end()。
    set<Node>::iterator split(int x) {
        if (x == n + 1) {
            return s.end();
        }
        auto it = s.lower_bound({x, 0, 0});
        if (it != s.end() && it->l == x) {
            return it;
        }
        auto [l, r, v] = *prev(it);
        s.erase(prev(it));
        s.insert({l, x - 1, v});
        return s.insert({x, r, v}).first;
    }
    // 必须先 split(r + 1)，再 split(l)，否则左迭代器可能失效。
    void assign(int l, int r, i64 v) {
        auto itr = split(r + 1), itl = split(l);
        s.erase(itl, itr);
        s.insert({l, r, v});
    }
    void add(int l, int r, i64 v) {
        auto itr = split(r + 1), itl = split(l);
        for (auto it = itl; it != itr; it++) {
            it->v += v;
        }
    }
    i64 sum(int l, int r) {
        auto itr = split(r + 1), itl = split(l);
        i64 ans = 0;
        for (auto it = itl; it != itr; it++) {
            ans += (it->r - it->l + 1) * it->v;
        }
        return ans;
    }
    // 按元素个数计重，第 k 小，要求 1 <= k <= r - l + 1。
    i64 kth(int l, int r, int k) {
        assert(1 <= k && k <= r - l + 1);
        auto itr = split(r + 1), itl = split(l);
        vector<pair<i64, int>> a;
        for (auto it = itl; it != itr; it++) {
            a.push_back({it->v, it->r - it->l + 1});
        }
        sort(a.begin(), a.end());
        for (auto [v, cnt] : a) {
            if (k <= cnt) {
                return v;
            }
            k -= cnt;
        }
        return 0;
    }
};

// ODT tr(n);                  // 初始全 0；ODT tr(n, v) 初始全 v。
// vector<i64> a = {0, 3, 1, 1, 5};
// ODT tr(a);                  // a[1..4]，a[0] 不用。
// tr.assign(2, 3, 4);         // [3, 4, 4, 5]
// tr.add(1, 2, -1);           // [2, 3, 4, 5]
// cout << tr.sum(1, 4);       // 14
// cout << tr.kth(1, 4, 2);    // 3
// 自定义统计：先分裂，再遍历 [itl, itr)，每段贡献由 v 和 r-l+1 计算。
