template <class Info>
struct SegmentTree {
    int n;
    vector<Info> info;
    SegmentTree(): n(0) {}
    SegmentTree(int N, Info v = Info()) {
        init(vector<Info>(N + 1, v));
    }
    SegmentTree(const vector<Info> &a) {
        init(a);
    }
    void init(const vector<Info> &a) {
        n = int(a.size()) - 1;
        info.assign(n << 2, Info());
        auto build = [&](auto &&self, int id, int l, int r) {
            if (l == r) {
                info[id] = a[l];
                return;
            }
            int mid = (l + r) >> 1;
            self(self, id * 2, l, mid);
            self(self, id * 2 + 1, mid + 1, r);
            pushUp(id);
        };
        build(build, 1, 1, n);
    }
    void pushUp(int id) {
        info[id] = info[id * 2] + info[id * 2 + 1];
    }
    void modify(int id, int l, int r, int x, const Info &v) {
        if (l == r) {
            info[id] = v;
            return;
        }
        int mid = (l + r) >> 1;
        if (x <= mid){
            modify(id * 2, l, mid, x, v);
        } else {
            modify(id * 2 + 1, mid + 1, r, x, v);
        }
        pushUp(id);
    }
    void modify(int x, const Info &v) {
        modify(1, 1, n, x, v);
    }
    Info rangeQuery(int id, int l, int r, int x, int y) {
        if (x > r || y < l) {
            return Info();
        }
        if (x <= l && y >= r) {
            return info[id];
        }
        int mid = (l + r) >> 1;
        return rangeQuery(id * 2, l, mid, x, y) + rangeQuery(id * 2 + 1, mid + 1, r, x, y);
    }
    Info rangeQuery(int l, int r) {
        return rangeQuery(1, 1, n, l, r);
    }
    template<class F>
    int findFirst(int id, int l, int r, int x, int y, F &&pred) {
        if (x > r || y < l) {
            return -1;
        }
        if (x <= l && y >= r && !pred(info[id])) {
            return -1;
        }
        if (l == r) {
            return l;
        }
        int mid = (l + r) >> 1;
        int res = findFirst(id * 2, l, mid, x, y, pred);
        if (res == -1) {
            res = findFirst(id * 2 + 1, mid + 1, r, x, y, pred);
        }
        return res;
    }
    template<class F>
    int findFirst(int l, int r, F &&pred) {
        return findFirst(1, 1, n, l, r, pred);
    }
    template<class F>
    int findLast(int id, int l, int r, int x, int y, F &&pred) {
        if (x > r || y < l) {
            return -1;
        }
        if (x <= l && y >= r && !pred(info[id])) {
            return -1;
        }
        if (l == r) {
            return l;
        }
        int mid = (l + r) >> 1;
        int res = findLast(id * 2 + 1, mid + 1, r, x, y, pred);
        if (res == -1) {
            res = findLast(id * 2, l, mid, x, y, pred);
        }
        return res;
    }
    template<class F>
    int findLast(int l, int r, F &&pred) {
        return findLast(1, 1, n, l, r, pred);
    }
};