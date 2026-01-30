template <class Info, class Tag>
struct LazySegmentTree {
    int n;
    vector<Info> info;
    vector<Tag> tag;
    LazySegmentTree(): n(0) {}
    LazySegmentTree(int N, Info v = Info()) {
        init(vector<Info>(N + 1, v));
    }
    LazySegmentTree(const vector<Info> &a) {
        init(a);
    }
    void init(const vector<Info> &a) {
        n = int(a.size()) - 1;
        info.assign(n << 2, Info());
        tag.assign(n << 2, Tag());
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
    void apply(int id, const Tag &t) {
        info[id].apply(t);
        tag[id].apply(t);
    }
    void pushDown(int id) {
        apply(id * 2, tag[id]);
        apply(id * 2 + 1, tag[id]);
        tag[id] = Tag();
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
        pushDown(id);
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
        pushDown(id);
        int mid = (l + r) >> 1;
        return rangeQuery(id * 2, l, mid, x, y) + rangeQuery(id * 2 + 1, mid + 1, r, x, y);
    }
    Info rangeQuery(int l, int r) {
        return rangeQuery(1, 1, n, l, r);
    }
    void rangeApply(int id, int l, int r, int x, int y, const Tag &t) {
        if (x > r || y < l) {
            return;
        }
        if (x <= l && y >= r) {
            apply(id, t);
            return;
        }
        pushDown(id);
        int mid = (l + r) >> 1;
        rangeApply(id * 2, l, mid, x, y, t);
        rangeApply(id * 2 + 1, mid + 1, r, x, y, t);
        pushUp(id);
    }
    void rangeApply(int l, int r, const Tag &t) {
        rangeApply(1, 1, n, l, r, t);
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
        pushDown(id);
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
        pushDown(id);
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