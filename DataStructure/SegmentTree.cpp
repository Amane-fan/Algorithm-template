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
    void set(int id, int l, int r, int x, const Info &v) {
        if (l == r) {
            info[id] = v;
            return;
        }
        int mid = (l + r) >> 1;
        if (x <= mid){
            set(id * 2, l, mid, x, v);
        } else {
            set(id * 2 + 1, mid + 1, r, x, v);
        }
        pushUp(id);
    }
    void set(int x, const Info &v) {
        set(1, 1, n, x, v);
    }
    Info get(int x) {
        return prod(x, x);
    }
    Info prod(int id, int l, int r, int x, int y) {
        if (x > r || y < l) {
            return Info();
        }
        if (x <= l && y >= r) {
            return info[id];
        }
        int mid = (l + r) >> 1;
        return prod(id * 2, l, mid, x, y) + prod(id * 2 + 1, mid + 1, r, x, y);
    }
    Info prod(int l, int r) {
        return prod(1, 1, n, l, r);
    }
    template<class F>
    int minLeft(int id, int l, int r, int x, int y, F &&pred) {
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
        int res = minLeft(id * 2, l, mid, x, y, pred);
        if (res == -1) {
            res = minLeft(id * 2 + 1, mid + 1, r, x, y, pred);
        }
        return res;
    }
    template<class F>
    int minLeft(int l, int r, F &&pred) {
        return minLeft(1, 1, n, l, r, pred);
    }
    template<class F>
    int maxRight(int id, int l, int r, int x, int y, F &&pred) {
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
        int res = maxRight(id * 2 + 1, mid + 1, r, x, y, pred);
        if (res == -1) {
            res = maxRight(id * 2, l, mid, x, y, pred);
        }
        return res;
    }
    template<class F>
    int maxRight(int l, int r, F &&pred) {
        return maxRight(1, 1, n, l, r, pred);
    }
};

struct Info {
    bool status;

    Info(): staus(false) {}

    friend Info operator+(const Info &a, const Info &b) {
        if (!a.status) {
            return b;
        }
        if (!b.status) {
            return a;
        }
        Info c;
        c.status = true;

        

        return c;
    }
    
};
