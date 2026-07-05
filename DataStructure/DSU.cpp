struct DSU {
    vector<int> p, siz;
    DSU() {}
    DSU(int n) {
        init(n);
    }
    void init(int n) {
        p.resize(n + 1);
        iota(p.begin(), p.end(), 0);
        siz.assign(n + 1, 1);
    }
    int find(int x) {
        if (x != p[x]) {
            p[x] = find(p[x]);
        }
        return p[x];
    }
    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) return false;
        siz[x] += siz[y];
        p[y] = x;
        return true;
    }
    int size(int x) {
        return siz[find(x)];
    }
    bool same(int x, int y) {
        return find(x) == find(y);
    }
};