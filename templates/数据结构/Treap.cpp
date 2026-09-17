mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());

template <class T>
struct Treap {

    struct Node {
        T val;
        int l;
        int r;
        int siz;
        int pri;
        Node(const T &v): val(v), l(0), r(0), siz(1), pri(rnd()) {}
    };

    vector<Node> tr;
    int root;

    Treap() {
        root = newNode(0);
        tr[0].siz = 0;
    }

    int newNode(const T &v) {
        tr.push_back(Node(v));
        return (int)tr.size() - 1;
    }

    void pushUp(int u) {
        if (u == 0) {
            return;
        }
        tr[u].siz = tr[tr[u].l].siz + tr[tr[u].r].siz + 1;
    }

    void split(int u, const T &k, int &x, int &y) {
        if (u == 0) {
            x = y = 0;
            return;
        }
        if (tr[u].val <= k) {
            x = u;
            split(tr[u].r, k, tr[u].r, y);
        } else {
            y = u;
            split(tr[u].l, k, x, tr[u].l);
        }
        pushUp(u);
    }

    int merge(int x, int y) {
        if (x == 0 || y == 0) {
            return x == 0 ? y : x;
        }
        if (tr[x].pri < tr[y].pri) {
            tr[x].r = merge(tr[x].r, y);
            pushUp(x);
            return x;
        } else {
            tr[y].l = merge(x, tr[y].l);
            pushUp(y);
            return y;
        }
    }

    void insert(const T &v) {
        int x, y;
        split(root, v, x, y);
        int z = newNode(v);
        root = merge(merge(x, z), y);
    }

    bool erase(const T &v) {
        int x, y, z;
        split(root, v, y, z);
        split(y, v - 1, x, y);
        if (y == 0) {
            merge(x, z);
            return false;
        }
        y = merge(tr[y].l, tr[y].r);
        root = merge(merge(x, y), z);
        return true;
    }

    bool eraseAll(const T &v) {
        int x, y, z;
        split(root, v, y, z);
        split(y, v - 1, x, y);
        if (y == 0) {
            merge(x, z);
            return false;
        }
        root = merge(x, z);
        return true;
    }

    int rank(const T &v) {
        int u = root;
        int cnt = 0;
        while (u != 0) {
            if (tr[u].val < v) {
                cnt += tr[tr[u].l].siz + 1;
                u = tr[u].r;
            } else {
                u = tr[u].l;
            }
        }
        return cnt + 1;
    }

    T kth(int k) {
        int u = root;
        while (u != 0) {
            int s = tr[tr[u].l].siz;
            if (s >= k) {
                u = tr[u].l;
            } else if (s + 1 == k) {
                return tr[u].val;
            } else {
                k -= s + 1;
                u = tr[u].r;
            }
        }
        return -1;
    }

    T prev(const T &v) {
        int x, y;
        split(root, v - 1, x, y);
        int u = x;
        while (u != 0) {
            if (tr[u].r == 0) {
                root = merge(x, y);
                return tr[u].val;
            }
            u = tr[u].r;
        }
        return -1;
    }

    T next(const T &v) {
        int x, y;
        split(root, v, x, y);
        int u = y;
        while (u != 0) {
            if (tr[u].l == 0) {
                root = merge(x, y);
                return tr[u].val;
            }
            u = tr[u].l;
        }
        return -1;
    }
};