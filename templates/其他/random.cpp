mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());

// 生成 [l, r] 范围内的数
int rng(int l, int r) {
    return rnd() % (r - l + 1) + l;
}

// 生成在 [l, r] 范围内的一个区间
pair<int, int> interval(int l = 1, int r = 5) {
    int x = rng(l, r);
    int y = rng(l, r);
    return minmax(x, y);
}

// 生成节点数在 [l, r] 范围内的一棵树
void tree(int l = 1, int r = 5) {
    int n = rng(l, r);
    cout << n << '\n';

    for (int u = 2; u <= n; u++) {
        int v = rng(1, u - 1);
        cout << u << " " << v << '\n';
    }
}

// 生成节点数在 [l, r] 范围内的一个无向连通图
void graph(int l = 1, int r = 5) {
    int n = rng(l, r);
    int m = rng(n - 1, n * (n - 1) / 2);
    cout << n << " " << m << '\n';
    set<pair<int, int>> S;
    vector<pair<int, int>> edges;
    for (int u = 2; u <= n; u++) {
        int v = rng(1, u - 1);
        S.insert({u, v});
        edges.push_back({u, v});
    }
    
    for (int i = n; i <= m; i++) {
        int u, v;
        do {
            u = rng(1, n);
            v = rng(1, n);
        } while (u == v || S.contains({u, v}));
        S.insert({u, v});
        edges.push_back({u, v});
    }
    for (int i = 0; i < m; i++) {
        auto [u, v] = edges[i];
        cout << u << " " << v << '\n';
    }
}