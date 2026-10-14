array<vector<int>, 2> Hierholzer(const vector<vector<array<int, 2>>> &g) {
    int n = g.size() - 1;
    if (n == 0) {
        return {vector<int>(), vector<int>()};
    }

    int m = 0;
    for (int u = 1; u <= n; ++u) {
        m += g[u].size();
    }
    m /= 2;

    int begin = 1;
    for (int u = 2; u <= n; ++u) {
        if (g[begin].size() == 0 && g[u].size() != 0) {
            begin = u;
        } else if (g[begin].size() % 2 != 1 && g[u].size() % 2 == 1) {
            begin = u;
        }
    }

    vector<int> path;
    vector<int> edge;
    vector<int> idx(n + 1, 0);
    vector<int> vis(m + 1, 0);

    auto dfs = [&](auto &&self, int u, int e) -> void {
        while (idx[u] < g[u].size()) {
            auto [v, e_] = g[u][idx[u]];
            ++idx[u];

            if (!vis[e_]) {
                vis[e_] = 1;
                self(self, v, e_);
            }
        }

        path.push_back(u);
        edge.push_back(e);
    };

    dfs(dfs, begin, 0);

    edge.pop_back();

    reverse(path.begin(), path.end());
    reverse(edge.begin(), edge.end());

    return {path, edge};
}

bool check(const vector<vector<array<int, 2>>> &g, const vector<int> &path) {
    int n = g.size() - 1;
    int m = 0;

    vector<map<int, int>> num(n + 1);

    for (int u = 1; u <= n; ++u) {
        for (auto [v, w] : g[u]) {
            num[u][v] += 1;
            m += 1;
        }
    }
    m /= 2;

    int ps = path.size();

    for (int i = 0; i + 1 < ps; ++i) {
        int u = path[i];
        int v = path[i + 1];

        if (u < 1 || u > n || v < 1 || v > n || num[u][v] == 0) {
            return false;
        }

        num[u][v] -= 1;
        num[v][u] -= 1;
        m -= 1;
    }

    return m == 0;
}