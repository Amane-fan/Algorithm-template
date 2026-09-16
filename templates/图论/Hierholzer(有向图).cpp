array<vector<int>, 2> Hierholzer(const vector<vector<array<int, 2>>> &adj) {
    int n = int(adj.size()) - 1;
    if (n == 0) {
        return {vector<int>(), vector<int>()};
    }
    vector<int> in(n + 1);
    for (int u = 1; u <= n; u++) {
        for (auto [v, e] : adj[u]) {
            in[v]++;
        }
    }
    int s = 1;
    for (int u = 2; u <= n; u++) {
        if (adj[s].size() == 0 && adj[u].size() != 0) {
            s = u;
        } else if (adj[s].size() != in[s] + 1 && adj[u].size() == in[u] + 1) {
            s = u;
        }
    }
    vector<int> path, edge;
    vector<int> idx(n + 1);
    auto dfs = [&](auto &&self, int u, int e) -> void {
        while (idx[u] < adj[u].size()) {
            auto [v, ne] = adj[u][idx[u]];
            idx[u]++;
            self(self, v, ne);
        }
        path.push_back(u);
        edge.push_back(e);
    };
    dfs(dfs, s, -1);

    edge.pop_back();
    reverse(edge.begin(), edge.end());
    reverse(path.begin(), path.end());
    return {path, edge};
}

bool check(const vector<vector<array<int, 2>>> &adj, const vector<int> &path) {
    int n = int(adj.size()) - 1;
    int m = 0;
    vector<map<int, int>> num(n + 1);
    for (int u = 1; u <= n; u++) {
        for (auto [v, e] : adj[u]) {
            num[u][v]++;
            m++;
        }
    }
    int ps = path.size();
    for (int i = 0; i + 1 < ps; i++) {
        int u = path[i];
        int v = path[i + 1];
        if (u < 1 || u > n || v < 1 || v > n || num[u][v] == 0) {
            return false;
        }
        num[u][v]--;
        m--;
    }
    return m == 0;
}