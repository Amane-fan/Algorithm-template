// 无向图
vector<int> Hierholzer(vector<vector<int>> adj) {
    int n = int(adj.size()) - 1;
    int odd = 0;
    vector<int> deg(n + 1);
    for (int u = 1; u <= n; u++) {
        for (auto v : adj[u]) {
            deg[u]++;
            if (u == v) {
                deg[u]++;
            }
        }
    }
    int s = 0;
    for (int i = n; i >= 1; i--) {
        if (deg[i] & 1) {
            odd++;
            s = i;
        }

        if (s == 0 && deg[i] != 0) {
            s = i;
        }
    }

    if (odd != 0 && odd != 2) {
        return {};
    }

    vector<int> path;
    vector<int> stk;
    stk.push_back(s);

    while (!stk.empty()) {
        int u = stk.back();
        if (adj[u].empty()) {
            path.push_back(u);
            stk.pop_back();
        } else {
            int v = adj[u].back();
            adj[u].pop_back();
            auto it = find(adj[v].begin(), adj[v].end(), u);
            if (it != adj[v].end()) {
                *it = adj[v].back();
                adj[v].pop_back();
            }
            stk.push_back(v);
        }
    }

    for (int u = 1; u <= n; u++) {
        if (!adj[u].empty()) {
            return {};
        }
    }

    reverse(path.begin(), path.end());
    return path;
}