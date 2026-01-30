// 有向图
// 保证至少有一条边即可，允许自环和重边
vector<int> Hierholzer(vector<vector<int>> adj) {
    int n = int(adj.size()) - 1;
    vector<int> in(n + 1), out(n + 1);
    for (int u = 1; u <= n; u++) {
        for (auto v : adj[u]) {
            in[v]++;
            out[u]++;
        }
    }

    int s = 0;
    int x = 0, y = 0;
    for (int i = 1; i <= n; i++) {
        if (abs(in[i] - out[i]) > 1) {
            return {};
        }

        if (s == 0 && in[i] == out[i] && in[i] != 0) {
            s = i;
        }

        if (out[i] - in[i] == 1) {
            x++;
            s = i;
        } else if (in[i] - out[i] == 1) {
            y++;
        }
    }

    if ((x != 0 || y != 0) && (x != 1 || y != 1)) {
        return {};
    }

    vector<int> stk;
    vector<int> path;
    stk.push_back(s);

    while (!stk.empty()) {
        int u = stk.back();
        if (adj[u].empty()) {
            path.push_back(u);
            stk.pop_back();
        } else {
            int v = adj[u].back();
            stk.push_back(v);
            adj[u].pop_back();
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