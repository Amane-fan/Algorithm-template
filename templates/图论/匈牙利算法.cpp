int n, m, e;
cin >> n >> m >> e;

vector<vector<int>> adj(n + 1);
for (int i = 0; i < e; i++) {
    int u, v;
    cin >> u >> v;
    adj[u].push_back(v);
}

vector<int> vis(m + 1);
vector<int> match(m + 1);
auto dfs = [&](auto &&self, int u) -> bool {
    for (auto v : adj[u]) {
        if (vis[v]) {
            continue;
        }
        vis[v] = true;

        if (match[v] == 0 || self(self, match[v])) {
            match[v] = u;
            return true;
        }
    }

    return false;
};

int ans = 0;
for (int i = 1; i <= n; i++) {
    fill(vis.begin() + 1, vis.end(), 0);
    ans += dfs(dfs, i);
}

cout << ans << '\n';
