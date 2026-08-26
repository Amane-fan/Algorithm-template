vector<int> siz(n + 1);
vector<int> g;
{
    auto dfs = [&](auto &&dfs, int u, int p = 0) -> void {
        siz[u] = 1;
        int mx = 0;
        for (auto v : adj[u]) {
            if (v == p) {
                continue;
            }
            dfs(dfs, v, u);
            siz[u] += siz[v];
            mx = max(mx, siz[v]);
        }
        mx = max(mx, n - siz[u]);

        if (mx <= n / 2) {
            g.push_back(u);
        }
    };
    dfs(dfs, 1);
}