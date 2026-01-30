constexpr ll inf = 1e18;
vector<ll> spfa(const vector<vector<array<int, 2>>> &adj, int s) {
    int n = adj.size() - 1;
    vector<ll> dis(n + 1, inf);
    vector<int> cnt(n + 1);
    vector<bool> vis(n + 1);
    dis[s] = 0;
    vis[s] = true;
    queue<int> q;
    q.push(s);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        vis[u] = false;
        for (auto [w, v] : adj[u]) {
            if (dis[v] > dis[u] + w) {
                dis[v] = dis[u] + w;
                cnt[v] = cnt[u] + 1;
                // 判断负环，要写>=点的个数，如果0位置也有点要写>=n+1
                if (cnt[v] >= n) {
                    return {};
                }
                if (!vis[v]) {
                    q.push(v);
                    vis[v] = true;
                }
            }
        }
    }
    return dis;
}