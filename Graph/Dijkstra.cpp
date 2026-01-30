constexpr i64 inf = numeric_limits<i64>::max() / 3;
vector<i64> dijkstra(const vector<vector<array<int, 2>>> &adj, int s) {
    int n = int(adj.size()) - 1;
    vector<i64> dis(n + 1, inf);
    dis[s] = 0;
    vector<int> vis(n + 1);
    priority_queue<pair<i64, int>, vector<pair<i64, int>>, greater<>> pq;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [_, u] = pq.top();
        pq.pop();
        if (vis[u]) {
            continue;
        }
        vis[u] = true;
        for (auto &[w, v] : adj[u]) {
            if (vis[v] || dis[u] + w >= dis[v]) {
                continue;
            }
            dis[v] = dis[u] + w;
            pq.push({dis[v], v});
        }
    }
    return dis;
}