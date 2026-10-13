// 点编号 [1, n]，有向边 adj[u].push_back({w, v})，最坏 O(nm)。
// s > 0：单源最短路，不可达为 inf；s = 0：所有点作源，检查全图负环。
// 有源点可达的负环返回空 vector；全图判负环用 spfa(adj).empty()。
constexpr i64 inf = numeric_limits<i64>::max() / 3;
vector<i64> spfa(const vector<vector<array<int, 2>>> &adj, int s = 0) {
    int n = int(adj.size()) - 1;
    vector<i64> dis(n + 1, inf);
    vector<int> cnt(n + 1);
    vector<bool> inq(n + 1);
    queue<int> q;
    for (int i = 1; i <= n; i++) {
        if (s == 0 || i == s) {
            dis[i] = 0;
            inq[i] = true;
            q.push(i);
        }
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        inq[u] = false;
        for (auto [w, v] : adj[u]) {
            if (dis[v] <= dis[u] + w) {
                continue;
            }
            dis[v] = dis[u] + w;
            cnt[v] = cnt[u] + 1; // 当前松弛路径的边数，不是入队次数。
            if (cnt[v] >= n) {
                return {};
            }
            if (!inq[v]) {
                inq[v] = true;
                q.push(v);
            }
        }
    }
    return dis;
}
