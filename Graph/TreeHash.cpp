mt19937_64 rnd(chrono::steady_clock::now().time_since_epoch().count());
const ull mask = rnd();
struct TreeHash {
    int n;
    vector<ull> h, rt;
    vector<vector<int>> adj;
    TreeHash(int N): n(N), adj(N + 1), h(N + 1), rt(N + 1) {}
    static ull shift(ull x) {
        x ^= mask;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x ^= mask;
        return x;
    }
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    void dfs1(int u, int p = 0) {
        h[u] = 1;
        for (auto v : adj[u]) {
            if (v == p) {
                continue;
            }
            dfs1(v, u);
            h[u] += shift(h[v]);
        }
    }
    void dfs2(int u, int p = 0) {
        for (auto v : adj[u]) {
            if (v == p) {
                continue;
            }
            rt[v] = h[v] + shift(rt[u] - shift(h[v]));
            dfs2(v, u);
        }
    }
    void work(int r = 1) {
        dfs1(r);
        rt[r] = h[r];
        dfs2(r);
    }
    auto tie_arrays() {
        return tie(h, rt);
    }
};