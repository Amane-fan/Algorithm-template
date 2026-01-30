int N = __lg(n - 1);
vector<int> dep(n + 1);
vector f(n + 1, vector<int>(N + 1));

auto dfs = [&](auto &&dfs, int u, int p) -> void {
    dep[u] = dep[p] + 1;
    f[u][0] = p;
    for (int i = 1; i <= N; i++) {
        f[u][i] = f[f[u][i - 1]][i - 1];
    }
    for (auto v : adj[u]) {
        if (v == p) continue;
        dfs(dfs, v, u);
    }
};
dfs(dfs, s, 0);

auto lca = [&](int x, int y) -> int {
    if (dep[x] < dep[y]) swap(x, y);
    int d = dep[x] - dep[y];
    for (int i = __lg(d); i >= 0; i--) {
        if (d >> i & 1) {
            x = f[x][i];
        }
    }

    if (x == y) return x;
    for (int i = N; i >= 0; i--) {
        if (f[x][i] != f[y][i]) {
            x = f[x][i];
            y = f[y][i];
        }
    }
    return f[x][0];
};