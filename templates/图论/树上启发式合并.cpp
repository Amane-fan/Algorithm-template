vector<int> siz(n + 1), dfn(n + 1), hvy(n + 1), rev(n + 1);
int now = 1;
auto dfs1 = [&](auto &&self, int u, int p) -> void {
    siz[u] = 1;
    dfn[u] = now;
    rev[now] = u;
    now++;
    int mx = 0, hc = 0;
    for (auto v : adj[u]) {
        if (v == p) {
            continue;
        }
        self(self, v, u);
        siz[u] += siz[v];
        if (siz[v] > mx) {
            mx = siz[v];
            hc = v;
        }
    }
    hvy[u] = hc;
};
dfs1(dfs1, 1, 0);

auto add = [&](int x) -> void {
    
};

auto delSubTree = [&](int x) -> void {
    
};

vector<i64> ans(n + 1);
auto dfs2 = [&](auto &&self, int u, int p, bool keep) -> void {
    for (auto v : adj[u]) {
        if (v == p || v == hvy[u]) {
            continue;
        }
        self(self, v, u, false);
    }

    if (hvy[u] != 0) {
        self(self, hvy[u], u, true);
    }

    add(u);

    for (auto v : adj[u]) {
        if (v == p || v == hvy[u]) {
            continue;
        }
        for (int i = dfn[v]; i < dfn[v] + siz[v]; i++) {
            add(rev[i]);
        }
    }

    // ans[u] = 

    if (!keep) {
        delSubTree(u);
    }
};
dfs2(dfs2, 1, 0, true);
