struct EBCC {
    int n;
    vector<vector<array<int, 2>>> adj;
    vector<int> stk;
    vector<int> dfn, low, bel;
    vector<bool> isBridge;

    int cur, cnt, ecnt;

    EBCC() {}

    EBCC(int N) {
        init(N);
    }

    void init(int N) {
        n = N;
        adj.assign(n + 1, {});
        dfn.assign(n + 1, -1);
        low.resize(n + 1);
        bel.assign(n + 1, -1);
        stk.clear();

        isBridge.assign(1, false);

        cur = cnt = 0;
        ecnt = 1;
    }

    void addEdge(int u, int v) {
        adj[u].push_back({v, ecnt});
        adj[v].push_back({u, ecnt});

        isBridge.push_back(false);

        ecnt++;
    }

    void dfs(int x, int pe) {
        dfn[x] = low[x] = cur++;
        stk.push_back(x);

        for (auto [y, id] : adj[x]) {
            if (id == pe) {
                continue;
            }

            if (dfn[y] == -1) {
                dfs(y, id);
                low[x] = min(low[x], low[y]);
                if (low[y] > dfn[x]) {
                    isBridge[id] = true;
                }
            } else if (dfn[y] < dfn[x]) {
                low[x] = min(low[x], dfn[y]);
            }
        }

        if (dfn[x] == low[x]) {
            cnt++;
            int y;
            do {
                y = stk.back();
                stk.pop_back();
                bel[y] = cnt;
            } while (y != x);
        }
    }

    pair<int, vector<int>> work() {
        for (int i = 1; i <= n; i++) {
            if (dfn[i] == -1) {
                dfs(i, -1);
            }
        }
        return {cnt, bel};
    }

    bool bridge(int id) {
        return isBridge[id];
    }
};