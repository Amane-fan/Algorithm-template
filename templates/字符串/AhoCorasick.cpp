struct AhoCorasick {
    struct Node {
        int link;
        int len;
        array<int, 26> nxt;
        Node(): link(0), len(0), nxt{} {}
    };
    vector<Node> trie;
    int newNode() {
        trie.push_back(Node{});
        return (int)trie.size() - 1;
    }
    AhoCorasick() {
        newNode();
    }
    void add(const string &s) {
        int u = 0;
        for (auto c : s) {
            int nxt = trie[u].nxt[c - 'a'];
            if (nxt == 0) {
                nxt = newNode();
                trie[u].nxt[c - 'a'] = nxt;
                trie[nxt].len = trie[u].len + 1;
            }
            u = nxt;
        }
    }
    void work() {
        queue<int> q;
        for (int i = 0; i < 26; i++) {
            int nxt = trie[0].nxt[i];
            if (nxt != 0) {
                q.push(nxt);
            }
        }
        while (!q.empty()) {
            auto u = q.front();
            q.pop();
            for (int i = 0; i < 26; i++) {
                if (trie[u].nxt[i] == 0) {
                    trie[u].nxt[i] = trie[trie[u].link].nxt[i];
                } else {
                    trie[trie[u].nxt[i]].link = trie[trie[u].link].nxt[i];
                    q.push(trie[u].nxt[i]);
                }
            }
        }
    }
    int nxt(int u, int x) {
        return trie[u].nxt[x];
    }
    int link(int u) {
        return trie[u].link;
    }
    int len(int u) {
        return trie[u].len;
    }
    int size() {
        return (int)trie.size() - 1;
    }
};