constexpr int N = 1e6;

int trie[N][26];
int tot = 0;

void clear() {
    for (int i = 0; i <= tot; i++) {
        fill(trie[i], trie[i] + 26, 0);
    }
    tot = 0;
}

void insert(const string &s) {
    int n = s.size();
    int p = 0;
    for (int i = 0; i < n; i++) {
        int &nxt = trie[p][s[i] - 'a'];
        if (nxt == 0) {
            nxt = ++tot;
        }
        p = nxt;
    }
}