vector<int> kmp(const string &s) {
    int n = (int)s.size() - 1;
    vector<int> pi(n + 1);
    for (int i = 2; i <= n; i++) {
        int j = pi[i - 1];
        while (j > 0 && s[j + 1] != s[i]) {
            j = pi[j];
        }
        if (s[j + 1] == s[i]) {
            pi[i] = j + 1;
        }
    }
    return pi;
}

vector<array<int, 26>> automaton(const string &s) {
    int n = (int)s.size() - 1;
    auto pi = kmp(s);
    vector<array<int, 26>> go(n + 1);
    for (int i = 0; i <= n; i++) {
        for (int c = 0; c < 26; c++) {
            if (i < n && s[i + 1] == c + 'a') {
                go[i][c] = i + 1;
            } else if (i == 0) {
                go[i][c] = 0;
            } else {
                go[i][c] = go[pi[i]][c];
            }
        }
    }
    return go;
}