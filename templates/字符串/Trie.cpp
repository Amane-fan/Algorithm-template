struct Node {
    int cnt;
    array<int, 26> nxt;
    Node(): cnt(0), nxt{} {}
};

vector<Node> trie{Node{}};

auto newNode = [&]() -> int {
    trie.push_back(Node{});
    return (int)trie.size() - 1;
};

auto insert = [&](const string &s) -> void {
    int u = 0;
    for (auto c : s) {
        int nxt = trie[u].nxt[c - 'a'];
        if (nxt == 0) {
            nxt = newNode();
            trie[u].nxt[c - 'a'] = nxt;
        }
        u = nxt;
    }
    trie[u].cnt++;
};