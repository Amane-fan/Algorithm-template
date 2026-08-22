const int B = max(1, int(n / sqrt(q)));
vector<int> bel(n + 1);
for (int i = 1; i <= n; i++) {
    bel[i] = (i + B - 1) / B;
}

// 将询问离线
vector<array<int, 3>> Q(q);
for (int i = 0; i < q; i++) {
    int l, r;
    cin >> l >> r;
    Q[i] = {l, r, i};
}

// 按左端点所在块的编号为第一关键字，右端点为第二关键字排序
sort(all(Q), [&](const auto &a, const auto &b) {
    if (bel[a[0]] != bel[b[0]]) {
        return bel[a[0]] < bel[b[0]];
    }
    if (bel[a[0]] & 1) {
        return a[1] < b[1];
    } else {
        return a[1] > b[1];
    }
});

vector<int> cnt(n + 1);
int cur = 0;

auto add = [&](int k) -> void {
    if (cnt[a[k]]++ == 0) {
        cur++;
    }
};

auto del = [&](int k) -> void {
    if (--cnt[a[k]] == 0) {
        cur--;
    }
};

vector<int> ans(q);

// x, y 代表询问区间；l, r 代表当前所在区间
// 先扩展，再删除
for (int i = 0, l = 1, r = 0; i < q; i++) {
    auto [x, y, id] = Q[i];
    while (l > x) add(--l); // 左扩展
    while (r < y) add(++r); // 右扩展
    while (l < x) del(l++); // 左删除
    while (r > y) del(r--); // 右删除
    ans[id] = cur;
}
