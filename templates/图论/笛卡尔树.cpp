// 小根堆
vector<int> lc(n + 1), rc(n + 1);
vector<int> stk;
for (int i = 1; i <= n; i++) {
    while (!stk.empty() && a[i] < a[stk.back()]) {
        lc[i] = stk.back();
        stk.pop_back();
    }
    if (!stk.empty()) {
        rc[stk.back()] = i;
    }
    stk.push_back(i);
}