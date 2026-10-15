// a 的长度为 2^n，a[s] 表示集合 s 的权值。
// 原地求 a[s] = sum_{t subseteq s} a[t]，包含空集和 s 自身。
// 时间复杂度 O(n * 2^n)，额外空间 O(1)。T 需支持 +=，注意溢出。
template <class T>
void subsetSum(vector<T> &a) {
    int n = a.size();
    assert(n > 0 && (n & (n - 1)) == 0);
    for (int bit = 1; bit < n; bit <<= 1) {
        for (int s = 0; s < n; s++) {
            if (s & bit) {
                a[s] += a[s ^ bit];
            }
        }
    }
}

template <class T>
void supersetSum(vector<T> &a) {
    int n = a.size();
    assert(n > 0 && (n & (n - 1)) == 0);
    for (int bit = 1; bit < n; bit <<= 1) {
        for (int s = 0; s < n; s++) {
            if (!(s & bit)) {
                a[s] += a[s | bit];
            }
        }
    }
}
