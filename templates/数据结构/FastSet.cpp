// 值域 [0, n]，0 <= n < INT_MAX，初始为空；重复插入、删除不存在的元素均无影响。
// contains O(1)，插入、删除、前驱后继 O(log_64(n))，空间 O(n) 位。
struct FastSet {
    int n;
    vector<vector<u64>> a;

    FastSet(int N = 0) {
        init(N);
    }
    void init(int N) {
        assert(0 <= N && N < INT_MAX);
        n = N;
        N++;
        a.clear();
        do {
            N = N / 64 + (N % 64 != 0);
            a.push_back(vector<u64>(N));
        } while (N > 1);
    }
    bool contains(int x) const {
        return 0 <= x && x <= n && (a[0][x / 64] >> (x % 64) & 1);
    }
    void insert(int x) {
        assert(0 <= x && x <= n);
        for (auto &v : a) {
            v[x / 64] |= u64(1) << (x % 64);
            x /= 64;
        }
    }
    void erase(int x) {
        assert(0 <= x && x <= n);
        for (auto &v : a) {
            v[x / 64] &= ~(u64(1) << (x % 64));
            if (v[x / 64]) {
                break;
            }
            x /= 64;
        }
    }
    // 最小的 >= x 的元素，不存在返回 n + 1。
    int next(int x) const {
        x = max(x, 0);
        if (x > n) {
            return n + 1;
        }
        for (int h = 0; h < int(a.size()); h++) {
            if (x / 64 >= int(a[h].size())) {
                break;
            }
            u64 v = a[h][x / 64] >> (x % 64);
            if (!v) {
                x = x / 64 + 1;
                continue;
            }
            x += __builtin_ctzll(v);
            for (int i = h - 1; i >= 0; i--) {
                x *= 64;
                x += __builtin_ctzll(a[i][x / 64]);
            }
            return x;
        }
        return n + 1;
    }
    // 最大的 <= x 的元素，不存在返回 -1。
    int prev(int x) const {
        x = min(x, n);
        for (int h = 0; h < int(a.size()); h++) {
            if (x < 0) {
                break;
            }
            u64 v = a[h][x / 64] << (63 - x % 64);
            if (!v) {
                x = x / 64 - 1;
                continue;
            }
            x -= __builtin_clzll(v);
            for (int i = h - 1; i >= 0; i--) {
                x = x * 64 + 63;
                x -= __builtin_clzll(a[i][x / 64]);
            }
            return x;
        }
        return -1;
    }
};
