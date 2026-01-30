struct FastGCD {
    int V;           // 值域上限
    int RADIO;       // 阈值, 通常为 sqrt(V)

    vector<int> np;          // np[i] > 0 表示 i 是合数
    vector<int> prime;       // 存储找到的素数
    vector<array<int, 3>> k; // k[i] 存储 i 的一种特殊三因子分解
    vector<vector<int>> sg; // 预计算的小范围 GCD 表
    int cnt;                      // 找到的素数数量

    /**
     * @brief 构造函数，执行所有预处理操作。
     * @param n 预处理的最大值 V。
     *
     * 预处理时间复杂度近似为 O(V * log(logV)) + O(RADIO^2)。
     * 空间复杂度为 O(V)。
     */
    FastGCD(int n) : V(n), RADIO(static_cast<int>(floor(sqrt(n)))), cnt(0) {
        np.resize(n + 1);
        prime.resize(n + 1);
        k.resize(n + 1);
        sg.resize(RADIO + 1, vector<int>(RADIO + 1));

        k[1] = {1, 1, 1};
        np[1] = 1;

        for (int i = 2; i <= V; i++) {
            if (!np[i]) {
                prime[++cnt] = i;
                k[i] = {1, 1, i};
            }
            for (int j = 1; j <= cnt && 1LL * prime[j] * i <= V; j++) {
                np[i * prime[j]] = 1;
                auto &tmp = k[i * prime[j]];
                
                tmp[0] = k[i][0] * prime[j];
                tmp[1] = k[i][1];
                tmp[2] = k[i][2];

                if (tmp[1] < tmp[0]) swap(tmp[1], tmp[0]);
                if (tmp[2] < tmp[1]) swap(tmp[2], tmp[1]);
                
                if (i % prime[j] == 0) {
                    break;
                }
            }
        }

        for (int i = 0; i <= RADIO; i++) {
            sg[i][0] = sg[0][i] = i;
        }

        for (int i = 1; i <= RADIO; i++) {
            for (int j = 1; j <= i; j++) {
                sg[i][j] = sg[j][i] = sg[j][i % j];
            }
        }
    }

    int query(int a, int b) const {
        if (a == 0) return b;
        if (b == 0) return a;

        int g = 1;
        for (int i = 0; i < 3; ++i) {
            int ka = k[a][i];
            if (ka == 1) continue;

            int cf;
            if (ka > RADIO) {
                cf = (b % ka == 0) ? ka : 1;
            } else {
                cf = sg[ka][b % ka];
            }
            g *= cf;
            b /= cf;
        }
        return g;
    }
};