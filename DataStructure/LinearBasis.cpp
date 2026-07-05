template <class T>
struct LinearBasis {
    static constexpr int N = __lg(numeric_limits<T>::max());
    array<T, N + 1> b;
    bool zero;

    LinearBasis() {
        zero = false;
        b.fill(0);
    }

    void insert(T x) {
        for (int i = N; i >= 0; i--) {
            if (x >> i & 1) {
                if (b[i] == 0) {
                    b[i] = x;
                    return;
                }
                x ^= b[i];
            }
        }
        zero = true;
    }

    T queryMax() {
        T ans = 0;
        for (int i = N; i >= 0; i--) {
            ans = max(ans, ans ^ b[i]);
        }
        return ans;
    }

    T queryMin() {
        if (zero) {
            return T(0);
        }
        T res;
        for (int i = 0; i <= N; i++) {
            if (b[i] != 0) {
                res = b[i];
                break;
            }
        }
        return res;
    }

    bool check(T x) {
        for (int i = N; i >= 0; i--) {
            if (x >> i & 1) {
                if (b[i] == 0) {
                    return false;
                }
                x ^= b[i];
            }
        }
        return true;
    }
};