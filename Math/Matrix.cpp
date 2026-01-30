constexpr int N = 2;
constexpr int mod = 1e9 + 7;
using Mat = array<array<i64, N>, N>;
 
Mat operator*(const Mat &a, const Mat &b) {
    Mat res {};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                res[i][j] += a[i][k] * b[k][j] % mod;
                res[i][j] %= mod;
            }
        }
    }
    return res;
}
 
template <class T>
T power(T a, i64 b) {
    Mat res {};
    for (int i = 0; i < N; i++) {
        res[i][i] = 1;
	}
 
    while (b) {
        if (b & 1) {
            res = res * a;
        }
        b >>= 1;
        a = a * a;
    }
    return res;
}