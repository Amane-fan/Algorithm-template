constexpr int N = 2;
using Mat = array<array<Z, N>, N>;
 
Mat operator*(const Mat &a, const Mat &b) {
    Mat res {};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
}
 
Mat power(Mat a, i64 b) {
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