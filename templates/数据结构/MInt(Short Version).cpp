// 需 i64 = long long；1 < P <= INT_MAX，不要直接修改 x。
// 求逆 / 除法要求 P 为质数且除数非零；power 的指数必须非负。
// 加减乘 O(1)，求逆 / 除法 O(log P)，快速幂 O(log b)。
template <class T>
T power(T a, i64 b) {
    assert(b >= 0);
    T res = 1;
    for (; b; b >>= 1, a *= a) {
        if (b & 1) {
            res *= a;
        }
    }
    return res;
}

template <int P>
struct MInt {
    static_assert(P > 1);
    i64 x;
    MInt(i64 v = 0) : x(v % P) {
        if (x < 0) {
            x += P;
        }
    }
    MInt inv() const {
        assert(x != 0);
        return power(*this, P - 2);
    }
    MInt operator-() const { return -x; }
    friend MInt operator+(MInt a, MInt b) { return a.x + b.x; }
    friend MInt operator-(MInt a, MInt b) { return a.x - b.x; }
    friend MInt operator*(MInt a, MInt b) { return a.x * b.x; }
    friend MInt operator/(MInt a, MInt b) { return a * b.inv(); }
    MInt &operator+=(MInt a) { return *this = *this + a; }
    MInt &operator-=(MInt a) { return *this = *this - a; }
    MInt &operator*=(MInt a) { return *this = *this * a; }
    MInt &operator/=(MInt a) { return *this = *this / a; }
    friend bool operator==(MInt a, MInt b) { return a.x == b.x; }
    friend bool operator!=(MInt a, MInt b) { return a.x != b.x; }
    friend ostream &operator<<(ostream &os, MInt a) { return os << a.x; }
};

constexpr int mod = 998244353;
using Z = MInt<mod>;
