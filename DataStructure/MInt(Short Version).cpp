template <class T>
T power(T a, i64 b) {
    T res{1};
    for (; b; b /= 2, a = a * a) {
        if (b & 1) {
            res = res * a;
        }
    }
    return res;
}

template <int P>
struct MInt {
    i64 x;
    MInt(): x{} {}
    MInt(i64 x_): x(x_) {
        x %= P;
        if (x < 0) {
            x += P;
        }
    }
    MInt inv() const {
        assert(x != 0);
        return power(*this, P - 2);
    }
    MInt& operator+=(const MInt &o) {
        x = (x + o.x) % P;
        return *this;
    }
    MInt& operator-=(const MInt &o) {
        x -= o.x;
        if (x < 0) {
            x += P;
        }
        return *this;
    }
    MInt& operator*=(const MInt &o) {
        x = (x * o.x) % P;
        return *this;
    }
    MInt& operator/=(const MInt &o) {
        x = x * o.inv() % P;
        return *this;
    }
    friend MInt operator+(MInt lhs, const MInt &rhs) { return lhs += rhs; }
    friend MInt operator-(MInt lhs, const MInt &rhs) { return lhs -= rhs; }
    friend MInt operator*(MInt lhs, const MInt &rhs) { return lhs *= rhs; }
    friend MInt operator/(MInt lhs, const MInt &rhs) { return lhs /= rhs; }
};

constexpr int mod = 998244353;
using Z = MInt<mod>;