template <class T>
std::array<T, 3> exgcd(const T& a, const T& b) {
    if (b == T(0)) {
        return {a, T(1), T(0)};
    }
    auto [g, x, y] = exgcd(b, a % b);
    return {g, y, x - a / b * y};
}

template <i64 mod>
class ModInt {
    static_assert(mod >= 1LL, "The modulus must be a positive integer.");
private:
    i64 value;
public:
    ModInt() : value(0LL) {}
 
    ModInt(const ModInt& v) : value(v.value) {}
 
    template <typename T>
    ModInt(const T& v): value(((i64)(v % mod) + mod) % mod) {}
 
    explicit operator i64() const {
        return value;
    }
 
    i64 getValue() const {
        return value;
    }
 
    static i64 getMod() {
        return mod;
    }
 
    ModInt inv() const {
        auto [g, x, y] = exgcd<i64>(value, mod);
        if (g != 1LL) {
            throw std::runtime_error("The modular inverse does not exist.");
        }
        ModInt res;
        res.value = x;
        if (res.value < 0LL) {
            res.value += mod;
        }
        return res;
    }
 
    template <typename V>
    ModInt pow(V b) const {
        i64 a = ((b >= V(0)) ? value : inv().value);
        ModInt res;
        res.value = ((mod >= 2LL) ? 1LL : 0LL);
        while (b != V(0)) {
            if (b % V(2) != V(0)) {
                res.value = res.value * a % mod;
            }
            a = a * a % mod;
            b = b / V(2);
        }
        return res;
    }
 
    ModInt& operator+=(const ModInt& b) {
        value += b.value;
        if (value >= mod) {
            value -= mod;
        }
        return *this;
    }
 
    ModInt& operator-=(const ModInt& b) {
        value -= b.value;
        if (value < 0LL) {
            value += mod;
        }
        return *this;
    }
 
    ModInt& operator*=(const ModInt& b) {
        value *= b.value;
        value %= mod;
        return *this;
    }
 
    ModInt& operator/=(const ModInt& b) {
        value *= b.inv().value;
        value %= mod;
        return *this;
    }
 
    ModInt& operator=(const ModInt& b) {
        value = b.value;
        return *this;
    }
 
    ModInt operator+() const {
        return *this;
    }
 
    ModInt operator-() const {
        ModInt res;
        res.value = -value;
        if (res.value < 0LL) {
            res.value += mod;
        }
        return res;
    }
 
    ModInt& operator++() {
        ++value;
        if (value >= mod) {
            value -= mod;
        }
        return *this;
    }
 
    ModInt operator++(int) {
        ModInt temp(*this);
        ++value;
        if (value >= mod) {
            value -= mod;
        }
        return temp;
    }
 
    ModInt& operator--() {
        --value;
        if (value < 0LL) {
            value += mod;
        }
        return *this;
    }
 
    ModInt operator--(int) {
        ModInt temp(*this);
        --value;
        if (value < 0LL) {
            value += mod;
        }
        return temp;
    }
 
    friend ModInt operator+(const ModInt& a, const ModInt& b) {
        ModInt res;
        res.value = a.value + b.value;
        if (res.value >= mod) {
            res.value -= mod;
        }
        return res;
    }
 
    friend ModInt operator-(const ModInt& a, const ModInt& b) {
        ModInt res;
        res.value = a.value - b.value;
        if (res.value < 0LL) {
            res.value += mod;
        }
        return res; 
    }
 
    friend ModInt operator*(const ModInt& a, const ModInt& b) {
        ModInt res;
        res.value = a.value * b.value % mod;
        return res;
    }
 
    friend ModInt operator/(const ModInt& a, const ModInt& b) {
        ModInt res;
        res.value = a.value * b.inv().value % mod;
        return res;
    }
 
    friend std::ostream& operator<<(std::ostream& os, const ModInt& x) {
        os << x.value;
        return os;
    }
 
    friend std::istream& operator>>(std::istream& is, ModInt& x) {
        if (is >> x.value) {
            x.value = (x.value % mod + mod) % mod;
        }
        return is;
    }
};

constexpr int mod = 998244353;
using Z = ModInt<mod>;