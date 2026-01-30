template <class T>
array<T, 3> exgcd(const T &a, const T &b) {
    if (b == T(0)) {
        return {a, T(1), T(0)};
    }
    auto [g, x, y] = exgcd(b, a % b);
    return {g, y, x - a / b * y};
}