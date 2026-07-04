i64 Lucas(i64 n, i64 m) {
    if (m < 0 || m > n) return 0;
    if (m == 0) return 1;

    i64 ni = n % mod;
    i64 mi = m % mod;
    if (mi > ni) return 0;

    return (i128)comb.binom(ni, mi) * Lucas(n / mod, m / mod, mod) % mod;
}