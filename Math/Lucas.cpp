i64 Lucas(i64 n, i64 m, i64 P) {
    if (m < 0 || m > n) return 0;
    if (m == 0) return 1;

    i64 ni = n % P;
    i64 mi = m % P;
    if (mi > ni) return 0;

    return (i128)comb.binom(ni, mi) * Lucas(n / P, m / P, P) % P;
}