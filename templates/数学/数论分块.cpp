i64 floorSum(i64 n) {
    i64 ans = 0;
    for (i64 l = 1, r; l <= n; l = r + 1) {
        r = n / (n / l);
        ans += (r - l + 1) * (n / l);
        if (r == n) {
            break; // 避免 n 为 i64 最大值时 r + 1 溢出。
        }
    }
    return ans;
}
