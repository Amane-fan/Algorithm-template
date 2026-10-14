using i128 = __int128;
istream &operator>>(istream &is, i128 &n) {
    string s;
    if (!(is >> s)) return is;
    n = 0;
    for (int i = s[0] == '-' || s[0] == '+'; i < (int)s.size(); i++) {
        n = n * 10 - (s[i] - '0');
    }
    if (s[0] != '-') n = -n;
    return is;
}
ostream &operator<<(ostream &os, i128 n) {
    if (n < 0) os << '-';
    string s;
    do {
        s += char('0' + abs(int(n % 10)));
        n /= 10;
    } while (n);
    reverse(s.begin(), s.end());
    return os << s;
}
