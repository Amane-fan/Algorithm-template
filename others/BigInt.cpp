struct BigInt {
    vector<int> a;  // 低位在前

    BigInt(long long x = 0) {
        *this = x;
    }

    BigInt(const string &s) {
        *this = s;
    }

    // long long -> BigInt
    BigInt& operator=(long long x) {
        a.clear();

        if (x == 0) {
            a.push_back(0);
            return *this;
        }

        while (x) {
            a.push_back(x % 10);
            x /= 10;
        }

        return *this;
    }

    // string -> BigInt
    BigInt& operator=(const string &s) {
        a.clear();

        for (int i = (int)s.size() - 1; i >= 0; i--)
            a.push_back(s[i] - '0');

        trim();
        return *this;
    }

    // 去除前导 0
    void trim() {
        while (a.size() > 1 && a.back() == 0)
            a.pop_back();
    }

    // 比较
    friend bool operator<(const BigInt &A, const BigInt &B) {
        if (A.a.size() != B.a.size())
            return A.a.size() < B.a.size();

        for (int i = (int)A.a.size() - 1; i >= 0; i--)
            if (A.a[i] != B.a[i])
                return A.a[i] < B.a[i];

        return false;
    }

    friend bool operator>(const BigInt &A, const BigInt &B) {
        return B < A;
    }

    friend bool operator<=(const BigInt &A, const BigInt &B) {
        return !(B < A);
    }

    friend bool operator>=(const BigInt &A, const BigInt &B) {
        return !(A < B);
    }

    friend bool operator==(const BigInt &A, const BigInt &B) {
        return A.a == B.a;
    }

    friend bool operator!=(const BigInt &A, const BigInt &B) {
        return !(A == B);
    }

    // 加法
    friend BigInt operator+(const BigInt &A, const BigInt &B) {
        BigInt C;
        C.a.clear();

        int t = 0;

        for (int i = 0;
             i < A.a.size() || i < B.a.size() || t;
             i++) {

            if (i < A.a.size()) t += A.a[i];
            if (i < B.a.size()) t += B.a[i];

            C.a.push_back(t % 10);
            t /= 10;
        }

        return C;
    }

    // 减法，要求 A >= B
    friend BigInt operator-(const BigInt &A, const BigInt &B) {
        BigInt C;
        C.a.clear();

        int t = 0;

        for (int i = 0; i < A.a.size(); i++) {
            t = A.a[i] - t;

            if (i < B.a.size())
                t -= B.a[i];

            C.a.push_back((t + 10) % 10);
            t = (t < 0);
        }

        C.trim();
        return C;
    }

    // 大整数 * 大整数
    friend BigInt operator*(const BigInt &A, const BigInt &B) {
        BigInt C;
        C.a.assign(A.a.size() + B.a.size(), 0);

        for (int i = 0; i < A.a.size(); i++)
            for (int j = 0; j < B.a.size(); j++)
                C.a[i + j] += A.a[i] * B.a[j];

        for (int i = 0; i + 1 < C.a.size(); i++) {
            C.a[i + 1] += C.a[i] / 10;
            C.a[i] %= 10;
        }

        C.trim();
        return C;
    }

    // 大整数 * int
    friend BigInt operator*(const BigInt &A, int b) {
        BigInt C;
        C.a.clear();

        long long t = 0;

        for (int i = 0; i < A.a.size() || t; i++) {
            if (i < A.a.size())
                t += 1LL * A.a[i] * b;

            C.a.push_back(t % 10);
            t /= 10;
        }

        C.trim();
        return C;
    }

    friend BigInt operator*(int b, const BigInt &A) {
        return A * b;
    }

    // 大整数 / int
    friend BigInt operator/(const BigInt &A, int b) {
        BigInt C;
        C.a.clear();

        long long r = 0;

        for (int i = (int)A.a.size() - 1; i >= 0; i--) {
            r = r * 10 + A.a[i];
            C.a.push_back(r / b);
            r %= b;
        }

        reverse(C.a.begin(), C.a.end());
        C.trim();

        return C;
    }

    // 大整数 % int
    friend int operator%(const BigInt &A, int b) {
        long long r = 0;

        for (int i = (int)A.a.size() - 1; i >= 0; i--)
            r = (r * 10 + A.a[i]) % b;

        return r;
    }

    // 输入
    friend istream& operator>>(istream &in, BigInt &x) {
        string s;
        in >> s;
        x = s;
        return in;
    }

    // 输出
    friend ostream& operator<<(ostream &out, const BigInt &x) {
        for (int i = (int)x.a.size() - 1; i >= 0; i--)
            out << x.a[i];

        return out;
    }
};