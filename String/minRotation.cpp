template <class T>
int minRotation(const T &s) {
    int n = s.size();
    int i = 0, j = 1, k = 0;

    while (i < n && j < n && k < n) {
        auto a = s[(i + k) % n];
        auto b = s[(j + k) % n];

        if (a == b) {
            ++k;
        } else {
            if (a > b) {
                i = i + k + 1;
                if (i == j) ++i;
            } else {
                j = j + k + 1;
                if (i == j) ++j;
            }
            k = 0;
        }
    }

    return min(i, j);
}