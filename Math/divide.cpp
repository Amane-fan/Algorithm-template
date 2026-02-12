template <class T>
T floor_div(const T &a, const T &b) {
    assert(b != 0);
    T q = a / b;
    T r = a % b;
    if (r != 0 && ((r > 0) != (b > 0))) {
        --q;
    }
    return q;
}
 
template <class T>
T ceil_div(const T &a, const T &b) {
    assert(b != 0);
    T q = a / b;
    T r = a % b;
    if (r != 0 && ((r > 0) == (b > 0))) {
        ++q;
    }
    return q;
}