template <class T>
vector<T> sparse(const vector<T> &a, int offset) {
    auto sp = a;
    sort(sp.begin() + offset, sp.end());
    sp.erase(unique(sp.begin() + offset, sp.end()), sp.end());
    return sp;
}