template <class T>
void hash_combine(size_t& seed, const T& v) {
    seed ^= hash<T>{}(v) 
          + 0x9e3779b97f4a7c15ULL 
          + (seed << 6) 
          + (seed >> 2);
}

struct Hash {
    size_t operator()(const array<int, 2> &a) const {
        size_t res = 0;
        for (auto &e : a) {
            hash_combine(res, e);
        }
        return res;
    };
};

unordered_map<array<int, 2>, int, Hash> M;