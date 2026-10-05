struct Hash {
    u64 seed = rnd();
    static u64 mix(u64 x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    size_t operator()(u64 x) const {
        return size_t(mix(x + seed));
    }
    template <class T, size_t N>
    size_t operator()(const array<T, N> &a) const {
        u64 h = seed ^ u64(N);
        for (const auto &x : a) {
            h = mix(h ^ u64(x));
        }
        return size_t(h);
    }
};
