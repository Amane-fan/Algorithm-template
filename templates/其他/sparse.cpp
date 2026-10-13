auto sp = a;
sort(sp.begin() + 1, sp.end());
sp.erase(unique(sp.begin() + 1, sp.end()), sp.end());

auto get = [&](i64 x) -> int {
    return lower_bound(sp.begin() + 1, sp.end(), x) - sp.begin();
};