vector<int> prime, minp;
void sieve(int n) {
    minp.assign(n + 1, 0);
    prime.clear();
    
    for (int i = 2; i <= n; i++) {
        if (minp[i] == 0) {
            minp[i] = i;
            prime.push_back(i);
        }

        for (auto p : prime) {
            if (1LL * p * i > n) {
                break;
            }
            minp[p * i] = p;
            if (minp[i] == p) {
                break;
            }
        }
    }
}