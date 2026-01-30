constexpr ll inf = 2e18;
template <class T>
struct LiChaoTree {
	struct Line {
		T a, b;
		Line(): a(0), b(-inf) {
		}
		Line(T a, T b): a(a), b(b) {
		}
		T get(T x) {
			return a * x + b;
		}
	};
	int N;
	vector<T> x;
	vector<Line> ST;
	LiChaoTree() {}
	LiChaoTree(const vector<T> &x2) {
		x = x2;
		sort(x.begin(), x.end());
		x.erase(unique(x.begin(), x.end()), x.end());
		int N2 = x.size();
		N = 1;
		while (N < N2) {
			N *= 2;
		}
		x.resize(N);
		for (int i = N2; i < N; i++) {
			x[i] = x[N2 - 1];
		}
		ST = vector<Line>(N * 2 - 1);
	}
	void addLine(Line L, int i, int l, int r) {
		T la = L.get(x[l]);
		T lb = ST[i].get(x[l]);
		T ra = L.get(x[r - 1]);
		T rb = ST[i].get(x[r - 1]);
		if (la <= lb && ra <= rb) {
			return;
		} else if (la >= lb && ra >= rb) {
			ST[i] = L;
		} else {
			int m = (l + r) / 2;
			T ma = L.get(x[m]);
			T mb = ST[i].get(x[m]);
			if (ma > mb) {
				swap(L, ST[i]);
				swap(la, lb);
				swap(ra, rb);
			}
			if (la > lb) {
				addLine(L, i * 2 + 1, l, m);
			}
			if (ra > rb) {
				addLine(L, i * 2 + 2, m, r);
			}
		}
	}
	void addLine(T a, T b) {
		addLine(Line(a, b), 0, 0, N);
	}
	T getMax(T x2) {
		int p = lower_bound(x.begin(), x.end(), x2) - x.begin();
		p += N - 1;
		T ans = -inf;
		ans = max(ans, ST[p].get(x2));
		while (p > 0) {
			p = (p - 1) / 2;
			ans = max(ans, ST[p].get(x2));
		}
		return ans;
	}
};