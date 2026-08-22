template <class T>
bool chmin(T &a, const T &b) {
	if (b < a) {
		a = b;
		return true;
	}
	return false;
}

template <class T>
bool chmax(T &a, const T &b) {
	if (b > a) {
		a = b;
		return true;
	}
	return false;
}