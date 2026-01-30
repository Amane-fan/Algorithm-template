using F = long double;
constexpr F eps = 1e-8;

template <class T>
struct Point {
    T x, y;
    Point(const T &x_ = 0, const T &y_ = 0) : x(x_), y(y_) {}
    Point operator+(const Point &p) const {
        return {x + p.x, y + p.y};
    }
    Point operator-(const Point &p) const {
        return {x - p.x, y - p.y};
    }
    Point operator*(const T &v) const {
        return {x * v, y * v};
    }
    Point operator/(const T &v) const {
        return {x / v, y / v};
    }
};

template <class T>
struct Line {
    Point<T> a;
    Point<T> b;
    Line(const Point<T> &a_ = Point<T>(), const Point<T> &b_ = Point<T>()): a(a_), b(b_) {}
};

template <class T>
std::ostream &operator<<(std::ostream &os, const Point<T> &p) {
    return os << "(" << p.x << ", " << p.y << ")";
}

template <class T>
std::istream &operator>>(std::istream &is, Point<T> &p) {
    is >> p.x >> p.y;
    return is;
}

template <class T>
T dot(const Point<T> &a, const Point<T> &b) {
    return a.x * b.x + a.y * b.y;
}

template <class T>
T cross(const Point<T> &a, const Point<T> &b) {
    return a.x * b.y - a.y * b.x;
}

template <class T>
T square(const Point<T> &p) {
    return dot(p, p);
}

template <class T>
double length(const Point<T> &p) {
    return sqrt(square(p));
}

template <class T>
double length(const Line<T> &l) {
    return length(l.a - l.b);
}

template <class T>
bool parallel(const Line<T> &l1, const Line<T> &l2) {
    return equal(cross(l1.a - l1.b, l2.a - l2.b), T(0));
}

template <class T>
Point<T> normalize(const Point<T> &p) {
    return p / length(p);
}

template <class T>
double distance(const Point<T> &a, const Point<T> &b) {
    return length(b - a);
}

template <class T>
double distancePL(const Point<T> &p, const Line<T> &l) {
    return abs(cross(l.a - p, l.b - p)) / length(l);
}

template <class T>
double distancePS(const Point<T> &p, const Line<T> &l) {
    if (dot(p - l.a, l.b - l.a) < 0) {
        return distance(p, l.a);
    }
    if (dot(p - l.b, l.a - l.b) < 0) {
        return distance(p, l.b);
    }
    return distancePL(p, l);
}

template <class T>
Point<T> lineIntersection(const Line<T> &l1, const Line<T> &l2) {
    return l1.a + (l1.b - l1.a) * cross(l2.a - l1.a, l2.b - l2.a) / cross(l2.b - l2.a, l1.b - l1.a);
}

template <class T>
bool equal(const T &x, const T &y) {
    if constexpr (is_floating_point_v<T>) {
        return fabs(x - y) < eps;
    } else {
        return x == y;
    }
}

template <class T>
auto getHull(vector<Point<T>> ps) {
    sort(ps.begin(), ps.end(), [&](const auto &p1, const auto &p2) {
        return equal(p1.x, p2.x) ? p1.y < p2.y : p1.x < p2.x;
    });
    vector<Point<T>> hi, lo;
    for (auto &p : ps) {
        while (lo.size() > 1 && cross(lo.back() - lo[lo.size() - 2], p - lo.back()) <= 0) {
            lo.pop_back();
        }
        lo.push_back(p);

        while (hi.size() > 1 && cross(hi.back() - hi[hi.size() - 2], p - hi.back()) >= 0) {
            hi.pop_back();
        }
        hi.push_back(p);
    }

    return make_pair(lo, hi);
}

template<class T>
bool pointOnLineLeft(const Point<T> &p, const Line<T> &l) {
    return cross(l.b - l.a, p - l.a) > 0;
}

template<class T>
bool pointOnSegment(const Point<T> &p, const Line<T> &l) {
    return cross(p - l.a, l.b - l.a) == 0 && std::min(l.a.x, l.b.x) <= p.x && p.x <= std::max(l.a.x, l.b.x)
        && std::min(l.a.y, l.b.y) <= p.y && p.y <= std::max(l.a.y, l.b.y);
}

template<class T>
bool pointInPolygon(const Point<T> &a, const std::vector<Point<T>> &p) {
    int n = p.size();
    for (int i = 0; i < n; i++) {
        if (pointOnSegment(a, Line(p[i], p[(i + 1) % n]))) {
            return true;
        }
    }
     
    int t = 0;
    for (int i = 0; i < n; i++) {
        auto u = p[i];
        auto v = p[(i + 1) % n];
        if (u.x < a.x && v.x >= a.x && pointOnLineLeft(a, Line(v, u))) {
            t ^= 1;
        }
        if (u.x >= a.x && v.x < a.x && pointOnLineLeft(a, Line(u, v))) {
            t ^= 1;
        }
    }
     
    return t == 1;
}