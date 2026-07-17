using F = long double;
constexpr F eps = 1e-8;
const F pi = acos(-1.L);

template <class T>
int sgn(T x) {
    if (fabs(x) < eps) {
        return 0;
    }
    return x < 0 ? -1 : 1;
}

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
    bool operator<(const Point<T> &o) const {
        if (sgn(x - o.x) == 0) {
            return y < o.y;
        }
        return x < o.x;
    }
    bool operator==(const Point<T> &o) const {
        return sgn(x - o.x) == 0 && sgn(y - o.y) == 0;
    }
    bool operator!=(const Point<T> &o) const {
        return !(*this == o);
    }
    friend ostream &operator<<(ostream &os, const Point<T> &p) {
        return os << "(" << p.x << ", " << p.y << ")";
    }
    friend istream &operator>>(istream &is, Point<T> &p) {
        is >> p.x >> p.y;
        return is;
    }
};

template <class T>
struct Line {
    Point<T> a;
    Point<T> b;
    Line(const Point<T> &a_ = Point<T>(), const Point<T> &b_ = Point<T>()): a(a_), b(b_) {}
};

// 点积
template <class T>
T dot(const Point<T> &a, const Point<T> &b) {
    return a.x * b.x + a.y * b.y;
}

// 叉积
template <class T>
T cross(const Point<T> &a, const Point<T> &b) {
    return a.x * b.y - a.y * b.x;
}

// 向量长度平方
template <class T>
T square(const Point<T> &p) {
    return dot(p, p);
}

// 向量长度
template <class T>
F length(const Point<T> &p) {
    return hypot<F>(p.x, p.y);
}

// 线段距离
template <class T>
F length(const Line<T> &l) {
    return length(l.a - l.b);
}

// 判断是否平行
template <class T>
bool parallel(const Line<T> &l1, const Line<T> &l2) {
    return sgn(cross(l1.a - l1.b, l2.a - l2.b)) == 0;
}

// 单位化向量
template <class T>
Point<F> normalize(const Point<T> &p) {
    F len = length(p);
    return Point{p.x / len, p.y / len};
}

// 两点间距离
template <class T>
F distance(const Point<T> &a, const Point<T> &b) {
    return length(b - a);
}

// 点到直线最短距离
template <class T>
F distancePL(const Point<T> &p, const Line<T> &l) {
    return abs(cross(l.a - p, l.b - p)) / length(l);
}

// 点到线段最短距离
template <class T>
F distancePS(const Point<T> &p, const Line<T> &l) {
    if (dot(p - l.a, l.b - l.a) < 0) {
        return distance(p, l.a);
    }
    if (dot(p - l.b, l.a - l.b) < 0) {
        return distance(p, l.b);
    }
    return distancePL(p, l);
}

// 向量逆时针旋转
template <class T>
Point<F> rotate(const Point<T> &p, F ang) {
    F s = sin(ang);
    F c = cos(ang);
    return Point{p.x * c - p.y * s, p.x * s + p.y * c};
}

// 计算两条直线交点
template<class T> 
Point<F> lineIntersection(const Line<T> &l1, const Line<T> &l2) { 
    F t = 1.L * cross(l2.b - l2.a, l1.a - l2.a) / cross(l2.b - l2.a, l1.a - l1.b);
    auto p = l1.b - l1.a;
    return Point{l1.a.x + p.x * t, l1.a.y + p.y * t};
}

// 凸包(仅支持整数)
template <class T>
auto getHull(vector<Point<T>> ps) {
    sort(ps.begin(), ps.end(), [&](const auto &p1, const auto &p2) {
        return p1.x == p2.x ? p1.y < p2.y : p1.x < p2.x;
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

// 判断点是否在直线左侧
template<class T>
bool pointOnLineLeft(const Point<T> &p, const Line<T> &l) {
    return cross(l.b - l.a, p - l.a) > 0;
}

// 判断点是否在线段上(仅支持整数)
template<class T>
bool pointOnSegment(const Point<T> &p, const Line<T> &l) {
    return cross(p - l.a, l.b - l.a) == 0 && min(l.a.x, l.b.x) <= p.x && p.x <= max(l.a.x, l.b.x)
        && min(l.a.y, l.b.y) <= p.y && p.y <= max(l.a.y, l.b.y);
}

// 判断点是否在多边形内部(仅支持整数)
template<class T>
bool pointInPolygon(const Point<T> &a, const vector<Point<T>> &p) {
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