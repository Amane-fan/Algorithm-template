// GNU C++17；整数坐标用 Point<i64>，浮点坐标用 Point<F>。
// 点积 / 叉积 / 面积和等均使用 T，须保证所有中间结果在 T 的范围内。
// 需要更大整数范围时使用 Point<i128>；整数坐标的除法仍是整数除法。
// 浮点判定用绝对 eps，需按题目尺度调整；排序、去重使用精确比较。
using F = long double;
constexpr F eps = 1e-8L;
const F pi = acos(-1.L);
// 符号：返回 -1/0/1；整数精确判定，浮点 |x| < eps 视为 0。
template <class T>
int sgn(T x) {
    if constexpr (is_floating_point_v<T>) {
        if (fabs(x) < eps) {
            return 0;
        }
    }
    return (x > 0) - (x < 0);
}

template <class T>
struct Point {
    T x, y;
    Point(const T &x = 0, const T &y = 0) : x(x), y(y) {}
    Point operator+(const Point &p) const { return {x + p.x, y + p.y}; }
    Point operator-(const Point &p) const { return {x - p.x, y - p.y}; }
    Point operator*(const T &v) const { return {x * v, y * v}; }
    Point operator/(const T &v) const { return {x / v, y / v}; }
    bool operator<(const Point &p) const {
        return x == p.x ? y < p.y : x < p.x;
    }
    bool operator==(const Point &p) const { return x == p.x && y == p.y; }
    bool operator!=(const Point &p) const { return !(*this == p); }
    friend ostream &operator<<(ostream &os, const Point &p) {
        return os << "(" << p.x << ", " << p.y << ")";
    }
    friend istream &operator>>(istream &is, Point &p) {
        return is >> p.x >> p.y;
    }
};

template <class T>
struct Line {
    Point<T> a, b;
    // 用两点表示直线 / 线段；有向直线的方向为 a -> b。
    Line(const Point<T> &a = {}, const Point<T> &b = {}) : a(a), b(b) {}
};

// 将坐标转为浮点类型 F，便于进行实数除法和几何构造。
template <class T>
Point<F> toFloat(const Point<T> &p) { return {F(p.x), F(p.y)}; }
// 两点判等：整数精确比较，浮点逐坐标使用 eps；不能用作排序比较器。
template <class T>
bool samePoint(const Point<T> &a, const Point<T> &b) {
    return sgn(a.x - b.x) == 0 && sgn(a.y - b.y) == 0;
}
// 向量点积 a·b，可用于判断夹角、计算投影。
template <class T>
T dot(const Point<T> &a, const Point<T> &b) {
    return a.x * b.x + a.y * b.y;
}
// 向量叉积 a×b；正数表示 b 在 a 的逆时针一侧。
template <class T>
T cross(const Point<T> &a, const Point<T> &b) {
    return a.x * b.y - a.y * b.x;
}
// 三点叉积：(b-a) × (c-a)，正数表示逆时针。
template <class T>
T cross(const Point<T> &a, const Point<T> &b, const Point<T> &c) {
    return cross(b - a, c - a);
}
// 向量长度的平方，比较距离时可避免开方。
template <class T>
T square(const Point<T> &p) { return dot(p, p); }
// 向量的欧几里得长度。
template <class T>
F length(const Point<T> &p) { return hypot(F(p.x), F(p.y)); }
// 两点间的欧几里得距离。
template <class T>
F distance(const Point<T> &a, const Point<T> &b) {
    return length(toFloat(b) - toFloat(a));
}
// 线段长度，即两个端点的距离。
template <class T>
F length(const Line<T> &l) { return distance(l.a, l.b); }
// 返回同方向的单位向量；调用者须保证向量非零。
template <class T>
Point<F> normalize(const Point<T> &p) {
    F len = length(p);
    assert(len > 0);
    return toFloat(p) / len;
}
// 向量逆时针旋转 90°，保持原坐标类型。
template <class T>
Point<T> rotate90(const Point<T> &p) { return {-p.y, p.x}; }
// 逆时针旋转 ang 弧度。
template <class T>
Point<F> rotate(const Point<T> &p, F ang) {
    F s = sin(ang), c = cos(ang);
    return {p.x * c - p.y * s, p.x * s + p.y * c};
}
// 两个非零向量的夹角，[0, pi]。
template <class T>
F angle(const Point<T> &a, const Point<T> &b) {
    auto u = normalize(a), v = normalize(b);
    return atan2(fabs(cross(u, v)), dot(u, v));
}
// 判断两条直线是否平行（含重合）；两条直线均须端点不同。
template <class T>
bool parallel(const Line<T> &a, const Line<T> &b) {
    return sgn(cross(a.b - a.a, b.b - b.a)) == 0;
}
// 判断点是否严格在有向直线 a -> b 左侧，在线上返回 false；要求 a != b。
template <class T>
bool pointOnLineLeft(const Point<T> &p, const Line<T> &l) {
    return sgn(cross(l.a, l.b, p)) > 0;
}
// 判断点是否在线段上，包含端点；允许线段退化为点。
template <class T>
bool pointOnSegment(const Point<T> &p, const Line<T> &l) {
    return sgn(cross(l.a, l.b, p)) == 0
        && sgn(p.x - min(l.a.x, l.b.x)) >= 0
        && sgn(p.x - max(l.a.x, l.b.x)) <= 0
        && sgn(p.y - min(l.a.y, l.b.y)) >= 0
        && sgn(p.y - max(l.a.y, l.b.y)) <= 0;
}
// 判断两线段是否相交，含端点接触、共线重叠及退化线段。
// strict=true 仅判断两段内部的非共线交叉。
template <class T>
bool segmentIntersection(const Line<T> &a, const Line<T> &b, bool strict = false) {
    int x = sgn(cross(a.a, a.b, b.a)), y = sgn(cross(a.a, a.b, b.b));
    int u = sgn(cross(b.a, b.b, a.a)), v = sgn(cross(b.a, b.b, a.b));
    if (x * y < 0 && u * v < 0) {
        return true;
    }
    return !strict && (pointOnSegment(a.a, b) || pointOnSegment(a.b, b)
                   || pointOnSegment(b.a, a) || pointOnSegment(b.b, a));
}
// 唯一直线交点；先检查非平行，平行/重合/退化直线不能调用。
template <class T>
Point<F> lineIntersection(const Line<T> &a, const Line<T> &b) {
    auto u = a.b - a.a, v = b.b - b.a;
    auto d = cross(u, v);
    assert(sgn(d) != 0);
    F t = F(cross(b.a - a.a, v)) / F(d);
    return toFloat(a.a) + toFloat(u) * t;
}
// 点在直线上的投影；若 a=b，约定投影为 a。
template <class T>
Point<F> projection(const Point<T> &p, const Line<T> &l) {
    auto a = toFloat(l.a), v = toFloat(l.b) - a;
    F d = square(v);
    if (d == 0) {
        return a;
    }
    return a + v * (dot(toFloat(p) - a, v) / d);
}
// 点关于直线的对称点；若 a=b，则返回关于点 a 的中心对称点。
template <class T>
Point<F> reflection(const Point<T> &p, const Line<T> &l) {
    return projection(p, l) * 2 - toFloat(p);
}
// 点到直线的最短距离；退化直线按点处理。
template <class T>
F distancePL(const Point<T> &p, const Line<T> &l) {
    return length(toFloat(p) - projection(p, l));
}
// 点到线段的最短距离；垂足越界时取端点，允许退化为点。
template <class T>
F distancePS(const Point<T> &p, const Line<T> &l) {
    auto a = toFloat(l.a), v = toFloat(l.b) - a, u = toFloat(p) - a;
    F d = square(v);
    if (d == 0) {
        return length(u);
    }
    F t = clamp(dot(u, v) / d, F(0), F(1));
    return length(u - v * t);
}
// 两线段的最短距离，相交时为 0；允许退化为点。
template <class T>
F distanceSS(const Line<T> &a, const Line<T> &b) {
    if (segmentIntersection(a, b)) {
        return 0;
    }
    return min({distancePS(a.a, b), distancePS(a.b, b),
                distancePS(b.a, a), distancePS(b.b, a)});
}
// 多边形顶点按边界顺序给出，不重复首点；有向面积的两倍，逆时针为正。
template <class T>
T area2(const vector<Point<T>> &p) {
    T ans = 0;
    for (int i = 1; i + 1 < int(p.size()); i++) {
        ans += cross(p[0], p[i], p[i + 1]);
    }
    return ans;
}
// 简单多边形的非负面积，O(n)；不足三个点时为 0。
template <class T>
F area(const vector<Point<T>> &p) { return fabs(F(area2(p))) / 2; }
// 闭合多边形的周长，O(n)；两点时为线段长度的两倍，空集/单点为 0。
template <class T>
F perimeter(const vector<Point<T>> &p) {
    F ans = 0;
    int n = p.size();
    for (int i = 0; i < n; i++) {
        ans += distance(p[i], p[(i + 1) % n]);
    }
    return ans;
}
// 判断点与简单多边形的位置关系，顺/逆时针均可，O(n)。
// 返回 -1 外部、0 边界、1 内部；空多边形返回 -1。
template <class T>
int pointPolygonRelation(const Point<T> &a, const vector<Point<T>> &p) {
    int n = p.size(), wn = 0;
    for (int i = 0; i < n; i++) {
        auto u = p[i], v = p[(i + 1) % n];
        if (pointOnSegment(a, Line<T>(u, v))) {
            return 0;
        }
        int x = sgn(u.y - a.y), y = sgn(v.y - a.y);
        int z = sgn(cross(u, v, a));
        if (x <= 0 && y > 0 && z > 0) {
            wn++;
        }
        if (y <= 0 && x > 0 && z < 0) {
            wn--;
        }
    }
    return wn ? 1 : -1;
}
// 判断点是否在简单多边形内，包含边界，O(n)。
template <class T>
bool pointInPolygon(const Point<T> &a, const vector<Point<T>> &p) {
    return pointPolygonRelation(a, p) >= 0;
}
// 构造凸包的上下凸链，O(n log n)；返回 {下凸链, 上凸链}，均从左到右。
// 精确去重，去掉边上共线中间点；整数精确判定，浮点使用 eps。
template <class T>
auto getHull(vector<Point<T>> ps) {
    sort(ps.begin(), ps.end());
    ps.erase(unique(ps.begin(), ps.end()), ps.end());
    vector<Point<T>> lo, hi;
    for (auto p : ps) {
        while (lo.size() > 1
            && sgn(cross(lo[lo.size() - 2], lo.back(), p)) <= 0) {
            lo.pop_back();
        }
        while (hi.size() > 1
            && sgn(cross(hi[hi.size() - 2], hi.back(), p)) >= 0) {
            hi.pop_back();
        }
        lo.push_back(p);
        hi.push_back(p);
    }
    return make_pair(lo, hi);
}
// 构造完整凸包，O(n log n)，逆时针且不重复首点。
// 空集返回空，单个不同点只返回一个。
// 全共线时仅保留两端点。
template <class T>
vector<Point<T>> convexHull(vector<Point<T>> ps) {
    auto [lo, hi] = getHull(move(ps));
    for (int i = int(hi.size()) - 2; i > 0; i--) {
        lo.push_back(hi[i]);
    }
    return lo;
}
// 旋转卡壳求直径平方，O(n)；输入必须是 convexHull 返回的凸包。
template <class T>
T convexDiameter2(const vector<Point<T>> &p) {
    int n = p.size();
    if (n < 2) {
        return 0;
    }
    if (n == 2) {
        return square(p[0] - p[1]);
    }
    T ans = 0;
    for (int i = 0, j = 1; i < n; i++) {
        int k = (i + 1) % n;
        while (cross(p[i], p[k], p[(j + 1) % n]) > cross(p[i], p[k], p[j])) {
            j = (j + 1) % n;
        }
        ans = max({ans, square(p[i] - p[j]), square(p[k] - p[j])});
    }
    return ans;
}

// using P = Point<i64>; vector<P> ps = {{0, 0}, {2, 0}, {0, 2}};
// auto hull = convexHull(ps);
// cout << area(hull) << ' ' << sqrt(F(convexDiameter2(hull))); // 2, sqrt(8)
// pointPolygonRelation(P(1, 1), hull) == 0；pointInPolygon 包含边界。
// Point 的 == 改为精确相等；浮点近似判等用 samePoint(a, b)。
