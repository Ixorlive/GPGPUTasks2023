#pragma once
#include "point.h"
#include <cmath>
#include <limits>

struct BBox {

    BBox() {
        clear();
    }

    explicit BBox(const Point &point) : BBox() {
        grow(point);
    }

    BBox(float fx, float fy) {
        clear();
        grow(fx, fy);
    }

    void clear() {
        minx = std::numeric_limits<int>::max();
        maxx = std::numeric_limits<int>::lowest();
        miny = minx;
        maxy = maxx;
    }

    bool contains(const Point &point) const {
        return point.x >= minx && point.x <= maxx && point.y >= miny && point.y <= maxy;
    }

    bool contains(float fx, float fy) const {
        int x = fx + 0.5;
        int y = fy + 0.5;
        return x >= minx && x <= maxx && y >= miny && y <= maxy;
    }

    bool empty() const {
        return minx > maxx;
    }

    bool operator==(const BBox &other) const {
        return minx == other.minx && maxx == other.maxx && miny == other.miny && maxy == other.maxy;
    }

    bool operator!=(const BBox &other) const {
        return !(*this == other);
    }

    void grow(const Point &point) {
        minx = std::min(minx, point.x);
        maxx = std::max(maxx, point.x);
        miny = std::min(miny, point.y);
        maxy = std::max(maxy, point.y);
    }

    void grow(float fx, float fy) {
        minx = std::min(minx, int(fx + 0.5));
        maxx = std::max(maxx, int(fx + 0.5));
        miny = std::min(miny, int(fy + 0.5));
        maxy = std::max(maxy, int(fy + 0.5));
    }

    void grow(const BBox &other) {
        grow(other.min());
        grow(other.max());
    }

    int minX() const {
        return minx;
    }
    int maxX() const {
        return maxx;
    }
    int minY() const {
        return miny;
    }
    int maxY() const {
        return maxy;
    }

    Point min() const {
        return Point{minx, miny};
    }
    Point max() const {
        return Point{maxx, maxy};
    }

private:
    int minx, maxx;
    int miny, maxy;
};
