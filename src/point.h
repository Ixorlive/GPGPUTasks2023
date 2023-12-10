#pragma once
#include <tuple>

struct Point {
    int x, y;

    bool operator==(const Point &rhs) const {
        return std::tie(x, y) == std::tie(rhs.x, rhs.y);
    }

    bool operator!=(const Point &rhs) const {
        return !(rhs == *this);
    }

    Point &operator+=(const Point &other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Point &operator-=(const Point &other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    Point operator-(const Point &other) const {
        Point result = *this;
        result -= other;
        return result;
    }
};
