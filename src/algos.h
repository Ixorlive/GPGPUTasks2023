#pragma once
#include "point.h"
#include <algorithm>
#include <cmath>
#include <vector>

void bresenham(std::vector<Point> &line_points, const Point &from, const Point &to) {
    line_points.clear();

    double x1 = from.x;
    double y1 = from.y;
    double x2 = to.x;
    double y2 = to.y;

    const bool steep = (fabs(y2 - y1) > fabs(x2 - x1));
    if (steep) {
        std::swap(x1, y1);
        std::swap(x2, y2);
    }
    const bool flip = x1 > x2;
    if (flip) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }

    const double dx = x2 - x1;
    const double dy = fabs(y2 - y1);

    double error = dx / 2.0;
    const int ystep = (y1 < y2) ? 1 : -1;
    int y = (int) y1;

    const int max_x = (int) x2;

    line_points.push_back(from);

    for (int x = (int) x1; x < max_x; x++) {
        Point point = (steep) ? Point{y, x} : Point{x, y};
        if (line_points.back() != point) {
            line_points.push_back(point);
        }

        error -= dy;
        if (error < 0) {
            y += ystep;
            error += dx;
        }
    }

    if (line_points.back() != to) {
        line_points.push_back(to);
    }

    if (line_points.size() > 2 && flip) {
        std::reverse(++line_points.begin(), --line_points.end());
    }
}

//Преобразует 0xbbbb в 0x0b0b0b0b
int spreadBits(int word) {
    word = (word ^ (word << 8)) & 0x00ff00ff;
    word = (word ^ (word << 4)) & 0x0f0f0f0f;
    word = (word ^ (word << 2)) & 0x33333333;
    word = (word ^ (word << 1)) & 0x55555555;
    return word;
}
