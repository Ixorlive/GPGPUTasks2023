#pragma once
#include "bbox.h"

bool compareFloats(float lhs, float rhs, float eps = 1.e-5) {
    return std::abs(lhs - rhs) < eps;
}

#pragma pack(push, 1)
struct Node {

    bool hasLeftChild() const {
        return child_left >= 0;
    }
    bool hasRightChild() const {
        return child_right >= 0;
    }
    bool isLeaf() const {
        return !hasLeftChild() && !hasRightChild();
    }

    bool operator==(const Node &other) const {
        bool res = std::tie(child_left, child_right, bbox) == std::tie(other.child_left, other.child_right, other.bbox);
        res &= compareFloats(mass, other.mass);
        res &= compareFloats(cmsx, other.cmsx);
        res &= compareFloats(cmsy, other.cmsy);
        return res;
    }

    bool operator!=(const Node &other) const {
        return !(*this == other);
    }

    int child_left, child_right;
    BBox bbox;

    // used only for nbody
    float mass;
    float cmsx;
    float cmsy;
};
#pragma pack(pop)
