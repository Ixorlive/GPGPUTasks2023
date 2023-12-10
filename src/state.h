#pragma once
#include <vector>

struct DeltaState {
    std::vector<float> dvx2d;
    std::vector<float> dvy2d;
};

struct State {

    State() {
    }
    State(int N) : pxs(N), pys(N), vxs(N), vys(N), mxs(N), coord_shift(0) {
    }

    std::vector<float> pxs;
    std::vector<float> pys;

    std::vector<float> vxs;
    std::vector<float> vys;

    std::vector<float> mxs;

    int coord_shift;
};
