__kernel void prefix_sum_naive(__global const uint *src, __global uint *res, const uint size, const uint d) {
    const size_t g_id = get_global_id(0);
    if (g_id >= size)
        return;
    const size_t pw = 1 << (d - 1);
    if (g_id >= pw) {
        res[g_id] = src[g_id - pw] + src[g_id];
    } else {
        res[g_id] = src[g_id];
    }
}

__kernel void reduce(__global unsigned int *data, __global unsigned int *partial_sums, const unsigned int size,
                     const uint d) {
    const size_t g_id = get_global_id(0);

    if (g_id < size / 2 / d) {
        uint k = 2 * d * g_id;
        data[k + 2 * d - 1] += data[k + d - 1];
    }
    if (g_id == 0 && (d << 1) >= size) {
        *partial_sums = data[size - 1];
        data[size - 1] = 0;
    }
}

__kernel void downsweep(__global unsigned int *data, const unsigned int size, const uint d) {
    const size_t g_id = get_global_id(0);

    if (g_id < size / 2 / d) {
        uint k = 2 * d * g_id;
        uint l = data[k + d - 1];
        data[k + d - 1] = data[k + 2 * d - 1];
        data[k + 2 * d - 1] += l;
    }
}