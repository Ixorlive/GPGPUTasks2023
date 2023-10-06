#define LESS(x, y) ((x) < (y))
#define LESS_EQ(x, y) ((x) <= (y))
#define COMP(x, y, eq) ((eq) ? LESS_EQ(x, y) : LESS(x, y))

__kernel void sort_workgroup(__global float *arr, __global float *res, const uint n) {
    const uint g_id = get_global_id(0);
    if (g_id >= n) {
        return;
    }
    const uint l_id = get_local_id(0);
    const uint g_size = get_local_size(0);
    const uint group_id = get_group_id(0);

    const float el = arr[g_id];

    const uint start_index = g_size * group_id;
    // thanks to the l1 cache it should be optimal?
    uint res_id = 0;
    for (int i = 0; i < g_size; ++i) {
        if (arr[start_index + i] < el || (arr[start_index + i] == el && i < l_id))
            ++res_id;
    }
    res[start_index + res_id] = el;
}

inline uint binary_search(__global float *arr, const uint n, const float val, const int eq) {
    uint l = 0;
    uint r = n;
    while (l < r) {
        uint mid = l + (r - l) / 2;
        if (COMP(val, arr[mid], eq)) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    if (l < n && !(COMP(val, arr[l], eq))) {
        ++l;
    }
    return l;
}

__kernel void merge(__global float *arr, __global float *res, const uint n, const uint block_size) {
    const uint g_id = get_global_id(0);
    if (g_id >= n) {
        return;
    }
    const uint block_id = g_id / block_size;
    const int is_second = block_id % 2;
    const float el = arr[g_id];
    uint pos = is_second ? g_id - block_size : g_id;
    pos += binary_search(arr + block_size * (block_id ^ 1), block_size, el, is_second ^ 1);
    res[pos] = el;
}

// 4
// 1 2 3 4 4 4 5 5 5 6 6 7 8