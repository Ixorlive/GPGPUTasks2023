#ifndef TILE_SIZE
    #error "Tile size must be init"
#endif

#ifndef NUM_BITS
    #error "Num bits must be init"
#endif

#define NUM_VALS (1 << NUM_BITS)

inline uint extract_value(const uint val, const uint iter) {
    return (val >> (iter * NUM_BITS)) & (NUM_VALS - 1);
}

__kernel void radix(__global const uint *as, __global const uint *prefix_matrix, __global uint *res, const uint as_size,
                    const uint num_rows, const uint num_cols, const uint iter) {
    const size_t g_id = get_global_id(0);
    if (g_id >= as_size) {
        return;
    }
    
    const size_t k = extract_value(as[g_id], iter);
    const size_t group_id = get_group_id(0);
    const size_t l_id = get_local_id(0);
    
    size_t idx = l_id;
    
    if (group_id + k > 0) {
        idx += prefix_matrix[k * num_cols + group_id - 1];
    }
    
    const size_t work_size = get_local_size(0);
    const size_t start_pos = group_id * work_size;

    for (uint i = 0; i < l_id; ++i) {
        const uint as_el = extract_value(as[start_pos + i], iter);
        if (as_el == k)
            break;
        --idx;
    }

    res[idx] = as[g_id];
}

__kernel void fill_count_matrix(__global const uint *as, __global uint *matrix, const uint size, const uint iter) {
    const size_t g_id = get_global_id(0);
    if (g_id >= size) {
        return;
    }
    const size_t group_id = get_group_id(0);
    const size_t l_id = get_local_id(0);
    
    __local uint local_matrix[NUM_VALS];
    
    if (l_id < NUM_VALS) {
        local_matrix[l_id] = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_inc(&local_matrix[extract_value(as[g_id], iter)]);
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_id < NUM_VALS) {
        matrix[group_id * NUM_VALS + l_id] = local_matrix[l_id];
    }
}

__kernel void matrix_transpose(__global const uint *matrix, __global uint *res, const uint M, const uint K) {
    const int g_col = get_global_id(0);
    const int g_row = get_global_id(1);

    __local uint tile[TILE_SIZE][TILE_SIZE];

    const int l_col = get_local_id(0);
    const int l_row = get_local_id(1);

    if (g_row < M && g_col < K)
        tile[l_row][l_col] = matrix[g_row * K + g_col];

    barrier(CLK_LOCAL_MEM_FENCE);

    const int diff = l_row - l_col;
    const int res_row = g_col + diff;
    const int res_col = g_row - diff;
    if (res_row < K && res_col < M) {
        res[res_row * M + res_col] = tile[l_col][l_row];
    }
}


__kernel void sort_workgroup(__global const uint *arr, __global uint *res, const uint n, const uint iter) {
    const uint g_id = get_global_id(0);
    if (g_id >= n) {
        return;
    }
    const uint l_id = get_local_id(0);
    const uint g_size = get_local_size(0);
    const uint group_id = get_group_id(0);

    const uint src_el = arr[g_id];
    const uint el = extract_value(arr[g_id], iter);

    const uint start_index = g_size * group_id;

    uint res_id = 0;
    for (int i = 0; i < g_size; ++i) {
        if (start_index + i >= n)
            break;
        const uint arr_val = extract_value(arr[start_index + i], iter);
        if (arr_val < el || (arr_val == el && i < l_id))
            ++res_id;
    }
    res[start_index + res_id] = src_el;
}

// naive impl
__kernel void prefix_sum(__global const uint *src, __global uint *res, const uint size, const uint d) {
    const size_t g_id = get_global_id(0);
    if (g_id >= size)
        return;
    const size_t pw = 1 << (d - 1);
    res[g_id] = src[g_id] + (g_id >= pw ? src[g_id - pw] : 0);
}