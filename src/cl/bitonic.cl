inline bool compare(const float x, const float y, bool is_less) {
    return is_less ? (x < y) : (x > y);
}

__kernel void bitonic(__global float *arr, const uint N, const uint sorted_size, const uint inner_block_len) {
    const size_t g_id = get_global_id(0);

    if (g_id >= N / 2) {
        return;
    }

    // first try
    // const size_t sequence_index = (g_id * 2) / seq_len;
    // const size_t block_index = (g_id * 2) / block_len;
    // const size_t element_id = (g_id % (block_len / 2)) + block_index * block_len;
    // const size_t paired_element_id = element_id + (block_len / 2);
    
    const size_t sorted_block_id = g_id / sorted_size;

    const size_t element_id = 2 * g_id - g_id % inner_block_len;
    const size_t paired_element_id = element_id + inner_block_len;

    if (compare(arr[element_id], arr[paired_element_id], sorted_block_id & 1)) {
        float temp = arr[element_id];
        arr[element_id] = arr[paired_element_id];
        arr[paired_element_id] = temp;
    }
}
