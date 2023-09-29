#ifndef TILE_SIZE
    #error "Tile size must be init"
#endif

/*
    Input: 
        Matrix M*K, M - row number, K - col number
    Result: 
        float *res -  transposed matrix
*/
__kernel void matrix_transpose(__global float *matrix, __global float *res, const uint M, const uint K) {
    const int g_col = get_global_id(0);
    const int g_row = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];

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

__kernel void matrix_transpose_no_coalesced(__global float *matrix, __global float *res, const uint M, const uint K) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    res[i * M + j] = matrix[j * K + i];
}