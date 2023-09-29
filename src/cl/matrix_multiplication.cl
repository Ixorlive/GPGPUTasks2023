#ifndef TILE_SIZE
    #error "TILE_SIZE must be init"
#endif

#ifndef WORK_PER_THREAD
    #error "WORK_PER_THREAD must be init"
#endif


__kernel void matrix_multiplication_naive(const __global float *as, const __global float *bs, __global float *cs,
                                          const uint M, const uint K, const uint N) {
    const size_t g_col = get_global_id(0);
    const size_t g_row = get_global_id(1);

    float ci = 0.0f;
    for (size_t i = 0; i < K; ++i) {
        if (g_col < N && g_row < M)
            ci += as[g_row * K + i] * bs[i * N + g_col];
    }
    if (g_col < N && g_row < M)
        cs[g_row * K + g_col] = ci;
}

__kernel void matrix_multiplication_local_mem(const __global float *as, const __global float *bs, __global float *cs,
                                              const uint M, const uint K, const uint N) {
    const size_t g_col = get_global_id(0);
    const size_t g_row = get_global_id(1);
    const size_t l_col = get_local_id(0);
    const size_t l_row = get_local_id(1);


    __local float Asub[TILE_SIZE][TILE_SIZE + 1];
    __local float Bsub[TILE_SIZE][TILE_SIZE + 1];

    float acc = 0.0f;

    for (int i_tile = 0; i_tile < K; i_tile += TILE_SIZE) {
        Asub[l_row][l_col] = (g_row < M && i_tile + l_col < N) ? as[g_row * K + (i_tile + l_col)] : 0.0f;
        Bsub[l_row][l_col] = (g_col < N && i_tile + l_row < K) ? bs[(i_tile + l_row) * N + g_col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; ++i) {
            acc += Asub[l_row][i] * Bsub[i][l_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (g_row < M && g_col < N)
        cs[g_row * N + g_col] = acc;
}

// same as above with register optimization
__kernel void matrix_multiplication_local_mem_opt(const __global float *as, const __global float *bs,
                                                  __global float *cs, const uint M, const uint K, const uint N) {
    const size_t g_col = get_global_id(0);
    const size_t l_col = get_local_id(0);
    const size_t l_row = get_local_id(1);
    const size_t g_row = get_group_id(1) * TILE_SIZE + l_row;

    const size_t rts = TILE_SIZE / WORK_PER_THREAD;

    __local float Asub[TILE_SIZE][TILE_SIZE + 1];
    __local float Bsub[TILE_SIZE][TILE_SIZE + 1];

    float acc[WORK_PER_THREAD];
    for (size_t i = 0; i < WORK_PER_THREAD; ++i) {
        acc[i] = 0.0f;
    }

    const int num_tiles = K / WORK_PER_THREAD;
    for (int i_tile = 0; i_tile < K; i_tile += TILE_SIZE) {
        for (size_t w = 0; w < TILE_SIZE; w += rts) {
            const bool a_cond = (g_row + w < M && i_tile + l_col < N);
            const bool b_cond = (g_col < N && i_tile + l_row + w < K);

            Asub[l_row + w][l_col] = a_cond ? as[(g_row + w) * K + i_tile + l_col] : 0.0f;
            Bsub[l_row + w][l_col] = b_cond ? bs[(i_tile + l_row + w) * N + g_col] : 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; ++i) {
            for (int w = 0, wi = 0; w < TILE_SIZE; w += rts, ++wi) {
                acc[wi] += Asub[l_row + w][i] * Bsub[i][l_col];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0, wi = 0; w < TILE_SIZE; w += rts, ++wi) {
        const size_t res_row = g_row + w;
        if (res_row < M && g_col < N)
            cs[res_row * N + g_col] = acc[wi];
    }
}