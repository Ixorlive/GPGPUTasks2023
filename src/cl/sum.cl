#define VALUES_PER_WORKITEM 64
#define WORKGROUP_SIZE 64

__kernel void sum_atomic(__global const uint *arr, __global uint *sum, const uint n) {
    const size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    atomic_add(sum, arr[i]);
}


__kernel void sum_loop(__global const uint *arr, __global uint *sum, const uint n) {
    const size_t idx = get_global_id(0);
    uint local_res = 0;
    
    for (uint i = 0; i < VALUES_PER_WORKITEM; ++i) {
        uint pos = idx * VALUES_PER_WORKITEM + i;
        if (pos < n) {
            local_res += arr[pos];
        }
    }
    atomic_add(sum, local_res);
}

__kernel void sum_loop_coalesced(__global const uint *arr, __global uint *sum, const uint n) {
    const size_t lid = get_local_id(0);
    const size_t wid = get_group_id(0);
    const size_t grs = get_local_size(0);
    
    uint local_res = 0;

    for (uint i = 0; i < VALUES_PER_WORKITEM; ++i) {
        uint pos = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (pos < n) {
            local_res += arr[pos];
        }
    }

    atomic_add(sum, local_res);
}

__kernel void sum_local_mem_thread(__global const uint *arr, __global uint *sum, const uint n) {
    const size_t gid = get_global_id(0);
    const size_t lid = get_local_id(0);

    __local uint buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid != 0) {
        return;
    }
    uint group_res = 0;
    for (int i = 0; i < WORKGROUP_SIZE; ++i) {
        group_res += buf[i];
    }
    atomic_add(sum, group_res);
}

__kernel void sum_tree_local(__global const uint *arr, __global uint *sum, const uint n) {
    const size_t gid = get_global_id(0);
    const size_t lid = get_local_id(0);

    __local uint buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nValues = WORKGROUP_SIZE / 2; nValues > 0; nValues /= 2) {
        if (lid < nValues) {
             buf[lid] += buf[lid + nValues];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}

__kernel void sum_tree_global(__global const uint *arr, __global uint *sum, const uint n) {
    const size_t gid = get_global_id(0);
    const size_t lid = get_local_id(0);

    __local uint buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nValues = WORKGROUP_SIZE / 2; nValues > 0; nValues /= 2) {
        if (lid < nValues) {
             buf[lid] += buf[lid + nValues];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const size_t wid = get_group_id(0);
        sum[wid] = buf[0];
    }
}
