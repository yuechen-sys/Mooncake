// clang-format off

#include <cooperative_groups.h>
#include <cstdio>
#include <cuda/atomic>

#include <mooncake_ep_configs.cuh>
#include <mooncake_ep_exception.cuh>
#include <mooncake_ep_launch.cuh>
#include <mooncake_ibgda/mlx5gda.h>
#include <mooncake_ep_utils.cuh>

namespace mooncake {

static __device__ void device_mutex_lock_system(uint32_t *mutex) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> lock(*mutex);
    // Spin until the mutex is acquired
    while (lock.exchange(1, cuda::memory_order_acquire) != 0);
}

static __device__ void device_mutex_unlock_system(uint32_t *mutex) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> lock(*mutex);
    // Release the mutex
    lock.store(0, cuda::memory_order_release);
}

static __device__ uint16_t device_byteswap(uint16_t x) {
    return __byte_perm(x, x, 0x2301);
}

static __device__ uint32_t device_byteswap(uint32_t x) {
    return __byte_perm(x, x, 0x0123);
}

static __device__ uint64_t device_byteswap(uint64_t x) {
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t lo = (uint32_t)(x);

    hi = __byte_perm(hi, hi, 0x0123);
    lo = __byte_perm(lo, lo, 0x0123);

    return ((uint64_t)lo << 32) | hi;
}

__device__ static inline uint16_t ptx_ld16_acq_sys_na(uint16_t *ptr) {
    uint16_t val;
    asm volatile("ld.acquire.sys.global.L1::no_allocate.b16 %0, [%1];" : "=h"(val) : "l"(ptr));
    return val;
}

// must be called with mutex locked
static __device__ void __mlx5gda_device_poll_cq(struct mlx5gda_qp_devctx *ctx, uint16_t expect) {
    uint16_t wq_tail = ctx->wq_tail;
    while ((int16_t)(wq_tail - expect) <= 0) {
        uint16_t cq_wqe_counter_be = ptx_ld16_acq_sys_na(&ctx->cq->wqe_counter);
        // printf("cq_wqe_counter_be=0x%x, expect=0x%x\n", cq_wqe_counter_be, expect);
        uint8_t opcode = ctx->cq->op_own >> 4;
        if (opcode == 0xD) {
            printf("Requester_Error: syndrome = 0x%lx\n", ctx->cq->timestamp >> 56);
        }
        EP_DEVICE_ASSERT(opcode == 0x0 || opcode == 0xF);
        wq_tail = device_byteswap(cq_wqe_counter_be) + 1;
    }
    if (wq_tail != ctx->wq_tail) {
        ctx->wq_tail = wq_tail;
    }
}

__device__ static inline void ptx_st32_rel_sys_na(uint32_t *ptr, uint32_t val) {
    asm volatile("st.release.sys.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ static inline void ptx_st64_rel_sys_na(uint64_t *ptr, uint64_t val) {
    asm volatile("st.release.sys.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}

/**
 * Ring DB to post WQs up to last_posted_wqe.
 *
 * Must be called with mutex locked.
 */
static __device__ void __mlx5gda_device_post_send_db(struct mlx5gda_qp_devctx *ctx) {
    // 1. update dbr
    uint32_t num_posted_wqe = (uint32_t)(ctx->wq_head);
    ptx_st32_rel_sys_na(&ctx->dbr->send_counter, device_byteswap((uint32_t)ctx->wq_head));
    // 2. ring db
    struct mlx5gda_wqebb *last_wqe = ctx->wq + ((num_posted_wqe - 1) & ctx->wqeid_mask);
    // printf("Last wqe=%lx -> bf=%p\n", *(uint64_t*)last_wqe, ctx->bf + ctx->bf_offset);
    ptx_st64_rel_sys_na((uint64_t*)(ctx->bf + ctx->bf_offset), *(uint64_t*)last_wqe);
    // 3. toggle bf
    ctx->bf_offset ^= MLX5GDA_BF_SIZE;
}

static __device__ void __mlx5gda_device_write_rdma_write_wqe(
    struct mlx5gda_qp_devctx *ctx, uint64_t laddr, __be32 lkey,
    uint64_t raddr, __be32 rkey, uint32_t bytes) {
    struct mlx5gda_rdma_write_wqe *wqe = (mlx5gda_rdma_write_wqe *)(ctx->wq + (ctx->wq_head & ctx->wqeid_mask));
    struct mlx5_wqe_ctrl_seg &ctrl_seg = wqe->ctrl;
    struct mlx5_wqe_raddr_seg &raddr_seg = wqe->raddr;
    struct mlx5_wqe_data_seg &data_seg = wqe->data;

    ctrl_seg = {};
    ctrl_seg.qpn_ds = device_byteswap((ctx->qpn << 8) | 3);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = device_byteswap(((uint32_t)ctx->wq_head << 8) | MLX5_OPCODE_RDMA_WRITE);

    raddr_seg.raddr = device_byteswap(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    data_seg.byte_count = device_byteswap(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = device_byteswap(laddr);

    ++ctx->wq_head;
}

struct mlx5_wqe_atomic_add_32_seg {
    __be32		add_data;
    __be32		field_boundary;
    __be64		compare;
};

static __device__ void __mlx5gda_device_write_rdma_atomic_add_wqe(
    struct mlx5gda_qp_devctx *ctx, const int& value, uint64_t laddr,
    __be32 lkey, uint64_t raddr, __be32 rkey) {
    struct mlx5gda_rdma_atomic_wqe *wqe = (mlx5gda_rdma_atomic_wqe *)(ctx->wq + (ctx->wq_head & ctx->wqeid_mask));
    struct mlx5_wqe_ctrl_seg &ctrl_seg = wqe->ctrl;
    struct mlx5_wqe_raddr_seg &raddr_seg = wqe->raddr;
    struct mlx5_wqe_data_seg &data_seg = wqe->data;

    ctrl_seg = {};
    ctrl_seg.qpn_ds = device_byteswap((ctx->qpn << 8) | 4);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = device_byteswap(MLX5_OPCODE_ATOMIC_MASKED_FA | ((uint32_t)ctx->wq_head << 8) | 0x08000000);

    raddr_seg.raddr = device_byteswap(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    auto atomic_add_32_seg = reinterpret_cast<mlx5_wqe_atomic_add_32_seg *>(&wqe->atomic);
    atomic_add_32_seg->add_data = device_byteswap((uint32_t)value);
    atomic_add_32_seg->field_boundary = 0;
    atomic_add_32_seg->compare = 0;

    data_seg.byte_count = device_byteswap((uint32_t)4);
    data_seg.lkey = lkey;
    data_seg.addr = device_byteswap(laddr);

    ++ctx->wq_head;
}

template <bool kUseFP8, int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * 32, 1) void
dispatch(void* packed_recv_x, float* packed_recv_x_scales,
         int* packed_recv_src_info, int64_t* packed_recv_layout_range,
         int* packed_recv_count, int32_t* active_ranks,
         void* mxa_buffer,
         int* rdma_send_signal_buffer, int* rdma_recv_signal_buffer,
         void* rdma_send_data_buffer, void* rdma_recv_data_buffer,
         void* cuda_counter_buffer, void* cuda_data_buffer,
         void* raddrs, void* rkeys, void* qp_devctxs,
         const int32_t* nvlink_available, void* const* ipc_peer_ptrs,
         const void* x, const int64_t* topk_idx,
         int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
         int* next_clean_buffer,
         int num_tokens, int num_max_dispatch_tokens_per_rank,
         int num_topk, int num_experts, int rank, int num_ranks,
         int64_t timeout_ticks,
         int phases) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
    constexpr float kFP8Margin = 1e-4, kFP8Amax = 448, kFP8AmaxInv = 1.0f / 448.0f;
    const int num_scales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: hidden data, FP8 scales, index at source
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
    const size_t num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16)));
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

    // IBGDA
    auto raddr_array = reinterpret_cast<uint64_t*>(raddrs);
    auto rkey_array = reinterpret_cast<uint32_t*>(rkeys);
    auto ctx_array = reinterpret_cast<mlx5gda_qp_devctx*>(qp_devctxs);
    const size_t num_qp_per_rank = MAX_QP_COUNT / num_ranks;

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // Expert counts
    __shared__ int shared_num_tokens_sent_per_expert[kNumWarpGroups];

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warp_id < num_warps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        const auto num_threads = (num_warps - 1) * 32;
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = reinterpret_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(rdma_send_data_buffer) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // Overlap top-k index read and source token index write
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            // FP8 cast
            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                    // Read
                    auto int4_value = __ldg(x_int4 + i);

                    if (kUseFP8) {
                        // Calculate local amax
                        auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
                        float fp32_values[kNumElemsPerRead];
                        float amax = kFP8Margin, scale, scale_inv;
                        #pragma unroll
                        for (int j = 0; j < kNumElemsPerRead; ++ j) {
                            fp32_values[j] = __bfloat162float(bf16_values[j]);
                            amax = fmaxf(amax, fabsf(fp32_values[j]));
                        }

                        // Reduce amax and scale
                        EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
                        amax = half_warp_reduce_max(amax), scale = kFP8Amax / amax, scale_inv = amax * kFP8AmaxInv;
                        if (lane_id == 0 or lane_id == 16)
                            rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

                        // Cast into send buffer
                        vec_t int2_value;
                        auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
                        #pragma unroll
                        for (int j = 0; j < kNumElemsPerRead; j += 2) {
                            float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                            fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                        }
                        rdma_x_vec[i] = int2_value;
                    } else {
                        // Reinterpret-cast is for C++14 compatibility
                        rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                    }
                }
            asm volatile("bar.sync 1, %0;" :: "r"(num_threads));

            // Issue IBGDA sends
            if (dst_expert_idx >= 0) {
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
                const auto dst_rank = dst_expert_idx / num_local_experts;
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_data_buffer) +
                                     dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     slot_idx * num_bytes_per_msg;
                if (dst_rank != rank) {
                    bool use_nvlink = nvlink_available[dst_rank] != 0;
                    if (use_nvlink) {
                        size_t offset = (char *)dst_ptr - (char *)(mxa_buffer);
                        void* peer_dst_ptr = (char *)ipc_peer_ptrs[dst_rank] + offset;
                        // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                        const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                        const auto* dst_int4_ptr = reinterpret_cast<int4*>(peer_dst_ptr);
                        UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                    } else {
                        if (lane_id == 0) {
                            uint64_t req_rptr_actual = raddr_array[dst_rank] + ((char *)dst_ptr - (char *)(mxa_buffer));
                            auto ctx = ctx_array + dst_rank * num_qp_per_rank + dst_expert_local_idx % num_qp_per_rank;
                            device_mutex_lock_system(&ctx->mutex);
                            __mlx5gda_device_write_rdma_write_wqe(ctx, src_ptr, device_byteswap(rkey_array[rank]), req_rptr_actual, device_byteswap(rkey_array[dst_rank]), num_bytes_per_msg);
                            __mlx5gda_device_post_send_db(ctx);
                            device_mutex_unlock_system(&ctx->mutex);
                        }
                    }
                } else {
                    // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                    const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                    const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
                    UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                }

                // Increase counter after finishing
                __syncwarp();
                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
            }
        }
    } else if (warp_id == num_warps - 1) {
        EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            // The first SM is also responsible for cleaning the next buffer
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += 32)
                next_clean_buffer[i] = 0;

            // Notify before executing `int_p`
            __syncwarp();
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += 32)
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
        }

        // This SM should be responsible for some destination experts, read `topk_idx` for them
        int expert_count[kNumWarpGroups] = {0};
        const auto expert_begin_idx = sm_id * kNumWarpGroups;
        const auto expert_end_idx = min(expert_begin_idx + kNumWarpGroups, num_experts);

        // Per lane count
        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx] ++;
        }

        // Warp reduce
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++ i) {
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
            }
        }
    }
    __syncthreads();

    // Issue count sends
    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * kNumWarpGroups];

        // Wait local sends issued and send expert counts
        while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != FINISHED_SUM_TAG * 2);
        if (dst_rank != rank) {
            bool use_nvlink = nvlink_available[dst_rank] != 0;
            if (use_nvlink) {
                int* signal_ptr = rdma_recv_signal_buffer + dst_expert_local_idx * num_ranks + rank;
                size_t offset = (char *)signal_ptr - (char *)(mxa_buffer);
                int* peer_signal_ptr = (int *)((char *)ipc_peer_ptrs[dst_rank] + offset);
                st_na_release(peer_signal_ptr, -num_tokens_sent - 1);
            } else {
                uint64_t laddr = (uint64_t)((char *)(raddr_array[rank]) + ((char *)(rdma_send_signal_buffer + dst_expert_local_idx * num_ranks + rank) - (char *)(mxa_buffer)));
                uint64_t rptr_actual = (uint64_t)((char *)(raddr_array[dst_rank]) + ((char *)(rdma_recv_signal_buffer + dst_expert_local_idx * num_ranks + rank) - (char *)(mxa_buffer)));
                auto ctx = ctx_array + dst_rank * num_qp_per_rank + dst_expert_local_idx % num_qp_per_rank;
                device_mutex_lock_system(&ctx->mutex);
                __mlx5gda_device_write_rdma_atomic_add_wqe(ctx, -num_tokens_sent - 1, laddr, device_byteswap(rkey_array[rank]), rptr_actual, device_byteswap(rkey_array[dst_rank]));
                __mlx5gda_device_post_send_db(ctx);
                device_mutex_unlock_system(&ctx->mutex);
            }
        } else {
            st_na_release(rdma_recv_signal_buffer + dst_expert_local_idx * num_ranks + rank, -num_tokens_sent - 1);
        }

        // Clean workspace for next use
        atomic_counter_per_expert[responsible_expert_idx] = 0;
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

        // Clean `packed_recv_count`
        if (dst_rank == 0)
            packed_recv_count[dst_expert_local_idx] = 0;
    }
    __syncwarp();

    // Receiving phase
    LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    if (phases & LOW_LATENCY_SEND_PHASE)
        cooperative_groups::this_grid().sync();

    // Receiving and packing
    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto rdma_recv_x_uint8 = reinterpret_cast<uint8_t*>(rdma_recv_data_buffer) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        const auto recv_x_int4 = reinterpret_cast<int4*>(packed_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        const auto recv_x_scales = packed_recv_x_scales + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_scales;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kNumWarpGroups], shared_recv_token_begin_idx[kNumWarpGroups];

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens, recv_token_begin_idx;
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
        if (sub_warp_id == 1 and lane_id == 0) {
            unsigned long long start_time = clock64();
            while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_signal_buffer + local_expert_idx * num_ranks + src_rank)) == 0) {
                unsigned long long end_time = clock64();
                if (timeout_ticks != -1 && end_time - start_time > timeout_ticks) {
                    active_ranks[src_rank] = 0;
                }
                if (!active_ranks[src_rank]) {
                    num_recv_tokens = -1;
                    break;
                }
            }
            num_recv_tokens = -num_recv_tokens - 1;
            recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 2), "r"(kNumWarpsPerGroup * 32));
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Copy tokens
        EP_DEVICE_ASSERT(num_scales <= 64);
        for (int i = sub_warp_id; i < num_recv_tokens; i += kNumWarpsPerGroup) {
            // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            if (lane_id == 0)
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            __syncwarp();

            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);

            // Copy scales
            if (kUseFP8) {
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                const auto dst_scales = reinterpret_cast<float*>(recv_x_scales + recv_token_begin_idx + i);
                const auto scale_stride = num_ranks * num_max_dispatch_tokens_per_rank;
                auto scale_0 = lane_id < num_scales ? ld_nc_global(src_scales + lane_id) : 0;
                auto scale_1 = (lane_id + 32) < num_scales ? ld_nc_global(src_scales + lane_id + 32) : 0;
                lane_id < num_scales ? dst_scales[lane_id * scale_stride] = scale_0 : 0.0f;
                (lane_id + 32) < num_scales ? dst_scales[(lane_id + 32) * scale_stride] = scale_1 : 0.0f;
            }
        }
    }
}

void dispatch(void* packed_recv_x, float* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count, int32_t* active_ranks,
              void* mxa_buffer,
              int* rdma_send_signal_buffer, int* rdma_recv_signal_buffer,
              void* rdma_send_data_buffer, void* rdma_recv_data_buffer,
              void* cuda_counter_buffer, void* cuda_data_buffer,
              void* raddrs, void* rkeys, void* qp_devctxs,
              const int32_t* nvlink_available, void* const* ipc_peer_ptrs,
              const void* x, const int64_t* topk_idx,
              int* next_clean_buffer,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks, bool use_fp8,
              void* workspace, cudaStream_t stream, int64_t timeout_ticks, int phases) {
    constexpr int kNumMaxTopK = 11;
    constexpr int kNumWarpsPerGroup = 4;
    constexpr int kNumWarpGroups = 8;
    EP_STATIC_ASSERT(kNumMaxTopK + 1 <= kNumWarpGroups * kNumWarpsPerGroup, "Too many top-k selections");

    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_experts, kNumWarpGroups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

#define DISPATCH_LAUNCH_CASE(hidden) { \
auto dispatch_func = use_fp8 ? dispatch<true, kNumWarpGroups, kNumWarpsPerGroup, hidden> : \
                               dispatch<false, kNumWarpGroups, kNumWarpsPerGroup, hidden>; \
LAUNCH_KERNEL(&cfg, dispatch_func, \
              packed_recv_x, packed_recv_x_scales, \
              packed_recv_src_info, packed_recv_layout_range, \
              packed_recv_count, active_ranks, \
              mxa_buffer, \
              rdma_send_signal_buffer, rdma_recv_signal_buffer, \
              rdma_send_data_buffer, rdma_recv_data_buffer, \
              cuda_counter_buffer, cuda_data_buffer, \
              raddrs, rkeys, qp_devctxs, \
              nvlink_available, ipc_peer_ptrs, \
              x, topk_idx, \
              atomic_counter_per_expert, atomic_finish_counter_per_expert, \
              next_clean_buffer, \
              num_tokens, num_max_dispatch_tokens_per_rank, \
              num_topk, num_experts, rank, num_ranks, timeout_ticks, phases); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumRecvUnrolls>
__forceinline__ __device__ void decode_and_accumulate_bf16(
    const uint32_t* ld_buffer, float* accum, const float weight) {
    #pragma unroll
    for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
        auto bf16_pack = *reinterpret_cast<const __nv_bfloat162*>(ld_buffer + k);
        accum[k * 2 + 0] += __bfloat162float(bf16_pack.x) * weight;
        accum[k * 2 + 1] += __bfloat162float(bf16_pack.y) * weight;
    }
}

template <int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * 32, 1) void
combine(void* combined_x, int32_t* active_ranks,
        void* mxa_buffer,
        int* rdma_send_signal_buffer, int* rdma_recv_signal_buffer,
        void* rdma_send_data_buffer, void* rdma_recv_data_buffer,
        void* cuda_counter_buffer, void* cuda_data_buffer,
        void* raddrs, void* rkeys, void* qp_devctxs,
        const int32_t* nvlink_available, void* const* ipc_peer_ptrs,
        const void* x, const int64_t* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int* next_clean_buffer,
        int* atomic_clean_flag,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int64_t timeout_ticks,
        int phases, bool zero_copy) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

#if MOONCAKE_EP_ENABLE_SM90_DEVICE_FEATURES
    // Use different unroll factors for send and recv phases
    constexpr int kNumMaxUnrolls = 4;
    constexpr int kNumSendUnrolls = kHidden % (32 * 4 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 ? 4 : 2;
    constexpr int kNumRecvUnrolls = 2;
    constexpr int hidden_bf16_int4_pad = align_up(static_cast<int>(hidden_bf16_int4), 32 * kNumSendUnrolls);
    EP_STATIC_ASSERT(kHidden % (32 * 2 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hidden_bf16_int4 % kNumSendUnrolls == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");
#endif

    // Message package
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // IBGDA
    auto raddr_array = reinterpret_cast<uint64_t*>(raddrs);
    auto rkey_array = reinterpret_cast<uint32_t*>(rkeys);
    auto ctx_array = reinterpret_cast<mlx5gda_qp_devctx*>(qp_devctxs);
    const size_t num_qp_per_rank = MAX_QP_COUNT / num_ranks;

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_experts; i += 32)
            next_clean_buffer[i] = 0;

        // Notify before executing `int_p`
        __syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x = reinterpret_cast<const int4*>(x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec = reinterpret_cast<uint8_t*>(rdma_send_data_buffer) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        // Unpack layout
        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

#if MOONCAKE_EP_ENABLE_SM90_DEVICE_FEATURES
        // TMA stuffs
        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;
        constexpr int kNumStages = 3;
        constexpr int kNumPrefetch = 1;
        EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

        auto smem_ptr = smem_buffer + warp_id * (kNumStages * (kNumTMABufferBytes + 16));
        uint32_t tma_phase = 0;
        auto tma_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<int4*>(smem_ptr + i * (kNumTMABufferBytes + 16)); });
        auto full_barriers = PatternVisitor(
            [=](const int& i) { return reinterpret_cast<uint64_t*>(smem_ptr + i * (kNumTMABufferBytes + 16) + kNumTMABufferBytes); });
        EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");

        // Initialize m-barriers
        if (lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            fence_barrier_init();
        }
        __syncwarp();

        constexpr int kNumIters = hidden_bf16_int4_pad / (32 * kNumSendUnrolls);
        auto tma_load_and_arrive = [&](const int& stage_idx, const int4* gmem_ptr, const int& num_bytes) {
            tma_load_1d(tma_buffers[stage_idx], gmem_ptr, full_barriers[stage_idx], num_bytes);
            mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_bytes);
        };
        auto get_num_tma_bytes = [&](const int& offset_int4) {
            return min(kNumTMABufferBytes, static_cast<int>((hidden_bf16_int4 - offset_int4) * sizeof(int4)));
        };
#endif

        // Issue IBGDA send
        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += kNumWarpsPerGroup) {
            const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

            // Copy directly to local rank, or copy to buffer and issue RDMA
            auto src_idx = __ldg(local_src_info + token_idx);
            const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_data_buffer) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
            bool use_nvlink = nvlink_available[dst_rank] != 0;
#if MOONCAKE_EP_ENABLE_SM90_DEVICE_FEATURES
            auto dst_p2p_ptr = dst_rank == rank
                                   ? reinterpret_cast<int4*>(dst_ptr)
                                   : (use_nvlink
                                          ? reinterpret_cast<int4*>(
                                                static_cast<char*>(ipc_peer_ptrs[dst_rank]) +
                                                (reinterpret_cast<char*>(dst_ptr) - reinterpret_cast<char*>(mxa_buffer)))
                                          : nullptr);

            if (not zero_copy or dst_p2p_ptr != nullptr) {
                const auto cpy_src_int4_ptr = zero_copy ? reinterpret_cast<int4*>(buf_ptr) : x_int4;
                const auto cpy_dst_int4_ptr =
                    dst_p2p_ptr == nullptr ? reinterpret_cast<int4*>(buf_ptr) : dst_p2p_ptr;

                if (elect_one_sync())
                    tma_load_and_arrive(0, cpy_src_int4_ptr, get_num_tma_bytes(0));
                __syncwarp();

                #pragma unroll
                for (int i = lane_id * kNumSendUnrolls, iter_idx = 0; i < hidden_bf16_int4_pad; i += 32 * kNumSendUnrolls, ++iter_idx) {
                    const int& stage_idx = iter_idx % kNumStages;
                    const int& next_stage_idx = (iter_idx + 1) % kNumStages;
                    if (iter_idx + 1 < kNumIters and elect_one_sync()) {
                        tma_store_wait<kNumStages - kNumPrefetch - 1>();
                        const auto& offset_int4 = i + 32 * kNumSendUnrolls;
                        tma_load_and_arrive(next_stage_idx, cpy_src_int4_ptr + offset_int4, get_num_tma_bytes(offset_int4));
                    }
                    __syncwarp();

                    EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
                    mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                    if (elect_one_sync())
                        tma_store_1d(tma_buffers[stage_idx], cpy_dst_int4_ptr + i, get_num_tma_bytes(i));
                    __syncwarp();
                }

                tma_store_wait<0>();
                __syncwarp();
            }

            if (dst_p2p_ptr == nullptr && lane_id == 0) {
                uint64_t req_rptr_actual = raddr_array[dst_rank] + (reinterpret_cast<char*>(dst_ptr) - reinterpret_cast<char*>(mxa_buffer));
                auto ctx = ctx_array + dst_rank * num_qp_per_rank + local_expert_idx % num_qp_per_rank;
                device_mutex_lock_system(&ctx->mutex);
                __mlx5gda_device_write_rdma_write_wqe(ctx, static_cast<uint64_t>(buf_ptr), device_byteswap(rkey_array[rank]), req_rptr_actual, device_byteswap(rkey_array[dst_rank]), num_bytes_per_slot);
                __mlx5gda_device_post_send_db(ctx);
                device_mutex_unlock_system(&ctx->mutex);
            }
#else
            if (dst_rank == rank) {
                const auto dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
                UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, dst_int4_ptr, x_int4, ld_nc_global, st_na_global);
            } else if (use_nvlink) {
                size_t offset = reinterpret_cast<char*>(dst_ptr) - reinterpret_cast<char*>(mxa_buffer);
                void* peer_dst_ptr = static_cast<char*>(ipc_peer_ptrs[dst_rank]) + offset;
                const auto dst_int4_ptr = reinterpret_cast<int4*>(peer_dst_ptr);
                UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, dst_int4_ptr, x_int4, ld_nc_global, st_na_global);
            } else {
                const auto buf_int4_ptr = reinterpret_cast<int4*>(buf_ptr);
                if (not zero_copy)
                    UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, buf_int4_ptr, x_int4, ld_nc_global, st_na_global);
                __syncwarp();

                if (lane_id == 0) {
                    uint64_t req_rptr_actual = raddr_array[dst_rank] + (reinterpret_cast<char*>(dst_ptr) - reinterpret_cast<char*>(mxa_buffer));
                    auto ctx = ctx_array + dst_rank * num_qp_per_rank + local_expert_idx % num_qp_per_rank;
                    device_mutex_lock_system(&ctx->mutex);
                    __mlx5gda_device_write_rdma_write_wqe(ctx, static_cast<uint64_t>(buf_ptr), device_byteswap(rkey_array[rank]), req_rptr_actual, device_byteswap(rkey_array[dst_rank]), num_bytes_per_slot);
                    __mlx5gda_device_post_send_db(ctx);
                    device_mutex_unlock_system(&ctx->mutex);
                }
            }
#endif
        }

        // Put finishing flag
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(kNumWarpsPerGroup * 32));
        if (sub_warp_id == 1 and lane_id == 0) {
            while (ld_acquire_global(atomic_clean_flag) == 0);
            if (dst_rank != rank) {
                bool use_nvlink = nvlink_available[dst_rank] != 0;
                if (use_nvlink) {
                    int* signal_ptr = rdma_recv_signal_buffer + global_expert_idx;
                    size_t offset = (char *)signal_ptr - (char *)(mxa_buffer);
                    int* peer_signal_ptr = (int *)((char *)ipc_peer_ptrs[dst_rank] + offset);
                    st_na_release(peer_signal_ptr, 1);
                } else {
                    uint64_t laddr = (uint64_t)((char *)(raddr_array[rank]) + ((char *)(rdma_send_signal_buffer + global_expert_idx) - (char *)(mxa_buffer)));
                    uint64_t req_rptr_actual = (uint64_t)((char *)(raddr_array[dst_rank]) + ((char *)(rdma_recv_signal_buffer + global_expert_idx) - (char *)(mxa_buffer)));
                    auto ctx = ctx_array + dst_rank * num_qp_per_rank + local_expert_idx % num_qp_per_rank;
                    device_mutex_lock_system(&ctx->mutex);
                    __mlx5gda_device_write_rdma_atomic_add_wqe(ctx, 1, laddr, device_byteswap(rkey_array[rank]), req_rptr_actual, device_byteswap(rkey_array[dst_rank]));
                    __mlx5gda_device_post_send_db(ctx);
                    device_mutex_unlock_system(&ctx->mutex);
                }
            } else {
                st_na_release(rdma_recv_signal_buffer + global_expert_idx, 1);
            }
            atomic_add_release_global(atomic_clean_flag, -1);
        }
        __syncwarp();

#if MOONCAKE_EP_ENABLE_SM90_DEVICE_FEATURES
        // Destroy m-barriers
        if (lane_id < kNumStages) {
            mbarrier_inval(full_barriers[lane_id]);
            fence_barrier_init();
        }
        __syncwarp();
#endif
    }

    // Receiving phase
    LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive
    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Invalid number of warps per group");
        if (sub_warp_id == 0 and lane_id == 0) {
            unsigned long long start_time = clock64();
            while (ld_acquire_sys_global(rdma_recv_signal_buffer + responsible_expert_idx) == 0) {
                unsigned long long end_time = clock64();
                if (timeout_ticks != -1 && end_time - start_time > timeout_ticks) {
                    active_ranks[src_rank] = 0;
                }
                if (!active_ranks[src_rank]) {
                    break;
                }
            }
        }
    }
    cooperative_groups::this_grid().sync();

#if MOONCAKE_EP_ENABLE_SM90_DEVICE_FEATURES
    constexpr int kMaxNumGroups = 2;
    const int num_decode_warps = hidden_bf16_int4_pad / (kNumRecvUnrolls * 32);
    const int num_groups = min(kMaxNumGroups, (num_threads / 32) / (num_decode_warps + 1));
    const int decode_warp_idx = __shfl_sync(0xffffffff, warp_id % (num_decode_warps + 1), 0);
    const int group_idx = __shfl_sync(0xffffffff, warp_id / (num_decode_warps + 1), 0);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(num_groups > 0);

    if (group_idx < num_groups) {
        constexpr int kNumStages = 3;
        constexpr int kNumTMABufferBytes = 16 * 2 + kHidden * 2;
        constexpr int kNumBF16PerWarpBytes = 32 * kNumRecvUnrolls * kNumElemsPerInt4 * sizeof(nv_bfloat16);
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kHidden * sizeof(nv_bfloat16);

        const auto smem_group_buffer = smem_buffer + kNumBytesPerGroup * group_idx;
        auto full_barriers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes); });
        auto empty_barriers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes + 8);
        });
        auto tma_ld_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint8_t*>(smem_group_buffer + i * kNumTMABufferBytes + 16);
        });
        auto tma_st_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint32_t*>(smem_group_buffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes);
        });
        uint32_t tma_phase = 0;
        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
        if (decode_warp_idx == num_decode_warps)
            tma_phase = (1u << kNumStages) - 1u;

        if (decode_warp_idx == num_decode_warps && lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            mbarrier_init(empty_barriers[lane_id], num_decode_warps);
        }
        if (decode_warp_idx == num_decode_warps && lane_id == 0)
            fence_barrier_init();
        asm volatile("bar.sync %0, %1;" ::"r"(group_idx + 1), "r"((num_decode_warps + 1) * 32));

        int stage_idx = 0, topk_idx_by_lane = 0;
        if (decode_warp_idx == num_decode_warps) {
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk)
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                for (int i = 0; i < num_topk; ++i) {
                    const int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (topk_idx_reg < 0)
                        continue;
                    if (!active_ranks[topk_idx_reg / num_local_experts])
                        continue;

                    mbarrier_wait<true>(empty_barriers[stage_idx], tma_phase, stage_idx);
                    auto buffer = reinterpret_cast<uint8_t*>(rdma_recv_data_buffer) +
                        (topk_idx_reg * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot;
                    if (elect_one_sync()) {
                        tma_load_1d(tma_ld_buffers[stage_idx], buffer, full_barriers[stage_idx], static_cast<int>(num_bytes_per_slot));
                        mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], static_cast<int>(num_bytes_per_slot));
                    }
                    __syncwarp();
                    stage_idx = (stage_idx + 1) % kNumStages;
                }
            }
        } else {
            float topk_weights_by_lane = 0.0f;
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk) {
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                    topk_weights_by_lane = __ldg(topk_weights + token_idx * num_topk + lane_id);
                }
                __syncwarp();

                float combined_values[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
                for (int i = 0; i < num_topk; ++i) {
                    const int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (topk_idx_reg < 0)
                        continue;
                    if (!active_ranks[topk_idx_reg / num_local_experts])
                        continue;
                    const auto topk_weight = __shfl_sync(0xffffffff, topk_weights_by_lane, i);

                    mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                    const int tma_offset = kNumBF16PerWarpBytes * decode_warp_idx;
                    decode_and_accumulate_bf16<kNumRecvUnrolls>(
                        reinterpret_cast<const uint32_t*>(
                            tma_ld_buffers[stage_idx] + tma_offset + kNumBF16PerWarpBytes / 32 * lane_id),
                        combined_values,
                        topk_weight);

                    if (elect_one_sync())
                        mbarrier_arrive(empty_barriers[stage_idx]);
                    stage_idx = (stage_idx + 1) % kNumStages;
                }

                tma_store_wait<0>();

                #pragma unroll
                for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
                    auto combined_pack = __nv_bfloat162(__float2bfloat16(combined_values[k * 2]),
                                                        __float2bfloat16(combined_values[k * 2 + 1]));
                    tma_st_buffers[decode_warp_idx][kNumRecvUnrolls * 4 * lane_id + k] =
                        *reinterpret_cast<uint32_t*>(&combined_pack);
                }
                tma_store_fence();
                if (elect_one_sync()) {
                    tma_store_1d(tma_st_buffers[decode_warp_idx],
                                 reinterpret_cast<int4*>(combined_x) +
                                     token_idx * hidden_bf16_int4 + decode_warp_idx * kNumRecvUnrolls * 32,
                                 kNumBF16PerWarpBytes);
                }
                __syncwarp();
            }
            tma_store_wait<0>();
        }
    }
#else
    // Reduce tokens with direct global-memory rereads on pre-Hopper targets.
    EP_DEVICE_ASSERT(num_topk <= 32 and hidden_bf16_int4 <= num_threads);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    if (thread_id < hidden_bf16_int4) {
        for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
            int reg_topk_idx[kNumMaxTopk];
            float reg_topk_weights[kNumMaxTopk];
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) {
                reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
                reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
            }

            float combined_values[kNumElemsPerInt4] = {0.0f};
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) if (reg_topk_idx[i] >= 0) {
                auto rdma_buffer_type = reinterpret_cast<const int*>(reinterpret_cast<uint8_t*>(rdma_recv_data_buffer) + (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot);
                auto rdma_buffer_row = reinterpret_cast<const uint8_t*>(rdma_buffer_type);

                auto x_vec = ld_nc_global(reinterpret_cast<const int4*>(rdma_buffer_row) + thread_id);
                const auto x_bf16 = reinterpret_cast<nv_bfloat16*>(&x_vec);
                #pragma unroll
                for (int j = 0; j < kNumElemsPerInt4; ++ j)
                    combined_values[j] += __bfloat162float(x_bf16[j]) * reg_topk_weights[i];
            }

            int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
            auto combined_bf16 = reinterpret_cast<nv_bfloat16*>(&combined_values);
            #pragma unroll
            for (int j = 0; j < kNumElemsPerInt4; ++ j)
                combined_bf16[j] = __float2bfloat16(combined_values[j]);
            (reinterpret_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4)[thread_id] = combined_int4;
        }
    }
#endif
}

void combine(void* combined_x, int32_t* active_ranks,
             void* mxa_buffer,
             int* rdma_send_signal_buffer, int* rdma_recv_signal_buffer,
             void* rdma_send_data_buffer, void* rdma_recv_data_buffer,
             void* cuda_counter_buffer, void* cuda_data_buffer,
             void* raddrs, void* rkeys, void* qp_devctxs,
             const int32_t* nvlink_available, void* const* ipc_peer_ptrs,
             const void* x, const int64_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int* next_clean_buffer,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             void* workspace, cudaStream_t stream,
             int64_t timeout_ticks, int phases, bool zero_copy) {
    constexpr int kNumWarpsPerGroup = 4;
    constexpr int kNumWarpGroups = 8;
    constexpr int kNumMaxTopk = 11;

    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_experts, kNumWarpGroups);

    // Check workspace
    auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

#if MOONCAKE_EP_ENABLE_SM90_HOST_FEATURES
    constexpr int kNumStages = 3;
    constexpr int kNumMaxUnrolls = 4;
    constexpr int kMaxNumGroups = 2;

    // Send buffer size
    const int num_send_tma_bytes = 32 * sizeof(int4) * kNumMaxUnrolls + 16;
    const int smem_send_size = num_warps * (kNumStages * num_send_tma_bytes);

    // Receive buffer size
    const int num_recv_tma_bytes = 16 * 2 + hidden * 2;
    const int smem_recv_size = kMaxNumGroups * (kNumStages * num_recv_tma_bytes + hidden * 2);

    // Total requirement
    const int smem_size = max(smem_send_size, smem_recv_size);
#endif

#define COMBINE_LAUNCH_CASE(hidden) { \
auto combine_func = combine<kNumWarpGroups, kNumWarpsPerGroup, hidden, kNumMaxTopk>; \
SET_SHARED_MEMORY_FOR_TMA(combine_func); \
LAUNCH_KERNEL(&cfg, combine_func, \
              combined_x, active_ranks, \
              mxa_buffer, \
              rdma_send_signal_buffer, rdma_recv_signal_buffer, \
              rdma_send_data_buffer, rdma_recv_data_buffer, \
              cuda_counter_buffer, cuda_data_buffer, \
              raddrs, rkeys, qp_devctxs, \
              nvlink_available, ipc_peer_ptrs, \
              x, topk_idx, topk_weights, src_info, layout_range, \
              next_clean_buffer, \
              atomic_clean_flag, \
              num_combined_tokens, hidden, num_topk, \
              num_max_dispatch_tokens_per_rank, \
              num_experts, rank, num_ranks, \
              timeout_ticks, phases, zero_copy); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace mooncake
