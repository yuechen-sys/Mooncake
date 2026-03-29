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

template <bool kUseFP8, int kHidden>
__global__ __launch_bounds__(1024, 1) void
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
         int num_warp_groups, int num_warps_per_group,
         int64_t timeout_ticks,
         int phases) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
    constexpr int kNumMaxWarpGroups = 32;

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
    __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

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
        int expert_count[kNumMaxWarpGroups] = {0};
        const auto expert_begin_idx = sm_id * num_warp_groups;
        const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);

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
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * num_warp_groups];

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
        __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups], shared_recv_token_begin_idx[kNumMaxWarpGroups];

        // Wait tokens to arrive. Sub-warp 1 owns the signal wait and range
        // publication, but the later group barrier means the copy loop still
        // starts only after the wait completes for the whole warp group.
        int num_recv_tokens, recv_token_begin_idx;
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
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
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 2), "r"(num_warps_per_group * 32));
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Copy tokens
        EP_DEVICE_ASSERT(num_scales <= 64);
        for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
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
              void* workspace, int num_device_sms, cudaStream_t stream,
              int64_t timeout_ticks, int phases) {
    constexpr int kNumMaxTopK = 11;
    // `bar.sync` uses barrier IDs in `[0, 15]`; dispatch recv uses
    // `warp_group_id + 2`, so keep the runtime group count within that range.
    const int num_warp_groups = min(cell_div(num_experts, num_device_sms), 14);
    const int num_warps_per_group = 32 / num_warp_groups;
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = cell_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

#define DISPATCH_LAUNCH_CASE(hidden) { \
auto dispatch_func = use_fp8 ? dispatch<true, hidden> : dispatch<false, hidden>; \
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
              num_topk, num_experts, rank, num_ranks, \
              num_warp_groups, num_warps_per_group, timeout_ticks, phases); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(1024, 1) void
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
        int num_warp_groups, int num_warps_per_group,
        int64_t timeout_ticks,
        int phases, bool zero_copy) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

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

        // Issue IBGDA send
        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
            const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

            // Copy directly to local rank, or copy to buffer and issue RDMA
            auto src_idx = __ldg(local_src_info + token_idx);
            const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_data_buffer) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
            if (dst_rank == rank) {
                const auto dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
                UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, dst_int4_ptr, x_int4, ld_nc_global, st_na_global);
            } else {
                bool use_nvlink = nvlink_available[dst_rank] != 0;
                if (use_nvlink) {
                    size_t offset = (char *)dst_ptr - (char *)(mxa_buffer);
                    void* peer_dst_ptr = (char *)ipc_peer_ptrs[dst_rank] + offset;
                    const auto dst_int4_ptr = reinterpret_cast<int4*>(peer_dst_ptr);
                    UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, dst_int4_ptr, x_int4, ld_nc_global, st_na_global);
                } else {
                    const auto buf_int4_ptr = reinterpret_cast<int4*>(buf_ptr);
                    if (not zero_copy)
                        UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, buf_int4_ptr, x_int4, ld_nc_global, st_na_global);
                    __syncwarp();

                    if (lane_id == 0) {
                        uint64_t req_rptr_actual = raddr_array[dst_rank] + ((char *)dst_ptr - (char *)(mxa_buffer));
                        auto ctx = ctx_array + dst_rank * num_qp_per_rank + local_expert_idx % num_qp_per_rank;
                        device_mutex_lock_system(&ctx->mutex);
                        __mlx5gda_device_write_rdma_write_wqe(ctx, (uint64_t) buf_ptr, device_byteswap(rkey_array[rank]), req_rptr_actual, device_byteswap(rkey_array[dst_rank]), num_bytes_per_slot);
                        __mlx5gda_device_post_send_db(ctx);
                        device_mutex_unlock_system(&ctx->mutex);
                    }
                }
            }
        }

        // Put finishing flag
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
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
    }

    // Receiving phase
    LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive
    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
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

    // Rebuild the recv side into a small multi-stage TMA ring with per-group
    // full/empty mbarriers, while keeping the direct output store.
    constexpr int kMaxRecvGroups = 2;
    constexpr int kNumStages = 3;
    constexpr int kNumRecvUnrolls = 2;
    constexpr size_t kNumTMABufferBytes = 16 + num_bytes_per_slot;
    constexpr size_t kNumBytesPerGroup = kNumStages * kNumTMABufferBytes;
    extern __shared__ __align__(16) uint8_t recv_smem_buffer[];
    const auto num_warps = num_threads / 32;
    const int num_recv_groups = min(kMaxRecvGroups, max(1, num_warps / 2));
    const int recv_warps_per_group = num_warps / num_recv_groups;
    const int num_decode_warps = recv_warps_per_group - 1;
    const int recv_group_idx = warp_id / recv_warps_per_group;
    const int recv_group_warp_idx = warp_id % recv_warps_per_group;
    const bool recv_group_active = recv_group_idx < num_recv_groups;

    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(num_recv_groups > 0 and recv_warps_per_group > 1 and
                     num_decode_warps > 0);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0,
                     "Invalid vectorization");
    const int warp_base_int4 =
        (recv_group_warp_idx - 1) * kNumRecvUnrolls * 32;
    const int max_group_iters =
        sm_id >= num_combined_tokens
            ? 0
            : cell_div(num_combined_tokens - sm_id,
                       num_sms * num_recv_groups);

    uint8_t* smem_group_buffer = recv_smem_buffer;
    uint64_t* full_barriers[kNumStages];
    uint64_t* empty_barriers[kNumStages];
    uint8_t* tma_ld_buffers[kNumStages];
    if (recv_group_active) {
        smem_group_buffer += recv_group_idx * kNumBytesPerGroup;
        #pragma unroll
        for (int stage = 0; stage < kNumStages; ++stage) {
            auto* stage_buffer = smem_group_buffer + stage * kNumTMABufferBytes;
            full_barriers[stage] =
                reinterpret_cast<uint64_t*>(stage_buffer);
            empty_barriers[stage] =
                reinterpret_cast<uint64_t*>(stage_buffer + 8);
            tma_ld_buffers[stage] = stage_buffer + 16;
        }
    }

    if (recv_group_active && recv_group_warp_idx == 0 && lane_id == 0) {
        #pragma unroll
        for (int stage = 0; stage < kNumStages; ++stage) {
            mbarrier_init(full_barriers[stage], 1);
            mbarrier_init(empty_barriers[stage], num_decode_warps);
        }
        fence_barrier_init();
    }
    __syncthreads();

    int stage_idx = 0;
    uint32_t empty_phase[kNumStages] = {1, 1, 1};
    uint32_t full_phase[kNumStages] = {0, 0, 0};

    for (int iter_idx = 0; iter_idx < max_group_iters; ++iter_idx) {
        const int token_idx =
            sm_id + num_sms * recv_group_idx +
            iter_idx * num_sms * num_recv_groups;
        const bool has_token =
            recv_group_active && token_idx < num_combined_tokens;

        if (!recv_group_active)
            continue;

        int topk_idx_by_lane = -1;
        float topk_weight_by_lane = 0.0f;
        if (lane_id < num_topk && has_token) {
            topk_idx_by_lane =
                static_cast<int>(__ldg(topk_idx + token_idx * num_topk +
                                       lane_id));
            if (recv_group_warp_idx != 0)
                topk_weight_by_lane = __ldg(topk_weights + token_idx * num_topk +
                                            lane_id);
        }
        __syncwarp();

        if (recv_group_warp_idx == 0) {
            #pragma unroll
            for (int i = 0; i < kNumMaxTopk; ++i) {
                const bool topk_in_range = i < num_topk;
                const int topk_idx_reg =
                    __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                const bool valid_topk =
                    has_token && topk_in_range && topk_idx_reg >= 0 &&
                    active_ranks[topk_idx_reg / num_local_experts] != 0;
                if (!valid_topk)
                    continue;

                mbarrier_wait(empty_barriers[stage_idx], empty_phase[stage_idx]);
                if (lane_id == 0) {
                    auto buffer =
                        reinterpret_cast<const uint8_t*>(rdma_recv_data_buffer) +
                        (topk_idx_reg * num_max_dispatch_tokens_per_rank +
                         token_idx) * num_bytes_per_slot;
                    tma_load_1d(tma_ld_buffers[stage_idx], buffer,
                                full_barriers[stage_idx],
                                static_cast<int>(num_bytes_per_slot));
                    mbarrier_arrive_and_expect_tx(
                        full_barriers[stage_idx],
                        static_cast<int>(num_bytes_per_slot));
                }
                __syncwarp();
                stage_idx = (stage_idx + 1) % kNumStages;
            }
        } else {
            float combined_values[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
            #pragma unroll
            for (int i = 0; i < kNumMaxTopk; ++i) {
                const bool topk_in_range = i < num_topk;
                const int topk_idx_reg =
                    __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                const bool valid_topk =
                    has_token && topk_in_range && topk_idx_reg >= 0 &&
                    active_ranks[topk_idx_reg / num_local_experts] != 0;
                if (!valid_topk)
                    continue;

                const float topk_weight =
                    __shfl_sync(0xffffffff, topk_weight_by_lane, i);
                mbarrier_wait(full_barriers[stage_idx], full_phase[stage_idx]);
                const auto tma_stage_ptr =
                    reinterpret_cast<const int4*>(tma_ld_buffers[stage_idx]) +
                    warp_base_int4;
                #pragma unroll
                for (int unroll_idx = 0; unroll_idx < kNumRecvUnrolls;
                     ++unroll_idx) {
                    const int hidden_idx =
                        warp_base_int4 + lane_id + unroll_idx * 32;
                    if (hidden_idx < hidden_bf16_int4) {
                        const auto x_vec =
                            tma_stage_ptr[lane_id + unroll_idx * 32];
                        const auto x_bf16 =
                            reinterpret_cast<const nv_bfloat16*>(&x_vec);
                        #pragma unroll
                        for (int j = 0; j < kNumElemsPerInt4; ++j)
                            combined_values[unroll_idx * kNumElemsPerInt4 + j] +=
                                __bfloat162float(x_bf16[j]) * topk_weight;
                    }
                }
                if (lane_id == 0)
                    mbarrier_arrive(empty_barriers[stage_idx]);
                stage_idx = (stage_idx + 1) % kNumStages;
            }

            if (has_token) {
                auto combined_row = reinterpret_cast<int4*>(combined_x) +
                    token_idx * hidden_bf16_int4 + warp_base_int4;
                #pragma unroll
                for (int unroll_idx = 0; unroll_idx < kNumRecvUnrolls;
                     ++unroll_idx) {
                    const int hidden_idx =
                        warp_base_int4 + lane_id + unroll_idx * 32;
                    if (hidden_idx < hidden_bf16_int4) {
                        int4 combined_int4;
                        auto combined_bf16 =
                            reinterpret_cast<nv_bfloat16*>(&combined_int4);
                        #pragma unroll
                        for (int j = 0; j < kNumElemsPerInt4; ++j)
                            combined_bf16[j] = __float2bfloat16(
                                combined_values[unroll_idx * kNumElemsPerInt4 + j]);
                        combined_row[lane_id + unroll_idx * 32] = combined_int4;
                    }
                }
            }
        }
    }
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
             void* workspace, int num_device_sms, cudaStream_t stream,
             int64_t timeout_ticks, int phases, bool zero_copy) {
    constexpr int kNumMaxTopk = 11;
    // `bar.sync` uses barrier IDs in `[0, 15]`; combine send uses
    // `warp_group_id + 1`, so keep the runtime group count within that range.
    const int num_warp_groups = min(cell_div(num_experts, num_device_sms), 15);
    const int num_warps_per_group = 32 / num_warp_groups;
    const int num_recv_per_sm = cell_div(num_combined_tokens, num_device_sms);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 1 and
                   num_recv_per_sm >= 0);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms =
        max(cell_div(num_experts, num_warp_groups),
            num_recv_per_sm == 0 ? 1
                                 : cell_div(num_combined_tokens, num_recv_per_sm));

    // Check workspace
    auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

#define COMBINE_LAUNCH_CASE(hidden) { \
auto combine_func = combine<hidden, kNumMaxTopk>; \
constexpr int kNumStages = 3; \
constexpr size_t kHiddenBytes = hidden * sizeof(nv_bfloat16); \
constexpr size_t kNumTMABufferBytes = 16 + kHiddenBytes; \
constexpr size_t kNumBytesPerGroup = kNumStages * kNumTMABufferBytes; \
const int num_recv_groups = min(2, max(1, num_warps / 2)); \
const size_t recv_smem_bytes = \
    (phases & LOW_LATENCY_RECV_PHASE) == 0 ? 0 : num_recv_groups * kNumBytesPerGroup; \
if (recv_smem_bytes > 0) { \
    CUDA_CHECK(cudaFuncSetAttribute( \
        combine_func, cudaFuncAttributeMaxDynamicSharedMemorySize, \
        static_cast<int>(recv_smem_bytes))); \
    CUDA_CHECK(cudaFuncSetAttribute( \
        combine_func, cudaFuncAttributePreferredSharedMemoryCarveout, 100)); \
    cfg.dynamicSmemBytes = recv_smem_bytes; \
} \
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
              num_warp_groups, num_warps_per_group, \
              timeout_ticks, phases, zero_copy); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace mooncake
