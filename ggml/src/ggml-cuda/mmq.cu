#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdint>

static __global__ void mmq_moe_tiles(const int32_t * bounds, int2 * tiles_small, int2 * tiles_large, int32_t * counts,
        const int n_experts, const int tile_small, const int tile_large, const int limit_small) {
    extern __shared__ int offsets[];
    int * small = offsets;
    int * large = offsets + blockDim.x;
    const int e = threadIdx.x;
    const int first = e < n_experts ? bounds[e] : 0;
    const int n = e < n_experts ? bounds[e + 1] - first : 0;
    int ns = n <= limit_small ? (n + tile_small - 1) / tile_small : 0;
    int nl = n >  limit_small ? (n + tile_large - 1) / tile_large : 0;
    const int own_small = ns;
    const int own_large = nl;
    small[e] = ns;
    large[e] = nl;
    __syncthreads();

    for (int step = 1; step < int(blockDim.x); step *= 2) {
        const int ps = e >= step ? small[e - step] : 0;
        const int pl = e >= step ? large[e - step] : 0;
        __syncthreads();
        small[e] = ns += ps;
        large[e] = nl += pl;
        __syncthreads();
    }
    if (e == int(blockDim.x) - 1) {
        counts[0] = ns;
        counts[1] = nl;
    }
    for (int j = 0; j < own_small; ++j) {
        tiles_small[ns - own_small + j] = make_int2(e, first + j*tile_small);
    }
    for (int j = 0; j < own_large; ++j) {
        tiles_large[nl - own_large + j] = make_int2(e, first + j*tile_large);
    }
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// overrides the src1 precision requested by the graph, "auto" keeps the requested one
static ggml_prec ggml_cuda_mmq_get_prec_env() {
    const char * env_c = getenv("GGML_CUDA_MMQ_PREC");
    if (env_c == nullptr) {
        return GGML_PREC_UNDEFINED;
    }
    std::string env_cpp = env_c;
    for (char & c : env_cpp) {
        c = std::tolower(c);
    }
    if (env_cpp == "q4") {
        return GGML_PREC_Q4;
    }
    if (env_cpp == "q8") {
        return GGML_PREC_Q8;
    }
    if (env_cpp != "auto") {
        GGML_LOG_WARN("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", __func__, env_cpp.c_str());
    }
    return GGML_PREC_UNDEFINED;
}

// src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
// default to the native W4A4 instructions on Blackwell.
static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_tensor * src0, const ggml_tensor * dst, const int cc) {
    static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();

    ggml_prec prec = prec_env;
    if (prec == GGML_PREC_UNDEFINED) {
        prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
    }

    // Q4 only for the FP4 types on Blackwell
    GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
    const bool can_use_q4 = (src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
    if (prec == GGML_PREC_Q8 || !can_use_q4) {
        return GGML_PREC_Q8;
    }
    return GGML_PREC_Q4;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_tensor * gate, ggml_tensor * down) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const ggml_prec prec_src1 = gate ? GGML_PREC_Q8 : ggml_cuda_mmq_get_prec_src1(src0, dst, cc);

    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    // MoE kernels read full tiles, including columns past the final expert's tokens.
    const int ncols_padding = ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, 128);
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows + ncols_padding);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
        ncols_padding * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
    // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
    int64_t ncols_opt = ne12;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, ncols_opt};

    static const bool compact_moe = [] {
        const char * value = getenv("GGML_CUDA_MMQ_MOE_COMPACT");
        return value != nullptr && atoi(value) != 0;
    }();
    if (compact_moe && ampere_mma_available(cc) && !fallback &&
            ne02 >= 64 && ne02 <= 1024 && ne12 >= 128 && ne12 <= 16384 && ne03 == 1 &&
            (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_1 || src0->type == GGML_TYPE_Q8_0)) {
        constexpr int tile_small = 16;
        const int tile_large = down ? 32 : gate ? 64 : 128;
        constexpr int limit_small = 16;
        const auto config_small = ggml_cuda_mmq_get_config(src0->type, tile_small, false, cc);
        const auto config_large = ggml_cuda_mmq_get_config(src0->type, tile_large, false, cc);
        const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
        if (config_small.type == GGML_TYPE_COUNT || config_large.type == GGML_TYPE_COUNT ||
                mmq_get_nbytes_shared(config_small, cc) > smpbo || mmq_get_nbytes_shared(config_large, cc) > smpbo) {
            ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
            return;
        }
        // Each small expert contributes at most one tile.
        const int64_t max_small = std::min(ne02, ne_get_rows);
        const int64_t max_large = ne02 + ne_get_rows/tile_large;
        ggml_cuda_pool_alloc<int2> tiles(ctx.pool(), max_small + max_large);
        ggml_cuda_pool_alloc<int32_t> counts(ctx.pool(), 2);
        int nthreads = 1;
        while (nthreads < ne02) {
            nthreads *= 2;
        }
        mmq_moe_tiles<<<1, nthreads, 2*nthreads*sizeof(int), stream>>>(expert_bounds.get(), tiles.get(),
                tiles.get() + max_small, counts.get(), ne02, tile_small, tile_large, limit_small);
        CUDA_CHECK(cudaGetLastError());

        args.expert_tiles = tiles.get();
        args.expert_tile_count = counts.get();
        args.expert_tiles_max = max_small;
        args.ncols_opt = tile_small;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (gate && down) {
            const ggml_tensor * weights = down->src[0];
            const int64_t hidden_padded = GGML_PAD(ne01, MATRIX_ROW_PADDING);
            const int padding_down = ggml_cuda_mmq_get_J_max(weights->type, false, cc, 128);
            const size_t blocks_hidden = ne_get_rows*ne01 / QK8_1_MMQ;
            const size_t blocks_padded = ne_get_rows*hidden_padded / QK8_1_MMQ + padding_down;
            ggml_cuda_pool_alloc<block_q8_1_mmq> hidden(ctx.pool(), blocks_padded);
            CUDA_CHECK(cudaMemsetAsync(hidden.get() + blocks_hidden, 0, (blocks_padded - blocks_hidden)*sizeof(block_q8_1_mmq), stream));
            const auto ds_layout = mmq_get_q8_1_ds_layout(weights->type);
            launch_mul_mat_q_moe_swiglu<tile_small, true>(ctx, args, (const char *) gate->data, stream, hidden.get(), ds_layout);
            args.expert_tiles = tiles.get() + max_small;
            args.expert_tile_count = counts.get() + 1;
            args.expert_tiles_max = max_large;
            args.ncols_opt = tile_large;
            launch_mul_mat_q_moe_swiglu<32, true>(ctx, args, (const char *) gate->data, stream, hidden.get(), ds_layout);

            if (ggml_backend_buffer_get_usage(weights->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
                const size_t size_data = ggml_nbytes(weights);
                const size_t size_alloc = ggml_backend_buffer_get_alloc_size(weights->buffer, weights);
                if (size_alloc > size_data) {
                    GGML_ASSERT(ggml_is_contiguously_allocated(weights) && !weights->view_src);
                    CUDA_CHECK(cudaMemsetAsync((char *) weights->data + size_data, 0, size_alloc - size_data, stream));
                }
            }
            const size_t ts = ggml_type_size(weights->type);
            mmq_args down_args = {
                (const char *) weights->data, weights->type, (const int *) hidden.get(), ids_dst.get(), expert_bounds.get(), (float *) down->data,
                nullptr,
                weights->ne[0], weights->ne[1], ne_get_rows, int64_t(weights->nb[1]/ts), ne_get_rows, int64_t(down->nb[1]/sizeof(float)),
                ne02, ne02, int64_t(weights->nb[2]/ts), 0, int64_t(down->nb[2]/sizeof(float)),
                1, 1, int64_t(weights->nb[3]/ts), 0, int64_t(down->nb[3]/sizeof(float)),
                ne12, tile_small};
            mmq_moe_tiles<<<1, nthreads, 2*nthreads*sizeof(int), stream>>>(expert_bounds.get(), tiles.get(),
                    tiles.get() + max_small, counts.get(), ne02, tile_small, 128, limit_small);
            down_args.expert_tiles = tiles.get();
            down_args.expert_tile_count = counts.get();
            down_args.expert_tiles_max = max_small;
            ggml_cuda_mul_mat_q_switch_type(ctx, down_args, stream, GGML_PREC_Q8);
            down_args.expert_tiles = tiles.get() + max_small;
            down_args.expert_tile_count = counts.get() + 1;
            down_args.expert_tiles_max = ne02 + ne_get_rows/128;
            down_args.ncols_opt = 128;
            ggml_cuda_mul_mat_q_switch_type(ctx, down_args, stream, GGML_PREC_Q8);
            return;
        }
        if (gate) {
            launch_mul_mat_q_moe_swiglu<tile_small>(ctx, args, (const char *) gate->data, stream);
            args.expert_tiles = tiles.get() + max_small;
            args.expert_tile_count = counts.get() + 1;
            args.expert_tiles_max = max_large;
            args.ncols_opt = tile_large;
            launch_mul_mat_q_moe_swiglu<64>(ctx, args, (const char *) gate->data, stream);
            return;
        }
#endif
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        args.expert_tiles = tiles.get() + max_small;
        args.expert_tile_count = counts.get() + 1;
        args.expert_tiles_max = max_large;
        args.ncols_opt = tile_large;
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        return;
    }

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
}

bool ggml_cuda_should_fuse_mmq_moe(const ggml_tensor * up, const ggml_tensor * glu, bool check_batch) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    static const bool enabled = [] {
        const char * compact = getenv("GGML_CUDA_MMQ_MOE_COMPACT");
        const char * swiglu = getenv("GGML_CUDA_MMQ_MOE_SWIGLU");
        return compact && atoi(compact) != 0 && swiglu && atoi(swiglu) != 0;
    }();
    if (!enabled || up->op != GGML_OP_MUL_MAT_ID || glu->op != GGML_OP_GLU ||
            ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
        return false;
    }
    const auto * weights = up->src[0];
    const auto * input = up->src[1];
    const auto & device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if (!ampere_mma_available(device.cc) || weights->type != GGML_TYPE_Q4_K ||
            input->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 ||
            weights->ne[1] % 128 != 0 || weights->ne[2] < 64 || weights->ne[2] > 1024 ||
            weights->ne[3] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(glu) ||
            input->nb[0] != sizeof(float) || input->nb[2] % input->nb[1] != 0 ||
            (check_batch && (input->ne[2] < 128 || input->ne[2] > 16384 ||
                !ggml_cuda_should_use_mmq(weights->type, device.cc, input->ne[2], weights->ne[2])))) {
        return false;
    }
    for (int J : {16, 64}) {
        const auto config = ggml_cuda_mmq_get_config(weights->type, J, false, device.cc);
        if (config.type == GGML_TYPE_COUNT || !config.use_mma_data_layout(device.cc) ||
                mmq_get_nbytes_shared(config, device.cc) + ggml_cuda_mmq_get_nbytes_shared_x(config, device.cc) > device.smpbo) {
            return false;
        }
    }
    return true;
#else
    GGML_UNUSED_VARS(up, glu, check_batch);
    return false;
#endif
}

bool ggml_cuda_should_fuse_mmq_moe_down(const ggml_tensor * up, const ggml_tensor * glu, const ggml_tensor * down, bool check_batch) {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_MMQ_MOE_Q8");
        return value && atoi(value) != 0;
    }();
    if (!enabled || !ggml_cuda_should_fuse_mmq_moe(up, glu, check_batch) || down->op != GGML_OP_MUL_MAT_ID ||
            down->src[1] != glu || down->src[2] != up->src[2] || !ggml_is_contiguous(down)) {
        return false;
    }
    const auto * weights = down->src[0];
    const auto & device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    if ((weights->type != GGML_TYPE_Q5_1 && weights->type != GGML_TYPE_Q8_0) ||
            weights->ne[0] != glu->ne[0] || weights->ne[1] % 128 != 0 || weights->ne[2] != up->src[0]->ne[2] ||
            weights->ne[3] != 1 || weights->nb[0] != ggml_type_size(weights->type) || down->type != GGML_TYPE_F32 ||
            (check_batch && !ggml_cuda_should_use_mmq(weights->type, device.cc, glu->ne[2], weights->ne[2]))) {
        return false;
    }
    for (int J : {16, 32}) {
        const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_Q4_K, J, false, device.cc);
        if (config.I % QK8_1_MMQ != 0 || J*sizeof(int) + J*config.I*sizeof(float) > device.smpbo) {
            return false;
        }
    }
    for (int J : {16, 128}) {
        const auto config = ggml_cuda_mmq_get_config(weights->type, J, false, device.cc);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, device.cc) > device.smpbo) {
            return false;
        }
    }
    return true;
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    // MUSA: the MMQ kernels compute wrong values on PH1 (MTT S5000).
    if (cc == GGML_CUDA_CC_PH1) {
        return false;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
