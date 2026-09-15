#include "qsa.cuh"
#include "top-k.cuh"
#include "fattn.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define launch_fattn qsa_launch_fattn
#define ggml_cuda_flash_attn_ext_mma_f16_case qsa_mma_case
#include "fattn-mma-f16.cuh"
#undef ggml_cuda_flash_attn_ext_mma_f16_case
#undef launch_fattn
template void qsa_mma_case<256, 256, 1, 8>(ggml_backend_cuda_context &, ggml_tensor *);
#endif

static __global__ void qsa_mask_scores(const float * score, const uint32_t * bits, float * masked, int nb, int words, int rows) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= int64_t(nb)*rows) { return; }
    const int row = i/nb, b = i%nb;
    const float x = score[i];
    masked[i] = (bits[int64_t(row)*words + b/32] & (uint32_t(1) << (b%32))) && isfinite(x) ? x : -INFINITY;
}

static __global__ void qsa_expand(const int * chosen, const int * partial, const int * cells, const int * tail,
        int * out, int nb, int nq, int r, int k, int count, int start, int rows) {
    const int width = k*r + r - 1;
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= int64_t(width)*rows) { return; }
    const int row = i/width, j = i%width, original_row = start + row;
    const int * row_tail = tail + int64_t(original_row)*(r - 1);
    int n_tail = 0;
    for (int t = 0; t < r - 1; ++t) { n_tail += row_tail[t] >= 0; }
    const int cut = partial[row], extra = cut >= 0 ? r - 1 - n_tail : 0;
    int cell = -1;
    if (j < k*r + extra) {
        // The boundary block contributes only the tokens left after the tail.
        const int at = cut >= 0 && j >= cut*r + extra ? j + r - extra : j;
        const int b = chosen[int64_t(row)*count + at/r];
        if (b >= 0 && b < nb) {
            cell = cells[(int64_t(original_row/nq)*nb + b)*r + at%r];
        }
    } else {
        int at = j - k*r - extra;
        for (int t = 0; t < r - 1; ++t) {
            if (row_tail[t] >= 0 && at-- == 0) { cell = row_tail[t]; break; }
        }
    }
    out[int64_t(original_row)*width + j] = cell;
}

static __global__ void qsa_reduce_scores(const float * dots, const uint32_t * bits, float * masked, int nb, int words, int rows, int heads) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= int64_t(nb)*rows) { return; }
    const int row = i/nb, b = i%nb;
    float sum = 0.0f;
    for (int h = 0; h < heads; ++h) { sum += fmaxf(dots[(int64_t(row)*heads + h)*nb + b], 0.0f); }
    masked[i] = bits[int64_t(row)*words + b/32] & (uint32_t(1) << (b%32)) ? sum : -INFINITY;
}

static __global__ void qsa_order(const float * scores, int * chosen, int * partial, int nb, int k, int count, int padded) {
    extern __shared__ int ids[];
    __shared__ float reduction[32];
    __shared__ int boundary;
    const int row = blockIdx.x;
    if (threadIdx.x == 0) {
        boundary = -1;
        partial[row] = -1;
    }
    __syncthreads();
    if (count > k) {
        float worst = INFINITY;
        int worst_id = -1;
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            const int b = chosen[int64_t(row)*count + i];
            const float score = b >= 0 && b < nb ? scores[int64_t(row)*nb + b] : -INFINITY;
            const float x = isfinite(score) ? score : -INFINITY;
            if (x < worst || (x == worst && b > worst_id)) {
                worst = x;
                worst_id = b;
            }
        }
        const float lowest = -block_reduce<block_reduce_method::MAX>(-worst, reduction);
        if (worst == lowest) { atomicMax(&boundary, worst_id); }
        __syncthreads();
    }
    const int full = min(count, k);
    const bool has_partial = boundary >= 0 && boundary < nb && isfinite(scores[int64_t(row)*nb + boundary]);
    for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        int b = i < full ? chosen[int64_t(row)*count + i] : INT_MAX;
        if (i < full && b == boundary) { b = chosen[int64_t(row)*count + full]; }
        ids[i] = b >= 0 && b < nb && isfinite(scores[int64_t(row)*nb + b]) ? b : INT_MAX;
    }
    __syncthreads();
    for (int span = 2; span <= padded; span *= 2) {
        for (int step = span/2; step; step /= 2) {
            for (int i = threadIdx.x; i < padded; i += blockDim.x) {
                const int j = i ^ step;
                if (j <= i) { continue; }
                const int a = ids[i], b = ids[j];
                const bool before = a < b;
                if (before == ((i & span) != 0)) {
                    ids[i] = b; ids[j] = a;
                }
            }
            __syncthreads();
        }
    }
    // Sort only full blocks, then insert the boundary in logical order.
    for (int i = threadIdx.x; i < full; i += blockDim.x) {
        const int b = ids[i];
        chosen[int64_t(row)*count + i + (has_partial && b > boundary)] = b;
        if (has_partial && b > boundary && (i == 0 || ids[i - 1] < boundary)) {
            chosen[int64_t(row)*count + i] = boundary;
            partial[row] = i;
        }
    }
    if (threadIdx.x == 0 && count > full) {
        if (!has_partial) {
            chosen[int64_t(row)*count + full] = INT_MAX;
        } else if (ids[full - 1] < boundary) {
            chosen[int64_t(row)*count + full] = boundary;
            partial[row] = full;
        }
    }
}

static void qsa_shape(ggml_tensor & t, int n0, int n1) {
    t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = 1; t.ne[3] = 1;
    t.nb[0] = sizeof(float);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) { t.nb[i] = t.nb[i - 1]*t.ne[i - 1]; }
}

void ggml_cuda_op_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * scores = dst->src[0];
    const auto * queries = dst->src[4];
    const int nb = dst->src[1]->ne[1], nq = dst->ne[1], rows = nq*dst->ne[3];
    const int r = dst->src[1]->ne[0], k = ggml_get_op_params_i32(dst, 0), words = (nb + 31)/32;
    const int heads = queries ? queries->ne[1] : 1;
    const int tile = std::min(nq, 32);
    const int count_blocks = std::min(nb, k + 1);
    int padded = 1;
    while (padded < std::min(nb, k)) { padded *= 2; }
    ggml_cuda_pool_alloc<float> dots(ctx.pool(), queries ? size_t(nb)*heads*tile : 1);
    ggml_cuda_pool_alloc<float> masked(ctx.pool(), size_t(nb)*tile);
    ggml_cuda_pool_alloc<int> chosen(ctx.pool(), size_t(count_blocks)*tile);
    ggml_cuda_pool_alloc<int> partial(ctx.pool(), tile);
    for (int start = 0; start < rows;) {
        const int count = std::min(tile, nq - start%nq);
        if (queries) {
            const int d = queries->ne[0];
            const float alpha = 1.0f, beta = 0.0f;
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
            const auto precision = CUBLAS_COMPUTE_32F;
#else
            const auto precision = CUBLAS_COMPUTE_32F_PEDANTIC;
#endif
            CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                nb, heads*count, d, &alpha,
                (const char *) scores->data + int64_t(start/nq)*scores->nb[2], CUDA_R_32F, d,
                (const float *) queries->data + int64_t(start)*heads*d, CUDA_R_32F, d,
                &beta, dots.ptr, CUDA_R_32F, nb, precision, CUBLAS_GEMM_DEFAULT));
            qsa_reduce_scores<<<(int64_t(nb)*count + 255)/256, 256, 0, ctx.stream()>>>(dots.ptr,
                (const uint32_t *) dst->src[2]->data + int64_t(start)*words, masked.ptr, nb, words, count, heads);
        } else {
            qsa_mask_scores<<<(int64_t(nb)*count + 255)/256, 256, 0, ctx.stream()>>>(
                (const float *) scores->data + int64_t(start)*nb,
                (const uint32_t *) dst->src[2]->data + int64_t(start)*words, masked.ptr, nb, words, count);
        }
        ggml_tensor input = *scores, selected = *dst;
        qsa_shape(input, nb, count); input.data = masked.ptr;
        qsa_shape(selected, count_blocks, count); selected.data = chosen.ptr; selected.src[0] = &input;
        ggml_cuda_op_top_k(ctx, &selected);
        // Visit selected blocks in ID order, independent of their scores.
        qsa_order<<<count, 256, padded*sizeof(int), ctx.stream()>>>(masked.ptr, chosen.ptr, partial.ptr, nb, k, count_blocks, padded);
        qsa_expand<<<(int64_t(dst->ne[0])*count + 255)/256, 256, 0, ctx.stream()>>>(chosen.ptr, partial.ptr,
            (const int *) dst->src[1]->data, (const int *) dst->src[3]->data, (int *) dst->data, nb, nq, r, k, count_blocks, start, count);
        start += count;
    }
    CUDA_CHECK(cudaGetLastError());
}

struct qsa_layout {
    int d, dv, nq, nh, nk, gqa, width;
    size_t q1, q2, q3, k1, k2, k3, v1, v2, v3;
};

template<bool kv_q8>
static __device__ __forceinline__ float qsa_value(const char * row, int c) {
    if constexpr (kv_q8) {
        const auto & block = ((const block_q8_0 *) row)[c/QK8_0];
        return __half2float(block.d)*block.qs[c%QK8_0];
    } else {
        return __half2float(((const half *) row)[c]);
    }
}

template<bool kv_q8>
static __global__ void qsa_qk(const char * q, const char * k, const int * indices, float * scores, qsa_layout p, float scale) {
    const int lane = threadIdx.x%32, warp = threadIdx.x/32;
    const int j = blockIdx.x*4 + warp, t = blockIdx.y, h = blockIdx.z%p.nh, s = blockIdx.z/p.nh;
    if (j >= p.width) { return; }
    const int cell = indices[(int64_t(s)*p.nq + t)*p.width + j];
    float dot = -INFINITY;
    if (cell >= 0 && cell < p.nk) {
        const auto * query = (const float *) (q + t*p.q1 + h*p.q2 + s*p.q3);
        const auto * key = k + cell*p.k1 + (h/p.gqa)*p.k2 + s*p.k3;
        dot = 0.0f;
        for (int c = lane; c < p.d; c += 32) { dot += query[c]*qsa_value<kv_q8>(key, c); }
        for (int offset = 16; offset; offset /= 2) { dot += __shfl_xor_sync(0xffffffff, dot, offset, 32); }
        dot *= scale;
    }
    if (lane == 0) { scores[((int64_t(s)*p.nq + t)*p.nh + h)*p.width + j] = dot; }
}

template<bool kv_q8>
static __global__ void qsa_av(const char * v, const int * indices, const float * scores, float * out, qsa_layout p) {
    extern __shared__ float weights[];
    __shared__ float reduction[256];
    const int tid = threadIdx.x, h = blockIdx.x%p.nh, t = (blockIdx.x/p.nh)%p.nq, s = blockIdx.x/(p.nh*p.nq);
    const auto * row = scores + int64_t(blockIdx.x)*p.width;
    const auto * ids = indices + (int64_t(s)*p.nq + t)*p.width;
    float mx = -INFINITY;
    for (int j = tid; j < p.width; j += 256) { mx = fmaxf(mx, row[j]); }
    reduction[tid] = mx;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (tid < step) { reduction[tid] = fmaxf(reduction[tid], reduction[tid + step]); }
        __syncthreads();
    }
    mx = reduction[0];
    float den = 0.0f;
    for (int j = tid; j < p.width; j += 256) {
        const float w = isfinite(mx) ? expf(row[j] - mx) : 0.0f;
        weights[j] = w; den += w;
    }
    __syncthreads();
    reduction[tid] = den;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (tid < step) { reduction[tid] += reduction[tid + step]; }
        __syncthreads();
    }
    den = reduction[0];
    for (int c = tid; c < p.dv; c += 256) {
        float sum = 0.0f;
        for (int j = 0; j < p.width; ++j) {
            const int cell = ids[j];
            if (cell >= 0 && cell < p.nk && weights[j] > 0) {
                const auto * val = v + cell*p.v1 + (h/p.gqa)*p.v2 + s*p.v3;
                sum += weights[j]*qsa_value<kv_q8>(val, c);
            }
        }
        out[int64_t(blockIdx.x)*p.dv + c] = den > 0 ? sum/den : 0.0f;
    }
}

static __global__ void qsa_zero_empty(const int * indices, float * out, int width, int row_size) {
    bool visible = false;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        visible |= indices[int64_t(blockIdx.x)*width + j] >= 0;
    }
    if (__syncthreads_or(visible)) { return; }
    for (int j = threadIdx.x; j < row_size; j += blockDim.x) { out[int64_t(blockIdx.x)*row_size + j] = 0.0f; }
}

static __global__ void qsa_mask_fill(half * mask, int64_t count) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < count) { mask[i] = __float2half(-INFINITY); }
}

static __global__ void qsa_mask_scatter(const int * indices, half * mask, int width, int nk, int rows) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < int64_t(width)*rows && indices[i] >= 0 && indices[i] < nk) {
        mask[(i/width)*nk + indices[i]] = __float2half(0.0f);
    }
}

static bool qsa_attn_dense(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0];
    const auto * indices = dst->src[5];
    const int64_t nk = dst->src[1]->ne[1], rows = q->ne[1]*q->ne[3];
    // Small batches keep indexed reduction order when KV cells move.
    // Bound dense prefill masks independently of the context limit.
    if (q->ne[1] < 32 || nk > 4096 || nk*rows > 4*1024*1024 || getenv("LLAMA_QSA_SCALAR")) { return false; }

    ggml_tensor dense = *dst, mask = *indices;
    mask.type = GGML_TYPE_F16;
    mask.ne[0] = nk;
    mask.nb[0] = sizeof(half);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) { mask.nb[i] = mask.nb[i - 1]*mask.ne[i - 1]; }
    dense.op = GGML_OP_FLASH_ATTN_EXT;
    dense.src[3] = &mask;
    dense.src[5] = nullptr;
    ggml_flash_attn_ext_set_n_kv_max(&dense, 0);
    if (!ggml_cuda_flash_attn_ext_supported(ctx.device, &dense)) { return false; }
    // Quantized KV must stay packed; use indexed attention if dense attention needs an F16 copy.
    if (ggml_cuda_flash_attn_ext_get_alloc_size(ctx.device, &dense) != ggml_nbytes(dst)) { return false; }

    ggml_cuda_pool_alloc<half> memory(ctx.pool(), nk*rows);
    mask.data = memory.ptr;
    qsa_mask_fill<<<(nk*rows + 255)/256, 256, 0, ctx.stream()>>>(memory.ptr, nk*rows);
    qsa_mask_scatter<<<(indices->ne[0]*rows + 255)/256, 256, 0, ctx.stream()>>>(
        (const int *) indices->data, memory.ptr, indices->ne[0], nk, rows);
    ggml_cuda_flash_attn_ext(ctx, &dense);
    qsa_zero_empty<<<rows, 128, 0, ctx.stream()>>>(
        (const int *) indices->data, (float *) dst->data, indices->ne[0], q->ne[2]*dst->src[2]->ne[0]);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template<bool kv_q8>
static void qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0]; const auto * k = dst->src[1]; const auto * v = dst->src[2];
    const auto * indices = dst->src[5];
    GGML_ASSERT(indices && indices->ne[0] <= 4096);
    const qsa_layout p = {int(q->ne[0]), int(v->ne[0]), int(q->ne[1]), int(q->ne[2]), int(k->ne[1]), int(q->ne[2]/k->ne[2]), int(indices->ne[0]),
        q->nb[1], q->nb[2], q->nb[3], k->nb[1], k->nb[2], k->nb[3], v->nb[1], v->nb[2], v->nb[3]};
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (p.d == 256 && p.dv == 256 && ggml_cuda_info().devices[ctx.device].cc >= GGML_CUDA_CC_AMPERE && !getenv("LLAMA_QSA_SCALAR")) {
        if constexpr (kv_q8) {
            ggml_cuda_flash_attn_ext_mma_f16_case_impl<256, 256, 1, 8, GGML_TYPE_Q8_0>(ctx, dst);
        } else {
            qsa_mma_case<256, 256, 1, 8>(ctx, dst);
        }
        qsa_zero_empty<<<p.nq*q->ne[3], 128, 0, ctx.stream()>>>((const int *) indices->data, (float *) dst->data, p.width, p.nh*p.dv);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif
    const int rows = p.nq*p.nh*q->ne[3];
    ggml_cuda_pool_alloc<float> scores(ctx.pool(), size_t(rows)*p.width);
    qsa_qk<kv_q8><<<dim3((p.width + 3)/4, p.nq, p.nh*q->ne[3]), 128, 0, ctx.stream()>>>(
        (const char *) q->data, (const char *) k->data, (const int *) indices->data, scores.ptr, p, ggml_get_op_params_f32(dst, 0));
    qsa_av<kv_q8><<<rows, 256, sizeof(float)*p.width, ctx.stream()>>>(
        (const char *) v->data, (const int *) indices->data, scores.ptr, (float *) dst->data, p);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (qsa_attn_dense(ctx, dst)) { return; }
    if (dst->src[1]->type == GGML_TYPE_Q8_0) {
        qsa_attn<true>(ctx, dst);
    } else {
        qsa_attn<false>(ctx, dst);
    }
}
