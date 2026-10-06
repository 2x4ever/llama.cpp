# Fork differences from upstream

Audited rebase code: `c94c992b9b4a7b40ba0b22d9db19d437844d16c9`, followed by the compact Native QSA storage refinement recorded below. Compared upstream base: `f0c41e0168dfd4b5ef72b21d1a311b24cc7a894a`. Inventory date: 2026-10-06. Validation distinguishes completed checks from untested configurations; do not treat source retention as proof of runtime equivalence.

`active` means a difference retained relative to that base, not a claim that newer upstream lacks an equivalent. Other allowed states are `partial` (upstream covers part), `upstream` (verified upstream replacement) and `retired` (removed without replacement). For each upstream match, record its commit, verified coverage, remaining delta and tests. [Maintenance policy and rebase procedure](../../FORK-MAINTENANCE.md).

## Inventory

F08 and F14 have partial upstream coverage; their remaining differences are described below. Other entries remain active. Commit IDs are anchors in the rebased fork history, not instructions to cherry-pick onto an arbitrary base.

| ID | Difference | Fork commits | Primary source |
| --- | --- | --- | --- |
| F01 | Configurable prefill/decode scheduling ratio | `05278c93b` | [server-context.cpp](../../tools/server/server-context.cpp), [arg.cpp](../../common/arg.cpp) |
| F02 | Indexed KV sequence range removal | `0d3afebbe` | [llama-kv-cells.h](../../src/llama-kv-cells.h), [llama-kv-cache.cpp](../../src/llama-kv-cache.cpp) |
| F03 | Conditional sampler checkpoint cloning | `93d61b25f` | [server-context.cpp](../../tools/server/server-context.cpp) |
| F04 | SSE2 sampling-candidate initialization | `dcc5d6d6a`, `6832d1bfb`, `ce769e5e4` | [sampling.cpp](../../common/sampling.cpp) |
| F05 | Native Q8_0 KV tiles in CUDA MMA Flash Attention | `2572402fa` | [fattn-mma-f16.cuh](../../ggml/src/ggml-cuda/fattn-mma-f16.cuh), [fattn.cu](../../ggml/src/ggml-cuda/fattn.cu) |
| F06 | Native Q4_0 KV tiles in CUDA MMA Flash Attention | `e925eec37` | same CUDA Flash Attention files as F05 |
| F07 | Compact attention-mask representation | `e377b3f45` | [llama-context.cpp](../../src/llama-context.cpp), [llama-kv-cache.cpp](../../src/llama-kv-cache.cpp), [fattn-common.cuh](../../ggml/src/ggml-cuda/fattn-common.cuh) |
| F08 | Native QSA and incremental indexer cache | `a5c535fe1`, `95d8d17bc`, `499ca8229` | [qwen4exp.cpp](../../src/models/qwen4exp.cpp), [llama-memory-hybrid-idx.cpp](../../src/llama-memory-hybrid-idx.cpp), [llama-qsa.h](../../src/llama-qsa.h), [qsa.cu](../../ggml/src/ggml-cuda/qsa.cu) |
| F09 | Vulkan event/command-buffer ownership fix | `b44110e58` | [ggml-vulkan.cpp](../../ggml/src/ggml-vulkan/ggml-vulkan.cpp), [ggml-vulkan-types.h](../../ggml/src/ggml-vulkan/ggml-vulkan-types.h) |
| F10 | Shared workspace pool and execution fences | `4661f0baf` | [ggml-alloc.c](../../ggml/src/ggml-alloc.c), [ggml-backend.cpp](../../ggml/src/ggml-backend.cpp), [ggml-backend-impl.h](../../ggml/src/ggml-backend-impl.h) |
| F11 | CPU pipeline workers and continuous submission | `ab0003b5c`, `89bebf207` | [llama-context.cpp](../../src/llama-context.cpp), [llama-ext.h](../../src/llama-ext.h), [server-context.cpp](../../tools/server/server-context.cpp) |
| F12 | MoE MMQ input padding for full tile reads | `508b9b99e` | [mmq.cu](../../ggml/src/ggml-cuda/mmq.cu) |
| F13 | Recurrent rollback history on cell relocation | `a1dca529d` | [llama-memory-recurrent.cpp](../../src/llama-memory-recurrent.cpp) |
| F14 | MTP integration in pipeline workers | `7855102fc` | [speculative.cpp](../../common/speculative.cpp), [llama-context.cpp](../../src/llama-context.cpp), [server-context.cpp](../../tools/server/server-context.cpp) |
| F15 | Separate CUDA and HIP unified-memory switches | `d72bab2fc` | [ggml-cuda.cu](../../ggml/src/ggml-cuda/ggml-cuda.cu), [build.md](../build.md) |
| F16 | RPC operation count/protocol patch update for QSA | `c94c992b9` | [ggml-rpc.h](../../ggml/include/ggml-rpc.h) |
| F17 | Fork inventory, maintenance policy and validation record | documentation following `c94c992b9` | [FORK-MAINTENANCE.md](../../FORK-MAINTENANCE.md), [AGENTS.md](../../AGENTS.md), this directory |
| F18 | Compact CUDA MoE work lists with per-expert tile widths | implementation following `d43f0dfcd` | [mmq.cu](../../ggml/src/ggml-cuda/mmq.cu), [mmq.cuh](../../ggml/src/ggml-cuda/mmq.cuh) |

## Adaptation to upstream f0c41e016 (2026-10-06)

F08 is `partial`: upstream `66e0c17ee` adds incremental pooled-key caching and sequence-relative groups; `889edf43d` adds LIGHTNING_INDEXER scoring and `3cf03257f` enables sparse CUDA attention. Retain the native compact F32 pooled cache, FP32 score calculation, full token budget and direct selected-index attention. Native groups now sort by (position, physical cell) and locate each query by its actual allocated cell, including gaps and nonzero starts. Native-enabled F16/Q8 contexts store raw-only indexer rows. Native-disabled and unsupported configurations retain upstream packed raw/pooled rows. Keeping a compact external cache avoids allocating one F32 pooled row for every physical KV cell. The host native and upstream layouts are both present; consolidating them requires a measured replacement with the same memory bound.

The native budget remains a deliberate difference: it fills `k*ratio+ratio-1` entries when enough history exists, while the new upstream keeps whole selected pools plus the actual incomplete tail. The two paths also differ in intermediate precision. Native disabled selects the new upstream path; neither implementation is asserted to be a reference-model oracle.

Spatial M-RoPE batches and queries sharing a scalar position with spatial cached cells fall back to upstream's causal mask. The fallback invalidates native keys and upstream pooled keys; later eligible text rebuilds native keys. This fallback can need dense-mask compute memory. Main and MTP graphs use the same eligibility checks. Ordinary text with four equal position coordinates remains eligible for Native QSA.

Upstream sparse CUDA attention appends live counts to its selected-index tensor; Native QSA passes a direct list with -1 padding. The internal dispatch distinguishes those layouts and does not read counts past a native list. Compact I32 masks remain a separate representation. Native quantized tile writers use the upstream MMA swizzle interface and keep the bounded preload storage.

F09 follows the moved command-buffer definition in ggml-vulkan-types.h. F10 keeps backend API version 3, the alloc_buffer_n interface, graph allocation dependencies and the new expert-copy callback; fences cover input copies, callback work and device execution. F11 submits through the canonical llama_batch_ext decode path used by llama_process. Queued work owns its storage, and unsupported mixed/embedding/decision-order batches fall back to the main context.

F12 remains active: upstream changes MoE sizing to ne12, but destination IDs still need full-tile padding. F13 gathers inactive history indices from the new state_copy layout using its i0 offset and retains the upstream rollback, split-replay and shared-sequence tests.

F14 is `partial`: upstream supplies original-row tracking and NextN output reordering, so the old fork batch_ids/scatter implementation is not replayed. Worker aggregation uses that ordering once. The retained MTP lane integration copies hidden/verification rows and per-sequence draft sampler state, candidates, RNG and temperature/seed configuration at drained boundaries. Probabilistic rejection remains available; PR #14 concurrent draft execution remains excluded.

F01-F04 and F15 remain separate differences after source review. F16 retains GGML_OP_COUNT=103 and RPC protocol patch version 1 for the two native QSA operations. See [VALIDATION.md](VALIDATION.md) for measured revisions and test coverage.

## Scheduling and sampling: F01-F04

F01 adds `--prefill-decode-ratio`. It changes admission of prompt work while decode is active, not GPU kernel arithmetic. Preserve mixed-batch scheduling semantics, the default and option parsing. Validation: `test-arg-parser` and overlapping prefill/decode server requests. An upstream replacement must expose equivalent behavior, not merely continuous batching.

F02 uses the existing per-sequence position index and `lower_bound` to remove a range instead of scanning the full KV capacity. Preserve negative/default range semantics, shared cells, the position index and empty-cell accounting. Validation: sequence remove/copy, fragmented save/restore and actual speculative rejection. No extra index is introduced.

F03 clones a sampler only when acceptance may need a checkpoint restore: FULL restore or recurrent restore beyond its rollback depth. Preserve the snapshot before mutation and require it on the restore path. Validation must actually reject draft tokens and exercise restore, including grammar and short recurrent histories; accepted-only speculation is insufficient.

F04 initializes candidate records with SSE2 and a scalar tail/fallback. Preserve every `{id, logit, p}` bit, alignment handling and exceptional float values. The explicit NEON variant was removed because Apple Clang already vectorized the loop and no benefit was measured. Do not restore it just because its commit appears in history. Validation: sampling tests and bitwise candidate comparisons on x86 and a fallback platform.

## Attention memory: F05-F07

F05/F06 load packed Q8_0/Q4_0 KV tiles and dequantize into bounded on-chip FP16 tile storage instead of allocating a full-context FP16 KV copy for the eligible CUDA MMA path. The supported native-quant path requires NVIDIA Turing or newer, compiled MMA support and a compatible type/shape/kernel selection; Q4_0 additionally requires a 256-dimensional query head. K and V types must match. Other paths can still need staging. These changes are independent of `GGML_CUDA_FORCE_MMQ`, which controls ordinary quantized matrix multiplication.

Preserve agreement between kernel dispatch and `ggml_cuda_flash_attn_ext_get_alloc_size`, strided/view addressing, quantization scales, tile bounds and synchronization around prefetch reuse. A flag or small reported KV cache is not proof of absent staging. Validate F16/Q8_0/Q4_0 operator outputs, views, quantized types actually compiled, peak allocation and throughput. Do not extend support by relaxing a capability gate without implementing and testing the selected kernel.

F07 replaces a dense query-by-KV mask with compact I32 position/sequence metadata and evaluates visibility inside attention. `LLAMA_FLASH_ATTN_COMPACT_MASK=1` uses it when smaller; `2` forces it for supported batches. The current model gate is GGUF `qwen35`, with unified causal KV, Flash Attention, full offload, no tensor placement overrides, no SWA, at most 32 sequences, head dimensions 256 and matching F16/Q8_0/Q4_0 K/V. The CUDA compact-mask kernel gate is Ampere or newer. Only devices used by the context's attention layers are probed, separately for main and MTP contexts.

Keep CPU reference semantics, M-RoPE/position coordinates, sequence visibility and CUDA MMA/vector handling aligned. Preserve the dense fallback. This is not the Qwen4Exp QSA representation and does not currently support V100. Validation: compact-versus-dense operator cases, main/MTP logs, large-context reservations and real inference. Treat a newer upstream mask API as a potential replacement only after checking these semantics and memory scaling.

## Qwen4Exp sparse attention: F08

Enable with `LLAMA_QSA_NATIVE=1` and causal Flash Attention on compatible Qwen4Exp layers. Main K/V must have matching F16 or Q8_0 types. Spatial M-RoPE and queries sharing a scalar position with spatial cached cells use the upstream fallback described in the adaptation section above. The graph uses `QSA_SELECT` and `QSA_ATTN`; these are added ggml operations with CPU and CUDA/HIP implementations and unsupported-backend dispatch guards. Relevant definitions are in [ggml.h](../../ggml/include/ggml.h), [ggml.c](../../ggml/src/ggml.c) and [CPU ops](../../ggml/src/ggml-cpu/ops.cpp).

The indexer keeps a compact F32 pooled-key cache and updates changed blocks. If distinct shared/divergent block layouts exceed its ceil(kv_size/ratio) capacity, it re-pools all blocks into temporary F32 output; raw-key gathers remain chunked. A FILL operation gives this temporary output a backend before chunk copies write its views. In native Q8 mode raw indexer K remains F16 while main KV stays Q8_0. Native-enabled F16/Q8 contexts store only raw indexer keys in the per-token cache. The upstream fallback pooled slots use a separate, same-type buffer allocated per layer on first use. Allocation is protected by a mutex shared by all worker contexts. The buffer stays alive until the memory object is destroyed, so cached graphs keep valid pointers. It is included in memory accounting and omitted from saved state; restore rebuilds the derived keys. This avoids the extra per-token pooled reservation in text-only native execution. A context that enters fallback retains that reservation afterwards, and fallback may still require dense-mask compute memory. Serialized indexer row widths change; snapshots with the previous packed native rows must be regenerated. Fallback allocations are preflighted before applying a microbatch, including runtime non-causal attention; allocation failure returns -2 before applying the microbatch or rolling back existing positions. Graph construction and reservation handle allocation exceptions. Returning from any fallback invalidates native pooled keys before reuse. Block visibility and membership are explicit; layout changes, sequence edits, clear and state restore must invalidate or rebuild the derived cache.

The selector computes scores in FP32, processes query tiles and expands chosen blocks into token IDs. Preserve the token budget `k * ratio + ratio - 1`: all valid tail members plus enough members of an extra ranked block must be included. Compact tail holes, leave unused slots as `-1`, and order traversal by logical block ID, not relocated physical KV address. The boundary-block reduction and parallel ordering avoid a serialized selection bottleneck. Equal-score candidate choice is backend-defined; universal bitwise agreement with stock is not promised.

Indexed D256 attention uses the optimized CUDA MMA path on Ampere or newer. Generic CUDA/HIP and CPU paths consume selected values directly, including Q8_0 dequantization, without a full FP16 main-KV copy. The generic CUDA path has FP32 score scratch proportional to selected width, heads and query batch. A bounded dense-FA shortcut is allowed only for at least 32 queries, physical KV span at most 4096, at most 8 MiB of temporary mask, and no extra FP16 KV staging. Preserve this bound and the small-query indexed fallback, which prevents physical relocation from changing short-decode reduction order.

V100 has a generic source path but no runtime validation here; it does not use the Ampere indexed fast path. Q4_0 main KV is not supported by native QSA. QSA operator output allocation and backend scratch are distinct; do not report output-sized allocation as zero total temporary memory.

Validation: CPU/CUDA/HIP selection and indexed attention, tail lengths, holes, shared/fragmented layouts, restored state, large block counts and CUDA memory/race checks. Compare PPL on matched F16/Q8_0 corpora and long-context decode against the explicit baseline. A previously underfilled token budget caused a quality discrepancy; retaining the formula and tests is essential. Final sampled PPL results are in [VALIDATION.md](VALIDATION.md), without a universal losslessness claim. An upstream QSA implementation must also preserve these layout and memory properties before replacing this entry.

## Shared execution: F09-F12

F09 makes Vulkan command-buffer reset/reuse the responsibility of the owning pool thread. A waiter can publish completion but must not reset another thread's command buffer. Validate cross-thread waits and delayed reuse; this fix is required by shared-pool worker execution on Vulkan.

F10 separates reusable device scratch from private inputs, outputs and inter-stage tensors. Pools share compatible default CUDA/ROCm/Vulkan device buffers; host and unsupported buffer types stay private. Allocation reference counts keep live buffers valid across growth. Execution fences serialize conflicting use of the same shared scratch while allowing work on distinct devices to overlap. Cancellation/failure must publish failure to dependents rather than leaving waiters blocked. Events cannot be recycled while earlier plans still reference them. Internal hooks live in `ggml-backend-impl.h`, not a new public scheduler API.

F11 uses separate CPU threads, graph/scheduler/backend state and outputs per worker, sharing model weights and unified KV. Prompt batches return after CPU submission so subsequent batches can keep GPUs busy; reading outputs and mutating state must wait for relevant GPU work. Server decode lanes run independent requests and batch excess requests within a lane. Preserve sequence ownership, memory mutexes, output ordering, lifetime, arrival/cancellation drains and fallback to the main context.

`LLAMA_PIPELINE_WORKERS=2..8` enables workers for GGUF `qwen35` and `qwen4exp`, with full layer offload across multiple devices, KQV offload, unified KV and no tensor placement overrides. `-ub` is the main physical batch limit and default worker limit. `LLAMA_PIPELINE_UBATCH` overrides only the worker limit, up to `-b`. `LLAMA_PIPELINE_STREAM=0` disables independent server decode lanes but retains worker prefill. The main context remains necessary for unsupported operations and single-request speculative verification. See the [server documentation](../../tools/server/README.md#cpu-pipeline-workers-experimental) for eligibility details.

`LLAMA_PIPELINE_POOL` is obsolete and unread in this baseline; compatible scratch pooling is automatic when workers are enabled. Old `LLAMA_PIPELINE_ASYNC`/`LLAMA_PIPELINE_CACHE` launchers describe an experimental executor absent from this baseline. Stock scheduler input copies provide a different form of pipelining; their activation log does not prove CPU worker activation. More workers do not guarantee more throughput, and private memory still grows.

F12 pads MoE MMQ temporary input storage for the full rows read by GPU tiles. The bug becomes visible with worker-dependent batch shapes but is not solved by limiting the worker count. Preserve bounds for every tile shape and expert arrangement when MMQ changes upstream. Validate MoE operators with memory checking and multiple worker/microbatch sizes.

Validation for this group: existing `test-alloc`, backend `events`/`workspace` modes, `test-thread-safety pipeline`, real request lifecycle tests, and balanced prefill/decode throughput. An upstream scheduler replacement must retain continuous submission, private boundaries, event ownership and error drains; async GPU submission alone is not equivalent to independent worker state.

## Recurrent state and MTP: F13-F14

F13 preserves the full recurrent rollback history when inactive cells move. Copying only the latest state corrupts later partial rollback even if ordinary generation succeeds. Validate with `test-recurrent-state-rollback relocation`, then speculative rejection and state restore. Retain this coverage when adopting upstream recurrent-memory changes.

F14 routes MTP hidden rows and logits in original input order through target prefill partitions and worker execution. Pending hidden states follow requests when entering/leaving lanes. Supported single-head draft-MTP uses per-lane speculative and draft contexts with shared draft weights/KV/scratch. Access to shared draft memory is serialized in this baseline; it is not the optional concurrent-draft implementation from PR #14.

Target rollback depth must cover the draft limit, the draft must support partial sequence removal, and worker microbatch size must exceed rollback depth plus one. Preserve synchronization for suffix removal, state ownership and failure cleanup. Chained heads and shared target/draft memory are outside this worker path. Validation combines `test-thread-safety pipeline-mtp` (target row routing/error recovery) with a real target-plus-draft server, acceptance/rejection, 1/2/4 concurrent requests, cancellation and slot reuse. An upstream equivalent must cover both target routing and draft lifecycle.

## Build/runtime compatibility: F15-F16

F15 separates `GGML_CUDA_ENABLE_UNIFIED_MEMORY` from `GGML_HIP_ENABLE_UNIFIED_MEMORY`. Presence enables the respective switch, even a value of `0`; unset it to disable. Preserve per-backend independence in a mixed process. It does not make managed memory efficient on every AMD/NVIDIA GPU or restrict it to weights. Validate default and independently enabled allocation paths; do not attribute an MMQ or rocBLAS result to this policy change.

F16 updates RPC's operation-count guard from 101 to 103 and protocol patch version from 0 to 1 for the two appended QSA operations. Existing operation IDs and wire layout remain unchanged at this baseline. The handshake checks major/minor, not patch equality; this change does not guarantee rejection of an incompatible peer. Native QSA over RPC requires compatible implementations on both ends and has not been validated end to end here.

On rebase, reconcile newly added upstream operations and protocol changes deliberately. Do not merely replace the assertion with the compiler's new count or delete the guard. Build RPC and run `test-rpc-multi-server`; record any required protocol/version migration. Operator insertion order, unsupported-backend switches and the QSA allocator contract must stay consistent.

## Documentation and retired experiments: F17

This inventory, the root maintenance guide, validation record, README entry point and AGENTS.md policy are fork-specific documentation. Update them together when behavior or upstream coverage changes. Historical validation is dated evidence and must not be rewritten as a new test run.

Explicit NEON candidate initialization is retired within F04. Experimental single-scheduler async execution and gfx906 MMQ tuning were not ported. Optional draft batching/concurrent execution remains outside the baseline. Keep these distinctions when reading old patches, branch names or performance tables. A feature branch is not evidence that its code is in master.

## CUDA MoE prefill: F18

`GGML_CUDA_MMQ_MOE_COMPACT=1` enables an experimental CUDA MMQ path. Unset or zero retains the original dispatch. Ordinary MMQ reserves a rectangular launch grid using the whole microbatch for every expert; sparse routing leaves many blocks empty and many nonempty tiles underfilled. F18 builds two GPU work lists from the existing expert boundaries. Experts with 1-16 routed rows use 16-column tiles; experts with more rows use 128-column tiles. Persistent blocks iterate only the listed work, with a grid capped at twice the device SM count. Empty experts contribute no tiles. The lists and their live counts are rebuilt on the GPU for every operation, including CUDA Graph replay, without a host readback.

The implementation reuses existing activation quantization, routed input ordering, MMQ tile arithmetic and output scatter. It does not change routing, expert weights, QSA, public ggml APIs or the selected expert count. Different tile/reduction choices can still change floating-point results; operator and model validation are required. Temporary list storage comes from the existing stream-local CUDA pool. Shared execution depends on its existing stream ownership and completion guarantees, not a new global pool.

Eligibility requires NVIDIA Ampere or newer with compiled MMA support, Q4_K/Q5_1/Q8_0 weights, 64-1024 experts, 128-16384 input tokens, one expert-weight sample, an output row count divisible by 128, and sufficient per-block shared memory for both tile sizes. Other quantizations, shapes and devices retain the existing path. Single-token decode is unchanged. The switch does not force MMQ when normal dispatch chooses another operation. CUDA sm_86 is the measured target; HIP, Vulkan, pre-Ampere NVIDIA and other model/quantization combinations have no performance claim.

F12 padding remains required: tile loads may read past an expert's live rows within the padded temporary allocation. The small-list capacity assumes at most one tile per small expert. Changing tile widths or the threshold requires updating these bounds and rerunning memory/race checks. Preserve strided expert weights, non-power-of-two row-tile counts and dynamic routing when replacing this path.

Compared upstream `f0c41e016` has no equivalent compact NVIDIA work list. Related AMD downstream work is discussed in [upstream discussion 26349](https://github.com/ggml-org/llama.cpp/discussions/26349); that is not evidence of coverage in the pinned upstream base. On rebase, replace F18 only after matched correctness, CUDA Graph/pool-lifetime and sparse/dense-routing throughput checks. See [VALIDATION.md](VALIDATION.md) for the exact measured configuration and limitations.
