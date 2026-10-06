# Fork validation record

This record separates current validation from historical evidence. Reproducible repository test entry points are in [FORK-MAINTENANCE.md](../../FORK-MAINTENANCE.md). Raw artifacts live outside the source tree; exact locations and hashes are recorded per validation date.

## Compact Native QSA storage: 2026-10-06

This refinement follows the rebase below. Its baseline is `c94c992b9b4a7b40ba0b22d9db19d437844d16c9` (the following `76e2b86c7` commit changes documentation only), not stock upstream. Native-enabled compatible Qwen4Exp contexts retain raw-only F16 indexer rows and the compact F32 pooled cache. The upstream fallback gets separate per-layer pooled storage on first use. No selection or attention arithmetic changes.

On Flash-Next UD-Q4_K_XL, four RTX 3090s, Q8_0 main KV, native QSA, context 500000, batch 2048, microbatch 512 and no managed memory, the per-GPU per-token indexer allocation falls from 732.75 MiB to 366.38 MiB. This saves 1465.5 MiB across four GPUs. The sampled peak on the fullest card falls from 23470 to 23104 MiB. Compute reservations are 304.30 MiB on CUDA0 and 424.30 MiB on CUDA1-3; CUDA_Host is 113.57 MiB. The 500K case validates startup/reservation, not processing a 500K prompt.

Throughput comparison uses the same real model on those four GPUs, context 131072, Q8_0 main KV, `-b 2048 -ub 512 -np 4`, eight CPU threads, native QSA, no CPU pipeline workers, UM or vision, and two fresh-prompt repetitions per depth. Values are mean server tokens/s; decode generates 128 tokens. Changes this small are not evidence of a speedup.

| Workload | Packed native storage | Compact native storage |
| --- | ---: | ---: |
| Prefill 8192 | 1093.31 | 1091.61 |
| Decode after 8192 | 62.36 | 62.98 |
| Prefill 65536 | 984.27 | 985.36 |
| Decode after 65536 | 58.84 | 58.80 |

The four generated text responses match the baseline. A later error-path-only adjustment moves fallback allocation preflight out of the generic rollback path; this does not change the successful graph. Targeted failure tests and the real five-worker lifecycle test were rerun after that adjustment. The exact final source patch and binary hashes are retained with the evidence.

Validation:

- ARM64 CPU/Accelerate and CUDA 12.9 sm86 server and relevant tests build. Final ARM CTest sampling, batch allocation, arguments, allocation and RPC checks pass 5/5. F16/Q8 synthetic text/spatial/text transitions, including saved spatial state restored into a fresh context, have bitwise-identical logits to the frozen packed-storage baseline on the same backend. Native-off and unsupported F32 CPU rollback cases pass. F16/Q8 recurrent rollback, split replay and shared-prefix tests pass on CPU and CUDA.
- Lazy allocation, eight concurrent acquisitions sharing one tensor, and zero-byte placeholders for no-alloc fitting pass. Switching causal attention off and back on rebuilds the derived cache; fresh-context restore and retry after allocation failure agree with uninterrupted execution on CPU.
- Fault injection fails the first or second fallback-layer allocation. Decode returns -2, preserves old positions and succeeds on retry. A spatial batch starting at the already cached last position with four recurrent rollback snapshots previously deleted that position during error cleanup; the final preflight bypasses this cleanup. F16/Q8 overlap, overlap-plus-restore and non-causal fault cases pass on ARM and CUDA. CUDA Compute Sanitizer memcheck of overlap/failure/retry/restore reports zero errors.
- The divergent shared-prefix overcapacity branch passes on CUDA. Real Flash-Next pipeline tests pass with two workers/F16 and five workers/Q8. Prefill, all-output prefill, decode replay, failure/drain/reuse and duplicate-sequence rejection have max_abs=0 in compared outputs. The final five-worker run was repeated after the last error-path fix.

Failure history is retained: one real-model attempt failed while allocating model weights before context construction, then passed on retry without a code change. An initial second-layer fault test did not inject failure because its hook covered one backend buffer type while layers were split across devices; the corrected test places both indexer layers on one GPU. These setup/load failures are not counted as passing tests.

Limitations: fallback storage stays allocated until its shared memory object is destroyed, to keep cached graph pointers valid. A context that uses spatial or non-causal fallback therefore regains the extra reservation; dense fallback masks can also cost memory. Spatial coverage uses generated embeddings, not a real vision tower. The native serialized indexer row width changes, so old packed-native snapshots must be regenerated. No new PPL run, reference-model quality comparison, HIP/Vulkan/P40/V100 execution, or mixed-backend model run was performed for this storage refinement.

Raw evidence is in `investigations/upstream-rebase-2026-10-06/compact-indexer` locally and `/home/user/fork-rebase-20261006/compact-indexer` on `user@192.168.50.53`. Start with `RESULTS.md`; `final-code.patch`, `source-hashes-final.json`, `final-binary-hashes.json`, test runners, result JSON and logs identify the measured code and commands. A portable `compact-indexer-validation.tar.gz` and adjacent SHA256 are stored in the parent directory on both hosts. The original server remains stopped.

## Upstream rebase and Native QSA adaptation: 2026-10-06

| Role | Commit |
| --- | --- |
| Published pre-rebase fork | `b54151e974f7bb2e6ac0b88bb80d92fb15af401c` |
| Pre-rebase fork including local maintenance docs | `85fade12817ec879a68b662bd59613605eba4979` |
| Previous upstream base | `37b53fd4545847188fdad29e38ba57875efc8228` |
| New upstream base | `f0c41e0168dfd4b5ef72b21d1a311b24cc7a894a` |
| Tested rebased code, before documentation | `c94c992b9b4a7b40ba0b22d9db19d437844d16c9` |
| Broad GPU test snapshot before the final temporary-buffer fix | `bb71bcfa77ff400671585306ceb8352773dc0725` |

The final code differs from the broad GPU test snapshot only by initializing the uncached QSA pooled tensor with a FILL operation, which assigns its backend when divergent shared-prefix layouts exceed cache capacity. Performance and PPL below were measured before this final fix; the ordinary cached graph is unchanged. Targeted final-fix checks are recorded separately.

The new base adds 445 upstream commits. The replay retains 22 non-merge commits and reconstructs 14 merge commits; F08 and F14 required semantic adaptation, not just conflict resolution. The compact native QSA cache and full token budget remain. Upstream sequence-relative grouping, packed indexer rows, sparse attention count metadata, batch_ext submission, recurrent history indexing and NextN row ordering are integrated as described in [CHANGES.md](CHANGES.md). PR #14 remains excluded. The dirty experimental checkout and backup refs were preserved.

Executed checks:

- ARM64 CPU/Accelerate Release: server and relevant tests built, CTest 5/5 passed (sampling, batch allocation, arguments, allocation and RPC). Synthetic Qwen4Exp F16/Q8 rollback, split replay and shared-sequence cases passed with strict realloc diagnostics; relocation passed with ordinary allocation. The relocation test grows from a 3-token replay to a 20-token prefill, so the strict diagnostic rejects that expected growth in both native and fallback modes. Native text/spatial/text graph transitions passed on a synthetic model. Existing CPU QSA operators passed 20/20; an independent selection oracle passed 28 rows including holes, tails, ties and nonfinite scores.
- CUDA 12.9, GNU 13.3, Release sm86 with all Flash Attention quantization variants and RPC: server, bench, perplexity and eight test/RPC targets built. Final QSA/top-k/MoE operators passed 439/439, retained upstream asymmetric FA cases 106/106, and lower-head Q8 tail cases 6/6. Sampling, 34 allocator cases, argument parsing and RPC multi-server passed. The complete source manifest verifies 3697 tracked files; 22 binary hashes identify the tested libraries and executables.
- CUDA attention validation on an earlier frozen build passed 96 output cases, 17 memcheck cases and 6 racecheck cases. Shared FA source hashes were unchanged in the final candidate; final QSA kernels were rerun after their later changes. Final expert multiplication memcheck passed 16/16 selected cases with zero errors. Small batches can use MMVQ; this does not assert all 16 invoked MMQ.
- CUDA, HIP gfx906 and Vulkan RADV VEGA20 each passed 1024 cross-thread event waits, 128 delayed waits and four-lane workspace tests covering private boundaries, views, reverse submission, cancellation, reuse and growth. Separate HIP and Vulkan backend builds with RPC succeeded. HIP QSA operators passed 20/20.
- Real Flash-Next UD-Q4_K_XL on four RTX 3090s passed 2 workers/F16 KV, 4 workers/Q8_0 KV and 5 workers/Q8_0 KV. Prefill, all-output prefill and sequential decode replay had max_abs=0; uneven completion, exceptions, duplicate-sequence rejection and reuse passed. These are correctness tests with a 64-token internal microbatch and 4096-token context, not throughput tests.

### Final temporary-pool allocation fix

A generated-model test with 256 KV cells, pooling ratio 4 and 96 sequences sharing a three-token prefix exceeded the 64-block compact cache with only 99 physical cells. The uncached F32 pooled destination had no backend assignment and aborted in allocation. The final one-line FILL initialization fixes that branch and leaves the cached graph unchanged.

On ARM CPU, the 96 divergent logits rows matched the same inputs in a larger cached context exactly; returning to cached mode after seq_keep also matched exactly. Independent one-sequence recomputation gave max_abs 1.19209e-7 and NMSE 1.32709e-13. A scheduler-level check of two pooling chunks, 8192+1 blocks, matched all output rows exactly and executed one initialization before the copies. This last check is not a full 8193-block model inference.

After the final one-line fix, all 11 CUDA build targets rebuilt. CUDA F16/Q8 overcapacity checks matched all 96 rows against the larger cached context exactly; the return-to-cached comparison against independent recomputation had max_abs at most 2.23517e-8. The focused Q8 overcapacity Compute Sanitizer memcheck passed with zero errors. The final CUDA build also reran the real Flash-Next five-worker Q8 pipeline lifecycle test: prefill, all-output prefill and replay again had max_abs=0, including failure recovery and reuse.

### Dense target and MTP integration

Qwen3.8-27B UD-Q8_K_L with mtp-Qwen3.8-27B-Q4_0 on two RTX 3090s, two CPU workers, unified KV, main Q8_0/draft Q4_0, compact masks in both contexts, context 65536, microbatch 512 and draft maximum 2. Existing pipeline and pipeline-mtp tests passed; prefill/replay and masked/unmasked hidden-row/logit ordering comparisons had max_abs=0.

The real server passed greedy/probabilistic draft modes at temperatures 0 and 1, with 1/2/4 requests and two repeats. Every mode activated two MTP lanes, exercised positive draft acceptance and rejection, and passed exact grammar output, arrival during streaming, cancellation while another request stayed active and canceled-slot reuse. Each owned server exited cleanly.

Some parallel seeded repeats differed in slot 0 at 2/4 requests. The harness correctly retained an exit-2 repeatability finding despite passing functional checks. A matched greedy/temperature-0 reproduction was stable on the old fork but also varied on unmodified new upstream, without fork workers or compact masks. This establishes that the observed non-repeatability is not unique to the worker adaptation; it does not identify its numerical or scheduling cause. Full concurrent-response identity is not a guarantee of this validation. The separate fixed-input row/logit tests above remain bitwise equal.

### Native QSA speed and memory

Same Flash-Next GGUF shards, four RTX 3090 UUIDs, CUDA 12.9 sm86, Q8_0 KV, split 23,24,23,26, context 131072, four slots, batch 2048, microbatch 512 and eight CPU threads. No UM, vision or CPU pipeline workers. Two repeats after warmup per case, one separate server process per implementation. These are whole-implementation comparisons, not isolation of one kernel or an alternated multi-launch statistical study. Prompt token hashes and loaded libraries were verified.

| Prompt tokens | Old fork prefill | New upstream prefill | Rebased fork prefill | Old fork decode | New upstream decode | Rebased fork decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 1040.66 | 1171.81 | 1095.28 | 62.52 | 58.65 | 63.13 |
| 32768 | 943.77 | 997.38 | 1025.40 | 59.75 | 49.70 | 59.87 |
| 65536 | 902.86 | 978.44 | 984.44 | 59.19 | 41.61 | 59.08 |

Rates are server tokens/s for fresh prefill and 128 generated tokens. Cached concurrent decode uses 8192 cached tokens plus one input token; its aggregate client-wall rates over two repeats were 86.68-87.64 tokens/s for two requests and 114.80-115.03 for four. The old fork gave 81.50-86.09 and 114.74-114.77; new upstream gave 81.46-82.06 and 105.67-106.02. CPU-worker throughput was not measured in this comparison.

At context 131072, rebased CUDA compute reservations were 251.54/371.54/371.54/371.54 MiB, compared with old fork 250.79/370.79/370.79/370.79 and upstream 1752.59/1872.59/1872.59/1872.60. CUDA_Host was 71.32/29.80/1110.58 MiB respectively. All used four scheduler copies. Largest sampled device usage was 21630/21526/23150 MiB.

At context 500000, the rebased fork started without UM: compute 305.05/425.05/425.05/425.05 MiB, CUDA_Host 113.57 MiB and largest sampled device usage 23470 MiB. Old fork reservations were 304.30/424.30/424.30/424.30 with peak 23104 MiB. Upstream failed its 6210.26 MiB allocation and its 3090.91 MiB fallback. This checks startup only; it is not a 500K-token inference test. The retained upstream packed raw/pooled indexer allocation costs an extra 96 MiB per GPU at 131072 and 366.375 MiB at the padded 500K context. The compact F32 native cache remains separate. Peak sampling interval was 200 ms.

### Native QSA quality

Holmes, context 8192, eight chunks, 32760 scored targets, same Q4 model and corpus, one run per case. These are the tool's rounded PPL and reported uncertainty, not byte comparisons of saved probabilities.

| KV | Old fork | New upstream | Rebased fork |
| --- | ---: | ---: | ---: |
| F16 | 1.0667 +/- 0.00329 | 1.0623 +/- 0.00303 | 1.0667 +/- 0.00329 |
| Q8_0 | 1.0618 +/- 0.00294 | 1.0635 +/- 0.00306 | 1.0618 +/- 0.00294 |

The rebased result matches the old fork at printed precision. The mixed upstream comparison does not establish a universal quality ordering or reference equivalence. Full-budget selection and intermediate precision remain deliberate native differences.

### Scope and retained evidence

P40 and V100 were not compiled or executed in this round. HIP/Vulkan have the operator/event/workspace coverage above, not full mixed-backend model or throughput coverage. No managed-memory overcommit, real vision quality, QSA-over-RPC or reference-model equivalence test ran. Spatial fallback was exercised only on a generated model; it can require dense-mask memory. The 500K case checks reservation, not a full prompt. Concurrent seeded MTP responses are not bitwise repeatable on the new upstream baseline.

Raw evidence includes pinned source archives/manifests, binary/model/corpus hashes, exact commands, build logs, operator/sanitizer results, model server requests/responses, timings, failure classifications and standalone regression harnesses. The broad CUDA snapshot source archive SHA256 is 29c1ede83ed52fe276e66d24f46832982a78ad2b7a3584f4f412b489e0a9abbd; final source differs only by the temporary-pool fix plus documentation. The Holmes corpus SHA256 is 585fca71a3c254647c2ad7b1ef6d6f2b1d563bf917ff77be85e79fdb471089dc. Model hashes are in model-hashes.json and the dense/MTP summaries.

The raw validation bundle is `validation-artifacts.tar.gz` (533 files, 2025115 bytes), SHA256 `f24b987a43b3cc4316bc552d3ccd492b89632944b91215a3df0426ba9c487c95`. It is retained on the test host at `user@192.168.50.53:/home/user/fork-rebase-20261006/validation-artifacts.tar.gz` and locally at `/Users/tabolin/dev2/llama-patches/investigations/upstream-rebase-2026-10-06/validation-artifacts.tar.gz`. `artifact-manifest.json` inside the bundle lists individual file hashes. Start with `VALIDATION-RESULT.md`; initial comparison/audit sections remain dated historical evidence. The original server is left stopped.

## Previous upstream rebase: 2026-09-16

| Role | Commit |
| --- | --- |
| Pre-rebase fork | `0b3e7a9d434ade7c9398bbe10ff04030e287a0f6` |
| Previous upstream base | `661643e43079a4ee6faab4c1895291767b67ea8d` |
| New upstream base | `37b53fd4545847188fdad29e38ba57875efc8228` |
| Tested rebased fork | `9e4f85897986f7d146d8c770b8c2b9c9f2f48bbc` |
| Subsequent RPC-only fix | `b54151e974f7bb2e6ac0b88bb80d92fb15af401c` |

The rebase incorporated 46 upstream commits without textual conflicts, retaining 20 local non-merge commits and 14 merge commits. `range-diff` showed unchanged patches apart from upstream context near QSA dispatch; the resulting tree matched an independent `merge-tree` result. This describes that rebase only, not a guarantee for a future one. PR #14 was rebased separately and was not included in master.

Historical build and regression results:

- ARM64 CPU/Accelerate: server and relevant tests compiled; sampling passed. Optional PR #14 compiled separately.
- Linux x86-64: CUDA sm_61/sm_86, HIP gfx906 and Vulkan built together. The source archive matched a manifest of all 3597 tracked files.
- 941 backend cases passed across HC operations, QSA selection/top-k/MoE on CUDA/HIP, and compact attention on CUDA.
- CUDA and Vulkan event tests passed 1024 cross-thread waits and 128 delayed waits per backend. Workspace checks covered four lanes, private boundaries, views, reverse submission, cancellation, reuse and growth.
- Qwen3.8-27B UD-Q8_K_L on two 3090s passed target pipeline, MTP row routing, failure/drain/reuse and recurrent relocation checks. Compared replay outputs were bitwise equal in these runs. Only the relocation mode of the recurrent test ran.
- Qwen4Exp Flash-Next UD-Q4_K_XL on four 3090s plus gfx906 passed two workers/F16 KV and four workers/Q8_0 KV. Prefill, all-output prefill and decode replay had `max_abs=0`; uneven completion, duplicate-sequence rejection, exceptions and reuse passed.
- The real Qwen3.8 target with a Q4_0 MTP head passed 1/2/4-request smoke cases, exact grammar output, arrival during streaming, cancellation with another request active and canceled-slot reuse.

P40 was compiled but not selected for model execution. No V100 run, managed-memory overcommit, vision validation or broad PPL comparison was part of this rebase suite. The original host was `user@192.168.50.53`, with sources/build under `/home/user/fork-rebase-20260916`. Local evidence: `investigations/upstream-rebase-2026-09-16/{VALIDATION.md,run-checks.py,run-server.py,checks.json,summary.json,range-diff.txt,validation-artifacts.tar.gz}`. These paths identify archived evidence, not portable dependencies.

## Express throughput check after the rebase

The candidate is `9e4f858`, with all loaded llama/ggml libraries from its build. Historical Qwen4Exp baseline: `84176694f796156b0bc2b1ac9c20c2db7295d310`; historical MTP baseline: `842da29bd1fe7791d076c901371d84cbbae54dea`. These are previous fork feature-validation runs, not stock upstream or a fresh run of immediately preceding master. The comparison does not isolate individual upstream commits.

Rates are tokens/s. Decode is aggregate client-wall-time throughput. The original report includes repetitions, ranges and acceptance counts; small changes are not asserted to be significant.

| Model / workers | Workload | Historical fork mean | Rebased fork mean |
| --- | --- | ---: | ---: |
| Qwen3.8 Q8 + MTP Q4 / 2 | Prefill 8192 | 1496.32 | 1510.85 |
| Qwen3.8 Q8 + MTP Q4 / 2 | Decode 1 request | 38.13 | 38.25 |
| Qwen3.8 Q8 + MTP Q4 / 2 | Decode 2 requests | 65.08 | 65.05 |
| Qwen3.8 Q8 + MTP Q4 / 2 | Decode 4 requests | 99.04 | 99.51 |
| Qwen4Exp Q4 / 2 | Prefill 10000 | 1336.10 | 1383.04 |
| Qwen4Exp Q4 / 2 | Decode 1 / 2 / 4 requests | 45.85 / 75.95 / 107.79 | 48.38 / 79.59 / 116.45 |
| Qwen4Exp Q4 / 4 | Prefill 10000 | 1708.71 | 1759.17 |
| Qwen4Exp Q4 / 4 | Decode 1 / 2 / 4 requests | 45.55 / 71.73 / 127.88 | 47.47 / 79.89 / 133.14 |

Qwen4Exp used four 3090s plus gfx906, native QSA, F16 KV, `-c 65536 -np 4 -b 2048 -ub 512 -t 8 -tb 8 -ts 23,24,24,24,12`; decode used a cached 2048-token prefix plus one token and 128 generated tokens. Three repeats after warmup per worker count.

MTP used two 3090s, target split `25,23`, draft on CUDA1, target Q8_0/draft Q4_0 KV, compact masks, two workers, `-c 65536 -np 4 -b 2048 -ub 512`, eight threads and draft limit two. Decode refreshed four distinct 8191-token prefixes, appended one token, and generated 256 tokens per request. Two new server runs were compared with three historical runs.

No UM, vision, P40 execution or overlapping benchmark jobs. Single-request MTP token IDs matched the recorded baseline. Long-context decode at 32K-128K and Vulkan performance were not measured. Evidence: `investigations/upstream-rebase-2026-09-16/performance/{PERFORMANCE.md,comparison.json,commands.json,baseline-manifest.json}` plus raw logs and runners in that directory.

## QSA quality regression to retain

The final native token-selection algorithm fills the full budget including the partial boundary block and tail. The final ordering implementation has byte-identical saved probabilities to the corrected-budget implementation for the measured Holmes runs. Use those results rather than earlier underfilled-budget tables.

Holmes, context 8192, eight chunks, 32760 scored targets, the same Q4 model, `-b 2048 -ub 512`, and matched main KV types:

| Main KV | Stock QSA PPL | Final native QSA PPL |
| --- | ---: | ---: |
| F16 | 1.067211556 | 1.067399375 |
| Q8_0 | 1.076745551 | 1.068463650 |

This is sampled evidence, not proof of losslessness or universal improvement. Native Q8 keeps raw indexer K in F16, so matching the main KV format does not make all intermediate arithmetic identical. FP32 score calculation, ties and attention reduction can change MoE routing.

The token-budget and ordering validations covered CPU/CUDA/HIP, tail lengths, holes, shuffled physical members, shared/restored prefixes, large block capacity and CUDA memcheck/racecheck. A Q8 device sequence-copy failure reproduced on the baseline and was not fixed in that work; the final ordering change did not retest it. Do not mark it resolved without a current reproduction. Evidence: `investigations/fork-prs-2026-09-15/pr08-budget-fix/README.md` and `pr08-budget-order/README.md` with their saved probabilities and manifests.

## RPC fix after the rebase

At `b54151e`, RPC count/version assertions were corrected for QSA. `GGML_RPC=ON` builds and `test-rpc-multi-server` passed on macOS ARM64 and Linux x86-64 with CPU RPC servers. The two-line header fix did not rerun the complete GPU suite or native QSA inference over RPC. It was committed and published after the rebase; the archived pre-commit report's final status line is historical. Evidence: `investigations/rpc-qsa-build-2026-09-16/VALIDATION.md` and its build/test logs.

## Recording the next validation

Append the exact old/new fork and upstream revisions, change IDs, toolchain/device/model/corpus hashes, commands, passed and skipped cases, peak-memory and throughput results, and an accessible raw-artifact location. Keep previous records dated. When an upstream replacement retires a local patch, include the regression evidence that justified replacement and update CHANGES.md in the same change.

## Compact CUDA MoE work lists, 2026-10-06 (F18)

Baseline: fork `d43f0dfcd517b92ac7bb115c7936bdeff115aa83`, upstream `f0c41e0168dfd4b5ef72b21d1a311b24cc7a894a`. Candidate: the F18 changes following that fork revision. CUDA 12.9.86, Release, sm_86, CPU/CUDA/RPC with static backend registration, CUDA Graph support and all Flash Attention KV types compiled. Runs used four RTX 3090 GPUs on `user@192.168.50.53`; P40 and gfx906 were excluded. No unified memory, MTP or vision tower. The earlier fork changes, including native QSA and pipeline workers, are present in both sides. This comparison isolates F18 and is not a comparison of the whole fork with stock upstream.

Correctness checks on the candidate:

- Existing `test-backend-ops` infrastructure: 98/98 Q4_K/Q5_1/Q8_0 `MUL_MAT_ID` cases passed against CPU. This includes 30 explicit routing cases with sparse, concentrated, skewed and boundary-length expert groups, broadcast and per-expert inputs, 384 output rows and strided expert weights.
- Compute Sanitizer 2025.2.1: memcheck passed all 30 routing cases with zero errors; racecheck passed 18 skewed/boundary cases with zero errors or warnings.
- A saved standalone harness reused the same graph while alternating all-small, all-large and mixed expert distributions for each supported weight type. All 54 comparisons passed against CPU, maximum NMSE `7.17673833e-5` (threshold `5e-4`). Nsight Systems recorded 51 `cudaGraphLaunch` calls, confirming actual captured replay rather than only uncaptured submission.
- CUDA source and the three changed type instantiations compiled for both sm_61 and sm_70. This is compile coverage, not execution coverage; F18 dispatch is disabled on those devices. HIP/MUSA/Vulkan were not rebuilt for F18.
- Independent source review covered list capacities, padding, strides, shared-memory barriers and stream-local pool lifetime. No new public API or test executable was added to the repository.

Model: `unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL`, HF snapshot `38bb39ee97821de2c9009abb7e93950eec396e66`. All expert weights fit on the four GPUs; the model's PLE data remained CPU-mapped. Native QSA was enabled, main KV Q8_0, tensor split `23,24,23,26`, eight CPU threads and no fit adjustment. Holmes corpus raw SHA-256: `585fca71a3c254647c2ad7b1ef6d6f2b1d563bf917ff77be85e79fdb471089dc`.

Fresh-prompt throughput, three repeats after a 1024-token warmup. Both sides used the same final binary with `GGML_CUDA_MMQ_MOE_COMPACT=0/1`; the timed requests had no profiler collection. Prompt caching was disabled, one request at a time, four server slots. Context size was 16384 for the 8192-token prompt and 33280 for the 32768-token prompt. CPU workers and worker microbatch size were explicitly selected through `LLAMA_PIPELINE_WORKERS` and `LLAMA_PIPELINE_UBATCH`.

| Workers | Prompt | Batch / microbatch | Original MMQ tok/s | Compact MMQ tok/s | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8192 | 2048 / 512 | 1178.20 | 1539.30 | +30.65% |
| 4 | 32768 | 8192 / 512 | 3016.17 | 3872.87 | +28.40% |
| 4 | 32768 | 8192 / 2048 | 2715.44 | 3120.19 | +14.91% |

With no workers, 128-token greedy decode after the 8192-token prompt measured 63.05 versus 63.87 tok/s. No decode speedup is attributed to F18; single-token execution uses the original path. All 128 generated token IDs matched across both modes and all repeats. A larger microbatch was slower for both paths in this setup; F18 does not remove pipeline balance and attention costs.

Separate Nsight Systems captures of the 32768-token, four-worker, 512-token case measured 15.145 versus 7.457 cumulative GPU seconds in Q4_K/Q5_1 expert matmul kernels. Building compact work lists cost 0.034 GPU seconds. These are summed kernel durations across GPUs, not elapsed request times. The original profile used the final binary; the compact profile preceded removal of unused J=32/64 instantiations and used the same J=16/128 executed kernels. Final throughput and correctness checks used the reduced set of instantiations. Build logs and final source and binary manifests accompany the captures.

Holmes PPL used eight 8192-token chunks with `-b 2048 -ub 512`, no workers, and `GGML_CUDA_MMQ_MOE_COMPACT=0/1`. Both baseline and candidate reported `1.0618 +/- 0.00294`. Values match at the executable's printed precision; this is not a claim of bitwise logits or universal quality equivalence.

Raw logs, exact commands, profiler exports, comparison scripts and source/binary manifests are archived under `investigations/moe-compact-2026-10-06` in the local project and `/home/user/moe-compact-20261006` on the test host. These are evidence locations, not build dependencies. The host launcher in that directory sets library search paths and selects the four 3090 GPUs by UUID unless `CUDA_VISIBLE_DEVICES` is already set. Its sm_86 build is not an all-backend replacement for an existing deployment.

## CUDA MoE MMQ SwiGLU fusion, 2026-10-06 (F19)

Baseline: `9da5fa77d2699008fc0fdeb9dfd10d3ea500995c` (F18). Candidate: F19 changes following that commit, with exact source hashes in `tested-source.json` and source/binary hashes in `verification.json`. The model, corpus, four RTX 3090 GPUs, CUDA 12.9 sm_86 toolchain, Q8_0 KV, native QSA, thread counts and tensor split match F18 above. No UM, MTP or vision tower. This isolates fusion relative to compact MMQ, not the whole fork relative to upstream.

Both sides use the same candidate binary with `GGML_CUDA_MMQ_MOE_COMPACT=1`; only `GGML_CUDA_MMQ_MOE_SWIGLU=0/1` changes. The following are means of three fresh-prompt requests after a 1024-token warmup, without profiler collection during the timed requests. The zero-worker pair was repeated after test compilation finished to avoid CPU interference. Published numbers use that final pair.

| Workers | Prompt | Batch / microbatch | Compact MMQ tok/s | Compact MMQ + fusion tok/s | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8192 | 2048 / 512 | 1533.56 | 1612.92 | +5.18% |
| 4 | 32768 | 8192 / 512 | 3892.16 | 4040.75 | +3.82% |

The zero-worker 128-token decode measured 62.97 versus 62.57 tok/s; this path is unchanged and no decode gain is claimed. All 128 output token IDs matched across modes and repeats. The single output token in each four-worker run also matched. Holmes PPL over eight 8192-token chunks was `1.0618 +/- 0.00294` in both modes, at the executable's printed precision. This sampled agreement does not establish universal bitwise equivalence.

Validation:

- `test-backend-ops`: 14/14 routed gate/up/GLU cases passed against CPU, covering broadcast and per-expert inputs, sparse/hot/concentrated/boundary routing, unsupported row and batch sizes, strided views, extra gate consumers, GEGLU and clamped SwiGLU fallback. Plain fused SwiGLU retains the `5e-4` NMSE threshold. The unsupported clamped case uses the existing quantized-GLU suite's `5e-3` threshold: the tighter plain-matmul threshold failed even with fusion disabled (`0.000650604` NMSE).
- Existing Q4_K/Q5_1/Q8_0 `MUL_MAT_ID` cases: 98/98 passed. Memcheck passed the 12 routing/shape/consumer cases with zero errors; racecheck passed eight skewed/boundary/fallback cases with zero hazards. These sanitizer runs preceded the two additional unsupported-GLU cases.
- Changing-routing graph replay passed 18 Q4_K comparisons through fusion and 36 Q5_1/Q8_0 fallback comparisons against CPU.
- Scheduler-allocated graphs with a computed activation, reused graph storage and batch sizes 256, 129, 512 and 64 passed 72 comparisons. A second fixture with separately allocated weight buffers and strided 384-row views of 512-row expert matrices passed another 72. Each Nsight capture recorded 108 fused kernel executions; the 64-token shape retained the fallback. These fixtures refill scheduler-owned inputs before each invocation, as their storage can be reused after the graph consumes them.
- A real four-worker, 32768-token profile recorded 5888 fused kernel launches. Their cumulative GPU duration was 3.869 seconds; this is a sum across GPUs, not request latency. The reported private/shared compute reservations were identical with fusion off/on (CUDA0: 40.00/192.04 MiB; CUDA1-2: 40.00/192.12 MiB; CUDA3: 3.98/192.12 MiB).
- The fused J=16 and J=64 kernels used 191 and 255 registers per thread with zero stack bytes in the sm_86 object. The selected width avoids spilling; high register use remains a throughput constraint.
- CUDA source and Q4_K/Q5_1/Q8_0 MMQ instantiations compiled for sm_61 and sm_70. F19 dispatch stays disabled there. HIP/MUSA/Vulkan and other NVIDIA architectures were not executed for F19.
- Independent source review found no blocking correctness issue. Its scheduler-lifetime and unsupported-GLU coverage requests were addressed by the tests above. `git diff --check` and ASCII checks of added C++/CUDA lines passed.

Evidence is in local `investigations/moe-fusion-2026-10-06` and remote `/home/user/moe-fusion-20261006`: operator/sanitizer logs, scheduler fixtures, profiles, matched performance and PPL JSON, patch and manifests. The remote `bin/` package was verified byte-identical to all 14 measured build binaries. `/home/user/moe-fusion-20261006/llama-server` sets its library search path and defaults to the four 3090 UUIDs; the two optimization switches must still be set explicitly. This is a CPU/CUDA/RPC sm_86 validation build, not an all-backend deployment build. The server was left stopped.

## Direct Q8 output for MoE SwiGLU, 2026-10-07 (F19 refinement)

Baseline: F19 with its FP32 SwiGLU output. Candidate: the direct-Q8 extension following `9da5fa77d`, with the unchanged F18 and F19 switches enabled. `GGML_CUDA_MMQ_MOE_Q8=0/1` selects the comparison in the same binary. The frozen earlier F19 build is `/home/user/moe-fusion-20261006/bin`; exact candidate source and binary hashes accompany the evidence below. Hardware, model snapshot, Holmes corpus, native QSA, Q8_0 KV, tensor split and CPU settings match the preceding F19 record. This measures the additional output-quantization fusion, not stock upstream versus the complete fork.

The final direct-Q8 path uses J=16/32 for gate/up, while FP32-output F19 stays at J=16/64 and down stays at J=16/128. The direct kernels use 177/249 registers per thread and zero stack bytes on sm_86. A J=64 direct-output trial used 56 stack bytes per thread; it is not the selected path.

Correctness and dispatch:

- `test-backend-ops`: 15/15 full gate/up/SwiGLU/down graphs passed against CPU at the unchanged `5e-4` NMSE limit. Cases include Q5_1 and Q8_0 down, broadcast and per-slot input, sparse/hot/concentrated/boundary routing, hidden widths 128/384/640, an extra SwiGLU consumer, small-batch fallback, zero input and unsupported Q4_0 down fallback. The 128-row cases exercise fallback; direct packing is exercised at 384/640 rows. Existing F19 tests passed 14/14 and existing Q4_K/Q5_1/Q8_0 MMID tests passed 98/98.
- Memcheck passed 15/15 cases with zero errors, including a repeat on the final packaged binary after the NaN guard correction below. Racecheck passed 11/11 skewed/boundary/fallback cases with zero hazards before that two-expression correction; allocation, indexing and synchronization did not change afterward.
- A changing-routing scheduler fixture passed 72 CPU comparisons at batches 256, 129, 512 and 64. Nsight recorded 68 CUDA Graph launches and 108 direct-Q8 kernel executions. Separate DS4/D4 fixtures included zero-input replay and compared all 4,428,288 output floats per layout with the FP32-output path: both were byte-identical. Both layout comparisons and all 15 operator cases were repeated after the final numerical correction.
- An all-NaN fixture exposed a D4 discrepancy: the initial zero guard selected a zero scale for NaN. The final guard handles only `amax == 0`; all other values follow the original quantizer's division. The original and corrected paths each preserve 33024/33024 NaN outputs. The failing and passing tests are archived as `nan-red-*` and `nan-green-*`.
- Holmes PPL over eight 8192-token chunks is `1.0618 +/- 0.00294` with direct Q8 disabled and enabled, including a repeated enabled run after the final correction. The initial 8192-token performance runs produced identical 128-token greedy outputs across both modes and all repeats. These are sampled comparisons, not a universal quality guarantee.
- CUDA sm_61/sm_70 compilation passed for the direct-output code, with runtime dispatch disabled there. That compilation preceded only the zero-guard expression change. HIP/MUSA/Vulkan and other NVIDIA devices were not executed for this refinement. Fresh source review covered layout, padding, consumers, allocation, replay and the final tile-width adjustment; the NaN correction was also independently reviewed.

A 32768-token, four-worker Nsight capture removed 2944 routing-helper launches and 2944 separate activation-quantization launches. Relative to FP32-output fusion, cumulative routing-helper duration fell from 0.213 to 0.109 GPU seconds, and activation quantization from 0.568 to 0.499 GPU seconds. Gate/up/SwiGLU cumulative duration was 3.866 seconds for FP32 output and 3.796 seconds for direct Q8 with J=32. These are summed durations across four GPUs, not elapsed request time. The eliminated helper work accounts for less than 1% of total cumulative kernel time in this workload. Compute reservations in the measured four-worker runs remained unchanged: CUDA0 private/shared 40.00/192.04 MiB, CUDA1-2 40.00/192.12 MiB, CUDA3 3.98/192.12 MiB. This traffic optimization does not remove the intermediate tensors from the graph allocator, and no total VRAM reduction is claimed.

Final throughput used the corrected binary, a full 32768-token warmup and five fresh-prompt requests per mode, without profiler collection. Direct Q8 ran first, then FP32 output. Both used four workers, four server slots, one request at a time, no prompt caching, context 33280 and batch/microbatch 8192/512. The table reports arithmetic means; the raw records are `verified-q0-w4.json` and `verified-q1-w4.json`.

| Workers | Prompt | FP32-output fusion tok/s | Direct-Q8 fusion tok/s | Change |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 32768 | 4056.87 | 4102.34 | +1.12% |

The baseline range was 4044.63-4067.22 tok/s; direct Q8 ranged from 4094.17 to 4107.69 tok/s. Earlier three-repeat, 8192-token runs with a shorter warmup measured 1605.79 versus 1622.03 tok/s (+1.01%) without workers. Those runs preceded the NaN correction and are supplemental evidence. Single-token decode retains the existing path; no decode acceleration is claimed. The measured prefill improvement is small and specific to this workload, so the new switch remains off by default.

Evidence is archived in local `investigations/moe-direct-q8-2026-10-06` and remote `/home/user/moe-direct-q8-20261006`, including commands, profiles, full test logs, source patch and source/binary manifests. The remote launcher `/home/user/moe-direct-q8-20261006/llama-server` sets its own library search path and defaults to the four 3090 UUIDs. Enable the path with `GGML_CUDA_MMQ_MOE_COMPACT=1 GGML_CUDA_MMQ_MOE_SWIGLU=1 GGML_CUDA_MMQ_MOE_Q8=1`. This is a CPU/CUDA/RPC sm_86 validation build, not an all-backend deployment. The server was left stopped.
