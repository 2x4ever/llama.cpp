# Fork validation record

This record separates current validation from historical evidence. Reproducible repository test entry points are in [FORK-MAINTENANCE.md](../../FORK-MAINTENANCE.md). Raw artifacts live outside the source tree; exact locations and hashes are recorded per validation date.

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
