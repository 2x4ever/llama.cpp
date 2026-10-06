# Maintaining the 2x4ever llama.cpp fork

Start here when changing this fork or rebasing it onto `ggml-org/llama.cpp`. This guide describes the maintained fork, not the earlier experimental `qsa-pipeline` checkout.

- [Fork changes](docs/fork/CHANGES.md) is the maintained inventory of differences from upstream, including source locations, invariants and replacement criteria.
- [Validation record](docs/fork/VALIDATION.md) records the last measured revisions, results and known gaps.
- [Server options](tools/server/README.md#cpu-pipeline-workers-experimental) describe pipeline activation and fallback behavior.

## Source of truth

Fork: <https://github.com/2x4ever/llama.cpp>. Upstream: <https://github.com/ggml-org/llama.cpp>.

The code baseline audited for this guide is fork `c94c992b9b4a7b40ba0b22d9db19d437844d16c9`, based on upstream `f0c41e0168dfd4b5ef72b21d1a311b24cc7a894a`. The 2026-10-06 update replayed the published fork `b54151e974f7bb2e6ac0b88bb80d92fb15af401c` and retained Native QSA after comparing it with the new upstream path. See the dated validation record for executed checks and remaining gaps. Later commits must update the inventory.

The optional draft batching/concurrency branch `codex/mtp-draft-concurrency` at `f6f2f90c93132c1ba7cfc4a65213a033cf85699b` is outside this baseline (PR #14). Do not merge it as part of a rebase. Its throughput benefit was not established. The older single-scheduler async executor, explicit ARM NEON sampler path and gfx906-specific MMQ tuning are also not part of the published feature set.

Legacy `investigations/` reports and launchers are historical evidence. They can describe superseded implementations and can contain machine-specific paths. They are not required to understand or build the fork; do not reapply old patches to the current tree. Do not infer that an untracked file is disposable.

## Documentation contract

Every change relative to upstream must have a separate, identifiable entry in [CHANGES.md](docs/fork/CHANGES.md), updated in the same commit as the implementation. An entry records the problem, current behavior, activation and limits, affected code, dependencies, tests and upstream status. Update existing entries for refinements; give independent changes their own IDs. Include build, protocol and documentation-only differences, not only features.

During every upstream update, review every active entry against the exact new upstream revision. Similar names or a conflict-free rebase are not proof of equivalent behavior. Check supported models, backends, types, ownership and failure paths, then run the relevant regressions.

When upstream covers a local change, replace the local implementation where appropriate and mark the entry `upstream`, linking the upstream commit and the verification evidence. If coverage is partial, mark it `partial` and describe the remaining delta. If the feature is removed without a replacement, mark it `retired` and explain why. Remove obsolete activation instructions and performance claims. Keep a short historical entry so a later rebase does not restore the duplicate patch. Unverified equivalents remain `active` with an explicit open question.

## Rebase procedure

Use a clean checkout of the published fork. Save dirty or untracked work separately and leave experimental checkouts alone. Do not reset a working tree to make this procedure run.

Confirm remote URLs, fetch, pin both tips, and create a backup before replaying commits. The old upstream base below must match CHANGES.md; replace it when that document records a newer base.

```sh
git remote -v
git status --short --branch
git fetch origin
git fetch upstream
FORK_OLD=$(git rev-parse origin/master)
FORK_OLD_BASE=f0c41e0168dfd4b5ef72b21d1a311b24cc7a894a
FORK_NEW_UPSTREAM=$(git rev-parse upstream/master)
FORK_REBASE_DATE=$(date +%Y%m%d)
git merge-base --is-ancestor "$FORK_OLD_BASE" "$FORK_OLD"
git branch "codex/backup-before-rebase-$FORK_REBASE_DATE" "$FORK_OLD"
git worktree add -b "codex/rebase-$FORK_REBASE_DATE" ../llama-rebase "$FORK_OLD"
cd ../llama-rebase
git rebase --rebase-merges --onto "$FORK_NEW_UPSTREAM" "$FORK_OLD_BASE"
```

If `upstream` is missing, add the URL above first. Choose unused branch/worktree names if these already exist. Stop if the recorded base is not an ancestor. Review upstream history from that base to the new tip before resolving conflicts; do not automatically choose all of `ours` or `theirs`. A rebase reverses the intuitive meaning of those sides.

Review the [invariants](docs/fork/CHANGES.md) for each conflicting subsystem. Explicitly review all four boundaries even when Git reports no conflicts: operator IDs/RPC, allocation versus kernel dispatch, shared-memory lifetime versus GPU completion, and recurrent state versus MTP rollback. Removing a local commit needs a documented upstream replacement or an explicit retirement decision.

```sh
git range-diff "$FORK_OLD_BASE..$FORK_OLD" "$FORK_NEW_UPSTREAM..HEAD"
git log --oneline --no-merges "$FORK_NEW_UPSTREAM..HEAD"
git diff --stat "$FORK_NEW_UPSTREAM...HEAD"
git diff --check
```

`range-diff` reviews the replayed patches; also inspect the merge topology and final tree. Counts need not stay constant when upstream absorbs a change. Keep the backup and record dropped/adapted commits. Update the baseline in this guide and CHANGES.md, and record the exact tested candidate in VALIDATION.md. Rebase unmerged feature branches separately, after validating the main branch. Publication is a separate action; this procedure does not push or merge anything.

For Native QSA storage changes, also check raw-only allocation, text/spatial/text fallback, saved-state restoration and the lifetime of shared worker graphs. Text-only native execution must not allocate the separate upstream fallback pooled buffer. Once used, that buffer remains alive with the memory object.

## Build and loaded-library checks

Use a new build directory so incompatible cached toolchain options cannot silently survive a rebase. Build all backends that the resulting binary will advertise. Include RPC in compile checks whenever ggml operation IDs change.

For a CUDA validation build on a 3090 host:

```sh
cmake -S . -B build-fork-check -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_RPC=ON -DGGML_BACKEND_DL=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_APP=OFF
cmake --build build-fork-check -j 8 --target llama-server llama-perplexity llama-bench \
    test-backend-ops test-thread-safety test-recurrent-state-rollback \
    test-sampling test-alloc test-arg-parser test-rpc-multi-server ggml-rpc-server
```

The historical mixed-backend build additionally used CUDA architectures `61;86`, `GGML_HIP=ON`, `CMAKE_HIP_ARCHITECTURES=gfx906`, `GGML_HIP_NO_VMM=ON`, `GGML_HIP_RCCL=OFF`, `GGML_VULKAN=ON` and `GGML_BACKEND_DL=ON`. With dynamic backend loading enabled, `test-rpc-multi-server` is not built; use a separate `GGML_BACKEND_DL=OFF` build for that test. Select one consistent ROCm compiler/header installation; the last host needed explicit `CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++` and `CMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm`. Source the installed Vulkan SDK's `setup-env.sh` before configuring when needed. The original host used `/home/user/1.4.328.1/setup-env.sh`. Adjust paths and CPU instruction flags to the actual host.

For V100 compilation include architecture `70` with a toolchain that supports it. Compilation does not establish native QSA runtime performance or compact-mask support on that card. See the feature limits in CHANGES.md.

Check `CMakeCache.txt`, compiler versions and `llama-server --list-devices`. Record GPU UUIDs because device numbering can change. For dynamic builds, set `LD_LIBRARY_PATH` to the candidate build's `bin` directory and needed vendor libraries, and inspect `/proc/<pid>/maps` or `LD_DEBUG=libs` on Linux to verify every loaded llama/ggml library. Old `.so` files can invalidate an otherwise correct A/B comparison.

`-DGGML_SCHED_MAX_COPIES=4` is a CMake option, already defaulting to four. `LLAMA_SCHED_MAX_COPIES=4` is not a runtime setting. Scheduler input copies and CPU pipeline workers are different mechanisms.

## Reproducible regression entry points

Commands below use existing repository tests. Run from the candidate root after setting `FORK_BIN` to the candidate `bin` directory. Clear inherited feature overrides first, then enable only those required by a case. Keep logs, exit codes, exact commits and model hashes. A skipped test or zero selected cases is not a pass.

```sh
FORK_BIN="$PWD/build-fork-check/bin"
"$FORK_BIN/test-sampling"
"$FORK_BIN/test-alloc"
"$FORK_BIN/test-arg-parser"
ctest --test-dir "$FORK_BIN/.." -R '^test-rpc-multi-server$' --output-on-failure
"$FORK_BIN/test-backend-ops" test -b CUDA0 -o QSA_SELECT,TOPK_QSA,TOPK_MOE
"$FORK_BIN/test-backend-ops" test -b CUDA0 -o QSA_ATTN
"$FORK_BIN/test-backend-ops" test -b CUDA0 -o FLASH_ATTN_EXT -p 'compact_mask=1'
"$FORK_BIN/test-backend-ops" events -b CUDA0
"$FORK_BIN/test-backend-ops" workspace -b CUDA0
```

CTest starts both local RPC servers and passes their endpoints to the client; do not invoke `test-rpc-multi-server` without endpoints. Repeat selection/indexed-attention tests on CPU and ROCm where available; repeat event/workspace tests on Vulkan. Run the relevant dense F16/Q8_0/Q4_0 Flash Attention cases as well as compact-mask cases. GPU memory safety checks are required when changing allocation sizes, index expansion or tile padding; use Compute Sanitizer on a supported CUDA host and the affected existing operator cases.

Set `FORK_DENSE_MODEL` to the exact target GGUF (historically Qwen3.8-27B UD-Q8_K_L) and `FORK_FLASH_MODEL` to the first shard of Qwen3.8-Flash-Next UD-Q4_K_XL. Paths, device lists and splits below are examples for the original hardware; do not download or substitute another quantization silently.

```sh
LLAMA_PIPELINE_WORKERS=0 LLAMA_FLASH_ATTN_COMPACT_MASK=1 \
    "$FORK_BIN/test-thread-safety" pipeline -m "$FORK_DENSE_MODEL" \
    -dev CUDA0,CUDA1 -ngl 99 -fit off -fa on -np 2 -ctk q8_0 -ctv q8_0
LLAMA_PIPELINE_WORKERS=0 LLAMA_FLASH_ATTN_COMPACT_MASK=1 \
    "$FORK_BIN/test-thread-safety" pipeline-mtp -m "$FORK_DENSE_MODEL" \
    -dev CUDA0,CUDA1 -ngl 99 -fit off -fa on -np 2 -ctk q8_0 -ctv q8_0
LLAMA_PIPELINE_WORKERS=0 \
    "$FORK_BIN/test-recurrent-state-rollback" relocation -m "$FORK_DENSE_MODEL" \
    -dev CUDA0,CUDA1 -ngl 99 -fit off -fa on -np 2 -ctk q8_0 -ctv q8_0
LLAMA_PIPELINE_WORKERS=0 LLAMA_QSA_NATIVE=1 \
    "$FORK_BIN/test-thread-safety" pipeline -m "$FORK_FLASH_MODEL" \
    -dev CUDA0,CUDA1,CUDA2,CUDA3,ROCm0 -ts 23,24,24,24,12 \
    -ngl 99 -fit off -fa on -np 4 -ctk q8_0 -ctv q8_0
```

Here `LLAMA_PIPELINE_WORKERS=0` prevents automatic setup in the initial context; the test itself enables workers. Its `-np` sets the tested worker count. It internally uses a 64-token microbatch and a 4096-token context; these are correctness tests, not throughput benchmarks. Repeat Qwen4Exp with F16 KV and 2/4/5 workers when changing scheduler, MoE or QSA code. `pipeline-mtp` tests target hidden-state/logit routing and failure handling; it does not replace an end-to-end test with a draft model. The recurrent test can skip non-recurrent models; check its output.

For MTP integration, run a real server with the target above and `mtp-Qwen3.8-27B-Q4_0.gguf`, `--spec-type draft-mtp --spec-draft-n-max 2`, target Q8_0 KV, draft Q4_0 KV, unified KV, compact masks and two workers. Check 1/2/4 requests, actual draft acceptance/rejection, grammar-constrained output, request arrival, cancellation while another request runs, and canceled-slot reuse. Verify `MTP pipeline lane ... active` appears for concurrent generation. Use loopback and a free port for tests; do not stop a production server implicitly.

For native QSA, verify both `QSA_SELECT` and `QSA_ATTN` in a scheduler debug trace (`GGML_SCHED_DEBUG=2` with a verbose build/run). `flash_attn = enabled` alone is insufficient. For workers, require `pipeline workers enabled`; the stock `pipeline parallelism enabled` line is not proof of our CPU workers. For compact masks, check main and MTP context logs separately and verify the selected case reaches the compact kernel.

## Quality, memory and throughput after rebase

1. Compare the candidate against the frozen pre-rebase fork to detect regressions. Compare separately against the new upstream if making a claim about fork benefit. Name every baseline by commit and feature settings; do not label an earlier experimental implementation as stock.
2. Keep weight/corpus/tokenizer hashes, KV type, prompt tokens, batch and microbatch sizes, context size, tensor split, device UUIDs, draft settings, thread counts and environment identical. Warm up, alternate independent server runs, exclude unrelated GPU jobs, and report repeats and ranges.
3. Measure fresh prefill and 1/2/4-request decode separately. Include 32K/64K prefixes when evaluating QSA scaling; short cached prompts cannot establish long-context behavior. Record whether throughput is per request or aggregate. Acceptance rate affects MTP comparisons.
4. Compare operator outputs and state save/restore, shared/separate prefixes and fragmented KV. For QSA changes, also compare matched PPL chunks/probabilities on F16 and Q8_0. FP32 scoring, tie selection and attention reduction can change MoE routing; passing one text or a short replay does not establish lossless quality.
5. Record startup reservation, private/shared scratch, actual peak VRAM and whether allocation fell back to fewer scheduler copies or another kernel. Repeat large-context reservations with UM disabled; managed-memory residency is not proof that unmanaged peak allocation fits.
6. In VALIDATION.md, separate tests run now from historical evidence, source inspection and skipped hardware. Missing models, V100, vision or Vulkan measurements remain explicit gaps. Do not promote a build-only result to runtime coverage.

## Passing this work to another maintainer

Provide this checkout, the inventory, the validation record, the candidate commit and exact model/toolchain locations. Keep raw logs and corpora outside the source tree, with an accessible artifact location and hashes in the new validation record. A clean clone contains the functional test entry points above; historical machine-specific scripts are optional evidence. An empty conflict list and a successful build alone do not complete a rebase.
