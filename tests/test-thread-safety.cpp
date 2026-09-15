// thread safety test
// - Loads a copy of the same model on each GPU, plus a copy on the CPU
// - Creates n_parallel (--parallel) contexts per model
// - Runs inference in parallel on each context

#include <array>
#include <thread>
#include <vector>
#include <atomic>
#include "llama.h"
#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "../src/llama-context.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

static void pipeline_require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static double pipeline_difference(const std::vector<float> & a, const std::vector<float> & b) {
    pipeline_require(a.size() == b.size(), "logit shape");
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        pipeline_require(std::isfinite(a[i]) && std::isfinite(b[i]), "finite logits");
        d = std::max(d, double(std::abs(a[i] - b[i])));
    }
    return d;
}
static int test_pipeline(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    common_init();
    common_params params;
    pipeline_require(common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON), "arguments");
    llama_backend_init();
    auto init = common_init_from_params(params, true);
    pipeline_require(init && init->model(), "model");
    auto * model = init->model();
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::vector<int> prefixes = {64, 128, 256, 384};
    const int seqs = 4;
    const int ubatch = 64;
    const int workers = 2;
    auto cp = common_context_params_to_llama(params);
    cp.n_ctx = 4096;
    cp.n_seq_max = seqs;
    cp.n_batch = std::max(1024, ubatch);
    cp.n_ubatch = ubatch;
    cp.n_rs_seq = 0;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.kv_unified = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context_ptr owner(llama_init_from_model(model, cp));
    pipeline_require(bool(owner), "context");
    std::string text;
    for (int i = 0; i < 1024; ++i) { text += "Pipeline validation paragraph " + std::to_string(i) + ": the river flows through the valley. Calculate 123 + 456 and explain the result.\n"; }
    auto tokens = common_tokenize(llama_model_get_vocab(model), text, true, false);
    auto token = [&](int seq, int pos) { return tokens[(size_t(seq)*4096 + pos)%tokens.size()]; };
    auto batch = llama_batch_init(cp.n_batch, 0, 1);
    auto prefill = [&] {
        std::vector<float> logits;
        for (int seq = 0; seq < seqs; ++seq) {
            for (int pos = 0; pos < prefixes[seq];) {
                const int count = std::min(ubatch, prefixes[seq] - pos);
                common_batch_clear(batch);
                for (int j = 0; j < count; ++j) { common_batch_add(batch, token(seq, pos + j), pos + j, {seq}, pos + j == prefixes[seq] - 1); }
                pipeline_require(llama_decode(owner.get(), batch) == 0, "prefill");
                pos += count;
            }
            const float * row = llama_get_logits_ith(owner.get(), -1);
            logits.insert(logits.end(), row, row + vocab);
        }
        return logits;
    };
    const auto reference = prefill();
    owner->set_pipeline(workers, ubatch);
    llama_memory_clear(llama_get_memory(owner.get()), true);
    const double prefill_error = pipeline_difference(reference, prefill());
    std::printf("Pipeline prefill: max_abs = %.9g\n", prefill_error);
    pipeline_require(prefill_error < 0.02, "pipeline prefill differs from synchronous decode");
    owner->set_pipeline(0, ubatch);
    llama_synchronize(owner.get());
    std::vector<std::vector<uint8_t>> states(seqs);
    for (int seq = 0; seq < seqs; ++seq) {
        states[seq].resize(llama_state_seq_get_size(owner.get(), seq));
        pipeline_require(llama_state_seq_get_data(owner.get(), states[seq].data(), states[seq].size(), seq) == states[seq].size(), "save");
    }
    const auto memory_size = llama_get_memory(owner.get())->memory_breakdown();
    struct execution {
        std::vector<int> ids, positions;
        std::vector<llama_token> tokens;
        std::vector<float> logits;
    };
    auto restore = [&] {
        llama_synchronize(owner.get());
        llama_memory_clear(llama_get_memory(owner.get()), true);
        for (int seq = 0; seq < seqs; ++seq) {
            pipeline_require(llama_state_seq_set_data(owner.get(), states[seq].data(), states[seq].size(), seq) == states[seq].size(), "restore");
        }
    };
    for (const std::string scenario : {"uneven", "exception", "duplicate", "reuse"}) {
        owner->set_pipeline(workers, ubatch);
        restore();
        struct state {
            int seqs, workers, vocab;
            std::vector<int> prefixes;
            std::vector<llama_batch> batches;
            std::vector<int> step, limit, last;
            std::vector<execution> trace;
            const std::vector<llama_token> * text;
            std::string scenario;
            int completions = 0;
        } data{seqs, workers, vocab, prefixes, {}, std::vector<int>(seqs, 0), {}, std::vector<int>(workers, -1), {}, &tokens, scenario};
        for (int i = 0; i < seqs; ++i) { data.limit.push_back(3 + (i*7)%16); }
        for (int i = 0; i < workers; ++i) { data.batches.push_back(llama_batch_init(seqs, 0, 1)); }
        bool threw = false;
        try {
            pipeline_require(llama_pipeline_stream(owner.get(), 24,
                [](void * ptr, uint32_t lane, llama_context * output, bool ready, llama_batch * next) {
                    auto & d = *static_cast<state *>(ptr);
                    if (ready) {
                        pipeline_require(d.last[lane] >= 0, "submitted output");
                        auto & entry = d.trace[d.last[lane]];
                        for (int row = 0; row < int(entry.ids.size()); ++row) {
                            const float * logits = llama_get_logits_ith(output, row);
                            entry.logits.insert(entry.logits.end(), logits, logits + d.vocab);
                            ++d.step[entry.ids[row]];
                        }
                        if (++d.completions == 2 && d.scenario == "exception") { throw std::runtime_error("injected callback failure"); }
                    }
                    if (!next) { return false; }
                    auto & b = d.batches[lane];
                    common_batch_clear(b);
                    execution entry;
                    for (int seq = lane; seq < d.seqs; seq += d.workers) {
                        if (d.step[seq] >= d.limit[seq] || (d.scenario == "duplicate" && !entry.ids.empty())) { continue; }
                        const int id = d.scenario == "duplicate" ? 0 : seq;
                        const int pos = d.prefixes[seq] + d.step[seq];
                        const llama_token token = (*d.text)[(size_t(seq)*4096 + pos)%d.text->size()];
                        common_batch_add(b, token, pos, {id}, true);
                        entry.ids.push_back(id);
                        entry.positions.push_back(pos);
                        entry.tokens.push_back(token);
                    }
                    if (!b.n_tokens) { return false; }
                    d.last[lane] = d.trace.size();
                    d.trace.push_back(std::move(entry));
                    *next = b;
                    return true;
                }, &data) == 0, "stream result");
        } catch (const std::exception &) {
            threw = true;
        }
        pipeline_require(threw == (scenario == "exception" || scenario == "duplicate"), "exception contract");
        // All submitted GPU work must be drained, including callbacks that throw.
        llama_synchronize(owner.get());
        double error = 0;
        if (!threw) {
            for (int seq = 0; seq < seqs; ++seq) {
                pipeline_require(data.step[seq] == data.limit[seq], "uneven completion count");
                pipeline_require(llama_memory_seq_pos_max(llama_get_memory(owner.get()), seq) == prefixes[seq] + data.limit[seq] - 1, "uneven final position");
            }
            owner->set_pipeline(0, ubatch);
            restore();
            // Replay the exact submitted order and shapes through ordinary synchronous decode.
            for (const auto & entry : data.trace) {
                common_batch_clear(batch);
                for (size_t i = 0; i < entry.ids.size(); ++i) { common_batch_add(batch, entry.tokens[i], entry.positions[i], {entry.ids[i]}, true); }
                pipeline_require(llama_decode(owner.get(), batch) == 0, "replay decode");
                std::vector<float> actual;
                for (size_t row = 0; row < entry.ids.size(); ++row) {
                    const float * logits = llama_get_logits_ith(owner.get(), row);
                    actual.insert(actual.end(), logits, logits + vocab);
                }
                error = std::max(error, pipeline_difference(entry.logits, actual));
            }
            std::fprintf(stderr, "Replay max_abs = %.9g\n", error);
            pipeline_require(error < 0.02, "pipeline trace differs from synchronous replay");
        }
        pipeline_require(llama_get_memory(owner.get())->memory_breakdown() == memory_size, "shared KV size unchanged");
        std::printf("Pipeline lifecycle: %s, expected exception = %d, max_abs_vs_replay = %.9g, submissions = %zu\n", scenario.c_str(), threw, error, data.trace.size());
        for (auto b : data.batches) { llama_batch_free(b); }
    }
    llama_batch_free(batch);
    owner.reset();
    init.reset();
    llama_backend_free();
    std::puts("Pipeline lifecycle checks passed");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "pipeline") { return test_pipeline(argc - 1, argv + 1); }
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    //llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
    //    if (level == GGML_LOG_LEVEL_ERROR) {
    //        common_log_add(common_log_main(), level, "%s", text);
    //    }
    //}, NULL);

    auto cparams = common_context_params_to_llama(params);

    // each context has a single sequence
    cparams.n_seq_max = 1;

    int dev_count = ggml_backend_dev_count();
    std::vector<std::array<ggml_backend_dev_t, 2>> gpus;
    for (int i = 0; i < dev_count; ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back({dev, nullptr});
        }
    }
    const int gpu_dev_count = (int)gpus.size();
    const int num_models = gpu_dev_count + 1 + 1; // GPUs + 1 CPU model + 1 layer split
    //const int num_models = std::max(1, gpu_dev_count);
    const int num_contexts = std::max(1, params.n_parallel);

    std::vector<llama_model_ptr> models;
    std::vector<std::thread> threads;
    std::atomic<bool> failed = false;

    for (int m = 0; m < num_models; ++m) {
        auto mparams = common_model_params_to_llama(params);

        if (m < gpu_dev_count) {
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            mparams.devices = gpus[m].data();
        } else if (m == gpu_dev_count) {
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            mparams.main_gpu = -1; // CPU model
        } else {
            mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;
        }

        llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (model == NULL) {
            LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
            return 1;
        }

        models.emplace_back(model);
    }

    for  (int m = 0; m < num_models; ++m) {
        auto * model = models[m].get();
        for (int c = 0; c < num_contexts; ++c) {
            threads.emplace_back([&, m, c, model]() {
                LOG_INF("Creating context %d/%d for model %d/%d\n", c + 1, num_contexts, m + 1, num_models);

                llama_context_ptr ctx { llama_init_from_model(model, cparams) };
                if (ctx == NULL) {
                    LOG_ERR("failed to create context\n");
                    failed.store(true);
                    return;
                }

                std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler { common_sampler_init(model, params.sampling), common_sampler_free };
                if (sampler == NULL) {
                    LOG_ERR("failed to create sampler\n");
                    failed.store(true);
                    return;
                }

                llama_batch batch = {};
                {
                    auto prompt = common_tokenize(ctx.get(), params.prompt, true);
                    if (prompt.empty()) {
                        LOG_ERR("failed to tokenize prompt\n");
                        failed.store(true);
                        return;
                    }
                    batch = llama_batch_get_one(prompt.data(), prompt.size());
                    if (llama_decode(ctx.get(), batch)) {
                        LOG_ERR("failed to decode prompt\n");
                        failed.store(true);
                        return;
                    }
                }

                const auto * vocab = llama_model_get_vocab(model);
                std::string result = params.prompt;

                for (int i = 0; i < params.n_predict; i++) {
                    llama_token token;
                    if (batch.n_tokens > 0) {
                        token = common_sampler_sample(sampler.get(), ctx.get(), batch.n_tokens - 1);
                    } else {
                        token = llama_vocab_bos(vocab);
                    }

                    result += common_token_to_piece(ctx.get(), token);

                    if (llama_vocab_is_eog(vocab, token)) {
                        break;
                    }

                    batch = llama_batch_get_one(&token, 1);

                    int ret = llama_decode(ctx.get(), batch);
                    if (ret == 1 && i > 0) {
                        LOG_INF("Context full, stopping generation.\n");
                        break;
                    }

                    if (ret != 0) {
                        LOG_ERR("Model %d/%d, Context %d/%d: failed to decode\n", m + 1, num_models, c + 1, num_contexts);
                        failed.store(true);
                        return;
                    }
                }

                LOG_INF("Model %d/%d, Context %d/%d: %s\n\n", m + 1, num_models, c + 1, num_contexts, result.c_str());

                llama_synchronize(ctx.get());
            });
        }
    }

    for (auto & thread : threads) {
        thread.join();
    }

    if (failed) {
        LOG_ERR("One or more threads failed.\n");
        return 1;
    }

    LOG_INF("All threads finished without errors.\n");
    return 0;
}
