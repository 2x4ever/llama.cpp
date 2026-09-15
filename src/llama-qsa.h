#pragma once

#include "llama-kv-cells.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>

struct llama_qsa_layout {
    struct token {
        int32_t cell;
        std::array<int32_t, 3> pos;
    };

    struct block {
        std::vector<int32_t> cells;
        std::array<int32_t, 4> pos;
    };

    struct sequence {
        std::vector<token> tokens;
        std::vector<std::pair<int64_t, int32_t>> blocks;
        bool ranked = false;
    };

    uint32_t ratio = 4;
    bool valid = false;
    bool pos_2d = false;
    std::map<llama_seq_id, sequence> sequences;
    std::vector<block> blocks;
    std::map<std::vector<int32_t>, int32_t> block_ids;
    std::vector<uint8_t> seen;
    std::vector<int32_t> dirty;

    token get_token(const llama_kv_cells & cells, int32_t i) const {
        const auto & ext = cells.ext_get(i);
        return { i, { cells.pos_get(i), pos_2d ? ext.y : 0, pos_2d ? ext.x : 0 } };
    }

    void append(sequence & seq, const token & t) {
        seq.tokens.push_back(t);
        const int64_t end = seq.ranked ? (int64_t) seq.tokens.size() - 1 : t.pos[0];
        if ((end + 1)%ratio || seq.tokens.size() < ratio) {
            return;
        }
        std::vector<int32_t> members;
        const size_t first = seq.tokens.size() - ratio;
        for (uint32_t j = 0; j < ratio; ++j) {
            if (!seq.ranked && seq.tokens[first + j].pos[0] != end - ratio + 1 + j) {
                return;
            }
            members.push_back(seq.tokens[first + j].cell);
        }
        auto entry = block_ids.emplace(members, (int32_t) blocks.size());
        if (entry.second) {
            const auto & p = seq.tokens[first].pos;
            blocks.push_back({ std::move(members), { p[0], p[1], p[2], p[0] } });
            dirty.push_back(entry.first->second);
        }
        seq.blocks.emplace_back(end, entry.first->second);
    }

    void rebuild(const llama_kv_cells & cells) {
        sequences.clear();
        blocks.clear();
        block_ids.clear();
        dirty.clear();
        seen.assign(cells.size(), 0);
        for (llama_seq_id id = 0; id < LLAMA_MAX_SEQ; ++id) {
            if (cells.seq_pos_min(id) < 0) {
                continue;
            }
            std::vector<token> ordered;
            for (uint32_t i = cells.used_min(); i < cells.used_max_p1(); ++i) {
                if (cells.seq_has(i, id)) {
                    ordered.push_back(get_token(cells, i));
                    seen[i] = 1;
                }
            }
            std::sort(ordered.begin(), ordered.end(), [](const token & a, const token & b) {
                return a.pos != b.pos ? a.pos < b.pos : a.cell < b.cell;
            });
            auto & seq = sequences[id];
            for (size_t i = 1; i < ordered.size(); ++i) {
                seq.ranked |= pos_2d && ordered[i - 1].pos[0] == ordered[i].pos[0];
            }
            for (const auto & t : ordered) {
                append(seq, t);
            }
        }
        valid = true;
    }

    void update(const llama_kv_cells & cells, const std::vector<uint32_t> & written, bool is_2d) {
        dirty.clear();
        bool reset = !valid || pos_2d != is_2d || seen.size() != cells.size();
        pos_2d = is_2d;
        std::vector<llama_seq_id> active;
        for (llama_seq_id id = 0; id < LLAMA_MAX_SEQ; ++id) {
            if (cells.seq_pos_min(id) >= 0) {
                active.push_back(id);
            }
        }
        if (!reset) {
            std::map<llama_seq_id, std::array<int32_t, 3>> last_pos;
            for (llama_seq_id id : active) {
                const auto it = sequences.find(id);
                if (it != sequences.end() && !it->second.tokens.empty()) {
                    last_pos[id] = it->second.tokens.back().pos;
                }
            }
            std::set<uint32_t> pending;
            for (uint32_t i : written) {
                reset |= seen[i] != 0 || !pending.insert(i).second;
                const auto t = get_token(cells, i);
                for (llama_seq_id id : active) {
                    if (!cells.seq_has(i, id)) {
                        continue;
                    }
                    auto & seq = sequences[id];
                    if (last_pos.count(id)) {
                        const auto & last = last_pos[id];
                        reset |= t.pos <= last || (pos_2d && !seq.ranked && t.pos[0] == last[0]);
                    }
                    last_pos[id] = t.pos;
                }
            }
        }
        if (reset) {
            rebuild(cells);
            return;
        }
        for (uint32_t i : written) {
            const auto t = get_token(cells, i);
            for (llama_seq_id id : active) {
                if (cells.seq_has(i, id)) {
                    append(sequences[id], t);
                }
            }
            seen[i] = 1;
        }
    }

    void row(llama_seq_id id, const std::array<int32_t, 3> & pos, uint32_t * bits, size_t words, int32_t * tail) const {
        std::fill(bits, bits + words, 0);
        std::fill(tail, tail + ratio - 1, -1);
        const auto it = sequences.find(id);
        if (it == sequences.end()) {
            return;
        }
        const auto & seq = it->second;
        const auto stop = std::upper_bound(seq.tokens.begin(), seq.tokens.end(), pos, [](const auto & p, const token & t) {
            return p < t.pos;
        });
        const int64_t q = seq.ranked ? stop - seq.tokens.begin() - 1 : pos[0];
        const int64_t start = (q + 1)/ratio*ratio;
        for (const auto & b : seq.blocks) {
            if (b.first >= start) {
                break;
            }
            bits[b.second/32] |= uint32_t(1) << (b.second%32);
        }
        if (seq.ranked) {
            for (int64_t j = start; j <= q; ++j) {
                tail[j - start] = seq.tokens[j].cell;
            }
        } else {
            auto first = std::lower_bound(seq.tokens.begin(), stop, start, [](const token & t, int64_t p) {
                return t.pos[0] < p;
            });
            for (auto cur = first; cur != stop; ++cur) {
                tail[cur->pos[0] - start] = cur->cell;
            }
        }
    }
};
