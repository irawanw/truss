// Flash-Next's MTP draft block as an engine::Drafter, on the last stage.
//
// The block is one DSA layer with routed experts, fed the join of the next token's embedding with the previous
// position's hidden state (res = eh_proj [RMSNorm(embed(token)) * enorm | RMSNorm_per_stream(h) * hnorm], per hc
// stream; llama-paw graph_mtp). Its caches must match a run over the TRUE hidden states, so every committed row is
// "committed" through it too (on_commit; position 0 has no previous state and is skipped), and draft() chains
// single-row steps from the last committed row whose cache rows and indexer block are tentative (restored after).
//
// accept() defers the commit of the accepted rows to the next draft(), which runs it together with the first draft
// row (pending state = the last accepted row, token = next): one MTP pass fewer per decode step. That row is the
// next commit's first row, so it is remembered (ahead) and the next commit skips it when its token matches.
#pragma once
#include "blocks/attn_dsa.h"
#include "blocks/moe.h"
#include "engine/model.h"
#include "model/qwen4exp/model.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace truss::qwen4exp {

class MtpDrafter : public engine::Drafter, public engine::Stateful {
public:
    MtpDrafter(engine::Stage & st, StageModel & sm, const Family & fam);
    ~MtpDrafter() override;
    void bind_experts(engine::Stage & st);

    // Drafter
    void on_commit(engine::Stage & st, const engine::Chunk & c) override;
    void on_accept(engine::Stage & st, int slot, int n, const engine::Window & w) override;
    void flush(engine::Stage & st, int slot) override;
    void park(engine::Stage & st, int slot) override;
    void unpark(engine::Stage & st, int slot) override;
    int draft(engine::Stage & st, int slot, int32_t next, int pos, int n, int32_t * out, const Sampling & s) override;
    engine::Stateful * state() override { return this; }

    // Stateful (its DSA layer's state is forwarded)
    void alloc_slots(engine::Stage & st, int n_slots, int seq_ctx, int spec_rows) override;
    void reset(engine::Stage & st, int slot) override;
    void accept(engine::Stage & st, int slot, int n, const engine::Window & w) override;
    void ck_buffers(int slot, std::vector<std::pair<float *, size_t>> & out) override;
    void ck_save(int slot, std::vector<int32_t> & host) override;
    void ck_load(int slot, const std::vector<int32_t> & host, size_t & at) override;
    void positional(std::vector<Positional> & out) const override { dsa_->positional(out); }

private:
    struct Slot {
        float * pending_h = nullptr;      // hidden state [hc][d] of the last committed row
        bool has_pending = false;
        bool dc_on = false;               // a deferred commit of dc_T rows at dc_pos0
        int dc_pos0 = 0, dc_T = 0;
        std::vector<int32_t> dc_tokens;
        bool ahead = false;               // draft() already computed the next commit's first row
        int32_t ahead_tok = -1;
        int ahead_pos = -1;
        float * res_keep = nullptr;       // the deferred rows while another slot uses the stage's buffers
    };
    void rows(engine::Stage & st, int slot, const float * h, const int * d_ids, int pos0, int T, bool stream,
              bool logits, bool calib = false);
    void commit(engine::Stage & st, int slot, const int32_t * tokens, int pos0, int T, bool stream);

    StageModel & sm_;
    const Config & c_;
    const Weights & w_;
    const Mtp & m_;
    std::unique_ptr<blocks::Dsa<DsaShape>> dsa_;
    std::unique_ptr<blocks::Moe> moe_;
    std::vector<Slot> s_;
    float * partial_snap_ = nullptr;      // the indexer's open block before a draft chain
    float * logits_ = nullptr;            // [n_vocab]
    int * draft_ids_ = nullptr;           // device [MAX_ROWS + 1]
    int32_t * draft_host_ = nullptr;      // pinned
    int32_t * commit_host_ = nullptr;     // pinned [2][MAX_ROWS + 1]: a decode commit's tokens (alternating halves)
    int commit_flip_ = 0;
    float * draft_prob_ = nullptr, * draft_prob_host_ = nullptr;
    float min_p_ = 0.f;
    // TRUSS_MTP_CALIB=/path/file.f32: the block's FFN input rows of true-path commits, for an offline Hessian
    FILE * calib_f_ = nullptr;
    float * calib_host_ = nullptr;
};

}  // namespace truss::qwen4exp
