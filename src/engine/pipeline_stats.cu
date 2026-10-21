// engine::Pipeline diagnostics: sums over the stages of the expert runtime's, CPU pool's and profile's counters.
#include "engine/pipeline.h"

namespace truss::engine {

int Pipeline::hot_experts() const
{
    int n = m_->n_hot();
    for (auto & p : front_) n += p->n_hot();
    return n;
}

size_t Pipeline::cold_bytes() const
{
    size_t n = 0;
    for (auto & p : front_) n += p->moe() ? p->moe()->store().cold_bytes() : 0;
    return n + (m_->moe() ? m_->moe()->store().cold_bytes() : 0);
}

void Pipeline::wait_pinned()
{
    each_stage([](Stage & m) { if (m.moe()) m.moe()->wait_pinned(); });
}

void Pipeline::lock_ms(double out[4], bool reset)
{
    Stage * st[2] = { front_.empty() ? nullptr : front_[0].get(), m_.get() };
    for (int i = 0; i < 2; ++i) {
        out[2 * i] = st[i] ? st[i]->lk.held_ns.load() / 1e6 : 0;
        out[2 * i + 1] = st[i] ? st[i]->lk.wait_ns.load() / 1e6 : 0;
        if (reset && st[i]) st[i]->lk.held_ns = 0, st[i]->lk.wait_ns = 0;
    }
}

void Pipeline::section_ms(double out[6], bool reset) const
{
    for (int i = 0; i < 6; ++i) out[i] = 0;
    auto add = [&](Stage & m) {
        double s[6];
        m.prof().ms(s, reset);
        for (int i = 0; i < 6; ++i) out[i] += s[i];
    };
    for (auto & p : front_) add(*p);
    add(*m_);
}

void Pipeline::section_moe_ms(double out[4], bool reset) const
{
    for (int i = 0; i < 4; ++i) out[i] = 0;
    auto add = [&](Stage & m) {
        double s[4];
        m.prof().moe_ms(s, reset);
        for (int i = 0; i < 4; ++i) out[i] += s[i];
    };
    for (auto & p : front_) add(*p);
    add(*m_);
}

void Pipeline::driver_ms(double out[3], long & n, bool reset) const
{
    for (int i = 0; i < 3; ++i) out[i] = 0;
    n = 0;
    auto add = [&](Stage & m) {
        if (!m.moe()) return;
        double s[3];
        long k;
        m.moe()->driver_ms(s, k, reset);
        for (int i = 0; i < 3; ++i) out[i] += s[i];
        n += k;
    };
    for (auto & p : front_) add(*p);
    add(*m_);
}

std::pair<long long, long> Pipeline::cpu_stats(bool reset)
{
    cpu::ExpertPool * p = m_->moe() ? m_->moe()->pool() : nullptr;   // a split's stages share one pool
    if (!p) return { 0, 0 };
    const auto out = std::make_pair(p->wait_us(), p->waits());
    if (reset) p->reset_stats();
    return out;
}

long long Pipeline::cpu_item_us() const
{
    cpu::ExpertPool * p = m_->moe() ? m_->moe()->pool() : nullptr;
    return p ? p->item_us() : 0;
}

void Pipeline::cpu_phase_us(long long out[4]) const
{
    cpu::ExpertPool * p = m_->moe() ? m_->moe()->pool() : nullptr;
    if (!p) {
        out[0] = out[1] = out[2] = out[3] = 0;
        return;
    }
    out[0] = p->gate_us(), out[1] = p->silu_us(), out[2] = p->hq_us(), out[3] = p->down_us();
}

void Pipeline::cpu_shape(long long out[4]) const
{
    cpu::ExpertPool * p = m_->moe() ? m_->moe()->pool() : nullptr;
    if (!p) {
        out[0] = out[1] = out[2] = out[3] = 0;
        return;
    }
    p->shape(out);
}

void Pipeline::cpu_shape_reset()
{
    if (cpu::ExpertPool * p = m_->moe() ? m_->moe()->pool() : nullptr) p->reset_stats();
}

void Pipeline::dyn_stats(long & cpu, long & pcie, double & ms_expert, double & ms_call) const
{
    cpu = pcie = 0, ms_expert = ms_call = 0;
    if (m_->moe()) m_->moe()->dyn_stats(cpu, pcie, ms_expert, ms_call);
    for (auto & p : front_)
        if (p->moe()) {
            long c, q;
            double a, b;
            p->moe()->dyn_stats(c, q, a, b);
            cpu += c, pcie += q;
        }
}

void Pipeline::split_stats(long long out[3]) const
{
    out[0] = out[1] = out[2] = 0;
    auto add = [&](Stage & m) {
        if (!m.moe()) return;
        long long s[3];
        m.moe()->split_stats(s);
        for (int i = 0; i < 3; ++i) out[i] += s[i];
    };
    for (auto & p : front_) add(*p);
    add(*m_);
}

size_t Pipeline::kv_lent() const
{
    size_t n = m_->moe() ? m_->moe()->lent_bytes() : 0;
    for (auto & p : front_) n += p->moe() ? p->moe()->lent_bytes() : 0;
    return n;
}

const runtime::ExpertStore * Pipeline::experts() const { return m_->moe() ? &m_->moe()->store() : nullptr; }

void Pipeline::profile_routes(bool on)
{
    // held per stage (H3): freeing a table that a chunk on that stage is still routing into is corruption
    each_stage_locked([on](Stage & m) { if (m.moe()) m.moe()->profile_routes(on); });
}

std::vector<float> Pipeline::route_counts() const
{
    std::vector<float> out;
    each_stage_locked([&](Stage & m) {
        DeviceGuard g(m.device());
        if (!m.moe()) return;
        const std::vector<float> r = m.moe()->route_counts();
        out.insert(out.end(), r.begin(), r.end());
    });
    return out;
}

void Pipeline::table_stats(double & ms, long & calls, long & rows, bool reset, double * wait_ms, double * phase_ms)
{
    ms = 0, calls = 0, rows = 0;
    if (wait_ms) *wait_ms = 0;
    if (phase_ms) phase_ms[0] = phase_ms[1] = phase_ms[2] = 0;
    each_stage([&](Stage & m) { m.model().table_stats(ms, calls, rows, reset, wait_ms, phase_ms); });
}

void Pipeline::expert_counts(long long out[6]) const
{
    for (int k = 0; k < 6; ++k) out[k] = 0;
    auto add = [&](Stage & m) {
        const runtime::MoeExec * x = m.moe();
        if (!x) return;
        const runtime::ExpertStore::Stats & s = x->store().stats();
        out[0] += s.experts_asked, out[1] += s.misses, out[2] += (long long) s.bytes;
        out[3] += s.hinted, out[4] += (long long) s.hint_bytes;
    };
    for (auto & p : front_) add(*p);
    add(*m_);
    long cpu, pcie;
    double a, b;
    dyn_stats(cpu, pcie, a, b);
    out[5] = cpu;
}

bool Pipeline::set_moe_option(const std::string & name, double v)
{
    bool ok = true, any = false;
    each_stage([&](Stage & m) {
        if (m.moe()) any = true, ok = m.moe()->set_option(name, v) && ok;
    });
    return any && ok;
}

}  // namespace truss::engine
