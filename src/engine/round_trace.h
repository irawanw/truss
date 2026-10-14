#pragma once
// TRUSS_ROUND_TRACE=<file>: the per-round interval log of track U (PLAN.md v1.6 U0) - what the two GPUs are not
// doing, and who is making them wait. Off unless the variable is set: every record site is one bool test, and the
// records go into a preallocated ring (no allocation, no I/O per record), written out by rt::dump().
//
// Two kinds of record, both on the host's steady clock:
//   host range  - a scope a thread spends in one place (a lock wait, a cudaStreamSynchronize, the pick callback),
//                 tagged by site and by the stage the thread is working on.
//   gpu work    - nanoseconds the device actually spent running kernels. A stage has one stream, so a held section
//                 is a run of kernels with host round trips between them: an event pair around the whole section
//                 only gives the elapsed SPAN (dead time inside it included), so events are stamped at each point
//                 where the host is about to wait for the device. The elapsed time between the section's first
//                 stamp and each of those is the work done up to there; the differences are the segments of real
//                 device work, and a host wait longer than the work it waited for is the device sitting idle while
//                 the stage is held. The segment after the last wait is read at the stage's next acquire, with a
//                 query - never a sync.
//
// The file: Hdr, then Recs in write order (not sorted; the reader sorts). Marks (kind 2) let the harness name the
// phases it measured. scripts/u_trace_report.py reads it.
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

namespace truss::engine::rt
{
enum Site : int32_t
{
    kNone = 0,
    kRound,       // Forward::round(), one slot, start to end
    kHeldFront,   // the front stage held (acquire -> release): the section's host time
    kHeldLast,    // the last stage held
    kLockFront,   // waiting for the front stage's lock (hand-off order)
    kLockLast,    // waiting for the last stage's lock
    kGpu,         // device work (see above); stage says which device
    kDraftMinP,   // draft(): the per-draft-step D2H of the guess probability + stream sync (an unsure guess ends the chain)
    kDraftEnd,    // draft(): the final sync, before the draft ids are read back on the host
    kFrontWait,   // round(): cudaEventSynchronize of the front stage's slot_done event, holding nothing
    kPick,        // round(): the caller's pick() callback (the last stage is held through it)
    kPickSync,    // pick(): the sync that waits for the sampled token (spec sampling)
    kPickSyncQ,   // pick(): the same for the argmax-of-q path
    kPlainSync,   // the no-window single step's sync (no room for a draft)
    kPromptFront, // prompt(): the front thread's chunk section
    kPromptLast,  // prompt(): the last stage's chunk section
    kGiveWay,     // prompt(): Strata's decode share - waiting out the decode window
    kFrontDrain,  // prompt(): the front thread's final stream sync
    kMark,        // the harness's phase mark; aux = the tag
    kStats,       // what the trace failed to read; dur = records dropped, aux = segments left unread
    kGpuSample,   // device work after a mid stamp (U2): what the pick's sync waits for beyond the queued layers/head
    kSiteCount
};
inline const char * site_name(int32_t s)
{
    static const char * n[kSiteCount] = { "none", "round", "held front", "held last", "lock front", "lock last",
                                          "gpu work", "draft min_p sync", "draft end sync", "front-event wait",
                                          "pick", "pick sync", "pick sync (argmax q)", "single step sync",
                                          "prompt front", "prompt last", "give way", "front drain", "mark",
                                          "stats", "gpu sampling" };
    return s >= 0 && s < kSiteCount ? n[s] : "?";
}

constexpr int32_t kKindHost = 0, kKindGpu = 1, kKindMark = 2, kKindStats = 3;

struct Rec
{
    int64_t t0 = 0, dur = 0;    // ns on steady_clock (gpu work: placed at the end of the segment, minus its length)
    int32_t site = 0, slot = -1, stage = -1, kind = kKindHost;
    int64_t aux = 0;            // marks: the tag; others: the round counter | kAuxPlacedBound (see below)
};

// A device segment read from a pair whose section had already been released (drain_stage) is placed at the host
// time of the last stamp INSIDE that section: that is a lower bound on when the device started it - the upper bound
// on when it ended is the sweep that happened to read the pair, which is not recorded. Such a segment is real device
// work at an uncertain position, so it carries this bit and the report keeps it out of the busy/idle split of any
// window it lands in instead of inventing busy time where the device never ran (W0).
constexpr int64_t kAuxPlacedBound = 1LL << 62;

struct Hdr
{
    char magic[8] = { 'T', 'R', 'U', 'S', 'S', 'R', 'T', '1' };   // constant; the reader sniffs the layout by SIZE,
                                                                  // because v3 shipped without this field
    int32_t ver = 6, nstage = 2;   // v6: aux carries kAuxPlacedBound; the layout is unchanged from v5
    int64_t epoch = 0;          // steady_clock at open, ns
    int32_t dev[4] = { -1, -1, -1, -1 };   // device of stage 0, 1, ... (the report names the GPU)
    int32_t pad = 0;
};

inline int64_t now() { return std::chrono::steady_clock::now().time_since_epoch().count(); }

struct StageState;   // below; registered here so any thread can read the segments its stages finished

struct Trace
{
    static constexpr size_t kCap = 2'000'000;   // ~40 rounds/s of two slots: minutes of both-decoding
    std::unique_ptr<Rec[]> rec;
    std::atomic<size_t> n{ 0 };
    size_t written = 0;         // records in the file (dump appends the rest)
    std::atomic<size_t> dropped{ 0 };
    FILE * f = nullptr;
    const char * path = nullptr;
    Hdr hdr;
    std::mutex dump_mu;   // held across the write (H3): close() must not fclose the FILE another thread is inside
    StageState * stages[4] = { nullptr, nullptr, nullptr, nullptr };   // by stage index (written once, idempotent)
    int64_t last_drain = 0;             // the last sweep for finished segments (throttled on the hot path)
    std::atomic<size_t> lost_seg{ 0 };  // pairs reused before their work could be read: busy time the file lacks
};
inline Trace & tr()
{
    static Trace t;
    return t;
}

// On once per process: the variable's value is read the first time anything is traced.
inline bool on()
{
    static const bool v = [] {
        const char * e = std::getenv("TRUSS_ROUND_TRACE");
        if (!e || !*e) return false;
        Trace & T = tr();
        T.path = e;
        T.rec.reset(new Rec[Trace::kCap]);
        T.hdr.epoch = now();
        std::fprintf(stderr, "TRUSS_ROUND_TRACE: %s\n", e);
        return true;
    }();
    return v;
}

inline void set_stage_device(int stage, int device)
{
    if (stage >= 0 && stage < 4) tr().hdr.dev[stage] = device;
}

inline void push(const Rec & r)
{
    Trace & T = tr();
    const size_t i = T.n.fetch_add(1, std::memory_order_relaxed);
    if (i < Trace::kCap) T.rec[i] = r;
    else T.dropped.fetch_add(1, std::memory_order_relaxed);
}

// What a thread is working on, for the records it makes. stage is set by Impl::acquire/release; slot and round by
// Forward::round / Forward::prompt.
struct Ctx
{
    int slot = -1, stage = -1;
    int64_t round = 0;
};
inline Ctx & ctx()
{
    static thread_local Ctx c;
    return c;
}

struct Scope
{
    int64_t t0 = 0;
    int32_t site = kNone;
    explicit Scope(Site s) : site(on() ? (int32_t) s : kNone) { if (site != kNone) t0 = now(); }
    Scope(const Scope &) = delete;
    Scope & operator=(const Scope &) = delete;
    ~Scope()
    {
        if (site != kNone)
        {
            const Ctx & c = ctx();
            push({ t0, now() - t0, site, c.slot, c.stage, kKindHost, c.round });
        }
    }
};
// ---- one stage's device-work accounting. A StageState lives in its Impl and covers the section it holds.
struct StageState
{
    cudaStream_t s = nullptr;
    int device = -1, stage = -1;
    // One event pair per held section, three in rotation: the pair of a released section stays unread until some
    // thread finds its closing event complete, and with two the next-but-one acquire could come around to it.
    cudaEvent_t evA[3] = { nullptr, nullptr, nullptr }, evB[3] = { nullptr, nullptr, nullptr };
    // An optional third stamp inside the section (U2): recorded without waiting, read by the next Wait, which then
    // splits its segment in two - the work queued before the stamp and the work enqueued after it.
    cudaEvent_t evM = nullptr;
    bool has_m = false;
    int cur = 0;
    double cum = 0;             // ms of device work already accounted for in this section
    int64_t last_t = 0;         // host time of the stamp cum was taken at: where the next segment starts
    bool pend[3] = { false, false, false };
    double pend_cum[3] = { 0, 0, 0 };
    int64_t pend_t0[3] = { 0, 0, 0 };
    int32_t pend_slot[3] = { -1, -1, -1 };
    int64_t pend_round[3] = { 0, 0, 0 };
    bool held = false;          // a section is in hand: its event pair is live
    // The pend[] fields are written by the thread that owns the stage and read by whichever thread sweeps (H2):
    // unsynchronised, the sweep can see pend[p] set before the values that go with it, or clear it a moment before
    // the owner re-records that pair, which invents busy time for the wrong slot. This mutex is the whole protocol:
    // once per section for the owner, once per sweep for the sweeper, never held across a device wait.
    std::mutex mu;
    ~StageState();   // defined below DevGuard: it unregisters the stage and releases the events
};

struct DevGuard
{
    int prev = -1;
    explicit DevGuard(int device)
    {
        cudaGetDevice(&prev);
        if (device >= 0 && device != prev) cudaSetDevice(device);
    }
    ~DevGuard() { if (prev >= 0) cudaSetDevice(prev); }
    DevGuard(const DevGuard &) = delete;
    DevGuard & operator=(const DevGuard &) = delete;
};

// The segment of device work between the stage's last stamp and this one, placed at the end of the segment.
inline void push_work(double & cum, int64_t & t0, double ms, int64_t end_ns, int32_t stage, int32_t slot,
                      int64_t round, int32_t site = kGpu)
{
    const double d = ms - cum;
    cum = ms;
    t0 = end_ns;
    if (d > 0.0) push({ end_ns - (int64_t) (d * 1e6), (int64_t) (d * 1e6), site, slot, stage, kKindGpu, round });
}

// A StageState lives inside an Impl. Unregister it before it dies: a later sweep - close(), or another stage's
// acquire - reads Trace::stages[], and after Forward's destructor those pointers are freed memory (review C1: every
// engine teardown did this, which for the served server is every restart). The events go with the stage.
inline StageState::~StageState()
{
    if (stage >= 0 && tr().stages[stage] == this) tr().stages[stage] = nullptr;
    if (device < 0) return;
    DevGuard g(device);
    for (int p = 0; p < 3; ++p)
    {
        if (evA[p]) cudaEventDestroy(evA[p]);
        if (evB[p]) cudaEventDestroy(evB[p]);
    }
    if (evM) cudaEventDestroy(evM);
}

// One stage's finished segments, read under its own mutex (H2). Never blocks on the device: a pair whose event is
// not complete is left for a later sweep, and one about to be reused counts as lost.
inline void drain_stage(StageState & st, int64_t not_before = 0)
{
    std::lock_guard<std::mutex> lk(st.mu);
    for (int p = 0; p < 3; ++p)
    {
        if (!st.pend[p]) continue;
        float ms = 0.f;
        DevGuard g(st.device);
        if (cudaEventQuery(st.evB[p]) != cudaSuccess) continue;
        if (cudaEventElapsedTime(&ms, st.evA[p], st.evB[p]) != cudaSuccess) { st.pend[p] = false; continue; }
        const double d = ms - st.pend_cum[p];
        const int64_t t0 = st.pend_t0[p];
        st.pend[p] = false;
        if (d > 0.0 && t0 >= not_before)   // placement is a lower bound only: see kAuxPlacedBound
            push({ t0, (int64_t) (d * 1e6), kGpu, st.pend_slot[p], st.stage, kKindGpu,
                   st.pend_round[p] | kAuxPlacedBound });
    }
}

// Every segment the stages have finished since the last sweep, each under its own mutex. A released section's last
// segment is the one its release event closes, and the device may still be running it when the host drops the lock -
// so it is read whenever a thread happens to look (a host wait, a phase mark, teardown) rather than at a point that
// would have to wait for the device. Placed where its work started, not where it was read.
inline void drain_stages(int64_t not_before = 0)
{
    Trace & T = tr();
    T.last_drain = now();
    for (int i = 0; i < 4; ++i)
        if (T.stages[i] && T.stages[i]->stage == i) drain_stage(*T.stages[i], not_before);
}

// Under the stage's lock, with the caller's device current: the section starts here on the device.
inline void stage_begin(StageState & st, cudaStream_t s, int device, int stage)
{
    if (!on()) return;
    st.s = s;
    st.device = device;
    st.stage = stage;
    tr().stages[stage] = &st;
    drain_stage(st);   // only its own pairs: sweeping the other stage under this lock is latency at the hand-off (M2)
    st.cur = (st.cur + 1) % 3;
    {
        std::lock_guard<std::mutex> lk(st.mu);
        if (st.pend[st.cur])   // two whole sections of slack and still unread: the device ran too far ahead to measure
        {
            st.pend[st.cur] = false;
            tr().lost_seg.fetch_add(1, std::memory_order_relaxed);
        }
    }
    st.cum = 0;
    st.held = true;
    st.last_t = now();
    st.has_m = false;
    DevGuard g(device);
    if (!st.evA[st.cur]) cudaEventCreate(&st.evA[st.cur]);
    if (!st.evB[st.cur]) cudaEventCreate(&st.evB[st.cur]);
    cudaEventRecord(st.evA[st.cur], s);
}

// A stamp inside the held section that the host does not wait for. The next Wait reads it; a section whose stamp is
// never read costs nothing, because cum is untouched by it and the release segment still covers the whole section.
inline void stamp_mid(StageState & st)
{
    if (!on() || !st.held) return;
    DevGuard g(st.device);
    if (!st.evM && cudaEventCreate(&st.evM) != cudaSuccess) return;
    if (cudaEventRecord(st.evM, st.s) == cudaSuccess) st.has_m = true;
}

// A host wait on this stage's own device, inside the held section: stamp before it, account the work after it.
struct Wait
{
    StageState * st = nullptr;
    int32_t slot = -1;
    int64_t round = 0;
    explicit Wait(StageState & s) : st(on() && s.held ? &s : nullptr)
    {
        if (!st) return;
        slot = ctx().slot;
        round = ctx().round;
        if (now() - tr().last_drain > 2'000'000) drain_stages();   // someone else's finished segment, cheaply
        DevGuard g(st->device);
        cudaEventRecord(st->evB[st->cur], st->s);
    }
    Wait(const Wait &) = delete;
    Wait & operator=(const Wait &) = delete;
    ~Wait()
    {
        if (!st) return;
        float ms = 0.f;
        DevGuard g(st->device);
        float mm = 0.f;
        const bool split = st->has_m && cudaEventElapsedTime(&mm, st->evA[st->cur], st->evM) == cudaSuccess;
        if (cudaEventElapsedTime(&ms, st->evA[st->cur], st->evB[st->cur]) == cudaSuccess)
        {
            // Host and device coincide at the return of the wait this object wraps - evB is complete now - so the
            // device passed evM at t_end - (ms - mm), NOT at the host time of the mid stamp. Anchoring on m_t drew
            // the queued-work segment 8-11 ms early, inside the PREVIOUS round's pick window, and the reader charged
            // the window it should have filled to "host wait on this GPU" = idle: 6.08 ms/round of invented idle
            // (track U W0, 10-14). Chain back from the sync's return instead.
            const int64_t t_end = now();
            if (split && mm > st->cum)   // what was already queued, drained by this wait
            {
                const int64_t d_ns = (int64_t) ((mm - st->cum) * 1e6);
                int64_t mid_end = t_end - (int64_t) ((ms - mm) * 1e6);
                if (mid_end < st->last_t + d_ns) mid_end = st->last_t + d_ns;   // never overlap the prior segment
                push_work(st->cum, st->last_t, mm, mid_end, st->stage, slot, round);
            }
            push_work(st->cum, st->last_t, ms, t_end, st->stage, slot, round,
                      split ? kGpuSample : kGpu);   // and what ran after the mid stamp
        }
        st->has_m = false;
    }
};

// At release: stamp what is enqueued so far. The device is often still running it when the lock drops, so the
// segment that stamp closes is left pending for the next sweep (see drain_stages).
inline void stage_end(StageState & st)
{
    if (!on() || !st.evB[st.cur]) return;
    DevGuard g(st.device);
    if (cudaEventRecord(st.evB[st.cur], st.s) == cudaSuccess)
    {
        std::lock_guard<std::mutex> lk(st.mu);
        st.pend[st.cur] = true;
        st.pend_cum[st.cur] = st.cum;
        st.pend_t0[st.cur] = st.last_t;
        st.pend_slot[st.cur] = ctx().slot;
        st.pend_round[st.cur] = ctx().round;
    }
    st.held = false;
}

// ---- output: append what is not yet in the file. Safe from any thread, safe to call twice.
// A stamp's record can take its index just as a dump reads the count, leaving an all-zero record in the file: the
// reader skips those (site 0 with t0 0).
inline void dump_locked()
{
    Trace & T = tr();
    if (!on()) return;
    // M1: the file carries what it failed to read, so the report can call a busy % what it is: a floor.
    push({ now(), (int64_t) T.dropped.load(std::memory_order_relaxed), kStats, -1, -1, kKindStats,
           (int64_t) T.lost_seg.load(std::memory_order_relaxed) });
    // H1: past kCap the count is beyond the allocation, so the file gets the first kCap records and no more.
    size_t n = T.n.load(std::memory_order_relaxed);
    if (n > Trace::kCap) n = Trace::kCap;
    if (n > T.written)
    {
        if (!T.f) T.f = std::fopen(T.path, T.written ? "ab" : "wb");
        if (T.f)
        {
            if (T.written == 0) std::fwrite(&T.hdr, sizeof(Hdr), 1, T.f);
            std::fwrite(T.rec.get() + T.written, sizeof(Rec), n - T.written, T.f);
            std::fflush(T.f);
            T.written = n;
        }
        else
            std::fprintf(stderr, "TRUSS_ROUND_TRACE: cannot write %s\n", T.path);
    }
}

inline void dump()
{
    if (!on()) return;
    std::lock_guard<std::mutex> lk(tr().dump_mu);
    dump_locked();
}

inline void mark(int tag)
{
    if (!on()) return;
    drain_stages();   // the phase's last segments, read before its boundary is recorded
    const Ctx & c = ctx();
    push({ now(), 0, kMark, c.slot, c.stage, kKindMark, tag });
    dump();   // a phase boundary is a good moment to make the file readable
}

inline void close()
{
    Trace & T = tr();
    if (!on()) return;
    drain_stages();
    {   // H3: the last write and the fclose under one lock, so a mark() on another thread cannot be inside fwrite
        std::lock_guard<std::mutex> lk(T.dump_mu);
        dump_locked();
        if (T.f) { std::fclose(T.f); T.f = nullptr; }
    }
    if (const size_t lost = T.lost_seg.load())
        std::fprintf(stderr, "TRUSS_ROUND_TRACE: %zu segment(s) went unread (busy time the file lacks)\n", lost);
    if (const size_t d = T.dropped.load())
        std::fprintf(stderr, "TRUSS_ROUND_TRACE: ring full, %zu record(s) dropped\n", d);
}
}  // namespace truss::engine::rt

#define TRUSS_RT_CAT2(a, b) rt_scope_##a##_##b
#define TRUSS_RT_CAT(a, b) TRUSS_RT_CAT2(a, b)
#define TRUSS_RT(site) ::truss::engine::rt::Scope TRUSS_RT_CAT(x, __LINE__)(site)
