#include "engine/util.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

namespace truss::engine {

namespace {
const auto load_t0 = std::chrono::steady_clock::now();
}

void load_mark(const char * what)
{
    static const bool on = std::getenv("TRUSS_LOAD_TIMES") != nullptr;
    if (on)
        std::fprintf(stderr, "load %7.2f s  %s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0).count(), what);
}

double spin_timeout_s()
{
    static const double s = [] {
        const char * e = std::getenv("TRUSS_SPIN_TIMEOUT_S");
        const double v = e ? std::atof(e) : 0.0;
        return v > 0 ? v : 60.0;
    }();
    return s;
}

bool robust_on()
{
    const char * env = std::getenv("TRUSS_ROBUST");
    return !env || std::atoi(env) != 0;
}

void inject_fault(const char * site, int slot)
{
    const char * env = std::getenv("TRUSS_FAULT");
    if (!env || !*env) return;
    const std::string spec = env;
    static std::mutex mu;
    static std::map<std::string, int> passes;   // site + slot + spec -> passes (a new spec re-arms from one)
    std::lock_guard<std::mutex> g(mu);
    for (size_t i = 0; i < spec.size();) {
        const size_t comma = spec.find(',', i);
        std::string one = spec.substr(i, comma == std::string::npos ? std::string::npos : comma - i);
        i = comma == std::string::npos ? spec.size() : comma + 1;
        int want = -1;
        const size_t at = one.rfind('@');
        if (at != std::string::npos) want = std::atoi(one.c_str() + at + 1), one.erase(at);
        const size_t colon = one.rfind(':');
        if (colon == std::string::npos || one.substr(0, colon) != site || (want >= 0 && want != slot)) continue;
        const std::string after = one.substr(colon + 1);
        const int n = ++passes[one + "@" + std::to_string(slot) + "|" + spec];
        if (n == 1 ? (after == "+" || after == "1") : std::atoi(after.c_str()) == n)
            throw std::runtime_error(std::string("truss: TRUSS_FAULT fired at ") + one + " on slot " +
                                     std::to_string(slot) + " pass " + std::to_string(n));
    }
}

}  // namespace truss::engine
