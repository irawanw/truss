#include "cpu/placement.h"

#include <dirent.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace truss::cpu {

const std::vector<std::vector<int>> & core_cpus()
{
    static const std::vector<std::vector<int>> cores = [] {
        std::vector<std::vector<int>> out;
        std::vector<std::pair<long, long>> keys;
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof set, &set) != 0) return out;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &set)) continue;
            auto topo = [c](const char * what) {
                char path[96];
                std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", c, what);
                long v = -1;
                if (FILE * f = std::fopen(path, "r")) {
                    if (std::fscanf(f, "%ld", &v) != 1) v = -1;
                    std::fclose(f);
                }
                return v;
            };
            const std::pair<long, long> key{ topo("physical_package_id"), topo("core_id") };
            const auto it = std::find(keys.begin(), keys.end(), key);
            if (it == keys.end()) keys.push_back(key), out.push_back({ c });
            else out[(size_t) (it - keys.begin())].push_back(c);
        }
        return out;
    }();
    return cores;
}

namespace {

std::vector<int> parse_list(const char * s)   // "40,41,44-47"
{
    std::vector<int> out;
    while (*s) {
        char * end = nullptr;
        const long a = std::strtol(s, &end, 10);
        if (end == s) break;
        long b = a;
        s = end;
        if (*s == '-') {
            b = std::strtol(s + 1, &end, 10);
            s = end;
        }
        for (long c = a; c <= b && c < CPU_SETSIZE; ++c)
            if (c >= 0) out.push_back((int) c);
        while (*s == ',' || *s == ' ') ++s;
    }
    return out;
}

std::string list_text(const std::vector<int> & v)
{
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s.empty() ? "none" : s;
}

}  // namespace

AuxPlan plan_aux(const std::vector<int> & worker_cpus)
{
    AuxPlan p;
    const char * e = std::getenv("TRUSS_AUX_CPUS");
    if (!e || !*e || !std::strcmp(e, "0") || !std::strcmp(e, "off")) return p;
    if (worker_cpus.empty()) {
        p.text = "TRUSS_AUX_CPUS: off (the pool is not pinned; set TRUSS_CPU_PIN=1)";
        return p;
    }
    const auto & cores = core_cpus();
    if (cores.empty()) {
        p.text = "TRUSS_AUX_CPUS: off (no CPU topology)";
        return p;
    }
    auto is_worker = [&](int c) { return std::find(worker_cpus.begin(), worker_cpus.end(), c) != worker_cpus.end(); };
    p.driver_cpu = cores[0][0];
    std::vector<std::vector<int>> spare;   // cores with no worker CPU, core 0 aside
    for (size_t i = 1; i < cores.size(); ++i)
        if (std::none_of(cores[i].begin(), cores[i].end(), is_worker)) spare.push_back(cores[i]);
    if (!spare.empty()) p.caller_cpu = spare[0][0];
    const bool automatic = !std::strcmp(e, "auto") || !std::strcmp(e, "1");
    if (automatic) {
        for (size_t k = 1; k < cores[0].size(); ++k)
            if (!is_worker(cores[0][k])) p.aux.push_back(cores[0][k]);
        for (size_t i = 0; i < spare.size(); ++i)
            for (size_t k = i == 0 ? 1 : 0; k < spare[i].size(); ++k) p.aux.push_back(spare[i][k]);
    } else {
        for (int c : parse_list(e))
            if (c != p.driver_cpu && c != p.caller_cpu && !is_worker(c)) p.aux.push_back(c);
    }
    if (p.aux.empty()) {
        p.text = std::string("TRUSS_AUX_CPUS=") + e + ": off (no spare CPU beside the " +
                 std::to_string(worker_cpus.size()) + " workers)";
        return p;
    }
    p.on = true;
    p.text = std::string("TRUSS_AUX_CPUS=") + e + ": driver on CPU " + std::to_string(p.driver_cpu) + ", caller on " +
             (p.caller_cpu >= 0 ? "CPU " + std::to_string(p.caller_cpu) : std::string("the aux set")) +
             ", other threads on CPUs " + list_text(p.aux) + " (workers on " + list_text(worker_cpus) + ")";
    return p;
}

pid_t this_tid()
{
    thread_local const pid_t tid = (pid_t) syscall(SYS_gettid);
    return tid;
}

bool pin_self(int cpu)
{
    if (cpu < 0) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof set, &set) == 0;
}

int sweep_aux(const AuxPlan & p, const std::vector<pid_t> & keep)
{
    if (!p.on) return 0;
    cpu_set_t aux;
    CPU_ZERO(&aux);
    for (int c : p.aux) CPU_SET(c, &aux);
    DIR * d = opendir("/proc/self/task");
    if (!d) return 0;
    int moved = 0;
    while (const dirent * de = readdir(d)) {
        char * end = nullptr;
        const long t = std::strtol(de->d_name, &end, 10);
        if (end == de->d_name || *end || t <= 0) continue;
        const pid_t tid = (pid_t) t;
        if (std::find(keep.begin(), keep.end(), tid) != keep.end()) continue;
        cpu_set_t cur;
        CPU_ZERO(&cur);
        if (sched_getaffinity(tid, sizeof cur, &cur) != 0) continue;   // gone
        if (CPU_EQUAL(&cur, &aux)) continue;
        if (CPU_COUNT(&cur) == 1) {   // pinned by someone: leave it, unless it holds the driver's or caller's CPU
            const bool ours = (p.driver_cpu >= 0 && CPU_ISSET(p.driver_cpu, &cur)) ||
                              (p.caller_cpu >= 0 && CPU_ISSET(p.caller_cpu, &cur));
            if (!ours) continue;
        }
        if (sched_setaffinity(tid, sizeof aux, &aux) == 0) ++moved;
    }
    closedir(d);
    return moved;
}

}  // namespace truss::cpu
