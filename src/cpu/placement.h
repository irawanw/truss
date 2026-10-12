// Thread placement beside the pinned CPU expert pool (TRUSS_AUX_CPUS; Strata 0.1.41's --aux-cpus).
//
// With TRUSS_CPU_PIN=1 each pool worker owns one logical CPU of its own physical core, but every other thread of the
// process floats: the driver thread, the thread that launches the GPU work, CUDA's own threads, the PLE readers, the
// host page-lock thread and the Python server's threads. One of them landing on a worker's CPU preempts that worker,
// and the whole phase barrier waits for it (memory: cpu-tier-straggler). TRUSS_AUX_CPUS moves those threads off the
// workers' cores; it moves threads only and never changes a result.
#pragma once

#include <sys/types.h>

#include <string>
#include <vector>

namespace truss::cpu {

// Physical cores this process may run on, in core order, each with its logical CPUs (sysfs). Read once, on the first
// call (the pool's constructor), before any thread is moved, so a later call is not narrowed by a moved thread's mask.
const std::vector<std::vector<int>> & core_cpus();

struct AuxPlan {
    bool on = false;
    int driver_cpu = -1;    // the driver thread (serves the doorbells, runs pool items): the pool's reserved core 0
    int caller_cpu = -1;    // the thread that runs the passes (GPU launches): a core with no worker; -1: aux
    std::vector<int> aux;   // every other thread
    std::string text;       // the start-up line
};

// TRUSS_AUX_CPUS: unset, "0" or "off": off. "auto" (or "1"): the SMT siblings of core 0, then the cores with no
// worker (the first one's first CPU takes the caller). A list ("40,41" or "44-47") is used as the aux set as given,
// minus the driver's, the caller's and the workers' CPUs. worker_cpus empty (the pool is not pinned): off.
AuxPlan plan_aux(const std::vector<int> & worker_cpus);

pid_t this_tid();
bool pin_self(int cpu);
// every thread of this process except keep goes to p.aux; a thread someone pinned to one CPU stays, unless that CPU
// is the driver's or the caller's (an earlier caller). Returns how many threads were moved.
int sweep_aux(const AuxPlan & p, const std::vector<pid_t> & keep);

}  // namespace truss::cpu
