// The kept cold copy (TRUSS_COLD_KEEP=<unix socket>): an ExpertStore's host cold copy lives in a memfd that a keeper
// process (tools/cold_keeper.py) holds between engine runs, so a restart maps the bytes instead of reading them again.
//
// Why: the DS41 cold copy is ~82 GB and this box's NVMe reads ~1 GB/s (power state capped, 10-21), so every load spent
// ~85 s reading it. With the keeper the next load of the same model and plan maps it in well under a second.
//
// Protocol (one request per connection, a text line each way, the fd as SCM_RIGHTS):
//   GET <slot> <key> <bytes>\n   -> "HIT\n" + fd | "MISS\n"   (a miss drops the slot's old copy, and evicts least
//                                                                recently used ones to fit <bytes> under the cap)
//   PUT <slot> <key> <bytes>\n + fd -> "OK\n"
// slot names the store (which file spans its layers come from), key its exact contents (slot + the cold expert list
// + size; the file identity in slot covers path, size and mtime: a rewritten pack misses). The copy is read-only once
// filled; only a store that finished its fill puts it.
//
// Every failure (no socket, keeper gone, a bad reply) returns -1 / false and the caller fills a private copy as
// before: the keeper is an accelerator, never needed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace truss::runtime::cold_keep {

// the socket path, or "" when TRUSS_COLD_KEEP is unset or "0"
std::string socket_path();

// the kept copy's fd (the caller maps and closes it) or -1
int get(const std::string & sock, uint64_t slot, uint64_t key, size_t bytes);

// hands a filled copy over (the keeper dups the fd; the caller still closes its own)
bool put(const std::string & sock, uint64_t slot, uint64_t key, size_t bytes, int fd);

}  // namespace truss::runtime::cold_keep
