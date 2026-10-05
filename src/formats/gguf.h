// GGUF v3 reader: memory-maps every shard of a (split) model file and exposes its metadata and tensors as one
// catalog. Read-only and zero-copy: a tensor's bytes are a pointer into the mapping, so the loader and the pack
// converter read from here without an intermediate buffer.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace truss::gguf {

// ggml tensor types, numbered as in ggml.h (the file stores these numbers).
enum class Type : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
    I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30,
    TQ1_0 = 34, TQ2_0 = 35, MXFP4 = 39, NVFP4 = 40, Q1_0 = 41, Q2_0 = 42,
};

struct TypeInfo {
    const char * name;
    int block;                             // elements per block
    int bytes;                             // bytes per block
};

// nullptr for a number this reader does not know (the file is then rejected at open).
const TypeInfo * type_info(Type t);

struct Tensor {
    std::string name;
    Type type;
    std::vector<int64_t> shape;            // ggml order: shape[0] is the contiguous dimension
    int shard;                             // index into File::shard_paths()
    uint64_t file_offset;                  // absolute, in that shard
    uint64_t bytes;
    const std::byte * data;                // into the mapping

    int64_t elements() const;
};

// A metadata value. Integers of every width are widened to int64 (uint64 values above INT64_MAX are rejected),
// floats to double; arrays hold one element kind.
using Value = std::variant<int64_t, double, bool, std::string, std::vector<int64_t>, std::vector<double>,
                           std::vector<bool>, std::vector<std::string>>;

// Copies n bytes at p, which points into a mapping of some open File, to dst with pread() on that shard instead of
// through the mapping: no page faults and no file pages mapped (RSS) for a bulk copy. False if p is not in a mapping
// of an open File or a read fails (the caller then copies through the mapping). Thread-safe.
bool read_mapped(const void * p, size_t n, void * dst);

class File {
public:
    // path: a single-file model or any shard of a split one ("...-0000N-of-0000M.gguf"); every shard is opened.
    // Throws std::runtime_error with the path and the reason on any malformed or missing input.
    static std::unique_ptr<File> open(const std::string & path);
    ~File();
    File(const File &) = delete;
    File & operator=(const File &) = delete;

    // Overlay (TRACKER #118, B7): map another GGUF (single or split) whose tensors replace this file's tensors of the
    // same name (and add any new ones); its metadata is ignored. The X3.1 experts ship as an experts-only GGUF over
    // the served one. Call before binding weights.
    void overlay(const std::string & path);
    size_t overlaid() const { return overlaid_.size(); }

    const std::vector<Tensor> & tensors() const { return tensors_; }
    const Tensor * find(std::string_view name) const;   // nullptr if absent
    const Tensor & at(std::string_view name) const;     // throws if absent

    const std::unordered_map<std::string, Value> & metadata() const { return kv_; }
    const Value * get(std::string_view key) const;      // nullptr if absent
    // Typed access; throws if the key is absent or holds another kind.
    int64_t get_int(std::string_view key) const;
    double get_float(std::string_view key) const;       // also accepts an integer value
    const std::string & get_string(std::string_view key) const;
    const std::vector<int64_t> & get_ints(std::string_view key) const;
    const std::vector<double> & get_floats(std::string_view key) const;
    const std::vector<bool> & get_bools(std::string_view key) const;

    const std::vector<std::string> & shard_paths() const { return paths_; }

    // Drop the file-backed pages of every shard from the process. The mapping stays valid (PROT_READ | MAP_SHARED):
    // a later touch just re-faults from disk, so this is safe once the tensors have been uploaded. Worth doing after
    // Forward's upload() — the shards are tens of GB of RSS that would otherwise sit resident beside every run
    // (TRACKER #72).
    void release_pages() const;

private:
    struct Mapping { void * addr; size_t size; int fd; };
    File() = default;
    void add_shard(const std::string & path, bool first, bool overlay = false);

    std::vector<std::string> paths_;
    std::vector<Mapping> maps_;
    std::vector<Tensor> tensors_;
    std::unordered_map<std::string, size_t> index_;
    std::unordered_map<std::string, Value> kv_;          // from shard 0; split.* keys of later shards are checked
    std::unordered_set<std::string> overlaid_;           // names replaced by overlay()
};

}  // namespace truss::gguf
