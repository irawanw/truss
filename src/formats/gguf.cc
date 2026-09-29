// GGUF v3 parser (spec: ggml/docs/gguf.md). Header, key/values and tensor infos are parsed from the mapped bytes
// with bounds checks; tensor data is never copied.
#include "formats/gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace truss::gguf {

namespace {

constexpr uint32_t MAGIC = 0x46554747;     // "GGUF" little-endian
constexpr uint32_t VERSION = 3;
constexpr uint64_t DEFAULT_ALIGNMENT = 32;

enum class ValueType : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, U64 = 10,
    I64 = 11, F64 = 12,
};

[[noreturn]] void fail(const std::string & path, const std::string & why)
{
    throw std::runtime_error("gguf: " + path + ": " + why);
}

// Sequential bounds-checked reader over one mapped shard.
class Cursor {
public:
    Cursor(const std::byte * p, size_t n, const std::string & path) : p_(p), n_(n), path_(path) {}

    template <class T> T pod()
    {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    std::string str()
    {
        const uint64_t len = pod<uint64_t>();
        need(len);
        std::string s(reinterpret_cast<const char *>(p_ + pos_), len);
        pos_ += len;
        return s;
    }

    size_t pos() const { return pos_; }
    const std::string & path() const { return path_; }

private:
    void need(uint64_t k) const
    {
        if (k > n_ - pos_) fail(path_, "truncated header at byte " + std::to_string(pos_));
    }

    const std::byte * p_;
    size_t n_, pos_ = 0;
    const std::string & path_;
};

int64_t read_int(Cursor & c, ValueType t)
{
    switch (t) {
        case ValueType::U8: return c.pod<uint8_t>();
        case ValueType::I8: return c.pod<int8_t>();
        case ValueType::U16: return c.pod<uint16_t>();
        case ValueType::I16: return c.pod<int16_t>();
        case ValueType::U32: return c.pod<uint32_t>();
        case ValueType::I32: return c.pod<int32_t>();
        case ValueType::I64: return c.pod<int64_t>();
        case ValueType::U64: {
            const uint64_t v = c.pod<uint64_t>();
            if (v > (uint64_t) std::numeric_limits<int64_t>::max()) fail(c.path(), "uint64 value out of range");
            return (int64_t) v;
        }
        default: fail(c.path(), "not an integer value type");
    }
}

bool is_int(ValueType t)
{
    return t == ValueType::U8 || t == ValueType::I8 || t == ValueType::U16 || t == ValueType::I16 ||
           t == ValueType::U32 || t == ValueType::I32 || t == ValueType::U64 || t == ValueType::I64;
}

double read_float(Cursor & c, ValueType t)
{
    return t == ValueType::F32 ? (double) c.pod<float>() : c.pod<double>();
}

Value read_value(Cursor & c, ValueType t)
{
    if (is_int(t)) return read_int(c, t);
    if (t == ValueType::F32 || t == ValueType::F64) return read_float(c, t);
    if (t == ValueType::BOOL) return c.pod<uint8_t>() != 0;
    if (t == ValueType::STRING) return c.str();
    if (t != ValueType::ARRAY) fail(c.path(), "unknown value type " + std::to_string((uint32_t) t));
    const auto et = (ValueType) c.pod<uint32_t>();
    const uint64_t n = c.pod<uint64_t>();
    if (is_int(et)) {
        std::vector<int64_t> v(n);
        for (auto & x : v) x = read_int(c, et);
        return v;
    }
    if (et == ValueType::F32 || et == ValueType::F64) {
        std::vector<double> v(n);
        for (auto & x : v) x = read_float(c, et);
        return v;
    }
    if (et == ValueType::BOOL) {
        std::vector<bool> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = c.pod<uint8_t>() != 0;
        return v;
    }
    if (et == ValueType::STRING) {
        std::vector<std::string> v(n);
        for (auto & x : v) x = c.str();
        return v;
    }
    fail(c.path(), "unsupported array element type " + std::to_string((uint32_t) et));
}

// "<stem>-0000N-of-0000M.gguf" -> (stem, M); M = 1 for a single file.
std::pair<std::string, int> split_name(const std::string & path)
{
    const std::string tail = "-00001-of-00000.gguf";
    if (path.size() > tail.size()) {
        const std::string t = path.substr(path.size() - tail.size());
        int n, m;
        char dot[6] = {};
        if (std::sscanf(t.c_str(), "-%5d-of-%5d.%5s", &n, &m, dot) == 3 && std::string(dot) == "gguf" && n >= 1 &&
            n <= m)
            return { path.substr(0, path.size() - tail.size()), m };
    }
    return { path, 1 };
}

std::string shard_path(const std::string & stem, int i, int m)
{
    char buf[32];
    std::snprintf(buf, sizeof buf, "-%05d-of-%05d.gguf", i + 1, m);
    return stem + buf;
}

}  // namespace

const TypeInfo * type_info(Type t)
{
    static const TypeInfo table[] = {
        { "F32", 1, 4 },      { "F16", 1, 2 },       { "Q4_0", 32, 18 },    { "Q4_1", 32, 20 },
        { nullptr, 0, 0 },    { nullptr, 0, 0 },     { "Q5_0", 32, 22 },    { "Q5_1", 32, 24 },
        { "Q8_0", 32, 34 },   { "Q8_1", 32, 40 },    { "Q2_K", 256, 84 },   { "Q3_K", 256, 110 },
        { "Q4_K", 256, 144 }, { "Q5_K", 256, 176 },  { "Q6_K", 256, 210 },  { "Q8_K", 256, 292 },
        { "IQ2_XXS", 256, 66 }, { "IQ2_XS", 256, 74 }, { "IQ3_XXS", 256, 98 }, { "IQ1_S", 256, 50 },
        { "IQ4_NL", 32, 18 }, { "IQ3_S", 256, 110 }, { "IQ2_S", 256, 82 },  { "IQ4_XS", 256, 136 },
        { "I8", 1, 1 },       { "I16", 1, 2 },       { "I32", 1, 4 },       { "I64", 1, 8 },
        { "F64", 1, 8 },      { "IQ1_M", 256, 56 },  { "BF16", 1, 2 },      { nullptr, 0, 0 },
        { nullptr, 0, 0 },    { nullptr, 0, 0 },     { "TQ1_0", 256, 54 },  { "TQ2_0", 256, 66 },
        { nullptr, 0, 0 },    { nullptr, 0, 0 },     { nullptr, 0, 0 },     { "MXFP4", 32, 17 },
        { "NVFP4", 64, 36 },  { "Q1_0", 128, 18 },   { "Q2_0", 64, 18 },
    };
    const auto i = (uint32_t) t;
    return i < std::size(table) && table[i].name ? &table[i] : nullptr;
}

int64_t Tensor::elements() const
{
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

std::unique_ptr<File> File::open(const std::string & path)
{
    std::unique_ptr<File> f(new File());
    const auto [stem, count] = split_name(path);
    if (count == 1) {
        f->add_shard(path, true);
    } else {
        for (int i = 0; i < count; ++i) f->add_shard(shard_path(stem, i, count), i == 0);
    }
    if (count > 1 && f->get_int("split.tensors.count") != (int64_t) f->tensors_.size())
        fail(path, "split.tensors.count " + std::to_string(f->get_int("split.tensors.count")) + " but shards hold " +
                   std::to_string(f->tensors_.size()));
    return f;
}

void File::add_shard(const std::string & path, bool first)
{
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) fail(path, std::strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        fail(path, std::strerror(errno));
    }
    void * addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (addr == MAP_FAILED) fail(path, std::string("mmap: ") + std::strerror(errno));
    const int shard = (int) maps_.size();
    maps_.push_back({ addr, (size_t) st.st_size });
    paths_.push_back(path);

    const auto * base = static_cast<const std::byte *>(addr);
    Cursor c(base, st.st_size, path);
    if (c.pod<uint32_t>() != MAGIC) fail(path, "not a GGUF file");
    if (const uint32_t v = c.pod<uint32_t>(); v != VERSION) fail(path, "GGUF version " + std::to_string(v));
    const uint64_t n_tensors = c.pod<uint64_t>(), n_kv = c.pod<uint64_t>();

    uint64_t alignment = DEFAULT_ALIGNMENT;
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = c.str();
        Value v = read_value(c, (ValueType) c.pod<uint32_t>());
        if (key == "general.alignment") {
            if (!std::holds_alternative<int64_t>(v) || std::get<int64_t>(v) <= 0) fail(path, "bad general.alignment");
            alignment = std::get<int64_t>(v);
        }
        if (key == "split.no" && std::get<int64_t>(v) != shard) fail(path, "split.no does not match file name");
        if (first) kv_.emplace(std::move(key), std::move(v));
    }

    const size_t first_tensor = tensors_.size();
    for (uint64_t i = 0; i < n_tensors; ++i) {
        Tensor t;
        t.name = c.str();
        const uint32_t n_dims = c.pod<uint32_t>();
        if (n_dims == 0 || n_dims > 4) fail(path, t.name + ": " + std::to_string(n_dims) + " dims");
        t.shape.resize(n_dims);
        for (auto & d : t.shape) {
            const uint64_t v = c.pod<uint64_t>();
            if (v == 0 || v > (1ull << 40)) fail(path, t.name + ": bad dim " + std::to_string(v));
            d = (int64_t) v;
        }
        t.type = (Type) c.pod<uint32_t>();
        const TypeInfo * ti = type_info(t.type);
        if (!ti) fail(path, t.name + ": unknown ggml type " + std::to_string((uint32_t) t.type));
        if (t.shape[0] % ti->block) fail(path, t.name + ": row not a whole number of " + ti->name + " blocks");
        t.bytes = (uint64_t) (t.elements() / ti->block) * ti->bytes;
        t.file_offset = c.pod<uint64_t>();       // relative to the data section for now
        t.shard = shard;
        if (index_.count(t.name)) fail(path, "duplicate tensor " + t.name);
        index_.emplace(t.name, tensors_.size());
        tensors_.push_back(std::move(t));
    }

    const uint64_t data_start = (c.pos() + alignment - 1) / alignment * alignment;
    for (size_t i = first_tensor; i < tensors_.size(); ++i) {
        Tensor & t = tensors_[i];
        if (t.file_offset % alignment) fail(path, t.name + ": misaligned data offset");
        t.file_offset += data_start;
        if (t.file_offset > (uint64_t) st.st_size || t.bytes > (uint64_t) st.st_size - t.file_offset)
            fail(path, t.name + ": data past end of file");
        t.data = base + t.file_offset;
    }
}

File::~File()
{
    for (auto & m : maps_) munmap(m.addr, m.size);
}

const Tensor * File::find(std::string_view name) const
{
    const auto it = index_.find(std::string(name));
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

const Tensor & File::at(std::string_view name) const
{
    if (const Tensor * t = find(name)) return *t;
    fail(paths_.front(), "no tensor " + std::string(name));
}

const Value * File::get(std::string_view key) const
{
    const auto it = kv_.find(std::string(key));
    return it == kv_.end() ? nullptr : &it->second;
}

namespace {
template <class T> const T & typed(const File & f, std::string_view key, const char * kind)
{
    const Value * v = f.get(key);
    if (!v) fail(f.shard_paths().front(), "no metadata key " + std::string(key));
    if (!std::holds_alternative<T>(*v)) fail(f.shard_paths().front(), std::string(key) + " is not " + kind);
    return std::get<T>(*v);
}
}  // namespace

int64_t File::get_int(std::string_view key) const { return typed<int64_t>(*this, key, "an integer"); }

double File::get_float(std::string_view key) const
{
    const Value * v = get(key);
    if (v && std::holds_alternative<int64_t>(*v)) return (double) std::get<int64_t>(*v);
    return typed<double>(*this, key, "a float");
}

const std::string & File::get_string(std::string_view key) const
{
    return typed<std::string>(*this, key, "a string");
}

const std::vector<int64_t> & File::get_ints(std::string_view key) const
{
    return typed<std::vector<int64_t>>(*this, key, "an integer array");
}

const std::vector<double> & File::get_floats(std::string_view key) const
{
    return typed<std::vector<double>>(*this, key, "a float array");
}

const std::vector<bool> & File::get_bools(std::string_view key) const
{
    return typed<std::vector<bool>>(*this, key, "a bool array");
}

}  // namespace truss::gguf
