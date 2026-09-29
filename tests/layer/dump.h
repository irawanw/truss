// Reader for tools/tk-parity/llama_dump output: index.tsv (name, dtype, ggml shape, file) + raw little-endian data.
#pragma once
#include <array>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace truss::test {

struct DumpTensor {
    std::array<int64_t, 4> ne;
    std::vector<float> f;                  // f32 tensors
    std::vector<int32_t> i;                // i32 tensors
    int64_t elements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};

class Dump {
public:
    explicit Dump(const std::string & dir) : dir_(dir)
    {
        std::ifstream in(dir + "/index.tsv");
        if (!in) throw std::runtime_error("dump: no index.tsv in " + dir);
        for (std::string line; std::getline(in, line);) {
            std::stringstream ss(line);
            std::string name, dtype, shape, file;
            std::getline(ss, name, '\t');
            std::getline(ss, dtype, '\t');
            std::getline(ss, shape, '\t');
            std::getline(ss, file, '\t');
            Entry e{ dtype, {}, file };
            std::sscanf(shape.c_str(), "%ld,%ld,%ld,%ld", &e.ne[0], &e.ne[1], &e.ne[2], &e.ne[3]);
            index_[name] = e;
        }
    }

    bool has(const std::string & name) const { return index_.count(name) != 0; }

    DumpTensor get(const std::string & name) const
    {
        const auto it = index_.find(name);
        if (it == index_.end()) throw std::runtime_error("dump: no tensor " + name);
        const Entry & e = it->second;
        DumpTensor t{ e.ne, {}, {} };
        std::ifstream f(dir_ + "/" + e.file, std::ios::binary);
        if (e.dtype == "i32") {
            t.i.resize(t.elements());
            f.read(reinterpret_cast<char *>(t.i.data()), t.i.size() * 4);
        } else {
            t.f.resize(t.elements());
            f.read(reinterpret_cast<char *>(t.f.data()), t.f.size() * 4);
        }
        if (!f) throw std::runtime_error("dump: short read " + e.file);
        return t;
    }

private:
    struct Entry { std::string dtype; std::array<int64_t, 4> ne; std::string file; };
    std::string dir_;
    std::map<std::string, Entry> index_;
};

}  // namespace truss::test
