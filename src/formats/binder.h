// Binds a model file's tensors to a family's weight views: every tensor checked against its expected shape (ggml
// order, trailing 1s ignored), every trellis expert table read and checked, and - at the end - every tensor in the
// file used exactly once, so a converter change that adds, drops or reshapes a tensor fails at load instead of
// computing garbage. Shared by every model family's weights.cc.
#pragma once
#include "formats/gguf.h"
#include "formats/trellis_table.h"

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace truss::formats {

class Binder {
public:
    Binder(const gguf::File & f, std::string family) : f_(f), fam_(std::move(family)) {}

    // The tensor, checked against `dims` (trailing 1s ignored on both sides). A dense matrix may also be a trellis
    // group ("<name>.m3_trellis" with its scales): then the handle is the group's words tensor and the parts are
    // bound here, so a model family names its weights the same way whatever format the file holds. A group wins over
    // a plain tensor of the same name (the X3.1 dense overlay shadows the base's Q8 tensor with a placeholder: the
    // group is the matrix).
    const gguf::Tensor * need(const std::string & name, std::initializer_list<int64_t> dims)
    {
        if (const gguf::Tensor * w = f_.find(name + ".m3_trellis")) return need_trellis(name, w, dims);
        const gguf::Tensor & t = f_.at(name);
        std::vector<int64_t> want(dims), have(t.shape);
        while (want.size() > 1 && want.back() == 1) want.pop_back();
        while (have.size() > 1 && have.back() == 1) have.pop_back();
        if (want != have) throw std::runtime_error(fam_ + ": " + name + " has shape " + str(have) + ", expected " +
                                                   str(want));
        use(name);
        return &t;
    }

    // A trellis-coded dense matrix: the handle is the group's words tensor, the parts are bound (so every tensor of
    // the file is used exactly once).
    const gguf::Tensor * need_trellis(const std::string & name, const gguf::Tensor * words,
                                      std::initializer_list<int64_t> dims)
    {
        const gguf::TrellisGroup * g = words->trellis;
        if (!g) throw std::runtime_error(fam_ + ": " + name + " is not a trellis matrix");
        std::vector<int64_t> want(dims), have{ g->in, g->out };
        while (want.size() > 1 && want.back() == 1) want.pop_back();
        while (have.size() > 1 && have.back() == 1) have.pop_back();
        if (want != have)
            throw std::runtime_error(fam_ + ": " + name + " is " + std::to_string(g->out) + "x" + std::to_string(g->in) +
                                     ", expected " + str(want));
        use(name + ".m3_trellis");
        use(name + ".m3_suh");
        use(name + ".m3_svh");
        if (g->meta) use(name + ".m3_meta");
        if (f_.find(name)) use(name);   // the shadowed plain tensor (the overlay's placeholder): bound, carries nothing
        return words;
    }

    formats::ExpertTable experts(const std::string & prefix, int64_t in, int64_t out, int n_expert)
    {
        formats::ExpertTable e = formats::read_expert_table(f_, prefix);
        if (e.in != in || e.out != out || e.n_expert != n_expert)
            throw std::runtime_error(fam_ + ": " + prefix + " is " + std::to_string(e.n_expert) + " x " +
                                     std::to_string(e.out) + "x" + std::to_string(e.in) + ", expected " +
                                     std::to_string(n_expert) + " x " + std::to_string(out) + "x" + std::to_string(in));
        for (const char * s : { ".m3_trellis", ".m3_meta", ".m3_suh", ".m3_svh" }) use(prefix + s);
        return e;
    }

    // the tensor if the file has it (checked like need()), else nullptr
    const gguf::Tensor * maybe(const std::string & name, std::initializer_list<int64_t> dims)
    {
        return f_.find(name) ? need(name, dims) : nullptr;
    }

    void check_all_used() const
    {
        std::string unused;
        int n = 0;
        for (const gguf::Tensor & t : f_.tensors())
            if (!used_.count(t.name) && n++ < 8) unused += " " + t.name;
        if (n) throw std::runtime_error(fam_ + ": " + std::to_string(n) + " tensors not bound:" + unused);
    }

private:
    void use(const std::string & name)
    {
        if (!used_.insert(name).second) throw std::runtime_error(fam_ + ": " + name + " bound twice");
    }

    static std::string str(const std::vector<int64_t> & v)
    {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
        return s + "]";
    }

    const gguf::File & f_;
    const std::string fam_;
    std::unordered_set<std::string> used_;
};

}  // namespace truss::formats
