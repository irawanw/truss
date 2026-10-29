// Fixture helpers shared by the vision checks (vit_check, ds41_vit_check): read a little-endian f4 / i8 .npy as
// floats, and compare a tap with its reference (cosine, max |diff|).
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static std::vector<float> npy(const std::string & p, std::vector<long> * shape = nullptr)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
    char m[10]; f.read(m, 10);
    unsigned hl = (unsigned char) m[8] | ((unsigned char) m[9] << 8);
    std::string h(hl, ' '); f.read(&h[0], hl);
    if (h.find("<f4") == std::string::npos && h.find("<i8") == std::string::npos) { fprintf(stderr, "%s: dtype not f4/i8: %s\n", p.c_str(), h.c_str()); exit(2); }
    std::vector<long> sh; size_t a = h.find("shape"), b = h.find('(', a), c = h.find(')', b);
    for (size_t i = b + 1; i < c;) { while (i < c && (h[i] == ' ' || h[i] == ',')) i++; if (i >= c) break; sh.push_back(strtol(h.c_str() + i, nullptr, 10)); while (i < c && h[i] != ',') i++; }
    long n = 1; for (long s : sh) n *= s;
    std::vector<float> v(n);
    if (h.find("<i8") != std::string::npos) { std::vector<long> t(n); f.read((char *) t.data(), n * 8); for (long i = 0; i < n; i++) v[i] = (float) t[i]; }
    else f.read((char *) v.data(), n * 4);
    if (shape) *shape = sh;
    return v;
}

static bool cmp(const char * name, const std::vector<float> & ref, const std::vector<float> & got, double need)
{
    if (ref.size() != got.size()) { printf("%-12s SIZE MISMATCH ref %zu got %zu\n", name, ref.size(), got.size()); return false; }
    double dot = 0, na = 0, nb = 0, mx = 0, amax = 0;
    for (size_t i = 0; i < ref.size(); i++) { dot += (double) ref[i] * got[i]; na += (double) ref[i] * ref[i]; nb += (double) got[i] * got[i]; mx = std::fmax(mx, std::fabs(ref[i] - got[i])); amax = std::fmax(amax, std::fabs(ref[i])); }
    const double cosv = dot / std::sqrt(na * nb + 1e-30);
    printf("%-12s cos %.6f  max|diff| %.4g  (ref absmax %.4g, rel %.2e)  %s\n", name, cosv, mx, amax, mx / amax, cosv >= need ? "ok" : "FAIL");
    return cosv >= need;
}
