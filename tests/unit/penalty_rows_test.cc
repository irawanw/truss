// sampling::penalty_rows (host): each verify row's penalty history against a direct construction - row t is the last
// w tokens of history + feed[0 .. t] - over history lengths shorter and longer than w, and nfeed 0 (truss_eval_sample).
#include "kernels/sampling/penalty.cuh"

#include <cstdio>
#include <vector>

using namespace truss;

int main()
{
    int fails = 0, cases = 0;
    for (int nh : {0, 3, 10, 64, 200})
        for (int w : {1, 8, 64})
            for (int nfeed : {0, 1, 4})
                for (int rows : {1, 4}) {
                    if (nfeed && rows > nfeed) continue;
                    std::vector<int32_t> hist(nh), feed(nfeed);
                    for (int i = 0; i < nh; ++i) hist[i] = 100 + i;
                    for (int i = 0; i < nfeed; ++i) feed[i] = 900 + i;
                    std::vector<int> out((size_t) rows * w);
                    sampling::penalty_rows(hist.data(), nh, feed.data(), nfeed, rows, w, out.data());
                    for (int t = 0; t < rows; ++t) {
                        std::vector<int> seq(hist.begin(), hist.end());
                        for (int i = 0; i < nfeed && i <= t; ++i) seq.push_back(feed[i]);
                        std::vector<int> want(w, -1);
                        for (int i = 0; i < w && i < (int) seq.size(); ++i) want[w - 1 - i] = seq[seq.size() - 1 - i];
                        fails += std::vector<int>(out.begin() + (size_t) t * w, out.begin() + (size_t) (t + 1) * w) != want;
                        ++cases;
                    }
                }
    std::printf("penalty_rows: %d rows checked, %d wrong  %s\n", cases, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
