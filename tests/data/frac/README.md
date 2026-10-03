Fractional-K decode fixtures (TRACKER #118, E3). From the X3.1 smoke encode of Flash-Next L24 expert 0
(exllamav3 v1.5.1, `b1_encode.py`): `k<K>_<proj>.trellis.u16` = the first 8 k-tile rows x first 40 n-tile columns of
the packed tiles ([8][40][16*K] uint16, k-slice major), `k<K>_<proj>.recon.f16` = exllamav3 `ext.reconstruct`
(rotated-basis codebook values, no Hadamard/scales) of the full matrix, rows 0..127 and columns 0..639 ([128][640]
fp16, input-major). K25 = 2.5 (KA 2, MASK 0xAAAA, 40 words/tile), K35 = 3.5 (KA 3, 56 words/tile).
