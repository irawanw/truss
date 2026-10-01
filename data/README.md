# data/

| file | what | source |
|---|---|---|
| `draft_vocab_en.bin` | 40,525 int32 token ids (Qwen3.8 Flash-Next vocabulary): the English/code subset the MTP draft head scores (`Forward::Options::draft_vocab_ids`, `truss_params.draft_vocab`) | Strata `data/draft_vocab_en.bin` (commit d6708a4), MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors; built by Strata's `tools/draft_vocab.py` |
| `usage_strata_rank.f32` | 48 × 512 float32, layer-major: Strata's shipped expert ranking as TRUSS usage scores (rank r of 24,576 → 24,576 − r), for `ExpertStore::plan` / `tk-bench-spec`'s usage argument. On the bench prompt it puts 9,960 experts in VRAM instead of 6,728 (our 8-prompt `usage_code_truss8.f32`) and halves the cold routings (TRACKER #78) | Strata `data/expert-profile.bin` (`STRP` v1, 24,576 ranked (layer, expert) pairs; commit 9259cad), MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors; converted by the snippet in `flashnext/20261001_strata_profile/reports/STRATA_PROFILE.md` §7 |
