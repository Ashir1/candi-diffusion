"""Denoise-vs-difficulty experiment toolkit for CANDI (read-only probes of the model).

Layout:
  metrics  -- pure functions: entropy, gold-NLL, top-k accuracy, quantile buckets, reveal sets
  harness  -- drive the unmodified CANDI model: trace / probe / gold-reconstruct / gold-renoise
  model    -- load the checkpoint, load gold text, generative perplexity
  plotting -- shared figure helpers (save, grouped bars, colours, run loading)
See ../run_experiments.py (collect data) and ../analyze.py (make figures), README_experiments.md.
"""
