"""Load the trained CANDI checkpoint, load gold text, and compute generative perplexity.

Loads the model from the config saved *inside* the checkpoint -- needs only ``omegaconf``
(not hydra). The trained config carries the right knobs; we top up any algo keys the current
code expects but the older trained schema lacks (from ``configs/algo/candi.yaml``).
"""
import gc
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (parent of denoise_diff/)
if REPO not in sys.path:
    sys.path.insert(0, REPO)                                          # so ``algo`` / ``dataloader`` import


def load_model(ckpt, device, scratch_dir=None, length=None):
    """Return (model, tokenizer, cfg). Drops torch.compile ``_orig_mod`` keys; ignores EMA."""
    from omegaconf import OmegaConf, open_dict
    for name, fn in [("cwd", os.getcwd), ("device_count", torch.cuda.device_count),
                     ("eval", eval), ("div_up", lambda x, y: (x + y - 1) // y)]:
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception:
            pass  # already registered

    print("[load] reading config from checkpoint (omegaconf only; no hydra) ...")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["hyper_parameters"]["config"]
    repo_algo = OmegaConf.load(os.path.join(REPO, "configs", "algo", "candi.yaml"))
    with open_dict(cfg):
        for k in repo_algo:                       # fill missing keys, keep trained values
            if k not in cfg.algo:
                cfg.algo[k] = repo_algo[k]
        if scratch_dir is not None:
            cfg.scratch_dir = scratch_dir
        if length is not None:
            cfg.model.length = length

    import algo
    import dataloader
    tok = dataloader.get_tokenizer(cfg)
    model = algo.CANDI(cfg, tokenizer=tok)
    sd = {k: v for k, v in ck["state_dict"].items() if "_orig_mod" not in k}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    model.ema = None                              # use trained weights directly (see HOWTO for EMA)
    try:
        model.metrics.to(device)
    except Exception:
        pass
    if not hasattr(model, "num_tokens"):
        model.num_tokens = cfg.model.length
    return model, tok, cfg


def load_gold(model, cfg, tok, n_seqs, gold_file=None, gold_hf=None):
    """Gold sequences for exp 4/6/7 -- real text tokenized + chunked to length L.

      gold_file PATH                   any UTF-8 text file
      gold_hf  NAME[:CONFIG[:SPLIT]]   a HF dataset, e.g. 'wikitext:wikitext-2-raw-v1:validation'
      (default)                        OpenWebText via the repo dataloader (needs scratch_dir/owt)
    """
    L = model.num_tokens

    def _chunk(ids):
        ids = torch.tensor(ids, dtype=torch.long)
        n = ids.numel() // L
        if n == 0:
            raise ValueError(f"text too short to fill one sequence of length {L}")
        return ids[: n * L].reshape(n, L)

    if gold_file:
        print(f"[gold] from text file {gold_file}")
        text = open(gold_file, encoding="utf-8", errors="ignore").read()
        seqs = _chunk(tok(text)["input_ids"])
    elif gold_hf:
        from datasets import load_dataset
        parts = (gold_hf.split(":") + [None, None])[:3]
        name, conf, split = parts[0], parts[1], parts[2] or "validation"
        print(f"[gold] from HF dataset {name}/{conf} [{split}]")
        ds = load_dataset(name, conf, split=split)
        col = "text" if "text" in ds.column_names else ds.column_names[0]
        text = "\n".join(t for t in ds[col][:20000] if t)
        seqs = _chunk(tok(text)["input_ids"])
    else:
        print("[gold] from OpenWebText via dataloader (needs scratch_dir holding owt/ cache)")
        import dataloader
        _, valid = dataloader.get_dataloaders(cfg, tok, skip_train=True, valid_seed=cfg.seed)
        rows, got = [], 0
        for batch in valid:
            ids = batch["input_ids"][:, :L]
            if ids.size(1) < L:
                continue
            rows.append(ids); got += ids.size(0)
            if got >= n_seqs:
                break
        seqs = torch.cat(rows, 0)
    return seqs[:n_seqs].to(model.device)


def gen_ppl(model, tok, tokens, L):
    """Generative perplexity (gpt2-large) of decoded ``tokens``; reference-free. None on failure."""
    try:
        texts = tok.batch_decode(tokens)
        model.metrics.gen_ppl.reset()
        model.metrics.record_generative_perplexity(texts, max_length=L, retokenize=True,
                                                    device=str(model.device))
        return float(model.metrics.gen_ppl.compute().item())
    except Exception as e:                                   # gpt2-large download / OOM etc.
        print(f"  [warn] gen_ppl failed: {e}")
        return None
    finally:
        free()


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
