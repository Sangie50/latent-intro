"""Tuned lens for the Section 5.1 (H1) follow-up.

  train  Fit one translator per layer for a single checkpoint, so that
         lm_head(norm(translator_l(h_l))) matches the model's final next-token
         distribution (KL) on generic text. Needs the GPU and the full model.
  apply  Re-read a run's saved hidden states (hiddens_<phase>.pt from run_model.py)
         through a trained lens and write results_tuned.json, in the same record
         format as results.json. Only loads the final norm and unembedding.

Translators are h + (h @ V) @ U.T + b with rank --rank, initialized to the identity
(i.e. to the logit lens). A full d x d map per layer does not fit next to a 32B model
on one GPU; the low-rank + bias form keeps most of the tuned lens's benefit.

Corpus: --text-file (plain text, documents separated by blank lines). If that file
does not exist, falls back to wikitext-103 via the `datasets` package if installed.

Run from the repo root, e.g.:
  python posttrain/tuned_lens.py train --model Qwen/Qwen2.5-Coder-7B
  python posttrain/tuned_lens.py apply --lens outputs/posttrain/lenses/Qwen_Qwen2.5-Coder-7B.pt \
      --run-dir outputs/posttrain/7b-base-plain-vinst-s42
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

PHASES = ["main", "controls", "baseline", "confusion"]
LENS_DIR = "outputs/posttrain/lenses"
LP_DECIMALS = 4


def lens_path(model_id: str) -> Path:
    return Path(LENS_DIR) / f"{model_id.replace('/', '_')}.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--tokenizer", help="Default: --model")
    t.add_argument("--out", help=f"Default: {LENS_DIR}/<model>.pt")
    t.add_argument("--text-file", default="data/lens_corpus.txt")
    t.add_argument("--rank", type=int, default=64)
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--eval-batches", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)

    a = sub.add_parser("apply")
    a.add_argument("--lens", help="Default: the lens for the run's --model")
    a.add_argument("--run-dir", required=True, nargs="+")
    a.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Translators


def translate(h: torch.Tensor, U: torch.Tensor, V: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h: [..., d] float32; U, V: [d, r]; b: [d]."""
    return h + (h @ V) @ U.T + b


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


# ---------------------------------------------------------------------------
# train


def load_corpus(path: str) -> list[str]:
    p = Path(path)
    if p.exists():
        docs = [d.strip() for d in p.read_text().split("\n\n")]
        return [d for d in docs if d]
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        sys.exit(
            f"No corpus at {path} and `datasets` is not installed. Provide a plain-text "
            "file of generic English (a few MB; documents separated by blank lines)."
        )
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    return [t.strip() for t in ds["text"] if len(t.strip()) > 200]


def token_chunks(tokenizer: Any, docs: list[str], seq_len: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    docs = [docs[i] for i in rng.permutation(len(docs))]
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    for d in docs:
        ids.extend(tokenizer.encode(d, add_special_tokens=False))
        ids.append(eos)
    n = len(ids) // seq_len
    return torch.tensor(ids[: n * seq_len]).view(n, seq_len)


def lens_kl(
    model: Any, h: torch.Tensor, target_logp: torch.Tensor, U, V, b
) -> torch.Tensor:
    """Mean KL(target || lens) over tokens for one layer."""
    t = translate(h.float(), U, V, b).to(model.lm_head.weight.dtype)
    logq = model.lm_head(model.model.norm(t)).float().log_softmax(-1)
    return (target_logp.exp() * (target_logp - logq)).sum(-1).mean()


def train(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda:0", torch_dtype=torch.bfloat16
    )
    model.eval()
    model.requires_grad_(False)
    dev = model.device
    n_layers = len(model.model.layers)
    d = model.config.hidden_size

    chunks = token_chunks(tokenizer, load_corpus(args.text_file), args.seq_len, args.seed)
    n_eval = args.eval_batches * args.batch_size
    needed = n_eval + args.steps * args.batch_size
    if len(chunks) < needed:
        print(f"Corpus has {len(chunks)} chunks, need {needed}; cycling training data.")
    eval_chunks, train_chunks = chunks[:n_eval], chunks[n_eval:]
    assert len(train_chunks) > 0, "Corpus too small"

    # Translator i reads hidden_states[i] (the residual entering layer i).
    U = torch.zeros(n_layers, d, args.rank, device=dev, requires_grad=True)
    V = (torch.randn(n_layers, d, args.rank, device=dev) / d**0.5).requires_grad_(True)
    b = torch.zeros(n_layers, d, device=dev, requires_grad=True)
    opt = torch.optim.Adam([U, V, b], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / args.steps)
    )

    def evaluate() -> tuple[list[float], list[float]]:
        """Per-layer KL for the logit lens (identity) and the current tuned lens."""
        logit_kl = np.zeros(n_layers)
        tuned_kl = np.zeros(n_layers)
        zU = torch.zeros_like(U[0])
        zb = torch.zeros_like(b[0])
        with torch.no_grad():
            for i in range(0, n_eval, args.batch_size):
                batch = eval_chunks[i : i + args.batch_size].to(dev)
                out = model(input_ids=batch, output_hidden_states=True)
                target = out.logits.float().log_softmax(-1)
                for layer in range(n_layers):
                    h = out.hidden_states[layer]
                    logit_kl[layer] += lens_kl(model, h, target, zU, V[layer], zb).item()
                    tuned_kl[layer] += lens_kl(model, h, target, U[layer], V[layer], b[layer]).item()
        n_batches = max(1, n_eval // args.batch_size)
        return (logit_kl / n_batches).tolist(), (tuned_kl / n_batches).tolist()

    logit_kl, _ = evaluate()
    print("eval KL, logit lens:", " ".join(f"{x:.2f}" for x in logit_kl), flush=True)

    for step in range(args.steps):
        idx = torch.arange(step * args.batch_size, (step + 1) * args.batch_size) % len(train_chunks)
        batch = train_chunks[idx].to(dev)
        with torch.no_grad():
            out = model(input_ids=batch, output_hidden_states=True)
            target = out.logits.float().log_softmax(-1)
            hiddens = [h.detach() for h in out.hidden_states[:n_layers]]
        del out
        opt.zero_grad(set_to_none=True)
        total = 0.0
        # One backward per layer keeps only one [B, T, vocab] graph alive at a time.
        for layer in range(n_layers):
            loss = lens_kl(model, hiddens[layer], target, U[layer], V[layer], b[layer]) / n_layers
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_([U, V, b], 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  mean KL {total:.3f}", flush=True)

    logit_kl, tuned_kl = evaluate()
    print("eval KL, logit lens:", " ".join(f"{x:.2f}" for x in logit_kl))
    print("eval KL, tuned lens:", " ".join(f"{x:.2f}" for x in tuned_kl))

    out_path = Path(args.out) if args.out else lens_path(args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": args.model,
            "n_layers": n_layers,
            "rank": args.rank,
            "U": U.detach().cpu(),
            "V": V.detach().cpu(),
            "b": b.detach().cpu(),
            "eval_kl_logit": logit_kl,
            "eval_kl_tuned": tuned_kl,
            "args": vars(args),
        },
        out_path,
    )
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# apply


def load_readout(model_id: str, device: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    """(unembedding [vocab, d], final norm weight [d], eps) straight from the checkpoint."""
    path = Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"]))
    config = json.loads((path / "config.json").read_text())
    wanted = {"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"}
    found: dict[str, torch.Tensor] = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if key in wanted:
                    found[key] = f.get_tensor(key)
    unembed = found.get("lm_head.weight")
    if unembed is None or config.get("tie_word_embeddings"):
        unembed = found["model.embed_tokens.weight"]
    eps = config.get("rms_norm_eps", 1e-6)
    return unembed.to(device), found["model.norm.weight"].to(device), eps


def tuned_logprobs(
    hs: torch.Tensor,
    token_ids: list[int],
    lens: dict[str, Any],
    unembed: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
) -> np.ndarray:
    """hs: [n_layers + 1, d]. Returns [n_layers + 1, len(token_ids)] log-probs.

    The last index is already the final-normed output, so it is read directly.
    """
    dev = unembed.device
    n = lens["n_layers"]
    with torch.inference_mode():
        h = hs.to(dev).float()
        U, V, b = (lens[k].to(dev) for k in ("U", "V", "b"))
        hv = torch.einsum("ld,ldr->lr", h[:n], V)
        t = h[:n] + torch.einsum("lr,ldr->ld", hv, U) + b
        normed = torch.cat([rms_norm(t, norm_w, eps), h[n:]])
        logp = (normed.to(unembed.dtype) @ unembed.T).float().log_softmax(-1)
    return logp[:, token_ids].cpu().numpy()


def apply(args: argparse.Namespace) -> None:
    cache: dict[str, Any] = {}
    for run_dir in map(Path, args.run_dir):
        results = json.loads((run_dir / "results.json").read_text())
        model_id = results["_meta"]["model"]
        lp_path = Path(args.lens) if args.lens else lens_path(model_id)
        lens = torch.load(lp_path)
        assert lens["model"] == model_id, f"Lens {lp_path} is for {lens['model']}, run is {model_id}"
        if model_id not in cache:
            cache[model_id] = load_readout(model_id, args.device)
        unembed, norm_w, eps = cache[model_id]

        out: dict[str, Any] = {"_meta": {**results["_meta"], "lens": str(lp_path)}}
        for phase in PHASES:
            hpath = run_dir / f"hiddens_{phase}.pt"
            if phase not in results or not hpath.exists():
                continue
            hs = torch.load(hpath)["hs"]
            token_ids = results[phase]["token_ids"]
            records = []
            for i, r in enumerate(results[phase]["records"]):
                lp = tuned_logprobs(hs[i], token_ids, lens, unembed, norm_w, eps)
                records.append({**r, "lp": np.round(lp, LP_DECIMALS).tolist()})
            out[phase] = {"token_ids": token_ids, "records": records}
            print(f"  {run_dir.name}: {phase} ({len(records)} records)", flush=True)
        (run_dir / "results_tuned.json").write_text(json.dumps(out))


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        train(args)
    else:
        apply(args)


if __name__ == "__main__":
    main()
