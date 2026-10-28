"""Section 5.1 (H1) follow-up: does post-training cause the final-layer suppression?

Runs one model at a time (fits a single GPU) and saves everything needed for a
cross-model comparison to an output directory. `posttrain/analyze.py` then
compares runs on CPU, and `posttrain/tuned_lens.py apply` re-reads the saved
hidden states through a tuned lens.

Phases (selectable with --phases):
  calibrate  top-of-mind samples at several vector strengths
  main       injection detection: per-layer logit lens + last-position hidden states
  controls   control questions (incl. confusing ones) under injection, per layer
  baseline   concept-in-context specificity baseline (no injection), per layer
  confusion  concept identification (per-layer MI)
  selfclaims unusual-but-true self-referential questions, no injection (H3 test)
  dla        direct logit attribution of attn/MLP outputs to the yes-no direction
  swap       replace the last N layers with another checkpoint's layers
  ablate     project out an (this model - other model) residual direction in late layers

Per-layer phases (main, controls, baseline, confusion, and the runs inside swap and
ablate) are stored as {"token_ids": [...], "records": [{<meta>, "lp": [[...]]}]},
where lp[layer][k] is the lens log-prob of token_ids[k]. main/controls/baseline/
confusion also save hiddens_<phase>.pt with one [n_layers + 1, d_model] tensor per
record, in the same order.

Run from the repo root, e.g.:
  python posttrain/run_model.py --model Qwen/Qwen2.5-Coder-7B-Instruct --tag qwen7b-instruct \
      --layer-range 9:19 --vector-strength 8 --fewshot --conditions primary
"""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as m  # noqa: E402  (reuses prompts, interventions, info docs)
from helpers import (  # noqa: E402
    Conversation,
    _model_cache_dir,
    apply_injection,
    prefill_hidden_states,
    token_id,
    train_vectors,
)
from posttrain.prompts import (  # noqa: E402
    PRIMARY_CONDITIONS,
    add_fewshot,
    context_baseline_prompts,
)
from repeng import ControlModel, ControlVector  # noqa: E402

# All 9 paper concepts by default; --concepts narrows this (e.g. the existence check).
CONCEPTS = list(m.CONCEPTS)

# Set from --fewshot in main(); applied to every conversation in last_pos_hiddens.
FEWSHOT = False

YES_VARIANTS = [" yes", " Yes", "yes", "Yes"]
NO_VARIANTS = [" no", " No", "no", "No"]

# Plain transcript format for base models. Every message ends in "\n\n" so that
# incremental tokenization in the helpers stays prefix-consistent.
PLAIN_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{% if message['content'] %}{{ message['content'] }}\n\n{% endif %}"
    "{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n\n"
    "{% else %}Assistant: {{ message['content'] }}\n\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant:{% endif %}"
)

# Unusual-but-true claims about the model itself (H3), plus ordinary true facts.
SELF_CLAIMS_TRUE_UNUSUAL = [
    "Do you process text as a sequence of tokens?",
    "Can information computed at earlier tokens influence your later computations?",
    "Do you have internal activations that represent concepts?",
    "Could a researcher modify your internal activations while you run?",
    "Is it possible for your intermediate layers to contain information you do not output?",
    "Do you produce each output by computing a probability distribution over tokens?",
    "Do you attend to earlier parts of this conversation when producing a reply?",
    "Are your intermediate computations stored in a key-value cache during generation?",
]
SELF_CLAIMS_TRUE_ORDINARY = m.CONTROL_QUESTIONS_YES

CONTROL_SUBSET = [
    ("no", m.CONTROL_QUESTIONS_NO[:4]),
    ("yes", m.CONTROL_QUESTIONS_YES[:4]),
    ("ambiguous", m.CONTROL_QUESTIONS_AMBIGUOUS[:3]),
    ("confusing", m.CONTROL_QUESTIONS_NO_CONFUSING),
]

# Conditions used by controls/confusion when --conditions all (the pre-existing subset).
ALL_CONTROL_CONDITIONS = [
    "inject_info-introspect",
    "inject_info-lorem-ipsum",
    "salient_info-introspect",
    "salient_info-lorem-ipsum",
]
ALL_CONFUSION_CONDITIONS = ["inject_info-introspect", "salient_info-lorem-ipsum", "inject_info-none"]

# Saved lens log-probs are rounded to keep results.json small.
LP_DECIMALS = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="Short name for this run (output dir)")
    p.add_argument(
        "--label",
        help="Group name for analyze.py; runs differing only in --seed share it (default: --tag)",
    )
    p.add_argument("--tokenizer", help="Tokenizer/chat template source (default: --model)")
    p.add_argument("--format", choices=["chat", "plain"], default="chat")
    p.add_argument("--fewshot", action="store_true", help="Prepend neutral yes/no turns")
    p.add_argument(
        "--conditions",
        choices=["primary", "all"],
        default="primary",
        help="primary = the two pre-registered conditions only",
    )
    p.add_argument("--concepts", help="Comma-separated subset of concepts (default: all 9)")
    p.add_argument("--layer-range", required=True, help="start:end steering layers")
    p.add_argument("--vector-strength", type=float, required=True)
    p.add_argument(
        "--vector-source",
        help="Model id whose cached vectors to reuse (default: train on --model)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--phases",
        default="main,controls,confusion,selfclaims",
        help="Comma-separated phases",
    )
    p.add_argument("--calibrate-strengths", default="2,4,8,12,16")
    p.add_argument("--swap-from", help="Checkpoint to take late layers from (swap phase)")
    p.add_argument("--swap-n", default="2,4,6,8")
    p.add_argument("--ablate-ref", help="Run dir of the reference model (ablate phase)")
    p.add_argument("--ablate-last", type=int, default=4)
    p.add_argument("--n-orderings", type=int, default=3)
    p.add_argument("--out", default="outputs/posttrain")
    return p.parse_args()


def last_pos_hiddens(
    tokenizer: Any,
    steps: list[dict[str, Any]],
    model: Any,
    steering_layers: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like helpers.logit_lens_helper, but returns raw last-position hidden states.

    Returns (base, expr), each [n_layers + 1, d_model] float32 on CPU. Index i < n
    is the residual entering layer i; index n is the final-normed output.
    """
    if FEWSHOT:
        steps = add_fewshot(steps)
    base_kv, expr_kv = DynamicCache(), DynamicCache()
    base_hs, expr_hs = None, None
    n_tokens = 0
    for i in range(len(steps)):
        if steps[i]["role"] == "system":
            continue
        tokens = cast(
            torch.Tensor,
            tokenizer.apply_chat_template(
                [
                    {"role": steps[j]["role"], "content": steps[j]["content"]}
                    for j in range(i + 1)
                ],
                continue_final_message=steps[i].get("continue_msg", False),
                return_tensors="pt",
            ),
        )[:, n_tokens:]
        n_tokens += tokens.shape[1]

        base_hs = prefill_hidden_states(base_kv, model, tokens)
        if cvec := steps[i].get("cmp_cvec"):
            wrapped = ControlModel(model, list(steering_layers))
            wrapped.set_control(cvec)
            expr_hs = prefill_hidden_states(expr_kv, wrapped, tokens)
            wrapped.reset()
            wrapped.unwrap()
        else:
            expr_hs = prefill_hidden_states(expr_kv, model, tokens)

    assert base_hs is not None and expr_hs is not None
    base = torch.stack([h[0, -1, :] for h in base_hs]).float().cpu()
    expr = torch.stack([h[0, -1, :] for h in expr_hs]).float().cpu()
    return base, expr


def lens_logprobs(model: Any, hs: torch.Tensor, token_ids: list[int]) -> np.ndarray:
    """Logit lens: [n_layers + 1, len(token_ids)] log-probs for selected tokens."""
    dev = model.lm_head.weight.device
    dtype = model.lm_head.weight.dtype
    with torch.inference_mode():
        h = hs.to(dev, dtype)
        normed = torch.cat([model.model.norm(h[:-1]), h[-1:]])
        logp = model.lm_head(normed).float().log_softmax(-1)
    return logp[:, token_ids].cpu().numpy()


def yes_no_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    return (
        [token_id(tokenizer, t) for t in YES_VARIANTS],
        [token_id(tokenizer, t) for t in NO_VARIANTS],
    )


def summarize_lens(
    model: Any, hs: torch.Tensor, yes_ids: list[int], no_ids: list[int]
) -> dict[str, list[float]]:
    lp = lens_logprobs(model, hs, yes_ids + no_ids)
    p_yes = np.exp(lp[:, : len(yes_ids)]).sum(1)
    p_no = np.exp(lp[:, len(yes_ids) :]).sum(1)
    return {
        "p_yes": p_yes.tolist(),
        "p_no": p_no.tolist(),
        "ld": (np.log(p_yes + 1e-12) - np.log(p_no + 1e-12)).tolist(),
    }


class Recorder:
    """Collects per-layer lens log-probs (and optionally hidden states) for one phase."""

    def __init__(self, model: Any, token_ids: list[int], keep_hiddens: bool):
        self.model = model
        self.token_ids = token_ids
        self.keep_hiddens = keep_hiddens
        self.records: list[dict[str, Any]] = []
        self.hiddens: list[torch.Tensor] = []

    def add(self, meta: dict[str, Any], hs: torch.Tensor) -> np.ndarray:
        lp = lens_logprobs(self.model, hs, self.token_ids)
        self.records.append({**meta, "lp": np.round(lp, LP_DECIMALS).tolist()})
        if self.keep_hiddens:
            self.hiddens.append(hs.half())
        return lp

    def result(self) -> dict[str, Any]:
        return {"token_ids": self.token_ids, "records": self.records}

    def save_hiddens(self, path: Path) -> None:
        torch.save({"hs": torch.stack(self.hiddens)}, path)


def norm_p_yes(lp: np.ndarray, n_yes: int) -> np.ndarray:
    """P(yes) normalized within yes+no, per layer."""
    p = np.exp(lp)
    yes = p[:, :n_yes].sum(1)
    return yes / (yes + p[:, n_yes:].sum(1) + 1e-12)


def conditions(args: argparse.Namespace, all_conditions: list[str] | None) -> list[str] | None:
    """Allowed prompt labels for a phase; None means no filter."""
    return PRIMARY_CONDITIONS if args.conditions == "primary" else all_conditions


def detection_prompts(allowed: list[str] | None) -> list[Conversation]:
    """5 framings x 4 info docs, yes/no format (introspection_yes_shift prompts)."""
    return [
        cv
        for cv in m.introspection_yes_shift_prompts()
        if cv.label.endswith("fmt-yes-no")
        and (allowed is None or cv.extra["prompt"] in allowed)
    ]


def run_detection(
    tokenizer: Any,
    model: Any,
    vectors: dict[str, ControlVector],
    args: argparse.Namespace,
    layers: list[int],
    save_hiddens: Path | None,
) -> dict[str, Any]:
    yes_ids, no_ids = yes_no_ids(tokenizer)
    rec = Recorder(model, yes_ids + no_ids, keep_hiddens=save_hiddens is not None)
    for cv in detection_prompts(conditions(args, None)):
        prompt = cv.extra["prompt"]
        base_final, inj_final = None, []
        for concept in CONCEPTS:
            inj = {"cmp_cvec": args.vector_strength * vectors[concept]}
            base, expr = last_pos_hiddens(
                tokenizer, apply_injection(cv, inj), model, layers  # type: ignore[arg-type]
            )
            if base_final is None:
                lp = rec.add({"prompt": prompt, "concept": None, "inj": False}, base)
                base_final = norm_p_yes(lp, len(yes_ids))[-1]
            lp = rec.add({"prompt": prompt, "concept": concept, "inj": True}, expr)
            inj_final.append(norm_p_yes(lp, len(yes_ids))[-1])
        print(
            f"  {prompt:40s} final P(yes) base={base_final:.3f} inj={np.mean(inj_final):.3f}",
            flush=True,
        )
    if save_hiddens is not None:
        rec.save_hiddens(save_hiddens)
    return rec.result()


def run_controls(
    tokenizer: Any,
    model: Any,
    vectors: dict[str, ControlVector],
    args,
    layers,
    save_hiddens: Path | None,
    groups: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    yes_ids, no_ids = yes_no_ids(tokenizer)
    fmt = m.CONTROL_QUESTION_ANSWER_FORMAT_YES_NO
    allowed = conditions(args, ALL_CONTROL_CONDITIONS)
    rec = Recorder(model, yes_ids + no_ids, keep_hiddens=save_hiddens is not None)
    for intervention in m.INTERVENTIONS:
        for info_label, info_doc in m.INFO_DOCUMENTS:
            prompt_label, user_turn_1 = m.make_turn_1_label_prompt(
                intervention, info_label, info_doc
            )
            if allowed is not None and prompt_label not in allowed:
                continue
            for group, questions in CONTROL_SUBSET:
                if groups is not None and group not in groups:
                    continue
                for q in questions:
                    cv = Conversation(
                        label=q,
                        yes_token=fmt.yes_token,
                        no_token=fmt.no_token,
                        extra={},
                        messages=[
                            {"role": "system", "content": ""},
                            {"role": "user", "content": user_turn_1, "add_injection": True},
                            {"role": "assistant", "content": m.ASST_TURN_1, "add_injection": True},
                            {"role": "user", "content": f"{q} {fmt.turn_2_format_desc}"},
                            {"role": "assistant", "content": fmt.assistant_prefix, "continue_msg": True},
                        ],
                    )
                    meta = {"prompt": prompt_label, "group": group, "q": q}
                    for i, concept in enumerate(CONCEPTS):
                        inj = {"cmp_cvec": args.vector_strength * vectors[concept]}
                        base, expr = last_pos_hiddens(
                            tokenizer, apply_injection(cv, inj), model, layers  # type: ignore[arg-type]
                        )
                        if i == 0:
                            rec.add({**meta, "concept": None, "inj": False}, base)
                        rec.add({**meta, "concept": concept, "inj": True}, expr)
            print(f"  controls {prompt_label} done", flush=True)
    if save_hiddens is not None:
        rec.save_hiddens(save_hiddens)
    return rec.result()


def run_baseline(tokenizer: Any, model: Any, args, save_hiddens: Path) -> dict[str, Any]:
    """Concept named in plain text vs. not; yes/no question about the conversation."""
    yes_ids, no_ids = yes_no_ids(tokenizer)
    rec = Recorder(model, yes_ids + no_ids, keep_hiddens=True)
    allowed = conditions(args, None) or [
        m.make_turn_1_label_prompt(i, label, doc)[0]
        for i in m.INTERVENTIONS
        for label, doc in m.INFO_DOCUMENTS
    ]
    for meta, cv in context_baseline_prompts(CONCEPTS, allowed):
        base, _ = last_pos_hiddens(tokenizer, apply_injection(cv, None), model, [])  # type: ignore[arg-type]
        rec.add(meta, base)
    print("  baseline done", flush=True)
    rec.save_hiddens(save_hiddens)
    return rec.result()


def run_confusion(
    tokenizer: Any,
    model: Any,
    vectors: dict[str, ControlVector],
    args,
    layers,
    save_hiddens: Path,
) -> dict[str, Any]:
    """Per-layer lens log-probs over the answer digits for [no injection] + CONCEPTS.

    Records are in shuffled answer order; meta["perm"][shuffled_idx] = canonical_idx.
    """
    canonical = [m.NO_INJECTION_STR] + CONCEPTS
    answer_ids = [token_id(tokenizer, str(i)) for i in range(len(canonical))]
    allowed = conditions(args, ALL_CONFUSION_CONDITIONS)
    rec = Recorder(model, answer_ids, keep_hiddens=True)
    for cv in m._concept_confusion_matrix_prompts(CONCEPTS):
        prompt = cv.extra["prompt"]
        if (allowed is not None and prompt not in allowed) or cv.extra["ordering_idx"] >= args.n_orderings:
            continue
        for actual in canonical:
            inj = (
                {}
                if actual == m.NO_INJECTION_STR
                else {"cmp_cvec": args.vector_strength * vectors[actual]}
            )
            _, expr = last_pos_hiddens(
                tokenizer, apply_injection(cv, inj), model, layers  # type: ignore[arg-type]
            )
            rec.add(
                {
                    "prompt": prompt,
                    "ordering": cv.extra["ordering_idx"],
                    "perm": cv.extra["perm"],
                    "actual": actual,
                    "labels": canonical,
                },
                expr,
            )
        print(f"  confusion {prompt} ordering {cv.extra['ordering_idx']} done", flush=True)
    rec.save_hiddens(save_hiddens)
    return rec.result()


def run_selfclaims(tokenizer: Any, model: Any) -> dict[str, Any]:
    yes_ids, no_ids = yes_no_ids(tokenizer)
    fmt = m.CONTROL_QUESTION_ANSWER_FORMAT_YES_NO
    out: dict[str, Any] = {}
    for group, questions in [
        ("self_unusual", SELF_CLAIMS_TRUE_UNUSUAL),
        ("ordinary", SELF_CLAIMS_TRUE_ORDINARY),
    ]:
        out[group] = []
        for q in questions:
            steps = [
                {"role": "system", "content": ""},
                {"role": "user", "content": f"{q} {fmt.turn_2_format_desc}"},
                {"role": "assistant", "content": fmt.assistant_prefix, "continue_msg": True},
            ]
            base, _ = last_pos_hiddens(tokenizer, steps, model, [])
            out[group].append({"q": q, **summarize_lens(model, base, yes_ids, no_ids)})
    return out


def component_modules(layer: Any) -> tuple[Any, Any]:
    """Modules whose outputs are added to the residual stream (attn, mlp)."""
    if hasattr(layer, "post_feedforward_layernorm"):  # OLMo-2: post-norm sublayers
        return layer.post_attention_layernorm, layer.post_feedforward_layernorm
    return layer.self_attn, layer.mlp


def run_dla(
    tokenizer: Any, model: Any, vectors: dict[str, ControlVector], args, layers
) -> dict[str, Any]:
    """Direct logit attribution to (" yes" - " no") at the answer position."""
    layer_mods = model.model.layers
    captured: dict[tuple[int, str], list[torch.Tensor]] = {}

    def make_hook(idx: int, kind: str):
        def hook(_mod, _inp, output):
            o = output[0] if isinstance(output, tuple) else output
            captured.setdefault((idx, kind), []).append(o[0, -1].float().cpu())

        return hook

    handles = []
    for i, layer in enumerate(layer_mods):
        attn, mlp = component_modules(layer)
        handles.append(attn.register_forward_hook(make_hook(i, "attn")))
        handles.append(mlp.register_forward_hook(make_hook(i, "mlp")))

    yes, no = token_id(tokenizer, " yes"), token_id(tokenizer, " no")
    direction = (model.lm_head.weight[yes] - model.lm_head.weight[no]).float().cpu()
    norm_w = model.model.norm.weight.float().cpu()
    eps = getattr(model.model.norm, "variance_epsilon", 1e-6)
    n_layers = len(layer_mods)

    out: dict[str, Any] = {}
    try:
        for cv in detection_prompts(PRIMARY_CONDITIONS):
            prompt = cv.extra["prompt"]
            rows = []
            for concept in CONCEPTS:
                captured.clear()
                inj = {"cmp_cvec": args.vector_strength * vectors[concept]}
                base, expr = last_pos_hiddens(
                    tokenizer, apply_injection(cv, inj), model, layers  # type: ignore[arg-type]
                )
                # Last forward of each module: [-2] = base run, [-1] = experiment run.
                res = {}
                for run_idx, hs in ((-2, base), (-1, expr)):
                    attn = torch.stack([captured[(i, "attn")][run_idx] for i in range(n_layers)])
                    mlp = torch.stack([captured[(i, "mlp")][run_idx] for i in range(n_layers)])
                    final_resid = hs[n_layers - 1] + attn[-1] + mlp[-1]
                    rms = final_resid.pow(2).mean().add(eps).sqrt()
                    scale = norm_w * direction / rms
                    res[run_idx] = ((attn @ scale).tolist(), (mlp @ scale).tolist())
                rows.append({
                    "concept": concept,
                    "base_attn": res[-2][0],
                    "base_mlp": res[-2][1],
                    "expr_attn": res[-1][0],
                    "expr_mlp": res[-1][1],
                })
            out[prompt] = rows
            print(f"  dla {prompt} done", flush=True)
    finally:
        for h in handles:
            h.remove()
    return out


def checkpoint_layer_state(model_id: str, layer_idx: int) -> dict[str, torch.Tensor]:
    path = Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"]))
    prefix = f"model.layers.{layer_idx}."
    state = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = f.get_tensor(key)
    return state


@contextmanager
def swapped_layers(model: Any, other_id: str, n_last: int) -> Iterator[None]:
    """Temporarily load the last n_last decoder layers from another checkpoint."""
    layer_mods = model.model.layers
    n = len(layer_mods)
    originals = {}
    for i in range(n - n_last, n):
        originals[i] = {k: v.detach().clone().cpu() for k, v in layer_mods[i].state_dict().items()}
        layer_mods[i].load_state_dict(checkpoint_layer_state(other_id, i))
    try:
        yield
    finally:
        for i, sd in originals.items():
            layer_mods[i].load_state_dict(sd)


def run_swap(tokenizer, model, vectors, args, layers) -> dict[str, Any]:
    """Detection plus always-no/always-yes controls (sanity check) for each swap size."""
    out: dict[str, Any] = {}
    for n_last in [int(x) for x in args.swap_n.split(",")]:
        print(f"  swap last {n_last} layers from {args.swap_from}", flush=True)
        with swapped_layers(model, args.swap_from, n_last):
            out[str(n_last)] = {
                "main": run_detection(tokenizer, model, vectors, args, layers, None),
                "controls": run_controls(
                    tokenizer, model, vectors, args, layers, None, groups=("no", "yes")
                ),
            }
    return out


@contextmanager
def ablated_direction(model: Any, directions: dict[int, torch.Tensor]) -> Iterator[None]:
    """Project unit directions out of the residual output of selected layers."""
    handles = []
    for idx, d in directions.items():
        d = d.to(model.device, model.dtype)

        def hook(_mod, _inp, output, d=d):
            h = output[0] if isinstance(output, tuple) else output
            h = h - (h @ d)[..., None] * d
            return (h, *output[1:]) if isinstance(output, tuple) else h

        handles.append(model.model.layers[idx].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def no_injection_hiddens(run_dir: Path) -> dict[str, torch.Tensor]:
    """Saved non-injected main-phase hidden states of a run, keyed by prompt."""
    records = json.loads((run_dir / "results.json").read_text())["main"]["records"]
    hs = torch.load(run_dir / "hiddens_main.pt")["hs"]
    return {r["prompt"]: hs[i].float() for i, r in enumerate(records) if not r["inj"]}


def run_ablate(tokenizer, model, vectors, args, layers, run_dir: Path) -> dict[str, Any]:
    """Ablate mean(this model - reference model) residual direction in the last layers.

    Directions come from saved non-injected final-position residuals of both runs, so
    both runs (this one and the reference) must have run the main phase first.
    """
    mine = no_injection_hiddens(run_dir)
    ref = no_injection_hiddens(Path(args.ablate_ref))
    prompts = sorted(set(mine) & set(ref))
    assert prompts, "No shared main-phase prompts between this run and --ablate-ref"
    mine_t = torch.stack([mine[p] for p in prompts])
    ref_t = torch.stack([ref[p] for p in prompts])
    n_layers = len(model.model.layers)
    directions = {}
    for idx in range(n_layers - args.ablate_last, n_layers):
        # hiddens index idx + 1 = residual leaving layer idx (last one is post-norm, skip)
        h = idx + 1 if idx + 1 < n_layers else idx
        d = (mine_t[:, h] - ref_t[:, h]).mean(0)
        directions[idx] = d / d.norm()
    out: dict[str, Any] = {}
    with ablated_direction(model, directions):
        out["main"] = run_detection(tokenizer, model, vectors, args, layers, None)
        out["controls"] = run_controls(
            tokenizer, model, vectors, args, layers, None, groups=("no", "yes")
        )
    return out


def run_calibrate(tokenizer, model, vectors, args, layers) -> dict[str, Any]:
    q = [
        {"role": "user", "content": m.TOP_OF_MIND_QUESTION},
        {"role": "assistant", "content": m.TOP_OF_MIND_PREFILL},
    ]
    ids = tokenizer.apply_chat_template(q, continue_final_message=True, return_tensors="pt")
    out: dict[str, Any] = {}
    for strength in [float(s) for s in args.calibrate_strengths.split(",")]:
        for concept in CONCEPTS[:3]:
            wrapped = ControlModel(model, layers)
            try:
                wrapped.set_control(vectors[concept] * strength)
                gen = wrapped.generate(
                    input_ids=ids.to(model.device), do_sample=False, max_new_tokens=30
                )
            finally:
                wrapped.unwrap()
            text = tokenizer.decode(gen[0, ids.shape[1] :], skip_special_tokens=True)
            out[f"{strength}_{concept}"] = text
            print(f"  [{strength:>5}] {concept:12s} {text!r}", flush=True)
    return out


def main() -> None:
    global FEWSHOT
    args = parse_args()
    phases = args.phases.split(",")
    start, end = map(int, args.layer_range.split(":"))
    layers = list(range(start, end))
    FEWSHOT = args.fewshot
    if args.concepts:
        CONCEPTS[:] = args.concepts.split(",")
        unknown = set(CONCEPTS) - set(m.CONCEPTS)
        assert not unknown, f"Unknown concepts: {unknown}"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if args.format == "plain":
        tokenizer.chat_template = PLAIN_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda:0", torch_dtype=torch.bfloat16
    )
    model.eval()

    run_dir = Path(args.out) / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    results: dict[str, Any] = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    results["_meta"] = {
        **results.get("_meta", {}),
        **{k: v for k, v in vars(args).items()},
        "label": args.label or args.tag,
        "n_layers": len(model.model.layers),
        "concepts": CONCEPTS,
        "n_yes_variants": len(YES_VARIANTS),
    }

    # Vectors are trained with this run's tokenizer/template and cached per model id;
    # --vector-source reuses another model's cached vectors (that run must go first).
    vector_source = args.vector_source or args.model
    if vector_source != args.model:
        cache = Path(_model_cache_dir(vector_source))
        missing = [c for c in CONCEPTS if not (cache / f"{c}_seed{args.seed}.gguf").exists()]
        if missing:
            sys.exit(f"No cached vectors for {vector_source}: {missing}. Run that model first.")
    vectors = train_vectors(
        tokenizer,
        model,
        None,
        CONCEPTS,
        m.OUTPUT_SUFFIXES,
        [],
        vector_source,
        seed=args.seed,
    )

    def save() -> None:
        results_path.write_text(json.dumps(results))

    for phase in phases:
        print(f"\n=== {args.tag}: {phase} ===", flush=True)
        if phase == "calibrate":
            results["calibrate"] = run_calibrate(tokenizer, model, vectors, args, layers)
        elif phase == "main":
            results["main"] = run_detection(
                tokenizer, model, vectors, args, layers, run_dir / "hiddens_main.pt"
            )
        elif phase == "controls":
            results["controls"] = run_controls(
                tokenizer, model, vectors, args, layers, run_dir / "hiddens_controls.pt"
            )
        elif phase == "baseline":
            results["baseline"] = run_baseline(
                tokenizer, model, args, run_dir / "hiddens_baseline.pt"
            )
        elif phase == "confusion":
            results["confusion"] = run_confusion(
                tokenizer, model, vectors, args, layers, run_dir / "hiddens_confusion.pt"
            )
        elif phase == "selfclaims":
            results["selfclaims"] = run_selfclaims(tokenizer, model)
        elif phase == "dla":
            results["dla"] = run_dla(tokenizer, model, vectors, args, layers)
        elif phase == "swap":
            assert args.swap_from, "--swap-from required"
            results["swap"] = {"from": args.swap_from, **run_swap(tokenizer, model, vectors, args, layers)}
        elif phase == "ablate":
            assert args.ablate_ref, "--ablate-ref required"
            results["ablate"] = {
                "ref": args.ablate_ref,
                "last": args.ablate_last,
                **run_ablate(tokenizer, model, vectors, args, layers, run_dir),
            }
        else:
            sys.exit(f"Unknown phase: {phase}")
        save()
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
