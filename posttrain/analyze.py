"""Section 5.1 (H1) follow-up: cross-model analysis of run_model.py outputs (CPU only).

Runs are grouped by the --label they were run with; runs in a group differ only in
--seed. Subcommands:

  exists      Step 0: is there a mid-layer injection signal at all? (exit code 1 if not)
  compare     Base vs. post-trained: attenuation, MI, specificity baselines, swap and
              ablation, and the pre-registered decision rule.
  stages      Attenuation across an ordered list of checkpoints (e.g. OLMo stages).
  selfclaims  H3 check: do late layers push unusual-but-true self-claims toward "no"?
  dla         Late-layer attention/MLP contributions to the yes-no direction.

Pre-registered metric (see the plan): signal at a layer = P(yes | treated) - P(yes |
control), P(yes) normalized within the yes/no token variants. The peak layer is chosen
on held-out folds (seeds if there are >= 2, otherwise concepts) and the signal is
averaged over peak +- PEAK_HALF_WIDTH on the left-out fold. Attenuation = peak-window
signal - final-layer signal, also reported as a fraction of the peak. Layers whose
yes+no probability mass is below --min-mass are excluded from the peak search.

Run from the repo root, e.g.:
  python posttrain/analyze.py compare --base 32b-base-plain-vinst --instruct 32b-instruct-plain
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posttrain.prompts import PRIMARY_CONDITIONS  # noqa: E402

# Pre-registered constants.
PEAK_HALF_WIDTH = 2
FIXED_WINDOW = (0.75, 0.95)  # relative depth, cross-check only
MIN_MASS = 1e-3  # min mean yes+no probability for a layer to be a peak candidate
EXIST_MIN_GAP = 0.05  # step 0: mean held-out peak gap must reach this, with CI > 0
SANITY_MAX_SHIFT = 0.10  # swaps: max change in control P(yes) and false-positive rate
N_BOOT = 2_000
RNG = np.random.default_rng(0)

PER_LAYER_PHASES = {"main", "controls", "baseline", "confusion"}


# ---------------------------------------------------------------------------
# Loading


@dataclass
class Run:
    path: Path
    meta: dict[str, Any]
    raw: dict[str, Any]  # results.json (logit lens)
    lensed: dict[str, Any]  # per-layer phases read through the requested lens

    @property
    def seed(self) -> int:
        return self.meta["seed"]

    @property
    def n_yes(self) -> int:
        return self.meta.get("n_yes_variants", 4)

    def phase(self, name: str) -> dict[str, Any] | None:
        src = self.lensed if name in PER_LAYER_PHASES else self.raw
        return src.get(name)


def load_runs(root: str, label: str, lens: str) -> list[Run]:
    runs = []
    for rpath in sorted(Path(root).glob("*/results.json")):
        raw = json.loads(rpath.read_text())
        if raw.get("_meta", {}).get("label") != label:
            continue
        lensed = raw
        if lens == "tuned":
            tpath = rpath.parent / "results_tuned.json"
            lensed = json.loads(tpath.read_text()) if tpath.exists() else {}
        runs.append(Run(rpath.parent, raw["_meta"], raw, lensed))
    if not runs:
        sys.exit(f"No runs with label {label!r} under {root}")
    return runs


def has_tuned(runs: list[Run]) -> bool:
    return all((r.path / "results_tuned.json").exists() for r in runs if "main" in r.raw)


# ---------------------------------------------------------------------------
# Signals


def p_yes(lp: list[list[float]], n_yes: int) -> tuple[np.ndarray, np.ndarray]:
    """(P(yes) normalized within yes+no, raw yes+no mass), each [n_layers + 1]."""
    p = np.exp(np.asarray(lp))
    y, n = p[:, :n_yes].sum(1), p[:, n_yes:].sum(1)
    return y / (y + n + 1e-12), y + n


@dataclass
class Signal:
    rows: np.ndarray  # [units, n_layers + 1] treated - control
    treated: np.ndarray  # [units, n_layers + 1] P(yes | treated)
    control: np.ndarray  # [units, n_layers + 1] P(yes | control)
    seeds: np.ndarray
    concepts: np.ndarray
    mass: np.ndarray  # [n_layers + 1] mean yes+no mass

    def __len__(self) -> int:
        return len(self.rows)

    def groups(self) -> np.ndarray:
        return self.seeds if len(np.unique(self.seeds)) >= 2 else self.concepts

    def valid(self, min_mass: float) -> np.ndarray:
        return self.mass >= min_mass


def build_signal(
    runs: list[Run],
    phase: str,
    select: Callable[[dict], bool],
    key_fields: tuple[str, ...],
    is_treated: Callable[[dict], bool],
) -> Signal | None:
    """Pair each treated record with the control record sharing key_fields."""
    rows, treated, control, seeds, concepts, masses = [], [], [], [], [], []
    for r in runs:
        ph = r.phase(phase)
        if not ph:
            continue
        recs = [x for x in ph["records"] if select(x)]
        ctrl = {tuple(x[k] for k in key_fields): x for x in recs if not is_treated(x)}
        ctrl_p = {k: p_yes(x["lp"], r.n_yes) for k, x in ctrl.items()}
        masses += [m for _, m in ctrl_p.values()]
        for x in recs:
            if not is_treated(x):
                continue
            pc, _ = ctrl_p[tuple(x[k] for k in key_fields)]
            pt, mt = p_yes(x["lp"], r.n_yes)
            rows.append(pt - pc)
            treated.append(pt)
            control.append(pc)
            seeds.append(r.seed)
            concepts.append(x.get("concept"))
            masses.append(mt)
    if not rows:
        return None
    return Signal(
        np.array(rows),
        np.array(treated),
        np.array(control),
        np.array(seeds),
        np.array(concepts, dtype=object),
        np.mean(masses, 0),
    )


def detection_signal(runs: list[Run], prompt: str, phase: str = "main") -> Signal | None:
    return build_signal(
        runs, phase, lambda x: x["prompt"] == prompt, ("prompt",), lambda x: x["inj"]
    )


def confusing_signal(runs: list[Run], prompt: str) -> Signal | None:
    return build_signal(
        runs,
        "controls",
        lambda x: x["prompt"] == prompt and x["group"] == "confusing",
        ("prompt", "group", "q"),
        lambda x: x["inj"],
    )


def context_signal(runs: list[Run], prompt: str) -> Signal | None:
    sig = build_signal(
        runs,
        "baseline",
        lambda x: x["prompt"] == prompt,
        ("prompt", "concept"),
        lambda x: x["kind"] == "concept",
    )
    if sig is not None:  # no injection involved, so folds are always over concepts
        sig.seeds = np.zeros(len(sig), dtype=int)
    return sig


def mi_curves(runs: list[Run], prompt: str) -> tuple[np.ndarray, np.ndarray] | None:
    """([n_runs, n_layers + 1] MI in bits over the concepts with 'no injection' dropped, seeds)."""
    curves, seeds = [], []
    for r in runs:
        ph = r.phase("confusion")
        recs = [x for x in (ph or {}).get("records", []) if x["prompt"] == prompt]
        if not recs:
            continue
        labels = recs[0]["labels"]
        n = len(labels)
        n_l = len(recs[0]["lp"])
        conf = np.zeros((n_l, n, n))
        counts = np.zeros(n)
        for x in recs:
            p = np.exp(np.asarray(x["lp"]))
            p /= p.sum(1, keepdims=True)
            a = labels.index(x["actual"])
            for s, c in enumerate(x["perm"]):
                conf[:, a, c] += p[:, s]
            counts[a] += 1
        conf /= counts[None, :, None]
        sub = conf[:, 1:, 1:]
        sub = sub / sub.sum(-1, keepdims=True)
        curves.append(entropy(sub.mean(1)) - entropy(sub).mean(1))
        seeds.append(r.seed)
    return (np.array(curves), np.array(seeds)) if curves else None


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log2(np.clip(p, 1e-12, None))).sum(-1)


# ---------------------------------------------------------------------------
# Attenuation


@dataclass
class Atten:
    rows: np.ndarray  # [units, n_layers + 1] signal
    groups: np.ndarray  # fold label per unit (cv only)
    valid: np.ndarray  # [n_layers + 1] layers allowed in the peak search / window
    method: Callable[..., tuple[np.ndarray, np.ndarray, list[int]] | None]
    peak: np.ndarray = field(init=False)  # per-unit peak-window signal, sign-aligned
    final: np.ndarray = field(init=False)  # per-unit final-layer signal, sign-aligned
    peak_layers: list[int] = field(init=False)
    _boot: dict[str, np.ndarray] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        got = self.method(self.rows, self.groups, self.valid)
        assert got is not None
        self.peak, self.final, self.peak_layers = got

    def boot(self) -> dict[str, np.ndarray]:
        """Resample units and rerun the whole procedure, peak selection included.

        Cached, so every comparison involving this signal uses the same resamples.
        """
        if self._boot is None:
            n = len(self.rows)
            pk, fn = [], []
            for _ in range(N_BOOT):
                idx = RNG.integers(0, n, n)
                got = self.method(self.rows[idx], self.groups[idx], self.valid)
                if got is not None:  # e.g. a resample that lost all but one fold
                    pk.append(got[0].mean())
                    fn.append(got[1].mean())
            pk_a, fn_a = np.array(pk), np.array(fn)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(np.abs(pk_a) > 1e-9, (pk_a - fn_a) / pk_a, np.nan)
            self._boot = {"peak": pk_a, "final": fn_a, "att": pk_a - fn_a, "frac": frac}
        return self._boot

    def point(self) -> dict[str, float]:
        pk, fn = self.peak.mean(), self.final.mean()
        return {"peak": pk, "final": fn, "att": pk - fn, "frac": (pk - fn) / pk if abs(pk) > 1e-9 else np.nan}

    def summary(self, n_layers: int) -> dict[str, Any]:
        b = self.boot()
        out: dict[str, Any] = {"n_units": len(self.peak)}
        for k, v in self.point().items():
            out[k] = float(v)
            out[f"{k}_ci"] = ci(b[k])
        out["peak_layers"] = self.peak_layers
        out["peak_depth"] = [round(layer / n_layers, 3) for layer in self.peak_layers]
        return out


def ci(samples: np.ndarray) -> list[float]:
    return [float(x) for x in np.nanpercentile(samples, [2.5, 97.5])]


def diff_ci(a: Atten, b: Atten, key: str) -> dict[str, Any]:
    """b - a for one statistic, the two signals resampled independently."""
    ba, bb = a.boot()[key], b.boot()[key]
    n = min(len(ba), len(bb))
    return {
        "diff": float(b.point()[key] - a.point()[key]),
        "ci": ci(bb[:n] - ba[:n]),
    }


def cv_method(rows: np.ndarray, groups: np.ndarray, valid: np.ndarray):
    """Peak layer chosen on the other folds, measured on the left-out fold."""
    final = rows.shape[1] - 1
    cand = valid.copy()
    cand[final] = False
    folds = np.unique(groups)
    if not cand.any() or len(folds) < 2:
        return None
    idx = np.where(cand)[0]
    peak, fin = np.empty(len(rows)), np.empty(len(rows))
    peak_layers = []
    for f in folds:
        test = groups == f
        mean_train = rows[~test].mean(0)
        p = int(idx[np.argmax(np.abs(mean_train[idx]))])
        sign = np.sign(mean_train[p]) or 1.0
        win = [layer for layer in range(p - PEAK_HALF_WIDTH, p + PEAK_HALF_WIDTH + 1) if 0 <= layer < final and cand[layer]]
        peak[test] = sign * rows[test][:, win].mean(1)
        fin[test] = sign * rows[test][:, final]
        peak_layers.append(p)
    return peak, fin, peak_layers


def fixed_method(rows: np.ndarray, groups: np.ndarray, valid: np.ndarray):
    """Pre-set relative-depth window; no peak selection (cross-check)."""
    del groups
    n_layers = rows.shape[1] - 1
    win = [layer for layer in range(n_layers) if FIXED_WINDOW[0] <= layer / n_layers <= FIXED_WINDOW[1] and valid[layer]]
    if not win:
        return None
    sign = np.sign(rows[:, win].mean()) or 1.0
    return sign * rows[:, win].mean(1), sign * rows[:, -1], win


def cv_attenuation(rows: np.ndarray, groups: np.ndarray, valid: np.ndarray) -> Atten | None:
    return Atten(rows, groups, valid, cv_method) if cv_method(rows, groups, valid) else None


def fixed_attenuation(rows: np.ndarray, valid: np.ndarray) -> Atten | None:
    groups = np.zeros(len(rows), dtype=int)
    return Atten(rows, groups, valid, fixed_method) if fixed_method(rows, groups, valid) else None


def signal_attenuation(sig: Signal, min_mass: float) -> tuple[Atten | None, Atten | None]:
    valid = sig.valid(min_mass)
    return cv_attenuation(sig.rows, sig.groups(), valid), fixed_attenuation(sig.rows, valid)


# ---------------------------------------------------------------------------
# Formatting


def fmt(s: dict[str, Any] | None, key: str) -> str:
    if s is None:
        return "n/a"
    lo, hi = s[f"{key}_ci"]
    return f"{s[key]:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def fmt_diff(d: dict[str, Any] | None) -> str:
    if d is None:
        return "n/a"
    return f"{d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"


def above(d: dict[str, Any] | None) -> bool:
    return d is not None and d["ci"][0] > 0


def includes_zero(d: dict[str, Any] | None) -> bool:
    return d is not None and d["ci"][0] <= 0 <= d["ci"][1]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
    print(f"\nSaved {path}")


# ---------------------------------------------------------------------------
# exists


def cmd_exists(args: argparse.Namespace) -> None:
    runs = load_runs(args.root, args.label, args.lens)
    n_layers = runs[0].meta["n_layers"]
    passed = False
    print(f"Existence check for {args.label} ({args.lens} lens)")
    for prompt in PRIMARY_CONDITIONS:
        sig = detection_signal(runs, prompt)
        if sig is None:
            print(f"  {prompt}: no data")
            continue
        cv, _ = signal_attenuation(sig, args.min_mass)
        if cv is None:
            print(f"  {prompt}: need >= 2 seeds or concepts and some readable layers")
            continue
        s = cv.summary(n_layers)
        ok = s["peak"] >= EXIST_MIN_GAP and s["peak_ci"][0] > 0
        passed |= ok
        print(f"  {prompt:28s} peak gap {fmt(s, 'peak')} at depth {s['peak_depth']}  -> {'PASS' if ok else 'fail'}")
    print(f"\nSignal exists: {'YES' if passed else 'NO'} (criterion: peak gap >= {EXIST_MIN_GAP}, CI > 0)")
    sys.exit(0 if passed else 1)


# ---------------------------------------------------------------------------
# compare


def pair_attenuation(base: list[Run], inst: list[Run], builder, prompt: str, min_mass: float):
    out = {}
    for name, runs in (("base", base), ("instruct", inst)):
        sig = builder(runs, prompt)
        if sig is None:
            out[name] = None
            continue
        cv, fixed = signal_attenuation(sig, min_mass)
        out[name] = {"sig": sig, "cv": cv, "fixed": fixed, "n_layers": runs[0].meta["n_layers"]}
    return out


def summarize_pair(pair: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("base", "instruct"):
        p = pair.get(name)
        out[name] = None if p is None else {
            "cv": p["cv"].summary(p["n_layers"]) if p["cv"] else None,
            "fixed": p["fixed"].summary(p["n_layers"]) if p["fixed"] else None,
        }
    b, i = pair.get("base"), pair.get("instruct")
    for kind in ("cv", "fixed"):
        if b and i and b[kind] and i[kind]:
            out[f"diff_{kind}"] = {k: diff_ci(b[kind], i[kind], k) for k in ("att", "frac")}
    return out


def evaluate_lens(args: argparse.Namespace, lens: str) -> dict[str, Any]:
    base = load_runs(args.root, args.base, lens)
    inst = load_runs(args.root, args.instruct, lens)
    res: dict[str, Any] = {"lens": lens, "conditions": {}}
    for prompt in PRIMARY_CONDITIONS:
        det = pair_attenuation(base, inst, detection_signal, prompt, args.min_mass)
        conf = pair_attenuation(base, inst, confusing_signal, prompt, args.min_mass)
        ctx = pair_attenuation(base, inst, context_signal, prompt, args.min_mass)
        c: dict[str, Any] = {
            "detection": summarize_pair(det),
            "confusing": summarize_pair(conf),
            "context": summarize_pair(ctx),
            "_raw": {"detection": det, "confusing": conf, "context": ctx},
        }
        # Specificity: does the baseline fade less than introspection in the instruct
        # model (fractions, since signal sizes differ), or about equally in both models?
        for bl in ("confusing", "context"):
            i_det, i_bl = det.get("instruct"), c["_raw"][bl].get("instruct")
            spec: dict[str, Any] = {"evaluable": False}
            if i_det and i_bl and i_det["cv"] and i_bl["cv"]:
                bl_sum = c[bl]["instruct"]["cv"]
                spec["evaluable"] = bl_sum["peak_ci"][0] > 0  # baseline has a signal to fade
                spec["intro_minus_baseline_frac"] = diff_ci(i_bl["cv"], i_det["cv"], "frac")
                spec["baseline_diff_frac"] = c[bl].get("diff_cv", {}).get("frac")
                spec["pass"] = above(spec["intro_minus_baseline_frac"]) or includes_zero(
                    spec["baseline_diff_frac"]
                )
            c[f"specificity_{bl}"] = spec
        mi = {}
        for name, runs in (("base", base), ("instruct", inst)):
            got = mi_curves(runs, prompt)
            if got is None:
                continue
            curves, seeds = got
            ones = np.ones(curves.shape[1], dtype=bool)
            cv = cv_attenuation(curves, seeds, ones)  # None with a single seed
            fixed = fixed_attenuation(curves, ones)
            nl = runs[0].meta["n_layers"]
            mi[name] = {
                "curve_mean": curves.mean(0).tolist(),
                "cv": cv.summary(nl) if cv else None,
                "fixed": fixed.summary(nl) if fixed else None,
            }
        c["mi"] = mi
        res["conditions"][prompt] = c
    return res


def final_gap(main: dict[str, Any], prompt: str, n_yes: int) -> tuple[dict[str, float], float]:
    """Final-layer ({concept: P(yes|inj) - P(yes|no inj)}, false-positive P(yes|no inj))."""
    recs = [x for x in main["records"] if x["prompt"] == prompt]
    fp = p_yes(next(x for x in recs if not x["inj"])["lp"], n_yes)[0][-1]
    return {x["concept"]: p_yes(x["lp"], n_yes)[0][-1] - fp for x in recs if x["inj"]}, fp


def control_means(controls: dict[str, Any], n_yes: int) -> dict[str, float]:
    """Final-layer mean P(yes) of non-injected always-no / always-yes control records."""
    out = {}
    for group in ("no", "yes"):
        vals = [p_yes(x["lp"], n_yes)[0][-1] for x in controls["records"] if x["group"] == group and not x["inj"]]
        if vals:
            out[group] = float(np.mean(vals))
    return out


def intervention_report(run: Run, results: dict[str, dict[str, Any]], predicted: int) -> dict[str, Any]:
    """Final-layer gap change vs. the unmodified model, per condition and setting.

    predicted = +1 if the gap should grow, -1 if it should shrink.
    """
    n_yes = run.n_yes
    ref_ctrl = control_means(run.raw["controls"], n_yes)
    out: dict[str, Any] = {}
    for setting, res in results.items():
        row: dict[str, Any] = {"conditions": {}}
        fp_shift = []
        for prompt in PRIMARY_CONDITIONS:
            ref, ref_fp = final_gap(run.raw["main"], prompt, n_yes)
            new, new_fp = final_gap(res["main"], prompt, n_yes)
            d = np.array([new[c] - ref[c] for c in ref])
            boot = d[RNG.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
            lo, hi = ci(boot)
            row["conditions"][prompt] = {
                "diff": float(d.mean()),
                "ci": [lo, hi],
                "as_predicted": (lo > 0) if predicted > 0 else (hi < 0),
            }
            fp_shift.append(abs(new_fp - ref_fp))
        ctrl = control_means(res["controls"], n_yes)
        shifts = [abs(ctrl[g] - ref_ctrl[g]) for g in ctrl if g in ref_ctrl] + fp_shift
        row["control_means"] = ctrl
        row["max_sanity_shift"] = float(max(shifts))
        row["sane"] = row["max_sanity_shift"] <= SANITY_MAX_SHIFT
        out[setting] = row
    return out


def swap_check(runs: list[Run], predicted: int) -> dict[str, Any] | None:
    run = next((r for r in runs if "swap" in r.raw and "main" in r.raw and "controls" in r.raw), None)
    if run is None:
        return None
    sw = {k: v for k, v in run.raw["swap"].items() if k != "from"}
    report = intervention_report(run, sw, predicted)
    sane = [n for n, r in report.items() if r["sane"]]
    ok = bool(sane) and all(
        c["as_predicted"] for n in sane for c in report[n]["conditions"].values()
    )
    return {"from": run.raw["swap"]["from"], "by_n": report, "sane_n": sane, "pass": ok}


def ablate_check(runs: list[Run]) -> dict[str, Any] | None:
    run = next((r for r in runs if "ablate" in r.raw and "main" in r.raw and "controls" in r.raw), None)
    if run is None:
        return None
    ab = run.raw["ablate"]
    return {"ref": ab["ref"], "last": ab["last"], **intervention_report(run, {"ablate": ab}, +1)["ablate"]}


def verdict(res: dict[str, Any], swaps: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    conds = res["conditions"]
    base_signal = all(
        (c["detection"]["base"] or {}).get("cv") and c["detection"]["base"]["cv"]["peak_ci"][0] > 0
        for c in conds.values()
    )
    c1 = all(above(c["detection"].get("diff_cv", {}).get("att")) for c in conds.values())
    c2 = None
    if swaps["into_instruct"] and swaps["into_base"]:
        c2 = swaps["into_instruct"]["pass"] and swaps["into_base"]["pass"]
    spec = [c[f"specificity_{bl}"] for c in conds.values() for bl in ("confusing", "context")]
    evaluable = [s for s in spec if s.get("evaluable")]
    c3 = all(s["pass"] for s in evaluable) if evaluable else None
    crit = {"base_has_signal": base_signal, "c1_attenuation": c1, "c2_swaps": c2, "c3_specificity": c3}

    if not base_signal:
        return "Inconclusive: no middle-layer signal in the base model", crit
    if c1 and c2 and c3:
        return "H1 supported", crit
    if c1:
        missing = [k for k in ("c2_swaps", "c3_specificity") if crit[k] is None]
        failed = [k for k in ("c2_swaps", "c3_specificity") if crit[k] is False]
        return f"Attenuation differs as H1 predicts, but not established (failed: {failed}, missing: {missing})", crit
    base_fades = all(
        (c["detection"]["base"]["cv"] or {}).get("att_ci", [0])[0] > 0 for c in conds.values()
    )
    if not base_fades:
        return "Neither model shows clear attenuation", crit
    generic = any(
        (c[bl]["base"] or {}).get("cv") and c[bl]["base"]["cv"]["att_ci"][0] > 0
        for c in conds.values()
        for bl in ("confusing", "context")
    )
    if generic:
        return "Generic late-layer fading: baseline signals also fade in the base model; says nothing about H1 vs H2", crit
    return "Base model fades as much and baselines don't: points to H2 (pretraining)", crit


def print_compare(res: dict[str, Any]) -> None:
    print(f"\n######## {res['lens']} lens ########")
    for prompt, c in res["conditions"].items():
        print(f"\n=== {prompt} ===")
        for sig in ("detection", "confusing", "context"):
            s = c[sig]
            print(f"  [{sig}]")
            for name in ("base", "instruct"):
                cv = (s[name] or {}).get("cv")
                fixed = (s[name] or {}).get("fixed")
                print(
                    f"    {name:9s} peak {fmt(cv, 'peak')}  final {fmt(cv, 'final')}  "
                    f"att {fmt(cv, 'att')}  frac {fmt(cv, 'frac')}  depth {cv['peak_depth'] if cv else 'n/a'}"
                )
                print(f"    {'':9s} fixed-window att {fmt(fixed, 'att')}  frac {fmt(fixed, 'frac')}")
            for kind in ("cv", "fixed"):
                d = s.get(f"diff_{kind}")
                if d:
                    print(f"    instruct-base ({kind}): att {fmt_diff(d['att'])}  frac {fmt_diff(d['frac'])}")
        for bl in ("confusing", "context"):
            sp = c[f"specificity_{bl}"]
            if "pass" not in sp:
                print(f"  specificity vs {bl}: n/a")
                continue
            print(
                f"  specificity vs {bl}: instruct intro-baseline frac {fmt_diff(sp['intro_minus_baseline_frac'])}, "
                f"baseline instruct-base frac {fmt_diff(sp['baseline_diff_frac'])} -> "
                f"{'pass' if sp['pass'] else 'FAIL'}{'' if sp['evaluable'] else ' (no baseline signal; ignored)'}"
            )
        for name, mi in c["mi"].items():
            print(
                f"  MI {name:9s} peak {fmt(mi['cv'], 'peak')}  att {fmt(mi['cv'], 'att')}  "
                f"fixed att {fmt(mi['fixed'], 'att')}"
            )


def print_interventions(name: str, rep: dict[str, Any] | None) -> None:
    if rep is None:
        print(f"\n{name}: not run")
        return
    print(f"\n{name}:")
    rows = rep.get("by_n", {"ablate": rep})
    for setting, r in rows.items():
        conds = "  ".join(
            f"{p}: {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]{' *' if c['as_predicted'] else ''}"
            for p, c in r["conditions"].items()
        )
        print(f"  {setting:>6}  {conds}  sanity shift {r['max_sanity_shift']:.3f} {'ok' if r['sane'] else 'BROKEN'}")
    if "pass" in rep:
        print(f"  -> {'as predicted' if rep['pass'] else 'NOT as predicted'} (sane N: {rep['sane_n']})")


def plot_compare(res: dict[str, Any], out_dir: Path, name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for prompt, c in res["conditions"].items():
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        for model, color in (("base", "tab:blue"), ("instruct", "tab:red")):
            p = c["_raw"]["detection"].get(model)
            if p is None:
                continue
            sig: Signal = p["sig"]
            x = np.arange(sig.rows.shape[1]) / p["n_layers"]
            axes[0].plot(x, sig.treated.mean(0), color=color, label=f"{model} inj")
            axes[0].plot(x, sig.control.mean(0), color=color, ls="--", label=f"{model} no inj")
            m, se = sig.rows.mean(0), sig.rows.std(0) / np.sqrt(len(sig))
            axes[1].plot(x, m, color=color, label=model)
            axes[1].fill_between(x, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.2)
            axes[2].semilogy(x, sig.mass, color=color, label=model)
            if model in c["mi"]:
                axes[3].plot(x, c["mi"][model]["curve_mean"], color=color, label=model)
        axes[2].axhline(MIN_MASS, color="gray", ls=":")
        for ax, title in zip(axes, ["P(yes)", "injection gap (95% CI)", "yes+no mass", "MI (bits)"]):
            ax.set_title(title)
            ax.set_xlabel("relative depth")
            ax.legend(fontsize=7)
        fig.suptitle(f"{name} | {prompt} | {res['lens']} lens")
        fig.tight_layout()
        path = out_dir / f"{name}_{prompt}_{res['lens']}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"Saved {path}")


def strip_raw(res: dict[str, Any]) -> dict[str, Any]:
    return {**res, "conditions": {p: {k: v for k, v in c.items() if k != "_raw"} for p, c in res["conditions"].items()}}


def cmd_compare(args: argparse.Namespace) -> None:
    name = args.name or f"{args.base}__vs__{args.instruct}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lenses = ["logit"]
    if has_tuned(load_runs(args.root, args.base, "logit")) and has_tuned(load_runs(args.root, args.instruct, "logit")):
        lenses.append("tuned")
    results = {lens: evaluate_lens(args, lens) for lens in lenses}

    base_runs = load_runs(args.root, args.base, "logit")
    inst_runs = load_runs(args.root, args.instruct, "logit")
    swaps = {
        "into_instruct": swap_check(inst_runs, +1),  # base late layers -> signal returns
        "into_base": swap_check(base_runs, -1),  # instruct late layers -> suppression appears
    }
    ablate = ablate_check(inst_runs)

    for lens in lenses:
        print_compare(results[lens])
        plot_compare(results[lens], out_dir, name)
    print_interventions("Swap: base layers into instruct (predict gap grows)", swaps["into_instruct"])
    print_interventions("Swap: instruct layers into base (predict gap shrinks)", swaps["into_base"])
    print_interventions("Ablation in instruct (predict gap grows)", ablate)

    final_lens = "tuned" if "tuned" in results else "logit"
    v, crit = verdict(results[final_lens], swaps)
    print(f"\n==== Decision rule ({final_lens} lens) ====")
    for k, val in crit.items():
        print(f"  {k:18s} {val}")
    print(f"  VERDICT: {v}")
    if final_lens == "logit":
        print("  NOTE: no tuned-lens results yet; the final claim requires the tuned lens.")
    elif verdict(results["logit"], swaps)[1]["c1_attenuation"] and not crit["c1_attenuation"]:
        print("  NOTE: attenuation difference appears with the logit lens only -> likely a lens artifact.")

    write_json(
        out_dir / f"{name}.json",
        {
            "base": args.base,
            "instruct": args.instruct,
            "min_mass": args.min_mass,
            "results": {lens: strip_raw(r) for lens, r in results.items()},
            "swaps": swaps,
            "ablate": ablate,
            "verdict": v,
            "criteria": crit,
            "verdict_lens": final_lens,
        },
    )


# ---------------------------------------------------------------------------
# stages


def cmd_stages(args: argparse.Namespace) -> None:
    labels = args.labels.split(",")
    out: dict[str, Any] = {"labels": labels, "lens": args.lens, "conditions": {}}
    print(f"Attenuation across stages ({args.lens} lens): {' -> '.join(labels)}")
    for prompt in PRIMARY_CONDITIONS:
        attens: list[Atten | None] = []
        rows = []
        for label in labels:
            runs = load_runs(args.root, label, args.lens)
            sig = detection_signal(runs, prompt)
            cv = signal_attenuation(sig, args.min_mass)[0] if sig else None
            attens.append(cv)
            rows.append({"label": label, "cv": cv.summary(runs[0].meta["n_layers"]) if cv else None})
        first = None
        for i in range(1, len(labels)):
            a0, ap, ai = attens[0], attens[i - 1], attens[i]
            rows[i]["vs_first"] = diff_ci(a0, ai, "att") if a0 and ai else None
            rows[i]["vs_previous"] = diff_ci(ap, ai, "att") if ap and ai else None
            if first is None and above(rows[i]["vs_first"]):
                first = labels[i]
        print(f"\n=== {prompt} ===")
        for r in rows:
            print(
                f"  {r['label']:32s} att {fmt(r['cv'], 'att')}  frac {fmt(r['cv'], 'frac')}  "
                f"vs first {fmt_diff(r.get('vs_first'))}  vs prev {fmt_diff(r.get('vs_previous'))}"
            )
        print(f"  First stage with attenuation above the first checkpoint: {first}")
        out["conditions"][prompt] = {"rows": rows, "first_stage_above_base": first}
    write_json(Path(args.out) / f"stages_{args.lens}.json", out)


# ---------------------------------------------------------------------------
# selfclaims


def cmd_selfclaims(args: argparse.Namespace) -> None:
    out: dict[str, Any] = {}
    attens: dict[str, dict[str, Atten]] = {}
    print("Self-claims (no injection): peak-window P(yes) - final P(yes); positive = late layers push toward 'no'")
    for label in args.labels.split(","):
        runs = load_runs(args.root, label, "logit")
        run = next((r for r in runs if "selfclaims" in r.raw), None)
        if run is None:
            print(f"  {label}: not run")
            continue
        nl = run.meta["n_layers"]
        out[label] = {}
        attens[label] = {}
        for group in ("self_unusual", "ordinary"):
            items = run.raw["selfclaims"][group]
            rows = np.array([np.array(x["p_yes"]) / (np.array(x["p_yes"]) + np.array(x["p_no"]) + 1e-12) for x in items])
            mass = np.mean([np.array(x["p_yes"]) + np.array(x["p_no"]) for x in items], 0)
            cv = cv_attenuation(rows, np.arange(len(rows)), mass >= args.min_mass)
            if cv is None:
                continue
            attens[label][group] = cv
            out[label][group] = {**cv.summary(nl), "final_p_yes": float(rows[:, -1].mean())}
            print(f"  {label:32s} {group:13s} drop {fmt(out[label][group], 'att')}  final P(yes) {rows[:, -1].mean():.3f}")
        a = attens[label]
        if "self_unusual" in a and "ordinary" in a:
            out[label]["self_minus_ordinary"] = diff_ci(a["ordinary"], a["self_unusual"], "att")
            print(f"  {label:32s} self - ordinary drop {fmt_diff(out[label]['self_minus_ordinary'])}")
    labels = [label for label in attens if "self_unusual" in attens[label]]
    if len(labels) == 2:
        d = diff_ci(attens[labels[0]]["self_unusual"], attens[labels[1]]["self_unusual"], "att")
        out["second_minus_first_self_unusual"] = d
        print(f"\n  {labels[1]} - {labels[0]} (self_unusual drop): {fmt_diff(d)}")
    write_json(Path(args.out) / "selfclaims.json", out)


# ---------------------------------------------------------------------------
# dla


def cmd_dla(args: argparse.Namespace) -> None:
    out: dict[str, Any] = {}
    for label in args.labels.split(","):
        run = next((r for r in load_runs(args.root, label, "logit") if "dla" in r.raw), None)
        if run is None:
            print(f"{label}: dla not run")
            continue
        nl = run.meta["n_layers"]
        late = range(int(0.75 * nl), nl)
        out[label] = {}
        print(f"\n{label} (layers {late.start}-{nl - 1}; negative = pushes toward 'no')")
        for prompt, rows in run.raw["dla"].items():
            comp = {}
            for kind in ("attn", "mlp"):
                base = np.mean([r[f"base_{kind}"] for r in rows], 0)
                eff = np.mean([np.array(r[f"expr_{kind}"]) - np.array(r[f"base_{kind}"]) for r in rows], 0)
                comp[kind] = {"base": base.tolist(), "injection_effect": eff.tolist()}
            effects = [(comp[k]["injection_effect"][i], i, k) for k in ("attn", "mlp") for i in late]
            top = sorted(effects)[: args.top]
            summary = {
                k: {
                    "late_sum_base": float(np.sum([comp[k]["base"][i] for i in late])),
                    "late_sum_injection_effect": float(np.sum([comp[k]["injection_effect"][i] for i in late])),
                }
                for k in ("attn", "mlp")
            }
            out[label][prompt] = {"per_layer": comp, "late": summary, "top_negative": top}
            print(f"  {prompt}")
            for k, s in summary.items():
                print(f"    {k:4s} late sum: no-inj {s['late_sum_base']:+.2f}  injection effect {s['late_sum_injection_effect']:+.2f}")
            print("    most negative injection effects: " + ", ".join(f"L{i} {k} {e:+.2f}" for e, i, k in top))
    write_json(Path(args.out) / "dla.json", out)


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/posttrain")
    p.add_argument("--out", default="outputs/posttrain/analysis")
    p.add_argument("--min-mass", type=float, default=MIN_MASS)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("exists")
    e.add_argument("--label", required=True)
    e.add_argument("--lens", choices=["logit", "tuned"], default="logit")

    c = sub.add_parser("compare")
    c.add_argument("--base", required=True, help="Label of the base-model runs")
    c.add_argument("--instruct", required=True, help="Label of the post-trained runs")
    c.add_argument("--name", help="Output file stem")

    s = sub.add_parser("stages")
    s.add_argument("--labels", required=True, help="Comma-separated, in training order")
    s.add_argument("--lens", choices=["logit", "tuned"], default="logit")

    sc = sub.add_parser("selfclaims")
    sc.add_argument("--labels", required=True)

    d = sub.add_parser("dla")
    d.add_argument("--labels", required=True)
    d.add_argument("--top", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    {
        "exists": cmd_exists,
        "compare": cmd_compare,
        "stages": cmd_stages,
        "selfclaims": cmd_selfclaims,
        "dla": cmd_dla,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
