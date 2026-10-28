# Post-training suppression experiment: working notes

Goal: test paper §5.1, hypothesis 1. The claim is that the late-layer suppression of the
introspection signal ("was a concept injected?") comes from post-training. If so, a **base**
model should carry the mid-layer signal through to the output, while its **instruct**
version suppresses it in the final layers.

## Files (all in `posttrain/`)

| File | Role |
|---|---|
| `run_plan.sh` | **The only thing you run.** One step at a time, from the repo root. Calls the files below. |
| `run_model.py` | Runs one model (GPU): steering vectors, injection, per-layer yes/no log-probs, hidden states. Writes `outputs/posttrain/<tag>/results.json` and `hiddens_*.pt`. |
| `prompts.py` | Primary conditions, few-shot turns for base models, concept-in-context baseline prompts. |
| `tuned_lens.py` | Trains a per-layer tuned lens (`train`) and re-reads saved hiddens through it (`apply` → `results_tuned.json`). |
| `analyze.py` | CPU-only analysis: `exists`, `compare`, `stages`, `selfclaims`, `dla`. Writes JSON and PNG files to `outputs/posttrain/analysis/`. |

Prerequisite: a working `uv sync`. It failed with a permission error in the Claude session,
so GPU code there was only syntax-checked. The analysis was tested on synthetic data
(H1 case → "H1 supported", identical models → no difference; null false-positive rate ~1.5–4.5%).

## Design in one paragraph

Inject a concept vector only while the first turn is processed, then remove it and ask
"did you detect an injected thought?". Read P(yes) at every layer with the logit lens and
the tuned lens. **Attenuation** = (gap at the peak window) − (gap at the final layer). The
peak is chosen on held-out folds and the window covers the peak ±2 layers. A fixed window
at 75–95% of depth is a cross-check. The bootstrap re-runs peak selection on every resample.
The primary conditions were fixed in advance: `salient_info-lorem-ipsum` and
`inject_info-introspect`.

**Decision rule.** H1 is supported only if all of these hold:
- the base model has a signal;
- **C1:** the 95% CI of (instruct − base) attenuation is above 0 on both primary conditions;
- **C2:** both swap directions (steps 7, 8) behave as predicted;
- **C3:** specificity passes (the confusing control questions and the concept-in-context
  baseline don't fade the same way).

The verdict uses the tuned lens once it exists.

## Steps and commands

```bash
export SIZE=32b          # 7b failed the existence check (see below)
export S_INST=16 S_BASE_VINST=12 S_BASE_VSELF=16
bash posttrain/run_plan.sh <step>
```

| Step | What | Status |
|---|---|---|
| 0cal / 0 | Calibrate instruct; existence check | 7B (S_INST=8): no signal → moved to 32B. 32B (S_INST=16): passed |
| 2 | Calibrate base, both vector sources | Done → S_BASE_VINST=12, S_BASE_VSELF=16 |
| 3 | Main runs: 4 run types × seeds 42 43 44, then 3 comparisons | Done (see "Step 3 results" below) |
| 4 | Controls + concept-in-context baseline | todo |
| 5 | Concept identification (mutual information) | todo |
| 6 | Tuned lens (needs `data/lens_corpus.txt` or the `datasets` package) | todo |
| 7 | Swaps, mean-direction ablation, DLA (only if step 3 shows a difference) | todo |
| 8 | Self-claims | todo |
| 9cal / 9 | OLMo-2 32B base/SFT/DPO/Instruct stages (only if step 7 supports H1); export `S_OLMO_BASE/SFT/DPO/INSTRUCT` | todo; OLMo model IDs still need checking |
| 10 | Manual write-up | todo |

Four run types (labels): `32b-instruct-plain`, `32b-instruct-chat`, `32b-base-plain-vinst`
(base model + instruct vectors; the **primary** comparison), and `32b-base-plain-vself`
(base model + its own vectors; secondary). Run dirs are `<label>-s<seed>`.

To re-read results without re-running models:

```bash
uv run python posttrain/analyze.py compare --base 32b-base-plain-vinst --instruct 32b-instruct-plain --name 32b-primary
uv run python posttrain/analyze.py compare --base 32b-base-plain-vself --instruct 32b-instruct-plain --name 32b-own-vectors
uv run python posttrain/analyze.py compare --base 32b-base-plain-vinst --instruct 32b-instruct-chat  --name 32b-instruct-chat
```

The verdict and the numbers are in `outputs/posttrain/analysis/<name>.json`, next to the plots.

## How the strengths were picked

Rule: choose the lowest strength where the "what are you thinking about" samples are clearly
on topic for cats, bread and love, and the text is still coherent.

- **Instruct 32B: 16.**
- **Base, instruct vectors: 12.** At 8, only bread showed up. At 12 all three did
  (a cats-and-dogs problem, a cake recipe, "you, my love").
- **Base, own vectors: 16.** These vectors never steer cleanly. At 16, love is clear, bread
  is weak and cats is absent. At 20 or more the text breaks down (fake turns, multiple-choice
  formats, Java code, repetition). The injection is therefore weaker than in the other runs,
  so a smaller gap in `32b-own-vectors` is expected and does **not** count against H1. In
  that comparison, mainly check whether base attenuation is near zero.

The "attention mask / pad_token_id" warnings during calibration are harmless.

## Results so far (step 3, seed 42, final-layer P(yes))

| Run | Condition | No injection | Injected | Change |
|---|---|---|---|---|
| instruct-plain | introspect | 0.095 | 0.355 | +0.26 |
| instruct-plain | lorem | 0.001 | 0.483 | +0.48 |
| instruct-chat | introspect | 0.042 | 0.313 | +0.27 |
| instruct-chat | lorem | 0.003 | 0.466 | +0.46 |
| base-vinst | introspect | 0.623 | 0.494 | **−0.13** |
| base-vinst | lorem | 0.532 | 0.668 | +0.14 |
| base-vself | introspect | 0.623 | 0.432 | −0.19 |
| base-vself | lorem | 0.532 | 0.439 | −0.09 |

What this means:
- **The instruct model detects injections at the output** in both formats.
- **The base model is near 50/50:** it says "yes" 53–62% of the time with nothing
  injected, and injection doesn't reliably raise that. At the final layer it detects
  *worse* than the instruct model.
- **This doesn't decide H1 on its own.** H1 is about the mid-layer signal being suppressed
  toward the output, and the layer-by-layer comparison measures that.
- **Warning sign:** if the base model also has no clear mid-layer signal, the "base has a
  signal" requirement fails. The verdict would then be "can't test H1 with this setup", not
  "H1 supported". The likely fix is the base model's high "yes" rate with nothing injected,
  e.g. by rebalancing or changing the few-shot examples in `prompts.py` (`FEWSHOT_TURNS`),
  then re-running steps 2–3.

## Step 3 results (all 3 seeds, logit lens)

The printed verdict is `base_has_signal True, c1_attenuation True` in all 3 comparisons.
C2 and C3 show as missing, which is expected until steps 4 and 7 run.
**Do not trust the C1 pass yet.** Per-layer check (script: `inspect_gap.py`, below):

| Layer (of 64) | Base-vinst lorem gap | Instruct lorem gap | Base-vinst introspect | Instruct introspect |
|---|---|---|---|---|
| ≤ 48 (depth ≤ 0.75) | ~0 | ~0 (one +0.47 blip at 48, mass 4%) | ~0 (−0.13 at 48) | ~0 |
| 56 | 0 | +0.71 | 0 | +0.01 |
| 60 | 0 | **+0.93** | 0 | +0.14 |
| 62 | +0.02 | +0.78 | −0.04 | **+0.48** |
| 63 | +0.22 | +0.44 | −0.18 | +0.27 |
| 64 (final) | +0.14 | +0.48 | **−0.13** | +0.26 |

Yes+no probability mass under the logit lens is < 1% until about layer 50 in every model.

What this means:
1. **The instruct model shows late suppression:** on lorem the gap drops from 0.93 at
   layer 60 to 0.48 at the output. This replicates the paper's observation.
2. **The base model has almost no signal to suppress.** Its gap is zero until layer 63.
   C1 passes mainly because the instruct model has a big peak and the base has none.
   Under H1, the base should have a mid-layer signal that survives to the output; that is
   not what we see.
3. **Sign-flip issue:** `cv_method` / `fixed_method` in `analyze.py` pick the peak by
   |gap| and flip the sign to match. On introspect, the base gap is *negative*
   (injection → more "no"), so it was flipped and reported as +0.077, and counted toward
   `base_has_signal`. For detection, H1 is directional, so the sign should be fixed at +
   (proposed fix, not yet applied).
4. **The base "peak" is diluted:** the peak is at layer 63, so the ±2 window averages in
   layers 61–62 (~0). That makes base attenuation negative by construction.
5. **The fixed-window cross-check (75–95% depth) is meaningless with the logit lens.** That
   window is layers 48–60, where base has no signal and low mass, so you see random signs
   and huge frac values (±199). It disagrees with cv on introspect (−0.113, CI below 0).
6. **The seeds are near-duplicates:** the control P(yes) is identical across seeds and the
   injected values differ by < 0.01. The effective sample size is ~9 concepts, not 27, so
   the CIs are optimistic.
7. The legend warning in the plots is harmless (cosmetic plot bug).

Conclusion so far: with the logit lens, mid layers can't be read (no yes/no mass), so
"does the base model carry a mid-layer signal?" is **unanswered**. The tuned lens
(step 6) is what answers it, so run it next.

## Next actions

1. **Run step 6 (tuned lens) next**, before 4/5. It only needs the step 3 hiddens.
   It needs `data/lens_corpus.txt` or the `datasets` package in the project env
   (check: `uv run python -c "import datasets"`; if missing: `uv add datasets`).
2. Decide on the sign-flip fix (item 3 above) before trusting `base_has_signal`.
3. If the tuned lens still shows no base signal: the base few-shot prompt is the
   likely culprit (base says "yes" 53–62% with nothing injected). Revisit `FEWSHOT_TURNS`
   and re-run steps 2–3.
4. Then steps 4 → 5 → 7 → 8.
5. `posttrain/` is untracked in git. Commit it so the code has history.

Per-layer inspection script, in the Claude session scratchpad (may be gone later; it reads
`results.json` and prints ctrl/inj P(yes), gap and mass per layer):
`uv run --no-project --with numpy python <scratchpad>/inspect_gap.py 32b-base-plain-vinst-s42 32b-instruct-plain-s42`
