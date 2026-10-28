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
| 6 | Tuned lens (needs `data/lens_corpus.txt` or the `datasets` package) | Done (wikitext). No base signal; see "Step 6 results" |
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
   `base_has_signal`. **Fixed (2026-10-28):** `detection_signal` is now `directional`, so
   the peak is the largest *positive* gap and the sign stays + (cv and fixed). Controls still
   use |gap|. Side effect: a gap that goes negative at the output now counts as positive
   attenuation (peak 0, final −0.13 → att +0.13), which makes C1 harder to pass, not easier.
   The step 3 numbers in this section predate the fix; re-run `compare`.
4. **The base "peak" is diluted:** the peak is at layer 63, so the ±2 window averages in
   layers 61–62 (~0). That makes base attenuation negative by construction.
5. **The fixed-window cross-check (75–95% depth) is meaningless with the logit lens.** That
   window is layers 48–60, where base has no signal and low mass, so you see random signs
   and huge frac values (±199). It disagrees with cv on introspect (−0.113, CI below 0).
6. **The seeds are near-duplicates:** the control P(yes) is identical across seeds and the
   injected values differ by < 0.01. The effective sample size is ~9 concepts, not 27, so
   the CIs are optimistic.
7. The legend warning in the plots is harmless (cosmetic plot bug).

**After the decision-rule fixes (2026-10-28):**
- `base_has_signal` now also needs peak ≥ `EXIST_MIN_GAP` (0.05), the same as step 0.
  Before that, a base introspect peak of +0.002 [+0.001, +0.004] passed.
- The H2 branch now says "baselines not run yet" when step 4 data is missing.

All three logit-lens comparisons now give **"Inconclusive: no middle-layer signal in the base
model"**. Base introspect: peak ≈ 0, final −0.13 (vself −0.19); injection pushes the base
toward "no". Base lorem (vinst): peak +0.078, but only at layer 63.

Conclusion so far: with the logit lens, mid layers can't be read (no yes/no mass), so
"does the base model carry a mid-layer signal?" is **unanswered**. The tuned lens
(step 6) is what answers it, so run it next.

## Step 6 results (tuned lens, 2026-10-28)

Lens quality is good. Mid-layer KL to the final output: logit lens ~9.5, tuned lens ~2.6
(instruct and base alike).

| | Base-vinst lorem | Base-vinst introspect | Base-vself lorem / introspect | Instruct-plain lorem | Instruct-plain introspect |
|---|---|---|---|---|---|
| Peak gap (depth) | +0.100 (0.98) | +0.002 [−0.003, +0.008] (0.64) | +0.027 / +0.020 (0.64) | +0.678 (0.89) | +0.317 (0.83) |
| Final gap | +0.130 | −0.127 | −0.091 / −0.193 | +0.479 | +0.259 |

Verdict in all three comparisons: **Inconclusive: no middle-layer signal in the base model.**
- The tuned lens doesn't rescue the base model. Its only positive gap is on lorem with
  instruct vectors, and only at the last layer or two.
- The instruct signal is also late: it peaks at 0.83–0.89 depth (layers 53–57), not mid-layer.
  It then attenuates to the output (lorem 0.68 → 0.48; chat introspect 0.44 → 0.27).
- So H1 can't be tested as set up. Two readings remain: (a) the base few-shot prompt
  doesn't teach the base model what the question means (it is at ~chance: P(yes) 0.53–0.62
  with nothing injected, and injection pushes it toward "no"); or (b) the base model has no
  yes/no-readable detection signal at all, i.e. post-training *creates* the readout rather than
  suppressing an existing one.
- **Not in the verdict:** under the tuned lens, injection pushes the base model toward "no" in
  layers 45–55. Mean gap over layers 47–51 (window picked after looking, so exploratory):
  base-vinst introspect −0.247 [−0.280, −0.215], base-vself introspect −0.314, base-vself
  lorem −0.123; 9/9 concepts negative in each. It shrinks toward the output (−0.33 at L49 →
  −0.13 final). The directional sign fix ignores this by design. The base readout is
  injection-sensitive; what's missing is the "injected → yes" direction H1 needs.
- Report: https://claude.ai/artifact/BfpjE5BKC2CfqqdonnVCqt
- `FEWSHOT_TURNS` is currently 4 world-knowledge questions (yes/no/yes/no), with nothing
  that depends on the conversation or the model's own state.

## Next actions

1. **Run step 6 (tuned lens) next**, before 4/5. It only needs the step 3 hiddens.
   It needs `data/lens_corpus.txt` or the `datasets` package in the project env
   (check: `uv run python -c "import datasets"`; if missing: `uv add datasets`).
2. ~~Sign-flip fix~~ done (item 3 above); re-run the three `compare` commands.
3. If the tuned lens still shows no base signal: the base few-shot prompt is the
   likely culprit (base says "yes" 53–62% with nothing injected). Revisit `FEWSHOT_TURNS`
   and re-run steps 2–3.
4. Then steps 4 → 5 → 7 → 8.
5. `posttrain/` is untracked in git. Commit it so the code has history.

Per-layer inspection script, in the Claude session scratchpad (may be gone later; it reads
`results.json` and prints ctrl/inj P(yes), gap and mass per layer):
`uv run --no-project --with numpy python <scratchpad>/inspect_gap.py 32b-base-plain-vinst-s42 32b-instruct-plain-s42`
