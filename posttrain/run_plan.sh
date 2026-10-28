#!/usr/bin/env bash
# Section 5.1 (H1) follow-up: the plan's steps, one per invocation.
#
#   SIZE=7b bash posttrain/run_plan.sh <step>
#
# Steps (in plan order):
#   0cal  calibrate the instruct model's injection strength      -> pick S_INST
#   0     existence check on the instruct model (5 concepts, 1 seed)
#           pass -> keep SIZE=7b;  fail -> redo 0cal and 0 with SIZE=32b
#   1     (nothing to run: analyze.py is the analysis code; it runs inside each step)
#   2     calibrate the base model with both vector sources       -> pick S_BASE_VINST, S_BASE_VSELF
#   3     main comparison, all seeds, both vector sources, both instruct formats
#   4     controls + concept-in-context baseline
#   5     concept identification (per-layer MI)
#   6     tuned lens: train per checkpoint, apply to all runs, re-analyze
#   7     causal tests: swaps in both directions, ablation, DLA (only if step 3 shows a difference)
#   8     self-claims (H1 vs H3)
#   9cal  OLMo-2 32B: calibrate every stage                       -> pick S_OLMO_*
#   9     OLMo-2 32B: stages base -> SFT -> DPO -> Instruct (only if step 7 supports H1)
#   10    (manual) search the OLMo SFT/DPO data for refusals to claim self-knowledge
#
# Strengths are chosen by reading the calibration samples: the lowest strength at which
# the top-of-mind answer is clearly about the injected concept but still coherent.
# Export them before the steps that need them, e.g. S_INST=8 S_BASE_VINST=8 S_BASE_VSELF=6.
#
# Vectors are cached per model id and seed (outputs/vectors/). Instruct plain-format runs
# go first in each step, so the instruct chat-format runs and the base runs with
# --vector-source reuse exactly the same vectors.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-"uv run python"}
OUT=outputs/posttrain
SIZE=${SIZE:-7b}
SEEDS=${SEEDS:-"42 43 44"}
FIRST_SEED=${SEEDS%% *}
STEP=${1:?usage: SIZE=7b|32b bash posttrain/run_plan.sh <step>}

case $SIZE in
  7b)
    BASE=Qwen/Qwen2.5-Coder-7B
    INST=Qwen/Qwen2.5-Coder-7B-Instruct
    LAYERS=9:19 # middle third of 28 layers
    ;;
  32b)
    BASE=Qwen/Qwen2.5-Coder-32B
    INST=Qwen/Qwen2.5-Coder-32B-Instruct
    LAYERS=21:43 # as in the paper
    ;;
  *) echo "SIZE must be 7b or 32b" >&2; exit 1 ;;
esac

L_INST=$SIZE-instruct-plain
L_INST_CHAT=$SIZE-instruct-chat
L_BASE_VINST=$SIZE-base-plain-vinst
L_BASE_VSELF=$SIZE-base-plain-vself
ALL_LABELS="$L_INST $L_INST_CHAT $L_BASE_VINST $L_BASE_VSELF"

need() { for v in "$@"; do [[ -n ${!v:-} ]] || { echo "Set $v first (see the calibration step)." >&2; exit 1; }; done; }

# run <label> <seed> <args...>: one run_model.py invocation; tag = <label>-s<seed>.
run() {
  local label=$1 seed=$2
  shift 2
  $PY posttrain/run_model.py --label "$label" --tag "$label-s$seed" --seed "$seed" \
    --fewshot --conditions primary --layer-range "$LAYERS" --out "$OUT" "$@"
}

# The four run types. Every later step reuses these, so a run dir's settings never change.
inst_plain() { need S_INST; run "$L_INST" "$1" --model "$INST" --format plain --vector-strength "$S_INST" "${@:2}"; }
inst_chat() { need S_INST; run "$L_INST_CHAT" "$1" --model "$INST" --format chat --vector-strength "$S_INST" "${@:2}"; }
base_vinst() { need S_BASE_VINST; run "$L_BASE_VINST" "$1" --model "$BASE" --format plain --vector-source "$INST" --vector-strength "$S_BASE_VINST" "${@:2}"; }
base_vself() { need S_BASE_VSELF; run "$L_BASE_VSELF" "$1" --model "$BASE" --format plain --vector-strength "$S_BASE_VSELF" "${@:2}"; }

# all_runs <seed> <args...>: the four run types for one seed, instruct plain first.
all_runs() { inst_plain "$@"; inst_chat "$@"; base_vinst "$@"; base_vself "$@"; }

compare_all() {
  $PY posttrain/analyze.py compare --base "$L_BASE_VINST" --instruct "$L_INST" --name "$SIZE-primary"
  $PY posttrain/analyze.py compare --base "$L_BASE_VSELF" --instruct "$L_INST" --name "$SIZE-own-vectors"
  $PY posttrain/analyze.py compare --base "$L_BASE_VINST" --instruct "$L_INST_CHAT" --name "$SIZE-instruct-chat"
}

calibrate() {
  local tag=$1
  shift
  $PY posttrain/run_model.py --tag "$tag" --seed "$FIRST_SEED" --layer-range "$LAYERS" --format plain \
    --vector-strength 0 --phases calibrate --out "$OUT" "$@"
}

case $STEP in
  0cal)
    calibrate "$SIZE-calibrate-instruct" --model "$INST"
    echo "Pick S_INST from the samples above, then run step 0."
    ;;

  0)
    need S_INST
    run "$SIZE-exists" "$FIRST_SEED" --model "$INST" --format plain --vector-strength "$S_INST" \
      --concepts programming,love,death,truth,music --phases main
    if $PY posttrain/analyze.py exists --label "$SIZE-exists"; then
      echo "Signal exists at $SIZE: continue with step 2 (SIZE=$SIZE)."
    else
      echo "No signal at $SIZE: rerun steps 0cal and 0 with SIZE=32b."
    fi
    ;;

  1)
    echo "Step 1 is the analysis code (posttrain/analyze.py); nothing to run."
    ;;

  2)
    # Base needs the instruct vectors cached first (step 0 trained them for the first seed).
    calibrate "$SIZE-calibrate-base-vinst" --model "$BASE" --vector-source "$INST"
    calibrate "$SIZE-calibrate-base-vself" --model "$BASE"
    echo "Pick S_BASE_VINST and S_BASE_VSELF to match the instruct model's steering at S_INST."
    ;;

  3)
    for seed in $SEEDS; do all_runs "$seed" --phases main; done
    compare_all
    ;;

  4)
    for seed in $SEEDS; do
      phases=controls
      [[ $seed == "$FIRST_SEED" ]] && phases=controls,baseline # baseline has no vectors; one seed
      all_runs "$seed" --phases "$phases"
    done
    compare_all
    ;;

  5)
    for seed in $SEEDS; do all_runs "$seed" --phases confusion; done
    compare_all
    ;;

  6)
    if [[ ! -f data/lens_corpus.txt ]] && ! $PY -c "import datasets" 2>/dev/null; then
      echo "Provide data/lens_corpus.txt (generic English, documents separated by blank lines)" >&2
      echo "or install the 'datasets' package for the wikitext fallback." >&2
      exit 1
    fi
    $PY posttrain/tuned_lens.py train --model "$INST"
    $PY posttrain/tuned_lens.py train --model "$BASE"
    for label in $ALL_LABELS; do
      for seed in $SEEDS; do $PY posttrain/tuned_lens.py apply --run-dir "$OUT/$label-s$seed"; done
    done
    compare_all
    ;;

  7)
    # Same tags as the first-seed runs, so the unmodified results sit next to the swaps.
    inst_plain "$FIRST_SEED" --phases swap --swap-from "$BASE"
    base_vinst "$FIRST_SEED" --phases swap --swap-from "$INST"
    inst_plain "$FIRST_SEED" --phases ablate,dla --ablate-ref "$OUT/$L_BASE_VINST-s$FIRST_SEED"
    base_vinst "$FIRST_SEED" --phases dla
    $PY posttrain/analyze.py compare --base "$L_BASE_VINST" --instruct "$L_INST" --name "$SIZE-primary"
    $PY posttrain/analyze.py dla --labels "$L_BASE_VINST,$L_INST"
    ;;

  8)
    inst_plain "$FIRST_SEED" --phases selfclaims
    inst_chat "$FIRST_SEED" --phases selfclaims
    base_vinst "$FIRST_SEED" --phases selfclaims
    $PY posttrain/analyze.py selfclaims --labels "$L_BASE_VINST,$L_INST"
    $PY posttrain/analyze.py selfclaims --labels "$L_BASE_VINST,$L_INST_CHAT"
    ;;

  9cal | 9)
    # OLMo-2 32B: 64 layers, same middle third as Qwen-32B. All stages use the Instruct
    # checkpoint's vectors so the only thing changing between stages is the weights.
    LAYERS=21:43
    OLMO_STAGES=(base:allenai/OLMo-2-0325-32B sft:allenai/OLMo-2-0325-32B-SFT
      dpo:allenai/OLMo-2-0325-32B-DPO instruct:allenai/OLMo-2-0325-32B-Instruct)
    OLMO_INST=allenai/OLMo-2-0325-32B-Instruct
    if [[ $STEP == 9cal ]]; then
      calibrate olmo32b-calibrate-instruct --model "$OLMO_INST"
      for entry in "${OLMO_STAGES[@]:0:3}"; do
        calibrate "olmo32b-calibrate-${entry%%:*}" --model "${entry#*:}" --vector-source "$OLMO_INST"
      done
      echo "Pick S_OLMO_BASE, S_OLMO_SFT, S_OLMO_DPO, S_OLMO_INSTRUCT, then run step 9."
      exit 0
    fi
    need S_OLMO_BASE S_OLMO_SFT S_OLMO_DPO S_OLMO_INSTRUCT
    labels=()
    for seed in $SEEDS; do
      # Instruct first: it trains the vectors the other stages reuse.
      for entry in "${OLMO_STAGES[3]}" "${OLMO_STAGES[@]:0:3}"; do
        stage=${entry%%:*} model=${entry#*:}
        strength_var=S_OLMO_${stage^^}
        phases=main,controls,confusion
        [[ $seed == "$FIRST_SEED" ]] && phases=$phases,baseline,selfclaims
        run "olmo32b-$stage" "$seed" --model "$model" --format plain --vector-source "$OLMO_INST" \
          --vector-strength "${!strength_var}" --phases "$phases"
      done
    done
    for entry in "${OLMO_STAGES[@]}"; do
      stage=${entry%%:*} model=${entry#*:}
      labels+=("olmo32b-$stage")
      $PY posttrain/tuned_lens.py train --model "$model"
      for seed in $SEEDS; do $PY posttrain/tuned_lens.py apply --run-dir "$OUT/olmo32b-$stage-s$seed"; done
    done
    stage_list=$(IFS=,; echo "${labels[*]}")
    $PY posttrain/analyze.py stages --labels "$stage_list" --lens logit
    $PY posttrain/analyze.py stages --labels "$stage_list" --lens tuned
    $PY posttrain/analyze.py compare --base olmo32b-base --instruct olmo32b-instruct --name olmo32b-base-vs-instruct
    $PY posttrain/analyze.py selfclaims --labels "olmo32b-base,olmo32b-instruct"
    ;;

  10)
    echo "Manual: search the OLMo-2 SFT and DPO training mixes (Tulu 3 style) for responses that"
    echo "deny introspection or self-knowledge, focusing on the stage step 9 flagged first."
    ;;

  *) echo "Unknown step: $STEP" >&2; exit 1 ;;
esac
