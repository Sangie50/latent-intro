"""Prompt content for the Section 5.1 (H1) follow-up that isn't in main.py.

- PRIMARY_CONDITIONS: the two pre-registered conditions the conclusion rests on.
- FEWSHOT_TURNS / add_fewshot: neutral worked yes/no turns prepended to every prompt,
  so base models answer in the "The answer is yes/no" format.
- context_baseline_prompts: specificity baseline where a concept appears in plain
  text (no injection) and the model answers a yes/no question about the context.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# main/helpers are imported inside the functions below so that analyze.py can import
# PRIMARY_CONDITIONS without torch.

# Pre-registered primary conditions (prompt labels from main.make_turn_1_label_prompt):
#   Vague Mechanism + Matched Lipsum Filler, Accurate Mechanism + Pro-Introspection Doc.
PRIMARY_CONDITIONS = ["salient_info-lorem-ipsum", "inject_info-introspect"]

# Neutral yes/no questions, balanced 2 yes / 2 no, nothing about introspection, and
# none of them overlap with the control questions in main.py.
FEWSHOT_TURNS = [
    ("Is Paris the capital of France?", "yes"),
    ("Do fish have fur?", "no"),
    ("Is a triangle a polygon?", "yes"),
    ("Is the Pacific Ocean smaller than a lake?", "no"),
]


def fewshot_messages() -> list[dict[str, Any]]:
    import main as m

    msgs: list[dict[str, Any]] = []
    for q, a in FEWSHOT_TURNS:
        msgs.append({"role": "user", "content": f"{q} {m.CONTROL_QUESTION_SUFFIX_YESNO}"})
        msgs.append({"role": "assistant", "content": f"The answer is {a}."})
    return msgs


def add_fewshot(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Insert the few-shot turns after the (possibly empty) system message."""
    i = 1 if steps and steps[0]["role"] == "system" else 0
    return [*steps[:i], *fewshot_messages(), *steps[i:]]


# Concept-in-context baseline. The concept sentence and the neutral sentence differ
# only in whether a topic is named; the question is about the conversation, not the
# model's internal state.
CONTEXT_CONCEPT_SENTENCE = "Before we begin: I have been thinking a lot about {concept} lately."
CONTEXT_NEUTRAL_SENTENCE = "Before we begin: I have been thinking a lot lately."
CONTEXT_QUESTION = "Did my first message mention a specific topic?"


def context_baseline_prompts(concepts: list[str], conditions: list[str]) -> list[tuple[dict[str, Any], Any]]:
    """(meta, Conversation) pairs; meta has prompt, concept, kind in {concept, neutral}."""
    import main as m
    from helpers import Conversation

    fmt = m.CONTROL_QUESTION_ANSWER_FORMAT_YES_NO
    out = []
    for intervention in m.INTERVENTIONS:
        for info_label, info_doc in m.INFO_DOCUMENTS:
            prompt_label, user_turn_1 = m.make_turn_1_label_prompt(
                intervention, info_label, info_doc
            )
            if prompt_label not in conditions:
                continue
            for concept in concepts:
                for kind, sentence in (
                    ("concept", CONTEXT_CONCEPT_SENTENCE.format(concept=concept)),
                    ("neutral", CONTEXT_NEUTRAL_SENTENCE),
                ):
                    cv = Conversation(
                        label=f"{prompt_label}_{concept}_{kind}",
                        yes_token=fmt.yes_token,
                        no_token=fmt.no_token,
                        extra={},
                        messages=[
                            {"role": "system", "content": ""},
                            {"role": "user", "content": f"{sentence}\n\n{user_turn_1}"},
                            {"role": "assistant", "content": m.ASST_TURN_1},
                            {
                                "role": "user",
                                "content": f"{m.USER_TURN_2_PREFIX} {CONTEXT_QUESTION} {fmt.turn_2_format_desc}",
                            },
                            {"role": "assistant", "content": fmt.assistant_prefix, "continue_msg": True},
                        ],
                    )
                    out.append(({"prompt": prompt_label, "concept": concept, "kind": kind}, cv))
    return out
