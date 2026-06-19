"""process_results for the vendored IFBench task.

Mirrors lm-eval's built-in ifeval utils, but imports the *IFBench* verifier
registry (vendored alongside this file) instead of lm-eval's IFEval one — IFBench
adds 57 new verifiable constraints whose check functions live in the local
instructions*.py modules. The InputExample interface (build_description /
get_instruction_args / check_following) is identical to IFEval, so the
strict/loose scoring logic is unchanged.
"""

import dataclasses
import os
import sys
from typing import Dict, Optional, Union

# lm-eval loads this module by file path (via `!function`) without putting its
# directory on sys.path, so make the vendored sibling modules importable here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Route nltk data to the persistent HF cache volume (fast ext4) instead of letting
# it land in the workspace next to this file. Set before nltk is imported/used.
_NLTK_DIR = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "nltk_data"
)
os.makedirs(_NLTK_DIR, exist_ok=True)
os.environ.setdefault("NLTK_DATA", _NLTK_DIR)

import instructions_registry  # noqa: E402  (vendored: eval_tasks/ifbench/)
import instructions_util  # noqa: E402

# IFBench verifiers use nltk (punkt, stopwords, POS tagger). Fetch once at import.
try:
    import nltk

    if _NLTK_DIR not in nltk.data.path:
        nltk.data.path.insert(0, _NLTK_DIR)
    instructions_util.download_nltk_resources()
except Exception:
    pass


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(inp, response):
    is_following_list = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(inp, response):
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    is_following_list = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for sample in all_responses:
            if sample.strip() and instruction.check_following(sample):
                is_following = True
                break
        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]
    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)
    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    return sum(flat_items) / len(flat_items)
