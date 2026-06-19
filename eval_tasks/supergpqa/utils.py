"""Doc preprocessing for the SuperGPQA custom lm-eval task.

SuperGPQA (m-a-p/SuperGPQA) is multiple-choice with a *variable* number of
options (up to 10), so we enumerate them as (A), (B), ... at load time and build
a zero-shot generative prompt that asks for the answer letter. Scoring is
exact-match on the gold `answer_letter` (the filter in supergpqa.yaml extracts
the parenthesised letter from the generation; `ignore_punctuation` drops the
parens before comparison).
"""

import datasets

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process(doc):
        options = doc["options"]
        choices = "\n".join(f"({LETTERS[i]}) {opt}" for i, opt in enumerate(options))
        query = (
            "Answer the following multiple choice question. The last line of your "
            'response must be exactly in the format "The answer is (X)" where X is '
            "the letter of the correct option.\n\n"
            f"Question: {doc['question']}\n{choices}\nAnswer:"
        )
        return {"query": query, "target": doc["answer_letter"]}

    return dataset.map(_process)
