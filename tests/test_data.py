"""`mix_corpora` — row-count weighting + deterministic shuffle over local fixtures.

Uses local .txt files only (no `hf:` specs, no network) via the shared
`llm_core.corpus.load_corpus`.
"""

import pytest

from llm_core.training.data import mix_corpora


@pytest.fixture
def corpora(tmp_path):
    domain = tmp_path / "domain.txt"
    general = tmp_path / "general.txt"
    domain.write_text("\n".join(f"domain line {i}" for i in range(20)) + "\n")
    general.write_text("\n".join(f"general line {i}" for i in range(20)) + "\n")
    return [
        {"role": "domain", "spec": str(domain), "weight": 0.5},
        {"role": "general", "spec": str(general), "weight": 0.5},
    ]


def test_row_counts_follow_weights(corpora):
    ds, role_counts = mix_corpora(corpora, total_samples=10, seed=0)
    assert role_counts == {"domain": 5, "general": 5}
    assert len(ds) == 10
    assert set(ds.column_names) == {"text", "role"}


def test_rows_tagged_with_role(corpora):
    ds, _ = mix_corpora(corpora, total_samples=10, seed=0)
    for row in ds:
        assert row["role"] in {"domain", "general"}
        assert (
            row["role"].split()[0] in row["text"]
        )  # 'domain'/'general' prefix matches tag


def test_shuffle_is_deterministic(corpora):
    a, _ = mix_corpora(corpora, total_samples=10, seed=123)
    b, _ = mix_corpora(corpora, total_samples=10, seed=123)
    assert a["text"] == b["text"]


def test_shuffle_differs_by_seed(corpora):
    a, _ = mix_corpora(corpora, total_samples=10, seed=1)
    b, _ = mix_corpora(corpora, total_samples=10, seed=2)
    assert a["text"] != b["text"]


def test_weight_capped_by_availability(corpora):
    """Asking for more rows than a corpus has is capped to what's available (20 each)."""
    _, role_counts = mix_corpora(corpora, total_samples=100, seed=0)
    assert role_counts == {"domain": 20, "general": 20}


def test_missing_spec_raises():
    with pytest.raises(ValueError):
        mix_corpora([{"role": "domain", "weight": 1.0}], total_samples=4)
