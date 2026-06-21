"""General sweep primitives — grid enumeration + Pareto front (`llm_core.sweep`)."""

from llm_core.sweep import enumerate_grid, pareto_front

_ID = "lr{learning_rate:g}_e{num_train_epochs:g}_n{num_samples}_r{lora_rank}"


def test_enumerate_grid_main_grid_plus_extra_dedups_center():
    # Main grid (2 lr × 1 epoch × 1 n at rank 16) + a rank sub-sweep at the lr2e-4 center;
    # the r16 sub-sweep point duplicates the main-grid center and must collapse to one.
    axes = {
        "learning_rate": [1e-4, 2e-4],
        "num_train_epochs": [1],
        "num_samples": [1000],
        "lora_rank": [16],
    }
    extra = [
        {
            "learning_rate": 2e-4,
            "num_train_epochs": 1,
            "num_samples": 1000,
            "lora_rank": r,
        }
        for r in (8, 16, 32)
    ]
    ids = [p["id"] for p in enumerate_grid(axes, extra=extra, id_format=_ID)]
    assert len(ids) == len(set(ids)) == 4
    assert ids.count("lr0.0002_e1_n1000_r16") == 1  # shared center appears once
    assert {"lr0.0002_e1_n1000_r8", "lr0.0002_e1_n1000_r32"} <= set(ids)


def test_enumerate_grid_keeps_axis_values_on_each_point():
    points = enumerate_grid({"a": [1, 2], "b": [9]}, id_format="{a}-{b}")
    assert points == [
        {"id": "1-9", "a": 1, "b": 9},
        {"id": "2-9", "a": 2, "b": 9},
    ]


def test_pareto_front_minimizes_both_axes():
    items = [
        {"id": "a", "x": -5.0, "y": 3.0},  # most learning, most forgetting
        {"id": "b", "x": -3.0, "y": 1.0},  # middle — non-dominated
        {"id": "c", "x": -1.0, "y": 0.5},  # least learning, least forgetting
        {"id": "d", "x": -2.0, "y": 2.0},  # dominated by b
    ]
    front = {p["id"] for p in pareto_front(items, "x", "y")}
    assert front == {"a", "b", "c"}


def test_pareto_front_drops_strictly_dominated_only():
    items = [{"id": "p", "x": 0.0, "y": 0.0}, {"id": "q", "x": 1.0, "y": 1.0}]
    assert {p["id"] for p in pareto_front(items, "x", "y")} == {"p"}
