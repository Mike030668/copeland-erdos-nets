"""Regression checks for CE-P1 execution and durable-state contracts."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_and_smoke_configs_are_closed_and_distinct():
    canonical = json.loads((ROOT / "configs" / "ce_p1_canonical.json").read_text())
    smoke = json.loads((ROOT / "configs" / "ce_p1_smoke.json").read_text())
    expected = ["Gaussian_Xavier_RMS", "CE_LCG_Xavier_RMS", "CE_SHUFFLED_Xavier_RMS"]
    assert canonical["experiment"]["conditions"] == expected
    assert smoke["experiment"]["conditions"] == expected
    assert canonical["seeds"] == [62, 63, 64, 65, 66]
    assert smoke["seeds"] == [1062]


def test_runner_records_live_path_b_state_and_preserves_best_checkpoint_selection():
    source = (ROOT / "scripts" / "run_ce_p1_seed_atomic.py").read_text()
    assert "class DriveLiveSync" in source
    assert "CE_P1_GDRIVE_LIVE=1 is required" in source
    assert "os.replace(temporary, path)" in source
    assert "best_checkpoint" in source and "resume_checkpoint" in source
    assert 'model.load_state_dict(torch.load(best_checkpoint' in source
    assert "sync.update(f\"{prefix}_checkpoint_{condition}.pt\", checkpoint)" in source
    assert 'publish("completed", "", epochs)' in source
