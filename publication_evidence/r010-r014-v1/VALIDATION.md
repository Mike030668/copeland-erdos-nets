# Gate A release validation — 2026-10-07

Accepted source copies: 160 files, all source/release bytes and SHA-256 equal.
Code/config/tests: identical to R014 base
`3b14595b99312ee14261235338fee9a5861e2cd2`.
Paper snapshots: identical to `12ae7f49ca38ed24a28aaad18b9b45b24862c650`
and `5af18387395009ee35f6c5e809934562733184d6` at their original public paths.
CE-P1 files are absent from the release tree and branch ancestry.

SHA inventory verification: `sha256sum -c` passed for every listed file.
The inventory is regenerated after the final receipt edit and verified against
committed blobs as well as the filesystem. Inventory files exclude themselves
where necessary to prevent recursive hashing; `SHA256SUMS` includes the CSV
inventory's own digest.

Inherited targeted test command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_r010_protocol.py tests/test_r011_protocol.py \
  tests/test_r012_protocol.py tests/test_r013_protocol.py \
  tests/test_r014_protocol.py -q
```

Outcome: **65 passed, 1 failed**, one existing unregistered `slow` marker warning.
Failure: `test_production_attention_hash_parity_seed42`, comparing current
`blocks.0.attn.k_proj.weight` tensor construction with a historical local R010
receipt. The independently constructed tensor did not match the recorded
historical tensor digest; no replacement digest was written into evidence.

The failure is recorded, not classified as a resolved runtime discrepancy.
The current host is Python 3.12.3, Torch 2.12.1+cu130, NumPy 2.4.6; it is not
the accepted R011 execution runtime. The test reads an untracked historical
scratch receipt and would skip when that receipt is absent. Its source SHA-256
`577f906b1a53692c95304fa704f0b2dc77a996bcbc6d56374319e2d6f81227e6`
matches the accepted base exactly. All production code remains unchanged.

No accepted training run, accepted metric recomputation, checkpoint re-selection
or new statistical analysis occurred. The inherited tests include small
synthetic CPU training as code validation only. No GPU experiment was launched.
No pre-commit configuration or project quality-gate script was present.
Copied canonical CSVs preserve their original CRLF bytes. Git whitespace
validation treats CR at end of line as retained source formatting.

Final DS intake must evaluate the disclosed test failure and provenance gaps.
This receipt does not declare Gate A PASS.
