# Public Release Checklist

Pre-release verification for `copeland-erdos-nets`.

## Secrets & Privacy

- [ ] No API keys, tokens, or credentials in any tracked file
- [ ] No private paths (`/home/mike`, `.control/`, `agent-rules-tree`) in public files
- [ ] `.control/` is in `.gitignore` and not tracked
- [ ] No PII in logs or test data

## Clean Clone Install

- [ ] `git clone` succeeds on a clean machine
- [ ] `make install` creates `.venv` and installs all dependencies
- [ ] `make check-env` passes
- [ ] `pip list` shows `datasets` and `transformers` installed

## Tests

- [ ] `make test` — at least 160 tests pass
- [ ] `make lint` — no new errors introduced

## Reproducibility Smoke

- [ ] R008b command starts without missing-dependency errors:
      `python scripts/run_transformer_screening.py --config configs/r008b_controls.json --output .tmp/r008b_smoke`
      (can be interrupted after first epoch)
- [ ] Tokenizer loads as GPT-2 with vocab_size = 50 257

## Artifact Integrity

- [ ] `paper_artifacts/artifacts_checksums.txt` contains SHA-256 (64 hex chars) for all artifacts
- [ ] `sha256sum -c paper_artifacts/artifacts_checksums.txt` passes
- [ ] `best_epoch` values in canonical CSVs are non-zero (4.0–7.2 range)
- [ ] `all_experiments_summary.csv` contains all 15 methods
- [ ] Figures match manuscript references (fig01–fig08)

## Claims Consistency

- [ ] README claims do not exceed `paper_artifacts/claims_ledger.md`
- [ ] No "CE superiority" or "SOTA" language anywhere in public files
- [ ] Parameter counts in README match actual model sizes
- [ ] `claims_ledger.md` C3 wording is cautious ("consistent with", not "driver")

## Documentation

- [ ] README links to `paper_artifacts/` and `claims_ledger.md`
- [ ] `CITATION.cff` present and valid
- [ ] `scripts/README.md` has script index
- [ ] `configs/README.md` has experiment index + vocab_size note
- [ ] `docs/README.md` has method documentation

## Final Steps

- [ ] Merge `prepare-publication` branch into `master`
- [ ] Set repository visibility to public on GitHub
- [ ] Tag release: `git tag v0.3.0`
