# MANUSCRIPT_EVIDENCE_BINDINGS_R010_R014.md

**Internal audit artifact. Not journal-facing prose.**

This file keeps the governance vocabulary removed from `paper_post_R014_v0_2.md`: claim IDs, DS classifications, canonical paths, evidence lineage, and source hashes. It does not alter any accepted result.

## Claim/evidence map

| Claim | Internal status/class | Canonical evidence | Source hash / provenance note |
|---|---|---|---|
| H0 | LEGACY_CONTEXT_ONLY | historical R008/R009/R008b paper-track artifacts | Legacy validation-PPL evidence; confounded for causal attribution; not comparable to corrected held-out test PPL. |
| C1 | SUPPORTED_FOR_TESTED_REGIME (negative result) | `R010_paired_attention_init_confirmation/metrics/per_seed.csv`; `metrics/summary.csv`; `metrics/paired_differences.csv` | per_seed `70995563095dcc233ad5a510738d5435562c519a1bedfeca1ef3207eaa0f5521`; summary `832c074fdad6862fc810e4aca2ac7cd88e41954b99a324336a1092e9d076e1d5`; paired `ac40565617c370f86d92211c730373fadccb204f7322ab41cfc0685a6c19d8e9`. |
| C2 | SUPPORTED_FOR_TESTED_REGIME | `R011_embedding_init_factorial_ablation/attempt_seed_atomic_v2/per_seed.csv`; `attempt_seed_atomic_v2/manifests/factorial_contrasts.csv`; `attempt_seed_atomic_v2/manifests/factorial_summary.csv` | Accepted R011 provenance limitation applies. No fully equivalent canonical release SHA is asserted here. |
| C3 | SCALE_DOMINATES | `R012_embedding_scale_redraw_factorial/per_seed.csv`; `manifests/factorial_summary.csv`; `manifests/factorial_contrasts.csv` | per_seed `7bc7b428efa2cb251954890a18ff9e81a687f5dadbccc125b3eb1f77c56b8d2b`; summary `5584218dd4167c53da8976522a799da87d8c40764c1688f1590990fc8894d205`; contrasts `6d1c363e299ee7eea9e52b8ec3cdeffa1de2d83e1eb0ccf9b6fd2f26679d997d`. Retained provenance limitations: full all-parameter `base_state` hash was not retained; unique provider-level physical Colab VM/session identity was not retained for every canonical attempt. Accepted evidence instead retains the base embedding hash, explicit non-embedding invariance hashes, exact factor construction, attention parity, all-epoch batch parity, and checkpoint evidence. These provenance limitations do not alter the accepted within-R012 factorial inference or `SCALE_DOMINATES` classification. |
| C4 | THRESHOLD_PLATEAU | `R013_embedding_scale_response_and_dynamics/per_seed.csv`; `manifests/dose_summary.csv`; `manifests/paired_contrasts.csv`; `manifests/paired_contrasts_summary.csv`; `manifests/scale_ladder.csv` | per_seed `72e4b649b663f5bee6afb0e6016ea37f3a3791289f1c4134df2905bca308e8fb`; dose summary `d2b5883bea6619672148cf26439d51e54caea3f30bc82ea4f3d060d5f70afd0a`; scale ladder `67ca39fdfaa99e80314026eedd61ffd406553875547cf8cfa0d1a3f149648eb4`; paired `4add3e154159c279e05a71352af07fff3ca78257bba40e9e20ef5db12d5f2e64`; paired summary `89e322b11b6e94c0cf4879a99205f97c7d2e8da51ba3ff5171f31a31a0da78db`. |
| C5 | SCALE_EFFECT_TRANSFERS | `R014_corrected_larger_model_embedding_scale_confirmation_v2/per_seed.csv`; `manifests/paired_contrasts.csv`; `manifests/paired_contrasts_summary.csv`; `manifests/scale_ladder.csv` | per_seed `0c8ec15c2bccc25b483f32cc0b07ed4d53b0a07dfc119d9203f0a9e80a70548b`; scale ladder `afd54743139595702abc89a67c8c6e8e33c17fc615de08d54a5c5c575b837a16`; paired `aba573d147786b1fd3c9ecefd19ed0b2a267a6be085241f4b3e25c798dfaf2b0`; paired summary `36dd74c0570fb5688d3dde92060f1e505ca0302bf23877e60f4d32554e338832`. |
| C6 | DIAGNOSTIC_ONLY | R013 dynamics plots/telemetry; R014 best-epoch/final-vs-best/validation-curve diagnostics | R013 has embedding RMS and gradient-L2 trajectories; R014 does **not** contain embedding RMS/gradient telemetry. No mechanism classification. |

## R011 accepted provenance limitation

The accepted seed-atomic v2 run used a global runtime that differed from the frozen smoke runtime. The deviation was common to all accepted R011 cells and was accepted as non-factorial. Exact historical source-bundle provenance was incomplete. R011 is therefore used for within-R011 paired factorial inference, while absolute cross-cycle values are not represented as exact same-runtime replication. This file deliberately does not retroactively strengthen that provenance.

## R012 accepted provenance limitations

Two limitations remain explicit and are not retrospectively strengthened: (1) a full all-parameter `base_state` hash was not retained; retained causal evidence consists of the base embedding hash, explicit non-embedding invariance hashes, exact factor construction, attention parity, all-epoch batch parity, and checkpoint evidence; and (2) a unique provider-level physical Colab VM/session identifier was not retained for every canonical attempt, so seed-atomic execution lineage is operational rather than an immutable provider-level VM identity. These are reproducibility/provenance limitations only. They do not invalidate the accepted within-R012 factorial inference and do not change the `SCALE_DOMINATES` result.

## Table-generation audit

`generated_tables/AUDIT_REPORT.json` must have `status: PASS`. The generator uses the accepted canonical CSV copies only and fails closed on any mismatch. The canonical R010–R014 summary files use the rounded Student-t critical value `t_{0.975,4}=2.776` for n=5 paired intervals; the generator reproduces that convention rather than substituting a different estimand.

## Figure lineage

See `FIGURE_BINDING_MANIFEST.csv`. R011 figure hashes are not promoted to canonical release assertions; the provenance limitation above remains binding.
