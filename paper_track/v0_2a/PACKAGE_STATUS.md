# PACKAGE_STATUS
```text
PACKAGE: paper_track/v0_2a — post-R014 manuscript technical package
PACKAGE_VERSION: v0_2a
PACKAGE_STATUS: ACCEPTED (DS Colleague, P5_v0_2a micro-revalidation)
SCIENTIFIC_GATE: P5_V0_2A_MICRO_REVALIDATION = PASS
ACCEPTED_MANUSCRIPT: paper_post_R014_v0_2a.md
ACCEPTED_MANUSCRIPT_SHA256: 1238b917dcc881c07e2a48603caca87e526bc733dcffe9e9034a23e12e93464b
SOURCE_V0_2_MANUSCRIPT_SHA256: e71edbe7e0c4a3d28af0d32a266234a8cad8e0c27b618d80b5e15525f6ca5c57 (immutable, unchanged by this micro-repair)
NUMERICAL_VALUES_CHANGED: false
FIGURES_CHANGED: false
EXPERIMENTS_RUN: false
DS_CLASSES_CHANGED: false
TABLES_2_6_TEXT_UNCHANGED: true
TABLE_7_CHANGED: true (table_7_reproducibility_v0_2a.md — the only table artifact that changed)
R014_PRIMARY_CONTRAST: D_xavier - D_ctor = -34.7442 [-37.5676, -31.9208] test PPL, n=5, CI excludes 0, 5/5 negative seeds
R014_DS_TRANSFER_CLASS: SCALE_EFFECT_TRANSFERS
R014_RELEASE_HEAD: 3b14595b99312ee14261235338fee9a5861e2cd2 (branch protocol/r014-larger-model-confirmation)
PRE_SUBMISSION_REVIEW: NOT_READY
READY_FOR_JOURNAL_SUBMISSION: NO
CURRENT_OWNER: DS_COLLEAGUE
LAST_UPDATED: 2026-09-14
```

## What this directory is

A self-contained, git-tracked technical mirror of the DS-accepted post-R014
manuscript package v0_2a, materialized from `@_gdrive_ds/paper_track/draft_paper/`
(Drive is not a git repository and not the paper source of truth — this
directory is). Every byte here is either unchanged since the frozen `v0_2`
package (figures, Tables 1-6, generator scripts, canonical evidence CSVs —
all sha256-verified against `V0_2_PACKAGE_MANIFEST.json` before copying) or
part of the DS-accepted `v0_2a` micro-repair delta (the manuscript itself,
the corrected Table 7, `MANUSCRIPT_EVIDENCE_BINDINGS_R010_R014.md`,
`FIGURE_TABLE_REGISTRY_R010_R014.md`, `PAPER_REVISION_CHANGELOG.md` — all
sha256-verified against `PACKAGE_MANIFEST_v0_2a.json` before copying, and
the manuscript's own hash cross-checked against the exact value DS
specified). No prose was authored or edited by the Architect; no claim,
number, figure, or DS classification was changed. See
`PAPER_REVISION_CHANGELOG.md`'s final section ("v0_2 → v0_2a bounded
provenance-disclosure repair") for the itemized delta, and
`MICRO_REVALIDATION_AUDIT.json` for the DS-facing PASS audit this
materialization mirrors verbatim.

The directory layout (`manuscript_figures/`, `generated_tables/`,
`scripts/`, `canonical_inputs/R0{10..14}/`) matches the manuscript's own
relative image/reference paths and the structure already recorded in
`V0_2_PACKAGE_MANIFEST.json`/the original `P5_v0_2_audit_package.zip` — not
an Architect-invented structure.
