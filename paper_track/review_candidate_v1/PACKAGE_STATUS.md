# PACKAGE_STATUS
```text
PACKAGE: paper_track/review_candidate_v1 — post-adversarial-re-review manuscript technical snapshot
PACKAGE_VERSION: v0_3 (review-candidate v1)
PACKAGE_STATUS: ACCEPTED (DS Colleague, Internal Adversarial Re-Review)
SCIENTIFIC_GATE: INTERNAL_ADVERSARIAL_RE_REVIEW = PASS
ACCEPTED_MANUSCRIPT_SOURCE: paper_post_R014_v0_3 (native Google Doc, file ID 1RMM6rLDt7zUvd7dodP2tEnwXAlJ1rXMUFD-EyWfR0Ng)
RAW_MARKDOWN_MIRROR: paper_post_R014_v0_3.md
RAW_MARKDOWN_MIRROR_SHA256: c43e0a3d074a04ec82297096574b8ec7204e1e6c09b87bd3facb52b7500728af
MIRROR_TRANSFORMATION: mechanical only (Docs-export markdown-escape unescaping + table/heading restructuring) -- see TECHNICAL_PROVENANCE_NOTE_v0_3.md for the itemized, auditable list; zero words/numbers/claims changed
TABLES_2_6_VALUES: RETAINED (byte-identical to v0_2a, verified)
TABLE_7: RETAINED (byte-identical to v0_2a's table_7_reproducibility_v0_2a.md, verified)
FIGURES: RETAINED, all 10 byte-identical to v0_2a, verified
NUMERICAL_EVIDENCE: RETAINED -- R014 primary contrast D_xavier - D_ctor = -34.7442 [-37.5676, -31.9208] unchanged
DS_SCIENTIFIC_CLASSIFICATIONS: UNCHANGED
PRECEDING_IMMUTABLE_SNAPSHOT: paper_track/v0_2a/ at commit 5af18387395009ee35f6c5e809934562733184d6, branch paper-track/v0_2a-materialization -- preserved, NOT modified by this package
DOCX_AUTHORING_GATE: AUTHORIZE_DOCX_REVIEW_MASTER (Paper Agent, in parallel with this snapshot)
PRE_SUBMISSION_REVIEW: NOT_READY
CURRENT_OWNER: DS_COLLEAGUE
LAST_UPDATED: 2026-09-14
```

## What this directory is

A self-contained, git-tracked technical snapshot of the DS-accepted
post-adversarial-re-review manuscript `paper_post_R014_v0_3`, materialized
from its native Google Doc form into raw Markdown for the first time (no
prior `.md` mirror existed for this version, unlike `v0_2a` which already
had one on Drive). `manuscript_figures/`, `generated_tables/table_{1..7}`,
`scripts/`, `canonical_inputs/R0{10..14}/`, the evidence-binding and
figure/table registry documents, and the bibliography sources are carried
forward **byte-identical** from the immutable `paper_track/v0_2a/` commit
(`5af18387395009ee35f6c5e809934562733184d6`) — verified via `git archive`
+ sha256 before inclusion, not copied from a mutable working tree. Only the
manuscript text itself changed between v0_2a and v0_3 (a full narrative/
prose restructure closing ten adversarial-review items A–J), and DS's own
re-review record confirms the numerical/tabular/figure core was retained
unchanged — independently re-verified here rather than merely trusted.

See `TECHNICAL_PROVENANCE_NOTE_v0_3.md` for the exact, itemized mechanical
normalization applied to the Doc export (markdown-escape unescaping and
table/heading restructuring only — no wording change), and
`PAPER_REVISION_CHANGELOG.md`'s final section for the v0_2a→v0_3 delta
summary quoting DS's own adversarial re-review verdict.
