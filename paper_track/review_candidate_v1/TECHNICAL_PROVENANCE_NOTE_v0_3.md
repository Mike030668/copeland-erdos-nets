# Technical Provenance Note — review-candidate v1 (v0_3 materialization)

This note is a factual record of the mechanical steps used to materialize
`paper_post_R014_v0_3` from its native Google Doc into this package. It
contains no scientific claims and changes no scientific content; it exists
so a reviewer can audit exactly what transformation, if any, the source
text went through.

## Source

- Google Doc: `paper_post_R014_v0_3` (file ID `1RMM6rLDt7zUvd7dodP2tEnwXAlJ1rXMUFD-EyWfR0Ng`), owner `puzitski.mikhail@gmail.com`, created 2026-09-14T06:10:53.890Z.
- Adversarial re-review record: `P5_v0_3_INTERNAL_ADVERSARIAL_REREVIEW_DS` (file ID `1xDLs_jWfaRcfCp2AX_7RRrUYSed_o_XXsnx-xT-3WqQ`), same owner.
- Both retrieved via read-only Drive content export; neither Doc was modified by this process.

## Mechanical normalization applied to the manuscript export

The Google Drive export of a native Doc into Markdown-like text introduces
two purely mechanical artifacts that do not reflect anything the author
typed, and were normalized as follows — **no word, number, citation, or
claim was added, removed, or reordered**:

1. **Backslash-escaped underscores.** The export escapes every literal `_`
   character as `\_` (e.g. `token\_emb.weight`, `d\_model`). This is the
   export mechanism guarding against accidental Markdown emphasis syntax,
   not an authored escape sequence. All 135 occurrences were unescaped
   back to plain `_` to match the raw-Markdown convention already used in
   `paper_post_R014_v0_2a.md` (which contains identical terms like
   `token_emb.weight` without escaping).

2. **Malformed table structure.** Doc tables have no native "header row"
   concept; the export emitted a blank placeholder header row, a
   `:-:`-style alignment separator, and then the actual (bold-formatted)
   header labels as if they were an ordinary body row, e.g.:
   ```
   |  |  |  |  |  |
   | :-: | :-: | :-: | :-: | :-: |
   | **Experiment** | **Scientific question** | ... |
   | R010 | ... |
   ```
   This renders as a broken table in standard Markdown. It was restructured
   into the conventional form (header-row text moved to the header
   position, single `|---|---|` separator, no bold markup on header
   cells) already used throughout `paper_post_R014_v0_2a.md`:
   ```
   | Experiment | Scientific question | ... |
   |---|---|
   | R010 | ... |
   ```
   Every cell's text content, every row, and every column is unchanged —
   only the row/marker arrangement was corrected to produce a table that
   Markdown renderers display correctly. The same treatment was applied to
   heading lines that the export wrapped in bold markers (e.g. `## **1.
   Introduction**` → `## 1. Introduction`), matching v0_2a's heading style,
   which also does not bold section titles.

No other transformation was applied. Every number, statistic, confidence
interval, citation, table cell value, and sentence in
`paper_post_R014_v0_3.md` is a verbatim transcription of the Doc's
exported text after only the two mechanical fixes above.

## Verification performed before assembling this package

- Confirmed via the DS adversarial re-review record that
  `TABLES_2_6_VALUES: RETAIN`, `FIGURE_SCIENTIFIC_LINEAGE: RETAIN`, and
  `NUMERICAL_EVIDENCE: RETAIN` — i.e. DS's own gate record states the
  numerical/figure core is unchanged from v0_2a.
- Independently verified this by comparing every number, confidence
  interval, and Table 7 cell embedded in the v0_3 manuscript text against
  the already-hash-verified v0_2a package (`paper_track/v0_2a/`, commit
  `5af18387395009ee35f6c5e809934562733184d6`) — exact match, including the
  R014 primary contrast (`-34.7442 [-37.5676, -31.9208]`) and all R010–R013
  values.
- All carried-forward files (10 figures, 6 base tables, Table 7, 2
  generator scripts, 20 canonical-evidence CSVs, evidence-binding and
  figure/table registry documents) were copied directly from the immutable
  `paper-track/v0_2a-materialization` git commit (`git archive`, not the
  working tree) and confirmed byte-identical (sha256) to that commit's
  blobs before inclusion here — see `ARTIFACTS.md` for the full hash list.
- `BIBLIOGRAPHY_v0_3.md` was extracted directly from this package's own
  `paper_post_R014_v0_3.md` References section (copy of already-transcribed
  text, not independently retyped), so it cannot introduce a bibliography/
  manuscript mismatch.

## What was deliberately left unresolved

Per DS's own re-review note (Section 2, "Remaining nonblocking
publication-edit items for DOCX stage"), the manuscript's Code/Data
Availability sentence still names the v0_2a commit/branch/path as "the
accepted technical snapshot for this manuscript revision," even though
this review-candidate v1 package now exists at a different commit/branch/
path. DS's note explicitly frames updating that sentence as a decision for
the DOCX authoring stage, not an automatic action for this technical
synchronization step — the sentence was left exactly as accepted, per the
Architect's "do not edit scientific prose independently" instruction.
