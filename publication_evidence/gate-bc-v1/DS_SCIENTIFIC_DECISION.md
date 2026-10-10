# Gate B+C — canonical scientific review and manuscript revision authority

To: Architect and Paper Agent (via Owner)
From: DS Colleague
Project: copeland-erdos-nets
Date: 2026-10-10
Type: SCIENTIFIC RESULT REVIEW / BOUNDED MANUSCRIPT REVISION RELEASE

TECHNICAL_INTAKE: PASS
GATE_B: INTERACTION_RESOLVED
GATE_C: C002_WORSE
PAPER_REVISION: AUTHORIZED
GATE_D: DEFERRED_BY_BOUNDED_SCOPE
GATE_E: NOT_REQUIRED_FOR_CURRENT_CLAIMS
NEW_CANONICAL_COMPUTE: NOT_AUTHORIZED_BY_THIS_DOCUMENT
JOURNAL_SUBMISSION_READINESS: PENDING_REVISED_MANUSCRIPT_AND_EVIDENCE_RELEASE

## 1. Decision

The required Gate B+C canonical block is accepted. Resume manuscript revision using the completed five-seed evidence. This document supersedes manuscript FROZEN and GateD HOLD_PENDING_BC for a revision explicitly limited to the investigated implementation, WikiText-2 and fixed training regime. It does not claim cross-corpus or training-regime robustness and does not authorize new training.

All reviewer remarks are NOT yet closed in the manuscript. Experimental separation and the RMS=0.02 control are complete; their interpretation, related-work addition, versioned public B+C evidence and revised claim/response matrices remain to be delivered. Gate D is a scope decision, not a completed robustness experiment.

## 2. Accepted evidence and audit scope

Primary package: `discussion/architect_reports/GATE_BC_canonical_execution/`.
Reviewed entry files: REPORT_FOR_DS.md, FINAL_TECHNICAL_SUMMARY.json, FINAL_EVIDENCE_SHA256SUMS, five TECHNICAL_AUDIT.json files, per-seed receipts and four excluded-attempt ledgers. Binding scientific rules: GATE_BC_DESIGN_DS_REVIEW.md, especially classification precedence; prior reviewer disposition: EXTERNAL_REVIEW_V2_DS_REVIEW.md. Gate A publication release and paper-readiness decisions were also consulted.

Accepted attempts:

| Seed | Attempt | Source SHA |
|---|---|---|
| 67 | canonical67-attempt2-routing-v1 | 7fc1e93604129eb17f6d9966809f24460776b0d0 |
| 68 | canonical68-attempt4-release-v1 | f10d70c531d84f219d55ea558378c9100723ec52 |
| 69 | canonical69-attempt1-release-v1 | f10d70c531d84f219d55ea558378c9100723ec52 |
| 70 | canonical70-attempt1-release-v1 | f10d70c531d84f219d55ea558378c9100723ec52 |
| 71 | canonical71-attempt1-release-v1 | f10d70c531d84f219d55ea558378c9100723ec52 |

Scientific config authority SHA256: `45dea06a3afd8eb523fdb875cde0b9759a9fcfcc2c4707cbc6f816a9ad4ea8aa`. Source difference is disclosed and accepted under the historical transport-only exact-source reviews; it must remain visible in public provenance.

DS independently checked the raw hashes of all 72 inventory entries: PASS. Independently recalculated the frozen estimands from all 25 accepted metrics rows. Checked 375 epoch rows (15 epochs per condition/seed), earliest minimum-validation checkpoint selection, one test evaluation per endpoint, exp(test_loss)=test_PPL, selected/current checkpoint size/hash metadata agreement with technical audits, identical within-seed batch-order hashes across conditions, all 42 construction/parity gates per seed, runtime and telemetry-equivalence receipts. The five resolved configs and five data manifests are respectively byte-identical. All 8,461 recorded durability readbacks report PASS. No partial failed-attempt metrics entered the calculations.

Audit boundary: DS did not redownload the five approximately 934 MB archives or the 50 checkpoint binaries and did not rerun training/evaluation or the engineering test suites. Their byte-level verification is accepted from the Architect's sealed-archive/readback and full technical audits; receipt hashes and endpoint metadata were cross-checked. Independent SA readback of seed71's final receipt is reported by the Architect, not a new DS SA operation. This is an evidence audit and independent statistical calculation, not independent model reproduction. Historical R011 exclusion remains disclosed; no accepted hash was changed.

Seed67 attempt1 and seed68 attempts1–3 remain entirely excluded and immutable. Failure causes are not inferred from scientific endpoints. No checkpoint, state or RNG splice is accepted.

## 3. Frozen construction and analysis

Let s_small=sqrt(2/(50257+128))≈0.00630035, s_large=RMS(W0)≈1 and u0=W0/s_large. The actual fixed-alpha intervention is token embeddings only, before positional addition: h=alpha*token_emb(ids)+pos_emb. No ongoing renormalization or positional/residual scaling.

| Condition | Stored weights | Fixed alpha | Initial forward level |
|---|---|---|---|
| B00 | s_small*u0 | 1 | small |
| B01 | exactly B00 bytes | s_large/s_small | large |
| B10 | exact W0 | s_small/s_large | small |
| B11 | exact W0 | 1 | large |
| C_002 | 0.02*u0 | 1 | 0.02 |

Analysis unit is the paired seed, n=5. Confidence intervals are the frozen paired Student-t intervals: mean ± 2.7764451051977987*sample_SD/sqrt(5), df=4. No historical pooling, post-hoc seed exclusions, equivalence margin or multiplicity-adjusted discovery claim is introduced. CI describes seed variability on the fixed corpus, not new-data uncertainty. Lower test PPL is better.

## 4. Results

| Condition | Mean test PPL | Sample SD |
|---|---:|---:|
| B00 | 239.634456 | 1.234881 |
| B01 | 265.136707 | 0.711087 |
| B10 | 307.850747 | 6.006105 |
| B11 | 313.391318 | 2.633425 |
| C_002 | 242.357056 | 2.046916 |

| Frozen contrast | Mean PPL difference | Sample SD | Paired 95% CI | Signs |
|---|---:|---:|---|---|
| Delta_P=(B10+B11-B00-B01)/2 | 58.235451 | 3.838773 | [53.468986, 63.001917] | 5/5 positive |
| Delta_F=(B01+B11-B00-B10)/2 | 15.521411 | 2.826719 | [12.011576, 19.031246] | 5/5 positive |
| I=B11-B10-B01+B00 | -19.961680 | 5.882122 | [-27.265299, -12.658061] | 5/5 negative |
| Delta_002=C_002-B00 | 2.722600 | 1.740615 | [0.561341, 4.883859] | 4/5 positive |

Interaction excludes zero and takes precedence: **INTERACTION_RESOLVED**. Both marginal large-level changes worsen PPL, but neither factor receives a DOMINANT label. Simple effects are mandatory:

| Simple effect | Mean | Sample SD | Paired 95% CI |
|---|---:|---:|---|
| Parameterization at small forward: B10-B00 | 68.216291 | 6.373470 | [60.302583, 76.129999] |
| Parameterization at large forward: B11-B01 | 48.254612 | 2.480104 | [45.175156, 51.334067] |
| Forward at small parameterization: B01-B00 | 25.502251 | 1.708900 | [23.380371, 27.624130] |
| Forward at large parameterization: B11-B10 | 5.540571 | 5.509993 | [-1.300989, 12.382131] |

The last simple effect is unresolved, not absent or equivalent. Negative interaction means the PPL penalties are sub-additive on the PPL scale in this design; it does not establish a general mechanism or a beneficial large scale.

C_002 has a positive lower CI bound: **C002_WORSE**. The penalty is modest and one seed has the opposite sign. This is an RMS=0.02 fixed-direction control in this implementation, not a replication of GPT-2 architecture/initialization or evidence that 0.02 is universally poor. No optimal scale is identified.

Per-seed primary contrasts, in seed order 67–71:

| Seed | Delta_P | Delta_F | I | Delta_002 |
|---|---:|---:|---:|---:|
| 67 | 52.903508 | 19.314765 | -9.589222 | 2.328906 |
| 68 | 62.165294 | 17.675247 | -21.582071 | 3.617936 |
| 69 | 61.489429 | 12.704350 | -23.992854 | 4.387209 |
| 70 | 56.133009 | 13.734833 | -22.950655 | 3.378024 |
| 71 | 58.486018 | 14.177860 | -21.693598 | -0.099073 |

Interpretation: the original token-embedding intervention changes both stored parameterization/optimizer coupling and initial forward magnitude. The matched-forward contrast establishes a parameterization-dependent training outcome under the fixed optimizer; fixed alpha also enters the gradient chain rule. A pure optimizer mechanism is not isolated. Initial matching does not imply forward magnitudes remain matched throughout training. R010–R014 numerical evidence remains intact; Gate B sharpens its attribution into a compound, condition-dependent interpretation.

## 5. Reviewer closure matrix

| Reviewer issue / planned gate | Evidence status | Remaining manuscript/publication work |
|---|---|---|
| P1-1, Gate A: one reproducible public evidence state | R010–R014 release previously PASS | Add immutable, versioned B+C evidence release; retain historical tag unchanged; cite precise new revision |
| P1-2, Gate B: stored RMS versus forward magnitude | Experimental requirement complete; INTERACTION_RESOLVED | Add factorial methods/results and all simple effects; revise title/abstract/discussion/conclusion to compound attribution |
| P1-3, Gate C: RMS=0.02 and practical context | RMS=0.02 requirement complete; C002_WORSE | Add result and bounded practical interpretation; add Takase et al. / Spike No More with accurate distinction |
| P1-4, Gate D: one corpus/regime and train-validation divergence | Not experimentally closed | Explicit fixed-regime/corpus limitation; defer robustness work for the current narrowly scoped paper |
| Gate E: additional runs for null/equivalence claims | Not required | Retain unresolved wording; no evidence-of-absence or equivalence claim |

Gate D may be reopened only for a separately agreed broader claim or journal strategy. The present decision follows the prior scope option; it does not guarantee that a future reviewer will accept that scope.

The accessible current editorial master (Drive ID 1W0twUWMseSTn-BkdXhohq3JSs9fFKgKfeQYob5TBi2Y) contains neither B+C factorial results nor Takase/Spike No More. Its existing title's “dominant role” must be reassessed in the new version. A previous Gate A tag cannot stand as the sole provenance for the new results.

## 6. Authorized next work — no new compute

Architect: prepare a separate versioned public B+C evidence candidate with canonical metrics/epoch rows, configs, scientific source provenance, construction/parity and durability evidence, excluded-attempt dispositions, this scientific decision and reproducible DS calculations. Use a reviewed public allowlist; exclude private credentials and private control artifacts. Preserve `publication-evidence-r010-r014-v1` and all historical targets. Deterministic tables/figures from accepted CSVs are permitted. Return exact candidate commit/tag and inventory for final DS evidence-release review. No training, checkpoint re-evaluation or scientific endpoint replacement.

Paper Agent: prepare a new derivative manuscript, preserving the latest Owner-approved text/edits and all historical R010–R014 numbers. Do not overwrite existing versions or treat an unapproved draft as Owner approval. Integrate the five-condition protocol, accepted attempts, source differences and exclusions; add the above results and intervals, simple effects and interaction precedence; update title/abstract/discussion/conclusion and limitations consistently. New B+C results must remain separate from historical series, not pooled.

Add Takase, Kiyono, Kobayashi and Suzuki, *Spike No More: Stabilizing the Pre-training of Large Language Models*, COLM 2025; primary source https://arxiv.org/abs/2312.16903v4 (checked 2026-10-10). Its stabilization framing is small sub-layers and large shortcut; the present endpoint/fixed-direction initialization intervention does not reproduce that full method or its LLM setting. Avoid claiming contradiction from different architectures/protocols. Citation numbering must be verified after insertion.

Deliver revised manuscript, change log, updated claim-evidence and reviewer-response matrices, canonical table/figure references and precise publication provenance. Mark scope limitations as acknowledged rather than empirically eliminated. Do not incorporate CE-P1 incomplete/NO_SIGNAL archives into canonical results. No full English rewrite is authorized merely by this scientific release.

NEXT_GATE: REVISED MANUSCRIPT + REVIEWER RESPONSE + VERSIONED B+C PUBLIC EVIDENCE INTAKE.
No further GPU or broad infrastructure test campaign is required to start or complete this bounded revision.
