# P5 v0_3 — Internal Adversarial Re-Review (DS)

Project: copeland-erdos-nets

Reviewed artifact: paper_post_R014_v0_3 (native Google Doc / uploaded review copy)

Prior gate: INTERNAL_ADVERSARIAL_REVIEW = MAJOR_MANUSCRIPT_REVISION_REQUIRED_BEFORE_DOCX

## DECISION

```text
INTERNAL_ADVERSARIAL_RE_REVIEW: PASS
SCIENTIFIC_CONTRADICTION: NONE
SCIENTIFIC_CORE: RETAIN
NUMERICAL_EVIDENCE: RETAIN
TABLES_2_6_VALUES: RETAIN
FIGURE_SCIENTIFIC_LINEAGE: RETAIN
NEW_EXPERIMENTS: NOT REQUIRED / NOT AUTHORIZED
R015/R016: NOT AUTHORIZED
DOCX_AUTHORING_GATE: AUTHORIZE_DOCX_REVIEW_MASTER
PRE_SUBMISSION_REVIEW: NOT READY
NEXT OWNER: Paper Agent for Word/DOCX review-master production; Architect only for explicit repository-side review-candidate synchronization.
```

## 1. Closure of prior adversarial blockers

A. External significance / contributions — CLOSED. The Introduction now gives four bounded contributions and frames the contribution as causal deconfounding of an initialization attribution, not as a universal initializer claim.

B. Constructor-scale origin — CLOSED. The manuscript states that the constructor condition is the native nn.Embedding constructor state in this implementation, with empirical RMS near 1 versus Xavier target near 0.0063 (~159x), and explicitly marks this as implementation-specific.

C. R012 factorial construction — CLOSED. The manuscript now defines the exact crossed construction from constructor (s0,u0) and one fresh-Xavier draw (s1,u1): S0D0=s0u0, S0D1=s0u1, S1D0=s1u0, S1D1=s1u1; no independent redraw is introduced for crossed cells.

D. Scale ladder — CLOSED. The manuscript states s_X=sqrt(2/(V+d_model)), r=s_X/RMS_constructor, the R013 ladder factors r/2, r, r^(2/3), r^(1/3), 1, and that R014 reuses the corresponding shape-parametric D_xavier/D_mid1/D_ctor levels.

E. Architecture/training envelope — CLOSED. Methods now describe the custom decoder-only Pre-LN Transformer, causal self-attention, GELU MLP, learned token/positional embeddings, final LayerNorm, direct token_emb+pos_emb input, no sqrt(d_model) multiplier, untied lm_head, no model dropout, AdamW lr=5e-4, wd=0.01, 15 epochs, and no scheduler/warmup/gradient clipping in the accepted runner.

F. Preprocessing / PPL comparability — CLOSED. Tokenization/chunking/drop_last semantics are stated and absolute PPL is explicitly bounded as protocol-specific rather than directly benchmark-comparable across pipelines.

G. Sequential test reuse — CLOSED. The manuscript now states that test is held out within each run but the same WikiText-2 test split is reused across R010-R014; fresh seeds are not independent dataset replications and intervals reflect seed variability conditional on the fixed corpus/test resource.

H. Statistical scope — CLOSED. Paired Student-t intervals are primary for seed-matched intervention contrasts; condition-mean intervals are defined as Student-t intervals over seed-level condition PPL values; primary/secondary/diagnostic contrast roles are distinguished; no p-values or retrospective multiplicity correction were introduced.

I. Causal wording — CLOSED FOR SCIENTIFIC GATE. Main Results, Discussion, Limitations and Conclusion now consistently restrict attribution to the tested factorial/intervention family and parity contracts, and R014 is described as reproduction at one larger tested capacity rather than broad capacity transfer.

J. Code/Data Availability — CLOSED FOR DOCX GATE. Repository, commit/branch/path technical snapshot, dataset configuration and tokenizer are stated.

Related Work — CLOSED FOR DOCX GATE. FixNorm/ScaleNorm and Conditioned Embedded Tokens are now explicitly distinguished from the present initialization-only RMS intervention, and no novelty claim is inferred from search absence.

## 2. Remaining nonblocking publication-edit items for DOCX stage

These are not scientific blockers and do not reopen P5:

- In the Abstract, prefer replacing the remaining phrase "localizes the dominant reconstructed difference to token_emb.weight" with the already-used bounded formulation "within the tested embedding-by-attention factorial, the dominant reconstructed contrast follows the token-embedding condition." This is wording normalization, not a claim change.

- Code/Data Availability currently calls commit 5af183... the technical snapshot "for this manuscript revision" although that commit is the accepted v0_2a technical snapshot and v0_3 is a later manuscript-only Google-Doc repair. Before external review, either rephrase it as the accepted technical evidence snapshot underlying the manuscript or, after Architect creates a review-candidate technical mirror, replace it with the new exact commit.

- Word conversion must typeset equations unambiguously. In particular ensure Eq. (3)/(5) render as base -> (s0,u0) and fresh -> (s1,u1); Eq. (9) renders r^(2/3), r^(1/3); Eq. (10) renders Delta_i=PPL_A,i-PPL_B,i; and subscripts/superscripts are not corrupted by Google-Docs export.

- The review master may move Table 7 and part/all of Fig. 7 diagnostics to an appendix/supplement to reduce display density, provided no scientific values or evidence lineage change.

- A final 2026 literature refresh remains mandatory before PRE_SUBMISSION_REVIEW, but it is not a blocker for DOCX authoring or fresh external/journal-style review.

## 3. DOCX transition authorization

```text
DOCX_AUTHORING_GATE = AUTHORIZE_DOCX_REVIEW_MASTER
```

Paper Agent is now authorized to create and polish a Word/DOCX review master from the accepted v0_3 scientific content. Recommended external-review artifact name: paper_post_R014_review_candidate_v1.docx.

The DOCX is a review surface, not the scientific source of truth. Accepted R010-R014 canonical evidence, generated table/figure values, evidence bindings, and repository manifests remain authoritative. Any material scientific change discovered during Word polishing must return to DS; pure typography/layout/caption/reference-style changes do not.

## 4. Architect synchronization rule

Architect is not authorized to change prose. A separate DS handoff may request technical synchronization of the accepted v0_3/review-candidate state into git. That task may materialize raw Markdown, update manifests/hashes, and produce a review-candidate commit, but may not alter claims, numbers, figures, or scientific classifications.

## 5. Next gates

Paper Agent: create DOCX review master and return it for DS publication-format intake.

Architect: if separately tasked, create review-candidate technical repository snapshot and return exact commit/branch/path.

DS after DOCX intake: verify scientific identity of DOCX, equation/table/figure/reference integrity, and review-facing formatting. Then authorize fresh external/journal-style review if clean.

PRE_SUBMISSION_REVIEW remains NOT READY until external-review feedback is processed, final literature refresh is completed, and any resulting blocking issues are closed.
