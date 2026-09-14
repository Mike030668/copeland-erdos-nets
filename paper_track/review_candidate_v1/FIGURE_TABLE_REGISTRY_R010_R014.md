\# FIGURE\_TABLE\_REGISTRY\_R010\_R014.md

&nbsp;

Project: copeland-erdos-nets / paper\_track

Purpose: figure/table registry for the new post-R014 manuscript. No hand-transcribed result tables. Every result display must be generated from canonical artifacts named below.

&nbsp;

\#\# Registry rules

&nbsp;

\- Primary numeric tables are generated from canonical CSV files only.

\- Legacy R008/R009 validation PPL may appear only in a historical/confound table and must be labeled non-comparable with R010-R014 held-out test PPL.

\- R010-R014 primary endpoint: held-out test PPL at validation-selected checkpoint.

\- Validation PPL is used for checkpoint selection and training-dynamics diagnostics, not as the main corrected performance endpoint.

\- Figures must record script/generation rule, input artifact, and claim ID.

&nbsp;

\#\# Proposed main-text tables

&nbsp;

| table\_id | proposed title | canonical artifact | metric / estimand | claim\_id | generation rule | guardrail |

|---|---|---|---|---|---|---|

| Table 1 | Compact R010-R014 experiment-design map | DS reviews R010-R014; accepted package manifests; resolved\_config.json files | experiment purpose, factor, paired control, endpoint, seeds, evidence package | C1-C5 | Manual design map allowed; numerical effect estimates stay in Tables 2-6 | Main-text orientation table only; source hierarchy stays in SOURCE\_OF\_TRUTH\_MAP\_R010\_R014.md. |

| Table 2 | R010 corrected attention-only reconstruction | R010\_paired\_attention\_init\_confirmation/metrics/per\_seed.csv; metrics/summary.csv; metrics/paired\_differences.csv | test\_PPL(method) and paired delta vs xavier\_g1.0 | C1 | Programmatically aggregate per\_seed.csv; bind paired effects to metrics/paired\_differences.csv; include n=5 and 95% CI | No validation-PPL comparison with legacy R008/R009. CE-LCG may appear as a corrected R010 control, not a causal/superiority claim. |

| Table 3 | R011 embedding × attention factorial | R011\_embedding\_init\_factorial\_ablation/attempt\_seed\_atomic\_v2/per\_seed.csv; canonical factorial\_contrasts.csv | A0B0/A0B1/A1B0/A1B1 held-out test PPL; embedding and attention contrasts | C2 | Aggregate cell means from per\_seed.csv; bind contrasts to canonical factorial\_contrasts.csv. Recompute only as verification, not as manuscript source. | Do not claim embedding reinit is unique historical cause; preserve accepted R011 provenance limitation. |

| Table 4 | R012 scale-vs-redraw decomposition | R012\_embedding\_scale\_redraw\_factorial/per\_seed.csv; factorial\_summary.csv; factorial\_contrasts.csv | scale contrasts, redraw contrasts, interaction | C3 | Generate 2×2 factorial table from canonical CSV; include CI and direction consistency | Do not cite spectral placeholders; spectral diagnostics were not collected. |

| Table 5 | R013 fresh-seed dose response | R013\_embedding\_scale\_response\_and\_dynamics/per\_seed.csv; dose\_summary.csv; manifests/paired\_contrasts.csv; manifests/paired\_contrasts\_summary.csv | dose means, adjacent paired contrasts, D\_xavier-D\_ctor | C4 | Generate dose-level means from per\_seed.csv; bind paired contrasts to manifests/paired\_contrasts.csv and manifests/paired\_contrasts\_summary.csv; include n=5 and CI | Low pair is unresolved, not statistically tied. |

| Table 6 | R014 larger-capacity transfer | R014\_corrected\_larger\_model\_embedding\_scale\_confirmation\_v2/per\_seed.csv; paired\_contrasts.csv; paired\_contrasts\_summary.csv | D\_xavier, D\_mid1, D\_ctor held-out test PPL; primary and secondary paired contrasts | C5 | Generate from per\_seed.csv; bind paired effects to canonical paired\_contrasts.csv and paired\_contrasts\_summary.csv; include all seed deltas and 95% CI | Scope is tested 4-layer/d\_model=256 family only. |

&nbsp;

| Table 7 | Reproducibility/provenance envelope | R010-R014 ARTIFACTS.md, commit.txt, resolved\_config.json, parity and checkpoint manifests | package status, runtime, source SHA where accepted, checkpoint persistence | C1-C5 | Extract path/status/hash counts from accepted package manifests | Do not rewrite retained provenance gaps as closed if DS marked them limitations. |

&nbsp;

\#\# Proposed main-text figures

&nbsp;

| figure\_id | proposed title | canonical artifact | metric / estimand | claim\_id | generation rule | guardrail |

|---|---|---|---|---|---|---|

| Fig. 1 | Causal reconstruction timeline | DS reviews R010-R014; source map | qualitative causal sequence | H0-C5 | Draw from source map: historical observation → R010 → R011 → R012 → R013 → R014 | No new numerical claim. |

| Fig. 2 | R010: attention-only corrected result | R010 plots or generated from metrics/per\_seed.csv and summary.csv | method mean test PPL / paired deltas vs xavier\_g1.0 | C1 | Prefer generated plot from per\_seed.csv; show CI from summary.csv | Emphasize failure of attention-only attribution. |

| Fig. 3 | R011: embedding effect by attention condition | R011 plots/fig01\_factorial\_test\_ppl.png and fig02\_embedding\_effect\_by\_seed.png | factorial cell means and paired embedding deltas | C2 | Use accepted plot if sourced from canonical rows; otherwise regenerate from per\_seed.csv | Attention main effect remains unresolved. |

| Fig. 4 | R012: scale dominates redraw | R012 fig01\_factorial\_test\_ppl.png, fig02\_scale\_effect\_by\_seed.png, fig03\_redraw\_effect\_by\_seed.png | scale vs redraw contrasts | C3 | Use accepted plots or regenerate from per\_seed.csv/factorial\_contrasts.csv | Do not infer mechanism beyond initial RMS scale. |

| Fig. 5 | R013: dose response and low-scale plateau | R013 fig01\_dose\_response\_test\_ppl.png; fig06\_primary\_contrast\_xavier\_minus\_ctor.png; fig07\_all\_contrasts\_ci.png | dose means; adjacent and constructor contrasts | C4 | Use R013 v3 plots documented as programmatically sourced from per\_seed.csv | Phrase D\_below-D\_xavier as unresolved at current precision. |

| Fig. 6 | R014: larger-capacity transfer | R014 fig01\_dose\_response\_test\_ppl.png; fig05\_primary\_contrast\_xavier\_minus\_ctor.png; fig06\_all\_contrasts\_ci.png | D\_xavier/D\_mid1/D\_ctor test PPL; paired contrasts | C5 | Use R014 v2 plots; or regenerate from per\_seed.csv if journal requires unified styling | Does not resolve below-Xavier region. |

| Fig. 7 | Training-dynamics diagnostics | R013 fig02\_best\_epoch\_by\_dose.png; fig03\_final\_vs\_best\_val\_ppl.png; fig04\_embedding\_rms\_trajectory.png; fig08\_embedding\_grad\_l2\_trajectory.png; R014 fig02/fig03/fig04 | best epoch, final/best validation PPL, RMS trajectory, gradient L2 | C6 | Use diagnostics only; optionally split into two figures if journal page budget permits | Diagnostic association only; no mechanism claim. |

&nbsp;

\#\# Appendix tables / figures

&nbsp;

Source hierarchy and Expert Review Closure Matrix are P0-P4 governance/closure artifacts. They may be cited in planning notes, but they are not main scientific result tables.

&nbsp;

| appendix\_id | proposed item | canonical artifact | purpose |

|---|---|---|---|

| App. Table A1 | Full R010 per-seed rows | R010 metrics/per\_seed.csv | auditability of corrected attention-only reconstruction. |

| App. Table A2 | Full R011 per-seed factorial rows | R011 attempt\_seed\_atomic\_v2/per\_seed.csv | auditability of embedding × attention factorial. |

| App. Table A3 | Full R012 per-seed rows and contrasts | R012 per\_seed.csv plus contrast artifacts | auditability of scale-vs-redraw decomposition. |

| App. Table A4 | Full R013 dose×seed rows | R013 per\_seed.csv | auditability of dose ladder. |

| App. Table A5 | Full R014 seed×dose rows | R014 per\_seed.csv | auditability of larger-capacity transfer. |

| App. Table A6 | Runtime and source-provenance limitations | R011-R014 environment/provenance addenda, source manifests | preserve DS-documented limitations. |

| App. Fig. A1 | Legacy old manuscript figures/tables index | paper\_2\_review\_v1.docx | historical reference only. |

&nbsp;

\#\# Artifact paths already verified during P0

&nbsp;

\- R010 package folder: discussion/architect\_reports/R010\_paired\_attention\_init\_confirmation/

\- R010 canonical summaries: metrics/summary.csv; metrics/paired\_differences.csv.

\- R011 package folder: discussion/architect\_reports/R011\_embedding\_init\_factorial\_ablation/attempt\_seed\_atomic\_v2/

\- R011 canonical rows and contrasts: per\_seed.csv; canonical factorial\_contrasts.csv. Any recomputation is verification only.

\- R012 package folder: discussion/architect\_reports/R012\_embedding\_scale\_redraw\_factorial/

\- R012 canonical rows: per\_seed.csv.

\- R013 package folder: discussion/architect\_reports/R013\_embedding\_scale\_response\_and\_dynamics/

\- R013 canonical rows and contrasts: per\_seed.csv; manifests/paired\_contrasts.csv; manifests/paired\_contrasts\_summary.csv.

\- R014 package folder: discussion/architect\_reports/R014\_corrected\_larger\_model\_embedding\_scale\_confirmation\_v2/

\- R014 canonical rows and contrasts: per\_seed.csv; paired\_contrasts.csv; paired\_contrasts\_summary.csv.

&nbsp;

\#\# Generation notes

&nbsp;

For manuscript tables, create a small reproducible script that reads canonical CSVs from the accepted packages and emits Markdown/LaTeX tables. Hand-entered numbers should be prohibited except in prose snippets directly copied from DS review and verified against canonical CSV.

&nbsp;

For visual uniformity, figures may be regenerated from canonical CSVs if journal style requires; the regenerated figure script must not change any scientific value. If any regenerated value differs from accepted artifacts, stop and report the contradiction to DS.

&nbsp;

&nbsp;

\#\# P5 v0\_2 clerical delivery binding — 2026-09-13

&nbsp;

Status: bounded manuscript repair delivered for \`P5 v0\_2 REVALIDATION\`. This section is clerical only: no accepted numerical result, DS scientific classification, claim boundary, or experiment status is changed. \`paper\_post\_R014\_v0\_1.md\` remains immutable.

&nbsp;

\#\#\# Manuscript and audit package

&nbsp;

\- \`paper\_post\_R014\_v0\_2.md\`: Drive ID \`11H\_sf9LcOaXq3JX9dP7NxyCgpLONkjWy\`; SHA-256 \`e71edbe7e0c4a3d28af0d32a266234a8cad8e0c27b618d80b5e15525f6ca5c57\`.

\- \`PAPER\_REVISION\_CHANGELOG.md\`: Drive ID \`1x8ZNAVZst3iUJxmE6DEZY1GIfGpBBed1\`; updated in place for v0\_2 while preserving the existing Drive identity.

\- Support folder: Drive ID \`1a3zsYKaHdeIoOlfihXOx8kNK2Pu6T20j\`.

\- \`MANUSCRIPT\_EVIDENCE\_BINDINGS\_R010\_R014.md\`: Drive ID \`11nvoUP2KP\_EQiDxcVv1O8q2JvugtZu\_o\`.

\- \`FIGURE\_BINDING\_MANIFEST.csv\`: Drive ID \`1uDnfnq4nP53ydr22Z\_VNaCqOP-eHn80U\`.

\- \`BIBLIOGRAPHY\_v0\_2.md\`: Drive ID \`1MQYAE4Idtln7F9ZXSRuW8faQemSeA8J2\`.

\- \`V0\_2\_PACKAGE\_MANIFEST.json\`: Drive ID \`1RvTJ-9cPblNBUkapkpXgDQbxHQDy1ZoS\`; manifest is generated last and self-excludes to avoid a circular self-hash.

\- \`AUDIT\_REPORT.json\`: Drive ID \`10NzJSPgzdCUUsEdKByIgLJAg25w0mx8-\`; status \`PASS\`, 21 canonical checks, fail-closed on mismatch.

\- \`P5\_v0\_2\_audit\_package.zip\`: Drive ID \`1MHGBgXHqtpEb5\_XeqeyhXDf7aI1XX8lK\`; SHA-256 \`7eecce1ac95f9a4cb71619ea372728e9ea0277230d0fcba8b45059db6696242c\`. The archive contains canonical R010–R014 snapshots, generated tables, audit report, scripts, figure files, manuscript, bibliography, changelog, and internal evidence bindings.

\- Table generator: Drive ID \`1t-TPb0Xu0xBvcrSHHBJ6ymAGLrB6yvoP\`; figure generator: Drive ID \`1BH50AF6i3IrHnETVABCOCgUhXkgUGQav\`.

&nbsp;

\#\#\# Materialized table bindings

&nbsp;

\- Table 1: \`table\_1\_experiment\_map.md\`, Drive ID \`10FGprgoqy6mgP0lo0EjsnxPs3c7pzcBI\`.

\- Table 2: \`table\_2\_R010.md\`, Drive ID \`1SK2RrL26tadn-aR6OBry0aNTB09BbsIu\`.

\- Table 3: \`table\_3\_R011.md\`, Drive ID \`1lgAo0e3mfgVTZZX9QzAaMniQw4XLj9uV\`.

\- Table 4: \`table\_4\_R012.md\`, Drive ID \`1JDOGQ5IQvTbahXwtHWfGjydlH8oOvIIE\`.

\- Table 5: \`table\_5\_R013\_doses.md\`, Drive ID \`1lFo5EBuJvufH8RCJBd6CtxbKbK6o2652\`.

\- Table 6: \`table\_6\_R014\_doses.md\`, Drive ID \`1337-7S63N83EeJoyyBh1Q8uwRBVYmE2V\`.

\- Table 7: \`table\_7\_reproducibility.md\`, Drive ID \`1s4wVZ5uG1Q4dHC3bsZr3KyvD\_2or7YYa\`.

&nbsp;

Tables 2–6 are generated from the canonical CSV snapshots and verified against accepted seed-level/summary artifacts. The generator reproduces the accepted rounded Student-t critical value \`t(0.975,4)=2.776\`; it does not introduce a new estimand.

&nbsp;

\#\#\# Materialized figure bindings

&nbsp;

Figure folder: Drive ID \`1UF1ztCMtWfg4TP6CLk2flaDV3om7JjWn\`.

&nbsp;

\- Fig. 1 \`fig01\_causal\_reconstruction.png\`: Drive ID \`16V\_SsVefpsKPdBFaqjzg3w1HJOU8Sm5p\`; manuscript-created qualitative timeline, no new numerical claim.

\- Fig. 2 \`fig02\_R010\_test\_ppl.png\`: Drive ID \`1iEdikTpWbxP6YUBVGTwSE41tcLwdo7HK\`; regenerated deterministically from canonical R010 per-seed rows.

\- Fig. 3 \`fig03\_R011\_factorial\_test\_ppl.png\`: Drive ID \`1YEJIqi\_1ldyw5tCCi-Hwbou7rmnehKXl\`; accepted R011 factorial plot lineage; R011 provenance limitation remains binding.

\- Fig. 4 \`fig04\_R012\_factorial\_test\_ppl.png\`: Drive ID \`1RKsrOciJbxy9yZn4QYjhEMJPE6vq6443\`; accepted R012 factorial plot lineage.

\- Fig. 5 \`fig05\_R013\_dose\_response\_test\_ppl.png\`: Drive ID \`1KHKQ-e7EnJ9PmLAiTazm2CojpbaTTSz8\`; accepted R013 dose-response plot lineage.

\- Fig. 6 \`fig06\_R014\_dose\_response\_test\_ppl.png\`: Drive ID \`1RYXnBfoWoBOFs3kQ7uZG1cPA3CPkQz2p\`; accepted R014 dose-response plot lineage.

\- Fig. 7a \`fig07a\_R013\_embedding\_rms\_trajectory.png\`: Drive ID \`1-ktVmz1mrC\_HMsQ-Tf-nHmkMCXBwxZUC\`; R013 embedding RMS diagnostic only.

\- Fig. 7b \`fig07b\_R013\_embedding\_grad\_l2\_trajectory.png\`: Drive ID \`11Y\_r4kaWzfItznwilR2pnw7zF9aqZJgw\`; R013 embedding gradient-L2 diagnostic only.

\- Fig. 7c \`fig07c\_R014\_best\_epoch\_by\_dose.png\`: Drive ID \`1ptaPowcW8b5CxpQEhTl\_mNOJb86WDyuV\`; R014 checkpoint-timing diagnostic only.

\- Fig. 7d \`fig07d\_R014\_final\_vs\_best\_val\_ppl.png\`: Drive ID \`1IAKv051fuZGKIb0H\_e3vPfhQbb0Qhsv6\`; R014 validation-degradation diagnostic only.

&nbsp;

R014 is not assigned R013 embedding RMS/gradient telemetry. Fig. 7 remains diagnostic and carries no mechanism classification.

&nbsp;

\#\#\# Delivery guardrails

&nbsp;

\- No new experiment was run.

\- R015/R016 were not opened.

\- No accepted numerical result was changed.

\- No internal scientific classification was changed.

\- \`paper\_post\_R014\_v0\_1.md\` was not modified.

\- Frozen prior manuscripts and Expert Review artifacts were not modified.

\- \`PRE\_SUBMISSION\_REVIEW\` is not declared ready by this delivery; the next gate is DS revalidation of \`paper\_post\_R014\_v0\_2.md\`.

&nbsp;

&nbsp;

\#\# P5 v0\_2a superseding exact-binding note — 2026-09-14

&nbsp;

This note supersedes only the path strings below. Historical registry rows remain unchanged. No scientific value, estimand, claim, classification, or experiment scope is modified.

&nbsp;

\- R011 Table 3 exact retained bindings:

  \- \`attempt\_seed\_atomic\_v2/manifests/factorial\_contrasts.csv\`

  \- \`attempt\_seed\_atomic\_v2/manifests/factorial\_summary.csv\`

\- R012 Table 4 exact retained bindings:

  \- \`manifests/factorial\_contrasts.csv\`

  \- \`manifests/factorial\_summary.csv\`

\- R013 Table 5 exact retained bindings:

  \- \`manifests/dose\_summary.csv\`

  \- \`manifests/paired\_contrasts.csv\`

  \- \`manifests/paired\_contrasts\_summary.csv\`

\- R014 Table 6 exact retained bindings:

  \- \`manifests/paired\_contrasts.csv\`

  \- \`manifests/paired\_contrasts\_summary.csv\`

&nbsp;

Interpretation: this is a clerical consistency amendment only. The machine-facing \`MANUSCRIPT\_EVIDENCE\_BINDINGS\_R010\_R014.md\` remains authoritative for the exact accepted evidence envelope, including the R011 and R012 retained provenance limitations.

&nbsp;