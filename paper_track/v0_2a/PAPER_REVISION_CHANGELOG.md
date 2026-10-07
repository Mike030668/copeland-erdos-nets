# PAPER_REVISION_CHANGELOG.md

Project: copeland-erdos-nets / paper_track  
Artifact: P5 change log for `paper_post_R014_v0_1`  
Status: P5 draft support file. Pre-submission review is not ready.

## 1. Old narrative → new narrative

| Old narrative element | New manuscript treatment |
|---|---|
| Historical selective-initialization gain was framed as attention-projection spectral/gain conditioning. | Retained as H0 historical observation only. The old attribution is tested and narrowed by R010. |
| Attention-only Q/K/V/O initialization was treated as the main causal path. | R010 shows attention-only same-path intervention does not reproduce the historical large gain in the tested corrected setup. |
| CE-LCG or Orthogonal attention initialization could be foregrounded as a superior method. | CE-LCG may appear as historical context and corrected R010 control only. Orthogonal is related-work/control context only. No superiority claim. |
| Validation-PPL legacy values were central to the narrative. | R010–R014 primary endpoint is held-out test PPL at validation-selected checkpoint. Legacy validation PPL is non-comparable historical context. |
| Spectral conditioning of attention was the main mechanism narrative. | Downstream mechanism remains unresolved. Dynamics are diagnostic only under C6. |
| Old manuscript could be patched. | New manuscript is reconstructed from P0–P4 and accepted R010–R014 evidence. Old manuscript is not grandfathered. |

## 2. Retired claims

| Retired claim | Replacement |
|---|---|
| Selective attention initialization alone improves PPL by 21–25%. | C1: corrected attention-only same-path intervention does not reproduce the historical large gain in the tested regime. |
| CE-LCG is uniquely beneficial or theoretically causal. | CE-LCG is a historical path element and corrected R010 control, not a causal/superiority claim. |
| Orthogonal attention initialization is superior. | Orthogonal is not a supported winner in the post-R014 manuscript. |
| Spectral conditioning of attention caused the historical gain. | R011/R012 support token-embedding initial RMS scale as the dominant tested control variable. |
| Lower embedding scale is universally better. | R013 supports THRESHOLD_PLATEAU only; half-Xavier vs Xavier is unresolved, and R014 does not test below-Xavier. |
| Result transfers across corpora or large LLMs. | Scope limited to WikiText-2, two tested capacities, fixed protocol, n=5 per corrected cycle. |

## 3. Retained material

| Material | Status |
|---|---|
| Transformer initialization and training-stability background | Retained with updated framing. |
| Attention-specific initialization literature | Retained as direct prior work, not as empirical comparison. |
| Historical R008/R009 observation | Retained only as motivation and reconstruction target. |
| Expert review / audit material | Retained as internal closure input, not empirical evidence. |
| Reproducibility guardrails | Retained and strengthened. |

## 4. New R010–R014 sections

| Section | Claim IDs | Evidence |
|---|---|---|
| R010 attention-only reconstruction | C1 | `R010_paired_attention_init_confirmation/metrics/per_seed.csv`, `metrics/summary.csv`, `metrics/paired_differences.csv`, R010 DS review |
| R011 embedding localization | C2 | `R011_embedding_init_factorial_ablation/attempt_seed_atomic_v2/per_seed.csv`, `attempt_seed_atomic_v2/manifests/factorial_contrasts.csv`, R011 DS review |
| R012 scale-vs-redraw decomposition | C3 | `R012_embedding_scale_redraw_factorial/per_seed.csv`, `manifests/factorial_summary.csv`, `manifests/factorial_contrasts.csv`, R012 DS review |
| R013 dose response | C4/C6 | `R013_embedding_scale_response_and_dynamics/per_seed.csv`, `dose_summary.csv`, `manifests/paired_contrasts.csv`, `manifests/paired_contrasts_summary.csv`, R013 DS review |
| R014 capacity transfer | C5/C6 | `R014_corrected_larger_model_embedding_scale_confirmation_v2/per_seed.csv`, `manifests/paired_contrasts.csv`, `manifests/paired_contrasts_summary.csv`, R014 DS review |

## 5. P5 confirmations

- No new experiments were run.
- R015/R016 were not opened.
- No accepted evidence was changed.
- No DS class was changed.
- Frozen manuscript was not edited.
- Expert review was not edited.
- Pre-submission review is not ready.

---

# v0_1 → v0_2 bounded repair

`paper_post_R014_v0_1.md` remains immutable. The following changes implement the P5 intake repair without changing any accepted result or experiment.

| Repair | v0_2 change |
|---|---|
| Methods self-contained | Added exact dataset/config/tokenizer, split handling, model sizes, training hyperparameters, seed blocks, validation-selected test endpoint, seed-matched contrasts, and Student-t paired CI procedure. |
| Embedding intervention | Added architecture facts (`token_emb + pos_emb`, no `sqrt(d_model)` multiplier, untied `lm_head`) and numbered RMS/scale equations; added exact R013/R014 target RMS ladders and ~159× constructor/Xavier ratio. |
| R011 limitation | Added accepted runtime/source-provenance limitation and restricted use to within-R011 paired factorial inference. |
| Complete result tables | Added full R011 factorial contrasts, numeric R012 redraw/interaction estimates, complete requested R013 contrast set, and all R014 primary/secondary contrasts. |
| Generated, audited tables | Added `scripts/generate_manuscript_tables.py`, canonical-input snapshots, `generated_tables/`, and fail-closed `AUDIT_REPORT.json` (21 checks PASS). |
| Governance separation | Removed visible claim IDs, internal classifications, raw evidence paths, and process terminology from journal-facing prose; added `MANUSCRIPT_EVIDENCE_BINDINGS_R010_R014.md`. |
| Figure package | Materialized Fig. 1–6 and diagnostic Fig. 7a–d; added `FIGURE_BINDING_MANIFEST.csv` and deterministic figure-generation script for manuscript-created Fig. 1–2. |
| Citation graph | Added consistent in-text numeric citations, Fixup/T-Fixup/ReZero/μP, WikiText, GPT-2, AdamW, and normalized NormFormer to ICLR 2022; preserved direct/adjacent 2018–2026 work. |
| Generality boundary | Added architecture-specific scale discussion and explicit non-transfer to different embedding scale/normalization/tied-head designs. |
| Historical observation | Added concrete legacy best-validation-PPL values with prominent confounded/non-comparable/non-causal warning. |
| Publication cleanup | Removed working-artifact/project/status/claim-map/raw-binding planning material from visible manuscript; moved provenance envelope to Reproducibility; adopted narrower publication title. |

## v0_2 confirmations

- No new experiments were run.
- R015/R016 were not opened.
- No accepted numerical result was changed.
- No internal scientific classification was changed.
- `paper_post_R014_v0_1.md` was not modified.
- The frozen old manuscript was not modified.
- The expert review was not modified.
- Pre-submission review is not declared ready.

---

# v0_2 → v0_2a bounded provenance-disclosure repair

`paper_post_R014_v0_2.md` remains immutable. This micro-repair makes no numerical, figure, classification, title, narrative, bibliography, or experimental-scope change.

| Repair | v0_2a change |
|---|---|
| R012 provenance disclosure | Added the two accepted R012 provenance limitations to Sections 12–13 and Table 7: no retained full all-parameter `base_state` hash; no immutable provider-level physical Colab VM/session ID for every canonical attempt. |
| R012 retained causal evidence | Explicitly listed base embedding hash, non-embedding invariance hashes, exact factor construction, attention parity, all-epoch batch parity, and checkpoint evidence. |
| C3 evidence bindings | Added the same two limitations without changing the accepted `SCALE_DOMINATES` classification or within-cycle factorial inference. |
| Eq. (1) clerical notation | Replaced ambiguous positional notation with `E_pos[:, :T, :]` notation matching implementation. |
| Registry | Append-only superseding exact-path note added; historical registry rows left unchanged. |
| Tables/Figures | Tables 2–6 and all figures unchanged; only Table 7 materialized artifact changes. |

## v0_2a confirmations

- No experiment was run.
- R015/R016 were not opened.
- No accepted numerical value or confidence interval changed.
- No figure changed.
- No DS scientific classification changed.
- `paper_post_R014_v0_2.md` was not modified.
- Pre-submission review is not declared ready.
