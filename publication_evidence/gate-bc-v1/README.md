# Gate B+C evidence v1 — candidate, not final release

Five paired seeds67–71, five conditions B00/B01/B10/B11/C_002,15epochs per cell.
Documentary scientific evidence for the fixed implementation, WikiText-2 and
training regime. DS technical intake PASS; GateB INTERACTION_RESOLVED;
GateC C002_WORSE. GateD deferred by bounded scope, not empirically closed.

## Reproduce documentary calculations

From repository root, using Python standard library only:

```bash
python3 publication_evidence/gate-bc-v1/analysis/calculate.py --out publication_evidence/gate-bc-v1/tables
python3 publication_evidence/gate-bc-v1/analysis/compare_ds.py
cd publication_evidence/gate-bc-v1
sha256sum -c SHA256SUMS
```

No torch, dataset access, model training, evaluation or credentials are required
for these calculations. Paired n5,df4,tcritical2.7764451051977987. Inputs are
25accepted metrics rows; all375epoch rows separately retained. Calculations
return both marginal effects, interaction,C002control and all four simple
effects. Classification is the fixed DS decision, not script-generated inference.
Intervals describe seed variation on one corpus/regime, not new-data uncertainty.

## Source and scientific limits

Seed67 executed7fc1e93604129eb17f6d9966809f24460776b0d0; seeds68–71 executed
f10d70c531d84f219d55ea558378c9100723ec52. Exact snapshots in source_snapshots/;
their difference is the historically reviewed transport-only remediation.
Candidate commit is packaging, not execution. Original public repository source
and frozen config are preserved, including documentary canonical_authorization
field; this evidence package does not authorize new training.

Read DS_SCIENTIFIC_DECISION.md in full. Token alpha affects token embeddings
before positional addition and the gradient chain rule; no pure optimizer
mechanism isolated and no ongoing forward renormalization. Large-forward
simple effect at large parameterization remains unresolved, not absent or
equivalent. C002 worse is a bounded fixed-direction control, not GPT-2
replication or universal0.02failure; no optimum or DOMINANT label.
Historical R010–R014 are not pooled or rewritten. No CE-P1 evidence included.
Four failed attempts are wholly excluded and immutable.

## Provenance and binary availability

SOURCE_TO_PUBLIC.json marks RAW_COPY versus PUBLIC_DERIVATIVE, raw source and
public hashes, omissions/reasons and unchanged scientific values. Sources are
logical project-relative evidence paths or git commit paths, not credentials.
RAW_COPY_AUDIT.json verifies raw-copy identity. DERIVATIVE_LINEAGE.json records
all public derivatives. Public numeric payloads retain every accepted row/value.

ARCHIVE_INVENTORY.json and CHECKPOINT_INVENTORY.json publish accepted hashes,
sizes and source provenance. Approximately4.7GB canonical archives and50checkpoint
binaries are retained privately and NOT distributed here. No publicly downloadable
checkpoints or independent model reproduction is claimed. Architect performed
binary/archive/readback audits; DS independently reviewed documentary receipts
and recalculated estimands, not checkpoint bytes or SA operations.
Recorded dataset/tokenizer revisions belong to this run; historical revision
continuity is UNRETAINED and not independently verified. Legacy seed67 durability
receipt lacks later transaction trace details; this is disclosed, not fabricated.
Final receipt own retry trace may remain local under the historical DS exception.

The exhaustive INVENTORY.csv/SHA256SUMS covers payloads; SHA256SUMS excludes
itself and INVENTORY.csv to avoid a hash cycle, inventories those controls by
size/digest conventions described in INVENTORY_POLICY.md. Inputs for generated
tables are hashed in tables/results.json. No figures are necessary for these
exact mappings; deterministic CSV tables are supplied.

Final annotated tag publication-evidence-gate-bc-v1 is PROPOSED only; final DS
exact-candidate review pending. Historical publication-evidence-r010-r014-v1
remains untouched. Paper Agent may revise a new derivative, preserving Owner
edits; journal readiness remains pending manuscript/response/evidence review.
