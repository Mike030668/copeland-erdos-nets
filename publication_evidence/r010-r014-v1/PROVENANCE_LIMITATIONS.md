# Retained provenance limitations

These are accepted historical limitations, not repaired execution receipts.

R011: canonical seed-atomic v2 used Python 3.13.15, Torch 2.11.0+cu128,
CUDA 12.8, NumPy 2.1.3, datasets 4.0.0 and transformers 5.15.1. This differs
from its frozen smoke/R010 runtime and must be disclosed for cross-series
absolute-PPL comparisons. Exact per-block source bundle hashes were checked
at execution but not persisted; they cannot be recovered. Selected checkpoint
bytes and file digests were not exported. The retained checkpoint manifest
records epochs/filenames only and does not enable checkpoint replay.
The partial staged-scope source-tree document is omitted because it references
private operational tooling; this does not upgrade its known provenance gap.
The retained `source_bundle_sha256.txt` documents the limitation verbatim.

R012: no full all-parameter base-state hash or unique physical VM identifier
for every canonical attempt was retained. Base embedding hashes, explicit
non-embedding invariance, attention/batch parity and retained source/runtime
and checkpoint receipts remain available. Spectral diagnostics were not
collected. Historical operational recycle causes are not newly inferred here.

R014: the interrupted first seed60 attempt did not reach terminal metrics or
parity packaging. Its own completed epoch count is unknown. The retained
v2 `seed_block_manifest.csv` explicitly records this gap; only the fresh
successful rerun contributes metrics. Package v2 repaired provenance only.

Dataset/tokenizer identity: accepted records name `Salesforce/wikitext`,
`wikitext-2-raw-v1`, and GPT-2 (`gpt2`); R014 records effective vocabulary
50257. Immutable dataset/tokenizer repository revision IDs are not established
by these release records. No current Hub revision is substituted retrospectively.
R012–R014 source manifests identify retained execution files; they do not
justify claiming an unrecorded upstream dataset revision.

Historical paths in machine-readable checkpoint manifests are source receipt
labels, not portable download locations. Large weights remain unpublished;
digests are supplied where retained. Reports, DS/control documentation,
private package snapshots and console logs are intentionally omitted from the
public tree; their source dispositions are inventoried.

Validation limitation: the inherited R011 production hash test depends on an
untracked local historical R010 receipt. It failed in the current environment
while 65 other targeted tests passed. The exact test and protocol code are
unchanged from the accepted base; the local regenerated tensor hash is not
accepted run evidence. This failure is disclosed for DS intake, not repaired
by modifying scientific code or accepted receipts.
