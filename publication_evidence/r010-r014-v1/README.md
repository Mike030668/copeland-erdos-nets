# R010–R014 publication evidence release candidate

This tree exposes immutable copies of DS-accepted canonical run evidence.
Code, configs and tests are inherited unchanged from accepted R014 protocol
commit `3b14595b99312ee14261235338fee9a5861e2cd2`.

Branch: `paper-release/r010-r014-evidence-v1`.
Proposed annotated tag: `publication-evidence-r010-r014-v1`.
Tag creation is pending DS/Owner approval of this release candidate.

`SOURCE_TO_RELEASE.csv` identifies every copied run artifact and verifies its
source/release SHA-256 equality. `SOURCE_DISPOSITION.csv` records omitted
source items. `PROTOCOL_INVENTORY.csv` binds the inherited public code to the
accepted base. `PAPER_SNAPSHOT_INVENTORY.csv` binds byte-identical copies of
the two accepted public paper snapshots to their original commits.

`RELEASE_ARTIFACTS.csv` inventories release evidence; `SHA256SUMS` binds the
evidence, code/config/test inventory, and paper snapshot inventory. Verify
from repository root with `sha256sum -c publication_evidence/r010-r014-v1/SHA256SUMS`.
The SHA manifest excludes only itself to avoid recursive hashing.

R011 uses only accepted seed-atomic v2 results; the excluded cross-VM v1 is
described in `ATTEMPT_LEDGER.csv`. R014 uses accepted package v2, including
the seed60 discarded attempt recorded in its retained seed-block manifest.
See `PROVENANCE_LIMITATIONS.md` before making reproducibility claims.

No accepted metrics, confidence intervals, selected checkpoints or claims
were recomputed or changed. No scientific training run was performed.
Existing small synthetic CPU tests are validation of the inherited code.
Large weights and private operational/control documents are not published.

The frozen paper snapshots are included byte-for-byte, including their
historical availability wording. Paper Agent updates the cited release tag
and commit only after DS Gate A PASS.
