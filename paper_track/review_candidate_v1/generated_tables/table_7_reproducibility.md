**Table 7. Reproducibility and provenance envelope.**

| Cycle | Main reproducibility controls | Retained boundary |
|---|---|---|
| R010 | Same-path intervention; base/changed/unchanged parameter hashes; batch-order and checkpoint records | Applies to corrected attention-only cycle |
| R011 | Seed-atomic factorial rows and paired factorial summaries | Global runtime differs from frozen smoke runtime; historical source-bundle provenance incomplete |
| R012 | Base embedding hash; explicit non-embedding invariance hashes; exact factor construction; scale/direction hashes and statistics; attention parity; all-epoch batch parity; checkpoints | Full all-parameter `base_state` hash not retained; unique provider-level physical Colab VM/session ID not retained for every canonical attempt; seed-atomic lineage is operational rather than immutable provider-level identity. These provenance limits do not alter the accepted within-cycle factorial inference. |
| R013 | Fresh seed block; exact RMS ladder; paired contrasts; checkpoints; source/runtime manifests; embedding dynamics | Below-Xavier versus Xavier unresolved at current precision |
| R014 | Larger-capacity resolved config; exact RMS ladder; paired contrasts; checkpoints; source/runtime manifests | Below-Xavier condition not tested; no embedding RMS/gradient telemetry |
