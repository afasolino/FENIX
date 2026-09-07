# ADR 0019: H3 v6 lsblk portability fallback

## Status
Accepted as a non-scientific portability correction to schema-v6 tooling.

## Context
`probe-storage` originally requested `PATH` and `MOUNTPOINTS` from `lsblk`. Rocky Linux 8 ships an older util-linux interface that rejects that output-column set, preventing device binding even though `findmnt` and the required block-device identity are available. This is an execution portability failure, not a change to the H3 storage model.

## Decision
Keep the existing modern `lsblk` query as the primary path. If it fails, retry with the older portable column set `NAME,MODEL,SERIAL,TRAN,SIZE,TYPE,PKNAME,MAJ:MIN,MOUNTPOINT`. Normalize the fallback result so `path` is derived from absolute `NAME` (`-p`) and `mountpoints` is derived from `MOUNTPOINT`. The scientific matching remains based on the `MAJ:MIN` identity returned independently by `findmnt`.

## Consequences
No H3 geometry, thresholds, timing model, queue-depth model, storage endpoint, or claim boundary changes. Evidence generated after this correction must bind to the new execution-repository commit. Tool qualification and storage binding are rerun after committing the patch so downstream prerequisites cannot mix evidence from different implementation commits.
