# ADR 0025 — Deterministic replay serialization and fio provenance closure

H3 final decisions bind finite-residency replay and physical fio calibration
through the exact SHA256 of the lower-tier miss stream and through a common
FENIX execution-repository identity.

Python gzip writers use the current time in the gzip header unless an explicit
mtime is supplied. Consequently, regenerating an otherwise identical residency
replay could change the compressed-file SHA and make equivalent semantic traces
fail the exact provenance gate.

Residency miss streams and logical-access traces are therefore written using
gzip mtime=0 with path-dependent original-filename metadata omitted. Identical
logical records now produce byte-identical compressed artifacts.

The fio achieved-queue-depth fraction is additionally clamped to [0,1] because
fio histogram percentages are independently rounded and may sum slightly above
100 percent.

Finally, fio summary verification now requires the sample manifest and each
physical fio-run provenance artifact to share the same FENIX execution
repository identity. This prevents a summary emitted by a newer implementation
from silently rebinding measurements executed under an older implementation.

No H3 workload, cache, storage, LPDDR, or decision threshold semantics are
changed.
