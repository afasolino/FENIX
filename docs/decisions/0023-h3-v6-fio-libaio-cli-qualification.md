# ADR 0023 — H3 fio libaio execution qualification

The first real H3 QD32 storage pilot exposed two execution-path defects that
were not visible in synthetic/unit qualification.

First, the pinned fio source revision had initially been compiled on the Rocky
8 workstation without the libaio development headers. The resulting fio binary
was source-pinned and clean but could not load the `libaio` engine required by
the H3 storage contract. Tool qualification therefore now requires the built
binary to advertise the `libaio` engine, and bootstrap fails immediately when
that engine is absent.

Second, H3 passed `--readonly=1` to fio. fio 3.42 defines `readonly` as a
boolean command-line flag and rejects the assignment form. Both the direct fio
calibration and page-cache replay now pass `--readonly`.

The storage semantics are unchanged: reads remain read-only, direct storage
calibration remains `libaio` + `direct=1`, configured queue depths are
unchanged, and page-cache replay remains non-direct. The fio source pin remains
`ab77643023f5d7e3c1b71a7576a564f368bf577a`; rebuilt binary bytes are recorded
in measurement provenance.

The failed pre-fix physical pilot is retained as negative qualification
evidence and is not consumed by H3 decisions. Because this change advances the
FENIX execution commit, final decision/prerequisite artifacts must be regenerated
under the new execution identity; immutable H1/H2 source traces and the already
captured host-memory measurement do not need to be repeated.
