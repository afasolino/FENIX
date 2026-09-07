# ADR 0024 — Omit terminal fio close records from H3 replay iologs

A controlled experiment with the pinned fio 3.42 source revision
`ab77643023f5d7e3c1b71a7576a564f368bf577a` established that terminal
`close` records corrupt fio's reported completed-byte accounting during
asynchronous `read_iolog` replay.

Four synthetic 64-read traces were exercised with one or two backing files,
with and without terminal close records, at QD1 and QD32.

At QD1 all cases reported the expected 262144 bytes.

At QD32, both traces containing terminal close records reported 64 total I/Os
but only 135168 completed read bytes, a deficit of 126976 bytes, exactly
31 x 4096 bytes = QD-1 operations.

At QD32, otherwise identical traces without terminal close records reported
the complete 262144 bytes. The result was invariant to use of one versus two
files.

The real H3 mixed pilot exhibited the same signature:
expected 4409466880 bytes, reported 4409339904 bytes, a deficit of exactly
126976 bytes, while achieving essentially full QD32 occupancy.

H3 therefore omits terminal file-close actions from generated fio replay
iologs and relies on normal fio job teardown to close file descriptors after
outstanding I/O is drained. Explicit add/open operations and all read
operations remain unchanged.

The strict equality gate between sampled bytes and fio-reported read bytes is
retained. No missing-byte tolerance or post-hoc correction is introduced.

The same rule is applied to the page-cache iolog generator for consistency and
to prevent the same issue if its execution strategy later becomes asynchronous.
