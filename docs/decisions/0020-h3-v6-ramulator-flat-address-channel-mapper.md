# ADR 0020 — Ramulator flat-address channel mapping

H3 v6 uses two pinned Ramulator2 frontend address contracts. `LatencyThroughputTrace`
constructs native DRAM address vectors, while `LoadStoreTrace` supplies a flat address.
Pinned `GenericDRAM` requires its channel mapper to populate `req.addr_vec[0]` before
selecting the channel controller. `PassThroughChannelMapper` is explicitly intended for
addr-vec-native frontends and only copies the flat address to `intra_channel_addr`; it does
not materialize the channel element for a flat-address request. This caused a workstation
segmentation fault as soon as a `LoadStoreTrace` LPDDR replay was dispatched.

The adapter now selects channel mapping by frontend contract: addr-vec-native streaming
uses `PassThroughChannelMapper`, while flat-address `LoadStoreTrace` uses upstream
`CacheLineInterleave(interleave_bits=0)`. For the one representative channel modeled by
H3, the latter sets channel 0 and preserves the flat intra-channel address, so it introduces
no inter-channel striping and does not change the LPDDR geometry, timing, trace contents,
or H3 claim boundary.

A regression test requires this explicit frontend-to-channel-mapper pairing and rejects
unknown address modes.
