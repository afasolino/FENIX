# ADR 0021 — LFU replay complexity hardening

The H3 v6 online byte-LFU policy is scientifically retained unchanged, but its first
workstation execution exposed an avoidable implementation cost: whenever a newly
observed candidate needed space, the replay sorted the complete resident cache again.
For realistic H3 capacities this repeated full-cache sort dominates execution time and
can make the replay appear stalled.

All LFU frequency and recency updates occur before admission begins. Candidate objects
are then processed in descending replacement-score order. Therefore an object admitted
earlier in the same epoch has a strictly greater replacement score than every later
candidate and cannot be a legal victim for those later candidates. The implementation
now builds one deterministic min-heap from the pre-admission resident set and reuses it
throughout the epoch. A failed admission restores every inspected heap element, exactly
matching the prior no-eviction behavior.

This is an algorithmic acceleration only. Capacity accounting, frequency density,
recency/key tie-breaking, same-timestamp atomicity, admission order, victim eligibility,
and resulting cache state are unchanged. A randomized differential regression compares
the optimized implementation against the original repeated-sort algorithm over multiple
capacities and hundreds of epochs and requires exact equality of hits, resident entries,
resident bytes, frequencies, recency state, and epoch count.
