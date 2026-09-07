"""Byte-accurate cache semantics for the FENIX H3 residency replay."""
from __future__ import annotations

from collections import Counter, OrderedDict
import heapq
from dataclasses import dataclass
from typing import Hashable, Iterable, Protocol

from analysis.h3.common import H3Error
from analysis.h3.trace import Geometry, TraceEvent


@dataclass(frozen=True)
class CacheObject:
    key: tuple[Hashable, ...]
    size_bytes: int
    kind: str
    storage_keys: tuple[tuple[Hashable, ...], ...]


class ByteCache(Protocol):
    capacity_bytes: int
    resident_bytes: int

    def classify_epoch(self, objects: Iterable[CacheObject]) -> dict[tuple[Hashable, ...], bool]: ...
    def commit_epoch(self, objects: Iterable[CacheObject]) -> None: ...


class ByteLRU:
    """Byte-capacity LRU with atomic pre-epoch hit classification."""

    def __init__(self, capacity_bytes: int):
        if int(capacity_bytes) < 0:
            raise H3Error("cache capacity cannot be negative")
        self.capacity_bytes = int(capacity_bytes)
        self.resident_bytes = 0
        self.entries: OrderedDict[tuple[Hashable, ...], CacheObject] = OrderedDict()

    def classify_epoch(self, objects: Iterable[CacheObject]) -> dict[tuple[Hashable, ...], bool]:
        unique = {obj.key: obj for obj in objects}
        return {key: key in self.entries for key in unique}

    def commit_epoch(self, objects: Iterable[CacheObject]) -> None:
        unique = {obj.key: obj for obj in objects}
        # Classification happened before this call. The deterministic order below
        # only defines post-epoch LRU state; it cannot manufacture same-timestamp hits.
        for key in sorted(unique, key=repr):
            obj = unique[key]
            if obj.size_bytes <= 0:
                raise H3Error(f"invalid cache object size for {key}: {obj.size_bytes}")
            if key in self.entries:
                self.entries.move_to_end(key)
                continue
            if obj.size_bytes > self.capacity_bytes:
                continue
            while self.entries and self.resident_bytes + obj.size_bytes > self.capacity_bytes:
                _, victim = self.entries.popitem(last=False)
                self.resident_bytes -= victim.size_bytes
            self.entries[key] = obj
            self.resident_bytes += obj.size_bytes


class ByteLFU:
    """Online size-aware LFU admission/eviction with atomic epoch semantics.

    Frequency is learned only from epochs that have already been classified.
    Admission uses cumulative accesses per resident byte. Therefore this is an
    online conventional policy rather than an oracle; same-timestamp accesses
    cannot train the cache early enough to become hits in that epoch.
    """

    def __init__(self, capacity_bytes: int):
        if int(capacity_bytes) < 0:
            raise H3Error("cache capacity cannot be negative")
        self.capacity_bytes = int(capacity_bytes)
        self.resident_bytes = 0
        self.entries: dict[tuple[Hashable, ...], CacheObject] = {}
        self.frequency: Counter[tuple[Hashable, ...]] = Counter()
        self.last_epoch: dict[tuple[Hashable, ...], int] = {}
        self.epoch = 0

    def classify_epoch(self, objects: Iterable[CacheObject]) -> dict[tuple[Hashable, ...], bool]:
        unique = {obj.key: obj for obj in objects}
        return {key: key in self.entries for key in unique}

    def _score(self, key: tuple[Hashable, ...], obj: CacheObject) -> tuple[float, int, str]:
        # Frequency density is the primary replacement metric. Recency and a
        # deterministic key representation break ties without altering frequency.
        return (
            float(self.frequency.get(key, 0)) / max(1, int(obj.size_bytes)),
            int(self.last_epoch.get(key, -1)),
            repr(key),
        )

    def commit_epoch(self, objects: Iterable[CacheObject]) -> None:
        materialized = list(objects)
        if not materialized:
            self.epoch += 1
            return
        for obj in materialized:
            if obj.size_bytes <= 0:
                raise H3Error(f"invalid cache object size for {obj.key}: {obj.size_bytes}")
            self.frequency[obj.key] += 1
            self.last_epoch[obj.key] = self.epoch
        unique = {obj.key: obj for obj in materialized}
        # Most valuable newly observed objects are considered first. A candidate
        # is admitted only when it is at least as valuable as every victim needed
        # to make room, preventing LFU from degrading into unconditional demand fill.
        candidates = sorted(
            unique.values(),
            key=lambda obj: self._score(obj.key, obj),
            reverse=True,
        )
        victim_heap: list[tuple[tuple[float, int, str], tuple[Hashable, ...]]] | None = None
        for obj in candidates:
            key = obj.key
            if key in self.entries or obj.size_bytes > self.capacity_bytes:
                continue
            needed = self.resident_bytes + obj.size_bytes - self.capacity_bytes
            if needed <= 0:
                self.entries[key] = obj
                self.resident_bytes += obj.size_bytes
                continue
            # The replacement scores are fixed for the rest of this epoch: all
            # frequency/recency learning happened above before admission starts.
            # Candidates are processed in strictly descending score order, so an
            # object admitted earlier in this loop can never be a legal victim for
            # a later candidate. Build the resident victim heap lazily once per
            # epoch rather than re-sorting the complete cache for every miss.
            if victim_heap is None:
                victim_heap = [
                    (self._score(victim_key, victim), victim_key)
                    for victim_key, victim in self.entries.items()
                ]
                heapq.heapify(victim_heap)

            reclaimed = 0
            chosen: list[tuple[tuple[float, int, str], tuple[Hashable, ...]]] = []
            candidate_score = self._score(key, obj)
            while victim_heap and victim_heap[0][0] <= candidate_score and reclaimed < needed:
                victim_score, victim_key = heapq.heappop(victim_heap)
                victim = self.entries[victim_key]
                chosen.append((victim_score, victim_key))
                reclaimed += victim.size_bytes

            if reclaimed < needed:
                # The original algorithm makes no eviction when the candidate
                # cannot reclaim enough admissible bytes. Restore the inspected
                # heap prefix exactly and leave cache state unchanged.
                for victim_item in chosen:
                    heapq.heappush(victim_heap, victim_item)
                continue

            for _, victim_key in chosen:
                victim = self.entries.pop(victim_key)
                self.resident_bytes -= victim.size_bytes
            self.entries[key] = obj
            self.resident_bytes += obj.size_bytes
        self.epoch += 1


def replacement_kind(policy: str) -> str:
    if policy.endswith("_lru"):
        return "lru"
    if policy.endswith("_lfu"):
        return "lfu"
    raise H3Error(f"unsupported replay policy {policy}")


def granularity_kind(policy: str) -> str:
    if policy.startswith("useful_object_"):
        return "useful_object"
    if policy.startswith("vm_granularity_"):
        return "vm_granularity"
    raise H3Error(f"unsupported replay policy {policy}")


def make_cache(policy: str, capacity_bytes: int) -> ByteCache:
    replacement = replacement_kind(policy)
    return ByteLRU(capacity_bytes) if replacement == "lru" else ByteLFU(capacity_bytes)


def pages_for_range(offset: int, length: int, page_bytes: int) -> tuple[int, ...]:
    if offset < 0 or length <= 0 or page_bytes <= 0:
        raise H3Error("invalid byte range/page geometry")
    first = offset // page_bytes
    last = (offset + length - 1) // page_bytes
    return tuple(range(first, last + 1))


def cache_objects(event: TraceEvent, geometry: Geometry, policy: str) -> list[CacheObject]:
    objects: list[CacheObject] = []
    granularity = granularity_kind(policy)
    if event.kind == "ple":
        if granularity == "useful_object":
            for row in event.object_ids:
                offset = int(row) * geometry.ple_row_bytes
                pages = pages_for_range(offset, geometry.ple_row_bytes, geometry.storage_page_bytes)
                objects.append(
                    CacheObject(
                        ("ple_row", int(row)),
                        geometry.ple_row_bytes,
                        "ple",
                        tuple(("ple_page", page) for page in pages),
                    )
                )
        else:
            page_ids: set[int] = set()
            for row in event.object_ids:
                offset = int(row) * geometry.ple_row_bytes
                page_ids.update(pages_for_range(offset, geometry.ple_row_bytes, geometry.storage_page_bytes))
            for page in sorted(page_ids):
                objects.append(
                    CacheObject(("ple_page", page), geometry.storage_page_bytes, "ple", (("ple_page", page),))
                )
    elif event.kind == "expert":
        if event.layer is None:
            raise H3Error("expert event has no layer")
        size = geometry.expert_bytes if granularity == "useful_object" else geometry.expert_stride_bytes
        for expert in event.object_ids:
            objects.append(
                CacheObject(
                    ("expert", int(event.layer), int(expert)),
                    size,
                    "expert",
                    (("expert", int(event.layer), int(expert)),),
                )
            )
    else:
        raise H3Error(f"unknown trace event kind {event.kind}")
    return objects


def storage_record(key: tuple[Hashable, ...], geometry: Geometry) -> tuple[str, int, int, int]:
    if key[0] == "ple_page":
        page = int(key[1])
        return "ple", page * geometry.storage_page_bytes, geometry.storage_page_bytes, geometry.storage_page_bytes
    if key[0] == "expert":
        layer, expert = int(key[1]), int(key[2])
        index = layer * geometry.experts + expert
        return "expert", index * geometry.expert_stride_bytes, geometry.expert_stride_bytes, geometry.expert_bytes
    raise H3Error(f"unknown storage object {key}")
