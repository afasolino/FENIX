"""Pinned Ramulator2 LPDDR5 qualification and access-pattern calibration."""
from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from typing import Any, Iterator

from analysis.h3.common import H3Error, sha256_file, stable_u64


def validate_resolved_profile(
    org: dict[str, Any], timing: dict[str, Any], lpddr: dict[str, Any], ncs: int
) -> None:
    if int(org.get("rank", -1)) != int(lpddr["ranks_per_channel"]):
        raise H3Error(f"Ramulator rank geometry drift: {org.get('rank')}")
    if int(org.get("channel_width", -1)) != int(lpddr["channel_width_bits"]):
        raise H3Error(f"Ramulator channel-width drift: {org.get('channel_width')}")
    required = lpddr["resolved_timing_required"]
    for key, expected in required.items():
        if int(timing.get(key, -1)) != int(expected):
            raise H3Error(
                f"Ramulator LPDDR timing drift for {key}: {timing.get(key)} != {expected}"
            )
    if int(timing.get("nCS", -1)) != int(ncs):
        raise H3Error(
            f"Ramulator nCS override did not resolve: {timing.get('nCS')} != {ncs}"
        )


def theoretical_channel_gb_s(lpddr: dict[str, Any]) -> float:
    return (
        float(lpddr["channel_width_bits"])
        * float(lpddr["data_rate_mt_s"])
        / 8.0
        / 1000.0
    )


def offered_frontend_gb_s(lpddr: dict[str, Any]) -> float:
    # One request/CK at clock_ratio=1. This is an offered-load qualification,
    # not a claim about achieved bandwidth.
    tck_ns = float(lpddr["resolved_timing_required"]["tCK_ps"]) / 1000.0
    return (
        float(lpddr["request_bytes"])
        * float(lpddr["frontend_clock_ratio"])
        / tck_ns
    )


def _controller(stats: dict[str, Any]) -> dict[str, Any]:
    try:
        return stats["memory_system"]["controller"]
    except (KeyError, TypeError) as exc:
        raise H3Error("Ramulator stats missing memory_system.controller") from exc


def extract_dram_layout(dram: Any) -> dict[str, Any]:
    """FENIX-local audited adapter for LatencyThroughputTrace.

    This intentionally mirrors the pinned upstream helper without importing
    Ramulator's private ``tests`` package, removing a brittle test-API dependency.
    """
    cls = type(dram)
    level_names = list(cls.levels.keys())
    org_dict, _ = dram.resolve()
    org_counts = [int(org_dict.get(name.lower(), 1)) for name in level_names]
    row_idx = level_names.index("Row")
    col_idx = level_names.index("Column")
    bank_positions = list(range(1, row_idx))
    bank_counts = [org_counts[i] for i in bank_positions]
    if "BankGroup" in level_names:
        bg_pos = level_names.index("BankGroup")
        idx = bank_positions.index(bg_pos)
        pos = bank_positions.pop(idx)
        count = bank_counts.pop(idx)
        bank_positions.append(pos)
        bank_counts.append(count)
    if "PseudoChannel" in level_names:
        pc_pos = level_names.index("PseudoChannel")
        idx = bank_positions.index(pc_pos)
        pos = bank_positions.pop(idx)
        count = bank_counts.pop(idx)
        bank_positions.append(pos)
        bank_counts.append(count)
    total_bank_units = math.prod(bank_counts)
    prefetch = int(cls.internal_prefetch_size)
    num_cols = org_counts[col_idx]
    return {
        "addr_vec_size": len(level_names),
        "bank_positions": bank_positions,
        "bank_counts": bank_counts,
        "total_bank_units": total_bank_units,
        "row_pos": row_idx,
        "col_pos": col_idx,
        "num_rows": org_counts[row_idx],
        "num_cols": num_cols,
        "internal_prefetch_size": prefetch,
        "num_cls": num_cols // prefetch,
    }


def _dram(lpddr: dict[str, Any], ncs: int) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import ramulator  # type: ignore

    dram = ramulator.dram.LPDDR5(
        org_preset=str(lpddr["org_preset"]),
        timing_preset=str(lpddr["timing_preset"]),
        rank=int(lpddr["ranks_per_channel"]),
        nCS=int(ncs),
    )
    org, timing = dram.resolve()
    validate_resolved_profile(org, timing, lpddr, ncs)
    return dram, org, timing


def _memory_system(dram: Any, addr_mapper: Any, lpddr: dict[str, Any]) -> Any:
    import ramulator  # type: ignore

    refresh = (
        ramulator.refresh_manager.AllBank()
        if bool(lpddr["refresh_enabled"])
        else ramulator.refresh_manager.NoRefresh()
    )
    ctrl = ramulator.controller.LPDDR5(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFSRowHit(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=addr_mapper,
        refresh_manager=refresh,
    )
    return ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
    )


def _finish_stats(
    stats: dict[str, Any], timing: dict[str, Any], requested: int, request_bytes: int
) -> dict[str, Any]:
    controller = _controller(stats)
    cycles = int(controller["cycles"])
    tck_ns = float(timing["tCK_ps"]) / 1000.0
    service_ns = cycles * tck_ns
    if service_ns <= 0:
        raise H3Error("Ramulator returned non-positive service time")
    reads = int(controller.get("num_read_reqs_served", 0))
    writes = int(controller.get("num_write_reqs_served", 0))
    served = reads + writes
    if served <= 0 or requested - served > 128:
        raise H3Error(
            f"Ramulator did not service calibration trace: served={served}, sent={requested}"
        )
    derived_total = served * request_bytes / service_ns
    stat_total = float(controller["total_throughput_MBps"]) / 1000.0
    if stat_total <= 0 or abs(derived_total - stat_total) / stat_total > 0.01:
        raise H3Error(
            f"Ramulator throughput/stat consistency failed: {derived_total} vs {stat_total}"
        )
    return {
        "controller_cycles": cycles,
        "tCK_ns": tck_ns,
        "service_ns": service_ns,
        "requests_sent_target": requested,
        "requests_served": served,
        "read_requests_served": reads,
        "write_requests_served": writes,
        "request_bytes": request_bytes,
        "bandwidth_gb_s": derived_total,
        "read_bandwidth_gb_s": reads * request_bytes / service_ns,
        "write_bandwidth_gb_s": writes * request_bytes / service_ns,
        "controller_reported_bandwidth_gb_s": stat_total,
        "stats": stats,
    }


def run_stream_point(lpddr: dict[str, Any], ncs: int, read_ratio: int) -> dict[str, Any]:
    """Run pinned upstream LatencyThroughputTrace streaming saturation."""
    import ramulator  # type: ignore

    dram, org, timing = _dram(lpddr, ncs)
    layout = extract_dram_layout(dram)
    frontend = ramulator.frontend.LatencyThroughputTrace(
        clock_ratio=int(lpddr["frontend_clock_ratio"]),
        nop_counter=1,
        num_probe_requests=0,
        num_streaming_requests=int(lpddr["ramulator_stream_requests"]),
        streaming_only=True,
        warmup_cycles=int(lpddr["ramulator_warmup_cycles"]),
        seed=12345,
        read_ratio=int(read_ratio),
        stream_cls=int(lpddr["stream_cls"]),
        stagger_stream_rows=True,
        **layout,
    )
    mem = _memory_system(dram, ramulator.addr_mapper.PassThroughAddrMapper(), lpddr)
    sim = ramulator.Simulation(frontend, mem)
    sim.run()
    result = _finish_stats(
        sim.stats,
        timing,
        int(lpddr["ramulator_stream_requests"]),
        int(lpddr["request_bytes"]),
    )
    peak = theoretical_channel_gb_s(lpddr)
    if result["bandwidth_gb_s"] > peak * 1.02:
        raise H3Error(
            f"Ramulator throughput exceeds x16 theoretical peak: {result['bandwidth_gb_s']}>{peak}"
        )
    return {
        "nCS": int(ncs),
        "read_ratio": int(read_ratio),
        "resolved_org": org,
        "resolved_timing": timing,
        "regime": "streaming",
        **result,
    }


def _capacity_bytes(org: dict[str, Any]) -> int:
    density_mbit = int(org["density"])
    ranks = int(org["rank"])
    return density_mbit * 1024 * 1024 // 8 * ranks


def _operations(mode: str, count: int, read_fraction: float) -> Iterator[str]:
    if mode == "read":
        yield from ("LD" for _ in range(count))
        return
    if mode == "write":
        yield from ("ST" for _ in range(count))
        return
    if mode != "mixed":
        raise H3Error(f"unknown LPDDR trace operation mode {mode}")
    if not 0.0 < read_fraction < 1.0:
        raise H3Error("mixed read fraction must lie strictly between zero and one")
    # Deterministic accumulator produces the requested fraction without PRNG
    # scheduling noise and preserves the exact address sequence across modes.
    acc = 0.0
    for _ in range(count):
        acc += read_fraction
        if acc >= 1.0:
            yield "LD"
            acc -= 1.0
        else:
            yield "ST"


def _write_regime_trace(
    path: Path,
    lpddr: dict[str, Any],
    org: dict[str, Any],
    regime: str,
    operation_mode: str,
) -> int:
    cfg = (lpddr.get("trace_regimes") or {}).get(regime)
    if not isinstance(cfg, dict):
        raise H3Error(f"LPDDR trace regime is not configured: {regime}")
    tx = int(lpddr["request_bytes"])
    capacity = _capacity_bytes(org)
    read_fraction = float(lpddr.get("mixed_read_fraction", 0.5))
    addresses: list[int] = []
    rng = random.Random(stable_u64(int(cfg.get("seed", 20260906)), regime, "lpddr"))
    if regime == "streaming":
        count = int(cfg["transactions"])
        addresses = [(index * tx) % max(tx, capacity - tx) for index in range(count)]
    elif regime == "ple_random_burst":
        objects = int(cfg["objects"])
        object_bytes = int(cfg.get("object_bytes", 160))
        tx_per_object = math.ceil(object_bytes / tx)
        max_base = capacity - tx_per_object * tx
        if max_base <= 0:
            raise H3Error("LPDDR modeled capacity is too small for PLE regime")
        for _ in range(objects):
            base = rng.randrange(0, max_base // tx + 1) * tx
            addresses.extend(base + index * tx for index in range(tx_per_object))
    elif regime == "expert_burst":
        bursts = int(cfg["bursts"])
        object_bytes = int(cfg["object_bytes"])
        tx_per_object = math.ceil(object_bytes / tx)
        object_span = tx_per_object * tx
        max_base = capacity - object_span
        if max_base <= 0:
            raise H3Error("LPDDR modeled capacity is too small for expert regime")
        alignment = int(cfg.get("alignment_bytes", 4096))
        for _ in range(bursts):
            base = rng.randrange(0, max_base // alignment + 1) * alignment
            addresses.extend(base + index * tx for index in range(tx_per_object))
    else:
        raise H3Error(f"unsupported LPDDR access regime {regime}")
    ops = _operations(operation_mode, len(addresses), read_fraction)
    with path.open("w", encoding="utf-8") as stream:
        for op, address in zip(ops, addresses, strict=True):
            stream.write(f"{op} {address}\n")
    return len(addresses)


def run_address_trace_point(
    lpddr: dict[str, Any], ncs: int, regime: str, operation_mode: str
) -> dict[str, Any]:
    """Replay deterministic flat-address traces through pinned LoadStoreTrace."""
    import ramulator  # type: ignore

    dram, org, timing = _dram(lpddr, ncs)
    with tempfile.TemporaryDirectory(prefix="fenix-h3-ramulator-") as tmp:
        trace = Path(tmp) / f"{regime}-{operation_mode}.trace"
        count = _write_regime_trace(trace, lpddr, org, regime, operation_mode)
        trace_sha = sha256_file(trace)
        frontend = ramulator.frontend.LoadStoreTrace(
            clock_ratio=int(lpddr["frontend_clock_ratio"]), path=str(trace)
        )
        mem = _memory_system(dram, ramulator.addr_mapper.RoBaRaCoCh(), lpddr)
        sim = ramulator.Simulation(frontend, mem)
        sim.run()
        result = _finish_stats(
            sim.stats, timing, count, int(lpddr["request_bytes"])
        )
    peak = theoretical_channel_gb_s(lpddr)
    if result["bandwidth_gb_s"] > peak * 1.02:
        raise H3Error(
            f"Ramulator trace throughput exceeds x16 theoretical peak: {result['bandwidth_gb_s']}>{peak}"
        )
    return {
        "nCS": int(ncs),
        "regime": regime,
        "operation_mode": operation_mode,
        "resolved_org": org,
        "resolved_timing": timing,
        "trace_sha256": trace_sha,
        "address_mapping": "RoBaRaCoCh",
        **result,
    }


class _NoAliasChannelLayout:
    """Collision-free sample-local placement for one representative LPDDR channel.

    The counterfactual oracle can have a larger logical address space than the
    physical 32-GiB platform.  Folding those addresses modulo channel capacity
    changes row/bank locality.  Instead, only the bounded sampled working set is
    embedded into the modeled channel.  Distinct logical objects receive
    disjoint page ranges; repeats reuse the same range.
    """

    def __init__(
        self,
        capacity_bytes: int,
        tx_bytes: int,
        seed: int,
        strategy: str = "hashed_bank_spread",
    ):
        self.capacity_bytes = int(capacity_bytes)
        self.tx_bytes = int(tx_bytes)
        self.page_bytes = 4096
        self.capacity_pages = self.capacity_bytes // self.page_bytes
        if self.capacity_pages <= 0:
            raise H3Error("modeled LPDDR channel has no allocatable pages")
        self.seed = int(seed)
        self.strategy = str(strategy)
        if self.strategy not in {"dense_first_touch", "hashed_bank_spread"}:
            raise H3Error(f"unknown no-alias LPDDR mapping strategy {self.strategy}")
        self.used_pages: set[int] = set()
        self.mapping: dict[tuple[Any, ...], tuple[int, int]] = {}
        self._dense_cursor = 0

    def _place(self, key: tuple[Any, ...], transaction_count: int) -> tuple[int, int]:
        if key in self.mapping:
            base, existing = self.mapping[key]
            if int(existing) != int(transaction_count):
                raise H3Error("actual-H3 object transaction span changed across repeated accesses")
            return base, existing
        span_bytes = max(self.tx_bytes, int(transaction_count) * self.tx_bytes)
        pages = math.ceil(span_bytes / self.page_bytes)
        if pages > self.capacity_pages:
            raise H3Error("one actual-H3 object is larger than modeled channel capacity")
        max_start = self.capacity_pages - pages
        if self.strategy == "dense_first_touch":
            candidate = self._dense_cursor
            probes = range(candidate, max_start + 1)
        else:
            candidate = stable_u64(
                self.seed, *key, "no-alias-channel-layout", self.strategy
            ) % (max_start + 1)
            probes = ((candidate + probe) % (max_start + 1) for probe in range(max_start + 1))
        for start_page in probes:
            occupied = False
            for page in range(start_page, start_page + pages):
                if page in self.used_pages:
                    occupied = True
                    break
            if occupied:
                continue
            for page in range(start_page, start_page + pages):
                self.used_pages.add(page)
            base = start_page * self.page_bytes
            self.mapping[key] = (base, int(transaction_count))
            if self.strategy == "dense_first_touch":
                self._dense_cursor = start_page + pages
            return base, int(transaction_count)
        raise H3Error(
            "actual-H3 sampled working set cannot be embedded collision-free in one modeled LPDDR channel"
        )

    def addresses(self, key: tuple[Any, ...], transaction_count: int) -> Iterator[int]:
        yield from self.addresses_at_indices(
            key, transaction_count, range(int(transaction_count))
        )

    def addresses_at_indices(
        self,
        key: tuple[Any, ...],
        transaction_count: int,
        indices: Iterable[int],
    ) -> Iterator[int]:
        base, count = self._place(key, transaction_count)
        for raw_index in indices:
            index = int(raw_index)
            if not 0 <= index < count:
                raise H3Error(
                    f"actual-H3 object-local transaction index {index} outside [0,{count})"
                )
            address = base + index * self.tx_bytes
            if address < 0 or address + self.tx_bytes > self.capacity_bytes:
                raise H3Error("no-alias LPDDR layout produced an out-of-range address")
            yield address

    def stats(self) -> dict[str, Any]:
        return {
            "mapping": "injective_sample_local_object_pages",
            "mapping_strategy": self.strategy,
            "logical_objects_mapped": len(self.mapping),
            "allocated_pages": len(self.used_pages),
            "allocated_bytes": len(self.used_pages) * self.page_bytes,
            "channel_capacity_bytes": self.capacity_bytes,
            "cross_object_aliases": 0,
            "modulo_address_folding": False,
        }


def _logical_trace_records(residency: dict[str, Any]) -> list[dict[str, Any]]:
    import gzip
    import json

    node = residency.get("logical_access_trace") or {}
    path = Path(str(node.get("path", "")))
    if not path.is_file():
        raise H3Error(f"residency logical access trace missing: {path}")
    if sha256_file(path) != node.get("sha256"):
        raise H3Error("residency logical access trace SHA mismatch")
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise H3Error("logical access trace contains non-object row")
                records.append(value)
    if len(records) != int(node.get("records", -1)):
        raise H3Error("logical access trace record-count mismatch")
    if not records:
        raise H3Error("logical access trace is empty")
    return records


def _channel_local_access(
    *,
    object_base: int,
    object_span_bytes: int,
    access_base: int,
    access_length_bytes: int,
    tx_bytes: int,
    channels: int,
    channel_index: int,
) -> tuple[int, list[int]]:
    """Return the object's channel-local span and indices touched by one access.

    The global address is transaction-striped over logical channels.  Removing
    the channel selector yields a compact object-local sequence for one
    representative channel.  The returned indices are stable across fills and
    later reads of the same storage object, so mixed traces preserve object
    identity without modulo aliasing.
    """
    object_base = int(object_base)
    object_span_bytes = int(object_span_bytes)
    access_base = int(access_base)
    access_length_bytes = int(access_length_bytes)
    tx_bytes = int(tx_bytes)
    channels = int(channels)
    channel_index = int(channel_index)
    if tx_bytes <= 0 or channels <= 0 or not 0 <= channel_index < channels:
        raise H3Error("invalid actual-H3 channel-local mapping geometry")
    if object_base % tx_bytes or access_base % tx_bytes:
        raise H3Error("actual-H3 object/access address is not transaction aligned")
    if object_span_bytes <= 0 or object_span_bytes % tx_bytes:
        raise H3Error("actual-H3 object span must be a positive transaction multiple")
    if access_length_bytes <= 0 or access_length_bytes % tx_bytes:
        raise H3Error("actual-H3 access length must be a positive transaction multiple")
    object_end = object_base + object_span_bytes
    access_end = access_base + access_length_bytes
    if access_base < object_base or access_end > object_end:
        raise H3Error("actual-H3 access lies outside its storage object")

    object_start_tx = object_base // tx_bytes
    object_tx_count = object_span_bytes // tx_bytes
    first_selected = (channel_index - (object_start_tx % channels)) % channels
    if first_selected >= object_tx_count:
        total_selected = 0
    else:
        total_selected = 1 + (object_tx_count - 1 - first_selected) // channels

    access_start_pos = (access_base - object_base) // tx_bytes
    access_tx_count = access_length_bytes // tx_bytes
    indices: list[int] = []
    for pos in range(access_start_pos, access_start_pos + access_tx_count):
        if pos < first_selected:
            continue
        delta = pos - first_selected
        if delta % channels == 0:
            indices.append(delta // channels)
    return total_selected, indices


def _mapped_access_addresses(
    layout: _NoAliasChannelLayout,
    key: tuple[Any, ...],
    *,
    object_base: int,
    object_span_bytes: int,
    access_base: int,
    access_length_bytes: int,
    tx_bytes: int,
    channels: int,
    channel_index: int,
) -> Iterator[int]:
    total, indices = _channel_local_access(
        object_base=object_base,
        object_span_bytes=object_span_bytes,
        access_base=access_base,
        access_length_bytes=access_length_bytes,
        tx_bytes=tx_bytes,
        channels=channels,
        channel_index=channel_index,
    )
    if total <= 0 or not indices:
        return
    yield from layout.addresses_at_indices(key, total, indices)

def _actual_object_transactions(
    records: list[dict[str, Any]],
    contract: dict[str, Any],
    org: dict[str, Any],
    layout: _NoAliasChannelLayout,
    *,
    stream_kind: str,
    start_record: int,
) -> Iterator[int]:
    """Yield collision-free channel-local addresses from actual H3 object IDs."""
    geometry = contract["geometry"]
    lpddr = contract["lpddr"]
    tx = int(lpddr["request_bytes"])
    channels = int(lpddr["logical_channels"])
    channel_index = int(lpddr.get("actual_trace_channel_index", 0))
    ple_data_bytes = int(geometry["ple_data_bytes"])
    ple_row_bytes = int(geometry["ple_row_bytes"])
    expert_useful = int(geometry["expert_useful_bytes"])
    expert_stride = int(geometry["expert_storage_stride_bytes"])
    experts_per_layer = int(geometry["experts_per_layer"])

    ordered = records[start_record:] + records[:start_record]
    while True:
        emitted_before = False
        for row in ordered:
            op = row.get("operation")
            if stream_kind == "read" and op == "logical_read":
                kind = str(row.get("kind"))
                if kind == "ple":
                    for object_id in row.get("object_ids", []):
                        object_id = int(object_id)
                        row_base = object_id * ple_row_bytes
                        row_end = row_base + ple_row_bytes
                        first_page = row_base // int(geometry["storage_page_bytes"])
                        last_page = (row_end - 1) // int(geometry["storage_page_bytes"])
                        for page in range(first_page, last_page + 1):
                            page_base = page * int(geometry["storage_page_bytes"])
                            overlap_start = max(row_base, page_base)
                            overlap_end = min(row_end, page_base + int(geometry["storage_page_bytes"]))
                            addresses = list(_mapped_access_addresses(
                                layout,
                                ("ple_page", page),
                                object_base=page_base,
                                object_span_bytes=int(geometry["storage_page_bytes"]),
                                access_base=overlap_start,
                                access_length_bytes=overlap_end - overlap_start,
                                tx_bytes=tx,
                                channels=channels,
                                channel_index=channel_index,
                            ))
                            if addresses:
                                emitted_before = True
                                yield from addresses
                elif kind == "expert":
                    layer = int(row.get("layer"))
                    for expert in row.get("object_ids", []):
                        expert = int(expert)
                        index = layer * experts_per_layer + expert
                        object_base = ple_data_bytes + index * expert_stride
                        addresses = list(_mapped_access_addresses(
                            layout,
                            ("expert", layer, expert),
                            object_base=object_base,
                            object_span_bytes=expert_stride,
                            access_base=object_base,
                            access_length_bytes=expert_useful,
                            tx_bytes=tx,
                            channels=channels,
                            channel_index=channel_index,
                        ))
                        if addresses:
                            emitted_before = True
                            yield from addresses
                else:
                    raise H3Error(f"unknown actual H3 logical read kind {kind}")
            elif stream_kind == "fill" and op == "fill_write":
                for fill in row.get("fills", []):
                    kind = str(fill.get("kind"))
                    storage_key = tuple(
                        fill.get("storage_key")
                        or [kind, int(fill["offset_bytes"]), int(fill["length_bytes"])]
                    )
                    length = int(fill["length_bytes"])
                    if kind == "ple":
                        if len(storage_key) != 2 or storage_key[0] != "ple_page":
                            raise H3Error("PLE fill lacks canonical ple_page storage key")
                        object_base = int(fill["offset_bytes"])
                        object_span = int(geometry["storage_page_bytes"])
                    elif kind == "expert":
                        if len(storage_key) != 3 or storage_key[0] != "expert":
                            raise H3Error("expert fill lacks canonical expert storage key")
                        object_base = ple_data_bytes + int(fill["offset_bytes"])
                        object_span = expert_stride
                    else:
                        raise H3Error(f"unknown actual H3 fill kind {kind}")
                    addresses = list(_mapped_access_addresses(
                        layout,
                        storage_key,
                        object_base=object_base,
                        object_span_bytes=object_span,
                        access_base=object_base,
                        access_length_bytes=length,
                        tx_bytes=tx,
                        channels=channels,
                        channel_index=channel_index,
                    ))
                    if addresses:
                        emitted_before = True
                        yield from addresses
        if not emitted_before:
            raise H3Error(f"actual H3 logical trace contains no {stream_kind} transactions")


def _write_actual_h3_transaction_trace_with_stats(
    path: Path,
    residency: dict[str, Any],
    contract: dict[str, Any],
    org: dict[str, Any],
    operation_mode: str,
    mapping_strategy: str = "hashed_bank_spread",
) -> tuple[int, dict[str, Any]]:
    if operation_mode not in {"read", "write", "mixed"}:
        raise H3Error(f"unknown actual H3 operation mode {operation_mode}")
    records = _logical_trace_records(residency)
    lpddr = contract["lpddr"]
    target = int(lpddr["actual_trace_sample_transactions"])
    if target <= 0:
        raise H3Error("actual H3 LPDDR trace target must be positive")
    seed = int(lpddr.get("actual_trace_seed", 20260906))
    start = stable_u64(
        seed,
        residency.get("stratum"),
        residency.get("policy"),
        residency.get("capacity_gib"),
        residency.get("phase_filter"),
        "common-source-start",
    ) % len(records)
    layout = _NoAliasChannelLayout(
        _capacity_bytes(org),
        int(lpddr["request_bytes"]),
        stable_u64(
            seed,
            residency.get("stratum"),
            residency.get("policy"),
            residency.get("capacity_gib"),
            residency.get("phase_filter"),
            mapping_strategy,
        ),
        strategy=mapping_strategy,
    )
    reads = _actual_object_transactions(
        records, contract, org, layout, stream_kind="read", start_record=start
    )
    fills = _actual_object_transactions(
        records, contract, org, layout, stream_kind="fill", start_record=start
    )
    counters = residency.get("counters") or {}
    read_bytes = float(counters.get("lpddr_read_useful_bytes", 0))
    fill_bytes = float(counters.get("lpddr_fill_write_bytes", 0))
    if read_bytes <= 0:
        raise H3Error("actual H3 trace has no logical read bytes")
    fill_fraction = fill_bytes / (read_bytes + fill_bytes) if fill_bytes > 0 else 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        accumulator = 0.0
        for _ in range(target):
            if operation_mode == "read":
                stream.write(f"LD {next(reads)}\n")
            elif operation_mode == "write":
                stream.write(f"ST {next(fills)}\n")
            else:
                accumulator += fill_fraction
                if fill_bytes > 0 and accumulator >= 1.0:
                    stream.write(f"ST {next(fills)}\n")
                    accumulator -= 1.0
                else:
                    stream.write(f"LD {next(reads)}\n")
    stats = layout.stats()
    stats.update({
        "sample_transactions": target,
        "representative_channel_index": int(lpddr.get("actual_trace_channel_index", 0)),
        "logical_channels": int(lpddr["logical_channels"]),
        "source_start_record": int(start),
        "mapping_strategy": str(mapping_strategy),
    })
    return target, stats


def write_actual_h3_transaction_trace(
    path: Path,
    residency: dict[str, Any],
    contract: dict[str, Any],
    org: dict[str, Any],
    operation_mode: str,
    mapping_strategy: str = "hashed_bank_spread",
) -> int:
    """Write a bounded actual-H3 trace using collision-free sample-local placement."""
    count, _ = _write_actual_h3_transaction_trace_with_stats(
        path, residency, contract, org, operation_mode, mapping_strategy
    )
    return count

def run_actual_h3_trace_point(
    lpddr: dict[str, Any],
    contract: dict[str, Any],
    residency: dict[str, Any],
    ncs: int,
    operation_mode: str,
    mapping_strategy: str = "hashed_bank_spread",
) -> dict[str, Any]:
    """Replay a bounded actual-H3 policy trace through pinned Ramulator2."""
    import ramulator  # type: ignore

    dram, org, timing = _dram(lpddr, ncs)
    with tempfile.TemporaryDirectory(prefix="fenix-h3-actual-ramulator-") as tmp:
        trace = Path(tmp) / f"actual-{operation_mode}.trace"
        count, mapping_stats = _write_actual_h3_transaction_trace_with_stats(
            trace, residency, contract, org, operation_mode, mapping_strategy
        )
        trace_sha = sha256_file(trace)
        frontend = ramulator.frontend.LoadStoreTrace(
            clock_ratio=int(lpddr["frontend_clock_ratio"]), path=str(trace)
        )
        mem = _memory_system(dram, ramulator.addr_mapper.RoBaRaCoCh(), lpddr)
        sim = ramulator.Simulation(frontend, mem)
        sim.run()
        result = _finish_stats(sim.stats, timing, count, int(lpddr["request_bytes"]))
    peak = theoretical_channel_gb_s(lpddr)
    if result["bandwidth_gb_s"] > peak * 1.02:
        raise H3Error("actual H3 Ramulator throughput exceeds x16 theoretical peak")
    return {
        "nCS": int(ncs),
        "regime": "actual_h3_policy_trace",
        "operation_mode": operation_mode,
        "resolved_org": org,
        "resolved_timing": timing,
        "trace_sha256": trace_sha,
        "address_mapping": "injective_sample_local_object_pages_then_RoBaRaCoCh",
        "address_mapping_strategy": str(mapping_strategy),
        "address_mapping_validation": mapping_stats,
        "trace_semantics": "actual H3 object IDs/order with collision-free representative-channel embedding; finite mixed trace uses measured read/fill byte ratio; mapping sensitivity is explicit",
        **result,
    }
