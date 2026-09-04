#!/usr/bin/env python3
"""Validate Qwen3.8 MoE hotness at request, phase, window, and cache timescales."""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from analysis.expert_locality import parse_layer_id
from analysis.process_ple_trace import load_jsonl


class MoEHotnessError(ValueError):
    pass


ExpertKey = tuple[int, int]


@dataclass
class RequestTrace:
    request_id: str
    stratum: str
    ordinal: int
    model_tokens: int
    metadata: dict[str, Any]
    layers: dict[int, list[tuple[int, ...]]]
    phase_layers: dict[str, dict[int, list[tuple[int, ...]]]]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise MoEHotnessError(f"{path}: expected JSON object")
    return value


def _geometry(model_campaign_path: Path) -> tuple[int, int, int, int]:
    payload = _load_json(model_campaign_path)
    model = payload.get("model")
    if not isinstance(model, dict):
        raise MoEHotnessError("model campaign geometry missing")
    return (
        int(model["num_hidden_layers"]),
        int(model["num_experts"]),
        int(model["experts_per_token"]),
        int(model["ple_addressable_rows"]),
    )


def _prompt_metadata(case_dir: Path) -> dict[int, dict[str, Any]]:
    payload = _load_json(case_dir / "prompts.json")
    rows = payload.get("prompts")
    if not isinstance(rows, list):
        raise MoEHotnessError(f"{case_dir}: prompts.json invalid")
    return {int(row["ordinal"]): dict(row) for row in rows}


def _uniform_occupancy(experts: int, topk: int, tokens: int) -> tuple[float, float]:
    if tokens <= 0:
        return 0.0, 0.0
    a = 1.0 - topk / experts
    b = ((experts - topk) * (experts - topk - 1)) / (experts * (experts - 1))
    a_t = a ** tokens
    b_t = b ** tokens
    p = 1.0 - a_t
    p_both = 1.0 - 2.0 * a_t + b_t
    expected = experts * p
    var = experts * p * (1.0 - p) + experts * (experts - 1) * (p_both - p * p)
    return expected, math.sqrt(max(0.0, var))


def _entropy_bits(counter: Mapping[int, int]) -> float | None:
    total = sum(int(v) for v in counter.values())
    if total <= 0:
        return None
    out = 0.0
    for value in counter.values():
        p = int(value) / total
        if p:
            out -= p * math.log2(p)
    return out


def _concentration(counter: Mapping[int, int], topk_values: Sequence[int]) -> dict[str, float | None]:
    total = sum(int(v) for v in counter.values())
    ordered = sorted((int(v) for v in counter.values()), reverse=True)
    return {
        str(k): (sum(ordered[:k]) / total if total else None)
        for k in topk_values
    }


def _counter_from_tokens(tokens: Sequence[tuple[int, ...]]) -> collections.Counter[int]:
    c: collections.Counter[int] = collections.Counter()
    for token in tokens:
        c.update(token)
    return c


def _sparse_cosine(left: Mapping[Any, int], right: Mapping[Any, int]) -> float | None:
    ln = math.sqrt(sum(int(v) ** 2 for v in left.values()))
    rn = math.sqrt(sum(int(v) ** 2 for v in right.values()))
    if not ln or not rn:
        return None
    if len(left) > len(right):
        left, right = right, left
    dot = sum(int(v) * int(right.get(k, 0)) for k, v in left.items())
    return dot / (ln * rn)


def _js(left: Mapping[Any, int], right: Mapping[Any, int]) -> float | None:
    ls = sum(int(v) for v in left.values())
    rs = sum(int(v) for v in right.values())
    if not ls or not rs:
        return None
    out = 0.0
    for key in set(left) | set(right):
        p = int(left.get(key, 0)) / ls
        q = int(right.get(key, 0)) / rs
        m = 0.5 * (p + q)
        if p:
            out += 0.5 * p * math.log2(p / m)
        if q:
            out += 0.5 * q * math.log2(q / m)
    return out


def _starts(length: int, window: int, maximum: int) -> list[int]:
    if window > length:
        return []
    count = length - window + 1
    if count <= maximum:
        return list(range(count))
    if maximum <= 1:
        return [0]
    hi = count - 1
    return sorted({round(i * hi / (maximum - 1)) for i in range(maximum)})


def _summary(values: Sequence[float | int | None]) -> dict[str, float | int | None]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(xs), "mean": mean(xs), "median": median(xs), "min": min(xs), "max": max(xs)}


def _eam(request: RequestTrace, phase: str = "all") -> collections.Counter[ExpertKey]:
    out: collections.Counter[ExpertKey] = collections.Counter()
    source = request.layers if phase == "all" else request.phase_layers.get(phase, {})
    for layer, tokens in source.items():
        for expert, count in _counter_from_tokens(tokens).items():
            out[(layer, expert)] += count
    return out


def load_case(
    case_dir: Path,
    *,
    num_layers: int,
    num_experts: int,
    experts_per_token: int,
    require_intrinsic: bool,
) -> list[RequestTrace]:
    evidence = _load_json(case_dir / "evidence.json")
    if evidence.get("trace_valid") is not True:
        raise MoEHotnessError(f"{case_dir.name}: trace_valid != true")
    case = evidence.get("case", {})
    if int(case.get("concurrency", 0)) != 1:
        raise MoEHotnessError("hotness validation requires concurrency=1")
    stratum = str(case["stratum"])

    clients = [r for r in load_jsonl(case_dir / "client.jsonl") if "error" not in r]
    clients.sort(key=lambda r: int(r["ordinal"]))
    client_by_id = {str(r["request_id"]): r for r in clients}
    request_ids = set(client_by_id)
    meta_by_ordinal = _prompt_metadata(case_dir)

    layers = {rid: {layer: [] for layer in range(num_layers)} for rid in request_ids}
    phase_layers = {
        rid: collections.defaultdict(lambda: {layer: [] for layer in range(num_layers)})
        for rid in request_ids
    }

    for event in load_jsonl(case_dir / "moe_normalized.jsonl"):
        rid = str(event["request_id"])
        if rid not in request_ids:
            raise MoEHotnessError(f"{case_dir.name}: MoE request outside client set")
        layer = parse_layer_id(event["layer"])
        selected = [int(v) for v in event.get("selected_expert_ids", [])]
        token_count = int(event.get("token_count", 0))
        if token_count <= 0:
            if len(selected) % experts_per_token:
                raise MoEHotnessError("cannot infer token_count")
            token_count = len(selected) // experts_per_token
        if len(selected) != token_count * experts_per_token:
            raise MoEHotnessError("selection width != token_count * experts_per_token")
        phase = str(event.get("phase", "unknown"))
        for index in range(token_count):
            token = tuple(selected[index * experts_per_token:(index + 1) * experts_per_token])
            if len(set(token)) != experts_per_token:
                raise MoEHotnessError("duplicate expert within one token")
            if any(not 0 <= expert < num_experts for expert in token):
                raise MoEHotnessError("expert outside geometry")
            layers[rid][layer].append(token)
            phase_layers[rid][phase][layer].append(token)

    out = []
    for client in clients:
        rid = str(client["request_id"])
        expected = int(client["prompt_tokens"]) + max(int(client["completion_tokens"]) - 1, 0)
        counts = [len(layers[rid][layer]) for layer in range(num_layers)]
        if min(counts) != max(counts):
            raise MoEHotnessError(f"{case_dir.name}:{rid}: layer token counts differ")
        observed = counts[0]
        if require_intrinsic and observed != expected:
            raise MoEHotnessError(f"{case_dir.name}:{rid}: intrinsic token mismatch {observed}!={expected}")
        ordinal = int(client["ordinal"])
        out.append(RequestTrace(
            request_id=rid,
            stratum=stratum,
            ordinal=ordinal,
            model_tokens=observed,
            metadata=dict(meta_by_ordinal.get(ordinal, {})),
            layers=layers[rid],
            phase_layers=dict(phase_layers[rid]),
        ))
    return out


def request_metrics(request: RequestTrace, *, num_experts: int, experts_per_token: int, topks: Sequence[int]) -> dict[str, Any]:
    unique, ratios, zscores, entropies, effective = [], [], [], [], []
    concentrations = {k: [] for k in topks}
    for tokens in request.layers.values():
        counter = _counter_from_tokens(tokens)
        observed = len(counter)
        expected, std = _uniform_occupancy(num_experts, experts_per_token, len(tokens))
        unique.append(observed)
        ratios.append(observed / expected if expected else 0.0)
        if std > 1e-12:
            zscores.append((observed - expected) / std)
        ent = _entropy_bits(counter)
        if ent is not None:
            entropies.append(ent / math.log2(num_experts))
            effective.append(2.0 ** ent)
        conc = _concentration(counter, topks)
        for k in topks:
            if conc[str(k)] is not None:
                concentrations[k].append(float(conc[str(k)]))
    return {
        "request_id": request.request_id,
        "stratum": request.stratum,
        "ordinal": request.ordinal,
        "model_tokens": request.model_tokens,
        "unique_experts_per_layer": _summary(unique),
        "uniform_expected_unique_experts_per_layer": _uniform_occupancy(num_experts, experts_per_token, request.model_tokens)[0],
        "observed_over_uniform_occupancy": _summary(ratios),
        "occupancy_zscore": _summary(zscores),
        "normalized_entropy": _summary(entropies),
        "effective_expert_count": _summary(effective),
        "concentration": {str(k): _summary(concentrations[k]) for k in topks},
    }


def rolling_metrics(
    requests: Sequence[RequestTrace],
    *,
    phases: Sequence[str],
    windows: Sequence[int],
    maximum_windows: int,
    num_experts: int,
    experts_per_token: int,
    topks: Sequence[int],
) -> list[dict[str, Any]]:
    rows = []
    for phase in phases:
        for window in windows:
            uniques, ratios, effective = [], [], []
            concentrations = {k: [] for k in topks}
            expected, _ = _uniform_occupancy(num_experts, experts_per_token, window)
            for request in requests:
                source = request.layers if phase == "all" else request.phase_layers.get(phase, {})
                for tokens in source.values():
                    for start in _starts(len(tokens), window, maximum_windows):
                        sample = tokens[start:start + window]
                        counter = _counter_from_tokens(sample)
                        uniques.append(len(counter))
                        ratios.append(len(counter) / expected if expected else 0.0)
                        ent = _entropy_bits(counter)
                        if ent is not None:
                            effective.append(2.0 ** ent)
                        conc = _concentration(counter, topks)
                        for k in topks:
                            if conc[str(k)] is not None:
                                concentrations[k].append(float(conc[str(k)]))
            if uniques:
                rows.append({
                    "phase": phase,
                    "window_tokens": window,
                    "uniform_expected_unique_experts": expected,
                    "observed_unique_experts": _summary(uniques),
                    "observed_over_uniform_occupancy": _summary(ratios),
                    "effective_expert_count": _summary(effective),
                    "concentration": {str(k): _summary(concentrations[k]) for k in topks},
                })
    return rows


def eam_similarity(requests: Sequence[RequestTrace]) -> dict[str, Any]:
    eams = {r.request_id: _eam(r) for r in requests}
    same_cos, same_js, cross_cos, cross_js = [], [], [], []
    matrix = collections.defaultdict(list)
    for i, left in enumerate(requests):
        for right in requests[i + 1:]:
            cos = _sparse_cosine(eams[left.request_id], eams[right.request_id])
            js = _js(eams[left.request_id], eams[right.request_id])
            if cos is None or js is None:
                continue
            pair = tuple(sorted((left.stratum, right.stratum)))
            matrix[pair].append((cos, js))
            if left.stratum == right.stratum:
                same_cos.append(cos); same_js.append(js)
            else:
                cross_cos.append(cos); cross_js.append(js)

    session_cos, session_js = [], []
    sessions = collections.defaultdict(list)
    for request in requests:
        sid = request.metadata.get("session_id")
        if sid is not None:
            sessions[str(sid)].append(request)
    for items in sessions.values():
        items.sort(key=lambda r: int(r.metadata.get("turn_index", r.ordinal)))
        for left, right in zip(items, items[1:]):
            cos = _sparse_cosine(eams[left.request_id], eams[right.request_id])
            js = _js(eams[left.request_id], eams[right.request_id])
            if cos is not None and js is not None:
                session_cos.append(cos); session_js.append(js)

    phase_cos, phase_js = [], []
    for request in requests:
        cos = _sparse_cosine(_eam(request, "prefill"), _eam(request, "decode"))
        js = _js(_eam(request, "prefill"), _eam(request, "decode"))
        if cos is not None and js is not None:
            phase_cos.append(cos); phase_js.append(js)

    return {
        "same_stratum": {"cosine": _summary(same_cos), "js": _summary(same_js)},
        "cross_stratum": {"cosine": _summary(cross_cos), "js": _summary(cross_js)},
        "same_session_consecutive": {"cosine": _summary(session_cos), "js": _summary(session_js)},
        "prefill_vs_decode": {"cosine": _summary(phase_cos), "js": _summary(phase_js)},
        "stratum_pair_matrix": [
            {"left": pair[0], "right": pair[1], "pairs": len(values),
             "cosine": _summary([v[0] for v in values]), "js": _summary([v[1] for v in values])}
            for pair, values in sorted(matrix.items())
        ],
    }


def _capacity_rows(contract: Mapping[str, Any], *, num_layers: int, slot_bytes: int) -> list[dict[str, Any]]:
    values = set(int(v) for v in contract["cache_replay"]["capacities_experts_per_layer"])
    for gib in contract["cache_replay"]["budget_gib"]:
        values.add(max(1, int(float(gib) * 1024**3) // (num_layers * slot_bytes)))
    return [
        {"experts_per_layer": cap,
         "total_expert_cache_bytes": cap * num_layers * slot_bytes,
         "total_expert_cache_gib": cap * num_layers * slot_bytes / 1024**3}
        for cap in sorted(values)
    ]


def _layer_frequency(requests: Sequence[RequestTrace], num_layers: int) -> list[collections.Counter[int]]:
    out = [collections.Counter() for _ in range(num_layers)]
    for request in requests:
        for layer in range(num_layers):
            out[layer].update(_counter_from_tokens(request.layers[layer]))
    return out


def _top_selected(counts: Sequence[collections.Counter[int]], capacity: int) -> list[set[int]]:
    return [{expert for expert, _ in counter.most_common(capacity)} for counter in counts]


def _simulate_static(requests: Sequence[RequestTrace], selected: Sequence[set[int]], *, num_layers: int) -> tuple[int, int]:
    resident = [set() for _ in range(num_layers)]
    hits = misses = 0
    for request in requests:
        for layer in range(num_layers):
            for token in request.layers[layer]:
                for expert in token:
                    if expert in selected[layer] and expert in resident[layer]:
                        hits += 1
                    else:
                        misses += 1
                        if expert in selected[layer]:
                            resident[layer].add(expert)
    return hits, misses


def _simulate_lru(requests: Sequence[RequestTrace], capacity: int, *, num_layers: int) -> tuple[int, int]:
    resident = [dict() for _ in range(num_layers)]
    hits = misses = epoch = 0
    for request in requests:
        for token_index in range(request.model_tokens):
            epoch += 1
            for layer in range(num_layers):
                token = request.layers[layer][token_index]
                cache = resident[layer]
                demanded = set(token)
                for expert in token:
                    if expert in cache: hits += 1
                    else: misses += 1
                for expert in demanded:
                    cache[expert] = epoch
                if len(cache) > capacity:
                    victims = sorted(cache, key=lambda e: (cache[e], e))
                    for expert in victims[:len(cache) - capacity]:
                        del cache[expert]
    return hits, misses


def _simulate_history_lfu(
    requests: Sequence[RequestTrace],
    capacity: int,
    *,
    num_layers: int,
    seed_history: Sequence[RequestTrace] = (),
) -> tuple[int, int]:
    history = _layer_frequency(seed_history, num_layers)
    resident = [set() for _ in range(num_layers)]
    hits = misses = 0
    for request in requests:
        selected = _top_selected(history, capacity)
        resident = [resident[layer] & selected[layer] for layer in range(num_layers)]
        for layer in range(num_layers):
            for token in request.layers[layer]:
                for expert in token:
                    if expert in selected[layer] and expert in resident[layer]:
                        hits += 1
                    else:
                        misses += 1
                        if expert in selected[layer]:
                            resident[layer].add(expert)
        for layer in range(num_layers):
            history[layer].update(_counter_from_tokens(request.layers[layer]))
    return hits, misses


def _simulate_belady(requests: Sequence[RequestTrace], capacity: int, *, num_layers: int) -> tuple[int, int]:
    hits = misses = 0
    for layer in range(num_layers):
        events = [set(token) for request in requests for token in request.layers[layer]]
        future = collections.defaultdict(collections.deque)
        for index, event in enumerate(events):
            for expert in event:
                future[expert].append(index)
        resident = set()
        inf = len(events) + 1
        for index, event in enumerate(events):
            for expert in event:
                q = future[expert]
                if not q or q[0] != index:
                    raise MoEHotnessError("Belady future-use index drift")
                q.popleft()
                if expert in resident: hits += 1
                else:
                    misses += 1
                    resident.add(expert)
            while len(resident) > capacity:
                victim = max(resident, key=lambda e: (future[e][0] if future[e] else inf, e))
                resident.remove(victim)
    return hits, misses


def _metric(policy: str, capacity: int, hits: int, misses: int, *, slot_bytes: int, num_layers: int) -> dict[str, Any]:
    total = hits + misses
    return {
        "policy": policy,
        "experts_per_layer": capacity,
        "total_cache_gib": capacity * num_layers * slot_bytes / 1024**3,
        "expert_selection_references": total,
        "hits": hits,
        "misses": misses,
        "selection_hit_rate": hits / total if total else None,
        "expert_object_transfer_bytes": misses * slot_bytes,
    }


def _round_robin(groups: Mapping[str, Sequence[RequestTrace]], names: Sequence[str]) -> list[RequestTrace]:
    sorted_groups = {name: sorted(groups[name], key=lambda r: r.ordinal) for name in names}
    out = []
    maximum = max(len(sorted_groups[name]) for name in names)
    for index in range(maximum):
        for name in names:
            if index < len(sorted_groups[name]):
                out.append(sorted_groups[name][index])
    return out


def cache_replay(requests: Sequence[RequestTrace], contract: Mapping[str, Any], *, num_layers: int, num_experts: int, slot_bytes: int) -> dict[str, Any]:
    domains = list(contract["cache_replay"]["domain_strata"])
    by_stratum = {name: [r for r in requests if r.stratum == name] for name in domains}
    if any(not by_stratum[name] for name in domains):
        raise MoEHotnessError("cache replay domain stratum missing")
    train_by_domain = {name: [r for r in by_stratum[name] if r.ordinal % 2 == 0] for name in domains}
    eval_by_domain = {name: [r for r in by_stratum[name] if r.ordinal % 2 == 1] for name in domains}
    global_train = [r for name in domains for r in train_by_domain[name]]
    global_eval = _round_robin(eval_by_domain, domains)

    output = []
    for caprow in _capacity_rows(contract, num_layers=num_layers, slot_bytes=slot_bytes):
        capacity = int(caprow["experts_per_layer"])
        global_selected = _top_selected(_layer_frequency(global_train, num_layers), capacity)
        h, m = _simulate_static(global_eval, global_selected, num_layers=num_layers)
        rows = [_metric("global_lfu_demand_fill", capacity, h, m, slot_bytes=slot_bytes, num_layers=num_layers)]

        rng = random.Random(20260904 + capacity)
        random_selected = [set(rng.sample(range(num_experts), min(capacity, num_experts))) for _ in range(num_layers)]
        h, m = _simulate_static(global_eval, random_selected, num_layers=num_layers)
        rows.append(_metric("random_static_demand_fill", capacity, h, m, slot_bytes=slot_bytes, num_layers=num_layers))

        h, m = _simulate_lru(global_eval, capacity, num_layers=num_layers)
        rows.append(_metric("token_atomic_lru", capacity, h, m, slot_bytes=slot_bytes, num_layers=num_layers))

        h, m = _simulate_history_lfu(global_eval, capacity, num_layers=num_layers, seed_history=global_train)
        rows.append(_metric("request_history_lfu_demand_fill", capacity, h, m, slot_bytes=slot_bytes, num_layers=num_layers))

        h, m = _simulate_belady(global_eval, capacity, num_layers=num_layers)
        rows.append(_metric("oracle_belady_token_atomic", capacity, h, m, slot_bytes=slot_bytes, num_layers=num_layers))

        in_domain_rates = []
        for name in domains:
            selected = _top_selected(_layer_frequency(train_by_domain[name], num_layers), capacity)
            h, m = _simulate_static(eval_by_domain[name], selected, num_layers=num_layers)
            in_domain_rates.append(h / (h + m) if h + m else 0.0)

        lodo_rates = []
        for held_out in domains:
            training = [r for name in domains if name != held_out for r in by_stratum[name]]
            selected = _top_selected(_layer_frequency(training, num_layers), capacity)
            h, m = _simulate_static(by_stratum[held_out], selected, num_layers=num_layers)
            lodo_rates.append(h / (h + m) if h + m else 0.0)

        output.append({
            **caprow,
            "mixed_holdout_policies": rows,
            "in_domain_lfu_selection_hit_rate": _summary(in_domain_rates),
            "leave_one_domain_out_lfu_selection_hit_rate": _summary(lodo_rates),
        })

    return {
        "evidence_kind": "trace_projection",
        "atomic_event": contract["cache_replay"]["atomic_event"],
        "first_touch": contract["cache_replay"]["first_touch"],
        "transfer_unit": contract["cache_replay"]["transfer_unit"],
        "interpretation": contract["cache_replay"]["interpretation"],
        "capacity_results": output,
    }


def analyze(trace_root: Path, contract_path: Path, model_campaign_path: Path) -> dict[str, Any]:
    contract = _load_json(contract_path)
    if contract.get("artifact_kind") != "fenix_moe_hotness_validation_contract":
        raise MoEHotnessError("unexpected hotness validation contract")
    num_layers, num_experts, experts_per_token, _ = _geometry(model_campaign_path)
    case_dirs = sorted(path for path in trace_root.glob("s-*-r01") if path.is_dir())
    if not case_dirs:
        raise MoEHotnessError("no robustness trace cases found")

    all_requests = []
    provenance = set()
    for case_dir in case_dirs:
        evidence = _load_json(case_dir / "evidence.json")
        provenance.add((evidence.get("repository_commit"), evidence.get("launch", {}).get("runtime_image_id"), evidence.get("frozen_corpus_sha256")))
        all_requests.extend(load_case(
            case_dir,
            num_layers=num_layers,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            require_intrinsic=True,
        ))
    if len(provenance) != 1:
        raise MoEHotnessError("cross-case provenance mismatch")

    topks = [int(v) for v in contract["request_level"]["topk"]]
    request_rows = [request_metrics(r, num_experts=num_experts, experts_per_token=experts_per_token, topks=topks) for r in all_requests]

    strata = {}
    for stratum in sorted({r.stratum for r in all_requests}):
        reqs = [r for r in all_requests if r.stratum == stratum]
        rows = [row for row in request_rows if row["stratum"] == stratum]
        strata[stratum] = {
            "requests": len(reqs),
            "model_tokens": sum(r.model_tokens for r in reqs),
            "request_mean_unique_experts_per_layer": _summary([row["unique_experts_per_layer"]["mean"] for row in rows]),
            "request_mean_observed_over_uniform_occupancy": _summary([row["observed_over_uniform_occupancy"]["mean"] for row in rows]),
            "request_mean_effective_expert_count": _summary([row["effective_expert_count"]["mean"] for row in rows]),
            "request_mean_concentration": {str(k): _summary([row["concentration"][str(k)]["mean"] for row in rows]) for k in topks},
            "rolling_windows": rolling_metrics(
                reqs,
                phases=contract["rolling_windows"]["phases"],
                windows=[int(v) for v in contract["rolling_windows"]["token_windows"]],
                maximum_windows=int(contract["rolling_windows"]["max_windows_per_sequence"]),
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                topks=[int(v) for v in contract["rolling_windows"]["topk"]],
            ),
        }

    h1_contract = _load_json(Path("configs/h1_h2_workload_robustness_v1.json"))
    slot_bytes = int(h1_contract["measured_geometry"]["expert_slot_bytes"])

    return {
        "schema_version": 1,
        "artifact_kind": "fenix_moe_hotness_validation",
        "evidence_kind": "local_measured_trace_analysis_plus_trace_projection",
        "hypothesis": contract["scientific_scope"]["hypothesis"],
        "trace_root": str(trace_root),
        "provenance_fingerprint": list(next(iter(provenance))),
        "geometry": {"num_layers": num_layers, "num_experts": num_experts, "experts_per_token": experts_per_token, "expert_slot_bytes": slot_bytes},
        "null_model": contract["model_null"],
        "request_count": len(all_requests),
        "request_metrics": request_rows,
        "strata": strata,
        "eam_similarity": eam_similarity(all_requests),
        "cache_replay": cache_replay(all_requests, contract, num_layers=num_layers, num_experts=num_experts, slot_bytes=slot_bytes),
        "interpretation_guard": {
            "global_union_is_not_hotness": True,
            "can_establish_h3": False,
            "cache_replay_is_expert_object_transfer_projection_only": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path("configs/moe_hotness_validation_v1.json"))
    parser.add_argument("--model-campaign", type=Path, default=Path("configs/campaign.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.trace_root, args.contract, args.model_campaign)
    except (MoEHotnessError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "artifact_kind": result["artifact_kind"],
        "request_count": result["request_count"],
        "output": str(args.out),
        "cache_capacities": [row["experts_per_layer"] for row in result["cache_replay"]["capacity_results"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
