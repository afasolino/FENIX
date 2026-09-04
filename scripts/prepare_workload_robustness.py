#!/usr/bin/env python3
"""Prepare the frozen natural-workload corpus for FENIX H1/H2 robustness.

This step is CPU/network only.  It resolves every declared Hugging Face
revision to an immutable commit SHA before reading rows, then freezes the
selected prompt material and source metadata under external/workloads/.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

DEFAULT_CONTRACT = Path("configs/h1_h2_workload_robustness_v1.json")
DEFAULT_OUT_ROOT = Path("external/workloads/h1_h2_workload_robustness_v1")


class WorkloadPreparationError(RuntimeError):
    """Raised when the declared workload cannot be prepared reproducibly."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise WorkloadPreparationError("unsupported robustness contract schema")
    if payload.get("artifact_kind") != "fenix_h1_h2_workload_robustness_contract":
        raise WorkloadPreparationError("unexpected robustness contract kind")
    for field in ("data_sources", "strata", "sampling", "trace", "h1", "h2"):
        if not isinstance(payload.get(field), dict):
            raise WorkloadPreparationError(f"contract field is missing: {field}")
    order = payload["trace"].get("strata_order")
    if not isinstance(order, list) or not order:
        raise WorkloadPreparationError("trace.strata_order must be a non-empty list")
    if set(order) != set(payload["strata"]):
        raise WorkloadPreparationError("trace.strata_order and strata keys differ")
    return payload


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _choice_prompt(question: object, choices: Iterable[object]) -> str:
    q = _clean_text(question)
    rendered = []
    for index, choice in enumerate(choices):
        rendered.append(f"{chr(ord('A') + index)}. {_clean_text(choice)}")
    return q + ("\n" + "\n".join(rendered) if rendered else "")


def _wildchat_turns(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("conversation")
    if not isinstance(raw, list):
        return []
    turns: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        content = _clean_text(item.get("content"))
        role = _clean_text(item.get("role")).lower()
        if not content or role not in {"user", "assistant"}:
            continue
        turns.append(
            {
                "role": role,
                "content": content,
                "language": _clean_text(item.get("language")),
                "toxic": bool(item.get("toxic", False)),
                "redacted": bool(item.get("redacted", False)),
            }
        )
    return turns


def _language_is_english(value: str) -> bool:
    normalized = value.strip().lower().replace("_", "-")
    return normalized in {"en", "eng", "english", "en-us", "en-gb"} or normalized.startswith("en-")


def _row_identity(row: Mapping[str, Any], fallback_text: str) -> str:
    for key in (
        "conversation_hash",
        "turn_identifier",
        "unique_id",
        "task_id",
        "_id",
        "id",
        "question_id",
    ):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return sha256_text(fallback_text)


def _sample_stream(
    dataset: Iterable[Mapping[str, Any]],
    *,
    eligible: Callable[[Mapping[str, Any]], bool],
    count: int,
    scan_limit: int,
) -> list[Mapping[str, Any]]:
    if count < 1:
        raise WorkloadPreparationError("sample count must be positive")
    selected: list[Mapping[str, Any]] = []
    scanned = 0
    for row in dataset:
        scanned += 1
        if eligible(row):
            selected.append(row)
            if len(selected) == count:
                return selected
        if scanned >= scan_limit:
            break
    raise WorkloadPreparationError(
        f"eligible source rows exhausted: selected={len(selected)} required={count} scanned={scanned}"
    )


def _balanced_sample(
    dataset: Iterable[Mapping[str, Any]],
    *,
    group_key: Callable[[Mapping[str, Any]], str],
    eligible: Callable[[Mapping[str, Any]], bool],
    count: int,
    scan_limit: int,
) -> list[Mapping[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    scanned = 0
    target_per_group = max(2, count)
    for row in dataset:
        scanned += 1
        if eligible(row):
            key = group_key(row) or "unknown"
            if len(buckets[key]) < target_per_group:
                buckets[key].append(row)
        if scanned >= scan_limit:
            break
    keys = sorted(key for key, values in buckets.items() if values)
    if not keys:
        raise WorkloadPreparationError("balanced source produced no eligible groups")
    selected: list[Mapping[str, Any]] = []
    cursor = 0
    while len(selected) < count:
        progress = False
        for key in keys:
            values = buckets[key]
            if cursor < len(values):
                selected.append(values[cursor])
                progress = True
                if len(selected) == count:
                    return selected
        if not progress:
            break
        cursor += 1
    raise WorkloadPreparationError(
        f"balanced source cannot supply requested rows: selected={len(selected)} required={count}"
    )


def _render_history(turns: list[dict[str, Any]], end_index: int) -> str:
    lines = []
    for turn in turns[: end_index + 1]:
        label = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{label}: {turn['content']}")
    return "\n\n".join(lines)


def _source_iterator(
    load_dataset,
    *,
    source: Mapping[str, Any],
    revision_sha: str,
    seed: int,
    shuffle_buffer_size: int,
    config: str | None = None,
):
    kwargs: dict[str, Any] = {
        "path": str(source["repo_id"]),
        "split": str(source["split"]),
        "streaming": True,
        "revision": revision_sha,
    }
    resolved_config = config if config is not None else source.get("config")
    if resolved_config:
        kwargs["name"] = str(resolved_config)
    dataset = load_dataset(**kwargs)
    # Buffer shuffle is used only to avoid source-order/time clustering.  The
    # selected corpus is frozen and hashed, so trace reproduction never
    # depends on re-running this shuffle.
    return dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)


def prepare(
    contract_path: Path,
    out_root: Path,
) -> dict[str, Any]:
    try:
        from datasets import get_dataset_config_names, load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise WorkloadPreparationError(
            "workload preparation requires the pinned workload-tools environment"
        ) from exc

    contract = load_contract(contract_path)
    seed = int(contract["seed"])
    sampling = contract["sampling"]
    buffer_size = int(sampling["shuffle_buffer_size"])
    scan_limit = int(sampling["max_rows_scanned_per_source"])
    sources = contract["data_sources"]
    strata = contract["strata"]

    api = HfApi()
    resolved: dict[str, dict[str, Any]] = {}
    for source_name, source in sources.items():
        info = api.dataset_info(
            repo_id=str(source["repo_id"]),
            revision=str(source.get("revision", "main")),
        )
        if not info.sha:
            raise WorkloadPreparationError(f"cannot resolve immutable SHA for {source_name}")
        resolved[source_name] = {
            "repo_id": str(source["repo_id"]),
            "declared_revision": str(source.get("revision", "main")),
            "resolved_revision": str(info.sha),
            "split": str(source["split"]),
            "config": source.get("config"),
            "license": source.get("license"),
            "paper": source.get("paper"),
            "origin": source.get("origin"),
        }

    records: list[dict[str, Any]] = []
    used_wildchat_ids: set[str] = set()

    # Independent natural English chat: select single-turn conversations so the
    # prompt is self-contained rather than an orphaned later turn.
    spec = strata["chat_en"]
    src = sources[spec["source"]]
    ds = _source_iterator(
        load_dataset,
        source=src,
        revision_sha=resolved[spec["source"]]["resolved_revision"],
        seed=seed + 11,
        shuffle_buffer_size=buffer_size,
    )

    def chat_ok(row: Mapping[str, Any]) -> bool:
        turns = _wildchat_turns(row)
        users = [turn for turn in turns if turn["role"] == "user"]
        if len(users) != 1:
            return False
        turn = users[0]
        if turn["toxic"] or turn["redacted"] or not _language_is_english(turn["language"]):
            return False
        n = len(turn["content"])
        return int(spec["min_characters"]) <= n <= int(spec["max_characters"])

    chat_rows = _sample_stream(ds, eligible=chat_ok, count=int(spec["requests"]), scan_limit=scan_limit)
    for ordinal, row in enumerate(chat_rows):
        turn = next(turn for turn in _wildchat_turns(row) if turn["role"] == "user")
        source_id = _row_identity(row, turn["content"])
        used_wildchat_ids.add(source_id)
        records.append(
            {
                "sample_id": f"chat_en:{ordinal:04d}",
                "stratum": "chat_en",
                "ordinal": ordinal,
                "source": spec["source"],
                "source_id": source_id,
                "source_revision": resolved[spec["source"]]["resolved_revision"],
                "language": "en",
                "render_mode": "native",
                "prompt": turn["content"],
            }
        )

    # Knowledge / MMLU-Pro, balanced over categories.
    spec = strata["knowledge"]
    src = sources[spec["source"]]
    ds = _source_iterator(
        load_dataset,
        source=src,
        revision_sha=resolved[spec["source"]]["resolved_revision"],
        seed=seed + 23,
        shuffle_buffer_size=buffer_size,
    )

    def mmlu_prompt(row: Mapping[str, Any]) -> str:
        options = row.get("options")
        if not isinstance(options, list):
            options = [row.get(key) for key in ("A", "B", "C", "D") if row.get(key) is not None]
        return _choice_prompt(row.get("question"), options)

    knowledge_rows = _balanced_sample(
        ds,
        group_key=lambda row: _clean_text(row.get("category") or row.get("subject")),
        eligible=lambda row: bool(_clean_text(row.get("question"))),
        count=int(spec["requests"]),
        scan_limit=scan_limit,
    )
    for ordinal, row in enumerate(knowledge_rows):
        prompt = mmlu_prompt(row)
        records.append(
            {
                "sample_id": f"knowledge:{ordinal:04d}",
                "stratum": "knowledge",
                "ordinal": ordinal,
                "source": spec["source"],
                "source_id": _row_identity(row, prompt),
                "source_revision": resolved[spec["source"]]["resolved_revision"],
                "group": _clean_text(row.get("category") or row.get("subject")),
                "language": "en",
                "render_mode": "native",
                "prompt": prompt,
            }
        )

    # Mathematical reasoning, balanced over subject and level.
    spec = strata["math"]
    src = sources[spec["source"]]
    ds = _source_iterator(
        load_dataset,
        source=src,
        revision_sha=resolved[spec["source"]]["resolved_revision"],
        seed=seed + 37,
        shuffle_buffer_size=buffer_size,
    )
    math_rows = _balanced_sample(
        ds,
        group_key=lambda row: f"{_clean_text(row.get('subject'))}:{_clean_text(row.get('level'))}",
        eligible=lambda row: bool(_clean_text(row.get("problem"))),
        count=int(spec["requests"]),
        scan_limit=scan_limit,
    )
    for ordinal, row in enumerate(math_rows):
        prompt = _clean_text(row.get("problem"))
        records.append(
            {
                "sample_id": f"math:{ordinal:04d}",
                "stratum": "math",
                "ordinal": ordinal,
                "source": spec["source"],
                "source_id": _row_identity(row, prompt),
                "source_revision": resolved[spec["source"]]["resolved_revision"],
                "group": f"{_clean_text(row.get('subject'))}:{_clean_text(row.get('level'))}",
                "language": "en",
                "render_mode": "native",
                "prompt": prompt,
            }
        )

    # Code: keep the native HumanEval function/docstring prompt without adding
    # a repeated instruction template.
    spec = strata["code"]
    src = sources[spec["source"]]
    ds = _source_iterator(
        load_dataset,
        source=src,
        revision_sha=resolved[spec["source"]]["resolved_revision"],
        seed=seed + 41,
        shuffle_buffer_size=buffer_size,
    )
    code_rows = _sample_stream(
        ds,
        eligible=lambda row: bool(_clean_text(row.get("prompt"))),
        count=int(spec["requests"]),
        scan_limit=scan_limit,
    )
    for ordinal, row in enumerate(code_rows):
        prompt = str(row.get("prompt"))
        records.append(
            {
                "sample_id": f"code:{ordinal:04d}",
                "stratum": "code",
                "ordinal": ordinal,
                "source": spec["source"],
                "source_id": _row_identity(row, prompt),
                "source_revision": resolved[spec["source"]]["resolved_revision"],
                "language": "en",
                "render_mode": "native",
                "prompt": prompt,
            }
        )

    # Multilingual MMLU: fixed locales and equal sample count per locale.
    spec = strata["multilingual"]
    src = sources[spec["source"]]
    available = set(
        get_dataset_config_names(
            str(src["repo_id"]),
            revision=resolved[spec["source"]]["resolved_revision"],
        )
    )
    output_ordinal = 0
    for locale_index, locale in enumerate(spec["locales"]):
        if locale not in available:
            raise WorkloadPreparationError(
                f"MMMLU locale {locale!r} missing at pinned revision; available={sorted(available)}"
            )
        ds = _source_iterator(
            load_dataset,
            source=src,
            config=locale,
            revision_sha=resolved[spec["source"]]["resolved_revision"],
            seed=seed + 100 + locale_index,
            shuffle_buffer_size=buffer_size,
        )
        rows = _balanced_sample(
            ds,
            group_key=lambda row: _clean_text(row.get("Subject") or row.get("subject")),
            eligible=lambda row: bool(_clean_text(row.get("Question") or row.get("question"))),
            count=int(spec["requests_per_locale"]),
            scan_limit=scan_limit,
        )
        for row in rows:
            question = row.get("Question") if row.get("Question") is not None else row.get("question")
            choices = [
                row.get("A"),
                row.get("B"),
                row.get("C"),
                row.get("D"),
            ]
            prompt = _choice_prompt(question, choices)
            records.append(
                {
                    "sample_id": f"multilingual:{output_ordinal:04d}",
                    "stratum": "multilingual",
                    "ordinal": output_ordinal,
                    "source": spec["source"],
                    "source_id": f"{locale}:{_row_identity(row, prompt)}",
                    "source_revision": resolved[spec["source"]]["resolved_revision"],
                    "group": _clean_text(row.get("Subject") or row.get("subject")),
                    "language": locale,
                    "render_mode": "native",
                    "prompt": prompt,
                }
            )
            output_ordinal += 1

    # Session-locality stratum: accumulated real conversation history, grouped
    # by session and kept in chronological user-turn order.
    spec = strata["session"]
    src = sources[spec["source"]]
    ds = _source_iterator(
        load_dataset,
        source=src,
        revision_sha=resolved[spec["source"]]["resolved_revision"],
        seed=seed + 59,
        shuffle_buffer_size=buffer_size,
    )

    def session_ok(row: Mapping[str, Any]) -> bool:
        turns = _wildchat_turns(row)
        users = [turn for turn in turns if turn["role"] == "user"]
        if len(users) < int(spec["turns_per_session"]):
            return False
        if any(turn["toxic"] or turn["redacted"] for turn in turns):
            return False
        first_language = next((turn["language"] for turn in turns if turn["role"] == "user"), "")
        return _language_is_english(first_language)

    session_rows = []
    scanned = 0
    for row in ds:
        scanned += 1
        turns = _wildchat_turns(row)
        if turns and session_ok(row):
            source_id = _row_identity(row, "\n".join(turn["content"] for turn in turns))
            if source_id not in used_wildchat_ids:
                session_rows.append(row)
                if len(session_rows) == int(spec["sessions"]):
                    break
        if scanned >= scan_limit:
            break
    if len(session_rows) != int(spec["sessions"]):
        raise WorkloadPreparationError("could not select the declared number of WildChat sessions")

    session_ordinal = 0
    for session_index, row in enumerate(session_rows):
        turns = _wildchat_turns(row)
        user_indices = [i for i, turn in enumerate(turns) if turn["role"] == "user"]
        source_id = _row_identity(row, "\n".join(turn["content"] for turn in turns))
        for turn_number, end_index in enumerate(user_indices[: int(spec["turns_per_session"])]):
            prompt = _render_history(turns, end_index)
            records.append(
                {
                    "sample_id": f"session:{session_ordinal:04d}",
                    "stratum": "session",
                    "ordinal": session_ordinal,
                    "source": spec["source"],
                    "source_id": source_id,
                    "source_revision": resolved[spec["source"]]["resolved_revision"],
                    "language": "en",
                    "session_id": f"wildchat-session-{session_index:02d}-{source_id[:12]}",
                    "turn_index": turn_number,
                    "render_mode": "session_suffix_fit",
                    "prompt": prompt,
                }
            )
            session_ordinal += 1

    # LongBench v2: preserve natural context and the original question/options.
    # The trace runner trims only the context prefix using the live tokenizer.
    spec = strata["long_context_8k"]
    src = sources[spec["source"]]
    ds = _source_iterator(
        load_dataset,
        source=src,
        revision_sha=resolved[spec["source"]]["resolved_revision"],
        seed=seed + 71,
        shuffle_buffer_size=buffer_size,
    )
    long_rows = _balanced_sample(
        ds,
        group_key=lambda row: _clean_text(row.get("domain")),
        eligible=lambda row: bool(_clean_text(row.get("context"))) and bool(_clean_text(row.get("question"))),
        count=int(spec["requests"]),
        scan_limit=scan_limit,
    )
    cap = int(spec["context_character_cap"])
    for ordinal, row in enumerate(long_rows):
        context = _clean_text(row.get("context"))[:cap]
        question = _clean_text(row.get("question"))
        choices = [row.get(f"choice_{letter}") for letter in "ABCD"]
        suffix = _choice_prompt(question, choices)
        source_text = context + "\n\n" + suffix
        records.append(
            {
                "sample_id": f"long_context_8k:{ordinal:04d}",
                "stratum": "long_context_8k",
                "ordinal": ordinal,
                "source": spec["source"],
                "source_id": _row_identity(row, source_text),
                "source_revision": resolved[spec["source"]]["resolved_revision"],
                "group": _clean_text(row.get("domain")),
                "sub_group": _clean_text(row.get("sub_domain")),
                "language": "en",
                "render_mode": "long_context_prefix_fit",
                "context": context,
                "suffix": suffix,
            }
        )

    expected_counts = {name: int(spec["requests"]) for name, spec in strata.items()}
    observed_counts: dict[str, int] = defaultdict(int)
    for record in records:
        observed_counts[str(record["stratum"])] += 1
    if dict(observed_counts) != expected_counts:
        raise WorkloadPreparationError(
            f"prepared stratum counts differ: observed={dict(observed_counts)} expected={expected_counts}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    corpus_path = out_root / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 1,
        "artifact_kind": "fenix_h1_h2_frozen_workload_corpus",
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "selection_seed": seed,
        "source_revisions": resolved,
        "tool_environment": {
            "python": platform.python_version(),
            "datasets": importlib.metadata.version("datasets"),
            "huggingface_hub": importlib.metadata.version("huggingface_hub"),
            "hf_xet": importlib.metadata.version("hf_xet"),
        },
        "selection_rule": sampling["selection_rule"],
        "stratum_counts": expected_counts,
        "record_count": len(records),
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
    }
    manifest_path = out_root / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()
    try:
        manifest = prepare(args.contract, args.out_root)
    except (WorkloadPreparationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
