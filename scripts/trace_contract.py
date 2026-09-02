#!/usr/bin/env python3
"""Typed workload contract for the FENIX trace-characterization campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts import workload_contract


DEFAULT_CAMPAIGN = Path("configs/campaign.json")
DEFAULT_EXPERIMENT = "trace_characterization"
TRACE_PROFILE = "trace_characterization_v1"
_FINE_FRAGMENTS = (
    " x", " a", " 0", " 1", " memory", " locality", " token", " data",
    " inference", ".", ",", ";", ":", "\n",
)


class TraceContractError(RuntimeError):
    """Raised when the trace campaign is absent or internally inconsistent."""


@dataclass(frozen=True)
class TraceContract:
    experiment: str
    workload_profile: str
    workload_corpus: Path
    seed: int
    requests_per_input_length: int
    input_tokens: tuple[int, ...]
    output_tokens: int
    temperature: float
    exact_concurrency: tuple[int, ...]
    aggregate_concurrency: tuple[int, ...]
    repetitions: int

    @property
    def concurrencies(self) -> tuple[int, ...]:
        return self.exact_concurrency + self.aggregate_concurrency


@dataclass(frozen=True)
class TraceCase:
    input_tokens: int
    concurrency: int
    repetition_index: int
    correlation_mode: str

    @property
    def case_id(self) -> str:
        return (
            f"i{self.input_tokens:06d}-c{self.concurrency:02d}-"
            f"r{self.repetition_index:02d}"
        )


@dataclass(frozen=True)
class PreparedTracePrompts:
    prompts: tuple[str, ...]
    prompt_hashes: tuple[str, ...]
    prompt_set_sha256: str
    prompt_tokens: int
    max_model_len: int | None
    tokenize_url: str


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TraceContractError(f"{field} must be a positive integer")
    return value


def _positive_int_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TraceContractError(f"{field} must be a non-empty list")
    result = tuple(_positive_int(item, field) for item in value)
    if len(set(result)) != len(result):
        raise TraceContractError(f"{field} contains duplicate values")
    return result


def load_trace_contract(
    campaign_path: Path = DEFAULT_CAMPAIGN,
    experiment: str = DEFAULT_EXPERIMENT,
) -> TraceContract:
    if not campaign_path.is_file():
        raise TraceContractError(
            f"campaign configuration does not exist: {campaign_path}"
        )
    payload = json.loads(campaign_path.read_text())
    try:
        raw = payload["experiments"][experiment]
    except (KeyError, TypeError) as exc:
        raise TraceContractError(
            f"campaign experiment is missing: {experiment}"
        ) from exc
    if not isinstance(raw, dict):
        raise TraceContractError(f"{experiment} must be an object")

    profile = raw.get("workload_profile")
    if profile != TRACE_PROFILE:
        raise TraceContractError(
            f"{experiment}.workload_profile must be {TRACE_PROFILE!r}"
        )
    temperature = raw.get("temperature")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TraceContractError(f"{experiment}.temperature must be numeric")

    exact = _positive_int_tuple(
        raw.get("exact_request_correlation_concurrency"),
        f"{experiment}.exact_request_correlation_concurrency",
    )
    aggregate = _positive_int_tuple(
        raw.get("aggregate_service_concurrency"),
        f"{experiment}.aggregate_service_concurrency",
    )
    if exact != (1,):
        raise TraceContractError(
            "exact request correlation is currently defined only for concurrency=1"
        )
    if set(exact) & set(aggregate):
        raise TraceContractError("exact and aggregate concurrency sets overlap")
    corpus = raw.get("workload_corpus")
    if not isinstance(corpus, str) or not corpus:
        raise TraceContractError(
            f"{experiment}.workload_corpus must name a versioned corpus file"
        )

    return TraceContract(
        experiment=experiment,
        workload_profile=profile,
        workload_corpus=Path(
            str(raw.get("workload_corpus", ""))
        ),
        seed=_positive_int(raw.get("seed"), f"{experiment}.seed"),
        requests_per_input_length=_positive_int(
            raw.get("requests_per_input_length"),
            f"{experiment}.requests_per_input_length",
        ),
        input_tokens=_positive_int_tuple(
            raw.get("input_tokens"), f"{experiment}.input_tokens"
        ),
        output_tokens=_positive_int(
            raw.get("output_tokens"), f"{experiment}.output_tokens"
        ),
        temperature=float(temperature),
        exact_concurrency=exact,
        aggregate_concurrency=aggregate,
        repetitions=_positive_int(
            raw.get("repetitions"), f"{experiment}.repetitions"
        ),
    )


def planned_cases(contract: TraceContract) -> tuple[TraceCase, ...]:
    cases: list[TraceCase] = []
    for repetition in range(1, contract.repetitions + 1):
        for input_tokens in contract.input_tokens:
            for concurrency in contract.concurrencies:
                mode = (
                    "exact_request_correlation"
                    if concurrency in contract.exact_concurrency
                    else "aggregate_service"
                )
                cases.append(
                    TraceCase(input_tokens, concurrency, repetition, mode)
                )
    return tuple(cases)


def load_prompt_corpus(contract: TraceContract) -> dict[str, tuple[str, ...]]:
    path = contract.workload_corpus
    if not path.is_file():
        raise TraceContractError(f"trace workload corpus does not exist: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise TraceContractError("trace workload corpus schema_version must be 1")
    if payload.get("profile") != contract.workload_profile:
        raise TraceContractError("trace workload corpus profile does not match campaign")
    seeds = payload.get("request_seeds")
    sentences = payload.get("continuation_sentences")
    if not isinstance(seeds, list) or not all(isinstance(x, str) and x.strip() for x in seeds):
        raise TraceContractError("trace workload corpus request_seeds are invalid")
    if len(seeds) != contract.requests_per_input_length:
        raise TraceContractError(
            "trace workload corpus seed count must equal requests_per_input_length"
        )
    if not isinstance(sentences, list) or not sentences or not all(
        isinstance(x, str) and x.strip() for x in sentences
    ):
        raise TraceContractError("trace workload corpus continuation_sentences are invalid")
    return {"request_seeds": tuple(seeds), "continuation_sentences": tuple(sentences)}


def _sentence_stream(
    *,
    contract: TraceContract,
    input_tokens: int,
    ordinal: int,
    sentences: tuple[str, ...],
    count: int,
) -> list[str]:
    output: list[str] = []
    for position in range(count):
        key = f"{contract.seed}:{input_tokens}:{ordinal}:{position}".encode()
        digest = hashlib.sha256(key).digest()
        output.append(sentences[int.from_bytes(digest[:4], "big") % len(sentences)])
    return output

def _build_exact_trace_prompt(
    *,
    contract: TraceContract,
    input_tokens: int,
    ordinal: int,
    request_seed: str,
    continuation_sentences: tuple[str, ...],
    token_count,
) -> str:
    header = (
        f"{request_seed.strip()} "
        "Continue with a careful systems analysis grounded in the supplied "
        "benchmark context."
    )
    header_count = token_count(header)
    if header_count > input_tokens:
        raise TraceContractError(
            f"trace prompt header exceeds target: {header_count} > {input_tokens}"
        )
    if header_count == input_tokens:
        return header

    sentences = _sentence_stream(
        contract=contract,
        input_tokens=input_tokens,
        ordinal=ordinal,
        sentences=continuation_sentences,
        count=max(128, input_tokens),
    )
    cache: dict[int, tuple[str, int]] = {0: (header, header_count)}

    def candidate(word_count: int) -> tuple[str, int]:
        if word_count not in cache:
            prompt = header + " " + " ".join(sentences[:word_count])
            cache[word_count] = (prompt, token_count(prompt))
        return cache[word_count]

    low = 0
    high = 1
    while high <= len(sentences) and candidate(high)[1] <= input_tokens:
        if candidate(high)[1] == input_tokens:
            return candidate(high)[0]
        low = high
        high *= 2
    high = min(high, len(sentences))
    if candidate(high)[1] <= input_tokens:
        raise TraceContractError(
            "deterministic trace corpus cannot bracket the target token count"
        )

    while low + 1 < high:
        middle = (low + high) // 2
        _, count = candidate(middle)
        if count <= input_tokens:
            low = middle
            if count == input_tokens:
                return candidate(middle)[0]
        else:
            high = middle

    prompt, current = candidate(low)
    fine_fragments = _FINE_FRAGMENTS + tuple(f" {value}" for value in range(2, 65))
    for _ in range(32):
        if current == input_tokens:
            return prompt
        best_prompt = None
        best_count = current
        for fragment in fine_fragments:
            trial = prompt + fragment
            trial_count = token_count(trial)
            if trial_count == input_tokens:
                return trial
            if current < trial_count < input_tokens and trial_count > best_count:
                best_prompt = trial
                best_count = trial_count
        if best_prompt is None:
            raise TraceContractError(
                "exact trace prompt construction stalled at "
                f"{current}/{input_tokens} rendered tokens"
            )
        prompt, current = best_prompt, best_count

    raise TraceContractError(
        "exact trace prompt construction exceeded the bounded fine-search budget"
    )


def prepare_trace_prompts(
    *,
    contract: TraceContract,
    input_tokens: int,
    chat_url: str,
    model: str,
    tokenize_url: str | None = None,
) -> PreparedTracePrompts:
    if input_tokens not in contract.input_tokens:
        raise TraceContractError(
            f"input_tokens={input_tokens} is outside the campaign contract"
        )
    resolved = tokenize_url or workload_contract.derive_tokenize_url(chat_url)
    corpus = load_prompt_corpus(contract)

    prompts: list[str] = []
    prompt_hashes: list[str] = []
    max_model_lens: set[int] = set()
    token_cache: dict[str, workload_contract.TokenizationResult] = {}

    def tokenize(prompt: str) -> workload_contract.TokenizationResult:
        if prompt not in token_cache:
            token_cache[prompt] = workload_contract.tokenize_prompt(
                resolved, model, prompt
            )
        return token_cache[prompt]

    def count(prompt: str) -> int:
        return tokenize(prompt).count

    for ordinal in range(contract.requests_per_input_length):
        prompt = _build_exact_trace_prompt(
            contract=contract,
            input_tokens=input_tokens,
            ordinal=ordinal,
            request_seed=corpus["request_seeds"][ordinal],
            continuation_sentences=corpus["continuation_sentences"],
            token_count=count,
        )
        verified = tokenize(prompt)
        if verified.count != input_tokens:
            raise TraceContractError(
                "trace prompt changed between construction and verification: "
                f"expected={input_tokens}, observed={verified.count}"
            )
        if verified.max_model_len is not None:
            max_model_lens.add(verified.max_model_len)
            if input_tokens + contract.output_tokens > verified.max_model_len:
                raise TraceContractError(
                    "trace input+output exceeds server max_model_len: "
                    f"{input_tokens}+{contract.output_tokens} > "
                    f"{verified.max_model_len}"
                )
        prompts.append(prompt)
        prompt_hashes.append(hashlib.sha256(prompt.encode()).hexdigest())

    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise TraceContractError("trace workload contains duplicate prompts")
    if len(max_model_lens) > 1:
        raise TraceContractError(
            f"server max_model_len changed during prompt preparation: {max_model_lens}"
        )

    set_digest = hashlib.sha256()
    for digest in prompt_hashes:
        set_digest.update(digest.encode())
        set_digest.update(b"\n")

    return PreparedTracePrompts(
        prompts=tuple(prompts),
        prompt_hashes=tuple(prompt_hashes),
        prompt_set_sha256=set_digest.hexdigest(),
        prompt_tokens=input_tokens,
        max_model_len=next(iter(max_model_lens), None),
        tokenize_url=resolved,
    )

def select_cases(
    cases: Iterable[TraceCase],
    *,
    input_tokens: int | None = None,
    concurrency: int | None = None,
    repetition_index: int | None = None,
) -> tuple[TraceCase, ...]:
    selected = tuple(
        case
        for case in cases
        if (input_tokens is None or case.input_tokens == input_tokens)
        and (concurrency is None or case.concurrency == concurrency)
        and (
            repetition_index is None
            or case.repetition_index == repetition_index
        )
    )
    if not selected:
        raise TraceContractError("case selector matched no predeclared trace case")
    return selected
