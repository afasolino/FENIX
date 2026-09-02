#!/usr/bin/env python3
"""Campaign-backed workload contracts for FENIX measurements.

This module owns the mapping from a versioned campaign experiment to the exact
rendered chat-token workload sent to the runtime. It deliberately asks the
running server's ``/tokenize`` endpoint to validate chat-template token counts;
client-side tokenizer assumptions are not accepted for evidence promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import requests


DEFAULT_CAMPAIGN = Path("configs/campaign.json")
DEFAULT_EXPERIMENT = "runtime_qualification"
DEFAULT_PROFILE = "runtime_qualification_v1"

_RUNTIME_SEED = (
    "FENIX runtime qualification workload. Analyze memory locality, "
    "conditional lookup traffic, expert routing, host-memory residency, "
    "and autoregressive decoding while treating the remaining deterministic "
    "text as benchmark material.\n"
)
_RUNTIME_BULK = (
    "Memory locality changes cache residency and transfer demand; conditional "
    "lookups select sparse rows; routed experts compete for host-memory "
    "capacity; deterministic decoding exposes repeatable runtime behavior. "
)
_FINE_FRAGMENTS = (
    " x",
    " a",
    " 0",
    " 1",
    " memory",
    " locality",
    " token",
    " data",
    " inference",
    ".",
    ",",
    ";",
    ":",
    "\n",
)


class WorkloadContractError(RuntimeError):
    """Raised when a campaign workload cannot be established exactly."""


@dataclass(frozen=True)
class WorkloadProfile:
    name: str
    seed: str
    bulk_fragment: str
    fine_fragments: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentContract:
    experiment: str
    workload_profile: str
    input_tokens: int
    output_tokens: int
    concurrency: int
    temperature: float
    warmup_requests: int
    measured_requests: int
    repetitions: int


@dataclass(frozen=True)
class TokenizationResult:
    count: int
    max_model_len: int | None


@dataclass(frozen=True)
class PreparedWorkload:
    prompt: str
    prompt_tokens: int
    max_model_len: int | None
    tokenize_url: str
    workload_profile: str

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode()).hexdigest()


WORKLOAD_PROFILES: Mapping[str, WorkloadProfile] = {
    DEFAULT_PROFILE: WorkloadProfile(
        name=DEFAULT_PROFILE,
        seed=_RUNTIME_SEED,
        bulk_fragment=_RUNTIME_BULK,
        fine_fragments=_FINE_FRAGMENTS,
    )
}


def _single_int(value: object, field: str) -> int:
    if not isinstance(value, list) or len(value) != 1:
        raise WorkloadContractError(
            f"{field} must contain exactly one value for this runner"
        )
    item = value[0]
    if not isinstance(item, int) or isinstance(item, bool) or item < 1:
        raise WorkloadContractError(
            f"{field} must contain one positive integer"
        )
    return item


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise WorkloadContractError(f"{field} must be a positive integer")
    return value


def load_experiment_contract(
    campaign_path: Path,
    experiment: str = DEFAULT_EXPERIMENT,
) -> ExperimentContract:
    """Load one exact single-endpoint experiment from ``campaign.json``."""

    if not campaign_path.is_file():
        raise WorkloadContractError(
            f"campaign configuration does not exist: {campaign_path}"
        )

    payload = json.loads(campaign_path.read_text())
    try:
        raw = payload["experiments"][experiment]
    except (KeyError, TypeError) as exc:
        raise WorkloadContractError(
            f"campaign experiment is missing: {experiment}"
        ) from exc

    if not isinstance(raw, dict):
        raise WorkloadContractError(
            f"campaign experiment must be an object: {experiment}"
        )

    profile = raw.get("workload_profile")
    if not isinstance(profile, str) or profile not in WORKLOAD_PROFILES:
        raise WorkloadContractError(
            f"{experiment}.workload_profile must name a supported profile"
        )

    temperature = raw.get("temperature")
    if not isinstance(temperature, (int, float)) or isinstance(
        temperature, bool
    ):
        raise WorkloadContractError(
            f"{experiment}.temperature must be numeric"
        )

    return ExperimentContract(
        experiment=experiment,
        workload_profile=profile,
        input_tokens=_single_int(
            raw.get("input_tokens"),
            f"{experiment}.input_tokens",
        ),
        output_tokens=_positive_int(
            raw.get("output_tokens"),
            f"{experiment}.output_tokens",
        ),
        concurrency=_single_int(
            raw.get("concurrency"),
            f"{experiment}.concurrency",
        ),
        temperature=float(temperature),
        warmup_requests=_positive_int(
            raw.get("warmup_requests"),
            f"{experiment}.warmup_requests",
        ),
        measured_requests=_positive_int(
            raw.get("measured_requests"),
            f"{experiment}.measured_requests",
        ),
        repetitions=_positive_int(
            raw.get("repetitions"),
            f"{experiment}.repetitions",
        ),
    )


def validate_repetition_index(
    contract: ExperimentContract,
    repetition_index: int,
) -> None:
    if not 1 <= repetition_index <= contract.repetitions:
        raise WorkloadContractError(
            "repetition_index must be within the predeclared campaign range "
            f"1..{contract.repetitions}, got {repetition_index}"
        )


def derive_tokenize_url(chat_url: str) -> str:
    """Map the OpenAI chat-completions URL to the runtime tokenize endpoint."""

    parts = urlsplit(chat_url)
    suffix = "/v1/chat/completions"
    if not parts.path.endswith(suffix):
        raise WorkloadContractError(
            "cannot derive /tokenize endpoint from chat URL; pass an explicit "
            "tokenize URL"
        )

    prefix = parts.path[: -len(suffix)]
    tokenize_path = f"{prefix}/tokenize" or "/tokenize"
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            tokenize_path,
            "",
            "",
        )
    )


def tokenize_prompt(
    tokenize_url: str,
    model: str,
    prompt: str,
    timeout_s: float = 60.0,
) -> TokenizationResult:
    """Ask the running server for the rendered chat-template token count."""

    response = requests.post(
        tokenize_url,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()

    count = payload.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise WorkloadContractError(
            f"/tokenize returned invalid count: {count!r}"
        )

    max_model_len = payload.get("max_model_len")
    if max_model_len is not None and (
        not isinstance(max_model_len, int)
        or isinstance(max_model_len, bool)
        or max_model_len < 1
    ):
        raise WorkloadContractError(
            f"/tokenize returned invalid max_model_len: {max_model_len!r}"
        )

    return TokenizationResult(
        count=count,
        max_model_len=max_model_len,
    )


def _cached_counter(
    token_count: Callable[[str], int],
) -> Callable[[str], int]:
    cache: dict[str, int] = {}

    def count(prompt: str) -> int:
        if prompt not in cache:
            value = token_count(prompt)
            if not isinstance(value, int) or isinstance(value, bool):
                raise WorkloadContractError(
                    f"token counter returned non-integer value: {value!r}"
                )
            cache[prompt] = value
        return cache[prompt]

    return count


def build_exact_token_prompt(
    *,
    target_tokens: int,
    token_count: Callable[[str], int],
    profile: WorkloadProfile,
    max_fine_steps: int = 256,
) -> str:
    """Construct a deterministic prompt whose rendered chat count is exact.

    The bulk stage uses a monotonic bounded search over a fixed corpus fragment.
    The fine stage re-measures candidate suffixes on every step because BPE
    boundaries can change after concatenation. The function fails closed when
    the live tokenizer cannot reach the exact target using the versioned
    profile; it never rounds to a nearby token count.
    """

    if target_tokens < 1:
        raise WorkloadContractError("target_tokens must be positive")

    count = _cached_counter(token_count)
    seed = profile.seed
    seed_count = count(seed)
    if seed_count > target_tokens:
        raise WorkloadContractError(
            f"profile seed already exceeds target: {seed_count} > "
            f"{target_tokens}"
        )
    if seed_count == target_tokens:
        return seed

    def with_bulk(repetitions: int) -> str:
        return seed + profile.bulk_fragment * repetitions

    low = 0
    high = 1
    high_count = count(with_bulk(high))
    while high_count <= target_tokens:
        low = high
        if high_count == target_tokens:
            return with_bulk(high)
        high *= 2
        if high > target_tokens * 2:
            raise WorkloadContractError(
                "bulk prompt search failed to bracket the target"
            )
        high_count = count(with_bulk(high))

    while low + 1 < high:
        middle = (low + high) // 2
        middle_count = count(with_bulk(middle))
        if middle_count <= target_tokens:
            low = middle
            if middle_count == target_tokens:
                return with_bulk(middle)
        else:
            high = middle

    prompt = with_bulk(low)
    current = count(prompt)

    fine_fragments = profile.fine_fragments + tuple(
        f" {value}" for value in range(2, 65)
    )

    for _ in range(max_fine_steps):
        if current == target_tokens:
            return prompt

        best_prompt: str | None = None
        best_count = current

        for fragment in fine_fragments:
            candidate = prompt + fragment
            candidate_count = count(candidate)
            if candidate_count == target_tokens:
                return candidate
            if current < candidate_count <= target_tokens:
                if candidate_count > best_count:
                    best_prompt = candidate
                    best_count = candidate_count

        if best_prompt is None:
            raise WorkloadContractError(
                "exact prompt construction stalled at "
                f"{current}/{target_tokens} rendered tokens"
            )

        prompt = best_prompt
        current = best_count

    raise WorkloadContractError(
        f"exact prompt construction exceeded {max_fine_steps} fine steps"
    )


def prepare_workload(
    *,
    contract: ExperimentContract,
    chat_url: str,
    model: str,
    tokenize_url: str | None = None,
) -> PreparedWorkload:
    """Build and independently revalidate the campaign workload."""

    resolved_tokenize_url = tokenize_url or derive_tokenize_url(chat_url)
    profile = WORKLOAD_PROFILES[contract.workload_profile]

    def count(prompt: str) -> int:
        return tokenize_prompt(
            resolved_tokenize_url,
            model,
            prompt,
        ).count

    prompt = build_exact_token_prompt(
        target_tokens=contract.input_tokens,
        token_count=count,
        profile=profile,
    )
    verified = tokenize_prompt(
        resolved_tokenize_url,
        model,
        prompt,
    )

    if verified.count != contract.input_tokens:
        raise WorkloadContractError(
            "workload changed between construction and verification: "
            f"expected {contract.input_tokens}, observed {verified.count}"
        )
    if (
        verified.max_model_len is not None
        and contract.input_tokens + contract.output_tokens
        > verified.max_model_len
    ):
        raise WorkloadContractError(
            "campaign input+output tokens exceed server max_model_len: "
            f"{contract.input_tokens}+{contract.output_tokens} > "
            f"{verified.max_model_len}"
        )

    return PreparedWorkload(
        prompt=prompt,
        prompt_tokens=verified.count,
        max_model_len=verified.max_model_len,
        tokenize_url=resolved_tokenize_url,
        workload_profile=profile.name,
    )


def record_token_mismatches(
    records: Sequence[Mapping[str, object]],
    *,
    contract: ExperimentContract,
    phase: str,
) -> list[dict[str, object]]:
    """Return detailed token-count deviations from the campaign contract."""

    mismatches: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if "error" in record:
            continue
        ordinal = record.get("ordinal", index)
        for field, expected in (
            ("prompt_tokens", contract.input_tokens),
            ("completion_tokens", contract.output_tokens),
        ):
            observed = record.get(field)
            if observed != expected:
                mismatches.append(
                    {
                        "phase": phase,
                        "ordinal": ordinal,
                        "field": field,
                        "expected": expected,
                        "observed": observed,
                    }
                )
    return mismatches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=DEFAULT_CAMPAIGN,
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument("--tokenize-url")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        contract = load_experiment_contract(
            args.campaign,
            args.experiment,
        )
        workload = prepare_workload(
            contract=contract,
            chat_url=args.url,
            model=args.model,
            tokenize_url=args.tokenize_url,
        )
    except WorkloadContractError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(workload.prompt)
    print(
        json.dumps(
            {
                "experiment": contract.experiment,
                "workload_profile": workload.workload_profile,
                "prompt_tokens": workload.prompt_tokens,
                "output_tokens": contract.output_tokens,
                "prompt_sha256": workload.prompt_sha256,
                "max_model_len": workload.max_model_len,
                "tokenize_url": workload.tokenize_url,
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
