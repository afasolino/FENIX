#!/usr/bin/env python3
"""Build or verify the FENIX mmap-ready PLE storage bank without CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from instrumentation.ple_bank import build_ple_bank, discover_ple_shards
from instrumentation.storage_bank import StorageBankError, validate_bank_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--tensor-prefix")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument(
        "--skip-data-sha256",
        action="store_true",
        help="verify manifest/length only; full SHA-256 is the default",
    )
    args = parser.parse_args()

    try:
        if args.verify is not None:
            if args.model_dir is not None or args.out_dir is not None or args.scan:
                raise StorageBankError("--verify cannot be combined with build arguments")
            manifest = validate_bank_manifest(
                args.verify,
                verify_data_sha256=not args.skip_data_sha256,
            )
            print(
                json.dumps(
                    {
                        "valid": True,
                        "manifest": str(args.verify),
                        "artifact_kind": manifest.get("artifact_kind"),
                        "data_bytes": manifest["data_bytes"],
                        "data_sha256": manifest["data_sha256"],
                        "layout": manifest.get("layout"),
                    },
                    indent=2,
                )
            )
            return 0

        if args.scan:
            if args.model_dir is None:
                raise StorageBankError("--model-dir is required for --scan")
            if args.out_dir is not None:
                raise StorageBankError("--scan does not accept --out-dir")
            prefix, records = discover_ple_shards(
                args.model_dir.resolve(), tensor_prefix=args.tensor_prefix
            )
            row_bytes = {record.data_bytes // record.rows for record in records}
            print(
                json.dumps(
                    {
                        "scanned": True,
                        "tensor_prefix": prefix,
                        "shard_count": len(records),
                        "dtype": records[0].dtype,
                        "embedding_width": records[0].shape[1],
                        "total_rows": sum(record.rows for record in records),
                        "data_bytes": sum(record.data_bytes for record in records),
                        "row_bytes": next(iter(row_bytes)) if len(row_bytes) == 1 else None,
                    },
                    indent=2,
                )
            )
            return 0

        if args.model_dir is None or args.out_dir is None:
            raise StorageBankError("--model-dir and --out-dir are required for build")
        artifact = build_ple_bank(
            model_dir=args.model_dir,
            output_dir=args.out_dir,
            tensor_prefix=args.tensor_prefix,
        )
        manifest = validate_bank_manifest(artifact.manifest_path)
        print(
            json.dumps(
                {
                    "built": True,
                    "data_path": str(artifact.data_path),
                    "manifest_path": str(artifact.manifest_path),
                    "data_bytes": artifact.data_bytes,
                    "data_sha256": artifact.data_sha256,
                    "layout": manifest["layout"],
                },
                indent=2,
            )
        )
        return 0
    except (StorageBankError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
