from pathlib import Path

from instrumentation.prepare_runtime_inplace import instrument_ple_bank_runtime


def _runtime_source() -> str:
    return '''import math

class Example:
    def construct(self):
        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
            quant_method=_get_ple_embedding_quant_method(
                quant_config,
                f"{prefix}.ngram_embedding",
                getattr(config, "ple_embedding_dtype", None),
            ),
        )

    def forward_impl(self):
        ngram_ids = torch.cat(id_blocks, dim=-1)
        if output_buffer is not None:
            output = output_buffer[:num_tokens, : self.embedding_dim]
            torch.index_select(
                self.ngram_embedding.weight,
                0,
                ngram_ids.reshape(-1),
                out=output.reshape(-1, self.head_dim),
            )
            return output
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def get_offload_output_dtype(self):
        embedding = getattr(self, "ngram_embedding", None)
        weight = getattr(embedding, "weight", None)

    def load_weights(self, weights):
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                pass
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
'''


def test_runtime_instrumentation_adds_opt_in_bank_without_removing_resident_path(
    tmp_path: Path,
):
    target = tmp_path / "ple_layer.py"
    target.write_text(_runtime_source())

    instrument_ple_bank_runtime(target)
    text = target.read_text()

    assert 'FENIX_PLE_STORAGE_MODE", "resident"' in text
    assert "FenixPleBankEmbedding.from_environment" in text
    assert "else:\n            self.ngram_embedding = VocabParallelEmbedding(" in text
    assert 'getattr(self.ngram_embedding, "_fenix_ple_bank", False)' in text
    assert "self.ngram_embedding.gather_into" in text
    assert "externalized_bank" in text
    assert 'name == "ngram_embedding.weight_scale"' in text
    assert 'loaded.add("ngram_embedding.weight")' in text
