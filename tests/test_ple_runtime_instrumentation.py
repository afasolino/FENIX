from pathlib import Path

from instrumentation.prepare_runtime_inplace import instrument_ple_address_trace


def test_ple_address_trace_targets_inline_ngram_ids(tmp_path: Path):
    path = tmp_path / "ple_layer.py"
    path.write_text(
        "def forward_impl(self, input_ids, query_start_loc, ngram_context):\n"
        "        ngram_ids = torch.cat(id_blocks, dim=-1)\n"
        "        return ngram_ids\n"
    )

    instrument_ple_address_trace(path)

    text = path.read_text()
    assert 'emit("ple_runtime"' in text
    assert '"physical_row_ids":ngram_ids.detach().cpu().tolist()' in text
    assert '"address_known_ns":_address_known_ns' in text
    assert 'compute_ngram_ids' not in text
    compile(text, str(path), "exec")


def test_prepare_runtime_no_longer_targets_missing_compute_ngram_ids():
    source = Path("instrumentation/prepare_runtime_inplace.py").read_text()
    assert "compute_ngram_ids" not in source
