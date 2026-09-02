from scripts.launch_vllm import emit_launch_preamble


class _TrackingStream:
    def __init__(self):
        self.parts = []
        self.flush_count = 0

    def write(self, text):
        self.parts.append(text)

    def flush(self):
        self.flush_count += 1

    def getvalue(self):
        return "".join(self.parts)


def test_launch_preamble_is_flushed_with_evidence_provenance():
    stream = _TrackingStream()
    environment = {"FENIX_TRACE": "0"}
    command = [
        "/python",
        "-m",
        "scripts.fenix_podman",
        "run",
        "fenix-qwen38:candidate",
        "serve",
        "/model",
    ]

    emit_launch_preamble(environment, command, stream=stream)

    text = stream.getvalue()
    assert '"FENIX_TRACE": "0"' in text
    assert "fenix-qwen38:candidate" in text
    assert stream.flush_count == 1
