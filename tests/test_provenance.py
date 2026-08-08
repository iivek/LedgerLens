import hashlib

from ledgerlens.harness import run_eval
from ledgerlens.runners.llamacpp import LlamaCppRunner, _sha256
from ledgerlens.data.unified import write_split
from PIL import Image


def test_sha256_of_real_file(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"fake gguf content")
    expected = hashlib.sha256(b"fake gguf content").hexdigest()[:16]
    assert _sha256(str(f)) == expected


def test_sha256_missing_file_returns_none(tmp_path):
    assert _sha256(str(tmp_path / "nope.gguf")) is None


def test_llamacpp_provenance_includes_model_hash_when_path_given(tmp_path):
    model = tmp_path / "student-q4km.gguf"
    model.write_bytes(b"x" * 1000)
    runner = LlamaCppRunner(model_path=str(model))
    prov = runner.provenance()
    assert prov["model_path"] == str(model)
    assert prov["model_sha256_16"] == hashlib.sha256(b"x" * 1000).hexdigest()[:16]
    assert prov["model_size_bytes"] == 1000


def test_llamacpp_provenance_without_paths_still_returns_dict():
    runner = LlamaCppRunner()
    prov = runner.provenance()
    assert "model_path" not in prov
    assert prov["runtime"] == "llama.cpp/llama-server"


def test_harness_row_includes_provenance_when_runner_supports_it(tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (10, 10), "white").save(tmp_path / "images" / "s0.png")
    write_split(
        [{"id": "s0", "image": "images/s0.png", "source": "test",
          "gold": {"vendor": "X", "total": 1.0, "line_items": []}}],
        tmp_path / "split.jsonl",
    )
    model = tmp_path / "m.gguf"
    model.write_bytes(b"weights")
    runner = LlamaCppRunner(model_path=str(model))
    row = run_eval(runner, tmp_path / "split.jsonl", tag="t", host="test",
                   runs_root=tmp_path / "runs", results_csv=tmp_path / "results.csv")
    assert row["prov_model_sha256_16"] == hashlib.sha256(b"weights").hexdigest()[:16]
    assert row["prov_runtime"] == "llama.cpp/llama-server"


def test_harness_row_has_no_provenance_keys_for_mock_runner(tmp_path):
    from ledgerlens.runners.mock import MockRunner
    (tmp_path / "images").mkdir()
    Image.new("RGB", (10, 10), "white").save(tmp_path / "images" / "s0.png")
    write_split(
        [{"id": "s0", "image": "images/s0.png", "source": "test",
          "gold": {"vendor": "X", "total": 1.0, "line_items": []}}],
        tmp_path / "split.jsonl",
    )
    row = run_eval(MockRunner(), tmp_path / "split.jsonl", tag="t", host="test",
                   runs_root=tmp_path / "runs", results_csv=tmp_path / "results.csv")
    assert not any(k.startswith("prov_") for k in row)
