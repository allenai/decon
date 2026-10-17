"""Reference-export regression tests; no dataset downloads or datasets install needed."""
import contextlib
import importlib.util
import io
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("decon_evals", ROOT / "python" / "evals.py")
evals = importlib.util.module_from_spec(SPEC)
with mock.patch.dict(sys.modules, {"datasets": mock.Mock()}):
    SPEC.loader.exec_module(evals)


class ReferenceExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "references"
        self.config = self.root / "evals.yaml"

    def run_export(self, config, *args):
        # JSON is valid YAML and avoids depending on formatting in these fixtures.
        self.config.write_text(json.dumps({"evals": config}))
        argv = ["evals.py", "--config", str(self.config), "--output-dir", str(self.output), *args]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return evals.main()

    @staticmethod
    def dataset_config(**kwargs):
        return {"hf_path": "example/reference", "splits": ["test"], "prompt": "question", **kwargs}

    def test_revision_forwarded_and_recorded_with_and_without_config(self):
        revision = "a" * 40
        for hf_config in (None, "subset"):
            with self.subTest(hf_config=hf_config):
                config = self.dataset_config(hf_revision=revision)
                if hf_config:
                    config["hf_config"] = hf_config
                with mock.patch.object(evals, "load_dataset", return_value={"test": [{"question": "Example question?"}]}) as load:
                    self.assertEqual(self.run_export({"example": config}, "--download"), 0)
                expected_args = ("example/reference", hf_config) if hf_config else ("example/reference",)
                load.assert_called_once_with(*expected_args, revision=revision)
                row = json.loads((self.output / "example_test-1.jsonl").read_text())
                self.assertEqual(row["hf_revision"], revision)
                self.assertEqual(row["question"], "Example question?")
                self.assertEqual(len(row["fingerprint"]), 16)

    def test_unpinned_configuration_retains_load_call(self):
        with mock.patch.object(evals, "load_dataset", return_value={"test": [{"question": "Example?"}]}) as load:
            self.assertEqual(self.run_export({"example": self.dataset_config()}, "--download"), 0)
        load.assert_called_once_with("example/reference")
        self.assertNotIn("hf_revision", json.loads((self.output / "example_test-1.jsonl").read_text()))

    def test_download_failure_preserves_previous_directory(self):
        self.output.mkdir()
        previous = self.output / "previous_test-1.jsonl"
        previous.write_bytes(b"previous reference bytes\n")
        good = {"test": [{"question": "Example?"}]}
        for failure in (RuntimeError("unavailable"), {"train": []}, {"test": []}, {"test": [{"question": ""}]}):
            with self.subTest(failure=failure):
                with mock.patch.object(evals, "load_dataset", side_effect=[good, failure]):
                    config = {"first": self.dataset_config(), "second": self.dataset_config()}
                    self.assertEqual(self.run_export(config, "--download"), 1)
                self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()},
                                 {previous.name: b"previous reference bytes\n"})
                self.assertEqual(list(self.root.glob(".references-*")), [])

    def test_successful_download_replaces_complete_set(self):
        self.output.mkdir()
        (self.output / "old_test-1.jsonl").write_text("old")
        with mock.patch.object(evals, "load_dataset", return_value={"test": [{"question": "Example?"}]}):
            self.assertEqual(self.run_export({"new": self.dataset_config()}, "--download"), 0)
        self.assertEqual([p.name for p in self.output.iterdir()], ["new_test-1.jsonl"])

    def test_single_eval_preserves_other_references_and_removes_stale_chunks(self):
        self.output.mkdir()
        (self.output / "example_test-2.jsonl").write_text(json.dumps({"eval_key": "example", "doc_id": 50}))
        overlapping = json.dumps({"eval_key": "example_other", "question": "Other?", "doc_id": 3})
        (self.output / "example_other_test-1.jsonl").write_text(overlapping)
        other = b'# comment lines are skipped\n{"eval_key": "other", "question": "Other reference?", "doc_id": 7}\n'
        (self.output / "other_test-1.jsonl").write_bytes(other)
        with mock.patch.object(evals, "load_dataset", return_value={"test": [{"question": "Example?"}]}):
            self.assertEqual(self.run_export({"example": self.dataset_config()}, "--eval", "example"), 0)
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), ["example_other_test-1.jsonl", "example_test-1.jsonl", "other_test-1.jsonl"])
        self.assertEqual((self.output / "example_other_test-1.jsonl").read_text(), overlapping)
        self.assertEqual((self.output / "other_test-1.jsonl").read_bytes(), other)
        # New doc_ids follow the preserved references (max 7), not the removed stale chunk.
        self.assertEqual(json.loads((self.output / "example_test-1.jsonl").read_text())["doc_id"], 8)

    def test_missing_split_fails_before_writing_and_empty_split_list_fails(self):
        with mock.patch.object(evals, "load_dataset", return_value={"test": [{"question": "Example?"}]}):
            for splits in (["test", "missing"], []):
                with self.subTest(splits=splits):
                    self.assertEqual(self.run_export({"example": self.dataset_config(splits=splits)}, "--download"), 1)
                    self.assertFalse(self.output.exists())

    def test_bundled_sources_require_every_requested_split(self):
        bundled = self.root / "bundled-evals"
        bundled.mkdir()
        with gzip.open(bundled / "example_train-1.jsonl.gz", "wt") as stream:
            stream.write(json.dumps({"eval_key": "example", "split": "train", "question": "Example?"}) + "\n")
        config = {"example": {"local_path": "bundled_evals/example_", "splits": ["train", "missing"]}}
        with mock.patch.object(evals, "__file__", str(self.root / "python" / "evals.py")):
            self.assertEqual(self.run_export(config, "--download"), 1)
        self.assertFalse(self.output.exists())

    def test_empty_bundled_split_fails(self):
        bundled = self.root / "bundled-evals"
        bundled.mkdir()
        with gzip.open(bundled / "example_train-1.jsonl.gz", "wt") as stream:
            stream.write("\n")
        config = {"example": {"local_path": "bundled_evals/example_", "splits": ["train"]}}
        with mock.patch.object(evals, "__file__", str(self.root / "python" / "evals.py")):
            self.assertEqual(self.run_export(config, "--download"), 1)
        self.assertFalse(self.output.exists())

    def test_unknown_eval_and_missing_config_fail(self):
        self.assertEqual(self.run_export({"example": self.dataset_config()}, "--eval", "missing"), 1)
        self.config.unlink()
        with mock.patch.object(sys, "argv", ["evals.py", "--download", "--config", str(self.config)]), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(evals.main(), 1)
        self.assertFalse(self.output.exists())

    def test_rename_failure_restores_previous_directory(self):
        self.output.mkdir()
        (self.output / "old.jsonl").write_bytes(b"previous\n")
        staged = self.root / "staged"
        staged.mkdir()
        real_rename = Path.rename

        def rename(path, destination):
            if path == staged:
                raise OSError("publication failed")
            return real_rename(path, destination)

        with mock.patch.object(Path, "rename", rename), self.assertRaises(OSError):
            evals.publish_reference_directory(staged, self.output)
        self.assertEqual((self.output / "old.jsonl").read_bytes(), b"previous\n")

    def test_supplemental_reference_config_has_only_verified_pinned_additions(self):
        config = yaml.safe_load((ROOT / "config" / "post-training-evals.yaml").read_text())["evals"]
        self.assertEqual(set(config), {"aime_2026", "hmmt_feb_2026", "ifbench", "ifeval_mt", "ifeval_mt_ood"})
        for entry in config.values():
            self.assertRegex(entry["hf_revision"], r"^[0-9a-f]{40}$")
            self.assertIn(entry["splits"], (["train"], ["test"]))
            self.assertIn(entry["prompt"], ("problem", "prompt"))
        self.assertEqual(config["ifeval_mt"]["hf_config"], "wildchat_unused_withRewrite")
        self.assertEqual(config["ifeval_mt_ood"]["hf_config"], "ood_wildchat_unused_withRewrite")


if __name__ == "__main__":
    unittest.main()
