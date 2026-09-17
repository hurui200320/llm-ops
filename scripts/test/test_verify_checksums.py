"""Tests for verify_checksums.py."""

import contextlib
import io
import sys
import unittest
from unittest import mock

import helpers
import verify_checksums


class VerifyModelTest(helpers.ScriptTestCase):
    def setUp(self):
        self.root = self.make_temp_dir()
        self.model_dir, self.digests = helpers.make_model(
            self.root, "gemma-4", "google-test", {"w.gguf": b"weights"}
        )
        verify_checksums.LIBRARY_ROOT = self.root
        self.addCleanup(setattr, verify_checksums, "LIBRARY_ROOT", None)

    def run_verify_model(self) -> tuple[int, int, int, int, str]:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            ok, fail, warn, total = verify_checksums.verify_model(self.model_dir)
        return ok, fail, warn, total, stdout.getvalue()

    def test_intact_file_passes(self):
        ok, fail, warn, total, output = self.run_verify_model()
        self.assertEqual((ok, fail, warn, total), (1, 0, 0, 1))
        self.assertIn("OK", output)

    def test_corrupt_file_fails(self):
        (self.model_dir / "w.gguf").write_bytes(b"tampered")
        ok, fail, warn, total, output = self.run_verify_model()
        self.assertEqual((ok, fail, warn, total), (0, 1, 0, 1))
        self.assertIn("FAIL", output)
        self.assertIn("expected:", output)

    def test_missing_gguf_warns(self):
        (self.model_dir / "w.gguf").unlink()
        ok, fail, warn, total, output = self.run_verify_model()
        self.assertEqual((ok, fail, warn, total), (0, 0, 1, 1))
        self.assertIn("MISSING", output)

    def test_invalid_companion_warns(self):
        (self.model_dir / "w.gguf.sha256").write_text("nothex\n", encoding="utf-8")
        ok, fail, warn, total, output = self.run_verify_model()
        self.assertEqual((ok, fail, warn, total), (0, 0, 1, 1))
        self.assertIn("INVALID", output)

    def test_binary_companion_warns(self):
        (self.model_dir / "w.gguf.sha256").write_bytes(b"\x93\xfa\xb0\xff" * 16)
        ok, fail, warn, total, output = self.run_verify_model()
        self.assertEqual((ok, fail, warn, total), (0, 0, 1, 1))
        self.assertIn("INVALID", output)

    def test_no_companion_files_warns(self):
        (self.model_dir / "w.gguf.sha256").unlink()
        ok, fail, warn, total, output = self.run_verify_model()
        self.assertEqual((ok, fail, warn, total), (0, 0, 1, 0))
        self.assertIn("No .gguf.sha256 files", output)

    def test_uppercase_digest_accepted(self):
        (self.model_dir / "w.gguf.sha256").write_text(
            self.digests["w.gguf"].upper(), encoding="utf-8"
        )
        ok, fail, warn, total, _ = self.run_verify_model()
        self.assertEqual((ok, fail, warn, total), (1, 0, 0, 1))

    def test_progress_bar_suppressed_when_not_a_tty(self):
        with mock.patch.object(verify_checksums, "_TTY", False):
            ok, fail, warn, total, output = self.run_verify_model()
        self.assertEqual((ok, fail, warn, total), (1, 0, 0, 1))
        self.assertNotIn("\r", output)
        self.assertNotIn("█", output)

    def test_progress_bar_drawn_when_tty(self):
        with mock.patch.object(verify_checksums, "_TTY", True):
            ok, fail, warn, total, output = self.run_verify_model()
        self.assertEqual((ok, fail, warn, total), (1, 0, 0, 1))
        self.assertIn("\r", output)


class MainTest(helpers.ScriptTestCase):
    def setUp(self):
        self.root = self.make_temp_dir()
        self.addCleanup(setattr, verify_checksums, "LIBRARY_ROOT", None)

    def run_main(self, *argv) -> tuple[object, str]:
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", ["verify_checksums.py", *argv]):
            with contextlib.redirect_stdout(stdout):
                try:
                    verify_checksums.main()
                    code = 0
                except SystemExit as e:
                    code = e.code
        return code, stdout.getvalue()

    def test_all_passed_exits_0(self):
        helpers.make_model(self.root, "gemma-4", "google-test", {"w.gguf": b"weights"})
        code, output = self.run_main("--library-root", str(self.root))
        self.assertEqual(code, 0)
        self.assertIn("passed verification", output)

    def test_failure_exits_1(self):
        model_dir, _ = helpers.make_model(
            self.root, "gemma-4", "google-test", {"w.gguf": b"weights"}
        )
        (model_dir / "w.gguf").write_bytes(b"tampered")
        code, output = self.run_main("--library-root", str(self.root))
        self.assertEqual(code, 1)
        self.assertIn("failed verification: 1", output)

    def test_model_filter(self):
        helpers.make_model(self.root, "gemma-4", "google-test", {"w.gguf": b"a"})
        model_dir, _ = helpers.make_model(
            self.root, "ornith-1.5", "ornith-test", {"w.gguf": b"b"}
        )
        (model_dir / "w.gguf").write_bytes(b"c")
        code, output = self.run_main("--library-root", str(self.root), "-m", "ornith")
        self.assertEqual(code, 1)
        self.assertIn("ornith-1.5", output)
        self.assertNotIn("gemma-4", output)

    def test_variant_without_readme_not_discovered(self):
        model_dir, _ = helpers.make_model(
            self.root, "gemma-4", "google-test", {"w.gguf": b"weights"}
        )
        (model_dir / "README.md").unlink()
        code, output = self.run_main("--library-root", str(self.root))
        self.assertEqual(code, 1)
        self.assertIn("No model directories containing README.md", output)

    def test_empty_library_exits_1(self):
        code, output = self.run_main("--library-root", str(self.root))
        self.assertEqual(code, 1)
        self.assertIn("No model directories containing README.md", output)

    def test_invalid_library_root_exits_2(self):
        code, output = self.run_main(
            "--library-root", str(self.root / "missing")
        )
        self.assertEqual(code, 2)
        self.assertIn("is not an existing directory", output)


if __name__ == "__main__":
    unittest.main()
