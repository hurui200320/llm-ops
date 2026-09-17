"""Tests for check_updates.py."""

import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import helpers
import check_updates

import verify_checksums


class ParseFrontMatterTest(helpers.ScriptTestCase):
    def test_parses_simple_fields(self):
        readme = self.make_temp_dir() / "README.md"
        readme.write_text(
            "---\nsource: https://huggingface.co/a/b\nnote: hello\n---\n\nbody\n",
            encoding="utf-8",
        )
        self.assertEqual(
            check_updates.parse_front_matter(readme),
            {"source": "https://huggingface.co/a/b", "note": "hello"},
        )

    def test_returns_none_without_front_matter(self):
        readme = self.make_temp_dir() / "README.md"
        readme.write_text("# plain markdown\n", encoding="utf-8")
        self.assertIsNone(check_updates.parse_front_matter(readme))

    def test_skips_comments_and_blank_lines(self):
        readme = self.make_temp_dir() / "README.md"
        readme.write_text(
            "---\n# a comment\n\nsource: https://huggingface.co/a/b\n---\n",
            encoding="utf-8",
        )
        self.assertEqual(
            check_updates.parse_front_matter(readme),
            {"source": "https://huggingface.co/a/b"},
        )


class ParseHfRepoTest(helpers.ScriptTestCase):
    def test_valid_repo(self):
        self.assertEqual(
            check_updates.parse_hf_repo("https://huggingface.co/a/b"),
            ("a/b", None),
        )

    def test_hf_co_short_host_valid(self):
        self.assertEqual(
            check_updates.parse_hf_repo("https://hf.co/a/b"),
            ("a/b", None),
        )

    def test_empty_source(self):
        repo_id, err = check_updates.parse_hf_repo("")
        self.assertIsNone(repo_id)
        self.assertTrue(err)

    def test_unsupported_host(self):
        repo_id, err = check_updates.parse_hf_repo("https://example.com/a/b")
        self.assertIsNone(repo_id)
        self.assertTrue(err)

    def test_short_path(self):
        repo_id, err = check_updates.parse_hf_repo("https://huggingface.co/a")
        self.assertIsNone(repo_id)
        self.assertTrue(err)

    def test_unsupported_scheme(self):
        repo_id, err = check_updates.parse_hf_repo("ftp://huggingface.co/a/b")
        self.assertIsNone(repo_id)
        self.assertTrue(err)


class ResolveLibraryRootTest(helpers.ScriptTestCase):
    MODULES = (check_updates, verify_checksums)

    def test_argument_wins_over_env_and_cwd(self):
        arg_root = self.make_temp_dir()
        env_root = self.make_temp_dir()
        cwd_root = self.make_temp_dir()
        with mock.patch.dict(os.environ, {"LLM_LIBRARY_ROOT": str(env_root)}):
            with contextlib.chdir(cwd_root):
                for module in self.MODULES:
                    with self.subTest(module=module.__name__):
                        self.assertEqual(
                            module.resolve_library_root(str(arg_root)), arg_root
                        )

    def test_env_used_without_argument(self):
        env_root = self.make_temp_dir()
        cwd_root = self.make_temp_dir()
        with mock.patch.dict(os.environ, {"LLM_LIBRARY_ROOT": str(env_root)}):
            with contextlib.chdir(cwd_root):
                for module in self.MODULES:
                    with self.subTest(module=module.__name__):
                        self.assertEqual(
                            module.resolve_library_root(None), env_root
                        )

    def test_cwd_used_without_argument_and_env(self):
        cwd_root = self.make_temp_dir()
        environ = dict(os.environ)
        environ.pop("LLM_LIBRARY_ROOT", None)
        with mock.patch.dict(os.environ, environ, clear=True):
            with contextlib.chdir(cwd_root):
                for module in self.MODULES:
                    with self.subTest(module=module.__name__):
                        self.assertEqual(
                            module.resolve_library_root(None), cwd_root
                        )

    def test_empty_argument_and_env_fall_through_to_cwd(self):
        cwd_root = self.make_temp_dir()
        environ = dict(os.environ)
        environ["LLM_LIBRARY_ROOT"] = ""
        with mock.patch.dict(os.environ, environ, clear=True):
            with contextlib.chdir(cwd_root):
                for module in self.MODULES:
                    with self.subTest(module=module.__name__):
                        self.assertEqual(
                            module.resolve_library_root(""), cwd_root
                        )

    def test_missing_argument_root_exits_2(self):
        missing = self.make_temp_dir() / "does-not-exist"
        for module in self.MODULES:
            with self.subTest(module=module.__name__):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as cm:
                        module.resolve_library_root(str(missing))
                self.assertEqual(cm.exception.code, 2)

    def test_missing_env_root_exits_2(self):
        missing = self.make_temp_dir() / "does-not-exist"
        with mock.patch.dict(os.environ, {"LLM_LIBRARY_ROOT": str(missing)}):
            for module in self.MODULES:
                with self.subTest(module=module.__name__):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit) as cm:
                            module.resolve_library_root(None)
                    self.assertEqual(cm.exception.code, 2)


class FetchLfsSha256Test(helpers.ScriptTestCase):
    POINTER_URL = "https://huggingface.co/example/test/raw/main/w.gguf"

    def test_success(self):
        digest = helpers.sha256_hex(b"weights")
        fake = helpers.FakeUrlopen(
            {self.POINTER_URL: helpers.FakeResponse(helpers.lfs_pointer(digest, 7))}
        )
        with mock.patch.object(check_updates, "urlopen", fake):
            result = check_updates.fetch_lfs_sha256("example/test", "w.gguf")
        self.assertEqual(result, (digest, 7))

    def test_not_found(self):
        fake = helpers.FakeUrlopen(
            {self.POINTER_URL: helpers.http_error(self.POINTER_URL, 404)}
        )
        with mock.patch.object(check_updates, "urlopen", fake):
            sha, err = check_updates.fetch_lfs_sha256("example/test", "w.gguf")
        self.assertIsNone(sha)
        self.assertIn("404", err)

    def test_non_lfs_response(self):
        fake = helpers.FakeUrlopen(
            {self.POINTER_URL: helpers.FakeResponse(b"plain text file")}
        )
        with mock.patch.object(check_updates, "urlopen", fake):
            sha, err = check_updates.fetch_lfs_sha256("example/test", "w.gguf")
        self.assertIsNone(sha)
        self.assertIn("LFS", err)

    def test_remote_path_with_space_is_percent_encoded(self):
        digest = helpers.sha256_hex(b"weights")
        url = "https://huggingface.co/example/test/raw/main/my%20file.gguf"
        fake = helpers.FakeUrlopen({url: helpers.FakeResponse(helpers.lfs_pointer(digest, 7))})
        with mock.patch.object(check_updates, "urlopen", fake):
            result = check_updates.fetch_lfs_sha256("example/test", "my file.gguf")
        self.assertEqual(result, (digest, 7))
        self.assertEqual([req.full_url for req in fake.requests], [url])


class CheckModelTest(helpers.ScriptTestCase):
    CONTENT = b"weights"

    def setUp(self):
        self.root = self.make_temp_dir()
        self.model_dir, self.digests = helpers.make_model(
            self.root, "gemma-4", "google-test", {"w.gguf": self.CONTENT}
        )
        check_updates.LIBRARY_ROOT = self.root
        self.addCleanup(setattr, check_updates, "LIBRARY_ROOT", None)

    def pointer_url(self, remote_path: str) -> str:
        return f"https://huggingface.co/example/test/raw/main/{remote_path}"

    def run_check_model(self, fake) -> tuple[int, int, int, str]:
        stdout = io.StringIO()
        with mock.patch.object(check_updates, "urlopen", fake):
            with contextlib.redirect_stdout(stdout):
                result = check_updates.check_model(self.model_dir)
        return *result, stdout.getvalue()

    def test_up_to_date(self):
        fake = helpers.FakeUrlopen(
            {
                self.pointer_url("w.gguf"): helpers.FakeResponse(
                    helpers.lfs_pointer(self.digests["w.gguf"], len(self.CONTENT))
                )
            }
        )
        updates, warns, total, output = self.run_check_model(fake)
        self.assertEqual((updates, warns, total), (0, 0, 1))
        self.assertIn("OK", output)

    def test_uppercase_digest_accepted(self):
        (self.model_dir / "w.gguf.sha256").write_text(
            self.digests["w.gguf"].upper(), encoding="utf-8"
        )
        fake = helpers.FakeUrlopen(
            {
                self.pointer_url("w.gguf"): helpers.FakeResponse(
                    helpers.lfs_pointer(self.digests["w.gguf"], len(self.CONTENT))
                )
            }
        )
        updates, warns, total, output = self.run_check_model(fake)
        self.assertEqual((updates, warns, total), (0, 0, 1))
        self.assertIn("OK", output)
        self.assertNotIn("UPDATE", output)

    def test_update_available(self):
        remote_digest = helpers.sha256_hex(b"new weights upstream")
        fake = helpers.FakeUrlopen(
            {
                self.pointer_url("w.gguf"): helpers.FakeResponse(
                    helpers.lfs_pointer(remote_digest, 99)
                )
            }
        )
        updates, warns, total, output = self.run_check_model(fake)
        self.assertEqual((updates, warns, total), (1, 0, 1))
        self.assertIn("UPDATE", output)

    def test_nested_remote_path(self):
        (self.model_dir / "w.gguf").unlink()
        (self.model_dir / "w.gguf.sha256").unlink()
        content = b"nested weights"
        digest = helpers.sha256_hex(content)
        (self.model_dir / "BF16").mkdir()
        (self.model_dir / "BF16" / "w.gguf").write_bytes(content)
        (self.model_dir / "BF16" / "w.gguf.sha256").write_text(digest, encoding="utf-8")
        url = self.pointer_url("BF16/w.gguf")
        fake = helpers.FakeUrlopen({url: helpers.http_error(url, 404)})
        stdout = io.StringIO()
        with mock.patch.object(check_updates, "urlopen", fake):
            with contextlib.redirect_stdout(stdout):
                result = check_updates.check_model(self.model_dir)
        self.assertEqual(result, (0, 1, 1))
        self.assertIn("BF16/w.gguf", stdout.getvalue())
        self.assertIn("404", stdout.getvalue())

    def test_missing_readme_warns(self):
        (self.model_dir / "README.md").unlink()
        updates, warns, total, output = self.run_check_model(helpers.FakeUrlopen({}))
        self.assertEqual((updates, warns, total), (0, 1, 0))
        self.assertIn("README.md not found", output)

    def test_missing_front_matter_warns(self):
        (self.model_dir / "README.md").write_text("# no front matter\n", encoding="utf-8")
        updates, warns, total, output = self.run_check_model(helpers.FakeUrlopen({}))
        self.assertEqual((updates, warns, total), (0, 1, 0))
        self.assertIn("No YAML front matter", output)

    def test_unsupported_source_skips_without_warning(self):
        (self.model_dir / "README.md").write_text(
            "---\nsource: https://example.com/a/b\n---\n", encoding="utf-8"
        )
        updates, warns, total, output = self.run_check_model(helpers.FakeUrlopen({}))
        self.assertEqual((updates, warns, total), (0, 0, 0))
        self.assertIn("SKIP", output)

    def test_no_companion_files_warns(self):
        for companion in self.model_dir.glob("*.sha256"):
            companion.unlink()
        updates, warns, total, output = self.run_check_model(helpers.FakeUrlopen({}))
        self.assertEqual((updates, warns, total), (0, 1, 0))
        self.assertIn("No .gguf.sha256 files", output)

    def test_binary_companion_warns_without_network_request(self):
        (self.model_dir / "w.gguf.sha256").write_bytes(b"\x93\xfa\xb0\xff" * 16)
        fake = helpers.FakeUrlopen({})
        updates, warns, total, output = self.run_check_model(fake)
        self.assertEqual((updates, warns, total), (0, 1, 1))
        self.assertIn("WARN", output)
        self.assertIn("valid 64-character hexadecimal hash", output)
        self.assertEqual(fake.requests, [])

    def test_sha256sum_style_companion_warns_not_update(self):
        (self.model_dir / "w.gguf.sha256").write_text(
            self.digests["w.gguf"] + "  w.gguf\n", encoding="utf-8"
        )
        fake = helpers.FakeUrlopen({})
        updates, warns, total, output = self.run_check_model(fake)
        self.assertEqual((updates, warns, total), (0, 1, 1))
        self.assertIn("WARN", output)
        self.assertNotIn("UPDATE", output)
        self.assertEqual(fake.requests, [])

    def test_hf_co_short_host_source_is_checked(self):
        (self.model_dir / "README.md").write_text(
            "---\nsource: https://hf.co/example/test\n---\n", encoding="utf-8"
        )
        fake = helpers.FakeUrlopen(
            {
                self.pointer_url("w.gguf"): helpers.FakeResponse(
                    helpers.lfs_pointer(self.digests["w.gguf"], len(self.CONTENT))
                )
            }
        )
        updates, warns, total, output = self.run_check_model(fake)
        self.assertEqual((updates, warns, total), (0, 0, 1))
        self.assertIn("OK", output)
        self.assertIn("[example/test]", output)


class MainTest(helpers.ScriptTestCase):
    CONTENT = b"weights"

    def setUp(self):
        self.root = self.make_temp_dir()
        self.addCleanup(setattr, check_updates, "LIBRARY_ROOT", None)

    def make_fixture(self):
        _, digests = helpers.make_model(
            self.root, "gemma-4", "google-test", {"w.gguf": self.CONTENT}
        )
        return digests

    def pointer_url(self, repo: str, remote_path: str) -> str:
        return f"https://huggingface.co/{repo}/raw/main/{remote_path}"

    def run_main(self, fake, *argv) -> tuple[object, str]:
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", ["check_updates.py", *argv]):
            with mock.patch.object(check_updates, "urlopen", fake):
                with contextlib.redirect_stdout(stdout):
                    try:
                        check_updates.main()
                        code = 0
                    except SystemExit as e:
                        code = e.code
        return code, stdout.getvalue()

    def test_all_up_to_date_exits_0(self):
        digests = self.make_fixture()
        fake = helpers.FakeUrlopen(
            {
                self.pointer_url("example/test", "w.gguf"): helpers.FakeResponse(
                    helpers.lfs_pointer(digests["w.gguf"], len(self.CONTENT))
                )
            }
        )
        code, output = self.run_main(fake, "--library-root", str(self.root))
        self.assertEqual(code, 0)
        self.assertIn("up to date", output)

    def test_update_available_exits_1(self):
        self.make_fixture()
        remote_digest = helpers.sha256_hex(b"updated upstream")
        fake = helpers.FakeUrlopen(
            {
                self.pointer_url("example/test", "w.gguf"): helpers.FakeResponse(
                    helpers.lfs_pointer(remote_digest, 123)
                )
            }
        )
        code, output = self.run_main(fake, "--library-root", str(self.root))
        self.assertEqual(code, 1)
        self.assertIn("Files with updates: 1", output)

    def test_model_filter_limits_requests(self):
        _, gemma_digests = helpers.make_model(
            self.root, "gemma-4", "google-test", {"w.gguf": self.CONTENT}
        )
        ornith_content = b"ornith weights"
        helpers.make_model(
            self.root,
            "ornith-1.5",
            "ornith-test",
            {"w.gguf": ornith_content},
            source="https://huggingface.co/example/ornith",
        )
        remote_digest = helpers.sha256_hex(b"ornith updated")
        fake = helpers.FakeUrlopen(
            {
                self.pointer_url("example/gemma", "w.gguf"): helpers.FakeResponse(
                    helpers.lfs_pointer(gemma_digests["w.gguf"], len(self.CONTENT))
                ),
                self.pointer_url("example/ornith", "w.gguf"): helpers.FakeResponse(
                    helpers.lfs_pointer(remote_digest, 456)
                ),
            }
        )
        code, output = self.run_main(fake, "--library-root", str(self.root), "-m", "ornith")
        self.assertEqual(code, 1)
        self.assertIn("Files with updates: 1", output)
        self.assertNotIn("gemma-4", output)
        self.assertEqual(
            [req.full_url for req in fake.requests],
            [self.pointer_url("example/ornith", "w.gguf")],
        )

    def test_no_matching_models_exits_1(self):
        self.make_fixture()
        fake = helpers.FakeUrlopen({})
        code, output = self.run_main(fake, "--library-root", str(self.root), "-m", "nothing")
        self.assertEqual(code, 1)
        self.assertIn("No model directories found matching", output)

    def test_empty_library_exits_1(self):
        fake = helpers.FakeUrlopen({})
        code, output = self.run_main(fake, "--library-root", str(self.root))
        self.assertEqual(code, 1)
        self.assertIn("No model directories containing README.md", output)

    def test_invalid_library_root_exits_2(self):
        fake = helpers.FakeUrlopen({})
        code, output = self.run_main(
            fake, "--library-root", str(self.root / "missing")
        )
        self.assertEqual(code, 2)
        self.assertIn("is not an existing directory", output)


if __name__ == "__main__":
    unittest.main()
