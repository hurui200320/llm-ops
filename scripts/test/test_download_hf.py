"""Tests for download_hf.py."""

import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import helpers
import download_hf


class DownloadHfTest(helpers.ScriptTestCase):
    URL = "https://huggingface.co/example/repo/resolve/main/w.gguf"
    POINTER_URL = "https://huggingface.co/example/repo/raw/main/w.gguf"

    PAYLOAD = b"model weights payload"

    def setUp(self):
        self.dest = self.make_temp_dir() / "dest"
        self.final_path = self.dest / "w.gguf"
        self.tmp_path = self.dest / "w.gguf.tmp"
        self.sha_path = self.dest / "w.gguf.sha256"
        self.digest = helpers.sha256_hex(self.PAYLOAD)

    def run_download(self, responses, *extra_args, url: str | None = None,
                     hf_token: str | None = None) -> tuple[object, helpers.FakeUrlopen, str]:
        url = url or self.URL
        fake = helpers.FakeUrlopen(responses)
        environ = dict(os.environ)
        environ.pop("HF_TOKEN", None)
        if hf_token is not None:
            environ["HF_TOKEN"] = hf_token
        stdout = io.StringIO()
        with mock.patch.dict(os.environ, environ, clear=True):
            with mock.patch.object(sys, "argv", ["download_hf", url, str(self.dest), *extra_args]):
                with mock.patch.object(download_hf, "urlopen", fake):
                    with contextlib.redirect_stdout(stdout):
                        try:
                            download_hf.main()
                            code = 0
                        except SystemExit as e:
                            code = e.code
        return code, fake, stdout.getvalue()

    def pointer_response(self, digest: str, size: int) -> helpers.FakeResponse:
        return helpers.FakeResponse(helpers.lfs_pointer(digest, size))

    def test_download_writes_file_and_companion(self):
        code, fake, _ = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
                self.URL: helpers.FakeResponse(self.PAYLOAD),
            }
        )
        self.assertEqual(code, 0)
        self.assertEqual(self.final_path.read_bytes(), self.PAYLOAD)
        self.assertEqual(self.sha_path.read_text(encoding="utf-8").strip(), self.digest)
        self.assertFalse(self.tmp_path.exists())

    def test_subdirectory_is_recreated(self):
        url = "https://huggingface.co/example/repo/resolve/main/BF16/w.gguf"
        pointer_url = "https://huggingface.co/example/repo/raw/main/BF16/w.gguf"
        code, fake, _ = self.run_download(
            {
                pointer_url: self.pointer_response(self.digest, len(self.PAYLOAD)),
                url: helpers.FakeResponse(self.PAYLOAD),
            },
            url=url,
        )
        self.assertEqual(code, 0)
        self.assertEqual((self.dest / "BF16" / "w.gguf").read_bytes(), self.PAYLOAD)

    def test_space_in_path_is_percent_encoded(self):
        url = "https://huggingface.co/example/repo/resolve/main/my file.gguf"
        pointer_url = "https://huggingface.co/example/repo/raw/main/my%20file.gguf"
        resolve_url = "https://huggingface.co/example/repo/resolve/main/my%20file.gguf"
        code, fake, _ = self.run_download(
            {
                pointer_url: self.pointer_response(self.digest, len(self.PAYLOAD)),
                resolve_url: helpers.FakeResponse(self.PAYLOAD),
            },
            url=url,
        )
        self.assertEqual(code, 0)
        self.assertEqual((self.dest / "my file.gguf").read_bytes(), self.PAYLOAD)
        self.assertEqual(
            [req.full_url for req in fake.requests], [pointer_url, resolve_url]
        )

    def test_percent_encoded_url_downloads_decoded_filename(self):
        url = "https://huggingface.co/example/repo/resolve/main/my%20file.gguf"
        pointer_url = "https://huggingface.co/example/repo/raw/main/my%20file.gguf"
        resolve_url = "https://huggingface.co/example/repo/resolve/main/my%20file.gguf"
        code, fake, _ = self.run_download(
            {
                pointer_url: self.pointer_response(self.digest, len(self.PAYLOAD)),
                resolve_url: helpers.FakeResponse(self.PAYLOAD),
            },
            url=url,
        )
        self.assertEqual(code, 0)
        self.assertEqual((self.dest / "my file.gguf").read_bytes(), self.PAYLOAD)
        self.assertEqual((self.dest / "my file.gguf.sha256").exists(), True)
        self.assertFalse((self.dest / "my%20file.gguf").exists())
        self.assertEqual(
            [req.full_url for req in fake.requests], [pointer_url, resolve_url]
        )

    def test_traversal_url_is_rejected_without_writes(self):
        url = "https://huggingface.co/example/repo/resolve/main/../outside.gguf"
        code, fake, output = self.run_download({}, url=url)
        self.assertEqual(code, 1)
        self.assertIn("Invalid file path", output)
        self.assertEqual(fake.requests, [])
        self.assertFalse(self.dest.exists())
        self.assertFalse((self.dest.parent / "outside.gguf").exists())

    def test_resume_sends_range_and_appends(self):
        split = 7
        self.dest.mkdir(parents=True)
        self.tmp_path.write_bytes(self.PAYLOAD[:split])
        code, fake, _ = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
                self.URL: helpers.FakeResponse(self.PAYLOAD[split:], status=206),
            }
        )
        self.assertEqual(code, 0)
        download_request = fake.requests[-1]
        self.assertEqual(download_request.get_header("Range"), f"bytes={split}-")
        self.assertEqual(self.final_path.read_bytes(), self.PAYLOAD)

    def test_server_ignoring_range_restarts_download(self):
        split = 7
        self.dest.mkdir(parents=True)
        self.tmp_path.write_bytes(self.PAYLOAD[:split])
        code, fake, _ = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
                self.URL: helpers.FakeResponse(self.PAYLOAD, status=200),
            }
        )
        self.assertEqual(code, 0)
        download_request = fake.requests[-1]
        self.assertEqual(download_request.get_header("Range"), f"bytes={split}-")
        self.assertEqual(self.final_path.read_bytes(), self.PAYLOAD)

    def test_complete_tmp_file_skips_download(self):
        self.dest.mkdir(parents=True)
        self.tmp_path.write_bytes(self.PAYLOAD)
        code, fake, _ = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
            }
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            [req.full_url for req in fake.requests], [self.POINTER_URL]
        )
        self.assertEqual(self.final_path.read_bytes(), self.PAYLOAD)

    def test_stale_tmp_removed_when_upstream_digest_changed(self):
        self.dest.mkdir(parents=True)
        self.tmp_path.write_bytes(b"stale partial of the old version")
        self.sha_path.write_text(helpers.sha256_hex(b"old version"), encoding="utf-8")
        new_payload = b"new version payload"
        new_digest = helpers.sha256_hex(new_payload)
        code, fake, output = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(new_digest, len(new_payload)),
                self.URL: helpers.FakeResponse(new_payload),
            }
        )
        self.assertEqual(code, 0)
        self.assertIn("stale", output)
        download_request = fake.requests[-1]
        self.assertIsNone(download_request.get_header("Range"))
        self.assertEqual(self.final_path.read_bytes(), new_payload)
        self.assertEqual(self.sha_path.read_text(encoding="utf-8").strip(), new_digest)
        self.assertFalse(self.tmp_path.exists())

    def test_digest_mismatch_deletes_tmp(self):
        other_digest = helpers.sha256_hex(b"different content")
        code, _, output = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(other_digest, len(self.PAYLOAD)),
                self.URL: helpers.FakeResponse(self.PAYLOAD),
            }
        )
        self.assertEqual(code, 1)
        self.assertIn("verification failed", output)
        self.assertFalse(self.tmp_path.exists())
        self.assertFalse(self.final_path.exists())

    def test_existing_up_to_date_file_skips_download(self):
        self.dest.mkdir(parents=True)
        self.final_path.write_bytes(self.PAYLOAD)
        self.sha_path.write_text(self.digest, encoding="utf-8")
        code, fake, output = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
            }
        )
        self.assertEqual(code, 0)
        self.assertIn("no download needed", output)
        self.assertEqual(
            [req.full_url for req in fake.requests], [self.POINTER_URL]
        )
        self.assertEqual(self.final_path.read_bytes(), self.PAYLOAD)

    def test_uppercase_local_digest_skips_download(self):
        self.dest.mkdir(parents=True)
        self.final_path.write_bytes(self.PAYLOAD)
        self.sha_path.write_text(self.digest.upper(), encoding="utf-8")
        code, fake, output = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
            }
        )
        self.assertEqual(code, 0)
        self.assertIn("no download needed", output)
        self.assertEqual(
            [req.full_url for req in fake.requests], [self.POINTER_URL]
        )
        self.assertEqual(self.final_path.read_bytes(), self.PAYLOAD)

    def test_corrupt_local_file_is_downloaded_again(self):
        self.dest.mkdir(parents=True)
        self.final_path.write_bytes(b"corrupt")
        self.sha_path.write_text(self.digest, encoding="utf-8")
        code, fake, output = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
                self.URL: helpers.FakeResponse(self.PAYLOAD),
            }
        )
        self.assertEqual(code, 0)
        self.assertIn("corrupt", output)
        self.assertEqual(self.final_path.read_bytes(), self.PAYLOAD)

    def test_pointer_http_error_exits_1(self):
        code, _, output = self.run_download(
            {
                self.POINTER_URL: helpers.http_error(self.POINTER_URL, 404),
            }
        )
        self.assertEqual(code, 1)
        self.assertIn("404", output)

    def test_token_sent_as_bearer_header(self):
        code, fake, _ = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
                self.URL: helpers.FakeResponse(self.PAYLOAD),
            },
            "--token",
            "hf_test_token",
        )
        self.assertEqual(code, 0)
        for request in fake.requests:
            self.assertEqual(request.get_header("Authorization"), "Bearer hf_test_token")

    def test_no_token_no_authorization_header(self):
        code, fake, _ = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
                self.URL: helpers.FakeResponse(self.PAYLOAD),
            }
        )
        self.assertEqual(code, 0)
        for request in fake.requests:
            self.assertIsNone(request.get_header("Authorization"))

    def test_hf_token_env_used_as_bearer_header(self):
        code, fake, _ = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
                self.URL: helpers.FakeResponse(self.PAYLOAD),
            },
            hf_token="hf_env_token",
        )
        self.assertEqual(code, 0)
        for request in fake.requests:
            self.assertEqual(request.get_header("Authorization"), "Bearer hf_env_token")

    def test_token_flag_overrides_hf_token_env(self):
        code, fake, _ = self.run_download(
            {
                self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
                self.URL: helpers.FakeResponse(self.PAYLOAD),
            },
            "--token",
            "hf_flag_token",
            hf_token="hf_env_token",
        )
        self.assertEqual(code, 0)
        for request in fake.requests:
            self.assertEqual(request.get_header("Authorization"), "Bearer hf_flag_token")

    def test_non_lfs_pointer_exits_1(self):
        code, _, output = self.run_download(
            {
                self.POINTER_URL: helpers.FakeResponse(b"plain text, not a pointer"),
            }
        )
        self.assertEqual(code, 1)
        self.assertIn("not an LFS pointer", output)

    def test_progress_bar_suppressed_when_not_a_tty(self):
        responses = {
            self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
            self.URL: helpers.FakeResponse(self.PAYLOAD),
        }
        with mock.patch.object(download_hf, "_TTY", False):
            code, _, output = self.run_download(responses)
        self.assertEqual(code, 0)
        self.assertNotIn("\r", output)
        self.assertNotIn("█", output)

    def test_progress_bar_drawn_when_tty(self):
        responses = {
            self.POINTER_URL: self.pointer_response(self.digest, len(self.PAYLOAD)),
            self.URL: helpers.FakeResponse(self.PAYLOAD),
        }
        with mock.patch.object(download_hf, "_TTY", True):
            code, _, output = self.run_download(responses)
        self.assertEqual(code, 0)
        self.assertIn("\r", output)


class ParseHfUrlTest(helpers.ScriptTestCase):
    def run_parse(self, url: str) -> tuple[object, str]:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            try:
                download_hf.parse_hf_url(url)
                code = 0
            except SystemExit as e:
                code = e.code
        return code, stdout.getvalue()

    def test_valid_url(self):
        self.assertEqual(
            download_hf.parse_hf_url(
                "https://huggingface.co/example/repo/resolve/main/BF16/w.gguf"
            ),
            ("example/repo", "main", "BF16/w.gguf"),
        )

    def test_short_host_alias_valid(self):
        self.assertEqual(
            download_hf.parse_hf_url("https://hf.co/example/repo/resolve/main/w.gguf"),
            ("example/repo", "main", "w.gguf"),
        )

    def test_percent_encoded_path_is_decoded(self):
        self.assertEqual(
            download_hf.parse_hf_url(
                "https://huggingface.co/example/repo/resolve/main/my%20file.gguf"
            ),
            ("example/repo", "main", "my file.gguf"),
        )

    def test_traversal_path_exits_1(self):
        for path in ("../../escape.gguf", "/etc/passwd", "BF16/../../escape.gguf"):
            with self.subTest(path=path):
                code, output = self.run_parse(
                    f"https://huggingface.co/example/repo/resolve/main/{path}"
                )
                self.assertEqual(code, 1)
                self.assertIn("Invalid file path", output)

    def test_unsupported_scheme_exits_1(self):
        code, output = self.run_parse("ftp://huggingface.co/a/b/resolve/main/w.gguf")
        self.assertEqual(code, 1)
        self.assertIn("Unsupported URL scheme", output)

    def test_unsupported_host_exits_1(self):
        code, output = self.run_parse("https://example.com/a/b/resolve/main/w.gguf")
        self.assertEqual(code, 1)
        self.assertIn("Unsupported hostname", output)

    def test_unexpected_path_exits_1(self):
        code, output = self.run_parse("https://huggingface.co/a/b")
        self.assertEqual(code, 1)
        self.assertIn("Unexpected URL path format", output)


if __name__ == "__main__":
    unittest.main()
