"""Shared helpers for the maintenance-script unit tests."""

import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class FakeResponse(io.BytesIO):
    """Bytes response mimicking the urlopen() result attributes the scripts use."""

    def __init__(self, content: bytes, status: int = 200):
        super().__init__(content)
        self.status = status


class FakeUrlopen:
    """
    urlopen() replacement backed by a canned URL -> response mapping.

    Responses may be FakeResponse instances or exception instances to raise.
    Every received request object is recorded in self.requests for assertions
    on headers (e.g. Range) and call counts.
    """

    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.requests: list[Request] = []

    def __call__(self, req: Request, timeout: float | None = None):
        url = req.full_url
        self.requests.append(req)
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


class ScriptTestCase(unittest.TestCase):
    """Base test case with temporary-directory plumbing."""

    def make_temp_dir(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)


def make_model(
    root: Path,
    family: str,
    variant: str,
    files: dict[str, bytes],
    source: str | None = "https://huggingface.co/example/test",
) -> tuple[Path, dict[str, str]]:
    """
    Create <root>/<family>/<variant>/ with a model README and GGUF files.

    Each entry in files maps a repository-relative path to its content bytes;
    a .sha256 companion holding the real digest is written next to it. When
    source is None no README is created; when source is an empty string the
    README has no front matter. Returns the model directory and the digests.
    """
    model_dir = root / family / variant
    model_dir.mkdir(parents=True, exist_ok=True)
    if source is not None:
        readme = model_dir / "README.md"
        if source == "":
            readme.write_text("# model without front matter\n", encoding="utf-8")
        else:
            readme.write_text(
                f"---\nsource: {source}\n---\n\n# Test model\n", encoding="utf-8"
            )

    digests: dict[str, str] = {}
    for rel, content in files.items():
        path = model_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digest = sha256_hex(content)
        path.with_name(path.name + ".sha256").write_text(digest, encoding="utf-8")
        digests[rel] = digest
    return model_dir, digests


def lfs_pointer(sha256: str, size: int) -> bytes:
    """Render a Hugging Face LFS pointer document."""
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{sha256}\n"
        f"size {size}\n"
    ).encode("utf-8")


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def http_error(url: str, code: int) -> HTTPError:
    return HTTPError(url, code, "HTTP error", hdrs=None, fp=None)
