"""The release guard must inspect staged blobs, even after a working-file edit."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_release", ROOT / "tools" / "check_release.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


@unittest.skipUnless(shutil.which("git"), "Git is required for index tests")
class ReleaseCheckTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".release-check-test-", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.run_git("init", "--quiet")
        self.write("README.md", "# Example source release\n")
        self.run_git("add", "README.md")

    def run_git(self, *args):
        subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, name, content):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def errors(self):
        return CHECKER.check(self.repo, staged=True)[2]

    def test_safe_source_passes(self):
        self.write("tools/example.py", "def add(a, b):\n    return a + b\n")
        self.run_git("add", "tools/example.py")
        self.assertEqual(self.errors(), [])

    def test_force_added_weights_fail(self):
        self.write(".gitignore", "*.pth\n")
        self.write("tools/model.pth", "model payload")
        self.run_git("add", ".gitignore")
        self.run_git("add", "--force", "tools/model.pth")
        self.assertTrue(any(name == "tools/model.pth"
                            for name, _ in self.errors()))

    def test_staged_symlink_fails(self):
        path = self.repo / "tools/link.py"
        path.parent.mkdir(exist_ok=True)
        path.symlink_to("../README.md")
        self.run_git("add", "tools/link.py")
        self.assertTrue(any("symlink" in reason for _, reason in self.errors()))

    def test_token_is_found_in_index_after_working_copy_is_cleaned(self):
        value = "ghp_" + "x" * 40
        self.write("tools/example.py", "TOKEN = " + repr(value) + "\n")
        self.run_git("add", "tools/example.py")
        self.write("tools/example.py", "TOKEN = None\n")
        self.assertEqual(CHECKER.check(self.repo, staged=False)[2], [])
        self.assertTrue(any(reason == "credential token"
                            for _, reason in self.errors()))

    def test_personal_paths_and_invalid_python_fail(self):
        personal = "/" + "home/" + "example/private/file"
        self.write("tools/paths.py", "LOCATION = " + repr(personal) + "\n")
        self.write("tools/broken.py", "def unfinished(\n")
        self.run_git("add", "tools")
        reasons = [reason for _, reason in self.errors()]
        self.assertIn("personal absolute path", reasons)
        self.assertTrue(any("invalid Python syntax" in reason for reason in reasons))

    def test_binary_disguised_as_source_fails(self):
        path = self.write("tools/binary.py", "")
        path.write_bytes(bytes([0, 1, 2, 3]))
        self.run_git("add", "tools/binary.py")
        self.assertTrue(any("binary" in reason for _, reason in self.errors()))

    def test_non_git_source_allowlist_and_large_file(self):
        plain = self.repo / "plain"
        plain.mkdir()
        (plain / "README.md").write_text("# Public source\n")
        self.assertEqual(CHECKER.check(plain)[2], [])
        (plain / "README.md").write_bytes(b"x" * (CHECKER.MAX_FILE_BYTES + 1))
        (plain / "private").mkdir()
        (plain / "private/note.md").write_text("private note")
        errors = CHECKER.check(plain)[2]
        self.assertTrue(any("exceeds" in reason for _, reason in errors))
        self.assertTrue(any(name == "private/note.md" for name, _ in errors))


if __name__ == "__main__":
    unittest.main()
