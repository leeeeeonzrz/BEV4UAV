#!/usr/bin/env python3
"""Check the code-only release; --staged inspects the actual Git index blobs.

This is a conservative publication guard, not a general-purpose secret detector.
It never prints matching content. Run before every push, including after git add.
"""

import argparse
import ast
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys


SOURCE_DIRS = {
    "mmdet3d", "projects", "configs", "tools", "requirements", "docs",
    "tests", ".github", "licenses",
}
ROOT_FILES = {
    "LICENSE", "NOTICE", "COPYING", "README.md", "README_zh-CN.md",
    "THIRD_PARTY_NOTICES.md", "CITATION.cff", "MANIFEST.in", "setup.py",
    "setup.cfg", "pyproject.toml", "requirements.txt", "Makefile",
    ".gitignore", ".gitattributes", ".pre-commit-config.yaml",
    "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md",
}
SOURCE_SUFFIXES = {
    ".py", ".pyi", ".cpp", ".cc", ".c", ".cu", ".cuh", ".h", ".hpp",
    ".sh", ".md", ".rst", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".in", ".cff",
}
PRIVATE_DIRS = {
    "data", "test_data", "sample_data", "samples", "checkpoints", "weights",
    "pretrained", "outputs", "results", "test_result", "work_dirs", "logs",
    "wandb", "mlruns", "paper_tools", "resources", "env_test", "build",
    "dist", "node_modules", "__pycache__", ".venv", "venv", ".vscode",
    ".idea", ".agents", ".codex", ".aws", ".ssh", ".git", "private",
}
FORBIDDEN_SUFFIXES = {
    ".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".engine", ".tflite",
    ".pkl", ".pickle", ".bin", ".npy", ".npz", ".h5", ".hdf5", ".db",
    ".sqlite", ".sqlite3", ".csv", ".tsv", ".json", ".jsonl", ".parquet",
    ".arrow", ".pcd", ".ply", ".las", ".laz", ".bag", ".mcap", ".tfrecord",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
    ".mp4", ".avi", ".mov", ".mp3", ".wav", ".pdf", ".docx", ".xlsx",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".deb", ".exe",
    ".so", ".dll", ".dylib", ".o", ".a", ".pyc", ".log", ".pem", ".key",
}
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024
SECRET_PATTERNS = (
    ("credential token", re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}"
        r"|AKIA[0-9A-Z]{16}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"
        r"|xox[baprs]-[A-Za-z0-9-]{20,})")),
    ("private key", re.compile(
        r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")),
    ("personal absolute path", re.compile(
        r"/(?:home|Users)/[A-Za-z0-9_.-]+(?:/|\\)"
        r"|(?<![A-Za-z:])[A-Za-z]:[\\/]Users[\\/][^\s\\/]+"
        r"|/" r"root/(?:\.ssh|\.aws|[^\s/]+/)")),
    ("embedded URL credentials", re.compile(
        r"https?://[^/\s:@]+:[^/\s@]+@")),
)


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def is_git_root(root):
    try:
        result = git(root, "rev-parse", "--show-toplevel")
        return Path(os.fsdecode(result.stdout).strip()).resolve() == root
    except (OSError, subprocess.CalledProcessError):
        return False


def path_issues(name):
    path = PurePosixPath(name)
    parts = path.parts
    if not parts or path.is_absolute() or ".." in parts:
        return ["invalid repository path"]
    lowered = [part.lower() for part in parts]
    problems = []
    if any(part in PRIVATE_DIRS or part.startswith("custom_train")
           or part.endswith(".egg-info") or part.startswith(".env")
           or part.startswith(".release-check-test-")
           for part in lowered):
        problems.append("private/generated directory or file")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        problems.append("data, weights, binary, or generated file type")
    if len(parts) == 1:
        if name not in ROOT_FILES:
            problems.append("file outside the source allowlist")
    elif parts[0] not in SOURCE_DIRS:
        problems.append("directory outside the source allowlist")
    elif path.suffix.lower() not in SOURCE_SUFFIXES:
        allowed_text = path.suffix == ".txt" and parts[0] in {
            "requirements", "licenses"}
        allowed_license = path.name in {"LICENSE", "NOTICE", "COPYING"}
        if not allowed_text and not allowed_license:
            problems.append("file outside the source allowlist")
    return problems


def inspect_blob(name, data):
    if len(data) > MAX_FILE_BYTES:
        return ["file exceeds 1 MiB source limit"]
    if any(byte < 9 or 13 < byte < 32 for byte in data):
        return ["binary/control bytes detected"]
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return ["file is not UTF-8 text"]
    problems = [label for label, pattern in SECRET_PATTERNS
                if pattern.search(content)]
    if name.endswith(".py"):
        try:
            ast.parse(content, filename=name)
        except (SyntaxError, ValueError) as error:
            line = getattr(error, "lineno", None)
            problems.append("invalid Python syntax" +
                            (" at line " + str(line) if line else ""))
    return problems


def filesystem_names(root):
    names = []
    for base, dirs, files in os.walk(root, followlinks=False):
        if Path(base) == root and ".git" in dirs:
            dirs.remove(".git")
        for directory in list(dirs):
            path = Path(base, directory)
            if path.is_symlink():
                names.append(path.relative_to(root).as_posix())
                dirs.remove(directory)
        names.extend(Path(base, filename).relative_to(root).as_posix()
                     for filename in files)
    return sorted(names)


def collect_entries(root, staged):
    if is_git_root(root):
        if staged:
            entries = []
            for record in git(root, "ls-files", "--stage", "-z").stdout.split(b"\0"):
                if not record:
                    continue
                metadata, raw_name = record.split(b"\t", 1)
                mode, object_id, stage = metadata.decode("ascii").split()
                name = raw_name.decode("utf-8")
                entries.append((name, mode, object_id, stage))
            return entries
        names = git(root, "ls-files", "-z").stdout.split(b"\0")
        return [(name.decode("utf-8"), None, None, None)
                for name in names if name]
    if staged:
        raise ValueError("--staged requires a Git repository at --root")
    return [(name, None, None, None) for name in filesystem_names(root)]


def check(root, staged=False):
    errors = []
    total_bytes = 0
    entries = collect_entries(root, staged)
    if not entries:
        return 0, 0, [(".", "no source files found")]
    for name, mode, object_id, stage in entries:
        problems = path_issues(name)
        if staged and stage != "0":
            problems.append("unmerged index entry")
        if mode is not None and mode not in {"100644", "100755"}:
            problems.append("symlinks/submodules are not allowed")
        if problems:
            errors.extend((name, problem) for problem in problems)
            continue
        if staged:
            size = int(git(root, "cat-file", "-s", object_id).stdout)
            if size > MAX_FILE_BYTES:
                errors.append((name, "file exceeds 1 MiB source limit"))
                continue
            data = git(root, "cat-file", "blob", object_id).stdout
        else:
            path = root / name
            try:
                info = path.lstat()
            except FileNotFoundError:
                errors.append((name, "tracked file is missing"))
                continue
            if not stat.S_ISREG(info.st_mode) or path.is_symlink():
                errors.append((name, "symlinks/nonregular files are not allowed"))
                continue
            if info.st_size > MAX_FILE_BYTES:
                errors.append((name, "file exceeds 1 MiB source limit"))
                continue
            data = path.read_bytes()
        total_bytes += len(data)
        errors.extend((name, problem) for problem in inspect_blob(name, data))
    if total_bytes > MAX_TOTAL_BYTES:
        errors.append((".", "source release exceeds 20 MiB total limit"))
    return len(entries), total_bytes, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--staged", action="store_true",
                        help="inspect all Git index paths and blobs")
    args = parser.parse_args()
    try:
        count, size, errors = check(args.root.resolve(), args.staged)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print("Release check could not complete: " + type(error).__name__,
              file=sys.stderr)
        return 2
    if errors:
        for name, reason in errors:
            print("FAIL " + name + ": " + reason, file=sys.stderr)
        print("Release check failed; do not publish.", file=sys.stderr)
        return 1
    print("Release check passed: %d source files, %d bytes (%s)." % (
        count, size, "Git index" if args.staged else "working files"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
