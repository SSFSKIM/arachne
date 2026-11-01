#!/usr/bin/env python3
"""Leak check: refuses commits and publication metadata that carry a denylisted term.

It reads every commit in a range and checks, for each one: every added line of a text file,
every changed file path, the commit message, and the author and committer (emails against an
allowlist, names against the denylist). Binary files are ruled by type: archives fail, ``.npz``
files pass only in the graphical units' control-data schema, a few image types pass after a
magic-byte check, and anything else binary fails. With ``--event`` (a GitHub event payload) or
``--branch`` it also checks a pull request's title, body and branch name, or a pushed ref's name.

The denylist is a secret. Matching is a case-insensitive substring match, and a failure names
where it is and the entry's position in the list, never the entry itself.

Exit status: 0 clean, 1 leak, 2 usage error or missing denylist.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import PurePosixPath

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
GITLINK_MODE = "160000"
ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".whl",
    ".egg",
    ".jar",
)
IMAGE_MAGIC = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".gif": (b"GIF87a", b"GIF89a"),
    ".webp": (b"RIFF",),
}
ALLOWED_EMAIL_PATTERNS = ("*@users.noreply.github.com", "noreply@github.com")
BINARY_SNIFF_BYTES = 8000  # git's own heuristic: a NUL byte in the first 8000 bytes


class UsageError(Exception):
    pass


class Denylist:
    def __init__(self, text: str):
        self.entries = [
            line.strip().lower()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self._any = (
            re.compile("|".join(re.escape(e) for e in self.entries))
            if self.entries
            else None
        )
        self._bytes = [e.encode("utf-8") for e in self.entries]
        self._any_bytes = (
            re.compile(b"|".join(re.escape(e) for e in self._bytes))
            if self._bytes
            else None
        )

    def hits(self, text: str) -> list[int]:
        """1-based positions of the entries found in ``text``."""
        if not text or self._any is None:
            return []
        low = text.lower()
        if not self._any.search(low):
            return []
        return [i for i, e in enumerate(self.entries, 1) if e in low]

    def hits_bytes(self, data: bytes) -> list[int]:
        low = data.lower()
        if self._any_bytes is None or not self._any_bytes.search(low):
            return []
        return [i for i, e in enumerate(self._bytes, 1) if e in low]


class Git:
    def __init__(self) -> None:
        self._batch: subprocess.Popen | None = None

    def run(self, *args: str) -> bytes:
        proc = subprocess.run(
            ["git", "-c", "core.quotePath=true", *args],
            capture_output=True,
        )
        if proc.returncode != 0:
            raise UsageError(
                f"git {args[0]} failed: {proc.stderr.decode(errors='replace').strip()}"
            )
        return proc.stdout

    def blob(self, sha: str) -> bytes:
        if self._batch is None:
            self._batch = subprocess.Popen(
                ["git", "cat-file", "--batch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        assert self._batch.stdin is not None
        assert self._batch.stdout is not None
        self._batch.stdin.write(sha.encode() + b"\n")
        self._batch.stdin.flush()
        header = self._batch.stdout.readline().split()
        if len(header) != 3:
            raise UsageError(f"cannot read blob {sha}")
        data = self._batch.stdout.read(int(header[2]))
        self._batch.stdout.read(1)  # the newline after the contents
        return data


def decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="surrogateescape")


def shown(path: str) -> str:
    """A path as it is safe to print (undecodable bytes escaped)."""
    return path.encode("utf-8", errors="surrogateescape").decode(
        "utf-8", errors="backslashreplace"
    )


_C_ESCAPES = {
    ord("a"): 7,
    ord("b"): 8,
    ord("t"): 9,
    ord("n"): 10,
    ord("v"): 11,
    ord("f"): 12,
    ord("r"): 13,
    ord('"'): 34,
    ord("\\"): 92,
}


def unquote_c(raw: bytes) -> bytes:
    """Undo git's C-style quoting of a path: a quoted name with backslash and octal escapes."""
    if not raw.startswith(b'"'):
        return raw
    body, out, i = raw[1:-1], bytearray(), 0
    while i < len(body):
        c = body[i]
        if c != 0x5C:
            out.append(c)
            i += 1
            continue
        nxt = body[i + 1]
        if 0x30 <= nxt <= 0x37:
            out.append(int(body[i + 1 : i + 4], 8))
            i += 4
        elif nxt in _C_ESCAPES:
            out.append(_C_ESCAPES[nxt])
            i += 2
        else:
            raise UsageError("cannot parse a quoted path")
    return bytes(out)


_HUNK = re.compile(rb"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def added_lines(patch: bytes) -> dict[str, list[tuple[int, str]]]:
    """Map each file in a ``-U0`` patch to its added lines as (line number, text)."""
    out: dict[str, list[tuple[int, str]]] = {}
    lines = patch.split(b"\n")
    path: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith(b"diff --git "):
            path = None
        elif line.startswith(b"+++ "):
            target = line[4:]
            # git ends the name with a tab when it holds a space; "" marks a deleted file
            if target.endswith(b"\t"):
                target = target[:-1]
            path = "" if target == b"/dev/null" else decode(unquote_c(target))[2:]
        elif line.startswith(b"@@ "):
            m = _HUNK.match(line)
            if m is None or path is None:
                raise UsageError("cannot parse a diff hunk")
            old_left = 1 if m.group(1) is None else int(m.group(1))
            new_left = 1 if m.group(3) is None else int(m.group(3))
            number = int(m.group(2))
            while (old_left > 0 or new_left > 0) and i < len(lines):
                body = lines[i]
                i += 1
                if body.startswith(b"+"):
                    if not path:
                        raise UsageError(
                            "cannot parse a diff hunk: an added line in a deleted file"
                        )
                    out.setdefault(path, []).append((number, decode(body[1:])))
                    number += 1
                    new_left -= 1
                elif body.startswith(b"-"):
                    old_left -= 1
                elif body.startswith(b" "):
                    number += 1
                    old_left -= 1
                    new_left -= 1
                elif not body.startswith(b"\\"):
                    raise UsageError("cannot parse a diff hunk body")
    return out


def changed_files(
    git: Git, parent: str, commit: str
) -> list[tuple[str, str, str, bool]]:
    """(path, status, new blob sha, git calls it binary) for every file changed against ``parent``."""
    raw = git.run(
        "diff-tree", "-r", "-z", "--no-renames", "--raw", "--no-abbrev", parent, commit
    )
    fields = raw.split(b"\0")[:-1]  # every field ends with a NUL
    entries = []
    for meta, path in zip(fields[0::2], fields[1::2], strict=True):
        _, mode, _, sha, status = decode(meta[1:]).split(" ")
        entries.append((decode(path), status, sha, mode))
    numstat = git.run(
        "diff-tree", "-r", "-z", "--no-renames", "--numstat", parent, commit
    )
    binary = set()
    for rec in numstat.split(b"\0"):
        if rec:
            added, _, path = rec.split(b"\t", 2)
            if added == b"-":
                binary.add(decode(path))
    return [
        (p, s, sha if mode != GITLINK_MODE else "", p in binary)
        for p, s, sha, mode in entries
    ]


def _only_zip64_fields(extra: bytes) -> bool:
    """Whether a zip extra field holds nothing but ZIP64 size records (id 0x0001)."""
    pos = 0
    while pos < len(extra):
        if pos + 4 > len(extra):
            return False
        kind, size = struct.unpack("<HH", extra[pos : pos + 4])
        if kind != 0x0001:
            return False
        pos += 4 + size
    return pos == len(extra)


def plain_container(data: bytes, z: zipfile.ZipFile) -> bool:
    """Whether a one-member zip has exactly the layout NumPy's writer emits, and nothing else.

    One local header at offset 0 (its extra field holding only ZIP64 sizes), the member's
    compressed bytes, the central directory right after them, an optional ZIP64 end record and
    locator, and an end record with no comment as the file's last 22 bytes: no archive or member
    comment, no data descriptor, and no gap, prefix or trailing bytes where text could hide.
    """
    (info,) = z.infolist()
    if len(data) < 22 or data[-22:-18] != b"PK\x05\x06" or data[-2:] != b"\0\0":
        return False  # the end record is not last, or it carries a comment
    if z.comment or info.comment or info.flag_bits & 0x08 or info.header_offset != 0:
        return False
    if not data.startswith(b"PK\x03\x04") or not _only_zip64_fields(info.extra):
        return False
    name_len, extra_len = struct.unpack("<HH", data[26:30])
    if not _only_zip64_fields(data[30 + name_len : 30 + name_len + extra_len]):
        return False
    if 30 + name_len + extra_len + info.compress_size != z.start_dir:
        return False  # bytes between the member and the central directory
    central_end = z.start_dir + 46 + len(info.filename.encode()) + len(info.extra)
    between = data[central_end : len(data) - 22]
    zip64_end = (
        len(between) == 76
        and between[:4] == b"PK\x06\x06"
        and between[56:60] == b"PK\x06\x07"
    )
    return between == b"" or zip64_end


def npz_in_schema(data: bytes) -> tuple[bool, bytes]:
    """Whether ``data`` is a graphical-units control file: one uint8 ``frame_data`` (H,W,C) or (N,H,W,C).

    When it is, also returns the decompressed ``frame_data.npy`` member, for the term scan. The
    member must be exactly an NPY header plus the array's bytes, because ``np.load`` ignores
    anything after the array.
    """
    try:
        import numpy as np
        from numpy.lib import format as npy
    except ImportError as exc:
        # the check cannot rule on an .npz without numpy, so it fails closed
        raise UsageError("numpy is needed to check .npz files") from exc
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if z.namelist() != ["frame_data.npy"] or not plain_container(data, z):
                return False, b""
            member = z.read("frame_data.npy")
        stream = io.BytesIO(member)
        version = npy.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = npy.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = npy.read_array_header_2_0(stream)
        else:
            return False, b""
    except Exception:
        return False, b""
    if dtype != np.uint8 or len(shape) not in (3, 4) or shape[-1] not in (3, 4):
        return False, b""
    size = 1
    for n in shape:
        size *= n
    if len(member) != stream.tell() + size * dtype.itemsize:
        return False, b""  # bytes after the array, or a short array
    return True, member


def check_binary(path: str, label: str, data: bytes, deny: Denylist) -> list[str]:
    """Rule a binary file; findings name it by ``label``, which is safe to print."""
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".npz":
        ok, member = npz_in_schema(data)
        if not ok:
            return [f"{label}: npz outside the control-data schema"]
        scanned = deny.hits_bytes(data) + deny.hits_bytes(member)
        return [
            f"{label}: denylisted term #{i} in binary content"
            for i in sorted(set(scanned))
        ]
    magic = IMAGE_MAGIC.get(suffix)
    if magic is None or not data.startswith(magic):
        return [f"{label}: binary file not allowed"]
    if suffix == ".webp" and data[8:12] != b"WEBP":
        return [f"{label}: binary file not allowed"]
    return [
        f"{label}: denylisted term #{i} in binary content"
        for i in deny.hits_bytes(data)
    ]


def check_commit(
    git: Git, sha: str, deny: Denylist, allowed_emails: set[str], stats: dict
) -> list[str]:
    problems = []
    short = sha[:12]
    meta = git.run("show", "-s", "--format=%P%x00%an%x00%ae%x00%cn%x00%ce%x00%B", sha)
    parents, author, author_email, committer, committer_email, message = decode(
        meta
    ).split("\0", 5)
    for role, name, email in (
        ("author", author, author_email),
        ("committer", committer, committer_email),
    ):
        problems += [f"commit {short}: {role} name term #{i}" for i in deny.hits(name)]
        email = email.lower()
        allowed = email in allowed_emails or any(
            fnmatch.fnmatchcase(email, p) for p in ALLOWED_EMAIL_PATTERNS
        )
        if not allowed:
            problems.append(f"commit {short}: {role} email not allowed")
        else:
            problems += [
                f"commit {short}: {role} email term #{i}" for i in deny.hits(email)
            ]
    problems += [f"commit {short}: message term #{i}" for i in deny.hits(message)]

    parent = parents.split()[0] if parents.split() else EMPTY_TREE
    files = changed_files(git, parent, sha)
    patch = git.run(
        "diff-tree", "-r", "-p", "-U0", "--no-renames", "--no-color", "--no-ext-diff", "--no-textconv",
        "--src-prefix=a/", "--dst-prefix=b/", parent, sha,
    )  # fmt: skip
    added = added_lines(patch)
    known = {path for path, *_ in files}
    if not set(added) <= known:
        raise UsageError(
            f"commit {short}: a patch path does not match the changed files"
        )

    for n, (path, status, blob_sha, git_binary) in enumerate(files, 1):
        if status == "D":
            continue
        stats["files"] += 1
        path_hits = deny.hits(path)
        problems += [
            f"commit {short}: file path denylisted term #{i}" for i in path_hits
        ]
        where = f"commit {short} file #{n}" if path_hits else shown(path)
        if PurePosixPath(path).name.lower().endswith(ARCHIVE_SUFFIXES):
            problems.append(f"{where}: archive not allowed")
            continue
        if not blob_sha:  # a submodule: its path is checked, it has no content here
            continue
        data = git.blob(blob_sha)
        if (
            git_binary
            or b"\0" in data[:BINARY_SNIFF_BYTES]
            or path.lower().endswith(".npz")
        ):
            problems += check_binary(path, where, data, deny)
            continue
        for number, text in added.get(path, []):
            stats["lines"] += 1
            problems += [
                f"{where}:{number}: denylisted term #{i}" for i in deny.hits(text)
            ]
    return problems


def check_metadata(event: dict, branch: str | None, deny: Denylist) -> list[str]:
    fields: list[tuple[str, str]] = []
    pr = event.get("pull_request")
    if pr:
        fields += [
            ("pull request title", pr.get("title") or ""),
            ("pull request body", pr.get("body") or ""),
            ("pull request branch", (pr.get("head") or {}).get("ref") or ""),
        ]
    elif event.get("ref"):
        ref = event["ref"]
        for prefix in ("refs/heads/", "refs/tags/"):
            if ref.startswith(prefix):
                ref = ref[len(prefix) :]
        fields.append(("branch name", ref))
    if branch:
        fields.append(("branch name", branch))
    return [
        f"{label}: denylisted term #{i}"
        for label, value in fields
        for i in deny.hits(value)
    ]


def parse_range(text: str) -> tuple[str, str]:
    base, sep, head = text.partition("..")
    if (
        not sep
        or not base
        or not head
        or head.startswith(".")
        or any(
            part.startswith("-") or any(c.isspace() for c in part)
            for part in (base, head)
        )
    ):
        raise UsageError(f"--range must look like <base>..<head>, got {text!r}")
    return base, head


def load_denylist(args: argparse.Namespace) -> Denylist:
    if args.denylist_env:
        deny = Denylist(os.environ.get(args.denylist_env, ""))
        missing = f"secret {args.denylist_env} unset"
    else:
        try:
            with open(args.denylist_file, encoding="utf-8") as fh:
                deny = Denylist(fh.read())
        except OSError:
            deny = Denylist("")
        missing = "denylist file missing or empty"
    if not deny.entries:
        raise UsageError(f"leak-check: no denylist ({missing})")
    return deny


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n\n")[0])
    parser.add_argument(
        "--range", required=True, help="<base>..<head>: the commits to check"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--denylist-env",
        metavar="VAR",
        help="read the denylist from this environment variable",
    )
    source.add_argument(
        "--denylist-file", metavar="PATH", help="read the denylist from this file"
    )
    parser.add_argument(
        "--event",
        metavar="PAYLOAD",
        help="a GitHub event payload (JSON) to check its metadata",
    )
    parser.add_argument("--branch", help="the name of the branch being pushed")
    parser.add_argument(
        "--allow-emails-from",
        metavar="REV",
        help="also allow every author and committer email in REV's history",
    )
    args = parser.parse_args(argv)

    try:
        deny = load_denylist(args)
        base, head = parse_range(args.range)
        git = Git()
        allowed_emails: set[str] = set()
        if args.allow_emails_from:
            log = git.run(
                "log", "--format=%ae%n%ce", "--end-of-options", args.allow_emails_from
            )
            allowed_emails = {
                e.strip().lower() for e in decode(log).splitlines() if e.strip()
            }
        event = {}
        if args.event:
            with open(args.event, encoding="utf-8") as fh:
                event = json.load(fh)
        commits = decode(
            git.run("rev-list", "--reverse", "--end-of-options", f"{base}..{head}")
        ).split()
        stats = {"files": 0, "lines": 0}
        problems = check_metadata(event, args.branch, deny)
        for sha in commits:
            problems += check_commit(git, sha, deny, allowed_emails, stats)
    except UsageError as exc:
        message = str(exc)
        print(
            message if message.startswith("leak-check:") else f"leak-check: {message}"
        )
        return 2

    for problem in problems:
        print(problem)
    if problems:
        print(
            f"leak-check: {len(problems)} problem(s) in {len(commits)} commit(s); the terms are not printed"
        )
        return 1
    print(
        f"leak-check: clean ({len(commits)} commits, {stats['files']} files, {stats['lines']} added lines)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
