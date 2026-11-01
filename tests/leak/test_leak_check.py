"""Tests for scripts/leak_check.py, on throwaway git repositories and synthetic denylists only."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "leak_check.py"

TERMS = ["zorblaxian", "quux-hollow.example", "frobnitz corp"]
CLEAN_EMAIL = "someone@users.noreply.github.com"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class Repo:
    def __init__(self, path: Path):
        self.path = path
        path.mkdir()
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": str(path),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        self.git("init", "-q", "-b", "main")
        self.base = self.commit({"README.md": "a clean start\n"}, "Initial commit")

    def git(self, *args: str, env: dict | None = None) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.path,
            env={**self.env, **(env or {})},
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def commit(
        self,
        files: dict[str, str | bytes | None],
        message: str = "Change",
        email: str = CLEAN_EMAIL,
        name: str = "Someone",
    ) -> str:
        for rel, content in files.items():
            target = self.path / rel
            if content is None:
                self.git("rm", "-q", rel)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")
            self.git("add", rel)
        who = {
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
        }
        self.git("commit", "-q", "--allow-empty", "-m", message, env=who)
        return self.git("rev-parse", "HEAD")


def run_check(
    repo: Repo,
    *extra: str,
    terms: list[str] | None = None,
    rng: str | None = None,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    args = [sys.executable, str(SCRIPT), "--range", rng or f"{repo.base}..HEAD"]
    if "--denylist-env" not in extra:
        listfile = repo.path.parent / "denylist.txt"
        listfile.write_text(
            "# synthetic\n" + "\n".join(TERMS if terms is None else terms) + "\n"
        )
        args += ["--denylist-file", str(listfile)]
    result = subprocess.run(
        [*args, *extra],
        cwd=repo.path,
        env={**repo.env, **(env or {})},
        capture_output=True,
        text=True,
    )
    for term in TERMS:
        assert term not in (result.stdout + result.stderr).lower(), "a term was printed"
    return result


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return Repo(tmp_path / "repo")


def npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


# --- the clean case -------------------------------------------------------------------------------


def test_clean_range_passes(repo: Repo):
    repo.commit({"src/a.py": "x = 1\n", "docs/b.md": "plain words\n"}, "Add files")
    result = run_check(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout


# --- commit contents ------------------------------------------------------------------------------


def test_added_line_fails_with_location(repo: Repo):
    repo.commit({"src/a.py": "x = 1\nurl = 'https://ZORBLAXIAN.dev'\n"})
    result = run_check(repo)
    assert result.returncode == 1
    assert "src/a.py:2: denylisted term #1" in result.stdout


def test_multi_word_term_matches(repo: Repo):
    repo.commit({"notes.txt": "made by Frobnitz Corp in 2026\n"})
    result = run_check(repo)
    assert result.returncode == 1
    assert "notes.txt:1: denylisted term #3" in result.stdout


def test_line_numbers_follow_the_new_file(repo: Repo):
    repo.commit({"a.txt": "one\ntwo\nthree\nfour\n"})
    repo.commit({"a.txt": "one\ntwo\nthree\nquux-hollow.example\nfour\n"})
    result = run_check(repo)
    assert "a.txt:4: denylisted term #2" in result.stdout


def test_removed_line_passes(repo: Repo):
    first = repo.commit({"a.txt": "keep\nzorblaxian\n"})
    repo.commit({"a.txt": "keep\n"})
    result = run_check(repo, rng=f"{first}..HEAD")
    assert result.returncode == 0, result.stdout


def test_deleted_file_passes(repo: Repo):
    first = repo.commit({"old/zorblaxian.txt": "zorblaxian\nmore\n"})
    repo.commit({"old/zorblaxian.txt": None, "new.txt": "fine\n"})
    result = run_check(repo, rng=f"{first}..HEAD")
    assert result.returncode == 0, result.stdout


def test_term_added_then_removed_inside_the_range_still_fails(repo: Repo):
    repo.commit({"a.txt": "zorblaxian\n"})
    repo.commit({"a.txt": "fine\n"})
    result = run_check(repo)
    assert result.returncode == 1
    assert "a.txt:1: denylisted term #1" in result.stdout


def test_path_fails_without_naming_the_path(repo: Repo):
    sha = repo.commit({"docs/zorblaxian-notes.md": "nothing here\n"})
    result = run_check(repo)
    assert result.returncode == 1
    assert f"commit {sha[:12]}: file path denylisted term #1" in result.stdout


def test_binary_path_is_checked(repo: Repo):
    repo.commit({"img/zorblaxian.png": PNG_MAGIC + b"\0\0data"})
    result = run_check(repo)
    assert result.returncode == 1
    assert "file path denylisted term #1" in result.stdout


def test_commit_message_fails(repo: Repo):
    sha = repo.commit({"a.txt": "fine\n"}, "Sync with quux-hollow.example")
    result = run_check(repo)
    assert result.returncode == 1
    assert f"commit {sha[:12]}: message term #2" in result.stdout


def test_merge_commit_content_is_checked(repo: Repo):
    repo.git("switch", "-q", "-c", "side")
    repo.commit({"side.txt": "side work\n"})
    repo.git("switch", "-q", "main")
    repo.commit({"main.txt": "main work\n"})
    repo.git(
        "merge",
        "-q",
        "--no-commit",
        "side",
        env={"GIT_AUTHOR_NAME": "x", "GIT_COMMITTER_NAME": "x"},
    )
    (repo.path / "resolved.txt").write_text("zorblaxian\n")
    repo.git("add", "resolved.txt")
    who = {
        "GIT_AUTHOR_NAME": "Someone",
        "GIT_AUTHOR_EMAIL": CLEAN_EMAIL,
        "GIT_COMMITTER_NAME": "Someone",
        "GIT_COMMITTER_EMAIL": CLEAN_EMAIL,
    }
    repo.git("commit", "-q", "-m", "Merge side", env=who)
    result = run_check(repo)
    assert result.returncode == 1
    assert "resolved.txt:1: denylisted term #1" in result.stdout


@pytest.mark.parametrize(
    "name", ['odd "name"\tfile.txt', "with space.txt", "ünïcode.txt"]
)
def test_unusual_paths_are_parsed(repo: Repo, name: str):
    repo.commit({name: "line\nzorblaxian\n"})
    result = run_check(repo)
    assert result.returncode == 1, result.stdout
    assert ":2: denylisted term #1" in result.stdout


# --- author and committer -------------------------------------------------------------------------


def test_private_author_email_fails_without_naming_it(repo: Repo):
    sha = repo.commit({"a.txt": "fine\n"}, email="person@private-mail.test")
    result = run_check(repo)
    assert result.returncode == 1
    assert f"commit {sha[:12]}: author email not allowed" in result.stdout
    assert f"commit {sha[:12]}: committer email not allowed" in result.stdout
    assert "private-mail" not in result.stdout


def test_github_noreply_emails_pass(repo: Repo):
    repo.commit({"a.txt": "one\n"}, email="12345+someone@users.noreply.github.com")
    repo.commit({"b.txt": "two\n"}, email="noreply@github.com")
    assert run_check(repo).returncode == 0


def test_emails_from_the_fork_base_history_are_allowed(tmp_path: Path):
    repo = Repo(tmp_path / "repo")
    old = repo.commit({"old.txt": "upstream\n"}, email="dev@upstream.test")
    repo.git("tag", "fork-base", old)
    repo.commit({"new.txt": "ours\n"}, email="dev@upstream.test")
    assert run_check(repo, rng=f"{old}..HEAD").returncode == 1
    assert (
        run_check(
            repo, "--allow-emails-from", "fork-base", rng=f"{old}..HEAD"
        ).returncode
        == 0
    )


def test_author_name_is_checked(repo: Repo):
    sha = repo.commit({"a.txt": "fine\n"}, name="Zorblaxian Bot")
    result = run_check(repo)
    assert result.returncode == 1
    assert f"commit {sha[:12]}: author name term #1" in result.stdout


# --- publication metadata -------------------------------------------------------------------------


def test_pull_request_body_fails(repo: Repo, tmp_path: Path):
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "pull_request": {
                    "title": "A fine title",
                    "body": "See the notes on quux-hollow.example",
                    "head": {"ref": "feature/ok"},
                }
            }
        )
    )
    result = run_check(repo, "--event", str(event))
    assert result.returncode == 1
    assert "pull request body: denylisted term #2" in result.stdout
    assert "title" not in result.stdout


def test_pull_request_title_and_branch_fail(repo: Repo, tmp_path: Path):
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "pull_request": {
                    "title": "Port the Zorblaxian fix",
                    "body": None,
                    "head": {"ref": "frobnitz corp-fix"},
                }
            }
        )
    )
    result = run_check(repo, "--event", str(event))
    assert result.returncode == 1
    assert "pull request title: denylisted term #1" in result.stdout
    assert "pull request branch: denylisted term #3" in result.stdout


def test_push_event_branch_name_fails(repo: Repo, tmp_path: Path):
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps({"ref": "refs/heads/try-zorblaxian", "before": "0" * 40})
    )
    result = run_check(repo, "--event", str(event))
    assert result.returncode == 1
    assert "branch name: denylisted term #1" in result.stdout


def test_branch_flag_fails(repo: Repo):
    result = run_check(repo, "--branch", "wip/quux-hollow.example")
    assert result.returncode == 1
    assert "branch name: denylisted term #2" in result.stdout


# --- binary files ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["bundle.zip", "src.tar.gz", "pkg.whl", "data.7z"])
def test_archive_fails(repo: Repo, name: str):
    repo.commit({f"assets/{name}": b"PK\x03\x04\0\0binary"})
    result = run_check(repo)
    assert result.returncode == 1
    assert f"assets/{name}: archive not allowed" in result.stdout


def test_archive_forced_to_text_still_fails(repo: Repo):
    repo.commit({".gitattributes": "*.zip diff\n", "a.zip": "plain text pretending\n"})
    result = run_check(repo)
    assert result.returncode == 1
    assert "a.zip: archive not allowed" in result.stdout


def test_well_formed_control_data_passes(repo: Repo):
    stack = np.zeros((3, 8, 10, 4), dtype=np.uint8)
    single = np.full((8, 10, 3), 200, dtype=np.uint8)
    repo.commit(
        {
            "tests/control_data/Stack.npz": npz_bytes(frame_data=stack),
            "tests/control_data/Single.npz": npz_bytes(frame_data=single),
        }
    )
    result = run_check(repo)
    assert result.returncode == 0, result.stdout


def test_npz_with_pickled_object_fails(repo: Repo):
    arr = np.empty(1, dtype=object)
    arr[0] = {"note": "anything"}
    buf = io.BytesIO()
    np.savez(buf, frame_data=arr, allow_pickle=True)
    repo.commit({"c.npz": buf.getvalue()})
    result = run_check(repo)
    assert result.returncode == 1
    assert "c.npz: npz outside the control-data schema" in result.stdout


@pytest.mark.parametrize(
    "arrays",
    [
        {"frame_data": np.array([["private words here"]] * 2)},  # <U dtype
        {"frame_data": np.array([[[b"bytes text"]]])},  # S dtype
        {
            "frame_data": np.zeros((2, 2, 3), dtype=np.uint8),
            "notes": np.zeros((2, 2, 3), dtype=np.uint8),
        },  # a second member
        {"other": np.zeros((2, 2, 3), dtype=np.uint8)},  # wrong name
        {"frame_data": np.zeros((2, 2, 3), dtype=np.float32)},  # wrong dtype
        {"frame_data": np.zeros((2, 2, 5), dtype=np.uint8)},  # wrong channel count
        {"frame_data": np.zeros((4, 4), dtype=np.uint8)},  # wrong rank
    ],
    ids=[
        "unicode",
        "bytes",
        "second-member",
        "wrong-name",
        "float",
        "channels",
        "rank",
    ],
)
def test_npz_outside_the_schema_fails(repo: Repo, arrays: dict):
    repo.commit({"tests/control_data/X.npz": npz_bytes(**arrays)})
    result = run_check(repo)
    assert result.returncode == 1
    assert (
        "tests/control_data/X.npz: npz outside the control-data schema" in result.stdout
    )


def test_npz_with_text_in_its_pixels_fails(repo: Repo):
    text = np.frombuffer(b"zorblaxian!!", dtype=np.uint8).reshape(1, 1, 4, 3)
    repo.commit({"t.npz": npz_bytes(frame_data=text)})
    result = run_check(repo)
    assert result.returncode == 1
    assert "t.npz: denylisted term #1 in binary content" in result.stdout


def test_npz_with_extra_zip_member_fails(repo: Repo):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("frame_data.npy", b"\x93NUMPY")
        z.writestr("readme.txt", "hello")
    repo.commit({"z.npz": buf.getvalue()})
    assert "z.npz: npz outside the control-data schema" in run_check(repo).stdout


def control_npz() -> bytes:
    return npz_bytes(frame_data=np.zeros((2, 4, 5, 4), dtype=np.uint8))


def test_npz_with_an_archive_comment_fails(repo: Repo):
    buf = io.BytesIO(control_npz())
    with zipfile.ZipFile(buf, "a") as z:
        z.comment = b"notes from zorblaxian"
    repo.commit({"tests/control_data/C.npz": buf.getvalue()})
    result = run_check(repo)
    assert result.returncode == 1
    assert (
        "tests/control_data/C.npz: npz outside the control-data schema" in result.stdout
    )


@pytest.mark.parametrize("extra", [b"zorblaxian", b"plain trailing words"])
def test_npz_with_trailing_bytes_fails(repo: Repo, extra: bytes):
    repo.commit({"tests/control_data/T.npz": control_npz() + extra})
    result = run_check(repo)
    assert result.returncode == 1
    assert (
        "tests/control_data/T.npz: npz outside the control-data schema" in result.stdout
    )


def test_npz_with_leading_bytes_fails(repo: Repo):
    repo.commit({"tests/control_data/L.npz": b"zorblaxian" + control_npz()})
    result = run_check(repo)
    assert result.returncode == 1
    assert "L.npz: npz outside the control-data schema" in result.stdout


def test_uncompressed_npz_container_bytes_are_scanned(repo: Repo):
    text = np.frombuffer(b"zorblaxian!!", dtype=np.uint8).reshape(1, 1, 4, 3)
    buf = io.BytesIO()
    np.savez(buf, frame_data=text)
    repo.commit({"u.npz": buf.getvalue()})
    result = run_check(repo)
    assert result.returncode == 1
    assert result.stdout.count("u.npz: denylisted term #1 in binary content") == 1


@pytest.mark.parametrize(
    "name", ["a: zorblaxian.dat", "x: y: zorblaxian.npz", "dir: zorblaxian/b.png"]
)
def test_binary_finding_never_names_a_term_bearing_path(repo: Repo, name: str):
    sha = repo.commit({name: b"\0binary bytes"})
    result = run_check(repo)  # run_check asserts no term reaches stdout or stderr
    assert result.returncode == 1
    assert f"commit {sha[:12]}: file path denylisted term #1" in result.stdout
    assert f"commit {sha[:12]} file #1: " in result.stdout


def test_allowed_image_passes(repo: Repo):
    repo.commit({"docs/logo.png": PNG_MAGIC + b"\0\0\0\rIHDR rest"})
    assert run_check(repo).returncode == 0


def test_image_with_wrong_magic_fails(repo: Repo):
    repo.commit({"docs/logo.png": b"\0not an image at all"})
    result = run_check(repo)
    assert result.returncode == 1
    assert "docs/logo.png: binary file not allowed" in result.stdout


def test_image_carrying_a_term_fails(repo: Repo):
    repo.commit({"docs/logo.png": PNG_MAGIC + b"\0\0tEXtComment\0made at zorblaxian"})
    result = run_check(repo)
    assert result.returncode == 1
    assert "docs/logo.png: denylisted term #1 in binary content" in result.stdout


def test_other_binary_fails(repo: Repo):
    repo.commit({"fonts/a.ttf": b"\0\1\0\0binary font"})
    result = run_check(repo)
    assert result.returncode == 1
    assert "fonts/a.ttf: binary file not allowed" in result.stdout


def test_text_file_marked_binary_fails_closed(repo: Repo):
    repo.commit({".gitattributes": "*.txt binary\n", "hidden.txt": "zorblaxian\n"})
    result = run_check(repo)
    assert result.returncode == 1
    assert "hidden.txt: binary file not allowed" in result.stdout


# --- failing closed -------------------------------------------------------------------------------


def test_empty_denylist_env_fails_closed(repo: Repo):
    result = run_check(
        repo, "--denylist-env", "LEAK_DENYLIST", env={"LEAK_DENYLIST": ""}
    )
    assert result.returncode == 2
    assert (
        "leak-check: no denylist (secret LEAK_DENYLIST unset)"
        in result.stdout + result.stderr
    )


def test_comment_only_denylist_file_fails_closed(repo: Repo):
    result = run_check(repo, terms=["# only a comment", ""])
    assert result.returncode == 2
    assert "no denylist" in result.stdout + result.stderr


def test_denylist_from_env_is_used(repo: Repo):
    repo.commit({"a.txt": "zorblaxian\n"})
    result = run_check(
        repo,
        "--denylist-env",
        "LEAK_DENYLIST",
        env={"LEAK_DENYLIST": "\n".join(TERMS)},
    )
    assert result.returncode == 1
    assert "a.txt:1: denylisted term #1" in result.stdout


def test_bad_range_is_a_usage_error(repo: Repo):
    assert run_check(repo, rng="--output=x..HEAD").returncode == 2
    assert run_check(repo, rng="no-such-rev..HEAD").returncode == 2


class _MakesDirWhenUnpickled:
    def __init__(self, target: Path):
        self.target = target

    def __reduce__(self):
        return (os.mkdir, (str(self.target),))


def test_pickle_is_never_loaded(repo: Repo):
    marker = repo.path.parent / "unpickled"
    arr = np.empty(1, dtype=object)
    arr[0] = _MakesDirWhenUnpickled(marker)
    buf = io.BytesIO()
    np.savez(buf, frame_data=arr, allow_pickle=True)
    repo.commit({"p.npz": buf.getvalue()})
    assert run_check(repo).returncode == 1
    assert not marker.exists()
