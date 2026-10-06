#!/usr/bin/env python3
"""Pre-submission checker for a Python CTF model file.

Python port of the organizers' R tooling (``utils/R/local_testing.R::run_toy_tests`` and
``scripts/check_submission.R``) for single-file Python submissions. Rule numbers refer to
https://jkpfactors.com/ctf/rules (mirrored in the organizer repo, ``docs/ctf_rules.md``).

Usage::

    python tools/test_submission.py --model submission/prism.py \\
        --requirements submission/requirements.txt --data DIR [--data DIR2 ...] \\
        [--static-only] [--cutoffs N] [--quick] [--seed S] [--strict] [--pypi] [--weights CSV]

Each DIR holds ``ctff_chars.parquet``, ``ctff_features.parquet`` and
``ctff_daily_ret.parquet`` (docs/DEV_SPEC.md, section 2).

Static checks (run once):
  * Rule 13: file < 1 MB, no NUL bytes, valid UTF-8, parses as Python.
  * Rule 11: module-level ``def main(chars: pd.DataFrame, features: pd.DataFrame,
    daily_ret: pd.DataFrame) -> pd.DataFrame`` (exact names, no defaults/varargs/decorators,
    defined once) and ``main()`` is not called at import time.
  * Rules 10/15 via ``ast``: no network/shell/native-code imports, no ``os.system``-style or
    ``eval``/``exec``/``compile``/``__import__`` calls, no ``os.environ`` writes, no file I/O
    outside the ``if __name__ == "__main__":`` block (which may read parquet / write CSV, as
    in the official example), no hardcoded secrets. A literal text scan for the Rule 15
    keywords (what a regex-based scanner would see, comments included) is reported as WARN.
  * Rules 4/8/16: every non-stdlib import is pinned with ``==`` in requirements.txt, every
    requirement line is an exact PyPI pin; pins that differ from the local venv are WARN.
    With ``--pypi`` (network), each pin must exist on PyPI, allow Python 3.13 and ship a
    CPython 3.13 linux x86_64 wheel; vulnerabilities PyPI/OSV lists are WARN.

Dynamic checks (per data dir; the module is re-imported fresh for every run and every run
gets deep copies of the inputs, so neither module state nor input mutation leaks between runs):
  1. Output contract (Rules 11, 12, 5): columns exactly id, eom, w; id integer; w float and
     finite; eom a date (datetime.date or midnight datetime64) serialized as YYYY-MM-DD; no
     NaN; no duplicated (id, eom); exactly the ctff_test (id, eom) set; non-zero gross
     exposure every test month; CSV < 50 MB; main() leaves its inputs unchanged; no network
     connection attempted (in-process socket guard).
  2. Determinism (Rule 18): a second run matches within |a-b| <= atol + rtol*|b|.
  3. Order invariance (Rule 18): shuffled rows/feature list/column order; and alternative
     input dtypes (date objects <-> datetime64[ns], bool <-> int ctff_test, StringDtype ->
     object, int32 <-> int64 id) give the same weights.
  4. Lookahead (Rule 1): for each cutoff c (always the second-to-last test month, others
     spread evenly over the test period), rerun on chars[eom <= c], daily_ret[date <= c];
     weights for eom <= c must match the full run. Plus a strict variant the organizers do
     not run: at the last test month, daily returns after it are removed and that month's
     ret_exc_lead1m is replaced by noise (a model that uses a month's own lead return
     passes the plain truncation test but fails this one).
  5. Summaries: runtime/rows/months per run; gross/net exposure, nonzero weights and the
     annualized Sharpe ratio of sum(w * ret_exc_lead1m) by eom over the test months.

``--weights CSV`` also runs check_submission.R's checks on a saved weights CSV (a required
submission file, Rule 5) against the first --data dir, and compares it with main()'s output
there if the dynamic checks ran. ``--quick`` skips the shuffle and dtype runs and uses one
cutoff. Exit code 1 if any check FAILs (``--strict`` also fails on WARN).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import errno
import importlib.metadata
import importlib.util
import io
import re
import resource
import socket
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-5, 1e-8  # Rule 18
MAX_SOURCE_BYTES = 1_000_000  # Rule 13 (same 1e6 threshold as check_submission.R)
MAX_OUTPUT_BYTES = 50_000_000  # Rule 12
OUTPUT_COLUMNS = ["id", "eom", "w"]
MAIN_ARGS = ["chars", "features", "daily_ret"]
DATA_FILES = {"chars": "ctff_chars.parquet", "features": "ctff_features.parquet",
              "daily_ret": "ctff_daily_ret.parquet"}
ALLOWED_ENV_KEYS = {"CTF_EXECUTION_MODE"}  # documented in Rule 14

# ── Rule 10/15 vocabulary ───────────────────────────────────────────────────────────────
PROHIBITED_MODULES: dict[str, set[str]] = {
    "network": {"socket", "ssl", "urllib", "urllib3", "requests", "http.client", "http.server",
                "httpx", "aiohttp", "ftplib", "smtplib", "poplib", "imaplib", "telnetlib",
                "xmlrpc", "websocket", "websockets", "paramiko", "pycurl", "boto3", "botocore"},
    "shell / native code": {"subprocess", "pty", "ctypes", "cffi"},
    "dynamic code": {"runpy", "code", "codeop"},
}
SHELL_CALLS = {"os.system", "os.popen", "os.fork", "os.forkpty", "os.startfile",
               "os.posix_spawn", "os.posix_spawnp", "pty.spawn"}
SHELL_PREFIXES = ("os.exec", "os.spawn", "subprocess.")
DYNAMIC_CALLS = {"eval", "exec", "compile", "__import__", "builtins.eval", "builtins.exec",
                 "builtins.compile", "builtins.__import__", "importlib.import_module",
                 "importlib.__import__", "runpy.run_path", "runpy.run_module", "pandas.eval",
                 "numexpr.evaluate"}
ENV_WRITE_CALLS = {"os.putenv", "os.unsetenv"}
ENV_WRITE_METHODS = {"update", "setdefault", "pop", "popitem", "clear", "__setitem__",
                     "__delitem__"}
FILE_IO_CALLS = {
    "open", "builtins.open", "io.open", "io.FileIO", "os.open", "os.fdopen", "codecs.open",
    "os.remove", "os.unlink", "os.rmdir", "os.removedirs", "os.rename", "os.renames",
    "os.replace", "os.mkdir", "os.makedirs", "os.listdir", "os.scandir", "os.walk",
    "os.chdir", "os.chmod", "os.chown", "os.truncate", "os.link", "os.symlink",
    "glob.glob", "glob.iglob", "pickle.dump", "pickle.load", "json.dump", "json.load",
    "marshal.dump", "marshal.load", "shelve.open", "sqlite3.connect", "numpy.save",
    "numpy.savez", "numpy.savez_compressed", "numpy.savetxt", "numpy.load", "numpy.loadtxt",
    "numpy.genfromtxt", "numpy.fromfile", "numpy.memmap", "joblib.dump", "joblib.load",
    "joblib.Memory", "torch.save", "torch.load", "zipfile.ZipFile", "tarfile.open",
    "gzip.open", "bz2.open", "lzma.open", "pyarrow.parquet.ParquetFile",
    "pyarrow.dataset.dataset",
}
# Methods that touch the filesystem whatever the receiver is.
FILE_IO_METHODS = {
    "to_pickle", "to_excel", "to_feather", "to_hdf", "to_sql", "to_stata", "to_orc",
    "to_clipboard", "write_ipc", "write_ipc_stream", "write_excel", "write_database",
    "write_delta", "write_avro", "sink_parquet", "sink_csv", "sink_ipc", "sink_ndjson",
    "savefig", "tofile", "save_model", "load_model", "write_text", "write_bytes",
    "read_text", "read_bytes", "touch", "mkdir", "unlink", "rmdir", "iterdir", "rglob", "open",
}
# Methods that return a string/bytes when called without a target path.
FILE_IO_METHODS_WITH_PATH = {"to_csv", "to_json", "to_html", "to_latex", "to_markdown",
                             "to_xml", "to_parquet", "write_csv", "write_json",
                             "write_ndjson", "write_parquet"}
PATH_KEYWORDS = {"path_or_buf", "buf", "path", "file", "fname", "filename", "target"}
READER_RE = re.compile(
    r"^(read|scan)_(parquet|csv|csv_batched|pickle|excel|json|ndjson|feather|hdf|sql\w*|table|"
    r"fwf|html|xml|stata|sas|spss|orc|ipc\w*|avro|delta|database\w*|clipboard|iceberg)$")
NAIVE_SCAN_RE = re.compile(
    r"\b(eval|exec|compile)\s*\(|__import__|\bos\s*\.\s*(system|popen)\b|\bsubprocess\b|"
    r"\bsocket\b|\burllib\d?\b|\brequests\b|\bhttp\.client\b|\bhttpx\b")
SECRET_PATTERNS = {
    "AWS access key": r"\b(AKIA|ASIA)[0-9A-Z]{16}\b",
    "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "GitHub token": r"\b(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})",
    "Slack token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "sk- API key": r"\bsk-(ant-|proj-)?[A-Za-z0-9_-]{20,}",
    "Google API key": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "credentials in URL": r"\b[a-z][a-z0-9+.-]*://[^\s/:@'\"]+:[^\s/@'\"]+@",
    "JWT": r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
}
SECRET_NAME_RE = re.compile(
    r"(?i)(^|_)(password|passwd|pwd|secret|token|api_?key|access_?key|private_?key|credentials?|"
    r"auth_?token)($|_)")

# ── Rules 4/8 vocabulary ────────────────────────────────────────────────────────────────
# Import name -> PyPI distributions that provide it (any one of them may be pinned).
IMPORT_TO_DISTS: dict[str, set[str]] = {
    "sklearn": {"scikit-learn"}, "skimage": {"scikit-image"}, "yaml": {"PyYAML"},
    "cv2": {"opencv-python", "opencv-python-headless", "opencv-contrib-python",
            "opencv-contrib-python-headless"},
    "PIL": {"Pillow"}, "bs4": {"beautifulsoup4"}, "dateutil": {"python-dateutil"},
    "attr": {"attrs"}, "jwt": {"PyJWT"}, "Crypto": {"pycryptodome"}, "OpenSSL": {"pyOpenSSL"},
    "sksurv": {"scikit-survival"}, "umap": {"umap-learn"}, "google": {"protobuf"},
    "xgboost": {"xgboost", "xgboost-cpu"}, "tensorflow": {"tensorflow", "tensorflow-cpu"},
    "psycopg2": {"psycopg2", "psycopg2-binary"},
}
PREINSTALLED = {"pandas", "numpy", "pyarrow", "boto3", "scipy", "scikit-learn", "polars",
                "joblib"}  # Rule 8
REQ_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<extras>\[[A-Za-z0-9._,\s-]*\])?\s*==\s*"
    r"(?P<version>[0-9][A-Za-z0-9.+!_-]*)\s*(?:;\s*(?P<marker>.+))?$")


# ═════════════════════════════════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════════════════════════════════
class Report:
    """Collects PASS/FAIL/WARN lines in the style of check_submission.R."""

    def __init__(self, strict: bool) -> None:
        self.strict = strict
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            self._emit("PASS", label, f" ({detail})" if detail else "")
        else:
            self._emit("FAIL", label, f": {detail}" if detail else "")
            self.failures.append(label)
        return ok

    def warn(self, label: str, detail: str = "") -> None:
        self._emit("WARN", label, f": {detail}" if detail else "")
        self.warnings.append(label)

    def skip(self, label: str, reason: str) -> None:
        self._emit("SKIP", label, f" ({reason})")

    def info(self, message: str) -> None:
        self._emit("INFO", message, "")

    @staticmethod
    def section(title: str) -> None:
        print(f"\n== {title} ==", flush=True)

    @staticmethod
    def _emit(status: str, label: str, tail: str) -> None:
        print(f"{status}  {label}{tail}", flush=True)

    def exit_code(self) -> int:
        return 1 if self.failures or (self.strict and self.warnings) else 0


@dataclass
class Hit:
    """A source location flagged by a static check."""

    line: int
    text: str

    def __str__(self) -> str:
        return f"L{self.line} {self.text}"


def fmt_hits(hits: list[Hit], limit: int = 4) -> str:
    shown = "; ".join(str(h) for h in hits[:limit])
    return shown + (f"; ... (+{len(hits) - limit} more)" if len(hits) > limit else "")


def snippet(source: str, node: ast.AST, width: int = 70) -> str:
    text = ast.get_source_segment(source, node) or ""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


# ═════════════════════════════════════════════════════════════════════════════════════════
# Static checks
# ═════════════════════════════════════════════════════════════════════════════════════════
def is_main_guard(stmt: ast.stmt) -> bool:
    """True for a module-level ``if __name__ == "__main__":``."""
    if not isinstance(stmt, ast.If) or not isinstance(stmt.test, ast.Compare):
        return False
    test = stmt.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    sides = [test.left, test.comparators[0]]
    has_name = any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
    has_main = any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides)
    return has_name and has_main


def collect_aliases(tree: ast.Module) -> dict[str, str]:
    """Map each name bound by an import to the dotted path it refers to."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
                else:
                    top = a.name.split(".")[0]
                    aliases[top] = top
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for a in node.names:
                if a.name != "*":
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def prohibited_category(module: str) -> str | None:
    for category, names in PROHIBITED_MODULES.items():
        if any(module == n or module.startswith(n + ".") for n in names):
            return category
    return None


class SourceScanner(ast.NodeVisitor):
    """Walks the module and records Rule 10/15 violations and file I/O.

    File I/O is allowed only inside the module-level ``if __name__ == "__main__":`` block,
    which the CTF pipeline never executes; everything else is checked everywhere.
    """

    def __init__(self, source: str, tree: ast.Module) -> None:
        self.source = source
        self.aliases = collect_aliases(tree)
        self.local_defs = {n.name for n in ast.walk(tree)
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        self.hits: dict[str, list[Hit]] = defaultdict(list)
        self.imports: list[tuple[str, int]] = []  # (dotted module, line)
        self.env_reads: list[Hit] = []
        self._allow_io = False
        self._depth = 0  # >0 inside function/class bodies
        self._env_seen: set[int] = set()

    def scan(self, tree: ast.Module) -> SourceScanner:
        for stmt in tree.body:
            if is_main_guard(stmt):
                assert isinstance(stmt, ast.If)
                self._allow_io = True
                for s in stmt.body:
                    self.visit(s)
                self._allow_io = False
                for s in stmt.orelse:
                    self.visit(s)
            else:
                self.visit(stmt)
        return self

    # ── helpers ──
    def resolve(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def flag(self, category: str, node: ast.AST) -> None:
        self.hits[category].append(Hit(getattr(node, "lineno", 0), snippet(self.source, node)))

    def _env_access(self, node: ast.AST, write: bool, key: ast.AST | None = None) -> None:
        if write:
            self.flag("os.environ write", node)
        elif not (isinstance(key, ast.Constant) and key.value in ALLOWED_ENV_KEYS):
            self.env_reads.append(Hit(getattr(node, "lineno", 0), snippet(self.source, node)))

    # ── scopes ──
    def _visit_scope(self, node: ast.AST) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = visit_Lambda = _visit_scope

    # ── imports ──
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self.imports.append((a.name, node.lineno))
            if category := prohibited_category(a.name):
                self.flag(f"{category} import", node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self.flag("relative import", node)
            return
        module = node.module or ""
        self.imports.append((module, node.lineno))
        for a in node.names:
            full = f"{module}.{a.name}"
            category = prohibited_category(module) or prohibited_category(full)
            if category:
                self.flag(f"{category} import", node)
            elif full in SHELL_CALLS or full in DYNAMIC_CALLS:
                self.flag("shell / dynamic-code import", node)

    # ── calls ──
    def visit_Call(self, node: ast.Call) -> None:
        name = self.resolve(node.func)
        bare = isinstance(node.func, ast.Name)
        if name == "getattr" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            # getattr(os, "system") / getattr(builtins, "eval"): check the attribute it names
            target = f"{self.resolve(node.args[0])}.{node.args[1].value}"
            if target in SHELL_CALLS or target.startswith(SHELL_PREFIXES):
                self.flag("shell execution call", node)
            if target in DYNAMIC_CALLS:
                self.flag("dynamic code call", node)
        if any(k.arg in ("backend", "prefer") and isinstance(k.value, ast.Constant)
               and k.value.value in ("threading", "threads") for k in node.keywords):
            self.flag("joblib threading backend", node)
        if name:
            if name in SHELL_CALLS or name.startswith(SHELL_PREFIXES):
                self.flag("shell execution call", node)
            if name in DYNAMIC_CALLS and not (bare and name in self.local_defs):
                self.flag("dynamic code call", node)
            if name in ENV_WRITE_CALLS:
                self.flag("os.environ write", node)
            if name == "os.getenv":
                self._env_access(node, write=False, key=node.args[0] if node.args else None)
            if bare and name == "main" and self._depth == 0 and not self._allow_io:
                self.flag("main() called at import time", node)
        if isinstance(node.func, ast.Attribute) and self.resolve(node.func.value) == "os.environ":
            self._env_seen.add(id(node.func.value))
            method = node.func.attr
            self._env_access(node, write=method in ENV_WRITE_METHODS,
                             key=node.args[0] if node.args else None)
        if not self._allow_io and self._is_file_io(name, node):
            self.flag("file I/O outside the __main__ block", node)
        self.generic_visit(node)

    def _is_file_io(self, name: str | None, node: ast.Call) -> bool:
        if name and (name in FILE_IO_CALLS or name.startswith("shutil.")):
            return True
        if not isinstance(node.func, ast.Attribute):
            return False
        attr = node.func.attr
        if READER_RE.match(attr) or attr in FILE_IO_METHODS:
            return True
        if attr in FILE_IO_METHODS_WITH_PATH:
            first = node.args[0] if node.args else None
            positional = first is not None and not (isinstance(first, ast.Constant) and first.value is None)
            keyword = any(k.arg in PATH_KEYWORDS and not (isinstance(k.value, ast.Constant) and k.value.value is None)
                          for k in node.keywords)
            return positional or keyword
        return False

    # ── os.environ access ──
    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self.resolve(node.value) == "os.environ":
            self._env_seen.add(id(node.value))
            self._env_access(node, write=not isinstance(node.ctx, ast.Load), key=node.slice)
        self.generic_visit(node)

    def _plain_env_reference(self, node: ast.Attribute | ast.Name) -> None:
        if id(node) not in self._env_seen and self.resolve(node) == "os.environ":
            self._env_access(node, write=not isinstance(node.ctx, ast.Load))
        self.generic_visit(node)

    visit_Attribute = visit_Name = _plain_env_reference


def secret_hits(source: str, tree: ast.Module) -> list[Hit]:
    """Hardcoded credentials: known token formats anywhere, and secret-named string constants."""
    hits = []
    for lineno, line in enumerate(source.splitlines(), 1):
        for label, pattern in SECRET_PATTERNS.items():
            if re.search(pattern, line):
                hits.append(Hit(lineno, label))
    for node in ast.walk(tree):
        pairs: list[tuple[str, ast.AST]] = []
        if isinstance(node, ast.Assign):
            pairs = [(t.id, node.value) for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            pairs = [(node.target.id, node.value)]
        elif isinstance(node, ast.keyword) and node.arg:
            pairs = [(node.arg, node.value)]
        elif isinstance(node, ast.Dict):
            pairs = [(k.value, v) for k, v in zip(node.keys, node.values, strict=True)
                     if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        for name, value in pairs:
            if (SECRET_NAME_RE.search(name) and isinstance(value, ast.Constant)
                    and isinstance(value.value, str) and len(value.value) >= 8
                    and " " not in value.value):
                hits.append(Hit(getattr(value, "lineno", 0), f"{name} = <{len(value.value)}-char string>"))
    return hits


def annotation_is_dataframe(node: ast.AST | None, aliases: dict[str, str]) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return False
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.insert(0, node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return False
    resolved = ".".join([aliases.get(node.id, node.id), *parts])
    return resolved in {"pandas.DataFrame", "pandas.core.frame.DataFrame"}


def main_signature_problems(tree: ast.Module, aliases: dict[str, str]) -> list[str]:
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "main"]
    if not defs:
        return ["no module-level def main(...)"]
    problems = []
    rebinds = [n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom))
               and "main" in _bound_names(n)]
    if len(defs) > 1 or rebinds:
        problems.append("main is bound more than once at module level")
    fn = defs[-1]
    if isinstance(fn, ast.AsyncFunctionDef):
        problems.append("main is async")
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args]
    if names != MAIN_ARGS:
        problems.append(f"parameters are {names}, expected {MAIN_ARGS}")
    if a.posonlyargs or a.vararg or a.kwonlyargs or a.kwarg or a.defaults:
        problems.append("extra parameter kinds (positional-only, *args, keyword-only, **kwargs or defaults)")
    unannotated = [x.arg for x in a.args if not annotation_is_dataframe(x.annotation, aliases)]
    if unannotated:
        problems.append(f"parameters not annotated pd.DataFrame: {unannotated}")
    if not annotation_is_dataframe(fn.returns, aliases):
        problems.append("return annotation is not -> pd.DataFrame")
    if fn.decorator_list:
        problems.append("main is decorated")
    return problems


def _bound_names(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    return set()


def normalize_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(text: str) -> tuple[dict[str, str], list[Hit]]:
    """Return ({normalized dist: version}, bad lines) for a requirements.txt body."""
    pins: dict[str, str] = {}
    bad: list[Hit] = []
    logical: list[tuple[int, str]] = []
    buffer, start = "", 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        if not buffer:
            start = lineno
        line = re.sub(r"(^|\s)#.*$", "", raw).rstrip()
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        logical.append((start, (buffer + line).strip()))
        buffer = ""
    for lineno, line in logical:
        line = re.sub(r"\s--hash[= ]\S+", "", line).strip()
        if not line:
            continue
        m = REQ_PIN_RE.match(line)
        if not m or "*" in m["version"]:
            bad.append(Hit(lineno, line))
            continue
        name = normalize_dist(m["name"])
        if name in pins:
            bad.append(Hit(lineno, f"{line} (duplicate)"))
        pins[name] = m["version"]
    return pins, bad


def dist_candidates(top: str, installed: dict[str, list[str]]) -> set[str]:
    """Distributions whose pin satisfies ``import top`` (known aliases + what is installed)."""
    names = IMPORT_TO_DISTS.get(top, set()) | set(installed.get(top, []))
    return {normalize_dist(d) for d in names or {top}}


def installed_version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def pin_vs_local(dist: str, version: str, tops: list[str], installed: dict[str, list[str]]) -> str | None:
    """None if the pinned dist is installed at the pinned version, else a short explanation."""
    local = installed_version(dist)
    if local == version:
        return None
    if local is not None:
        return f"{dist}=={version} (local {local})"
    providers = [f"`import {t}` here comes from {d} {installed_version(d)}" for t in tops
                 if dist in dist_candidates(t, installed) for d in installed.get(t, [])]
    return f"{dist}=={version} not installed locally" + (f"; {', '.join(providers)}" if providers else "")


def check_file_basics(report: Report, path: Path, what: str) -> str | None:
    """Rule 13 checks; returns the decoded text if it is valid UTF-8."""
    data = path.read_bytes()
    report.check(len(data) < MAX_SOURCE_BYTES, f"Rule 13: {what} under 1 MB", f"{len(data) / 1e6:.3f} MB")
    report.check(b"\x00" not in data, f"Rule 13: {what} has no binary (NUL) bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        report.check(False, f"Rule 13: {what} is valid UTF-8", str(e))
        return None
    report.check(True, f"Rule 13: {what} is valid UTF-8")
    return text


def check_requirements(report: Report, scanner: SourceScanner, req_path: Path, model_dir: Path,
                       pypi: bool = False) -> None:
    tops = sorted({m.split(".")[0] for m, _ in scanner.imports if m})
    third_party = [t for t in tops if t not in sys.stdlib_module_names and t != "__future__"]
    local = [t for t in third_party if (model_dir / f"{t}.py").exists() or (model_dir / t).is_dir()]
    report.check(not local, "Rule 4: self-contained (no imports of local modules)", ", ".join(local))
    report.info(f"non-stdlib imports: {', '.join(third_party) or '(none)'}")
    if not req_path.exists():
        report.check(not third_party, "Rule 4: requirements file exists", f"{req_path} not found")
        return
    text = check_file_basics(report, req_path, "requirements file")
    if text is None:
        return
    pins, bad = parse_requirements(text)
    report.check(not bad, "Rule 4: every requirement is an exact `name==version` PyPI pin", fmt_hits(bad))
    installed = importlib.metadata.packages_distributions()
    unpinned = [t for t in third_party if not dist_candidates(t, installed) & pins.keys()]
    report.check(not unpinned, "Rules 4/8: every non-stdlib import is pinned with == in requirements",
                 f"unpinned: {', '.join(unpinned)}" if unpinned else "")
    mismatched = [m for d, v in sorted(pins.items()) if (m := pin_vs_local(d, v, third_party, installed))]
    if mismatched:
        report.warn("Rule 4: pins match the versions these tests run with", "; ".join(mismatched))
    else:
        report.check(True, "Rule 4: pins match the versions these tests run with")
    network = sorted(pins.keys() & {normalize_dist(n) for n in PROHIBITED_MODULES["network"]})
    if network:
        report.warn("Rules 10/16: network packages pinned", ", ".join(network))
    report.info("pins: " + ", ".join(f"{d}=={v}{' (pre-installed)' if d in PREINSTALLED else ''}"
                                     for d, v in sorted(pins.items())))
    if pypi:
        check_pins_on_pypi(report, pins)
    else:
        report.skip("Rules 4/8/16: pins on PyPI (exists, CPython 3.13 linux wheel, known vulnerabilities)",
                    "offline; use --pypi")
    report.skip("Rule 16: full scan (OSV-Scanner, pip-audit, GuardDog)", f"run e.g. `pip-audit -r {req_path}`")


WHEEL_RE = re.compile(r"^(?P<name>.+?)-(?P<ver>[^-]+)(-\d[^-]*)?-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$")


def wheel_fits_ctf(filename: str) -> bool:
    """Whether a wheel installs on CPython 3.13, glibc linux x86_64 (the assumed CTF runtime)."""
    m = WHEEL_RE.match(filename)
    if not m:
        return False
    pys, abis, plats = (set(m[k].split(".")) for k in ("py", "abi", "plat"))
    if not ("any" in plats or any(p.startswith("manylinux") and p.endswith("_x86_64") for p in plats)):
        return False
    if "abi3" in abis:
        return any((v := re.fullmatch(r"cp3(\d+)", py)) and int(v[1]) <= 13 for py in pys)
    if "cp313" in abis:
        return "cp313" in pys
    return "none" in abis and bool(pys & {"py3", "py313", "cp313"})


def requires_python_ok(spec: str | None) -> bool | None:
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        return None
    return SpecifierSet(spec).contains("3.13", prereleases=True)


def pypi_release(dist: str, version: str) -> dict[str, Any] | None:
    """PyPI JSON for one release; None if it does not exist (raises on network errors)."""
    import json
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{dist}/{version}/json", timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def check_pins_on_pypi(report: Report, pins: dict[str, str]) -> None:
    missing, no_wheel, bad_python, vulns = [], [], [], []
    for dist, version in sorted(pins.items()):
        try:
            release = pypi_release(dist, version)
        except OSError as e:
            report.warn("Rules 4/8/16: PyPI reachable", f"{dist}: {e}")
            return
        if release is None:
            missing.append(f"{dist}=={version}")
            continue
        if not any(wheel_fits_ctf(u["filename"]) for u in release.get("urls", [])):
            no_wheel.append(f"{dist}=={version}")
        if requires_python_ok(release["info"].get("requires_python")) is False:
            bad_python.append(f"{dist}=={version} (requires_python {release['info']['requires_python']})")
        ids = [v.get("id", "?") for v in release.get("vulnerabilities") or [] if not v.get("withdrawn")]
        if ids:
            more = ", ..." if len(ids) > 3 else ""
            vulns.append(f"{dist}=={version}: {len(ids)} advisories ({', '.join(ids[:3])}{more})")
    report.check(not missing, "Rule 4: every pinned version exists on PyPI", ", ".join(missing))
    report.check(not bad_python, "Rule 8: every pin supports Python 3.13 (requires_python)", ", ".join(bad_python))
    report.check(not no_wheel, "Rule 8: every pin has a wheel for CPython 3.13 on linux x86_64", ", ".join(no_wheel))
    if vulns:
        report.warn("Rule 16: PyPI/OSV lists known vulnerabilities (CTF rejects HIGH/CRITICAL)", "; ".join(vulns))
    else:
        report.check(True, "Rule 16: no known vulnerabilities listed on PyPI/OSV for the pins")


def run_static_checks(report: Report, model: Path, req_path: Path, pypi: bool = False) -> bool:
    """Static checks; returns False if the file cannot be parsed (dynamic checks pointless)."""
    report.section(f"Static checks: {model}")
    report.info(f"Python {sys.version.split()[0]}, pandas {pd.__version__}, numpy {np.__version__}")
    if sys.version_info[:2] != (3, 13):
        report.warn("Rule 8: checks run on Python 3.13 (the CTF runtime)", sys.version.split()[0])
    source = check_file_basics(report, model, "model script")
    if source is None:
        return False
    try:
        tree = ast.parse(source, filename=str(model))
    except SyntaxError as e:
        report.check(False, "model script parses as Python", f"line {e.lineno}: {e.msg}")
        return False
    report.check(True, "model script parses as Python")
    scanner = SourceScanner(source, tree).scan(tree)
    problems = main_signature_problems(tree, scanner.aliases)
    report.check(not problems, "Rule 11: defines main(chars: pd.DataFrame, features: pd.DataFrame, "
                 "daily_ret: pd.DataFrame) -> pd.DataFrame", "; ".join(problems))
    report.check(not scanner.hits.get("main() called at import time"), "Rule 11: main() is not called at import time",
                 fmt_hits(scanner.hits.get("main() called at import time", [])))
    for category in ["network import", "shell / native code import", "dynamic code import",
                     "shell / dynamic-code import", "shell execution call", "dynamic code call",
                     "relative import", "os.environ write", "file I/O outside the __main__ block"]:
        found = scanner.hits.get(category, [])
        report.check(not found, f"Rules 10/15: no {category}", fmt_hits(found))
    if scanner.hits.get("joblib threading backend"):
        report.warn("Rule 9: joblib threading backend (may deadlock; use loky)",
                    fmt_hits(scanner.hits["joblib threading backend"]))
    if scanner.env_reads:
        report.warn("Rule 15: environment reads other than CTF_EXECUTION_MODE", fmt_hits(scanner.env_reads))
    secrets = secret_hits(source, tree)
    report.check(not secrets, "Rule 15: no hardcoded API keys, passwords or tokens", fmt_hits(secrets))
    naive = [Hit(i, " ".join(line.split())[:70]) for i, line in enumerate(source.splitlines(), 1)
             if NAIVE_SCAN_RE.search(line)]
    if naive:
        report.warn("Rule 15 (literal scan, comments included): text a regex scanner may flag", fmt_hits(naive))
    else:
        report.check(True, "Rule 15 (literal scan, comments included): no prohibited keywords in the text")
    check_requirements(report, scanner, req_path, model.parent, pypi)
    return True


# ═════════════════════════════════════════════════════════════════════════════════════════
# Dynamic checks: data, model loading and runs
# ═════════════════════════════════════════════════════════════════════════════════════════
@dataclass
class Inputs:
    chars: pd.DataFrame
    features: pd.DataFrame
    daily_ret: pd.DataFrame

    def frames(self) -> dict[str, pd.DataFrame]:
        return {"chars": self.chars, "features": self.features, "daily_ret": self.daily_ret}

    def deep_copy(self) -> Inputs:
        return Inputs(*(df.copy(deep=True) for df in self.frames().values()))


@dataclass
class RunResult:
    label: str
    seconds: float = float("nan")
    output: Any = None
    error: str | None = None
    keyed: pd.DataFrame | None = None  # id/eom/w normalized for comparisons
    net_attempts: list[str] = field(default_factory=list)


def to_datetime_ns(s: pd.Series) -> pd.Series:
    # Strings: fixed ISO format, so one malformed value cannot change the inferred format.
    iso = isinstance(first_valid(s), str)
    out = pd.to_datetime(s, errors="coerce", format="ISO8601" if iso else None)
    if isinstance(out.dtype, pd.DatetimeTZDtype):
        out = out.dt.tz_localize(None)
    return out.astype("datetime64[ns]")


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"true", "t", "1"})


def keyed_frame(ids: pd.Series, eoms: pd.Series, w: pd.Series | None = None) -> pd.DataFrame:
    """Canonical (id float64, eom datetime64[ns][, w float64]) frame for joins and comparisons."""
    out = pd.DataFrame({"id": pd.to_numeric(ids, errors="coerce").astype("float64").to_numpy(),
                        "eom": to_datetime_ns(eoms).to_numpy()})
    if w is not None:
        out["w"] = pd.to_numeric(w, errors="coerce").astype("float64").to_numpy()
    return out


def load_inputs(data_dir: Path) -> Inputs:
    return Inputs(*(pd.read_parquet(data_dir / DATA_FILES[k]) for k in ("chars", "features", "daily_ret")))


def load_model(path: Path) -> ModuleType:
    """Import the model file as a fresh module named after its stem.

    Registered in sys.modules (with its directory on sys.path) so that joblib/loky workers
    can unpickle functions defined in it by reference.
    """
    name = path.stem
    existing = sys.modules.get(name)
    if name in sys.stdlib_module_names or (existing is not None and getattr(existing, "__file__", None) != str(path)):
        raise RuntimeError(f"model module name {name!r} clashes with an existing module; rename the file")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def network_blocked(attempts: list[str]) -> Iterator[None]:
    """Make internet sockets fail inside this process, as in the isolated CTF runtime (Rule 10)."""
    originals = {name: getattr(socket.socket, name) for name in ("connect", "connect_ex", "sendto")}

    def guarded(name: str) -> Callable[..., Any]:
        def wrapper(self: socket.socket, *args: Any) -> Any:
            if self.family in (socket.AF_INET, socket.AF_INET6):
                attempts.append(f"socket.{name}({args[-1]!r})")
                raise OSError(errno.ENETUNREACH, "network blocked by test_submission.py (Rule 10)")
            return originals[name](self, *args)
        return wrapper

    for name in originals:
        setattr(socket.socket, name, guarded(name))
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(socket.socket, name, fn)


def run_model(model: Path, inputs: Inputs, label: str) -> RunResult:
    """Fresh import + one main() call on the given inputs (which the model may consume)."""
    res = RunResult(label)
    print(f"-- run [{label}]: rows chars {len(inputs.chars):,}, daily_ret {len(inputs.daily_ret):,}", flush=True)
    with network_blocked(res.net_attempts):
        t0 = time.perf_counter()
        try:
            module = load_model(model)
            t0 = time.perf_counter()
            res.output = module.main(inputs.chars, inputs.features, inputs.daily_ret)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - report any model failure
            traceback.print_exception(e, file=sys.stdout)
            frames = traceback.extract_tb(e.__traceback__)
            where = next((f for f in reversed(frames) if f.filename == str(model)), None)
            loc = f" at {model.name}:{where.lineno} in {where.name}()" if where else ""
            res.error = f"{type(e).__name__}: {e}{loc}"
        finally:
            res.seconds = time.perf_counter() - t0
    if res.error is None:
        res.keyed = normalize_output(res.output)
    return res


def normalize_output(pf: Any) -> pd.DataFrame | None:
    if not isinstance(pf, pd.DataFrame) or not set(OUTPUT_COLUMNS) <= set(pf.columns):
        return None
    if pf.columns.duplicated().any():
        return None
    return keyed_frame(pf["id"], pf["eom"], pf["w"])


# ── perturbed inputs ─────────────────────────────────────────────────────────────────────
def shuffled_inputs(inp: Inputs, seed: int) -> Inputs:
    rng = np.random.default_rng(seed)

    def shuffle(df: pd.DataFrame) -> pd.DataFrame:
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
        return df[list(rng.permutation(np.asarray(df.columns, dtype=object)))]

    return Inputs(shuffle(inp.chars), shuffle(inp.features), shuffle(inp.daily_ret))


def first_valid(s: pd.Series) -> Any:
    idx = s.first_valid_index()
    return None if idx is None else s.loc[idx]


def flip_dtypes(df: pd.DataFrame, changes: set[str]) -> pd.DataFrame:
    """The 'other' representation of each column: what a different loader might hand main()."""
    df = df.copy(deep=True)
    for col in df.columns:
        s = df[col]
        if s.dtype == object and isinstance(first_valid(s), dt.date):
            df[col] = to_datetime_ns(s)
            changes.add("date objects -> datetime64[ns]")
        elif pd.api.types.is_datetime64_any_dtype(s):
            df[col] = pd.Series(s.dt.date, index=s.index, dtype=object)
            changes.add("datetime64 -> date objects")
        elif col == "ctff_test" and pd.api.types.is_bool_dtype(s):
            df[col] = s.astype("int32")
            changes.add("ctff_test bool -> int")
        elif col == "ctff_test" and pd.api.types.is_integer_dtype(s):
            df[col] = s.astype(bool)
            changes.add("ctff_test int -> bool")
        elif isinstance(s.dtype, pd.StringDtype):
            df[col] = s.astype(object)
            changes.add("StringDtype -> object")
        elif col == "id" and pd.api.types.is_integer_dtype(s):
            df[col] = s.astype("int64" if s.dtype.itemsize < 8 else "int32")
            changes.add(f"id {s.dtype} -> {df[col].dtype}")
    return df


def alt_dtype_inputs(inp: Inputs) -> tuple[Inputs, str]:
    changes: set[str] = set()
    out = Inputs(*(flip_dtypes(df, changes) for df in inp.frames().values()))
    return out, ", ".join(sorted(changes)) or "no convertible columns"


def truncated_inputs(inp: Inputs, cutoff: pd.Timestamp) -> Inputs:
    """The organizers' truncation: chars[eom <= c], daily_ret[date <= c]."""
    chars = inp.chars[(to_datetime_ns(inp.chars["eom"]) <= cutoff).to_numpy()].reset_index(drop=True)
    daily = inp.daily_ret[(to_datetime_ns(inp.daily_ret["date"]) <= cutoff).to_numpy()].reset_index(drop=True)
    return Inputs(chars, inp.features.copy(deep=True), daily)


def strict_inputs(inp: Inputs, cutoff: pd.Timestamp, seed: int) -> Inputs:
    """Truncation at c plus noise in place of the month-c lead returns (unknown at c)."""
    out = truncated_inputs(inp, cutoff)
    at_cutoff = (to_datetime_ns(out.chars["eom"]) == cutoff).to_numpy()
    noise = np.random.default_rng(seed).normal(0.0, 0.1, int(at_cutoff.sum()))
    out.chars.loc[at_cutoff, "ret_exc_lead1m"] = noise
    return out


def choose_cutoffs(test_eoms: list[pd.Timestamp], n: int) -> list[pd.Timestamp]:
    """Second-to-last test month plus n-1 others spread evenly from the first test month."""
    if not test_eoms:
        return []
    last = max(len(test_eoms) - 2, 0)
    idx = {last} if n <= 1 else {round(last * k / (n - 1)) for k in range(n)}
    return [test_eoms[i] for i in sorted(idx)]


# ── comparisons and contract ─────────────────────────────────────────────────────────────
@dataclass
class Comparison:
    n_common: int = 0
    only_ref: int = 0
    only_new: int = 0
    n_bad: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0
    duplicated: bool = False

    @property
    def ok(self) -> bool:
        return not self.duplicated and self.only_ref == self.only_new == self.n_bad == 0

    def detail(self) -> str:
        if self.duplicated:
            return "duplicated (id, eom) in an output; cannot compare"
        parts = [f"{self.n_common:,} rows, max |diff| {self.max_abs:.2e}, max rel diff {self.max_rel:.2e}"]
        if self.n_bad:
            parts.append(f"{self.n_bad:,} outside tolerance")
        if self.only_ref or self.only_new:
            parts.append(f"{self.only_ref:,} rows only in the reference run, {self.only_new:,} only in this run")
        return "; ".join(parts)


def compare_weights(ref: pd.DataFrame, new: pd.DataFrame, rtol: float = RTOL, atol: float = ATOL) -> Comparison:
    """Rule 18 tolerance |new - ref| <= atol + rtol * |ref| on the outer join of (id, eom)."""
    if ref.duplicated(["id", "eom"]).any() or new.duplicated(["id", "eom"]).any():
        return Comparison(duplicated=True)
    m = ref.merge(new, on=["id", "eom"], how="outer", suffixes=("_ref", "_new"), indicator=True)
    both = m[m["_merge"] == "both"]
    a, b = both["w_new"].to_numpy(), both["w_ref"].to_numpy()
    diff = np.abs(a - b)
    bad = ~(diff <= atol + rtol * np.abs(b))  # NaN counts as a mismatch, like R's anyNA()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(b) > 0, diff / np.abs(b), np.where(diff > 0, np.inf, 0.0))
    finite = ~np.isnan(diff)
    return Comparison(
        n_common=len(both), only_ref=int((m["_merge"] == "left_only").sum()),
        only_new=int((m["_merge"] == "right_only").sum()), n_bad=int(bad.sum()),
        max_abs=float(diff[finite].max()) if finite.any() else 0.0,
        max_rel=float(rel[finite].max()) if finite.any() else 0.0)


@dataclass
class Item:
    ok: bool
    label: str
    detail: str = ""


def eom_type_problem(s: pd.Series) -> str | None:
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is not None:
            return f"timezone-aware {s.dtype}"
        return None if (s.dropna() == s.dropna().dt.normalize()).all() else "datetime64 with time of day"
    if s.dtype == object:
        bad = [v for v in s.dropna() if not isinstance(v, dt.date)
               or (isinstance(v, dt.datetime) and v != dt.datetime.combine(v.date(), dt.time(), v.tzinfo))]
        return None if not bad else f"{len(bad)} non-date values, e.g. {bad[0]!r}"
    return f"dtype {s.dtype}"


def contract_items(pf: Any, expected: pd.DataFrame, keyed: pd.DataFrame | None) -> list[Item]:
    """Rule 11/12/5 output contract; ``expected`` holds the ctff_test (id, eom) keys."""
    if not isinstance(pf, pd.DataFrame):
        return [Item(False, "Rule 11: main() returns a pandas DataFrame", type(pf).__name__)]
    cols = [str(c) for c in pf.columns]
    items = [Item(True, "Rule 11: main() returns a pandas DataFrame"),
             Item(cols == OUTPUT_COLUMNS, "Rule 12: columns are exactly id, eom, w",
                  "" if cols == OUTPUT_COLUMNS else f"got {cols}")]
    if keyed is None:
        return items
    pf = pf.loc[:, OUTPUT_COLUMNS]
    n_na = pf.isna().sum()
    id_s, eom_s, w_s = pf["id"], pf["eom"], pf["w"]
    w_num = pd.to_numeric(w_s, errors="coerce")
    items += [
        Item(len(pf) > 0, "Rule 12: output is non-empty", f"{len(pf):,} rows"),
        Item(int(n_na.sum()) == 0, "Rule 12: no missing values",
             ", ".join(f"{c} {n:,}" for c, n in n_na.items() if n)),
        Item(pd.api.types.is_integer_dtype(id_s) and not pd.api.types.is_bool_dtype(id_s),
             "Rule 12: id is an integer dtype", str(id_s.dtype)),
        Item(pd.api.types.is_float_dtype(w_s), "Rule 12: w is a float dtype", str(w_s.dtype)),
        Item(bool(np.isfinite(w_num.dropna()).all()), "Rule 12: w is finite (no inf)"),
    ]
    problem = eom_type_problem(eom_s)
    items.append(Item(problem is None, "Rule 12: eom is a date (datetime.date or midnight datetime64)",
                      problem or str(eom_s.dtype)))
    csv_text = pf.to_csv(index=False)
    csv_eom = pd.read_csv(io.StringIO(csv_text), usecols=["eom"], dtype=str)["eom"]
    bad_fmt = ~csv_eom.fillna("").str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    items.append(Item(not bad_fmt.any(), "Rule 12: eom serializes as YYYY-MM-DD",
                      f"{int(bad_fmt.sum()):,} bad, e.g. {csv_eom[bad_fmt].iloc[0]!r}" if bad_fmt.any() else ""))
    size = len(csv_text.encode())
    items.append(Item(size < MAX_OUTPUT_BYTES, "Rule 12: output under 50 MB as CSV", f"{size / 1e6:.2f} MB"))
    n_dup = int(keyed.duplicated(["id", "eom"]).sum())
    items.append(Item(n_dup == 0, "Rule 12: no duplicated (id, eom)", f"{n_dup:,} duplicated" if n_dup else ""))
    items.append(coverage_item(keyed, expected))
    items.append(exposure_item(keyed, expected))
    return items


def coverage_item(keyed: pd.DataFrame, expected: pd.DataFrame) -> Item:
    m = expected.drop_duplicates().merge(keyed[["id", "eom"]].drop_duplicates(), on=["id", "eom"],
                                         how="outer", indicator=True)
    missing, extra = m[m["_merge"] == "left_only"], m[m["_merge"] == "right_only"]

    def examples(df: pd.DataFrame) -> str:
        return ", ".join(f"({r.id:.0f}, {r.eom:%Y-%m-%d})" if pd.notna(r.id) and pd.notna(r.eom)
                         else f"({r.id}, {r.eom})" for r in df.head(3).itertuples())

    detail = f"{len(missing):,} missing, {len(extra):,} extra"
    if len(missing):
        detail += f"; missing e.g. {examples(missing)}"
    if len(extra):
        detail += f"; extra e.g. {examples(extra)}"
    return Item(len(missing) == 0 and len(extra) == 0,
                "Rules 5/12: a weight for every ctff_test (id, eom), and nothing else", detail)


def exposure_item(keyed: pd.DataFrame, expected: pd.DataFrame) -> Item:
    months = pd.Index(expected["eom"].unique())
    gross = keyed.assign(a=keyed["w"].abs()).groupby("eom")["a"].sum().reindex(months, fill_value=0.0)
    zero = gross[~(gross > 0)]
    detail = (f"{len(zero)} month(s) with zero gross, e.g. {', '.join(f'{d:%Y-%m-%d}' for d in zero.index[:3])}"
              if len(zero) else f"{len(months)} test months")
    return Item(len(zero) == 0, "non-zero gross exposure in every test month", detail)


def mutation_detail(before: pd.DataFrame, after: pd.DataFrame) -> str | None:
    if (list(after.columns) == list(before.columns) and after.index.equals(before.index)
            and after.dtypes.equals(before.dtypes) and after.equals(before)):
        return None
    parts = []
    added = [c for c in after.columns if c not in before.columns]
    removed = [c for c in before.columns if c not in after.columns]
    if added:
        parts.append(f"added {added[:4]}")
    if removed:
        parts.append(f"removed {removed[:4]}")
    if len(after) != len(before):
        parts.append(f"rows {len(before):,} -> {len(after):,}")
    elif after.index.equals(before.index):
        changed = [c for c in before.columns if c in after.columns and not after[c].equals(before[c])]
        if changed:
            parts.append(f"modified {changed[:4]}{' ...' if len(changed) > 4 else ''}")
    else:
        parts.append("row order/index changed")
    return "; ".join(parts) or "column order or values changed"


# ── per-directory driver ─────────────────────────────────────────────────────────────────
@dataclass
class DirContext:
    report: Report
    model: Path
    pristine: Inputs
    expected: pd.DataFrame  # ctff_test (id, eom) keys, keyed form
    returns: pd.DataFrame  # keyed (id, eom) + r for the test rows
    seed: int
    runs: list[RunResult] = field(default_factory=list)

    def run(self, inputs: Inputs, label: str) -> RunResult:
        res = run_model(self.model, inputs, label)
        self.runs.append(res)
        return res

    def ran(self, res: RunResult, what: str) -> bool:
        """Report a failed run; successful non-base runs are summarized in the run table."""
        if res.error is None:
            return True
        return self.report.check(False, f"model imports and main() runs without error [{what}]", res.error)


def expected_keys(chars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = chars[as_bool(chars["ctff_test"]).to_numpy()]
    keys = keyed_frame(test["id"], test["eom"])
    rets = keys.assign(r=pd.to_numeric(test["ret_exc_lead1m"], errors="coerce").to_numpy()
                       if "ret_exc_lead1m" in test else np.nan)
    return keys, rets


def report_items(report: Report, items: list[Item], aggregate_label: str | None = None,
                 base_failed: set[str] | None = None) -> bool:
    """Print each item, or one aggregated line for a non-base run.

    Failures already reported for the base run are only named, not detailed again.
    """
    if aggregate_label is None:
        return all([report.check(i.ok, i.label, i.detail) for i in items])
    failed = [i for i in items if not i.ok]
    repeated = [i.label for i in failed if i.label in (base_failed or set())]
    new = [f"{i.label}{' (' + i.detail + ')' if i.detail else ''}" for i in failed
           if i.label not in (base_failed or set())]
    detail = "; ".join(new + ([f"as in the base run: {', '.join(repeated)}"] if repeated else []))
    return report.check(not failed, aggregate_label, detail)


def check_comparison(ctx: DirContext, base: RunResult, res: RunResult, label: str,
                     restrict: pd.Timestamp | None = None) -> bool:
    if not ctx.ran(res, res.label):
        return False
    if res.keyed is None:
        return ctx.report.check(False, label, "output lacks id/eom/w columns")
    ref = base.keyed if restrict is None else base.keyed[base.keyed["eom"] <= restrict]
    cmp = compare_weights(ref, res.keyed)
    return ctx.report.check(cmp.ok, label, cmp.detail())


def test_data_dir(report: Report, model: Path, data_dir: Path, args: argparse.Namespace) -> pd.DataFrame | None:
    """All dynamic checks on one data dir; returns the base run's keyed weights (or None)."""
    report.section(f"Dynamic checks: {data_dir}")
    missing = [f for f in DATA_FILES.values() if not (data_dir / f).exists()]
    if not report.check(not missing, "data dir has the three CTF parquet files", ", ".join(missing)):
        return None
    pristine = load_inputs(data_dir)
    keys, rets = expected_keys(pristine.chars)
    test_eoms = [pd.Timestamp(e) for e in sorted(keys["eom"].dropna().unique())]
    all_eoms = to_datetime_ns(pristine.chars["eom"])
    report.info(f"chars {pristine.chars.shape[0]:,} x {pristine.chars.shape[1]}, {all_eoms.nunique()} months "
                f"({all_eoms.min():%Y-%m-%d} .. {all_eoms.max():%Y-%m-%d}); features {len(pristine.features)}; "
                f"daily_ret {len(pristine.daily_ret):,}; test {len(keys):,} rows in {len(test_eoms)} months; "
                f"eom dtype {pristine.chars['eom'].dtype}")
    if not report.check(len(test_eoms) > 0, "data has ctff_test rows"):
        return None
    ctx = DirContext(report, model, pristine, keys, rets, args.seed)

    passed = pristine.deep_copy()
    base = ctx.run(passed, "base")
    if ctx.ran(base, "base"):
        report.check(True, "model imports and main() runs without error [base]", f"{base.seconds:.1f} s")
    if base.error is not None or base.keyed is None:
        if base.error is None:
            report_items(report, contract_items(base.output, keys, None))
        report.skip("remaining dynamic checks", "base run failed")
        return None
    base_items = contract_items(base.output, keys, base.keyed)
    report_items(report, base_items)
    base_failed = {i.label for i in base_items if not i.ok}
    changed = {k: d for k, df in passed.frames().items() if (d := mutation_detail(pristine.frames()[k], df))}
    report.check(not changed, "main() leaves its input DataFrames unchanged",
                 "; ".join(f"{k}: {d}" for k, d in changed.items()))
    del passed

    if not check_comparison(ctx, base, ctx.run(pristine.deep_copy(), "repeat"),
                            "Rule 18: deterministic (second run gives the same weights)"):
        report.info("the model is not deterministic, so the comparisons below may fail for that reason alone")
    if args.quick:
        report.skip("Rule 18: shuffled rows/features/columns and alternative dtypes", "--quick")
    else:
        res = ctx.run(shuffled_inputs(pristine, args.seed), "shuffled")
        check_comparison(ctx, base, res, "Rule 18: same weights when input rows, features and columns are shuffled")
        alt, changes = alt_dtype_inputs(pristine)
        res = ctx.run(alt, "dtypes")
        check_comparison(ctx, base, res, f"Rule 18: same weights with alternative input dtypes ({changes})")
        if res.error is None:
            report_items(report, contract_items(res.output, keys, res.keyed),
                         "Rule 12: output contract also holds with alternative input dtypes", base_failed)

    for c in choose_cutoffs(test_eoms, 1 if args.quick else args.cutoffs):
        res = ctx.run(truncated_inputs(pristine, c), f"cutoff {c:%Y-%m-%d}")
        n_months = sum(e <= c for e in test_eoms)
        check_comparison(ctx, base, res, f"Rule 1: no lookahead (same weights for the {n_months} test month(s) "
                         f"up to {c:%Y-%m-%d} when later data is removed)", restrict=c)
        if res.error is None:
            report_items(report, contract_items(res.output, keys[keys["eom"] <= c], res.keyed),
                         f"Rule 12: output contract holds on data truncated at {c:%Y-%m-%d}", base_failed)
    last = test_eoms[-1]
    res = ctx.run(strict_inputs(pristine, last, args.seed), f"strict {last:%Y-%m-%d}")
    check_comparison(ctx, base, res, f"Rule 1 (strict): same weights when the {last:%Y-%m-%d} lead returns are "
                     "noise and later daily returns are removed")

    attempts = sorted({a for r in ctx.runs for a in r.net_attempts})
    n_runs = sum(bool(r.net_attempts) for r in ctx.runs)
    report.check(not attempts, "Rule 10: no network connection attempted (in-process guard)",
                 f"in {n_runs} of {len(ctx.runs)} runs: {'; '.join(attempts[:3])}" if attempts else "")
    print_run_table(ctx.runs)
    print_performance(report, base.keyed, rets)
    return base.keyed


def check_weights_csv(report: Report, csv_path: Path, data_dir: Path, base: pd.DataFrame | None) -> None:
    """check_submission.R's output checks on a saved weights CSV (Rules 5, 12)."""
    report.section(f"Weights CSV: {csv_path} (against {data_dir})")
    if not report.check(csv_path.is_file(), "Rule 5: weights CSV exists", str(csv_path)):
        return
    size = csv_path.stat().st_size
    report.check(size < MAX_OUTPUT_BYTES, "Rule 12: output under 50 MB", f"{size / 1e6:.2f} MB")
    out = pd.read_csv(csv_path, dtype={"eom": str})
    cols = [str(c) for c in out.columns]
    if not report.check(cols == OUTPUT_COLUMNS, "Rule 12: columns are exactly id, eom, w",
                        "" if cols == OUTPUT_COLUMNS else f"got {cols}"):
        return
    ids = pd.to_numeric(out["id"], errors="coerce")
    report.check(len(out) > 0, "Rule 12: output is non-empty", f"{len(out):,} rows")
    report.check(not out.isna().any().any(), "Rule 12: no missing values")
    report.check(bool(ids.notna().all() and (ids == ids.round()).all()), "Rule 12: id is integer")
    if pd.api.types.is_float_dtype(out["id"]):
        report.warn("Rule 12: id written as an integer (no decimal point)", "id column parses as float")
    bad_eom = ~out["eom"].fillna("").str.fullmatch(r"\d{4}-\d{2}-\d{2}") | to_datetime_ns(out["eom"]).isna()
    report.check(not bad_eom.any(), "Rule 12: eom is a YYYY-MM-DD date", f"{int(bad_eom.sum()):,} bad")
    w = pd.to_numeric(out["w"], errors="coerce")
    report.check(bool(w.notna().all() and np.isfinite(w).all()), "Rule 12: w is numeric and finite")
    keyed = keyed_frame(out["id"], out["eom"], out["w"])
    n_dup = int(keyed.duplicated(["id", "eom"]).sum())
    report.check(n_dup == 0, "Rule 12: no duplicated (id, eom)", f"{n_dup:,}" if n_dup else "")
    chars = pd.read_parquet(data_dir / DATA_FILES["chars"], columns=["id", "eom", "ctff_test", "ret_exc_lead1m"])
    keys, rets = expected_keys(chars)
    for item in (coverage_item(keyed, keys), exposure_item(keyed, keys)):
        report.check(item.ok, item.label, item.detail)
    if base is not None:
        cmp = compare_weights(base, keyed)
        report.check(cmp.ok, "CSV matches the weights main() returned on this data", cmp.detail())
    print_performance(report, keyed, rets)


def print_run_table(runs: list[RunResult]) -> None:
    print("\n  run                      seconds        rows  months  status")
    for r in runs:
        rows = months = "-"
        if r.keyed is not None:
            rows, months = f"{len(r.keyed):,}", f"{r.keyed['eom'].nunique()}"
        status = "ok" if r.error is None else "error"
        print(f"  {r.label:<24} {r.seconds:>7.1f} {rows:>11} {months:>7}  {status}")
    print(flush=True)


def print_performance(report: Report, keyed: pd.DataFrame, rets: pd.DataFrame) -> None:
    m = rets.merge(keyed, on=["id", "eom"], how="inner")
    held_missing = int((m["r"].isna() & (m["w"] != 0)).sum())
    if held_missing:
        report.warn("held positions without ret_exc_lead1m", f"{held_missing:,}")
    g = (m.assign(pr=m["w"] * m["r"], a=m["w"].abs(), nz=m["w"] != 0)
          .groupby("eom").agg(ret=("pr", "sum"), gross=("a", "sum"), net=("w", "sum"), nz=("nz", "sum")))
    mean, sd = g["ret"].mean() * 12, g["ret"].std(ddof=1) * np.sqrt(12)
    sharpe = mean / sd if len(g) > 1 and sd > 0 else float("nan")
    report.info(f"performance over {len(g)} test months: mean {mean:.2%}/yr, vol {sd:.2%}/yr, "
                f"Sharpe {sharpe:.2f} (annualized); avg gross {g['gross'].mean():.3f} "
                f"(min {g['gross'].min():.3f}), avg net {g['net'].mean():.3f}, "
                f"avg nonzero weights {g['nz'].mean():.1f}/month")


# ═════════════════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════════════════
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="model .py file defining main()")
    ap.add_argument("--requirements", type=Path, help="requirements.txt (default: next to the model)")
    ap.add_argument("--data", type=Path, action="append", default=[],
                    help="directory with the three ctff_*.parquet files (repeatable)")
    ap.add_argument("--static-only", action="store_true", help="skip running main()")
    ap.add_argument("--cutoffs", type=int, default=2, help="number of truncation cutoffs (default 2)")
    ap.add_argument("--quick", action="store_true", help="skip shuffle/dtype runs, use 1 cutoff")
    ap.add_argument("--seed", type=int, default=2026, help="seed for shuffles and noise (default 2026)")
    ap.add_argument("--strict", action="store_true", help="exit 1 on WARN as well as FAIL")
    ap.add_argument("--weights", type=Path,
                    help="also check a saved weights CSV against the first --data dir (runs with --static-only too)")
    ap.add_argument("--pypi", action="store_true",
                    help="query PyPI for each pin: exists, CPython 3.13 linux wheel, known vulnerabilities")
    args = ap.parse_args(argv)
    if (not args.static_only or args.weights) and not args.data:
        ap.error("--data is required unless --static-only (and for --weights)")
    if args.cutoffs < 1:
        ap.error("--cutoffs must be >= 1")
    args.model = args.model.resolve()
    args.requirements = (args.requirements or args.model.parent / "requirements.txt").resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = Report(args.strict)
    if not args.model.is_file():
        report.check(False, "model file exists", str(args.model))
        return report.exit_code()
    parsed = run_static_checks(report, args.model, args.requirements, args.pypi)
    first_base = None
    if parsed and not args.static_only:
        sys.path.insert(0, str(args.model.parent))
        bases = [test_data_dir(report, args.model, d.resolve(), args) for d in args.data]
        first_base = bases[0]
    elif not args.static_only:
        report.skip("dynamic checks", "model script did not pass the basic static checks")
    if args.weights:
        check_weights_csv(report, args.weights.resolve(), args.data[0].resolve(), first_base)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB -> GB on Linux
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    print(f"\nPeak RSS: {peak:.2f} GB (largest child process {children:.2f} GB)")
    if report.failures:
        print(f"\n{len(report.failures)} check(s) FAILED for {args.model.name}:")
        for label in report.failures:
            print(f"  - {label}")
    else:
        print(f"\nAll checks passed for {args.model.name}"
              + (f" ({len(report.warnings)} warning(s))" if report.warnings else ""))
    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
