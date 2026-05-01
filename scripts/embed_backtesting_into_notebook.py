"""One-shot: splice embedded ``backtesting/`` sources into ``hybrid_results_dashboard.ipynb``

and delete standalone ``backtesting/**/*.py`` so deliverable stays notebook-centric.

The notebook should include markdown containing ``<!-- EMBED_LOADER_AFTER_PART1 -->``
(Part 1 anchor). The regenerated embed cells are inserted immediately after that
cell so executable Part 1 cells stay above the loader.

Repo root::
    python scripts/embed_backtesting_into_notebook.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "backtesting" / "hybrid_results_dashboard.ipynb"
BT_ROOT = REPO_ROOT / "backtesting"

LOADER_FILES: list[tuple[str, str, Path]] = [
    ("backtesting", "__init__.py", BT_ROOT / "__init__.py"),
    ("backtesting.core", "core/__init__.py", BT_ROOT / "core" / "__init__.py"),
    ("backtesting.core.ips", "core/ips.py", BT_ROOT / "core" / "ips.py"),
    ("backtesting.core.walk_forward", "core/walk_forward.py", BT_ROOT / "core" / "walk_forward.py"),
    ("backtesting.core.data", "core/data.py", BT_ROOT / "core" / "data.py"),
    ("backtesting.core.backtest", "core/backtest.py", BT_ROOT / "core" / "backtest.py"),
    ("backtesting.hypotheses.base", "hypotheses/base.py", BT_ROOT / "hypotheses" / "base.py"),
    ("backtesting.hypotheses.h1_market_stress", "hypotheses/h1_market_stress.py", BT_ROOT / "hypotheses" / "h1_market_stress.py"),
    ("backtesting.hypotheses.h2_growth_cycle", "hypotheses/h2_growth_cycle.py", BT_ROOT / "hypotheses" / "h2_growth_cycle.py"),
    ("backtesting.hypotheses.h3_two_stage", "hypotheses/h3_two_stage.py", BT_ROOT / "hypotheses" / "h3_two_stage.py"),
    ("backtesting.hypotheses.h4_stagflation", "hypotheses/h4_stagflation.py", BT_ROOT / "hypotheses" / "h4_stagflation.py"),
    ("backtesting.hypotheses", "hypotheses/__init__.py", BT_ROOT / "hypotheses" / "__init__.py"),
    ("backtesting.models.base", "models/base.py", BT_ROOT / "models" / "base.py"),
    ("backtesting.models.xgb_model", "models/xgb_model.py", BT_ROOT / "models" / "xgb_model.py"),
    ("backtesting.models.lstm_model", "models/lstm_model.py", BT_ROOT / "models" / "lstm_model.py"),
    ("backtesting.models.transformer_model", "models/transformer_model.py", BT_ROOT / "models" / "transformer_model.py"),
    ("backtesting.models.return_forecaster", "models/return_forecaster.py", BT_ROOT / "models" / "return_forecaster.py"),
    ("backtesting.models", "models/__init__.py", BT_ROOT / "models" / "__init__.py"),
    ("backtesting.prepare", "prepare.py", BT_ROOT / "prepare.py"),
    ("backtesting.train", "train.py", BT_ROOT / "train.py"),
    ("backtesting.run_all_baselines", "run_all_baselines.py", BT_ROOT / "run_all_baselines.py"),
    ("backtesting.run_threshold_search", "run_threshold_search.py", BT_ROOT / "run_threshold_search.py"),
]


CODE_TAG = "# __AUTO_EMBED_BACKTESTING_LOADER__"
META_TAG = "###__EMBED_BACKTESTING_META__"
PART1_EMBED_ANCHOR = "<!-- EMBED_LOADER_AFTER_PART1 -->"

OLD_ROOT_BLOCK = '''# Resolve repo root for imports and for paths below. Cwd may be the repo root
# or `.../backtesting/` (e.g. notebook opened from that folder).
_root = Path.cwd().resolve()
if not (_root / "backtesting" / "__init__.py").is_file():
    _root = _root.parent
if (_root / "backtesting" / "__init__.py").is_file():
    sys.path.insert(0, str(_root))
ROOT = _root
'''

NEW_ROOT_BLOCK = '''# Repo root — works **without** ``backtesting/__init__.py`` on disk.
_root = Path.cwd().resolve()
if (_root / "hybrid_results_dashboard.ipynb").is_file():
    ROOT = _root.parent.resolve()
elif (_root / "backtesting" / "hybrid_results_dashboard.ipynb").is_file():
    ROOT = _root.resolve()
else:
    ROOT = _root.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
'''


def _cell_source_lines(blob: str) -> list[str]:
    out = []
    for ln in blob.splitlines(keepends=True):
        if not ln.endswith("\n"):
            ln += "\n"
        out.append(ln)
    return out


def build_embed_cell_body() -> str:
    regs: list[str] = []
    for fq, rel, p in LOADER_FILES:
        if not p.is_file():
            raise FileNotFoundError(f"Missing loader source file: {p}")
        regs.append(f"    REGISTER({fq!r}, {rel!r}, {repr(p.read_text(encoding='utf-8'))})")

    return (
        CODE_TAG
        + "\n"
        + '''"""Notebook-only copy of ``backtesting/**/*.py``. No separate ``.py`` tree required."""

from __future__ import annotations

import sys
import types
from pathlib import Path

_QUEUE: list[tuple[str, str, str]] = []


def _resolve_paths():
    cwd = Path.cwd().resolve()
    if (cwd / "hybrid_results_dashboard.ipynb").is_file():
        return cwd.parent.resolve(), cwd.resolve()
    proj_nb = cwd / "backtesting" / "hybrid_results_dashboard.ipynb"
    if proj_nb.is_file():
        return cwd.resolve(), proj_nb.parent.resolve()
    return cwd.resolve(), (cwd / "backtesting").resolve()


PROJECT_ROOT, BACKTEST_PKG_ROOT = _resolve_paths()


def REGISTER(full_name: str, rel_under_backtesting: str, source: str) -> None:
    _QUEUE.append((full_name, rel_under_backtesting, source))


def _purge_backtesting_modules() -> None:
    for key in list(sys.modules):
        if key == "backtesting" or key.startswith("backtesting."):
            del sys.modules[key]


def _ensure_parents(full_name: str) -> None:
    parts = full_name.split(".")
    if len(parts) <= 1:
        return
    for j in range(1, len(parts)):
        pname = ".".join(parts[:j])
        if pname in sys.modules:
            continue
        pkg = types.ModuleType(pname)
        pkg.__path__ = []
        sys.modules[pname] = pkg


def _attach_child(full_name: str, mod: types.ModuleType) -> None:
    parts = full_name.split(".")
    if len(parts) <= 1:
        return
    parent = ".".join(parts[:-1])
    setattr(sys.modules[parent], parts[-1], mod)


def _exec_one(full_name: str, rel_under_backtesting: str, source: str) -> types.ModuleType:
    _ensure_parents(full_name)
    parts = full_name.split(".")
    fake_path = (BACKTEST_PKG_ROOT / rel_under_backtesting).resolve()
    mod = types.ModuleType(full_name)
    mod.__file__ = str(fake_path)
    if len(parts) == 1:
        mod.__path__ = [str(BACKTEST_PKG_ROOT)]
        mod.__package__ = parts[0]
    else:
        mod.__package__ = ".".join(parts[:-1])
    sys.modules[full_name] = mod
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    _attach_child(full_name, mod)
    return mod


def _register_embedded_module_sources() -> None:
    """Each ``REGISTER`` holds ``repr(one old .py file)`` staged for exec."""
    _QUEUE.clear()
'''
        + "\n".join(regs)
        + """

def LOAD_EMBEDDED() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    for fq, rel, src in list(_QUEUE):
        _exec_one(fq, rel, src)


_EMBED_INITIALIZED = False


def init_embedded_backtesting(*, reload_: bool = False) -> None:
    global _EMBED_INITIALIZED
    if _EMBED_INITIALIZED and not reload_:
        return
    _purge_backtesting_modules()
    _register_embedded_module_sources()
    LOAD_EMBEDDED()
    _EMBED_INITIALIZED = True


def train_main(extra_argv: list[str] | None = None) -> None:
    import sys
    mod = sys.modules["backtesting.train"]
    old = sys.argv[:]
    try:
        base = [mod.__file__]
        sys.argv = base if not extra_argv else base + list(extra_argv)
        mod.main()
    finally:
        sys.argv = old


def run_all_baselines_main(extra_argv: list[str] | None = None) -> None:
    import sys
    mod = sys.modules["backtesting.run_all_baselines"]
    old = sys.argv[:]
    try:
        base = [mod.__file__]
        sys.argv = base if not extra_argv else base + list(extra_argv)
        mod.main()
    finally:
        sys.argv = old


def run_threshold_search_main(extra_argv: list[str] | None = None) -> None:
    import sys
    mod = sys.modules["backtesting.run_threshold_search"]
    old = sys.argv[:]
    try:
        base = [mod.__file__]
        sys.argv = base if not extra_argv else base + list(extra_argv)
        mod.main()
    finally:
        sys.argv = old


init_embedded_backtesting()
"""
    )


def _fix_attach_indent_bug(embed_src: str) -> str:
    """Fix typo: dedent inside _attach_child if present."""
    bad = """def _attach_child(full_name: str, mod: types.ModuleType) -> None:
    parts = full_name.split(".")
    if len(parts) <= 1:
        return
        parent = ".".join(parts[:-1])
        setattr(sys.modules[parent], parts[-1], mod)"""
    good = """def _attach_child(full_name: str, mod: types.ModuleType) -> None:
    parts = full_name.split(".")
    if len(parts) <= 1:
        return
    parent = ".".join(parts[:-1])
    setattr(sys.modules[parent], parts[-1], mod)"""
    if bad in embed_src:
        embed_src = embed_src.replace(bad, good)
    return embed_src


def main() -> None:
    embed_body = _fix_attach_indent_bug(build_embed_cell_body())
    compile(embed_body, "<embedded loader>", "exec")

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def _strip_old_embed_cells() -> None:
        def _embed_markdown(cell: dict) -> bool:
            if cell.get("cell_type") != "markdown":
                return False
            return "".join(cell.get("source") or []).lstrip().startswith(META_TAG)

        def _embed_code(cell: dict) -> bool:
            return (
                cell.get("cell_type") == "code"
                and "".join(cell.get("source") or []).lstrip().startswith(CODE_TAG)
            )

        cells[:] = [c for c in cells if not (_embed_markdown(c) or _embed_code(c))]

    _strip_old_embed_cells()

    insert_at = 1
    for i, c in enumerate(cells):
        if c.get("cell_type") != "markdown":
            continue
        if PART1_EMBED_ANCHOR in "".join(c.get("source") or []):
            insert_at = i + 1
            break

    md_intro = (
        f"{META_TAG}\n\n"
        "## Part 2 — Embedded ``backtesting`` runtime\n\n"
        "**What this cell is:** Notebook-only **`repr()` payloads** plus a loader (`init_embedded_backtesting`). "
        "Run it before any ``from backtesting …`` imports. Standalone ``backtesting/**/*.py`` "
        "files are removed from disk.\n\n"
        "Current sources include **JPY via inverted USD/JPY** in ``ASSET_ORDER`` / ``prepare_price_panel``.\n\n"
        "To **retrain** or resweep after editing embedded sources, open **Part 4** "
        "(end of notebook; tagged `optional-retraining`). **Skip Part 4** when `artifacts/` "
        "already contains the runs you chart in Part 3.\n\n"
        "CLI mirrors: **`train_main()`**, **`run_all_baselines_main()`**, **`run_threshold_search_main()`** "
        "(optional `extra_argv`).\n"
    )
    embed_md = {"cell_type": "markdown", "metadata": {}, "source": _cell_source_lines(md_intro)}
    embed_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _cell_source_lines(embed_body),
    }
    cells.insert(insert_at, embed_md)
    cells.insert(insert_at + 1, embed_code)

    patched = False
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        blob = "".join(cell.get("source") or [])
        if OLD_ROOT_BLOCK not in blob:
            continue
        blob = blob.replace(OLD_ROOT_BLOCK, NEW_ROOT_BLOCK)
        monte_m = re.search(
            r"requires cell 2 run for `taa_returns`",
            blob,
        )
        if monte_m:
            blob = blob.replace(
                "requires cell 2 run for `taa_returns`",
                "requires embedded loader + benchmark cell for `taa_returns`",
            )
        cell["source"] = _cell_source_lines(blob)
        patched = True
        break

    if not patched:
        raise RuntimeError("Could not find OLD_ROOT_BLOCK in notebook code cells.")

    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        blob = "".join(cell.get("source") or [])
        needles = (
            "- JPY is not used anywhere in the rerun portfolio universe because the IPS specifies Swiss franc, not JPY, and no local CHF series exists to trade.\n",
            "- JPY is not used anywhere in the rerun portfolio universe because the IPS specifies Swiss franc, not JPY, and no local CHF series exists to trade.",
        )
        repl = (
            "- **JPY** sleeve: Zion **USD/JPY** inverted to **`1/adj_close`** "
            "(USD‑based yen sleeve); Swiss franc still unavailable.\n"
        )
        patched_intro = False
        for needle in needles:
            if needle in blob:
                blob = blob.replace(needle, repl)
                patched_intro = True
                break
        if patched_intro:
            cell["source"] = _cell_source_lines(blob)

    nb["nbformat"] = nb.get("nbformat", 4)
    nb["nbformat_minor"] = nb.get("nbformat_minor", 5)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    removed = []
    for path in BT_ROOT.rglob("*.py"):
        path.unlink()
        removed.append(path.relative_to(REPO_ROOT))
    if not removed:
        print("Warning: no .py removed under backtesting/")
    print(f"Patched notebook: {NB_PATH}")
    print(f"Removed {len(removed)} .py files from backtesting/ — only notebook + artifacts remain as code.")


if __name__ == "__main__":
    main()
