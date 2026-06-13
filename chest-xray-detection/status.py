"""
status.py — Quick project dashboard for Chest X-Ray Pneumonia Detection.

Prints a one-screen summary of project readiness in under a second.
No heavy imports — just file-existence checks.

Usage:
    xray-status              (after pip install -e .)
    python status.py         (from the chest-xray-detection/ folder)
"""

import sys
import os
from pathlib import Path
from datetime import datetime


# ── Colour helpers ────────────────────────────────────────────────────────────

USE_COLOUR = sys.platform != "win32" and sys.stdout.isatty()
GREEN  = "\033[92m" if USE_COLOUR else ""
RED    = "\033[91m" if USE_COLOUR else ""
YELLOW = "\033[93m" if USE_COLOUR else ""
CYAN   = "\033[96m" if USE_COLOUR else ""
BOLD   = "\033[1m"  if USE_COLOUR else ""
DIM    = "\033[2m"  if USE_COLOUR else ""
RESET  = "\033[0m"  if USE_COLOUR else ""


def _fmt_size(path: Path) -> str:
    """Return human-readable file size, e.g. '29.1 MB'."""
    b = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def _fmt_mtime(path: Path) -> str:
    """Return last-modified date as 'YYYY-MM-DD HH:MM'."""
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _check(label: str, path: Path, note: str = "", fix: str = "") -> bool:
    """Print one status row. Returns True if the item exists."""
    exists = path.exists()
    icon  = f"{GREEN}✓{RESET}" if exists else f"{RED}✗{RESET}"
    label_str = f"{BOLD}{label:<28}{RESET}"

    if exists:
        detail = f"{DIM}{_fmt_size(path)}  {_fmt_mtime(path)}{RESET}"
        if note:
            detail += f"  {CYAN}{note}{RESET}"
        print(f"  {icon}  {label_str}  {detail}")
    else:
        hint = f"  {DIM}→ {fix}{RESET}" if fix else ""
        print(f"  {icon}  {label_str}  {RED}not found{RESET}{hint}")

    return exists


def run_status() -> bool:
    """
    Print the status dashboard. Returns True if the project is fully ready.
    """
    # Import only config (pure Python, no torch) to get absolute paths
    try:
        from config import (
            DATA_DIR, CHECKPOINT, TORCHSCRIPT_PATH,
            _PROJECT_ROOT, NUM_EPOCHS, BATCH_SIZE,
        )
    except Exception as exc:
        print(f"{RED}✗  Could not load config.py: {exc}{RESET}")
        return False

    project_root = Path(_PROJECT_ROOT)
    data_path    = Path(DATA_DIR)
    ckpt_path    = Path(CHECKPOINT)
    ts_path      = Path(TORCHSCRIPT_PATH)
    app_path     = project_root / "app.py"

    # ── Header ────────────────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print(f"{BOLD}┌─────────────────────────────────────────────────────┐{RESET}")
    print(f"{BOLD}│   Chest X-Ray Pneumonia Detection — Status          │{RESET}")
    print(f"{BOLD}│   {DIM}{now}{RESET}{BOLD}                            │{RESET}")
    print(f"{BOLD}└─────────────────────────────────────────────────────┘{RESET}")
    print()

    # ── Section 1: Data ───────────────────────────────────────────────────────
    print(f"  {BOLD}DATA{RESET}")

    data_ok = data_path.exists() and data_path.is_dir()
    if data_ok:
        # Count images quickly
        n = sum(1 for _ in data_path.rglob("*") if _.suffix.lower() in (".jpeg", ".jpg", ".png"))
        splits = {s: sum(1 for _ in (data_path / s).rglob("*")
                         if _.suffix.lower() in (".jpeg", ".jpg", ".png"))
                  if (data_path / s).is_dir() else 0
                  for s in ("train", "val", "test")}
        split_str = "  ".join(f"{s}:{c:,}" for s, c in splits.items() if c)
        icon = f"{GREEN}✓{RESET}"
        print(f"  {icon}  {BOLD}{'Dataset (data/)': <28}{RESET}  {DIM}{n:,} images  [{split_str}]{RESET}")
    else:
        icon = f"{RED}✗{RESET}"
        print(f"  {icon}  {BOLD}{'Dataset (data/)': <28}{RESET}  {RED}not found{RESET}"
              f"  {DIM}→ run xray-download{RESET}")
    print()

    # ── Section 2: Models ─────────────────────────────────────────────────────
    print(f"  {BOLD}MODELS{RESET}")
    ckpt_ok = _check(
        "best_model.pth",
        ckpt_path,
        fix="run xray-train",
    )
    ts_ok = _check(
        "TorchScript (.pt)",
        ts_path,
        note="(used by FastAPI)" if ts_path.exists() else "",
        fix="run xray-export",
    )
    print()

    # ── Section 3: Application ────────────────────────────────────────────────
    print(f"  {BOLD}APPLICATION{RESET}")
    app_ok = _check("app.py (FastAPI server)", app_path)

    # Check if API is reachable (non-blocking, 0.3 s timeout)
    api_live = False
    try:
        import urllib.request
        req = urllib.request.urlopen("http://localhost:8000/health", timeout=0.3)
        api_live = req.getcode() == 200
    except Exception:
        pass

    api_icon = f"{GREEN}✓{RESET}" if api_live else f"{YELLOW}○{RESET}"
    api_msg  = "responding" if api_live else f"not running  {DIM}→ run xray-serve{RESET}"
    print(f"  {api_icon}  {BOLD}{'API server':28}{RESET}  {api_msg}")
    print()

    # ── Section 4: Configuration snapshot ────────────────────────────────────
    print(f"  {BOLD}CONFIG  {DIM}(edit config.py to change){RESET}")
    print(f"  {DIM}  NUM_EPOCHS  = {NUM_EPOCHS}   BATCH_SIZE = {BATCH_SIZE}{RESET}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    all_ready = data_ok and ckpt_ok and ts_ok and app_ok
    missing   = []
    if not data_ok:  missing.append("dataset")
    if not ckpt_ok:  missing.append("best_model.pth")
    if not ts_ok:    missing.append("TorchScript")

    if all_ready:
        print(f"  {GREEN}{BOLD}Ready.{RESET}  All components present.")
        print()
        print(f"  {DIM}xray-evaluate   xray-predict   xray-gradcam{RESET}")
        print(f"  {DIM}xray-visualize  xray-report    xray-serve{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}Not fully ready.{RESET}  Missing: {', '.join(missing)}")
    print()

    return all_ready


if __name__ == "__main__":
    ok = run_status()
    sys.exit(0 if ok else 1)
