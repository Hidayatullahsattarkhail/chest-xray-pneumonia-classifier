"""
test_api.py — Validate the FastAPI server by sending test images and checking predictions.

Usage:
    # 1. Start the server in one terminal:
    #    uvicorn app:app --port 8000
    #
    # 2. Run this script in another terminal:
    python test_api.py

    # Test more images per class (default is 5):
    python test_api.py --num 10

    # Point to a different server:
    python test_api.py --url http://localhost:8000

What it does:
  - Sends test images to POST /predict
  - Compares the returned prediction to the true label (from the folder name)
  - Prints a pass/fail table
  - Prints a summary: total, correct, wrong, accuracy
"""

import os
import sys
import argparse
import random
import time

import requests


# ── Configuration ─────────────────────────────────────────────────────────────

from config import REQUEST_TIMEOUT, CLASS_NAMES, TEST_DATA_DIR as TEST_DIR
from config import API_URL as DEFAULT_URL, NUM_TEST_IMAGES as DEFAULT_NUM

# Terminal colour codes (fall back to plain text on Windows)
GREEN  = "\033[92m" if sys.platform != "win32" else ""
RED    = "\033[91m" if sys.platform != "win32" else ""
YELLOW = "\033[93m" if sys.platform != "win32" else ""
RESET  = "\033[0m"  if sys.platform != "win32" else ""
BOLD   = "\033[1m"  if sys.platform != "win32" else ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_test_images(test_dir: str, num_per_class: int) -> list[tuple[str, str]]:
    """
    Randomly pick `num_per_class` images from each class in the test folder.

    Returns a list of (image_path, true_label) tuples.
    NORMAL images come first, then PNEUMONIA.
    """
    samples = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"{YELLOW}Warning: folder not found — {class_dir}{RESET}")
            continue
        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        chosen = random.sample(files, min(num_per_class, len(files)))
        for fname in chosen:
            samples.append((os.path.join(class_dir, fname), class_name))
    return samples


def check_server(base_url: str) -> bool:
    """
    Hit GET /health to confirm the server is up before running tests.
    Returns True if healthy, False otherwise.
    """
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        data = resp.json()
        print(f"  Server status : {data.get('status', '?')}")
        print(f"  Model loaded  : {data.get('model', '?')}")
        print(f"  Device        : {data.get('device', '?')}")
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def send_prediction(base_url: str, image_path: str) -> tuple[str, float, float]:
    """
    POST one image to /predict and return (predicted_label, confidence, latency_ms).

    Raises:
        requests.HTTPError if the server returns an error status.
        requests.Timeout if the server does not respond in time.
    """
    start = time.time()
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{base_url}/predict",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")},
            timeout=REQUEST_TIMEOUT,
        )
    latency_ms = (time.time() - start) * 1000
    resp.raise_for_status()
    data = resp.json()
    return data["prediction"], data["confidence"], latency_ms


def result_row(index, filename, true_label, pred_label, confidence, latency_ms, correct):
    """Format one row of the results table."""
    status = f"{GREEN}PASS ✓{RESET}" if correct else f"{RED}FAIL ✗{RESET}"
    conf_str     = f"{confidence * 100:.1f}%"
    latency_str  = f"{latency_ms:.0f}ms"
    filename_short = filename[:36] + "…" if len(filename) > 37 else filename
    return (
        f"  {index:>3}.  "
        f"{filename_short:<38}  "
        f"True: {true_label:<12}  "
        f"Pred: {pred_label:<12}  "
        f"Conf: {conf_str:<7}  "
        f"{latency_str:<6}  "
        f"{status}"
    )


# ── Main test runner ──────────────────────────────────────────────────────────

def run_tests(base_url: str = DEFAULT_URL, num_per_class: int = DEFAULT_NUM):
    print()
    print("=" * 70)
    print(f"  {BOLD}Chest X-Ray API Test Runner{RESET}")
    print("=" * 70)
    print(f"  Server : {base_url}")
    print(f"  Images : {num_per_class} per class  ({num_per_class * len(CLASS_NAMES)} total)")
    print()

    # ── 1. Health check ───────────────────────────────────────────────────────
    print(f"{BOLD}[1/3] Health check...{RESET}")
    if not check_server(base_url):
        print(f"\n{RED}ERROR: Cannot connect to the server at {base_url}{RESET}")
        print()
        print("Make sure the API is running:")
        print("  uvicorn app:app --port 8000")
        print()
        sys.exit(1)
    print(f"  {GREEN}Server is up and healthy.{RESET}")

    # ── 2. Collect images ─────────────────────────────────────────────────────
    print(f"\n{BOLD}[2/3] Collecting test images...{RESET}")
    if not os.path.isdir(TEST_DIR):
        print(f"{RED}ERROR: Test folder not found: '{TEST_DIR}'{RESET}")
        print("Run download_dataset.py first.")
        sys.exit(1)

    samples = collect_test_images(TEST_DIR, num_per_class)
    if not samples:
        print(f"{RED}ERROR: No test images found in '{TEST_DIR}'.{RESET}")
        sys.exit(1)

    print(f"  Found {len(samples)} images to test.")

    # ── 3. Send predictions ───────────────────────────────────────────────────
    print(f"\n{BOLD}[3/3] Sending predictions to {base_url}/predict...{RESET}")
    print()
    print(
        f"  {'#':>3}   "
        f"{'File':<38}  "
        f"{'True label':<18}  "
        f"{'Prediction':<18}  "
        f"{'Conf':<7}  "
        f"{'Time':<6}  "
        f"Result"
    )
    print("  " + "─" * 110)

    results    = []
    total_ms   = 0.0
    errors     = 0

    for i, (path, true_label) in enumerate(samples, start=1):
        fname = os.path.basename(path)
        try:
            pred_label, confidence, latency_ms = send_prediction(base_url, path)
            correct = (pred_label == true_label)
            total_ms += latency_ms
            results.append({
                "path":       path,
                "true":       true_label,
                "pred":       pred_label,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "correct":    correct,
            })
            print(result_row(i, fname, true_label, pred_label, confidence, latency_ms, correct))

        except requests.exceptions.Timeout:
            print(f"  {i:>3}.  {fname:<38}  {RED}TIMEOUT — server did not respond in {REQUEST_TIMEOUT}s{RESET}")
            errors += 1

        except requests.exceptions.HTTPError as e:
            print(f"  {i:>3}.  {fname:<38}  {RED}HTTP ERROR — {e}{RESET}")
            errors += 1

        except Exception as e:
            print(f"  {i:>3}.  {fname:<38}  {RED}ERROR — {e}{RESET}")
            errors += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  {BOLD}RESULTS SUMMARY{RESET}")
    print("=" * 70)

    if results:
        num_correct = sum(1 for r in results if r["correct"])
        num_wrong   = len(results) - num_correct
        accuracy    = num_correct / len(results) * 100
        avg_latency = total_ms / len(results)

        acc_color = GREEN if accuracy >= 80 else YELLOW if accuracy >= 60 else RED

        print(f"  Total tested  : {len(results)}")
        print(f"  Correct (PASS): {GREEN}{num_correct}{RESET}")
        print(f"  Wrong   (FAIL): {RED}{num_wrong}{RESET}")
        if errors:
            print(f"  Errors        : {YELLOW}{errors}{RESET}")
        print(f"  Accuracy      : {acc_color}{accuracy:.1f}%{RESET}")
        print(f"  Avg latency   : {avg_latency:.0f}ms per request")

        # Per-class breakdown
        print()
        print("  Per-class breakdown:")
        for class_name in CLASS_NAMES:
            class_results = [r for r in results if r["true"] == class_name]
            if class_results:
                class_correct = sum(1 for r in class_results if r["correct"])
                class_acc     = class_correct / len(class_results) * 100
                bar_filled    = int(class_acc / 5)
                bar           = "█" * bar_filled + "░" * (20 - bar_filled)
                print(f"    {class_name:<12}  {bar}  {class_correct}/{len(class_results)}  ({class_acc:.1f}%)")

        # Highlight failures for easy review
        failures = [r for r in results if not r["correct"]]
        if failures:
            print()
            print(f"  {RED}Failed images:{RESET}")
            for r in failures:
                print(f"    ✗  {os.path.basename(r['path']):<40} "
                      f"True: {r['true']:<12}  Pred: {r['pred']}  ({r['confidence']*100:.1f}%)")
        else:
            print()
            print(f"  {GREEN}All predictions were correct! 🎉{RESET}")

    else:
        print(f"  {RED}No results — all requests failed.{RESET}")

    print("=" * 70)
    print()


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send test images to the FastAPI server and validate predictions."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Base URL of the API server (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=DEFAULT_NUM,
        help=f"Number of images per class to test (default: {DEFAULT_NUM})",
    )
    args = parser.parse_args()

    run_tests(base_url=args.url, num_per_class=args.num)
