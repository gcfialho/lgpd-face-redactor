"""Screenshots of the web app for the README, taken by driving the real UI in headless Chromium.

Uses the clip and reference photo produced by `demo/make_demo.py` (public-domain NASA footage).

    uv run python demo/make_demo.py --out docs/img
    uv run --with playwright python demo/screenshots.py --out docs/img
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"


def wait_for_server(timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/healthz", timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise SystemExit("The web app did not start")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path(".demo"))
    parser.add_argument("--out", type=Path, default=Path("docs/img"))
    args = parser.parse_args()
    clip, subject = args.work / "clip.mp4", args.work / "subject.jpg"
    if not clip.exists() or not subject.exists():
        raise SystemExit("Run demo/make_demo.py first")
    args.out.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen([sys.executable, "-m", "lgpd_face_redactor.web.app", "--port", str(PORT)])
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1180, "height": 860}, device_scale_factor=1.5)

            page.goto(BASE)
            page.set_input_files("input[name=file]", str(clip))
            page.click("details summary")
            page.set_input_files("input[name=references]", str(subject))
            page.click("form[action='/jobs'] button[type=submit]")

            page.wait_for_selector(".gallery", timeout=600_000)
            page.screenshot(path=str(args.out / "web-gallery.png"), full_page=True)

            page.click("form[action$='/render'] button[type=submit]")
            page.wait_for_selector(".result", timeout=600_000)
            page.wait_for_function(
                "() => { const v = document.querySelector('video'); return !v || v.readyState >= 2 }"
            )
            page.evaluate("() => { const v = document.querySelector('video'); if (v) v.currentTime = 2 }")
            page.wait_for_timeout(1000)
            page.locator(".result").screenshot(path=str(args.out / "web-result.png"))
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=30)
    print("Saved", *(str(p) for p in sorted(args.out.glob("web-*.png"))))


if __name__ == "__main__":
    main()
