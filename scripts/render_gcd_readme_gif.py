"""Render the README GCD concept-emergence GIF from the live visualiser.

Drives the mechinterp-viz app (the `viz/` git submodule) with Playwright so the
README animation shows the exact "Transcoder feature alignment" and
"Inter-layer direction similarity" panels as rendered by the hosted
visualiser, stepping through each ranked anchor for the requested concept.
The middle panel (normally "Delta trajectory and permutation null") is
swapped for a small grid of the top-6 aligned transcoder features' own
activation-profile plots — the same per-feature images the site shows in its
click-to-inspect panel — so the GIF surfaces feature-level detail instead.

Requires:
  - the `viz` submodule checked out (`git submodule update --init viz`) and
    its dependencies installed (`npm install` inside `viz/`)
  - `pip install playwright && playwright install chromium`
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import time
import urllib.request
from contextlib import closing
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
VIZ_DIR = REPO_ROOT / "viz"
TOP_K_FEATURES = 6


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1.0)
            return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"Dev server at {url} did not start within {timeout}s")


def _concept_json_path(concept: str) -> Path:
    filename = concept.strip().lower().replace(" ", "_") + "_T0.json"
    return VIZ_DIR / "data" / filename


def _top_features_by_anchor(concept: str) -> list[list[dict]]:
    """Top-K aligned features per anchor, in the same order the app's anchor buttons use."""
    data = json.loads(_concept_json_path(concept).read_text())
    anchors = sorted(data["anchors"], key=lambda a: (a["position"], a["rank"]))
    result = []
    for anchor in anchors:
        features = [f for f in anchor["features"] if f.get("image_url")]
        top = sorted(features, key=lambda f: abs(f.get("score", 0.0)), reverse=True)[:TOP_K_FEATURES]
        result.append([{**f, "image_url": f"/data/{f['image_url']}"} for f in top])
    return result


# Inline styles for the injected mini-grid keep it visually consistent with the rest of the
# app (same border/radius/font tokens as .concept-panel and .concept-feature-plot in
# ConceptView.css) without depending on classes that don't exist for a multi-feature layout.
_MINIGRID_CSS = """
.gif-feature-minigrid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; padding: 0 15px 12px; }
.gif-feature-card { display: flex; flex-direction: column; }
.gif-feature-card img { display: block; width: 100%; height: 138px; object-fit: contain;
  background: white; border: 1px solid #e1e7ef; border-radius: 8px; }
.gif-feature-card .gif-feature-name { margin-top: 4px; color: #27364d; font-family: monospace;
  font-size: 0.7rem; }
.gif-feature-card .gif-feature-score { color: #718096; font-size: 0.68rem; }
"""


def _inject_top_features_panel(page, features: list[dict]) -> None:
    page.evaluate(
        """([features, css]) => {
            if (!document.getElementById('gif-minigrid-style')) {
                const style = document.createElement('style');
                style.id = 'gif-minigrid-style';
                style.textContent = css;
                document.head.appendChild(style);
            }
            const panel = document.querySelectorAll('.concept-plots-grid > .concept-panel')[1];
            if (!panel) return;
            const cards = features.map(f => `
                <div class="gif-feature-card">
                    <img src="${f.image_url}" alt="${f.feature}" />
                    <span class="gif-feature-name">${f.feature}</span>
                    <span class="gif-feature-score">score ${f.score.toFixed(3)}</span>
                </div>
            `).join('');
            panel.innerHTML = `
                <div class="concept-panel-heading">
                    <div><h2>Top transcoder features</h2></div>
                </div>
                <div class="gif-feature-minigrid">${cards}</div>
            `;
        }""",
        [features, _MINIGRID_CSS],
    )


def capture_frames(concept: str, out_dir: Path, viewport_width: int, scale: int) -> list[Image.Image]:
    if not (VIZ_DIR / "node_modules").exists():
        raise RuntimeError("viz/node_modules is missing — run `npm install` in viz/ first")

    top_features_by_anchor = _top_features_by_anchor(concept)

    port = _free_port()
    server = subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", str(port), "--strictPort"],
        cwd=VIZ_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://localhost:{port}/"
    frames: list[Image.Image] = []
    try:
        _wait_for_server(url)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport={"width": viewport_width, "height": 900},
                device_scale_factor=scale,
            )
            page.goto(url, wait_until="networkidle")
            page.get_by_text(concept, exact=True).click()
            page.wait_for_timeout(2000)

            content = page.locator(".concept-content")
            content.wait_for()
            anchor_buttons = page.locator(".concept-anchor-context button").all()
            if len(anchor_buttons) != len(top_features_by_anchor):
                raise RuntimeError(
                    f"Anchor button count ({len(anchor_buttons)}) does not match anchors in "
                    f"{_concept_json_path(concept).name} ({len(top_features_by_anchor)})"
                )
            for i, button in enumerate(anchor_buttons):
                button.click()
                page.wait_for_timeout(900)
                _inject_top_features_panel(page, top_features_by_anchor[i])
                page.wait_for_timeout(300)
                frame_path = out_dir / f"frame_{i + 1:02d}.png"
                content.screenshot(path=str(frame_path))
                frames.append(Image.open(frame_path).convert("RGB"))
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    return frames


def render_gif(
    frames: list[Image.Image],
    out_path: Path,
    target_width: int,
    frame_ms: int,
    last_frame_ms: int,
) -> None:
    resized = []
    for im in frames:
        w, h = im.size
        new_h = round(h * target_width / w)
        resized.append(im.resize((target_width, new_h), Image.LANCZOS))

    # Quantize every frame against one shared palette so colors don't flicker between frames.
    combined = Image.new("RGB", (target_width * len(resized), resized[0].height))
    for i, im in enumerate(resized):
        combined.paste(im, (i * target_width, 0))
    palette = combined.quantize(colors=256, method=Image.MEDIANCUT)
    quantized = [im.quantize(palette=palette, dither=Image.FLOYDSTEINBERG) for im in resized]

    durations = [frame_ms] * len(quantized)
    durations[-1] = last_frame_ms

    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", default="gcd", help="Preset button text to click (carry, gcd, residue class)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs/_static/images/gcd_concept_emergence.gif")
    parser.add_argument("--width", type=int, default=1400, help="Output GIF width in pixels")
    parser.add_argument("--capture-width", type=int, default=1500, help="Browser viewport width")
    parser.add_argument("--scale", type=int, default=2, help="Device pixel ratio used for capture")
    parser.add_argument("--frame-ms", type=int, default=1400)
    parser.add_argument("--last-frame-ms", type=int, default=2800)
    args = parser.parse_args()

    tmp_dir = REPO_ROOT / ".gif_frames_tmp"
    tmp_dir.mkdir(exist_ok=True)
    try:
        frames = capture_frames(args.concept, tmp_dir, args.capture_width, args.scale)
        render_gif(frames, args.out, args.width, args.frame_ms, args.last_frame_ms)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
