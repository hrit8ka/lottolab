"""Run this first on your Mac:  python -m ingestion.probe

Reports, for each source: HTTP status, whether the static HTML already contains
draw rows, any CSV/JSON/XHR hints in the page, and robots.txt rules. Output is
saved to data/external/probe_report.txt -- paste it back to me.
"""
from __future__ import annotations
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from .sources import SOURCES, fetch, rows_from_html

HINT_RE = re.compile(r"""["'(]([^"'()\s]*(?:\.csv|\.json|/api/|ajax|export|download)[^"'()\s]*)""", re.I)


def probe_one(name: str, url: str) -> list[str]:
    out = [f"=== {name}: {url}"]
    try:
        r = fetch(url)
    except Exception as e:  # noqa: BLE001
        return out + [f"  FAILED: {e}"]
    out.append(f"  status={r.status_code} bytes={len(r.text)} content-type={r.headers.get('content-type')}")
    good, bad = rows_from_html(r.text)
    out.append(f"  draw-like rows in static HTML: {len(good)} valid, {len(bad)} rejected")
    if good:
        out.append(f"  sample: {good[:2]}")
    hints = sorted({urljoin(url, h) for h in HINT_RE.findall(r.text)})
    out.append(f"  csv/json/api hints ({len(hints)}):")
    out += [f"    {h}" for h in hints[:25]]
    try:
        root = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        rb = fetch(root + "/robots.txt")
        out.append("  robots.txt:")
        out += [f"    {line}" for line in rb.text.splitlines()[:15]]
    except Exception as e:  # noqa: BLE001
        out.append(f"  robots.txt unavailable: {e}")
    return out


def main():
    lines: list[str] = []
    for name, url in SOURCES.items():
        lines += probe_one(name, url) + [""]
    text = "\n".join(lines)
    print(text)
    p = Path("data/external/probe_report.txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(f"\nSaved to {p}")


if __name__ == "__main__":
    main()
