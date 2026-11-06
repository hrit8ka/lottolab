"""Yelo past-results parser and downloader.

Page structure (seen 2026-11-06): a table of rows
  [ "04 November, 2026 - Wednesday  Loto Result" | numbers separated by <br> | jackpot ]
mixing three games: Loto (6 numbers), Loto Plus (6, separate draw), Loto Vert (7 numbers,
daily). We keep ONLY rows labelled "Loto Result" with exactly 6 numbers.
"""
from __future__ import annotations
import re
import time
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

from .validator import InvalidDraw, validate_draw

URL = "https://www.yelo.mu/lottery/results/history"
CACHE_DIR = Path("data/raw/yelo_html")
DATE_RE = re.compile(r"(\d{1,2}) ([A-Za-z]+),? (\d{4})")


def parse_yelo_html(html: str, game: str = "Loto Result") -> tuple[list, list]:
    """Returns (valid_rows, rejected). `game` is the exact label to keep."""
    soup = BeautifulSoup(html, "lxml")
    good, bad = [], []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        label = " ".join(tds[0].get_text(" ", strip=True).split())
        m = DATE_RE.search(label)
        if not m:
            continue
        rest = label.replace(m.group(0), "")
        # exact game match: "Loto Result" must not match "Loto Plus Result"/"Loto Vert Result"
        if game not in rest:
            continue
        nums = [int(x) for x in re.findall(r"\d+", tds[1].get_text(" ", strip=True))]
        try:
            good.append(validate_draw(m.group(0), nums))
        except InvalidDraw as e:
            bad.append((label + " | " + str(nums), str(e)))
    return good, bad


def _goto(page, tries: int = 3):
    """Ad/tracker-heavy pages never reach 'networkidle', so wait for the DOM + dropdowns instead."""
    for i in range(tries):
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector("select", timeout=30000)
            return
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def _find_selects(page):
    game_sel = month_sel = None
    for i in range(page.locator("select").count()):
        sel = page.locator("select").nth(i)
        opts = [o.strip() for o in sel.locator("option").all_inner_texts()]
        if "Loto Vert" in opts:
            game_sel = sel
        elif any(re.fullmatch(r"\d{4}-\d{2}", o) for o in opts):
            month_sel = sel
    if game_sel is None or month_sel is None:
        raise RuntimeError("Could not find the Game / Month dropdowns; the page layout may have changed.")
    return game_sel, month_sel


def _dump_debug(page, note: str) -> str:
    """Write what the filter form looks like so the selection logic can be corrected."""
    lines = [note, f"current url: {page.url}"]
    for i in range(page.locator("form").count()):
        f = page.locator("form").nth(i)
        lines.append(f"form[{i}] action={f.get_attribute('action')} method={f.get_attribute('method')}")
    for i in range(page.locator("select").count()):
        sel = page.locator("select").nth(i)
        opts = sel.locator("option")
        vals = [(opts.nth(j).get_attribute("value"), opts.nth(j).inner_text().strip()) for j in range(min(opts.count(), 6))]
        lines.append(f"select[{i}] name={sel.get_attribute('name')} id={sel.get_attribute('id')} "
                     f"onchange={sel.get_attribute('onchange')} first options(value,label)={vals}")
    btns = page.locator("button, input[type=submit]")
    for i in range(min(btns.count(), 6)):
        lines.append(f"button[{i}]: {btns.nth(i).inner_text()[:40]!r} type={btns.nth(i).get_attribute('type')}")
    text = "\n".join(lines)
    out = Path("data/external/yelo_debug.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text


def _apply_filters(page, month: str):
    game_sel, _ = _find_selects(page)
    game_sel.select_option(label="Loto")
    page.wait_for_timeout(2000)            # the page may reload itself
    _, month_sel = _find_selects(page)
    month_sel.select_option(label=month)
    page.wait_for_timeout(3000)
    btn = page.locator("button[type=submit], input[type=submit]")
    if btn.count():
        btn.first.click()
        page.wait_for_timeout(3000)
    page.wait_for_load_state("domcontentloaded")


def download_all(delay: float = 3.0, headless: bool = True) -> None:
    """Select Game=Loto and each month, cache the HTML ONLY if every draw on the page
    falls inside the requested month (otherwise the filter did not apply: stop and dump diagnostics).
    Requires: pip install playwright && python -m playwright install chromium"""
    from playwright.sync_api import sync_playwright

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        _goto(page)
        _, month_sel = _find_selects(page)
        months = [o.strip() for o in month_sel.locator("option").all_inner_texts()]
        months = [m for m in months if re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", m)]
        print(f"{len(months)} months available")
        for k, m in enumerate(months):
            out = CACHE_DIR / f"{m}.html"
            if out.exists() and k != 0:    # always refresh the newest month
                continue
            _goto(page)
            _apply_filters(page, m)
            html = page.content()
            good, bad = parse_yelo_html(html)
            wrong = [d for d, _ in good if d.strftime("%Y-%m") != m]
            if wrong or (k == 0 and not good):
                msg = _dump_debug(page, f"Month filter did not apply for {m}: {len(wrong)} of {len(good)} draws outside the month.")
                print(msg)
                print("\nStopped. Nothing wrong was saved. Paste the text above (also in data/external/yelo_debug.txt) to Claude.")
                browser.close()
                return
            out.write_text(html)
            print(f"{m}: {len(good)} Loto draws, {len(bad)} rejected")
            time.sleep(delay)
        browser.close()


if __name__ == "__main__":
    download_all()
