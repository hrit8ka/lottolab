"""Parsers and fetchers for Loto result sources.

IMPORTANT: none of these have been run against the live sites yet (the build
environment had no access to them). Run `python -m ingestion.probe` on your Mac
first; it reports what each site actually exposes so we can pick the right method.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .validator import InvalidDraw, validate_draw

HEADERS = {"User-Agent": "LotoLab-research/0.1 (personal research project)"}

SOURCES = {
    "lottotech": "https://www.lottotech.mu/loto/",
    "yelo": "https://www.yelo.mu/lottery/results/history",
    "myt": "https://www.myt.mu/sinformer/loterie/archive.php",
}

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4}|\d{1,2} [A-Za-z]{3,9} \d{4})\b")
INT_RE = re.compile(r"\b\d{1,2}\b")


def fetch(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r


def rows_from_dataframe(df: pd.DataFrame) -> tuple[list, list]:
    """Tolerant CSV parser: finds a date column and six number columns.
    Returns (valid_rows, rejected[(raw, reason)])."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), df.columns[0])
    num_cols = [c for c in df.columns if c != date_col]
    good, bad = [], []
    for _, row in df.iterrows():
        raw = row.to_dict()
        try:
            nums = []
            for c in num_cols:
                v = row[c]
                if isinstance(v, str):
                    nums += [int(x) for x in INT_RE.findall(v)]
                elif pd.notna(v):
                    nums.append(int(v))
            if len(nums) > 6:  # some exports append a bonus ball: keep first six
                nums = nums[:6]
            good.append(validate_draw(row[date_col], nums))
        except (InvalidDraw, ValueError) as e:
            bad.append((str(raw), str(e)))
    return good, bad


def rows_from_csv_file(path: Path) -> tuple[list, list]:
    return rows_from_dataframe(pd.read_csv(path, sep=None, engine="python"))


def rows_from_html(html: str) -> tuple[list, list]:
    """Heuristic: any <tr> with a date and six ints in 1..40."""
    soup = BeautifulSoup(html, "lxml")
    good, bad = [], []
    for tr in soup.find_all("tr"):
        text = " ".join(tr.get_text(" ", strip=True).split())
        m = DATE_RE.search(text)
        if not m:
            continue
        rest = text.replace(m.group(0), " ")
        nums = [int(x) for x in INT_RE.findall(rest)]
        try:
            good.append(validate_draw(m.group(0), nums[:6]))
        except InvalidDraw as e:
            bad.append((text, str(e)))
    return good, bad


def fetch_html_rows(url: str) -> tuple[list, list]:
    return rows_from_html(fetch(url).text)
