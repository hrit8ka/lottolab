"""python -m ingestion.run_pipeline [--csv-dir data/raw] [--url URL ...]

Ingests every CSV in --csv-dir (e.g. exports downloaded from Yelo) and any
--url pages, validates, stores in SQLite, flags conflicts, writes a report.
Incremental: identical existing draws are counted as duplicates.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from . import sources, storage, yelo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="data/raw")
    ap.add_argument("--url", action="append", default=[])
    ap.add_argument("--yelo-html-dir", default=str(yelo.CACHE_DIR), help="cached Yelo month pages")
    ap.add_argument("--db", default=str(storage.DB_PATH))
    args = ap.parse_args()

    con = storage.connect(Path(args.db))
    report = []

    for f in sorted(Path(args.csv_dir).glob("*.csv")):
        good, bad = sources.rows_from_csv_file(f)
        st = storage.upsert_draws(con, good, source=f.name)
        for raw, why in bad:
            storage.log_rejected(con, raw, why, f.name)
        report.append(f"{f.name}: {len(good)} valid, {len(bad)} rejected, {st}")

    for f in sorted(Path(args.yelo_html_dir).glob("*.html")):
        good, bad = yelo.parse_yelo_html(f.read_text())
        st = storage.upsert_draws(con, good, source=f"yelo:{f.stem}")
        for raw, why in bad:
            storage.log_rejected(con, raw, why, f"yelo:{f.stem}")
        report.append(f"{f.name}: {len(good)} valid, {len(bad)} rejected, {st}")

    for url in args.url:
        good, bad = sources.fetch_html_rows(url)
        st = storage.upsert_draws(con, good, source=url)
        for raw, why in bad:
            storage.log_rejected(con, raw, why, url)
        report.append(f"{url}: {len(good)} valid, {len(bad)} rejected, {st}")

    total = con.execute("SELECT COUNT(*), MIN(draw_date), MAX(draw_date) FROM draws").fetchone()
    nconf = con.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0]
    report.append(f"TOTAL in DB: {total[0]} draws, {total[1]} -> {total[2]}; conflicts logged: {nconf}")
    text = "\n".join(report)
    print(text)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/processed/ingestion_report.txt").write_text(text)


if __name__ == "__main__":
    main()
