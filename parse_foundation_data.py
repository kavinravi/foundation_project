"""Split the wide ``Foundation Project Data(Sheet1).csv`` into one CSV per
datasource.

The input sheet stores 13 datasources side by side. Each datasource occupies
four columns: ``[Symbol, Date, Last Price, <blank separator>]``. The first row
holds the symbol; the first cell of the second row holds a type marker
(``Price`` or ``Signal``) and a few blocks include an annotation label such as
``Agg Bonds`` or ``LIBOR/SOFR`` in the third row.

Output files match the layout of ``data/0_5Y_TIPS_2002_D.csv``:

    Security,<Symbol>
    Start Date,<YYYY-MM-DD 00:00:00>
    End Date,<YYYY-MM-DD 00:00:00>
    Period,<D|M|W>
    ,
    Date,PX_LAST
    <YYYY-MM-DD 00:00:00>,<value>
    ...

Signal-typed datasources are written to ``data/signals/`` and the rest to
``data/``.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "Foundation Project Data(Sheet1).csv"
DATA_DIR = ROOT / "data"
SIGNALS_DIR = DATA_DIR / "signals"

TYPE_MARKERS = {"price", "signal"}


@dataclass
class DataSource:
    symbol: str
    col_offset: int
    filename: str


SOURCES: list[DataSource] = [
    DataSource("LF98TRUU", 0, "LF98TRUU.csv"),
    DataSource("LF98OAS", 4, "LF98OAS.csv"),
    DataSource("SPX", 8, "SPX.csv"),
    DataSource("VIX", 12, "VIX.csv"),
    DataSource("Unemployment", 16, "Unemployment.csv"),
    DataSource("Umich CONNSENT", 20, "UMICH_CONNSENT.csv"),
    DataSource("Conference Board CONCCONF", 24, "CB_CONCCONF.csv"),
    DataSource("ISM Manufacturing PMI", 28, "ISM_Manufacturing_PMI.csv"),
    DataSource("LBUSTRUU", 32, "LBUSTRUU.csv"),
    DataSource("LBUSOAS", 36, "LBUSOAS.csv"),
    DataSource("Fed Funds Rate", 40, "Fed_Funds_Rate.csv"),
    DataSource("US0003M", 44, "US0003M.csv"),
    DataSource("RTY Index", 48, "RTY_Index.csv"),
]


def parse_date(raw: str) -> datetime | None:
    raw = raw.strip()
    if not raw:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def parse_price(raw: str) -> float | None:
    raw = raw.strip().replace(",", "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def detect_period(dates: list[datetime]) -> str:
    """Return ``'D'``, ``'W'`` or ``'M'`` based on the median spacing."""
    if len(dates) < 2:
        return "D"
    sorted_dates = sorted(dates)
    deltas = [
        (sorted_dates[i + 1] - sorted_dates[i]).days
        for i in range(len(sorted_dates) - 1)
    ]
    deltas = [d for d in deltas if d > 0]
    if not deltas:
        return "D"
    median = statistics.median(deltas)
    if median >= 25:
        return "M"
    if median >= 5:
        return "W"
    return "D"


def determine_type_and_note(rows: list[list[str]], col_offset: int) -> tuple[str, str | None]:
    """Inspect the leftmost column of each block to find ``Price``/``Signal``
    and any extra annotation (e.g. ``Agg Bonds``)."""
    type_label: str | None = None
    note: str | None = None
    for row in rows[1:]:
        cell = row[col_offset].strip() if col_offset < len(row) else ""
        if not cell:
            continue
        lowered = cell.lower()
        if lowered in TYPE_MARKERS and type_label is None:
            type_label = cell.capitalize()
        elif lowered not in TYPE_MARKERS and note is None:
            note = cell
        if type_label and note:
            break
    return type_label or "Price", note


def extract_series(
    rows: list[list[str]], col_offset: int
) -> list[tuple[datetime, float]]:
    """Pull (date, price) pairs from a single block, skipping blank rows."""
    series: dict[datetime, float] = {}
    for row in rows[1:]:
        if col_offset + 2 >= len(row):
            continue
        date = parse_date(row[col_offset + 1])
        price = parse_price(row[col_offset + 2])
        if date is None or price is None:
            continue
        series[date] = price
    return sorted(series.items())


def write_csv(
    path: Path,
    symbol: str,
    note: str | None,
    period: str,
    series: list[tuple[datetime, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    security = f"{symbol} ({note})" if note else symbol
    start, end = series[0][0], series[-1][0]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Security", security])
        writer.writerow(["Start Date", start.strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["End Date", end.strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["Period", period])
        writer.writerow(["", ""])
        writer.writerow(["Date", "PX_LAST"])
        for date, value in series:
            writer.writerow([date.strftime("%Y-%m-%d %H:%M:%S"), value])


def main() -> None:
    with SRC.open(newline="") as f:
        rows = list(csv.reader(f))

    DATA_DIR.mkdir(exist_ok=True)
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    summary: list[tuple[str, str, int, str, str, str]] = []
    for source in SOURCES:
        type_label, note = determine_type_and_note(rows, source.col_offset)
        series = extract_series(rows, source.col_offset)
        if not series:
            print(f"[WARN] {source.symbol}: no data, skipping")
            continue
        period = detect_period([d for d, _ in series])
        target_dir = SIGNALS_DIR if type_label.lower() == "signal" else DATA_DIR
        out_path = target_dir / source.filename
        write_csv(out_path, source.symbol, note, period, series)
        summary.append(
            (
                source.symbol,
                type_label,
                len(series),
                period,
                series[0][0].strftime("%Y-%m-%d"),
                series[-1][0].strftime("%Y-%m-%d"),
            )
        )

    print(f"\n{'Symbol':<28} {'Type':<7} {'Rows':>6} {'P':>3}  {'Start':<11} {'End':<11}")
    print("-" * 74)
    for symbol, type_label, n, period, start, end in summary:
        print(f"{symbol:<28} {type_label:<7} {n:>6} {period:>3}  {start:<11} {end:<11}")


if __name__ == "__main__":
    main()
