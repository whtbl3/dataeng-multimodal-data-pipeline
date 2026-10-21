"""Idempotency check (roadmap #8): same video + params + code -> same rows in every table.

Compare two export dirs table by table with EXCEPT ALL in both directions, ignoring the
columns that change on every run (run_id, processed_at, failed_at). Rows only need to be
the same SET, not the same order or bytes: Parquet metadata and row groups may differ.
"""

from pathlib import Path

import duckdb

# change on every run by design: never part of "same result"
VOLATILE = ("run_id", "processed_at", "failed_at")


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def comparable_columns(
    con: duckdb.DuckDBPyConnection, path: Path, ignore: tuple[str, ...]
) -> list[str]:
    """SQL expressions for every column worth comparing, struct fields flattened.

    Input:  audio_segments.parquet, ignore=("run_id", "processed_at", "failed_at")
    Output: ['"video_id"', ..., '"provenance"."model" AS "provenance.model"', ...]
            provenance.run_id / provenance.processed_at are left out like top-level ones
    """
    src = f"'{path}'"
    exprs: list[str] = []
    for name, col_type, *_ in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall():
        if name in ignore:
            continue
        if col_type.startswith("STRUCT"):
            fields = con.sql(f"SELECT {_quote(name)}.* FROM {src} LIMIT 0").columns
            exprs += [
                f"{_quote(name)}.{_quote(f)} AS {_quote(name + '.' + f)}"
                for f in fields
                if f not in ignore
            ]
        else:
            exprs.append(_quote(name))
    return exprs


def _count_except(con, cols: str, a: Path, b: Path) -> int:
    """Rows of a not matched in b. EXCEPT ALL counts duplicates: 2 copies vs 1 -> 1."""
    return con.execute(
        f"SELECT count(*) FROM (SELECT {cols} FROM '{a}' EXCEPT ALL SELECT {cols} FROM '{b}')"
    ).fetchone()[0]


def compare_table(path1: Path, path2: Path, ignore: tuple[str, ...]) -> dict:
    """Compare one table exported twice.

    Output: {"rows_1": 78, "rows_2": 78, "only_in_1": 0, "only_in_2": 0,
             "differing_columns": {}}            identical
            differing_columns: {"provenance.code_version": 78} -> which column explains it
    """
    con = duckdb.connect()
    try:
        cols_1 = comparable_columns(con, path1, ignore)
        cols_2 = comparable_columns(con, path2, ignore)
        if cols_1 != cols_2:
            return {"schema_differs": {"dir1": cols_1, "dir2": cols_2}}
        cols = ", ".join(cols_1)
        result = {
            "rows_1": con.execute(f"SELECT count(*) FROM '{path1}'").fetchone()[0],
            "rows_2": con.execute(f"SELECT count(*) FROM '{path2}'").fetchone()[0],
            # both directions: one direction alone misses rows that only dir2 has
            "only_in_1": _count_except(con, cols, path1, path2),
            "only_in_2": _count_except(con, cols, path2, path1),
            "differing_columns": {},
        }
        if result["only_in_1"] or result["only_in_2"]:
            # which column explains the difference: compare each column on its own
            for expr in cols_1:
                n = _count_except(con, expr, path1, path2) + _count_except(
                    con, expr, path2, path1
                )
                if n:
                    label = expr.split(" AS ")[-1].strip('"')
                    result["differing_columns"][label] = n
        return result
    finally:
        con.close()


def compare_dirs(dir1: Path, dir2: Path, ignore: tuple[str, ...] = VOLATILE) -> dict:
    """Compare every *.parquet table of two export dirs. {table: compare_table result}."""
    names = sorted(
        {p.stem for p in Path(dir1).glob("*.parquet")}
        | {p.stem for p in Path(dir2).glob("*.parquet")}
    )
    result: dict[str, dict] = {}
    for name in names:
        p1, p2 = Path(dir1) / f"{name}.parquet", Path(dir2) / f"{name}.parquet"
        if not p1.exists() or not p2.exists():
            result[name] = {"missing_in": "dir1" if not p1.exists() else "dir2"}
        else:
            result[name] = compare_table(p1, p2, ignore)
    return result


def is_identical(result: dict) -> bool:
    return bool(result) and all(
        t.get("only_in_1") == 0 and t.get("only_in_2") == 0 for t in result.values()
    )


def render_comparison(result: dict, ignore: tuple[str, ...]) -> str:
    """One line per table, then the verdict."""
    lines = [f"ignored columns: {', '.join(ignore)}"]
    for table, r in result.items():
        if "missing_in" in r:
            lines.append(f"  {table:<15} MISSING in {r['missing_in']}")
        elif "schema_differs" in r:
            lines.append(f"  {table:<15} SCHEMA DIFFERS")
        else:
            same = r["only_in_1"] == 0 and r["only_in_2"] == 0
            line = (
                f"  {table:<15} {'same' if same else 'DIFF':<4}  rows {r['rows_1']} / "
                f"{r['rows_2']}  only in dir1 {r['only_in_1']}  only in dir2 {r['only_in_2']}"
            )
            if r["differing_columns"]:
                cols = ", ".join(
                    f"{c} ({n})" for c, n in r["differing_columns"].items()
                )
                line += f"\n      differing columns: {cols}"
            lines.append(line)
    lines.append("IDEMPOTENT" if is_identical(result) else "NOT IDEMPOTENT")
    return "\n".join(lines)
